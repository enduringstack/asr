"""Deterministic source-isolated split and sampling helpers."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar


SPLITS = ("train", "calibration", "test")


class ProtocolRow(Protocol):
    id: str
    label: str
    split: str
    source_group: str


RowT = TypeVar("RowT", bound=ProtocolRow)


@dataclass(frozen=True)
class SourceAggregation:
    """Mean logits and one truth label for each original recording."""

    source_ids: list[str]
    logits: list[list[float]]
    targets: list[int]


def aggregate_source_logits(
    rows: Sequence[dict[str, object]],
    logits: Sequence[Sequence[float]],
    targets: Sequence[int],
) -> SourceAggregation:
    """Average consecutive windows from one recording into one decision.

    ``source_group`` remains the train/calibration/test leakage boundary, but
    it can represent a location containing several independent recordings.
    Runtime temporal smoothing must therefore aggregate by ``source_id``.
    """
    if not (len(rows) == len(logits) == len(targets)):
        raise ValueError("rows, logits and targets must have equal length")
    grouped: dict[str, list[tuple[list[float], int]]] = {}
    for row, values, target in zip(rows, logits, targets):
        source_id = str(row.get("source_id", ""))
        if not source_id:
            raise ValueError("source_id must not be empty")
        numeric_values = [float(value) for value in values]
        if not numeric_values:
            raise ValueError("logits must not be empty")
        grouped.setdefault(source_id, []).append((numeric_values, int(target)))

    source_ids: list[str] = []
    mean_logits: list[list[float]] = []
    source_targets: list[int] = []
    for source_id, values in grouped.items():
        dimensions = {len(value) for value, _ in values}
        if len(dimensions) != 1:
            raise ValueError(f"logit dimension conflict for source {source_id}")
        source_truth = {target for _, target in values}
        if len(source_truth) != 1:
            raise ValueError(f"target conflict for source {source_id}")
        count = float(len(values))
        source_ids.append(source_id)
        mean_logits.append([
            sum(value[index] for value, _ in values) / count
            for index in range(next(iter(dimensions)))
        ])
        source_targets.append(next(iter(source_truth)))
    return SourceAggregation(source_ids, mean_logits, source_targets)


def map_official_scene(scene: str) -> str:
    """Map official TAU/TUT scene names to the product taxonomy."""
    return {
        "metro": "metro",
        "shopping_mall": "shopping_mall",
        "cafe/restaurant": "cafe_restaurant",
    }.get(scene, "other")


def split_source_group(source_group: str) -> str:
    """Assign one immutable source group to a 70/15/15 split."""
    if not source_group:
        raise ValueError("source_group must not be empty")
    digest = hashlib.sha256(source_group.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 15:
        return "test"
    if bucket < 30:
        return "calibration"
    return "train"


def frozen_split_map(
    source_groups: Iterable[str], frozen: dict[str, str]
) -> dict[str, str]:
    """Keep every previously assigned source in its original evaluation split."""
    invalid = {value for value in frozen.values() if value not in SPLITS}
    if invalid:
        raise ValueError(f"invalid frozen splits: {sorted(invalid)}")
    return {
        source_group: frozen.get(source_group, split_source_group(source_group))
        for source_group in set(source_groups)
    }


def stratified_split_map(
    group_strata: Iterable[tuple[str, str]],
) -> dict[str, str]:
    """Allocate source groups while balancing sample counts per stratum.

    Repeated ``(group, stratum)`` pairs are intentional: their frequency is the
    number of samples contributed by that source group.  Balancing those
    weights avoids a split with the right number of recording locations but a
    badly skewed number of audio windows.
    """
    stratum_by_group: dict[str, str] = {}
    weight_by_group: Counter[str] = Counter()
    for source_group, stratum in group_strata:
        previous = stratum_by_group.setdefault(source_group, stratum)
        if previous != stratum:
            raise ValueError(
                f"source group {source_group} occurs in multiple strata: "
                f"{previous}, {stratum}"
            )
        weight_by_group[source_group] += 1
    groups_by_stratum: dict[str, list[str]] = {}
    for source_group, stratum in stratum_by_group.items():
        groups_by_stratum.setdefault(stratum, []).append(source_group)

    result: dict[str, str] = {}
    for stratum, groups in sorted(groups_by_stratum.items()):
        ordered = sorted(
            groups,
            key=lambda value: (
                -weight_by_group[value],
                hashlib.sha256(f"{stratum}:{value}".encode("utf-8")).digest(),
                value,
            ),
        )
        count = len(ordered)
        if count == 1:
            result[ordered[0]] = "train"
            continue
        if count == 2:
            result[ordered[0]] = "train"
            result[ordered[1]] = "test"
            continue

        total_weight = sum(weight_by_group[group] for group in ordered)
        target = {
            "train": total_weight * 0.70,
            "calibration": total_weight * 0.15,
            "test": total_weight * 0.15,
        }
        assigned = {split: 0 for split in SPLITS}
        # Stable tie order is chosen so twenty equally weighted groups become
        # exactly 14/3/3. Larger groups are placed first to minimize imbalance.
        split_priority = ("train", "calibration", "test")
        for source_group in ordered:
            split = max(
                split_priority,
                key=lambda name: target[name] - assigned[name],
            )
            result[source_group] = split
            assigned[split] += weight_by_group[source_group]
    return result


def round_robin_limit(rows: Iterable[RowT], limit: int) -> list[RowT]:
    """Prefer source diversity before selecting another row from a group."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    grouped: dict[str, list[RowT]] = {}
    for row in sorted(rows, key=lambda value: (value.source_group, value.id)):
        grouped.setdefault(row.source_group, []).append(row)

    selected: list[RowT] = []
    depth = 0
    while len(selected) < limit:
        appended = False
        for source_group in sorted(grouped):
            values = grouped[source_group]
            if depth >= len(values):
                continue
            selected.append(values[depth])
            appended = True
            if len(selected) == limit:
                break
        if not appended:
            break
        depth += 1
    return selected


def balanced_sample_weights(
    rows: Iterable[RowT | dict[str, object]],
    *,
    class_balance_power: float,
    sample_unit: str,
) -> list[float]:
    """Build class-balanced weights without over-counting repeated windows."""
    values = list(rows)
    if sample_unit not in {"item", "source"}:
        raise ValueError(f"unknown sample unit: {sample_unit}")
    def field(row: RowT | dict[str, object], name: str) -> str:
        if isinstance(row, dict):
            return str(row[name])
        return str(getattr(row, name))

    item_count_by_class = Counter(field(row, "label") for row in values)
    item_count_by_group = Counter(field(row, "source_group") for row in values)
    groups_by_class: dict[str, set[str]] = {}
    for row in values:
        groups_by_class.setdefault(field(row, "label"), set()).add(
            field(row, "source_group")
        )
    weights: list[float] = []
    for row in values:
        label = field(row, "label")
        source_group = field(row, "source_group")
        if sample_unit == "source":
            population = len(groups_by_class[label])
            repeated_windows = item_count_by_group[source_group]
        else:
            population = item_count_by_class[label]
            repeated_windows = 1
        weights.append(
            1.0
            / (float(population) ** class_balance_power)
            / float(repeated_windows)
        )
    return weights


def validate_source_isolation(rows: Iterable[ProtocolRow]) -> None:
    split_by_group: dict[str, str] = {}
    label_by_group: dict[str, str] = {}
    for row in rows:
        if row.split not in SPLITS:
            raise ValueError(f"unknown split: {row.split}")
        previous = split_by_group.setdefault(row.source_group, row.split)
        if previous != row.split:
            raise ValueError(
                f"source leakage: {row.source_group} occurs in {previous} and {row.split}"
            )
        previous_label = label_by_group.setdefault(row.source_group, row.label)
        if previous_label != row.label:
            raise ValueError(
                f"source label conflict: {row.source_group} occurs as "
                f"{previous_label} and {row.label}"
            )


def summarize_split_counts(rows: Iterable[ProtocolRow]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for row in rows:
        if row.split not in counts:
            raise ValueError(f"unknown split: {row.split}")
        counts[row.split][row.label] += 1
    return {
        split: dict(sorted(values.items()))
        for split, values in counts.items()
        if values
    }
