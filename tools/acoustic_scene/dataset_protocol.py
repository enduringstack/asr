"""Deterministic source-isolated split and sampling helpers."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from typing import Protocol, TypeVar


SPLITS = ("train", "calibration", "test")


class ProtocolRow(Protocol):
    id: str
    label: str
    split: str
    source_group: str


RowT = TypeVar("RowT", bound=ProtocolRow)


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


def validate_source_isolation(rows: Iterable[ProtocolRow]) -> None:
    split_by_group: dict[str, str] = {}
    for row in rows:
        if row.split not in SPLITS:
            raise ValueError(f"unknown split: {row.split}")
        previous = split_by_group.setdefault(row.source_group, row.split)
        if previous != row.split:
            raise ValueError(
                f"source leakage: {row.source_group} occurs in {previous} and {row.split}"
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
