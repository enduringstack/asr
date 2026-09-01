from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from dataset_protocol import (  # noqa: E402
    map_official_scene,
    round_robin_limit,
    split_source_group,
    stratified_split_map,
    summarize_split_counts,
    validate_source_isolation,
)


@dataclass(frozen=True)
class Row:
    id: str
    label: str
    split: str
    source_group: str


class DatasetProtocolTest(unittest.TestCase):
    def test_official_scene_mapping_keeps_targets_and_hard_negatives(self) -> None:
        self.assertEqual(map_official_scene("metro"), "metro")
        self.assertEqual(map_official_scene("shopping_mall"), "shopping_mall")
        self.assertEqual(map_official_scene("cafe/restaurant"), "cafe_restaurant")
        self.assertEqual(map_official_scene("train"), "high_speed_train")
        self.assertEqual(map_official_scene("airport"), "other")

    def test_split_is_deterministic_and_has_expected_population_ratio(self) -> None:
        first = [split_source_group(f"group-{index}") for index in range(10_000)]
        second = [split_source_group(f"group-{index}") for index in range(10_000)]
        self.assertEqual(first, second)
        counts = {name: first.count(name) for name in ("train", "calibration", "test")}
        self.assertTrue(6800 <= counts["train"] <= 7200, counts)
        self.assertTrue(1400 <= counts["calibration"] <= 1600, counts)
        self.assertTrue(1400 <= counts["test"] <= 1600, counts)

    def test_round_robin_uses_each_group_before_repeating(self) -> None:
        rows = [
            Row("a1", "metro", "train", "a"),
            Row("a2", "metro", "train", "a"),
            Row("b1", "metro", "train", "b"),
            Row("b2", "metro", "train", "b"),
            Row("c1", "metro", "train", "c"),
        ]
        selected = round_robin_limit(rows, 4)
        self.assertEqual([row.id for row in selected], ["a1", "b1", "c1", "a2"])

    def test_stratified_split_balances_small_source_group_sets(self) -> None:
        groups = [
            (f"metro-{index}", "metro") for index in range(20)
        ] + [
            (f"train-{index}", "train") for index in range(20)
        ]
        mapping = stratified_split_map(groups)
        for prefix in ("metro", "train"):
            values = [mapping[f"{prefix}-{index}"] for index in range(20)]
            self.assertEqual(values.count("train"), 14)
            self.assertEqual(values.count("calibration"), 3)
            self.assertEqual(values.count("test"), 3)

    def test_stratified_split_balances_weighted_source_groups(self) -> None:
        groups = (
            [("large", "mall")] * 7
            + [("medium", "mall")] * 3
            + [("small-a", "mall")]
            + [("small-b", "mall")]
        )
        mapping = stratified_split_map(groups)
        weighted_counts = {name: 0 for name in ("train", "calibration", "test")}
        for group, _ in groups:
            weighted_counts[mapping[group]] += 1
        self.assertEqual(set(mapping.values()), {"train", "calibration", "test"})
        self.assertLessEqual(max(weighted_counts.values()), 8)

    def test_source_group_cannot_cross_splits(self) -> None:
        rows = [
            Row("a", "metro", "train", "shared"),
            Row("b", "other", "test", "shared"),
        ]
        with self.assertRaisesRegex(ValueError, "source leakage"):
            validate_source_isolation(rows)

    def test_summary_keeps_split_and_class_counts(self) -> None:
        rows = [
            Row("a", "metro", "train", "one"),
            Row("b", "metro", "calibration", "two"),
            Row("c", "other", "test", "three"),
        ]
        self.assertEqual(
            summarize_split_counts(rows),
            {
                "train": {"metro": 1},
                "calibration": {"metro": 1},
                "test": {"other": 1},
            },
        )


if __name__ == "__main__":
    unittest.main()
