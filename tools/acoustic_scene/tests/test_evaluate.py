from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from evaluate import apply_unknown_threshold, metric_report  # noqa: E402


class EvaluateTest(unittest.TestCase):
    def test_unknown_threshold_routes_only_low_confidence_rows_to_other(self) -> None:
        rows = [
            np.asarray([0.70, 0.10, 0.10, 0.05, 0.03, 0.02]),
            np.asarray([0.21, 0.20, 0.19, 0.15, 0.14, 0.11]),
        ]
        self.assertEqual(apply_unknown_threshold(rows, 0.4, 5), [0, 5])

    def test_metric_report_uses_all_declared_classes(self) -> None:
        report = metric_report([0, 1, 2], [0, 1, 1], ["a", "b", "other"])
        self.assertAlmostEqual(report["accuracy"], 2 / 3)
        self.assertAlmostEqual(report["macro_recall"], 2 / 3)
        self.assertEqual(len(report["confusion_matrix"]), 3)


if __name__ == "__main__":
    unittest.main()
