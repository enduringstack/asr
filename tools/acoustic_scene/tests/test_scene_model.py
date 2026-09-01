from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from scene_model import backbone_spec, fuse_hierarchical_logits  # noqa: E402


class SceneModelTest(unittest.TestCase):
    def test_backbone_specs_match_efficientat_names_and_widths(self) -> None:
        self.assertEqual(backbone_spec("dymn04_as"), ("dymn", 0.4))
        self.assertEqual(backbone_spec("dymn10_as"), ("dymn", 1.0))
        self.assertEqual(backbone_spec("mn04_as"), ("mn", 0.4))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            backbone_spec("unknown")

    def test_known_head_moves_probability_between_targets_and_other(self) -> None:
        primary = torch.zeros(2, 6)
        raw = torch.cat((primary, torch.tensor([[5.0], [-5.0]])), dim=1)
        fused = fuse_hierarchical_logits(raw)
        probabilities = torch.softmax(fused, dim=1)
        self.assertGreater(float(probabilities[0, :5].sum()), 0.95)
        self.assertGreater(float(probabilities[1, 5]), 0.95)
        self.assertTrue(torch.isfinite(fused).all())

    def test_hierarchical_fusion_requires_seven_outputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "seven"):
            fuse_hierarchical_logits(torch.zeros(1, 6))


if __name__ == "__main__":
    unittest.main()
