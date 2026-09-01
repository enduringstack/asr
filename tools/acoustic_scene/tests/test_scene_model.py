from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from scene_model import (  # noqa: E402
    backbone_spec,
    fuse_expert_logits,
    fuse_hierarchical_logits,
)
from audio_features import load_config  # noqa: E402


class SceneModelTest(unittest.TestCase):
    def test_frontend_matches_official_efficientat_frequency_contract(self) -> None:
        config = load_config()
        self.assertEqual(config["sample_rate"], 32_000)
        self.assertEqual(config["window_samples"], 320_000)
        self.assertEqual(config["n_fft"], 1024)
        self.assertEqual(config["win_length"], 800)
        self.assertEqual(config["hop_length"], 320)
        self.assertEqual(config["f_max"], 15_000)

    def test_backbone_specs_match_efficientat_names_and_widths(self) -> None:
        self.assertEqual(backbone_spec("dymn04_as"), ("dymn", 0.4))
        self.assertEqual(backbone_spec("dymn10_as"), ("dymn", 1.0))
        self.assertEqual(backbone_spec("dymn20_as"), ("dymn", 2.0))
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

    def test_experts_keep_group_mass_and_reassign_inside_each_group(self) -> None:
        base = torch.tensor([[2.0, 1.0, 3.0, 0.0, -1.0, -2.0]])
        experts = torch.tensor([[-4.0, 4.0, -5.0, 5.0]])
        fused = fuse_expert_logits(torch.cat((base, experts), dim=1))
        self.assertEqual(tuple(fused.shape), (1, 6))
        self.assertGreater(float(fused[0, 1]), float(fused[0, 0]))
        self.assertGreater(float(fused[0, 3]), float(fused[0, 2]))
        self.assertTrue(torch.isfinite(fused).all())

        base_probability = torch.softmax(base, dim=1)
        fused_probability = torch.softmax(fused, dim=1)
        self.assertAlmostEqual(
            float(base_probability[0, :2].sum()),
            float(fused_probability[0, :2].sum()),
            places=6,
        )
        self.assertAlmostEqual(
            float(base_probability[0, 2:4].sum()),
            float(fused_probability[0, 2:4].sum()),
            places=6,
        )

    def test_expert_fusion_requires_ten_outputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "ten"):
            fuse_expert_logits(torch.zeros(1, 9))

    def test_copying_base_pair_logits_into_experts_preserves_predictions(self) -> None:
        base = torch.randn(4, 6)
        raw = torch.cat((base, base[:, :4]), dim=1)
        self.assertTrue(torch.allclose(fuse_expert_logits(raw), base, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
