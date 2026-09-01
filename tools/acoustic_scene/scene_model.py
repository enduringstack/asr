"""Model contracts shared by acoustic-scene training and ONNX export."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


BACKBONE_SPECS: dict[str, tuple[str, float]] = {
    "mn04_as": ("mn", 0.4),
    "dymn04_as": ("dymn", 0.4),
    "dymn10_as": ("dymn", 1.0),
}


def backbone_spec(backbone: str) -> tuple[str, float]:
    try:
        return BACKBONE_SPECS[backbone]
    except KeyError as error:
        raise ValueError(f"Unsupported EfficientAT backbone: {backbone}") from error


def fuse_hierarchical_logits(raw_logits: torch.Tensor) -> torch.Tensor:
    """Fold a known-vs-other logit into six scene logits."""
    if raw_logits.shape[-1] != 7:
        raise ValueError("hierarchical scene output must contain seven logits")
    scene_logits = raw_logits[..., :6]
    known_logit = raw_logits[..., 6:7]
    known_log_probability = F.logsigmoid(known_logit)
    other_log_probability = F.logsigmoid(-known_logit)
    return torch.cat(
        (
            scene_logits[..., :5] + known_log_probability,
            scene_logits[..., 5:6] + other_log_probability,
        ),
        dim=-1,
    )


class HierarchicalSceneModel(nn.Module):
    """EfficientAT backbone with a seventh known/open-set training output."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def raw_forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_logits, embedding = self.backbone(features)
        return raw_logits, embedding

    def training_forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_logits, embedding = self.raw_forward(features)
        return fuse_hierarchical_logits(raw_logits), raw_logits[..., 6], embedding

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_logits, embedding = self.raw_forward(features)
        return fuse_hierarchical_logits(raw_logits), embedding
