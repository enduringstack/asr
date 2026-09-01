"""Model contracts shared by acoustic-scene training and ONNX export."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


BACKBONE_SPECS: dict[str, tuple[str, float]] = {
    "mn04_as": ("mn", 0.4),
    "dymn04_as": ("dymn", 0.4),
    "dymn10_as": ("dymn", 1.0),
    "dymn20_as": ("dymn", 2.0),
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


def fuse_expert_logits(raw_logits: torch.Tensor) -> torch.Tensor:
    """Use dedicated experts inside the transport and indoor-venue groups.

    The first six values are the ordinary scene logits. Values 6:8 decide
    metro versus high-speed train and values 8:10 decide shopping mall versus
    cafe/restaurant. Each expert preserves the probability mass assigned to
    its parent group, so it cannot turn an unrelated scene into transport or
    an indoor venue by itself.
    """
    if raw_logits.shape[-1] != 10:
        raise ValueError("expert scene output must contain ten logits")
    scene_logits = raw_logits[..., :6]
    transport_gate = torch.logsumexp(scene_logits[..., :2], dim=-1, keepdim=True)
    venue_gate = torch.logsumexp(scene_logits[..., 2:4], dim=-1, keepdim=True)
    transport = transport_gate + F.log_softmax(raw_logits[..., 6:8], dim=-1)
    venue = venue_gate + F.log_softmax(raw_logits[..., 8:10], dim=-1)
    return torch.cat((transport, venue, scene_logits[..., 4:6]), dim=-1)


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


class ExpertSceneModel(nn.Module):
    """EfficientAT backbone with transport and indoor-venue expert outputs."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def raw_forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_logits, embedding = self.backbone(features)
        return raw_logits, embedding

    def training_forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_logits, embedding = self.raw_forward(features)
        return (
            fuse_expert_logits(raw_logits),
            raw_logits[..., :6],
            raw_logits[..., 6:8],
            raw_logits[..., 8:10],
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_logits, embedding = self.raw_forward(features)
        return fuse_expert_logits(raw_logits), embedding
