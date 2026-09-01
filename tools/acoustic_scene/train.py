#!/usr/bin/env python3
"""Fine-tune an EfficientAT backbone for the six product acoustic scenes."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from audio_features import WaveformFrontend, load_config, load_waveform
from scene_model import HierarchicalSceneModel, backbone_spec


class SceneDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, augment: bool) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.root = manifest_path.parent
        self.classes: list[str] = manifest["classes"]
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}
        self.items = [item for item in manifest["items"] if item["split"] == split]
        self.augment = augment
        config = load_config()
        self.sample_rate = int(config["sample_rate"])
        self.window_samples = int(config["window_samples"])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]
        samples = load_waveform(
            self.root / item["path"], self.sample_rate, self.window_samples
        )
        waveform = torch.from_numpy(samples)
        if self.augment:
            shift = random.randint(0, max(0, waveform.numel() - 1))
            waveform = torch.roll(waveform, shift)
            gain_db = random.uniform(-7.0, 5.0)
            waveform = waveform * (10.0 ** (gain_db / 20.0))
            if random.random() < 0.35:
                noise_scale = random.uniform(0.0005, 0.006)
                waveform = waveform + torch.randn_like(waveform) * noise_scale
            if random.random() < 0.30:
                waveform = torchaudio.functional.lowpass_biquad(
                    waveform,
                    self.sample_rate,
                    random.uniform(3_500.0, 7_500.0),
                )
            if random.random() < 0.20:
                waveform = torchaudio.functional.highpass_biquad(
                    waveform,
                    self.sample_rate,
                    random.uniform(40.0, 220.0),
                )
            waveform = waveform.clamp(-1.0, 1.0)
        return waveform, torch.tensor(self.class_to_index[item["label"]], dtype=torch.long)


def load_model(
    efficientat_root: Path,
    num_classes: int,
    backbone: str = "mn04_as",
    objective: str = "direct",
) -> torch.nn.Module:
    efficientat_root = efficientat_root.resolve()
    sys.path.insert(0, str(efficientat_root))
    previous_directory = Path.cwd()
    try:
        # EfficientAT resolves its AudioSet labels and checkpoint cache relative
        # to the repository root during import.
        os.chdir(efficientat_root)
        family, width = backbone_spec(backbone)
        output_count = num_classes + 1 if objective == "hierarchical" else num_classes
        # EfficientAT prints the entire architecture on every construction,
        # which buries epoch metrics in thousands of log lines.
        with contextlib.redirect_stdout(io.StringIO()):
            if family == "mn":
                from models.mn.model import get_model  # type: ignore

                model = get_model(
                    num_classes=output_count,
                    pretrained_name=backbone,
                    width_mult=width,
                    head_type="mlp",
                    input_dim_f=128,
                    input_dim_t=1000,
                    se_dims="c",
                )
            else:
                from models.dymn.model import get_model  # type: ignore

                model = get_model(
                    num_classes=output_count,
                    pretrained_name=backbone,
                    width_mult=width,
                    pretrain_final_temp=1.0,
                )
    finally:
        os.chdir(previous_directory)

    if objective == "hierarchical":
        return HierarchicalSceneModel(model)
    return model


def metrics(targets: list[int], predictions: list[int], classes: list[str]) -> dict[str, object]:
    return {
        "accuracy": accuracy_score(targets, predictions),
        "macro_recall": recall_score(targets, predictions, average="macro", zero_division=0),
        "per_class_recall": {
            name: value
            for name, value in zip(
                classes,
                recall_score(
                    targets,
                    predictions,
                    labels=list(range(len(classes))),
                    average=None,
                    zero_division=0,
                ),
            )
        },
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=list(range(len(classes)))
        ).tolist(),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    frontend: WaveformFrontend,
    loader: DataLoader,
    device: torch.device,
    classes: list[str],
) -> tuple[float, dict[str, object]]:
    model.eval()
    frontend.eval()
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    for waveforms, labels in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        logits, _ = model(frontend(waveforms).unsqueeze(1))
        loss = F.cross_entropy(logits, labels)
        losses.append(float(loss.cpu()))
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return float(np.mean(losses)), metrics(targets, predictions, classes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--efficientat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        type=Path,
        help="continue fine-tuning from a compatible acoustic-scene checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--backbone",
        choices=("mn04_as", "dymn04_as", "dymn10_as"),
        default="mn04_as",
    )
    parser.add_argument(
        "--objective", choices=("direct", "hierarchical"), default="direct"
    )
    parser.add_argument("--known-loss-weight", type=float, default=0.35)
    parser.add_argument(
        "--other-binary-weight",
        type=float,
        default=5.0,
        help="relative weight for unknown/other targets in the known-vs-other head",
    )
    parser.add_argument(
        "--class-balance-power",
        type=float,
        default=0.5,
        help="0 uses natural sampling and 1 makes all scene classes equiprobable",
    )
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument(
        "--head-only-epochs",
        type=int,
        default=3,
        help="warm up the randomly initialized classifier before full fine-tuning",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="training device; auto prefers Apple MPS",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config()
    classes: list[str] = list(config["classes"])
    train_set = SceneDataset(args.manifest, "train", augment=True)
    calibration_set = SceneDataset(args.manifest, "calibration", augment=False)
    class_counts = Counter(item["label"] for item in train_set.items)
    weights = [
        1.0 / (class_counts[item["label"]] ** args.class_balance_power)
        for item in train_set.items
    ]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )
    calibration_loader = DataLoader(
        calibration_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device_name = args.device
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    device = torch.device(device_name)
    model = load_model(
        args.efficientat_root, len(classes), args.backbone, args.objective
    ).to(device)
    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=False)
        if resumed.get("classes") != classes:
            raise RuntimeError("resume checkpoint class order does not match")
        if resumed.get("backbone") != args.backbone:
            raise RuntimeError("resume checkpoint backbone does not match")
        if resumed.get("objective") != args.objective:
            raise RuntimeError("resume checkpoint objective does not match")
        model.load_state_dict(resumed["model_state"])
    frontend = WaveformFrontend(config, augment=True).to(device)
    classifier = model.backbone.classifier if isinstance(
        model, HierarchicalSceneModel
    ) else model.classifier
    head_parameters = list(classifier.parameters())
    head_parameter_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in head_parameter_ids
    ]
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": 6e-5},
        {"params": head_parameters, "lr": 4e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = (-1.0, -1.0)
    best_report: dict[str, object] | None = None
    patience = 0
    history: list[dict[str, object]] = []
    for epoch in range(args.epochs):
        backbone_trainable = epoch >= args.head_only_epochs
        for parameter in backbone_parameters:
            parameter.requires_grad_(backbone_trainable)
        model.train()
        frontend.train()
        losses: list[torch.Tensor] = []
        for waveforms, labels in train_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            soft_targets: torch.Tensor | None = None
            if args.mixup_alpha > 0 and random.random() < 0.5:
                mix_weight = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                permutation = torch.randperm(waveforms.shape[0], device=device)
                waveforms = (
                    waveforms * mix_weight
                    + waveforms[permutation] * (1.0 - mix_weight)
                )
                one_hot = F.one_hot(labels, num_classes=len(classes)).float()
                soft_targets = (
                    one_hot * mix_weight
                    + one_hot[permutation] * (1.0 - mix_weight)
                )
            features = frontend(waveforms).unsqueeze(1)
            if isinstance(model, HierarchicalSceneModel):
                logits, known_logits, _ = model.training_forward(features)
                if soft_targets is None:
                    scene_loss = F.cross_entropy(
                        logits, labels, label_smoothing=0.05
                    )
                    known_targets = (
                        labels != train_set.class_to_index["other"]
                    ).float()
                else:
                    scene_loss = -(
                        soft_targets * F.log_softmax(logits, dim=1)
                    ).sum(dim=1).mean()
                    known_targets = 1.0 - soft_targets[
                        :, train_set.class_to_index["other"]
                    ]
                binary_weights = 1.0 + (1.0 - known_targets) * (
                    args.other_binary_weight - 1.0
                )
                binary_loss = F.binary_cross_entropy_with_logits(
                    known_logits, known_targets, reduction="none"
                )
                loss = scene_loss + args.known_loss_weight * (
                    binary_loss * binary_weights
                ).mean()
            else:
                logits, _ = model(features)
                if soft_targets is None:
                    loss = F.cross_entropy(
                        logits, labels, label_smoothing=0.05
                    )
                else:
                    loss = -(
                        soft_targets * F.log_softmax(logits, dim=1)
                    ).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            # Avoid synchronizing MPS once per batch. The optimizer operations
            # are ordered on the same command queue; one scalar sync at the end
            # of the epoch is sufficient for reporting.
            losses.append(loss.detach())
        scheduler.step()
        calibration_loss, report = evaluate(
            model, frontend, calibration_loader, device, classes
        )
        row = {
            "epoch": epoch + 1,
            "training_stage": "full" if backbone_trainable else "head_only",
            "train_loss": float(torch.stack(losses).mean().cpu()),
            "calibration_loss": calibration_loss,
            **report,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        accuracy = float(report["accuracy"])
        macro_recall = float(report["macro_recall"])
        # The release gate requires both metrics. Maximizing the weaker metric
        # prevents a macro-heavy checkpoint from winning while accuracy drops.
        score = (min(accuracy, macro_recall), (accuracy + macro_recall) / 2.0)
        if score > best_score:
            best_score = score
            best_report = report
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "config": config,
                "epoch": epoch + 1,
                "calibration": report,
                "backbone": args.backbone,
                "objective": args.objective,
                "known_loss_weight": args.known_loss_weight,
                "other_binary_weight": args.other_binary_weight,
                "class_balance_power": args.class_balance_power,
                "mixup_alpha": args.mixup_alpha,
                "efficientat_commit": config["efficientat_commit"],
            }, args.output / "best.pt")
        else:
            patience += 1
            if patience >= 8:
                break

    (args.output / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "device": str(device),
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "backbone": args.backbone,
        "training_samples": len(train_set),
        "calibration_samples": len(calibration_set),
        "class_counts": class_counts,
        "objective": args.objective,
        "known_loss_weight": args.known_loss_weight,
        "other_binary_weight": args.other_binary_weight,
        "class_balance_power": args.class_balance_power,
        "mixup_alpha": args.mixup_alpha,
        "head_only_epochs": args.head_only_epochs,
        "resumed_from": str(args.resume.resolve()) if args.resume else None,
        "best_gate_minimum": best_score[0],
        "best_macro_recall": (
            float(best_report["macro_recall"]) if best_report else 0.0
        ),
        "best_accuracy": float(best_report["accuracy"]) if best_report else 0.0,
    }
    (args.output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
