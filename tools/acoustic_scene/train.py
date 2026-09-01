#!/usr/bin/env python3
"""Fine-tune an EfficientAT backbone for the six product acoustic scenes."""

from __future__ import annotations

import argparse
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
            waveform = waveform.clamp(-1.0, 1.0)
        return waveform, torch.tensor(self.class_to_index[item["label"]], dtype=torch.long)


def load_model(
    efficientat_root: Path, num_classes: int, backbone: str = "mn04_as"
) -> torch.nn.Module:
    efficientat_root = efficientat_root.resolve()
    sys.path.insert(0, str(efficientat_root))
    previous_directory = Path.cwd()
    try:
        # EfficientAT resolves its AudioSet labels and checkpoint cache relative
        # to the repository root during import.
        os.chdir(efficientat_root)
        if backbone == "mn04_as":
            from models.mn.model import get_model  # type: ignore

            model = get_model(
                num_classes=num_classes,
                pretrained_name=backbone,
                width_mult=0.4,
                head_type="mlp",
                input_dim_f=128,
                input_dim_t=1000,
                se_dims="c",
            )
        elif backbone == "dymn04_as":
            from models.dymn.model import get_model  # type: ignore

            model = get_model(
                num_classes=num_classes,
                pretrained_name=backbone,
                width_mult=0.4,
                pretrain_final_temp=1.0,
            )
        else:
            raise ValueError(f"Unsupported EfficientAT backbone: {backbone}")
    finally:
        os.chdir(previous_directory)

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
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--backbone", choices=("mn04_as", "dymn04_as"), default="mn04_as"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config()
    classes: list[str] = list(config["classes"])
    train_set = SceneDataset(args.manifest, "train", augment=True)
    validation_set = SceneDataset(args.manifest, "validation", augment=False)
    class_counts = Counter(item["label"] for item in train_set.items)
    weights = [1.0 / class_counts[item["label"]] for item in train_set.items]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(args.efficientat_root, len(classes), args.backbone).to(device)
    frontend = WaveformFrontend(config, augment=True).to(device)
    head_parameters = list(model.classifier.parameters())
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

    best_macro = -1.0
    patience = 0
    history: list[dict[str, object]] = []
    for epoch in range(args.epochs):
        model.train()
        frontend.train()
        losses: list[float] = []
        for waveforms, labels in train_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            features = frontend(waveforms).unsqueeze(1)
            logits, _ = model(features)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        validation_loss, report = evaluate(
            model, frontend, validation_loader, device, classes
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            **report,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        macro = float(report["macro_recall"])
        if macro > best_macro:
            best_macro = macro
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "config": config,
                "epoch": epoch + 1,
                "validation": report,
                "backbone": args.backbone,
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
        "validation_samples": len(validation_set),
        "class_counts": class_counts,
        "best_macro_recall": best_macro,
    }
    (args.output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
