#!/usr/bin/env python3
"""Evaluate an exported scene ONNX model on the held-out manifest split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

from audio_features import WaveformFrontend, load_config, load_waveform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = load_config()
    classes: list[str] = list(manifest.get("classes", config["classes"]))
    class_to_index = {name: index for index, name in enumerate(classes)}
    frontend = WaveformFrontend(config, augment=False).eval()
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    targets: list[int] = []
    predictions: list[int] = []
    probability_rows: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for item in manifest["items"]:
        if item["split"] != "validation":
            continue
        waveform = load_waveform(
            args.manifest.parent / item["path"],
            int(config["sample_rate"]),
            int(config["window_samples"]),
        )
        with torch.no_grad():
            power = frontend.power_spectrogram(torch.from_numpy(waveform).unsqueeze(0)).numpy()
        logits = session.run(None, {"power_spectrogram": power})[0][0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        prediction = int(probabilities.argmax())
        target = class_to_index[item["label"]]
        targets.append(target)
        predictions.append(prediction)
        probability_rows.append(probabilities)
        rows.append({
            "id": item["id"],
            "truth": item["label"],
            "prediction": classes[prediction],
            "confidence": float(probabilities[prediction]),
            "probabilities": {
                name: float(value) for name, value in zip(classes, probabilities)
            },
            "correct": prediction == target,
        })
    elapsed = time.perf_counter() - started
    other_index = class_to_index["other"]
    threshold_reports: list[dict[str, float]] = []
    for threshold in np.arange(0.30, 0.81, 0.01):
        threshold_predictions = [
            int(values.argmax()) if float(values.max()) >= float(threshold) else other_index
            for values in probability_rows
        ]
        threshold_reports.append({
            "threshold": float(round(float(threshold), 2)),
            "accuracy": float(accuracy_score(targets, threshold_predictions)),
            "macro_recall": float(recall_score(
                targets, threshold_predictions, average="macro", zero_division=0
            )),
        })
    best_threshold = max(
        threshold_reports,
        key=lambda row: (row["macro_recall"], row["accuracy"], -abs(row["threshold"] - 0.52)),
    )
    report = {
        "model": str(args.model),
        "samples": len(rows),
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
        "elapsed_seconds": elapsed,
        "rtf": elapsed / (len(rows) * int(config["window_seconds"])),
        "threshold_sweep": threshold_reports,
        "recommended_threshold": best_threshold,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
