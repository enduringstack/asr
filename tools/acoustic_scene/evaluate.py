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


def apply_unknown_threshold(
    probability_rows: list[np.ndarray], threshold: float, other_index: int
) -> list[int]:
    return [
        int(values.argmax()) if float(values.max()) >= threshold else other_index
        for values in probability_rows
    ]


def metric_report(
    targets: list[int], predictions: list[int], classes: list[str]
) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_recall": float(recall_score(
            targets, predictions, average="macro", zero_division=0
        )),
        "per_class_recall": {
            name: float(value)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("calibration", "test"), default="calibration"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="frozen unknown-rejection threshold; required for blind test",
    )
    args = parser.parse_args()
    if args.split == "test" and args.threshold is None:
        parser.error("--threshold is required when --split test opens the blind set")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

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
        if item["split"] != args.split:
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
    if not rows:
        raise RuntimeError(f"manifest contains no rows for split {args.split}")
    other_index = class_to_index["other"]
    threshold_reports: list[dict[str, float]] = []
    best_threshold: dict[str, float] | None = None
    if args.split == "calibration":
        for threshold in np.arange(0.30, 0.81, 0.01):
            threshold_predictions = apply_unknown_threshold(
                probability_rows, float(threshold), other_index
            )
            threshold_reports.append({
                "threshold": float(round(float(threshold), 2)),
                "accuracy": float(accuracy_score(targets, threshold_predictions)),
                "macro_recall": float(recall_score(
                    targets, threshold_predictions, average="macro", zero_division=0
                )),
            })
        best_threshold = max(
            threshold_reports,
            key=lambda row: (
                min(row["macro_recall"], row["accuracy"]),
                (row["macro_recall"] + row["accuracy"]) / 2.0,
                -abs(row["threshold"] - 0.52),
            ),
        )
    applied_threshold = args.threshold
    if applied_threshold is None and best_threshold is not None:
        applied_threshold = best_threshold["threshold"]
    thresholded_predictions = apply_unknown_threshold(
        probability_rows, float(applied_threshold), other_index
    ) if applied_threshold is not None else predictions
    for row, thresholded_prediction, target in zip(
        rows, thresholded_predictions, targets
    ):
        row["threshold_prediction"] = classes[thresholded_prediction]
        row["threshold_correct"] = thresholded_prediction == target
    raw_metrics = metric_report(targets, predictions, classes)
    thresholded_metrics = metric_report(targets, thresholded_predictions, classes)
    report = {
        "model": str(args.model),
        "split": args.split,
        "samples": len(rows),
        **raw_metrics,
        "elapsed_seconds": elapsed,
        "rtf": elapsed / (len(rows) * int(config["window_seconds"])),
        "threshold_sweep": threshold_reports,
        "recommended_threshold": best_threshold,
        "applied_threshold": applied_threshold,
        "thresholded": thresholded_metrics,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
