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
from dataset_protocol import aggregate_source_logits


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
    parser.add_argument(
        "--evaluation-unit",
        choices=("source", "window"),
        default="source",
        help="average adjacent windows from one recording before scoring",
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
    item_rows: list[dict[str, object]] = []
    window_targets: list[int] = []
    window_logits: list[np.ndarray] = []
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
        target = class_to_index[item["label"]]
        item_rows.append(item)
        window_targets.append(target)
        window_logits.append(np.asarray(logits, dtype=np.float64))
    elapsed = time.perf_counter() - started
    if not item_rows:
        raise RuntimeError(f"manifest contains no rows for split {args.split}")
    window_probability_rows = []
    for logits in window_logits:
        probabilities = np.exp(logits - logits.max())
        window_probability_rows.append(probabilities / probabilities.sum())
    window_predictions = [int(values.argmax()) for values in window_probability_rows]
    if args.evaluation_unit == "source":
        aggregated = aggregate_source_logits(item_rows, window_logits, window_targets)
        decision_ids = aggregated.source_ids
        decision_logits = [np.asarray(values) for values in aggregated.logits]
        targets = aggregated.targets
        windows_by_source: dict[str, list[str]] = {}
        for item in item_rows:
            windows_by_source.setdefault(str(item["source_id"]), []).append(str(item["id"]))
    else:
        decision_ids = [str(item["id"]) for item in item_rows]
        decision_logits = window_logits
        targets = window_targets
        windows_by_source = {str(item["id"]): [str(item["id"])] for item in item_rows}
    probability_rows: list[np.ndarray] = []
    for logits in decision_logits:
        probabilities = np.exp(logits - logits.max())
        probability_rows.append(probabilities / probabilities.sum())
    predictions = [int(values.argmax()) for values in probability_rows]
    rows: list[dict[str, object]] = []
    for decision_id, target, prediction, probabilities in zip(
        decision_ids, targets, predictions, probability_rows
    ):
        rows.append({
            "id": decision_id,
            "window_ids": windows_by_source[decision_id],
            "truth": classes[target],
            "prediction": classes[prediction],
            "confidence": float(probabilities[prediction]),
            "probabilities": {
                name: float(value) for name, value in zip(classes, probabilities)
            },
            "correct": prediction == target,
        })
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
        "evaluation_unit": args.evaluation_unit,
        "samples": len(rows),
        "evaluated_windows": len(item_rows),
        **raw_metrics,
        "window_metrics": metric_report(window_targets, window_predictions, classes),
        "elapsed_seconds": elapsed,
        "rtf": elapsed / (len(item_rows) * int(config["window_seconds"])),
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
