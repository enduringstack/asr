#!/usr/bin/env python3
"""Export the fine-tuned scene classifier and verify ONNX parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from audio_features import PowerToLogMel, WaveformFrontend, load_config, load_waveform
from train import load_model


class ExportModel(nn.Module):
    def __init__(self, mel_basis: torch.Tensor, model: nn.Module) -> None:
        super().__init__()
        self.mel = PowerToLogMel(mel_basis)
        self.model = model

    def forward(self, power_spectrogram: torch.Tensor) -> torch.Tensor:
        features = self.mel(power_spectrogram).unsqueeze(1)
        # Both upstream EfficientAT forward implementations call squeeze(),
        # which removes the batch dimension for a single window. Run the
        # backbone and classifier explicitly so ONNX always returns [N, C].
        if hasattr(self.model, "features"):
            features = self.model.features(features)
        else:
            features = self.model._feature_forward(features)
        return self.model.classifier(features)


class SceneCalibrationReader(CalibrationDataReader):
    def __init__(self, manifest_path: Path, frontend: WaveformFrontend,
                 config: dict[str, object], limit: int = 64) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = [
            manifest_path.parent / item["path"]
            for item in manifest["items"]
            if item["split"] == "train"
        ][:limit]
        rows: list[dict[str, np.ndarray]] = []
        for path in paths:
            waveform = load_waveform(
                path, int(config["sample_rate"]), int(config["window_samples"])
            )
            with torch.no_grad():
                power = frontend.power_spectrogram(
                    torch.from_numpy(waveform).unsqueeze(0)
                ).numpy()
            rows.append({"power_spectrogram": power})
        self._rows = iter(rows)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._rows, None)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--efficientat-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-int8", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes: list[str] = checkpoint["classes"]
    config = checkpoint.get("config", load_config())
    backbone = checkpoint.get("backbone", config.get("backbone", "mn04_as"))
    model = load_model(args.efficientat_root, len(classes), backbone)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    frontend = WaveformFrontend(config, augment=False).eval()
    export_model = ExportModel(frontend.mel_basis, model).eval()

    input_frames = int(config["window_samples"]) // int(config["hop_length"])
    static_batch = backbone == "dymn04_as"
    example_batch = 1 if static_batch else 2
    example = torch.rand(
        example_batch, int(config["n_fft"]) // 2 + 1, input_frames
    )
    model_path = args.output / "model.fp32.onnx"
    torch.onnx.export(
        export_model,
        (example,),
        model_path,
        input_names=["power_spectrogram"],
        output_names=["logits"],
        dynamic_axes=None if static_batch else {
            "power_spectrogram": {0: "batch"}, "logits": {0: "batch"}
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(model_path))

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_output = export_model(example).numpy()
    onnx_output = session.run(None, {"power_spectrogram": example.numpy()})[0]
    maximum_difference = float(np.max(np.abs(torch_output - onnx_output)))
    if maximum_difference > 2e-4:
        raise RuntimeError(f"ONNX parity failed: max abs diff {maximum_difference}")

    int8_report: dict[str, object] = {}
    if not args.skip_int8:
        int8_path = args.output / "model.int8.onnx"
        quantize_static(
            model_input=str(model_path),
            model_output=str(int8_path),
            calibration_data_reader=SceneCalibrationReader(
                args.manifest, frontend, config
            ),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
        )
        onnx.checker.check_model(onnx.load(int8_path))
        int8_session = ort.InferenceSession(
            str(int8_path), providers=["CPUExecutionProvider"]
        )
        int8_output = int8_session.run(
            None, {"power_spectrogram": example.numpy()}
        )[0]
        int8_report = {
            "int8_model": int8_path.name,
            "int8_sha256": sha256(int8_path),
            "int8_bytes": int8_path.stat().st_size,
            "int8_max_abs_difference": float(
                np.max(np.abs(torch_output - int8_output))
            ),
        }

    # The fixture is consumed by the native C++ frontend parity test.
    rng = np.random.default_rng(20260901)
    waveform = rng.standard_normal(int(config["window_samples"])).astype(np.float32) * 0.01
    with torch.no_grad():
        power = frontend.power_spectrogram(torch.from_numpy(waveform).unsqueeze(0)).numpy()
    np.savez_compressed(
        args.output / "frontend_fixture.npz",
        waveform=waveform,
        power=power,
    )
    (args.output / "labels.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    report = {
        "model": model_path.name,
        "sha256": sha256(model_path),
        "bytes": model_path.stat().st_size,
        "opset": 17,
        "onnxruntime_export_check": ort.__version__,
        "max_abs_difference": maximum_difference,
        "input_shape": [
            1 if static_batch else "batch",
            int(config["n_fft"]) // 2 + 1,
            input_frames,
        ],
        "classes": classes,
        "source_checkpoint": str(args.checkpoint),
        "backbone": backbone,
        **int8_report,
    }
    (args.output / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
