#!/usr/bin/env python3
"""Bundle the exported model and pinned, playable scene fixtures into the app."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import onnx


DISPLAY_NAMES = {
    "metro": "地铁",
    "high_speed_train": "高铁 / 城际列车",
    "shopping_mall": "商场",
    "cafe_restaurant": "咖啡厅 / 餐厅",
    "concert": "音乐现场",
    "other": "其他",
}


def quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_input_shape(path: Path) -> list[int | str]:
    model = onnx.load(path, load_external_data=False)
    if not model.graph.input:
        raise RuntimeError("scene ONNX has no input")
    shape: list[int | str] = []
    for dimension in model.graph.input[0].type.tensor_type.shape.dim:
        if dimension.dim_value > 0:
            shape.append(int(dimension.dim_value))
        else:
            shape.append(dimension.dim_param or "dynamic")
    return shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-manifest", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        help="optional model to replace; omit when refreshing demo audio only",
    )
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    fixed_root = args.fixed_manifest.resolve().parent
    raw_root = repo / "entry/src/main/resources/rawfile"
    audio_root = raw_root / "test/acoustic_scene"
    model_root = raw_root / "acoustic-scene-classifier"
    audio_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.fixed_manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = manifest["items"]
    generated: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for stale in audio_root.glob("*.wav"):
        stale.unlink()
    for row in rows:
        label = row["label"]
        index = counts.get(label, 0)
        counts[label] = index + 1
        target_name = f"{label}-{index}.wav"
        source = fixed_root / row["path"]
        target = audio_root / target_name
        shutil.copy2(source, target)
        generated.append({**row, "bundled_path": f"test/acoustic_scene/{target_name}"})

    bundled_model = model_root / "model.onnx"
    info_path = model_root / "MODEL_INFO.json"
    if args.model:
        replacement_model = args.model.resolve()
        input_shape = model_input_shape(replacement_model)
        # The currently deployed native FFT branch is the legacy 16 kHz
        # [257, 1000] contract. v4 research exports intentionally use the
        # official 32 kHz [513, 1000] frontend and must not be silently copied
        # until the recorder/native branch is upgraded and endpoint-tested.
        batch_dimension = input_shape[0] if input_shape else None
        compatible_frontend = len(input_shape) == 3 and input_shape[1:] == [257, 1000] and (
            batch_dimension == 1 or isinstance(batch_dimension, str)
        )
        if not compatible_frontend:
            raise RuntimeError(
                f"scene model input {input_shape} is incompatible with the current "
                "HarmonyOS native frontend [1, 257, 1000]; do not deploy this model"
            )
        shutil.copy2(replacement_model, bundled_model)
        info: dict[str, object] = {
            "model": "EfficientAT DyMN04 fine-tuned acoustic-scene classifier",
            "model_sha256": sha256(bundled_model),
            "model_bytes": bundled_model.stat().st_size,
            "model_input_shape": input_shape,
            "classes": list(DISPLAY_NAMES),
        }
    else:
        if not bundled_model.exists() or not info_path.exists():
            parser.error("--model is required when the app has no existing bundled model")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        # A refreshed corpus is not comparable with a legacy validation run.
        # Keep the deployed model identity, but never present stale metrics as
        # if they had been measured on the newly generated fixtures.
        info.pop("evaluation", None)
        info["evaluation_status"] = "not_evaluated_on_current_v4_protocol"
    info["fixed_audio_split"] = manifest.get("split", "calibration")
    info["fixed_audio_duration_seconds"] = manifest.get("duration_seconds", 10)
    info["fixed_audio_count"] = len(generated)
    info["fixed_audio_sha256"] = {
        Path(row["bundled_path"]).name: sha256(raw_root / row["bundled_path"])
        for row in generated
    }
    if args.evaluation:
        evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
        evaluation_summary = {
            key: value for key, value in evaluation.items()
            if key not in {"rows", "threshold_sweep"}
        }
        if "model" in evaluation_summary:
            evaluation_summary["model"] = Path(str(evaluation_summary["model"])).name
        info["evaluation"] = evaluation_summary
    info_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "import { AcousticSceneLabel } from './AcousticSceneTypes';",
        "",
        "export interface AcousticSceneTestCase {",
        "  id: string;",
        "  dataset: string;",
        "  sourceId: string;",
        "  expectedLabel: AcousticSceneLabel;",
        "  expectedDisplayName: string;",
        "  audioPath: string;",
        "  truthBasis: string;",
        "  license: string;",
        "  attribution: string;",
        "}",
        "",
        "/** Generated from the pinned local evaluation manifest. */",
        "export function buildAcousticSceneTestCases(): AcousticSceneTestCase[] {",
        "  return [",
    ]
    for row in generated:
        lines.extend([
            "    {",
            f"      id: '{quote(row['id'])}',",
            f"      dataset: '{quote(row['dataset'])}',",
            f"      sourceId: '{quote(row['source_id'])}',",
            f"      expectedLabel: '{quote(row['label'])}',",
            f"      expectedDisplayName: '{quote(DISPLAY_NAMES[row['label']])}',",
            f"      audioPath: '{quote(row['bundled_path'])}',",
            f"      truthBasis: '{quote(row['truth_basis'])}',",
            f"      license: '{quote(row['license'])}',",
            f"      attribution: '{quote(row['attribution'])}',",
            "    },",
        ])
    lines.extend(["  ];", "}", ""])
    data_path = repo / "entry/src/main/ets/common/AcousticSceneTestData.ets"
    data_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
