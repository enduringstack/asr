#!/usr/bin/env python3
"""Bundle the exported model and pinned, playable scene fixtures into the app."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
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
    shutil.copy2(args.model.resolve(), bundled_model)
    info = {
        "model": "EfficientAT DyMN04 fine-tuned acoustic-scene classifier",
        "model_sha256": sha256(bundled_model),
        "model_bytes": bundled_model.stat().st_size,
        "classes": list(DISPLAY_NAMES),
        "fixed_audio_count": len(generated),
        "fixed_audio_sha256": {
            Path(row["bundled_path"]).name: sha256(raw_root / row["bundled_path"])
            for row in generated
        },
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
    (model_root / "MODEL_INFO.json").write_text(
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
