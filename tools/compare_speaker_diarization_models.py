#!/usr/bin/env python3
"""Compare speaker diarization outputs for multiple embedding models.

This is a product-sanity tool, not a DER benchmark. The 14 meeting clips do not
have speaker-turn ground truth, so we compare stability metrics such as speaker
count, segments per minute, short-island count, and secondary-speaker duration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import sherpa_onnx


SAMPLE_RATE = 16000


def decode_audio(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "-",
    ]
    data = subprocess.check_output(cmd)
    return np.frombuffer(data, dtype=np.float32).copy()


def make_diarizer(rawfile: Path, embedding_model: Path, threshold: float) -> sherpa_onnx.OfflineSpeakerDiarization:
    pyannote = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig()
    pyannote.model = str(rawfile / "sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx")

    segmentation = sherpa_onnx.OfflineSpeakerSegmentationModelConfig()
    segmentation.pyannote = pyannote
    segmentation.num_threads = 2
    segmentation.provider = "cpu"
    segmentation.debug = False

    embedding = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
    embedding.model = str(embedding_model)
    embedding.num_threads = 2
    embedding.provider = "cpu"
    embedding.debug = False

    clustering = sherpa_onnx.FastClusteringConfig()
    clustering.num_clusters = -1
    clustering.threshold = threshold

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation,
        embedding,
        clustering,
        0.25,
        0.50,
    )
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def summarize_segments(segments: list[object], duration: float) -> dict[str, object]:
    speaker_durations: dict[int, float] = {}
    short_segments = 0
    overlaps = 0
    previous_end = -1.0
    normalized = []

    for segment in segments:
        start = float(segment.start)
        end = float(segment.end)
        speaker = int(segment.speaker)
        seg_duration = max(0.0, end - start)
        speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + seg_duration
        if seg_duration < 2.0:
            short_segments += 1
        if previous_end >= 0.0 and start < previous_end:
            overlaps += 1
        previous_end = max(previous_end, end)
        normalized.append({"start": round(start, 2), "end": round(end, 2), "speaker": speaker})

    sorted_durations = sorted(speaker_durations.values(), reverse=True)
    primary = sorted_durations[0] if sorted_durations else 0.0
    secondary = sum(sorted_durations[1:])
    speaker_count = len(speaker_durations)
    return {
        "speakers": speaker_count,
        "segments": len(segments),
        "segments_per_min": len(segments) / max(duration / 60.0, 1e-6),
        "short_segments": short_segments,
        "overlaps": overlaps,
        "primary_speaker_sec": primary,
        "secondary_speaker_sec": secondary,
        "secondary_ratio": secondary / max(primary + secondary, 1e-6),
        "speaker_durations": {str(k): round(v, 2) for k, v in sorted(speaker_durations.items())},
        "preview_segments": normalized[:12],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=Path("/Users/cannkit/Downloads/audios"))
    parser.add_argument("--out-dir", type=Path, default=Path("/Users/cannkit/ASR/benchmarks/diarization_68m"))
    parser.add_argument("--rawfile", type=Path, default=Path("/Users/cannkit/ASR/entry/src/main/resources/rawfile"))
    parser.add_argument("--thresholds", default="1.20,1.35,1.50")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    project_root = args.rawfile.parents[4]
    models = [
        ("38M-eres2net-base", args.rawfile / "sherpa-onnx-3dspeaker-sv-zh-cn/model.onnx"),
        (
            "68M-eres2netv2",
            project_root /
            "dev_assets/speaker_models/sherpa-onnx-3dspeaker-eres2netv2-sv-zh-cn-16k-common/model.onnx",
        ),
    ]

    audio_files = sorted(args.audio_dir.glob("chinese_meeting_room_discussion_*.mp3"))
    if not audio_files:
        raise RuntimeError(f"No meeting mp3 files found in {args.audio_dir}")

    diarizers: dict[tuple[str, float], sherpa_onnx.OfflineSpeakerDiarization] = {}
    for model_name, model_path in models:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        for threshold in thresholds:
            diarizers[(model_name, threshold)] = make_diarizer(args.rawfile, model_path, threshold)

    rows = []
    for audio_index, audio in enumerate(audio_files, 1):
        samples = decode_audio(audio)
        duration = len(samples) / SAMPLE_RATE
        print(f"\n[{audio_index:02d}/{len(audio_files):02d}] {audio.name} duration={duration:.1f}s")
        for model_name, model_path in models:
            for threshold in thresholds:
                diarizer = diarizers[(model_name, threshold)]
                started = time.perf_counter()
                result = diarizer.process(samples)
                elapsed = time.perf_counter() - started
                segments = result.sort_by_start_time()
                summary = summarize_segments(segments, duration)
                row = {
                    "audio": audio.name,
                    "model": model_name,
                    "model_path": str(model_path),
                    "threshold": threshold,
                    "duration_sec": duration,
                    "runtime_sec": elapsed,
                    "rtf": elapsed / duration if duration > 0 else 0.0,
                    **summary,
                }
                rows.append(row)
                print(
                    f"  {model_name} th={threshold:.2f} "
                    f"speakers={row['speakers']} segments={row['segments']} "
                    f"short={row['short_segments']} secondary={row['secondary_ratio']:.1%} rtf={row['rtf']:.3f}"
                )

    with (args.out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_fields = [
        "audio",
        "model",
        "threshold",
        "duration_sec",
        "runtime_sec",
        "rtf",
        "speakers",
        "segments",
        "segments_per_min",
        "short_segments",
        "overlaps",
        "primary_speaker_sec",
        "secondary_speaker_sec",
        "secondary_ratio",
    ]
    with (args.out_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in csv_fields})

    write_report(args.out_dir, rows, audio_files, thresholds)
    print(f"\nWrote {args.out_dir / 'report.md'}")
    return 0


def write_report(out_dir: Path, rows: list[dict[str, object]], audio_files: list[Path], thresholds: list[float]) -> None:
    report = out_dir / "report.md"
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), float(row["threshold"])), []).append(row)

    with report.open("w", encoding="utf-8") as f:
        f.write("# Speaker Diarization Embedding A/B\n\n")
        f.write(f"- Audio files: {len(audio_files)}\n")
        f.write("- Segmentation model: `sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx`\n")
        f.write("- Metrics are stability proxies because the 14 meeting clips do not include speaker-turn ground truth.\n\n")

        f.write("## Summary\n\n")
        f.write("| Model | Threshold | Avg speakers | Avg segments | Avg short segments | Avg secondary ratio | Avg RTF |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for key in sorted(grouped):
            items = grouped[key]
            f.write(
                f"| {key[0]} | {key[1]:.2f} | "
                f"{np.mean([float(r['speakers']) for r in items]):.2f} | "
                f"{np.mean([float(r['segments']) for r in items]):.2f} | "
                f"{np.mean([float(r['short_segments']) for r in items]):.2f} | "
                f"{np.mean([float(r['secondary_ratio']) for r in items]):.2%} | "
                f"{np.mean([float(r['rtf']) for r in items]):.3f} |\n"
            )

        f.write("\n## Per Audio At Current Threshold 1.20\n\n")
        f.write("| # | Audio | 38M speakers/segments | 68M speakers/segments | 38M short | 68M short |\n")
        f.write("|---:|---|---:|---:|---:|---:|\n")
        for i, audio in enumerate(audio_files, 1):
            old = next((r for r in rows if r["audio"] == audio.name and r["model"] == "38M-eres2net-base" and abs(float(r["threshold"]) - 1.2) < 1e-6), None)
            new = next((r for r in rows if r["audio"] == audio.name and r["model"] == "68M-eres2netv2" and abs(float(r["threshold"]) - 1.2) < 1e-6), None)
            if old and new:
                f.write(
                    f"| {i} | {audio.name} | {old['speakers']}/{old['segments']} | "
                    f"{new['speakers']}/{new['segments']} | {old['short_segments']} | {new['short_segments']} |\n"
                )

        f.write("\n## Readout\n\n")
        f.write("- Lower average speaker count/short-segment count is not automatically better; it is a proxy for less over-splitting.\n")
        f.write("- If 68M reduces short islands at similar speaker count, it is a safer upgrade.\n")
        f.write("- If both models over-split single-speaker-looking clips, threshold tuning is more important than the embedding model alone.\n")


if __name__ == "__main__":
    raise SystemExit(main())
