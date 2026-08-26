#!/usr/bin/env python3
"""Build the labeled AISHELL-3 open-set voiceprint resources used by the app.

The source manifest is produced by the local model-selection evaluation.  This
script deliberately preserves that protocol:

* 10 enrolled speakers and 10 speakers that are never enrolled;
* five disjoint enrollment utterances per enrolled speaker;
* two query groups per speaker, each averaging three other utterances;
* a concatenated playback WAV for every query group so the phone UI can play
  exactly the three utterances whose embeddings are scored.

The generated manifest records the original speaker id, source relative path,
duration and SHA-256.  It is an auditable test fixture, not a user profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import wave


QUERY_GROUPS = ((5, 6, 7), (8, 9, 10))
SILENCE_SECONDS = 0.25


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wav_info(path: pathlib.Path) -> tuple[int, int, int, int, float]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.getnframes()
    return channels, sample_width, sample_rate, frames, frames / sample_rate


def require_pcm_mono(path: pathlib.Path) -> tuple[int, int, int]:
    channels, sample_width, sample_rate, frames, _ = wav_info(path)
    if channels != 1 or sample_width != 2 or sample_rate <= 0 or frames <= 0:
        raise ValueError(
            f"Expected mono PCM16 WAV, got channels={channels}, "
            f"width={sample_width}, rate={sample_rate}, frames={frames}: {path}"
        )
    return sample_width, sample_rate, frames


def concatenate_wavs(paths: list[pathlib.Path], target: pathlib.Path) -> None:
    if not paths:
        raise ValueError("Cannot concatenate an empty query group")
    first_width, first_rate, _ = require_pcm_mono(paths[0])
    silence = b"\x00" * int(round(first_rate * SILENCE_SECONDS)) * first_width
    payloads: list[bytes] = []
    for path in paths:
        width, rate, _ = require_pcm_mono(path)
        if width != first_width or rate != first_rate:
            raise ValueError(f"Inconsistent WAV format in query group: {path}")
        with wave.open(str(path), "rb") as source:
            payloads.append(source.readframes(source.getnframes()))
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(first_width)
        output.setframerate(first_rate)
        for index, payload in enumerate(payloads):
            if index > 0:
                output.writeframes(silence)
            output.writeframes(payload)


def source_path(data_root: pathlib.Path, relative: str) -> pathlib.Path:
    prefix = "test/wav/"
    if not relative.startswith(prefix):
        raise ValueError(f"Unexpected AISHELL-3 source path: {relative}")
    return data_root / relative.removeprefix(prefix)


def copy_sample(source: pathlib.Path, target: pathlib.Path, source_relative: str) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    require_pcm_mono(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    _, _, sample_rate, frames, duration = wav_info(target)
    return {
        "path": target.name,
        "source": source_relative,
        "sampleRate": sample_rate,
        "frames": frames,
        "durationSec": round(duration, 6),
        "sha256": sha256(target),
    }


def build_speaker(
    kind: str,
    ordinal: int,
    speaker_id: str,
    files: list[str],
    data_root: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict:
    prefix = f"{kind}{ordinal:02d}"
    enrollment: list[dict] = []
    if kind == "known":
        for output_index, source_index in enumerate(range(5), start=1):
            relative = files[source_index]
            source = source_path(data_root, relative)
            target = output_dir / f"{prefix}-enroll{output_index:02d}.wav"
            enrollment.append(copy_sample(source, target, relative))

    query_groups: list[dict] = []
    for group_index, source_indices in enumerate(QUERY_GROUPS, start=1):
        parts: list[dict] = []
        part_paths: list[pathlib.Path] = []
        for part_index, source_index in enumerate(source_indices, start=1):
            relative = files[source_index]
            source = source_path(data_root, relative)
            target = output_dir / f"{prefix}-query{group_index:02d}-part{part_index:02d}.wav"
            parts.append(copy_sample(source, target, relative))
            part_paths.append(target)
        playback = output_dir / f"{prefix}-query{group_index:02d}-playback.wav"
        concatenate_wavs(part_paths, playback)
        _, _, sample_rate, frames, duration = wav_info(playback)
        query_groups.append({
            "query": group_index,
            "parts": parts,
            "playback": {
                "path": playback.name,
                "sampleRate": sample_rate,
                "frames": frames,
                "durationSec": round(duration, 6),
                "sha256": sha256(playback),
            },
        })

    used_sources = [item["source"] for item in enrollment]
    for group in query_groups:
        used_sources.extend(item["source"] for item in group["parts"])
    if len(used_sources) != len(set(used_sources)):
        raise ValueError(f"Enrollment/query leakage for {speaker_id}")

    return {
        "kind": kind,
        "ordinal": ordinal,
        "speakerId": speaker_id,
        "personId": f"aishell3-test-{kind}-{ordinal:02d}" if kind == "known" else None,
        "displayName": f"AISHELL 已知 {ordinal:02d}" if kind == "known" else "未知说话人",
        "enrollment": enrollment,
        "queries": query_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--source-data", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    protocol = source_manifest["protocol"]
    test_known = protocol["test_known"]
    test_unknown = protocol["test_unknown"]
    if len(test_known) != 10 or len(test_unknown) != 10:
        raise ValueError("Device protocol requires exactly 10 known and 10 unknown speakers")
    if set(test_known) & set(test_unknown):
        raise ValueError("Known and unknown speaker sets overlap")
    if protocol["enrollment_indices"] != [0, 1, 2, 3, 4]:
        raise ValueError("Unexpected enrollment split in source manifest")

    by_speaker = {item["speaker"]: item["files"] for item in source_manifest["speakers"]}
    args.output.mkdir(parents=True, exist_ok=True)
    for existing in args.output.glob("*.wav"):
        existing.unlink()

    speakers: list[dict] = []
    for index, speaker_id in enumerate(test_known, start=1):
        speakers.append(build_speaker(
            "known", index, speaker_id, by_speaker[speaker_id], args.source_data, args.output
        ))
    for index, speaker_id in enumerate(test_unknown, start=1):
        speakers.append(build_speaker(
            "unknown", index, speaker_id, by_speaker[speaker_id], args.source_data, args.output
        ))

    manifest = {
        "schemaVersion": 1,
        "dataset": "AISHELL/AISHELL-3",
        "datasetRevision": source_manifest.get("revision", "main"),
        "license": source_manifest.get("license", "Apache-2.0"),
        "split": source_manifest.get("split", "test"),
        "protocol": {
            "knownSpeakers": 10,
            "unknownSpeakers": 10,
            "enrollmentUtterancesPerKnownSpeaker": 5,
            "queryGroupsPerSpeaker": 2,
            "utterancesPerQueryGroup": 3,
            "identityThreshold": 0.60,
            "enrollmentAndQueryDisjoint": True,
            "unknownSpeakersNeverEnrolled": True,
        },
        "speakers": speakers,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    wavs = sorted(args.output.glob("*.wav"))
    total_seconds = sum(wav_info(path)[4] for path in wavs)
    total_bytes = sum(path.stat().st_size for path in wavs)
    print(json.dumps({
        "manifest": str(manifest_path),
        "speakers": len(speakers),
        "wavFiles": len(wavs),
        "audioSecondsIncludingPlaybackCopies": round(total_seconds, 3),
        "bytes": total_bytes,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
