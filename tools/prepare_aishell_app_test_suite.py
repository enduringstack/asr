#!/usr/bin/env python3
"""Build the fixed AISHELL-1 through AISHELL-5 labeled App evaluation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import wave

from huggingface_hub import HfApi, hf_hub_download
import soundfile as sf


SAMPLE_COUNT_PER_DATASET = 20
AISHELL1_OFFICIAL_REPO = "AISHELL/AISHELL-1"
AISHELL1_FILE_REPO = "shenyunhang/AISHELL-1"
AISHELL3_REPO = "AISHELL/AISHELL-3"
AISHELL3_REVISION = "f20d5db4a31fe779ef07bb1af4ea92da5c786622"
AISHELL4_REPO = "AISHELL/AISHELL-4"
AISHELL4_REVISION = "aada72727856313b19d4a030383c426364931dbf"
AISHELL4_SESSION = "S_R004S04C01"
AISHELL5_SOURCE_URL = "https://www.openslr.org/resources/159/Dev.tar.gz"
WINDOW_SECONDS = 12.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wav_metadata(path: Path) -> tuple[int, int, float]:
    with wave.open(str(path), "rb") as source:
        sample_rate = source.getframerate()
        frames = source.getnframes()
        channels = source.getnchannels()
        width = source.getsampwidth()
    if channels != 1 or width != 2:
        raise RuntimeError(f"Expected mono 16-bit PCM WAV: {path}")
    return sample_rate, frames, frames / sample_rate


def remove_generated_wavs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for existing in directory.glob("*.wav"):
        existing.unlink()


def load_aishell1_transcripts(path: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            fields = line.strip().split(maxsplit=1)
            if len(fields) == 2:
                transcripts[fields[0]] = fields[1]
    return transcripts


def stable_rank(seed: str, path: str) -> str:
    return hashlib.sha256(f"{seed}:{path}".encode("utf-8")).hexdigest()


def prepare_aishell1(output_root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    api = HfApi()
    official = api.dataset_info(AISHELL1_OFFICIAL_REPO, files_metadata=False)
    mirror = api.dataset_info(AISHELL1_FILE_REPO, files_metadata=False)
    if not official.sha or not mirror.sha:
        raise RuntimeError("AISHELL-1 revisions are unavailable")
    transcript_path = Path(hf_hub_download(
        AISHELL1_OFFICIAL_REPO,
        "data_aishell/transcript/aishell_transcript_v0.8.txt",
        repo_type="dataset",
        revision=official.sha,
    ))
    transcripts = load_aishell1_transcripts(transcript_path)
    paths = sorted(
        item.rfilename for item in mirror.siblings
        if item.rfilename.startswith("data_aishell/wav/test/")
        and item.rfilename.endswith(".wav")
    )
    by_speaker: dict[str, list[str]] = {}
    for path in paths:
        speaker = PurePosixPath(path).parts[-2]
        by_speaker.setdefault(speaker, []).append(path)
    speakers = sorted(by_speaker)[:10]
    if len(speakers) != 10:
        raise RuntimeError(f"Expected at least 10 AISHELL-1 test speakers, found {len(speakers)}")

    seed = "asr-app-aishell1-test-v2"
    selected: list[str] = []
    used_texts: set[str] = set()
    for speaker in speakers:
        speaker_count = 0
        for path in sorted(by_speaker[speaker], key=lambda value: stable_rank(seed, value)):
            utterance_id = PurePosixPath(path).stem
            reference = "".join(transcripts.get(utterance_id, "").split())
            if not reference or reference in used_texts:
                continue
            selected.append(path)
            used_texts.add(reference)
            speaker_count += 1
            if speaker_count == 2:
                break
    if len(selected) != SAMPLE_COUNT_PER_DATASET:
        raise RuntimeError(f"Expected 20 AISHELL-1 samples, selected {len(selected)}")

    output_dir = output_root / "aishell1"
    remove_generated_wavs(output_dir)
    samples: list[dict[str, object]] = []
    for index, source_path in enumerate(selected, 1):
        utterance_id = PurePosixPath(source_path).stem
        speaker_id = PurePosixPath(source_path).parts[-2]
        cached = Path(hf_hub_download(
            AISHELL1_FILE_REPO, source_path, repo_type="dataset", revision=mirror.sha,
        ))
        filename = f"{index:02d}-{utterance_id}.wav"
        target = output_dir / filename
        shutil.copyfile(cached, target)
        sample_rate, frames, duration = wav_metadata(target)
        samples.append({
            "id": f"aishell1-{utterance_id.lower()}",
            "datasetId": "aishell1",
            "dataset": "AISHELL-1",
            "split": "test",
            "sourceId": utterance_id,
            "audioPath": f"test/asr_aishell_suite/aishell1/{filename}",
            "durationSec": round(duration, 6),
            "reference": transcripts[utterance_id],
            "speakerIds": [speaker_id],
            "speakerTurns": [],
            "condition": "安静室内 · 近讲朗读",
            "license": "Apache-2.0",
            "sha256": file_sha256(target),
            "sampleRate": sample_rate,
            "frames": frames,
            "sourcePath": source_path,
            "supportsAsr": True,
            "supportsVoiceprint": True,
            "supportsDiarization": False,
        })
        print(f"AISHELL-1 {index:02d}/20 {speaker_id} {utterance_id}")
    dataset = {
        "id": "aishell1",
        "name": "AISHELL-1",
        "split": "test",
        "license": "Apache-2.0",
        "source": AISHELL1_OFFICIAL_REPO,
        "revision": official.sha,
        "fileMirror": AISHELL1_FILE_REPO,
        "fileMirrorRevision": mirror.sha,
        "protocol": "10 位说话人，每人 2 条不重复语句，固定哈希抽样",
    }
    return dataset, samples


def load_tab_transcripts(lines: list[str]) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for line in lines:
        fields = line.rstrip("\r\n").split("\t", maxsplit=1)
        if len(fields) == 2 and fields[0] and fields[1]:
            transcripts[fields[0]] = fields[1]
    return transcripts


def prepare_aishell2(output_root: Path, archive: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not archive.is_file():
        raise RuntimeError(
            f"AISHELL-2 iOS test archive is missing: {archive}. "
            "Extract AISHELL-DEV-TEST-SET/iOS/test.tar.gz from the official evaluation ZIP first."
        )
    with tarfile.open(archive, "r:gz") as source:
        transcript_stream = source.extractfile("test/trans.txt")
        if transcript_stream is None:
            raise RuntimeError("AISHELL-2 archive has no test/trans.txt")
        transcripts = load_tab_transcripts(
            transcript_stream.read().decode("utf-8").splitlines()
        )
        audio_members = [
            member for member in source.getmembers()
            if member.isfile() and member.name.startswith("test/wav/") and member.name.endswith(".wav")
        ]
        by_speaker: dict[str, list[tarfile.TarInfo]] = {}
        for member in audio_members:
            speaker = PurePosixPath(member.name).parts[-2]
            by_speaker.setdefault(speaker, []).append(member)
        speakers = sorted(by_speaker)
        if len(speakers) != 10:
            raise RuntimeError(f"Expected 10 AISHELL-2 test speakers, found {len(speakers)}")

        seed = "asr-app-aishell2-ios-test-v1"
        selected: list[tarfile.TarInfo] = []
        used_texts: set[str] = set()
        for speaker in speakers:
            count = 0
            ranked = sorted(by_speaker[speaker], key=lambda item: stable_rank(seed, item.name))
            for member in ranked:
                utterance_id = PurePosixPath(member.name).stem
                reference = "".join(transcripts.get(utterance_id, "").split())
                if not reference or reference in used_texts:
                    continue
                selected.append(member)
                used_texts.add(reference)
                count += 1
                if count == 2:
                    break
        if len(selected) != SAMPLE_COUNT_PER_DATASET:
            raise RuntimeError(f"Expected 20 AISHELL-2 samples, selected {len(selected)}")

        output_dir = output_root / "aishell2"
        remove_generated_wavs(output_dir)
        samples: list[dict[str, object]] = []
        for index, member in enumerate(selected, 1):
            utterance_id = PurePosixPath(member.name).stem
            speaker_id = PurePosixPath(member.name).parts[-2]
            filename = f"{index:02d}-{utterance_id}.wav"
            target = output_dir / filename
            extracted = source.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Cannot extract AISHELL-2 audio: {member.name}")
            with target.open("wb") as sink:
                shutil.copyfileobj(extracted, sink)
            sample_rate, frames, duration = wav_metadata(target)
            samples.append({
                "id": f"aishell2-{utterance_id.lower()}",
                "datasetId": "aishell2",
                "dataset": "AISHELL-2",
                "split": "AISHELL-2018A-EVAL test / iOS",
                "sourceId": utterance_id,
                "audioPath": f"test/asr_aishell_suite/aishell2/{filename}",
                "durationSec": round(duration, 6),
                "reference": transcripts[utterance_id],
                "speakerIds": [speaker_id],
                "speakerTurns": [],
                "condition": "手机近讲 · iOS 设备录音",
                "license": "Apache-2.0",
                "sha256": file_sha256(target),
                "sampleRate": sample_rate,
                "frames": frames,
                "sourcePath": member.name,
                "supportsAsr": True,
                "supportsVoiceprint": True,
                "supportsDiarization": False,
            })
            print(f"AISHELL-2 {index:02d}/20 {speaker_id} {utterance_id}")
    dataset = {
        "id": "aishell2",
        "name": "AISHELL-2",
        "split": "AISHELL-2018A-EVAL test / iOS",
        "license": "Apache-2.0",
        "source": "https://www.aishelltech.com/aishell_2018_eval",
        "archive": "AISHELL-DEV-TEST-SET/iOS/test.tar.gz",
        "archiveSha256": file_sha256(archive),
        "protocol": "官方公开评测集的 10 位说话人，每人 2 条不重复语句，固定哈希抽样",
    }
    return dataset, samples


def load_aishell3_text(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            fields = line.rstrip("\n").split("\t", maxsplit=1)
            if len(fields) != 2:
                continue
            utterance_id = Path(fields[0]).stem
            tokens = fields[1].split()
            result[utterance_id] = "".join(tokens[0::2])
    return result


def prepare_aishell3(project_root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    voiceprint_root = project_root / "entry/src/main/resources/rawfile/test/aishell3_voiceprint"
    source_manifest = json.loads((voiceprint_root / "manifest.json").read_text(encoding="utf-8"))
    content_path = Path(hf_hub_download(
        AISHELL3_REPO, "test/content.txt", repo_type="dataset", revision=AISHELL3_REVISION,
    ))
    transcripts = load_aishell3_text(content_path)
    samples: list[dict[str, object]] = []
    known_speakers = [speaker for speaker in source_manifest["speakers"] if speaker["kind"] == "known"]
    for speaker in known_speakers[:10]:
        for query in speaker["queries"][:2]:
            part_ids = [Path(part["source"]).stem for part in query["parts"]]
            missing = [part_id for part_id in part_ids if part_id not in transcripts]
            if missing:
                raise RuntimeError(f"Missing AISHELL-3 transcripts: {missing}")
            relative_path = f"test/aishell3_voiceprint/{query['playback']['path']}"
            local_path = project_root / "entry/src/main/resources/rawfile" / relative_path
            _, _, duration = wav_metadata(local_path)
            samples.append({
                "id": f"aishell3-{speaker['speakerId'].lower()}-query{query['query']:02d}",
                "datasetId": "aishell3",
                "dataset": "AISHELL-3",
                "split": "test",
                "sourceId": "+".join(part_ids),
                "audioPath": relative_path,
                "durationSec": round(duration, 6),
                "reference": "".join(transcripts[part_id] for part_id in part_ids),
                "speakerIds": [speaker["speakerId"]],
                "speakerTurns": [],
                "condition": "高保真录音 · 3 句拼接",
                "license": "Apache-2.0",
                "sha256": file_sha256(local_path),
                "sourcePaths": [part["source"] for part in query["parts"]],
                "supportsAsr": True,
                "supportsVoiceprint": True,
                "supportsDiarization": False,
            })
    if len(samples) != SAMPLE_COUNT_PER_DATASET:
        raise RuntimeError(f"Expected 20 AISHELL-3 samples, selected {len(samples)}")
    dataset = {
        "id": "aishell3",
        "name": "AISHELL-3",
        "split": "test",
        "license": "Apache-2.0",
        "source": AISHELL3_REPO,
        "revision": AISHELL3_REVISION,
        "protocol": "10 位已知说话人，每人 2 组；每组由 3 条同人语句拼接",
    }
    return dataset, samples


def parse_textgrid(path: Path) -> tuple[float, list[dict[str, object]]]:
    """Parse the IntervalTiers used by AISHELL-4/5 without adding a Praat dependency."""
    intervals: list[dict[str, object]] = []
    speaker_id = ""
    in_interval = False
    start = 0.0
    end = 0.0
    maximum = 0.0
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if line.startswith("xmax = ") and not in_interval:
            try:
                maximum = max(maximum, float(line.split("=", maxsplit=1)[1].strip()))
            except ValueError:
                pass
        name_match = re.fullmatch(r'name = "(.*)"', line)
        if name_match:
            speaker_id = name_match.group(1).replace('""', '"')
            continue
        if re.fullmatch(r"intervals \[\d+\]:", line):
            in_interval = True
            start = 0.0
            end = 0.0
            continue
        if not in_interval:
            continue
        if line.startswith("xmin = "):
            start = float(line.split("=", maxsplit=1)[1].strip())
            continue
        if line.startswith("xmax = "):
            end = float(line.split("=", maxsplit=1)[1].strip())
            continue
        text_match = re.fullmatch(r'text = "(.*)"', line)
        if text_match:
            text = text_match.group(1).replace('""', '"').strip()
            if speaker_id and text and end > start:
                intervals.append({
                    "speakerId": speaker_id,
                    "startSec": start,
                    "endSec": end,
                    "text": text,
                })
            in_interval = False
    intervals.sort(key=lambda value: (
        float(value["startSec"]), float(value["endSec"]), str(value["speakerId"])
    ))
    return maximum, intervals


def build_labeled_windows(intervals: list[dict[str, object]], maximum: float,
                          seed: str) -> list[tuple[float, float, list[dict[str, object]]]]:
    """Choose non-overlapping windows whose boundaries lie in corpus-labeled silence."""
    active = sorted(
        (float(item["startSec"]), float(item["endSec"])) for item in intervals
    )
    merged: list[tuple[float, float]] = []
    for start, end in active:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    boundaries: list[float] = [0.0]
    for index in range(len(merged) - 1):
        gap_start = merged[index][1]
        gap_end = merged[index + 1][0]
        if gap_end - gap_start >= 0.20:
            boundaries.append((gap_start + gap_end) / 2.0)
    boundaries.append(maximum)

    candidates: list[tuple[float, float, list[dict[str, object]], int, int, float, str]] = []
    for start_index, window_start in enumerate(boundaries[:-1]):
        possible_ends = [
            value for value in boundaries[start_index + 1:]
            if 8.0 <= value - window_start <= 18.0
        ]
        if not possible_ends:
            continue
        for window_end in possible_ends:
            turns = [
                item for item in intervals
                if float(item["startSec"]) >= window_start
                and float(item["endSec"]) <= window_end
            ]
            if not turns:
                continue
            speaker_count = len({str(item["speakerId"]) for item in turns})
            character_count = sum(len(str(item["text"])) for item in turns)
            coverage = sum(float(item["endSec"]) - float(item["startSec"]) for item in turns)
            tie = stable_rank(seed, f"{window_start:.3f}-{window_end:.3f}")
            candidates.append((window_start, window_end, turns, speaker_count,
                               character_count, coverage, tie))

    candidates.sort(key=lambda value: (
        -value[3], -value[4], -value[5], abs((value[1] - value[0]) - WINDOW_SECONDS), value[6]
    ))
    selected: list[tuple[float, float, list[dict[str, object]]]] = []
    for window_start, window_end, turns, _speakers, _characters, _coverage, _tie in candidates:
        overlaps = any(
            window_start < selected_end and window_end > selected_start
            for selected_start, selected_end, _selected_turns in selected
        )
        if overlaps:
            continue
        selected.append((window_start, window_end, turns))
        if len(selected) == SAMPLE_COUNT_PER_DATASET:
            break
    if len(selected) != SAMPLE_COUNT_PER_DATASET:
        raise RuntimeError(f"Expected 20 labeled windows, selected {len(selected)}")
    selected.sort(key=lambda value: value[0])
    return selected


def write_window_samples(dataset_id: str, dataset_name: str, split: str, session_id: str,
                         condition: str, license_name: str, audio_path: Path,
                         windows: list[tuple[float, float, list[dict[str, object]]]],
                         output_root: Path, channel: int = 0) -> list[dict[str, object]]:
    output_dir = output_root / dataset_id
    remove_generated_wavs(output_dir)
    samples: list[dict[str, object]] = []
    with sf.SoundFile(audio_path) as source:
        sample_rate = source.samplerate
        if sample_rate != 16000:
            raise RuntimeError(f"Expected 16 kHz {dataset_name} audio, found {sample_rate} Hz")
        if channel < 0 or channel >= source.channels:
            raise RuntimeError(f"Audio channel {channel} is unavailable in {audio_path}")
        for index, (window_start, window_end, source_turns) in enumerate(windows, 1):
            start_frame = round(window_start * sample_rate)
            end_frame = round(window_end * sample_rate)
            source.seek(start_frame)
            audio = source.read(end_frame - start_frame, dtype="float32", always_2d=True)[:, channel]
            filename = f"{index:02d}-{session_id}-{window_start:07.3f}.wav"
            target = output_dir / filename
            sf.write(target, audio, sample_rate, subtype="PCM_16", format="WAV")
            turns: list[dict[str, object]] = []
            for turn in source_turns:
                turns.append({
                    "speakerId": str(turn["speakerId"]),
                    "startSec": round(float(turn["startSec"]) - window_start, 3),
                    "endSec": round(float(turn["endSec"]) - window_start, 3),
                    "text": str(turn["text"]),
                })
            speaker_ids = sorted({str(turn["speakerId"]) for turn in turns})
            reference = "".join(str(turn["text"]) for turn in turns)
            _, frames, duration = wav_metadata(target)
            samples.append({
                "id": f"{dataset_id}-{session_id.lower()}-{index:02d}",
                "datasetId": dataset_id,
                "dataset": dataset_name,
                "split": split,
                "sourceId": f"{session_id}@{window_start:.3f}-{window_end:.3f}",
                "audioPath": f"test/asr_aishell_suite/{dataset_id}/{filename}",
                "durationSec": round(duration, 6),
                "reference": reference,
                "speakerIds": speaker_ids,
                "speakerTurns": turns,
                "condition": condition,
                "license": license_name,
                "sha256": file_sha256(target),
                "sampleRate": sample_rate,
                "frames": frames,
                "sourceAudio": audio_path.name,
                "sourceStartSec": round(window_start, 6),
                "sourceEndSec": round(window_end, 6),
                "sourceChannel": channel + 1,
                "supportsAsr": True,
                "supportsVoiceprint": True,
                "supportsDiarization": True,
            })
            print(
                f"{dataset_name} {index:02d}/20 {window_start:.3f}-{window_end:.3f} "
                f"speakers={','.join(speaker_ids)}"
            )
    return samples


def prepare_aishell4(output_root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    textgrid = Path(hf_hub_download(
        AISHELL4_REPO, f"test/TextGrid/{AISHELL4_SESSION}.TextGrid",
        repo_type="dataset", revision=AISHELL4_REVISION,
    ))
    audio = Path(hf_hub_download(
        AISHELL4_REPO, f"test/wav/{AISHELL4_SESSION}.flac",
        repo_type="dataset", revision=AISHELL4_REVISION,
    ))
    maximum, intervals = parse_textgrid(textgrid)
    windows = build_labeled_windows(intervals, maximum, "asr-app-aishell4-test-v1")
    samples = write_window_samples(
        "aishell4", "AISHELL-4", "test", AISHELL4_SESSION,
        "真实会议 · 8 通道阵列第 1 通道", "CC BY-SA 4.0", audio, windows, output_root,
    )
    dataset = {
        "id": "aishell4",
        "name": "AISHELL-4",
        "split": "test",
        "license": "CC BY-SA 4.0",
        "source": AISHELL4_REPO,
        "revision": AISHELL4_REVISION,
        "session": AISHELL4_SESSION,
        "sourceAudioSha256": file_sha256(audio),
        "protocol": "在全体说话人均静音的边界切出 20 个互不重叠片段，保留 TextGrid 逐段真值",
    }
    return dataset, samples


def copy_tar_member(source: tarfile.TarFile, member_name: str, target: Path) -> None:
    member = source.getmember(member_name)
    if not member.isfile():
        raise RuntimeError(f"AISHELL-5 archive member is not a file: {member_name}")
    extracted = source.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"Cannot read AISHELL-5 archive member: {member_name}")
    with target.open("wb") as sink:
        shutil.copyfileobj(extracted, sink)


def prepare_aishell5(output_root: Path, archive: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not archive.is_file():
        raise RuntimeError(f"AISHELL-5 development archive is missing: {archive}")
    session_id = "001"
    microphone_id = "DX01C01"
    with tempfile.TemporaryDirectory(prefix="aishell5-selected-") as temporary:
        temp_root = Path(temporary)
        textgrid = temp_root / f"{microphone_id}.TextGrid"
        audio = temp_root / f"{microphone_id}.wav"
        with tarfile.open(archive, "r:gz") as source:
            copy_tar_member(source, f"Dev/{session_id}/{microphone_id}.TextGrid", textgrid)
            copy_tar_member(source, f"Dev/{session_id}/{microphone_id}.wav", audio)
        maximum, intervals = parse_textgrid(textgrid)
        windows = build_labeled_windows(intervals, maximum, "asr-app-aishell5-dev-v1")
        samples = write_window_samples(
            "aishell5", "AISHELL-5", "Dev", f"{session_id}-{microphone_id}",
            "真实车内双人会话 · 车载麦克风 1", "CC BY-SA 4.0",
            audio, windows, output_root,
        )
        source_audio_sha256 = file_sha256(audio)
    dataset = {
        "id": "aishell5",
        "name": "AISHELL-5",
        "split": "Dev",
        "license": "CC BY-SA 4.0",
        "source": "https://www.openslr.org/159/",
        "archiveUrl": AISHELL5_SOURCE_URL,
        "archiveSha256": file_sha256(archive),
        "session": session_id,
        "microphone": microphone_id,
        "sourceAudioSha256": source_audio_sha256,
        "protocol": "在全体说话人均静音的边界切出 20 个互不重叠片段，保留 TextGrid 逐段真值",
    }
    return dataset, samples


def ets_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_ets(samples: list[dict[str, object]], output: Path) -> None:
    lines = [
        "// Generated by tools/prepare_aishell_app_test_suite.py. Do not edit by hand.",
        "",
        "export interface GroundTruthSpeakerTurn {",
        "  speakerId: string;",
        "  startSec: number;",
        "  endSec: number;",
        "  text: string;",
        "}",
        "",
        "export interface AsrTestCase {",
        "  id: string;",
        "  datasetId: string;",
        "  dataset: string;",
        "  split: string;",
        "  sourceId: string;",
        "  speakerIds: string[];",
        "  speakerTurns: GroundTruthSpeakerTurn[];",
        "  reference: string;",
        "  audioPath: string;",
        "  durationSec: number;",
        "  condition: string;",
        "  license: string;",
        "  supportsAsr: boolean;",
        "  supportsVoiceprint: boolean;",
        "  supportsDiarization: boolean;",
        "}",
        "",
        "export function buildAsrTestCases(): AsrTestCase[] {",
        "  return [",
    ]
    for sample in samples:
        speaker_ids = ", ".join(ets_string(str(value)) for value in sample["speakerIds"])
        turns = sample["speakerTurns"]
        lines.extend([
            "    {",
            f"      id: {ets_string(str(sample['id']))},",
            f"      datasetId: {ets_string(str(sample['datasetId']))},",
            f"      dataset: {ets_string(str(sample['dataset']))},",
            f"      split: {ets_string(str(sample['split']))},",
            f"      sourceId: {ets_string(str(sample['sourceId']))},",
            f"      speakerIds: [{speaker_ids}],",
            "      speakerTurns: [",
        ])
        for turn in turns:
            lines.append(
                "        { speakerId: %s, startSec: %.3f, endSec: %.3f, text: %s },"
                % (ets_string(str(turn["speakerId"])), float(turn["startSec"]),
                   float(turn["endSec"]), ets_string(str(turn["text"])))
            )
        lines.extend([
            "      ],",
            f"      reference: {ets_string(str(sample['reference']))},",
            f"      audioPath: {ets_string(str(sample['audioPath']))},",
            f"      durationSec: {float(sample['durationSec']):.6f},",
            f"      condition: {ets_string(str(sample['condition']))},",
            f"      license: {ets_string(str(sample['license']))},",
            f"      supportsAsr: {str(bool(sample['supportsAsr'])).lower()},",
            f"      supportsVoiceprint: {str(bool(sample['supportsVoiceprint'])).lower()},",
            f"      supportsDiarization: {str(bool(sample['supportsDiarization'])).lower()},",
            "    },",
        ])
    lines.extend(["  ];", "}", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def write_suite(output_root: Path, ets_output: Path,
                datasets: list[dict[str, object]], samples: list[dict[str, object]]) -> None:
    manifest = {
        "schemaVersion": 2,
        "language": "zh-CN",
        "sampleCountPerDataset": SAMPLE_COUNT_PER_DATASET,
        "predictionBundled": False,
        "datasets": datasets,
        "samples": samples,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "ATTRIBUTION.md").write_text(
        "# AISHELL-1～5 中文语音固定测试集\n\n"
        "每套数据选择 20 组，保留转写、说话人和可用的时间段真值。"
        "来源、版本、抽样协议、许可证和文件 SHA-256 见 `manifest.json`。\n\n"
        "App 不包含预计算的模型预测；所有输出均由端侧模型现场产生。\n",
        encoding="utf-8",
    )
    write_ets(samples, ets_output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("entry/src/main/resources/rawfile/test/asr_aishell_suite"),
    )
    parser.add_argument(
        "--ets-output", type=Path,
        default=Path("entry/src/main/ets/common/AsrTestData.ets"),
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["aishell1", "aishell3"],
        choices=["aishell1", "aishell2", "aishell3", "aishell4", "aishell5"],
    )
    parser.add_argument(
        "--aishell2-archive", type=Path,
        default=Path("/tmp/asr-aishell-suite-20260903/aishell2-ios/test.tar.gz"),
    )
    parser.add_argument(
        "--aishell5-archive", type=Path,
        default=Path("/tmp/asr-aishell-suite-20260903/AISHELL5-Dev.tar.gz"),
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    datasets: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    for dataset_id in args.datasets:
        if dataset_id == "aishell1":
            dataset, selected = prepare_aishell1(args.output_root)
        elif dataset_id == "aishell2":
            dataset, selected = prepare_aishell2(args.output_root, args.aishell2_archive)
        elif dataset_id == "aishell3":
            dataset, selected = prepare_aishell3(project_root)
        elif dataset_id == "aishell4":
            dataset, selected = prepare_aishell4(args.output_root)
        else:
            dataset, selected = prepare_aishell5(args.output_root, args.aishell5_archive)
        datasets.append(dataset)
        samples.extend(selected)
    write_suite(args.output_root, args.ets_output, datasets, samples)
    print(f"wrote {len(samples)} samples across {len(datasets)} datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
