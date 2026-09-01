#!/usr/bin/env python3
"""Prepare a reproducible, source-grouped acoustic-scene corpus.

The script downloads only selected entries from large Zenodo ZIP archives by
using HTTP ranges. Full source corpora are never copied into the repository.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import requests
import soundfile as sf
from remotezip import RemoteZip

from aporee_source import (
    AporeeCandidate,
    CONTINUOUS_SCENE_OFFSETS_SECONDS,
    fetch_audio_window,
    has_minimum_audio,
    is_commercial_license,
    load_candidates,
)
from dataset_protocol import (
    SPLITS,
    frozen_split_map,
    map_official_scene,
    round_robin_limit,
    split_source_group,
    stratified_split_map,
    summarize_split_counts,
    validate_source_isolation,
)


SAMPLE_RATE = 32_000
WINDOW_SAMPLES = SAMPLE_RATE * 10
PREPARED_DIR = "prepared-32k"
FIXED_TEST_DIR = "fixed_tests-32k"
APOREE_WINDOW_DIR = "aporee-windows-32k"
APOREE_DOWNLOAD_WORKERS = 12
CLASSES = [
    "metro",
    "high_speed_train",
    "shopping_mall",
    "cafe_restaurant",
    "concert",
    "other",
]
COMMERCIAL_CC = {
    "http://creativecommons.org/publicdomain/zero/1.0/",
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/licenses/by/3.0/",
    "https://creativecommons.org/licenses/by/3.0/",
    "http://creativecommons.org/licenses/by/4.0/",
    "https://creativecommons.org/licenses/by/4.0/",
}

TAU_RECORD = "6337421"
TAU_PREFIX = "TAU-urban-acoustic-scenes-2022-mobile-development"
TAU_META_URL = (
    "https://zenodo.org/api/records/6337421/files/"
    "TAU-urban-acoustic-scenes-2022-mobile-development.meta.zip/content"
)
TAU_ARCHIVES = [
    f"https://zenodo.org/api/records/{TAU_RECORD}/files/"
    f"{TAU_PREFIX}.audio.{index}.zip/content"
    for index in range(1, 17)
]
TAU_MIRROR_REPO = "quinnlue/tau-urban-acoustic-scenes-2022-mobile-split5"
TAU_MIRROR_REVISION = "f8279077c020195bc8828ffe11e8a2647929d3a6"
TAU_MIRROR_RAW = (
    f"https://huggingface.co/datasets/{TAU_MIRROR_REPO}/resolve/"
    f"{TAU_MIRROR_REVISION}"
)

TUT_RECORD = "1040168"
TUT_PREFIX = "TUT-acoustic-scenes-2017-evaluation"
TUT_META_URL = (
    "https://zenodo.org/api/records/1040168/files/"
    "TUT-acoustic-scenes-2017-evaluation.meta.zip/content"
)
TUT_ARCHIVES = [
    f"https://zenodo.org/api/records/{TUT_RECORD}/files/"
    f"{TUT_PREFIX}.audio.{index}.zip/content"
    for index in range(1, 5)
]

TUT_DEV_RECORD = "400515"
TUT_DEV_PREFIX = "TUT-acoustic-scenes-2017-development"
TUT_DEV_META_URL = (
    "https://zenodo.org/api/records/400515/files/"
    "TUT-acoustic-scenes-2017-development.meta.zip/content"
)
TUT_DEV_ARCHIVES = [
    f"https://zenodo.org/api/records/{TUT_DEV_RECORD}/files/"
    f"{TUT_DEV_PREFIX}.audio.{index}.zip/content"
    for index in range(1, 11)
]

FSD_REPO = "quinnlue/FSD50K-16k"
FSD_REVISION = "2a60d475f4e2f0db624a902881af2df79d656145"
FSD_RAW = f"https://huggingface.co/datasets/{FSD_REPO}/resolve/main"
FSD_INFO = f"{FSD_RAW}/metadata/original/dev_clips_info_FSD50K.json"
FSD_LABELS = f"{FSD_RAW}/metadata/labels/dev.csv"
APOREE_CACHE = "metadata/aporee/candidates-v4.json"


@dataclass
class ManifestItem:
    id: str
    path: str
    label: str
    split: str
    dataset: str
    source_id: str
    source_group: str
    subscene: str
    truth_basis: str
    license: str
    attribution: str
    sha256: str


@dataclass
class SourceSelection:
    id: str
    label: str
    split: str
    dataset: str
    source_id: str
    source_group: str
    subscene: str
    truth_basis: str
    license: str
    attribution: str
    source_paths: list[str]
    mix_source_paths: list[str] | None = None


def stable_bucket(value: str, modulo: int = 5) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % modulo


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> Path:
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(12):
        try:
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            output.write(block)
            temporary.replace(target)
            return target
        except (requests.RequestException, OSError):
            temporary.unlink(missing_ok=True)
            if attempt == 11:
                raise
            time.sleep(min(60, 2 ** min(attempt, 5)))
    raise RuntimeError(f"unable to download {url}")


def ensure_metadata(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    downloads = root / "downloads"
    tau_zip = download(TAU_META_URL, downloads / "tau2022-meta.zip")
    tut_zip = download(TUT_META_URL, downloads / "tut2017-meta.zip")
    tut_dev_zip = download(TUT_DEV_META_URL, downloads / "tut2017-dev-meta.zip")
    tau_dir = root / "metadata" / "tau2022"
    tut_dir = root / "metadata" / "tut2017"
    tut_dev_dir = root / "metadata" / "tut2017-dev"
    if not (tau_dir / TAU_PREFIX / "meta.csv").exists():
        shutil.unpack_archive(tau_zip, tau_dir)
    if not (tut_dir / TUT_PREFIX / "evaluation_setup" / "evaluate.txt").exists():
        shutil.unpack_archive(tut_zip, tut_dir)
    if not (tut_dev_dir / TUT_DEV_PREFIX / "meta.txt").exists():
        shutil.unpack_archive(tut_dev_zip, tut_dev_dir)

    fsd_dir = root / "metadata" / "fsd50k"
    fsd_info = download(FSD_INFO, fsd_dir / "dev_clips_info_FSD50K.json")
    fsd_labels = download(FSD_LABELS, fsd_dir / "dev.csv")
    return (
        tau_dir / TAU_PREFIX,
        tut_dir / TUT_PREFIX,
        tut_dev_dir / TUT_DEV_PREFIX,
        fsd_info,
        fsd_labels,
    )


def limited_per_stratum(
    rows: Iterable[SourceSelection],
    limits: dict[tuple[str, str, str], int],
) -> list[SourceSelection]:
    grouped: dict[tuple[str, str, str], list[SourceSelection]] = {}
    for row in rows:
        key = (row.label, row.subscene, row.split)
        if limits.get(key, 0) > 0:
            grouped.setdefault(key, []).append(row)
    selected: list[SourceSelection] = []
    for key in sorted(grouped):
        selected.extend(round_robin_limit(grouped[key], limits[key]))
    return selected


def tau_group_key(filename: str) -> str:
    return re.sub(r"-\d+-[^-]+\.wav$", "", filename)


def build_tau_selections(
    tau_dir: Path, available_root: Path | None = None
) -> list[SourceSelection]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with (tau_dir / "meta.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["source_label"] != "a":
                continue
            grouped.setdefault(tau_group_key(row["filename"]), []).append(row)

    rows: list[SourceSelection] = []
    for group_key, group_rows in grouped.items():
        if len(group_rows) != 10:
            continue
        scene = group_rows[0]["scene_label"]
        label = map_official_scene(scene)
        location = group_rows[0]["identifier"]
        source_group = f"tau-location:{location}"
        rows.append(SourceSelection(
            id=f"tau-{Path(group_key).name}",
            label=label,
            split=split_source_group(source_group),
            dataset="TAU Urban Acoustic Scenes 2022 Mobile",
            source_id=group_key,
            source_group=source_group,
            subscene=scene,
            truth_basis=f"官方文件级场景标签：{scene}",
            license="other-nc",
            attribution="Tampere University / TAU Urban Acoustic Scenes 2022 Mobile",
            source_paths=sorted(item["filename"] for item in group_rows),
        ))
    if available_root is not None:
        rows = [
            row for row in rows
            if all(
                (available_root / path.removeprefix("audio/")).exists()
                and (available_root / path.removeprefix("audio/")).stat().st_size > 0
                for path in row.source_paths
            )
        ]
    split_by_group = stratified_split_map(
        (row.source_group, row.subscene) for row in rows
    )
    for row in rows:
        row.split = split_by_group[row.source_group]
    limits: dict[tuple[str, str, str], int] = {}
    for scene, label in (("metro", "metro"), ("shopping_mall", "shopping_mall")):
        limits[(label, scene, "train")] = 300
        limits[(label, scene, "calibration")] = 80
        limits[(label, scene, "test")] = 80
    negative_scenes = {
        row.subscene for row in rows if row.label == "other"
    }
    for scene in negative_scenes:
        limits[("other", scene, "train")] = 40
        limits[("other", scene, "calibration")] = 15
        limits[("other", scene, "test")] = 15
    return limited_per_stratum(rows, limits)


def tut_source_group(source_path: str) -> str:
    name = Path(source_path).stem
    return re.sub(r"_\d+_\d+$", "", name)


def build_tut_evaluation_selections(tut_dir: Path) -> list[SourceSelection]:
    evaluate_path = tut_dir / "evaluation_setup" / "evaluate.txt"
    map_path = tut_dir / "evaluation_setup" / "map.txt"
    labels: dict[str, str] = {}
    with evaluate_path.open(encoding="utf-8") as stream:
        for line in stream:
            filename, label = line.rstrip("\n").split("\t")
            labels[filename] = label
    original_by_eval: dict[str, str] = {}
    with map_path.open(encoding="utf-8") as stream:
        for line in stream:
            original, evaluated = line.rstrip("\n").split("\t")
            original_by_eval[evaluated] = original

    rows: list[SourceSelection] = []
    for filename, scene in labels.items():
        label = map_official_scene(scene)
        original = original_by_eval.get(filename, filename)
        group = tut_source_group(original)
        source_group = f"tut2017-eval-recording:{group}"
        rows.append(SourceSelection(
            id=f"tut-eval-{Path(filename).stem}",
            label=label,
            split=split_source_group(source_group),
            dataset="TUT Acoustic Scenes 2017 Evaluation",
            source_id=original,
            source_group=source_group,
            subscene=scene,
            truth_basis=f"官方文件级场景标签：{scene}",
            license="other-nc",
            attribution="Tampere University / TUT Acoustic Scenes 2017",
            source_paths=[filename],
        ))
    return rows


def build_tut_development_selections(tut_dir: Path) -> list[SourceSelection]:
    rows: list[SourceSelection] = []
    with (tut_dir / "meta.txt").open(encoding="utf-8") as stream:
        for line in stream:
            filename, scene, recording = line.rstrip("\n").split("\t")
            label = map_official_scene(scene)
            source_group = f"tut2017-dev-recording:{recording}"
            rows.append(SourceSelection(
                id=f"tut-dev-{Path(filename).stem}",
                label=label,
                split=split_source_group(source_group),
                dataset="TUT Acoustic Scenes 2017 Development",
                source_id=filename,
                source_group=source_group,
                subscene=scene,
                truth_basis=f"官方文件级场景标签：{scene}",
                license="other-nc",
                attribution="Tampere University / TUT Acoustic Scenes 2017",
                source_paths=[filename],
            ))
    return rows


def limit_tut_selections(rows: list[SourceSelection]) -> list[SourceSelection]:
    split_by_group = stratified_split_map(
        (row.source_group, row.subscene) for row in rows
    )
    for row in rows:
        row.split = split_by_group[row.source_group]
    limits: dict[tuple[str, str, str], int] = {}
    for scene, label in (
        ("cafe/restaurant", "cafe_restaurant"),
        ("train", "high_speed_train"),
    ):
        limits[(label, scene, "train")] = 180
        limits[(label, scene, "calibration")] = 60
        limits[(label, scene, "test")] = 60
    negative_scenes = {
        row.subscene for row in rows if row.label == "other"
    }
    for scene in negative_scenes:
        limits[("other", scene, "train")] = 15
        limits[("other", scene, "calibration")] = 5
        limits[("other", scene, "test")] = 5
    return limited_per_stratum(rows, limits)


def text_for_fsd(info: dict[str, object]) -> str:
    return " ".join([
        str(info.get("title", "")),
        str(info.get("description", "")),
        " ".join(str(value) for value in info.get("tags", [])),
    ])


def build_fsd_index(labels_path: Path) -> tuple[dict[int, tuple[str, int, set[str]]], dict[int, set[str]]]:
    by_split: dict[str, list[tuple[int, set[str]]]] = {"train": [], "val": []}
    labels_by_id: dict[int, set[str]] = {}
    with labels_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source_id = int(row["fname"])
            labels = set(row["labels"].split(","))
            labels_by_id[source_id] = labels
            by_split[row["split"]].append((source_id, labels))
    index: dict[int, tuple[str, int, set[str]]] = {}
    for split, values in by_split.items():
        public_split = "validation" if split == "val" else "train"
        for row_index, (source_id, labels) in enumerate(sorted(values)):
            index[source_id] = (public_split, row_index, labels)
    return index, labels_by_id


def select_fsd_ids(info_path: Path, labels_path: Path) -> tuple[list[int], list[int], list[int], list[int]]:
    info: dict[str, dict[str, object]] = json.loads(info_path.read_text(encoding="utf-8"))
    _, labels_by_id = build_fsd_index(labels_path)
    high_speed: list[int] = []
    concert: list[int] = []
    music: list[int] = []
    crowd: list[int] = []
    high_speed_pattern = re.compile(
        r"high[- ]?speed|very fast|relatively high speed|fast moving train|fast train|"
        r"passes? express|express no\.?|full speed|speed 80 km/h|\bice passenger|"
        r"\bice train|\btgv\b",
        re.IGNORECASE,
    )
    concert_pattern = re.compile(
        r"concert|jazz[- ]?club|music[- ]?festival|recital|live[- ]music|gig",
        re.IGNORECASE,
    )
    for source_id_text, metadata in info.items():
        source_id = int(source_id_text)
        license_name = str(metadata.get("license", ""))
        if license_name not in COMMERCIAL_CC:
            continue
        labels = labels_by_id.get(source_id, set())
        text = text_for_fsd(metadata)
        if "Train" in labels and high_speed_pattern.search(text):
            high_speed.append(source_id)
        if labels.intersection({"Crowd", "Applause"}) and concert_pattern.search(text):
            concert.append(source_id)
        if "Music" in labels and not labels.intersection({"Crowd", "Applause"}):
            music.append(source_id)
        if labels.intersection({"Crowd", "Applause"}) and source_id not in concert:
            crowd.append(source_id)
    return sorted(high_speed), sorted(concert), sorted(music), sorted(crowd)


def resolve_fsd_audio_url(source_id: int, index: dict[int, tuple[str, int, set[str]]]) -> str:
    split, row_index, _ = index[source_id]
    endpoint = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": FSD_REPO,
        "config": "default",
        "split": split,
        "offset": row_index,
        "length": 1,
    }
    for attempt in range(12):
        try:
            response = requests.get(endpoint, params=params, timeout=(30, 180))
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "0") or 0)
                time.sleep(max(retry_after, min(60, 2 ** min(attempt, 5))))
                continue
            response.raise_for_status()
            row = response.json()["rows"][0]["row"]
            if int(row["freesound_id"]) != source_id:
                raise RuntimeError(f"FSD row mismatch for {source_id}: {row['freesound_id']}")
            return str(row["audio"][0]["src"])
        except (requests.RequestException, KeyError, IndexError, RuntimeError):
            if attempt == 11:
                raise
            time.sleep(min(60, 2 ** min(attempt, 5)))
    raise RuntimeError(f"unable to resolve FSD50K row {source_id}")


def fetch_fsd_audio(root: Path, source_id: int, audio_url: str) -> Path:
    target = root / "raw" / "fsd50k" / f"{source_id}.flac"
    if target.exists() and target.stat().st_size > 0:
        return target
    return download(audio_url, target)


def build_fsd_selections(
    root: Path,
    info_path: Path,
    labels_path: Path,
    cache_only: bool = False,
) -> list[SourceSelection]:
    info: dict[str, dict[str, object]] = json.loads(info_path.read_text(encoding="utf-8"))
    index, _ = build_fsd_index(labels_path)
    high_speed, real_concert, music_pool, crowd_pool = select_fsd_ids(info_path, labels_path)
    rng = random.Random(20260901)
    # Keep this small, license-audited FSD subset pinned. Broader concurrent
    # Dataset Viewer extraction is rate-limited and must not make the corpus
    # irreproducible; the larger gain in this revision comes from official
    # TAU/TUT scene audio.
    legacy_high_speed_slots = list(range(6))
    rng.shuffle(legacy_high_speed_slots)
    high_speed_rng = random.Random(20260902)
    high_speed_rng.shuffle(high_speed)
    rng.shuffle(real_concert)
    rng.shuffle(music_pool)
    rng.shuffle(crowd_pool)
    high_speed = high_speed[:24]
    real_concert = real_concert[:24]
    music_pool = music_pool[:24]
    crowd_pool = crowd_pool[:24]
    high_speed_split_by_group = stratified_split_map(
        (
            f"freesound-uploader:{info[str(source_id)].get('uploader', source_id)}",
            "explicit_high_speed",
        )
        for source_id in high_speed
    )
    music_by_split = {
        split: [
            source_id for source_id in music_pool
            if split_source_group(f"freesound:{source_id}") == split
        ]
        for split in SPLITS
    }
    crowd_by_split = {
        split: [
            source_id for source_id in crowd_pool
            if split_source_group(f"freesound:{source_id}") == split
        ]
        for split in SPLITS
    }
    if any(not music_by_split[split] or not crowd_by_split[split] for split in SPLITS):
        raise RuntimeError("FSD50K three-way music/crowd pools are empty")
    required_ids = sorted(set(high_speed + real_concert + music_pool + crowd_pool))
    existing: dict[int, Path] = {}
    for source_id in required_ids:
        cached_path = root / "raw" / "fsd50k" / f"{source_id}.flac"
        if cached_path.exists() and cached_path.stat().st_size > 0:
            existing[source_id] = cached_path
    missing_ids = [source_id for source_id in required_ids if source_id not in existing]
    print(
        f"FSD50K: {len(existing)} cached, resolving {len(missing_ids)} selected CC0/CC BY clips",
        flush=True,
    )
    if cache_only and missing_ids:
        raise RuntimeError(
            f"cache-only mode is missing {len(missing_ids)} FSD50K files"
        )
    resolved_urls: dict[int, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_by_id = {
            executor.submit(resolve_fsd_audio_url, source_id, index): source_id
            for source_id in missing_ids
        }
        for progress, future in enumerate(
            concurrent.futures.as_completed(future_by_id), start=1
        ):
            source_id = future_by_id[future]
            resolved_urls[source_id] = future.result()
            if progress % 10 == 0 or progress == len(missing_ids):
                print(f"FSD50K: resolved {progress}/{len(missing_ids)}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        resolved = executor.map(
            lambda source_id: fetch_fsd_audio(root, source_id, resolved_urls[source_id]), missing_ids
        )
        downloaded = dict(zip(missing_ids, resolved))
    paths = {**existing, **downloaded}

    rows: list[SourceSelection] = []
    for label, source_ids, basis in [
        ("high_speed_train", high_speed, "FSD50K Train 标签 + TGV/ICE/高速列车元数据"),
        ("concert", real_concert, "FSD50K Crowd/Applause 标签 + 音乐现场元数据"),
    ]:
        for source_id in source_ids:
            metadata = info[str(source_id)]
            if label == "high_speed_train":
                # Several Freesound contributors upload numbered variants from
                # the same recording session. Split by uploader so a near-
                # duplicate series cannot cross the evaluation boundary.
                uploader = str(metadata.get("uploader", source_id))
                source_group = f"freesound-uploader:{uploader}"
                product_split = high_speed_split_by_group[source_group]
            else:
                source_group = f"freesound:{source_id}"
                product_split = split_source_group(source_group)
            rows.append(SourceSelection(
                id=f"fsd-{source_id}",
                label=label,
                split=product_split,
                dataset="FSD50K",
                source_id=str(source_id),
                source_group=source_group,
                subscene="explicit_high_speed" if label == "high_speed_train" else "live_concert",
                truth_basis=basis,
                license=str(metadata["license"]),
                attribution=f"{metadata.get('uploader', 'Freesound contributor')} · Freesound {source_id}",
                source_paths=[str(paths[source_id])],
            ))

    # Music-only and audience-only clips are hard negatives: either sound alone is
    # not sufficient evidence that the phone is at a live concert.
    for split in SPLITS:
        for subscene, source_ids in (
            ("music_without_live_audience", music_by_split[split]),
            ("crowd_without_music", crowd_by_split[split]),
        ):
            for source_id in source_ids:
                metadata = info[str(source_id)]
                rows.append(SourceSelection(
                    id=f"fsd-negative-{source_id}",
                    label="other",
                    split=split,
                    dataset="FSD50K",
                    source_id=str(source_id),
                    source_group=f"freesound:{source_id}",
                    subscene=subscene,
                    truth_basis="单独音乐或观众声音，仅作为音乐现场困难负样本",
                    license=str(metadata["license"]),
                    attribution=f"{metadata.get('uploader', 'Freesound contributor')} · Freesound {source_id}",
                    source_paths=[str(paths[source_id])],
                ))

        # Synthetic mixtures are training augmentation, never evaluation truth.
        if split != "train":
            continue
        mix_count = 80
        selected_music = music_by_split[split]
        selected_crowd = crowd_by_split[split]
        for local_index in range(mix_count):
            music_id = selected_music[local_index % len(selected_music)]
            crowd_id = selected_crowd[(local_index * 7 + 3) % len(selected_crowd)]
            rows.append(SourceSelection(
                id=f"fsd-concert-mix-{split}-{local_index:03d}",
                label="concert",
                split=split,
                dataset="FSD50K synthetic mixture",
                source_id=f"{music_id}+{crowd_id}",
                source_group=f"freesound-mixture:{music_id}+{crowd_id}",
                subscene="synthetic_live_concert",
                truth_basis="CC 音乐与 CC 观众/掌声的可复现混合增强",
                license="mixed CC0/CC BY; see source manifest",
                attribution=(
                    f"{info[str(music_id)].get('uploader', '')} / {music_id}; "
                    f"{info[str(crowd_id)].get('uploader', '')} / {crowd_id}"
                ),
                source_paths=[str(paths[music_id])],
                mix_source_paths=[str(paths[crowd_id])],
            ))
    return rows


def build_aporee_selections(
    root: Path, *, cache_only: bool
) -> list[SourceSelection]:
    """Add real field recordings without downloading the multi-terabyte corpus.

    Candidate metadata is snapshotted on the first online run.  The audio path
    is then streamed through ffmpeg and only the normalized ten-second window
    is retained.  All recordings from one Radio Aporee map location stay in a
    single product split.
    """
    candidates = load_candidates(root / APOREE_CACHE, cache_only=cache_only)
    if not candidates:
        print("Radio Aporee: no cached candidates", flush=True)
        return []
    frozen: dict[str, str] = {}
    existing_manifest = root / "dataset_manifest.json"
    if existing_manifest.exists():
        existing_payload = json.loads(existing_manifest.read_text(encoding="utf-8"))
        for item in existing_payload.get("items", []):
            if item.get("dataset") == "Radio Aporee field recordings":
                frozen[str(item["source_group"])] = str(item["split"])
    split_by_group = frozen_split_map(
        (candidate.location_group for candidate in candidates), frozen
    )
    rows: list[SourceSelection] = []
    candidate_by_id: dict[str, AporeeCandidate] = {}
    for candidate in candidates:
        split = split_by_group[candidate.location_group]
        target = root / "raw" / APOREE_WINDOW_DIR / f"{candidate.identifier}.wav"
        if cache_only and not has_minimum_audio(target):
            continue
        candidate_by_id[candidate.identifier] = candidate
        rows.append(SourceSelection(
            id=f"aporee-{candidate.identifier}",
            label=candidate.label,
            split=split,
            dataset="Radio Aporee field recordings",
            source_id=candidate.identifier,
            source_group=candidate.location_group,
            subscene=candidate.title,
            truth_basis=(
                "Radio Aporee 标题明确为目标场景外部/近邻/自然同名词困难负样本"
                if candidate.label == "other"
                else "Radio Aporee 地点标题 + 实地录音元数据"
            ),
            license=candidate.license,
            attribution=f"{candidate.creator} · Radio Aporee / Internet Archive",
            source_paths=[str(target)],
        ))

    split_limits = {"train": 140, "calibration": 30, "test": 30}
    selected: list[SourceSelection] = []
    for label in sorted({row.label for row in rows}):
        for split in SPLITS:
            selected.extend(round_robin_limit(
                (
                    row for row in rows
                    if row.label == label and row.split == split
                ),
                split_limits[split],
            ))
    recordings = selected
    selected = []
    for row in recordings:
        for offset in CONTINUOUS_SCENE_OFFSETS_SECONDS:
            suffix = "" if offset == 0 else f"-t{offset}"
            target = (
                root / "raw" / APOREE_WINDOW_DIR
                / f"{row.source_id}{suffix}.wav"
            )
            if cache_only and not has_minimum_audio(target):
                continue
            selected.append(replace(
                row,
                id=f"{row.id}{suffix}",
                source_paths=[str(target)],
            ))
    print(
        f"Radio Aporee: selected {len(recordings)}/{len(candidates)} "
        f"license-audited recordings -> {len(selected)} windows",
        flush=True,
    )
    if cache_only:
        return selected

    def fetch(row: SourceSelection) -> Path | None:
        candidate = candidate_by_id[row.source_id]
        target = Path(row.source_paths[0])
        offset_match = re.search(r"-t(\d+)$", row.id)
        offset = int(offset_match.group(1)) if offset_match else 0
        resolved = fetch_audio_window(
            candidate,
            target,
            sample_rate=SAMPLE_RATE,
            seconds=10,
            cache_only=False,
            start_seconds=offset,
            optional=offset > 0,
        )
        if resolved is None and offset == 0:
            raise RuntimeError(f"Radio Aporee audio was not resolved: {row.source_id}")
        return resolved

    available: list[SourceSelection] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=APOREE_DOWNLOAD_WORKERS
    ) as executor:
        for progress, (row, resolved) in enumerate(
            zip(selected, executor.map(fetch, selected)), start=1
        ):
            if resolved is not None:
                available.append(row)
            if progress % 25 == 0 or progress == len(selected):
                print(
                    f"Radio Aporee: decoded {progress}/{len(selected)} windows",
                    flush=True,
                )
    print(
        f"Radio Aporee: retained {len(available)}/{len(selected)} windows "
        "with at least five seconds of audio",
        flush=True,
    )
    return available


def archive_index(urls: list[str], cache_path: Path) -> dict[str, tuple[str, str]]:
    completed_urls: set[str] = set()
    result: dict[str, tuple[str, str]] = {}
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        if "entries" in raw:
            result = {
                key: (value[0], value[1])
                for key, value in raw["entries"].items()
            }
            completed_urls = set(raw.get("completed_urls", []))
        else:
            return {key: (value[0], value[1]) for key, value in raw.items()}
    for url in urls:
        if url in completed_urls:
            continue
        for attempt in range(12):
            try:
                with RemoteZip(url, timeout=(30, 180)) as archive:
                    for name in archive.namelist():
                        if name.endswith(".wav"):
                            result[name.split("/", 1)[-1]] = (url, name)
                completed_urls.add(url)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({
                    "completed_urls": sorted(completed_urls),
                    "entries": result,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"indexed remote archive {len(completed_urls)}/{len(urls)}", flush=True)
                break
            except Exception:
                if attempt == 11:
                    raise
                time.sleep(min(90, 3 * (2 ** min(attempt, 5))))
    return result


def extract_remote_entries(root: Path, dataset: str, paths: list[str], urls: list[str]) -> dict[str, Path]:
    index = archive_index(urls, root / "metadata" / f"{dataset}-archive-index.json")
    by_url: dict[str, list[tuple[str, str]]] = {}
    output: dict[str, Path] = {}
    for relative in paths:
        normalized = relative.removeprefix("audio/")
        key = f"audio/{normalized}"
        if key not in index:
            raise KeyError(f"{dataset} archive entry not found: {relative}")
        url, archive_name = index[key]
        target = root / "raw" / dataset / normalized
        output[relative] = target
        if not target.exists():
            by_url.setdefault(url, []).append((archive_name, relative))
    batches: list[tuple[str, list[tuple[str, str]]]] = []
    for url, entries in by_url.items():
        # Opening a multi-gigabyte remote ZIP through the local proxy is much
        # more expensive than reading many small members from one connection.
        # Amortize central-directory requests while keeping each retry bounded.
        for start in range(0, len(entries), 100):
            batches.append((url, entries[start:start + 100]))

    def extract_archive_batch(work: tuple[str, list[tuple[str, str]]]) -> int:
        url, entries = work
        for attempt in range(12):
            try:
                with RemoteZip(url, timeout=(30, 180)) as archive:
                    for archive_name, relative in entries:
                        target = output[relative]
                        if target.exists() and target.stat().st_size > 0:
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_suffix(target.suffix + ".part")
                        temporary.write_bytes(archive.read(archive_name))
                        temporary.replace(target)
                return len(entries)
            except Exception:
                if attempt == 11:
                    raise
                time.sleep(min(90, 3 * (2 ** min(attempt, 5))))
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for progress, _ in enumerate(executor.map(extract_archive_batch, batches), start=1):
            if progress % 20 == 0 or progress == len(batches):
                print(f"{dataset}: extracted batch {progress}/{len(batches)}", flush=True)
    return output


def download_tau_mirror_entries(root: Path, paths: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for relative in paths:
        normalized = relative.removeprefix("audio/")
        output[relative] = root / "raw" / "tau2022" / normalized
    missing = [relative for relative in paths if not output[relative].exists()]
    print(
        f"TAU mirror: {len(paths) - len(missing)} cached, downloading {len(missing)} files",
        flush=True,
    )

    def fetch(relative: str) -> Path:
        normalized = relative.removeprefix("audio/")
        return download(f"{TAU_MIRROR_RAW}/audio/{normalized}", output[relative])

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        for progress, _ in enumerate(executor.map(fetch, missing), start=1):
            if progress % 100 == 0 or progress == len(missing):
                print(f"TAU mirror: downloaded {progress}/{len(missing)}", flush=True)
    return output


def list_tau_mirror_paths(root: Path) -> set[str]:
    cache = root / "metadata" / "tau2022-mirror-paths.json"
    if cache.exists():
        return set(json.loads(cache.read_text(encoding="utf-8")))
    url = (
        f"https://huggingface.co/api/datasets/{TAU_MIRROR_REPO}/tree/"
        f"{TAU_MIRROR_REVISION}/audio"
    )
    params: dict[str, object] | None = {
        "recursive": "false", "expand": "false", "limit": 1000,
    }
    paths: list[str] = []
    while url:
        response: requests.Response | None = None
        for attempt in range(8):
            try:
                response = requests.get(url, params=params, timeout=(30, 180))
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 7:
                    raise
                time.sleep(min(60, 2 ** attempt))
        if response is None:
            raise RuntimeError("unable to list TAU mirror")
        paths.extend(str(item["path"]) for item in response.json() if item["type"] == "file")
        url = response.links.get("next", {}).get("url", "")
        params = None
        print(f"TAU mirror: indexed {len(paths)} files", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(sorted(paths), ensure_ascii=False), encoding="utf-8")
    return set(paths)


def load_audio(path: Path) -> np.ndarray:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return np.asarray(mono, dtype=np.float32)


def crop_or_pad(samples: np.ndarray, seed: str) -> np.ndarray:
    if samples.size > WINDOW_SAMPLES:
        maximum = samples.size - WINDOW_SAMPLES
        offset = stable_bucket(seed, maximum + 1) if maximum > 0 else 0
        return samples[offset:offset + WINDOW_SAMPLES]
    if samples.size < WINDOW_SAMPLES:
        return np.pad(samples, (0, WINDOW_SAMPLES - samples.size))
    return samples


def materialize(root: Path, selection: SourceSelection, resolved_paths: list[Path]) -> ManifestItem:
    if selection.dataset == "TAU Urban Acoustic Scenes 2022 Mobile":
        samples = np.concatenate([load_audio(path) for path in resolved_paths])
    else:
        samples = load_audio(resolved_paths[0])
    samples = crop_or_pad(samples, selection.id)
    if selection.mix_source_paths:
        crowd = crop_or_pad(load_audio(Path(selection.mix_source_paths[0])), selection.id + ":mix")
        music_rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-9))
        crowd_rms = float(np.sqrt(np.mean(np.square(crowd)) + 1e-9))
        crowd = crowd * (music_rms / max(crowd_rms, 1e-5)) * 0.55
        samples = np.clip(samples * 0.85 + crowd, -1.0, 1.0)
    output = root / PREPARED_DIR / selection.split / selection.label / f"{selection.id}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, samples, SAMPLE_RATE, subtype="PCM_16")
    return ManifestItem(
        id=selection.id,
        path=str(output.relative_to(root)),
        label=selection.label,
        split=selection.split,
        dataset=selection.dataset,
        source_id=selection.source_id,
        source_group=selection.source_group,
        subscene=selection.subscene,
        truth_basis=selection.truth_basis,
        license=selection.license,
        attribution=selection.attribution,
        sha256=sha256_file(output),
    )


def validate_manifest(items: list[ManifestItem]) -> None:
    try:
        validate_source_isolation(items)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    for item in items:
        if item.label not in CLASSES:
            raise RuntimeError(f"unknown label: {item.label}")
    for label in CLASSES:
        for split in SPLITS:
            count = sum(item.label == label and item.split == split for item in items)
            if count == 0:
                raise RuntimeError(f"empty class/split: {label}/{split}")


def write_fixed_tests(root: Path, items: list[ManifestItem]) -> None:
    """Write three playable 30-second calibration sessions per class.

    The product decision is based on temporal smoothing, so a fixture must be
    one continuous recording rather than three unrelated ten-second clips.
    Demo fixtures intentionally come from ``calibration``: repeatedly running
    them in the app must not open or tune against the held-out blind test set.
    """
    destination = root / FIXED_TEST_DIR
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("*.wav"):
        stale.unlink()
    rows: list[dict[str, str]] = []
    for label in CLASSES:
        candidates = [
            item for item in items
            if item.label == label
            and item.split == "calibration"
            and item.dataset == "Radio Aporee field recordings"
            and (item.license in COMMERCIAL_CC or is_commercial_license(item.license))
        ]
        by_source: dict[str, list[ManifestItem]] = {}
        for item in candidates:
            by_source.setdefault(item.source_id, []).append(item)
        continuous_sources = [
            sorted(source_items, key=lambda value: value.id)
            for source_items in by_source.values()
            if len(source_items) == len(CONTINUOUS_SCENE_OFFSETS_SECONDS)
        ]
        ordered = sorted(continuous_sources, key=lambda value: value[0].source_id)
        chosen: list[list[ManifestItem]] = []
        seen_groups: set[str] = set()
        for source_items in ordered:
            if source_items[0].source_group in seen_groups:
                continue
            chosen.append(source_items)
            seen_groups.add(source_items[0].source_group)
            if len(chosen) == 3:
                break
        if len(chosen) < 3:
            for source_items in ordered:
                if source_items in chosen:
                    continue
                chosen.append(source_items)
                if len(chosen) == 3:
                    break
        if len(chosen) < 3:
            raise RuntimeError(
                f"need three continuous calibration recordings for {label}, "
                f"found {len(chosen)}"
            )
        for fixture_index, source_items in enumerate(chosen):
            item = source_items[0]
            samples = np.concatenate([
                load_audio(root / source_item.path) for source_item in source_items
            ])
            expected_samples = WINDOW_SAMPLES * len(CONTINUOUS_SCENE_OFFSETS_SECONDS)
            if samples.size != expected_samples:
                raise RuntimeError(
                    f"continuous fixture {item.source_id} has {samples.size} samples, "
                    f"expected {expected_samples}"
                )
            target = destination / f"{label}-{fixture_index}.wav"
            sf.write(target, samples, SAMPLE_RATE, subtype="PCM_16")
            row = asdict(item)
            row["id"] = f"calibration-session-{item.source_id}"
            row["path"] = target.name
            row["truth_basis"] = (
                f"{item.truth_basis}; 同一原始录音的连续 0–30 秒校准展示片段"
            )
            row["sha256"] = sha256_file(target)
            rows.append(row)
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "split": "calibration",
                "duration_seconds": 30,
                "items": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="build only from complete cached source groups; never fetch audio",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    tau_dir, tut_dir, tut_dev_dir, fsd_info, fsd_labels = ensure_metadata(root)
    tau = build_tau_selections(
        tau_dir, root / "raw" / "tau2022" if args.cache_only else None
    )
    if args.cache_only:
        tau_mirror: list[SourceSelection] = []
        tau_official: list[SourceSelection] = []
    else:
        tau_available = list_tau_mirror_paths(root)
        tau_mirror = [
            row for row in tau
            if all(path in tau_available for path in row.source_paths)
        ]
        tau_official = [row for row in tau if row not in tau_mirror]
    tut_evaluation = build_tut_evaluation_selections(tut_dir)
    tut_development = build_tut_development_selections(tut_dev_dir)
    if args.cache_only:
        tut_evaluation = [
            row for row in tut_evaluation
            if all(
                (root / "raw" / "tut2017" / path.removeprefix("audio/")).exists()
                and (root / "raw" / "tut2017" / path.removeprefix("audio/")).stat().st_size > 0
                for path in row.source_paths
            )
        ]
        tut_development = [
            row for row in tut_development
            if all(
                (root / "raw" / "tut2017-dev" / path.removeprefix("audio/")).exists()
                and (root / "raw" / "tut2017-dev" / path.removeprefix("audio/")).stat().st_size > 0
                for path in row.source_paths
            )
        ]
    tut = limit_tut_selections(tut_evaluation + tut_development)
    print(
        f"selected TAU={len(tau)} TUT={len(tut)} "
        f"(development={sum(row.dataset.endswith('Development') for row in tut)})",
        flush=True,
    )
    fsd = build_fsd_selections(
        root, fsd_info, fsd_labels, cache_only=args.cache_only
    )
    aporee = build_aporee_selections(root, cache_only=args.cache_only)
    selections = tau + tut + fsd + aporee

    tau_mirror_paths = sorted({path for row in tau_mirror for path in row.source_paths})
    tau_official_paths = sorted({path for row in tau_official for path in row.source_paths})
    tut_evaluation_paths = sorted({
        path for row in tut
        if row.dataset == "TUT Acoustic Scenes 2017 Evaluation"
        for path in row.source_paths
    })
    tut_development_paths = sorted({
        path for row in tut
        if row.dataset == "TUT Acoustic Scenes 2017 Development"
        for path in row.source_paths
    })
    if args.cache_only:
        resolved_tau = {
            path: root / "raw" / "tau2022" / path.removeprefix("audio/")
            for row in tau for path in row.source_paths
        }
    else:
        resolved_tau = download_tau_mirror_entries(root, tau_mirror_paths)
        resolved_tau.update(extract_remote_entries(
            root, "tau2022", tau_official_paths, TAU_ARCHIVES
        ))
    print(f"downloaded TAU entries={len(resolved_tau)}", flush=True)
    if args.cache_only:
        resolved_tut = {
            path: root / "raw" / "tut2017" / path.removeprefix("audio/")
            for path in tut_evaluation_paths
        }
        resolved_tut_development = {
            path: root / "raw" / "tut2017-dev" / path.removeprefix("audio/")
            for path in tut_development_paths
        }
    else:
        resolved_tut = extract_remote_entries(
            root, "tut2017", tut_evaluation_paths, TUT_ARCHIVES
        )
        resolved_tut_development = extract_remote_entries(
            root, "tut2017-dev", tut_development_paths, TUT_DEV_ARCHIVES
        )
    print(
        f"downloaded TUT entries={len(resolved_tut) + len(resolved_tut_development)}",
        flush=True,
    )

    items: list[ManifestItem] = []
    for selection in selections:
        if selection.dataset == "TAU Urban Acoustic Scenes 2022 Mobile":
            resolved = [resolved_tau[path] for path in selection.source_paths]
        elif selection.dataset == "TUT Acoustic Scenes 2017 Evaluation":
            resolved = [resolved_tut[path] for path in selection.source_paths]
        elif selection.dataset == "TUT Acoustic Scenes 2017 Development":
            resolved = [resolved_tut_development[path] for path in selection.source_paths]
        else:
            resolved = [Path(path) for path in selection.source_paths]
        items.append(materialize(root, selection, resolved))

    validate_manifest(items)
    manifest = {
        "version": 4,
        "sample_rate": SAMPLE_RATE,
        "classes": CLASSES,
        "sources": {
            "tau_record": TAU_RECORD,
            "tau_transport_mirror": TAU_MIRROR_REPO,
            "tau_transport_revision": TAU_MIRROR_REVISION,
            "tut_record": TUT_RECORD,
            "tut_development_record": TUT_DEV_RECORD,
            "fsd_repository": FSD_REPO,
            "fsd_revision": FSD_REVISION,
            "radio_aporee_collection": "radio-aporee-maps",
            "radio_aporee_candidate_snapshot": APOREE_CACHE,
            "radio_aporee_split_policy": "frozen-existing-then-stable-hash",
        },
        "items": [asdict(item) for item in items],
    }
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_fixed_tests(root, items)

    for split, counts in summarize_split_counts(items).items():
        print(split, counts)
    print(f"manifest: {root / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
