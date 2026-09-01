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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import requests
import soundfile as sf
from remotezip import RemoteZip


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = SAMPLE_RATE * 10
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

FSD_REPO = "quinnlue/FSD50K-16k"
FSD_REVISION = "2a60d475f4e2f0db624a902881af2df79d656145"
FSD_RAW = f"https://huggingface.co/datasets/{FSD_REPO}/resolve/main"
FSD_INFO = f"{FSD_RAW}/metadata/original/dev_clips_info_FSD50K.json"
FSD_LABELS = f"{FSD_RAW}/metadata/labels/dev.csv"


@dataclass
class ManifestItem:
    id: str
    path: str
    label: str
    split: str
    dataset: str
    source_id: str
    source_group: str
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


def ensure_metadata(root: Path) -> tuple[Path, Path, Path, Path]:
    downloads = root / "downloads"
    tau_zip = download(TAU_META_URL, downloads / "tau2022-meta.zip")
    tut_zip = download(TUT_META_URL, downloads / "tut2017-meta.zip")
    tau_dir = root / "metadata" / "tau2022"
    tut_dir = root / "metadata" / "tut2017"
    if not (tau_dir / TAU_PREFIX / "meta.csv").exists():
        shutil.unpack_archive(tau_zip, tau_dir)
    if not (tut_dir / TUT_PREFIX / "evaluation_setup" / "evaluate.txt").exists():
        shutil.unpack_archive(tut_zip, tut_dir)

    fsd_dir = root / "metadata" / "fsd50k"
    fsd_info = download(FSD_INFO, fsd_dir / "dev_clips_info_FSD50K.json")
    fsd_labels = download(FSD_LABELS, fsd_dir / "dev.csv")
    return tau_dir / TAU_PREFIX, tut_dir / TUT_PREFIX, fsd_info, fsd_labels


def limited_per_label(rows: Iterable[SourceSelection], limits: dict[tuple[str, str], int]) -> list[SourceSelection]:
    grouped: dict[tuple[str, str], dict[str, list[SourceSelection]]] = {}
    for row in sorted(rows, key=lambda item: item.id):
        key = (row.label, row.split)
        if limits.get(key, 0) <= 0:
            continue
        grouped.setdefault(key, {}).setdefault(row.source_group, []).append(row)

    selected: list[SourceSelection] = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        ordered_groups = sorted(group_rows)
        limit = limits[key]
        depth = 0
        selected_count = 0
        while selected_count < limit:
            appended = False
            for source_group in ordered_groups:
                values = group_rows[source_group]
                if depth >= len(values):
                    continue
                selected.append(values[depth])
                selected_count += 1
                appended = True
                if selected_count >= limit:
                    break
            if not appended:
                break
            depth += 1
    return selected


def tau_group_key(filename: str) -> str:
    return re.sub(r"-\d+-[^-]+\.wav$", "", filename)


def build_tau_selections(tau_dir: Path, available_paths: set[str] | None = None) -> list[SourceSelection]:
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
        if available_paths is not None and any(
            item["filename"] not in available_paths for item in group_rows
        ):
            continue
        scene = group_rows[0]["scene_label"]
        if scene == "metro":
            label = "metro"
        elif scene == "shopping_mall":
            label = "shopping_mall"
        else:
            label = "other"
        location = group_rows[0]["identifier"]
        split = "validation" if stable_bucket(f"tau:{location}") == 0 else "train"
        rows.append(SourceSelection(
            id=f"tau-{Path(group_key).name}",
            label=label,
            split=split,
            dataset="TAU Urban Acoustic Scenes 2022 Mobile",
            source_id=group_key,
            source_group=f"tau-location:{location}",
            truth_basis=f"官方文件级场景标签：{scene}",
            license="other-nc",
            attribution="Tampere University / TAU Urban Acoustic Scenes 2022 Mobile",
            source_paths=sorted(item["filename"] for item in group_rows),
        ))
    return limited_per_label(rows, {
        ("metro", "train"): 48,
        ("metro", "validation"): 12,
        # Prefer geographic diversity to many correlated snippets: the
        # round-robin limiter below yields 18 train locations and three held-out
        # locations. TAU remains non-commercial; production must replace this
        # source with appropriately licensed in-domain captures.
        ("shopping_mall", "train"): 18,
        ("shopping_mall", "validation"): 3,
        ("other", "train"): 80,
        ("other", "validation"): 20,
    })


def tut_source_group(source_path: str) -> str:
    name = Path(source_path).stem
    return re.sub(r"_\d+_\d+$", "", name)


def build_tut_selections(tut_dir: Path) -> list[SourceSelection]:
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
        if scene == "cafe/restaurant":
            label = "cafe_restaurant"
        elif scene == "train":
            label = "high_speed_train"
        else:
            label = "other"
        original = original_by_eval.get(filename, filename)
        group = tut_source_group(original)
        split = "validation" if stable_bucket(f"tut:{group}") == 0 else "train"
        rows.append(SourceSelection(
            id=f"tut-{Path(filename).stem}",
            label=label,
            split=split,
            dataset="TUT Acoustic Scenes 2017 Evaluation",
            source_id=original,
            source_group=f"tut-recording:{group}",
            truth_basis=f"官方文件级场景标签：{scene}",
            license="other-nc",
            attribution="Tampere University / TUT Acoustic Scenes 2017",
            source_paths=[filename],
        ))
    return limited_per_label(rows, {
        ("cafe_restaurant", "train"): 48,
        ("cafe_restaurant", "validation"): 12,
        ("high_speed_train", "train"): 42,
        ("high_speed_train", "validation"): 10,
        ("other", "train"): 60,
        ("other", "validation"): 15,
    })


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


def build_fsd_selections(root: Path, info_path: Path, labels_path: Path) -> list[SourceSelection]:
    info: dict[str, dict[str, object]] = json.loads(info_path.read_text(encoding="utf-8"))
    index, _ = build_fsd_index(labels_path)
    high_speed, real_concert, music_pool, crowd_pool = select_fsd_ids(info_path, labels_path)
    # Keep the music/concert pools pinned when the high-speed keyword list is
    # expanded. The original six-item shuffle is replayed solely to preserve
    # the established deterministic RNG sequence for those unrelated pools.
    rng = random.Random(20260901)
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
    music_train = [
        source_id for source_id in music_pool
        if stable_bucket(f"fsd-pool:{source_id}") != 0
    ]
    music_validation = [
        source_id for source_id in music_pool
        if stable_bucket(f"fsd-pool:{source_id}") == 0
    ]
    crowd_train = [
        source_id for source_id in crowd_pool
        if stable_bucket(f"fsd-pool:{source_id}") != 0
    ]
    crowd_validation = [
        source_id for source_id in crowd_pool
        if stable_bucket(f"fsd-pool:{source_id}") == 0
    ]
    if not music_validation or not crowd_validation:
        raise RuntimeError("FSD50K validation pools are empty")
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
    resolved_urls: dict[int, str] = {}
    for progress, source_id in enumerate(missing_ids, start=1):
        resolved_urls[source_id] = resolve_fsd_audio_url(source_id, index)
        if progress % 5 == 0 or progress == len(missing_ids):
            print(f"FSD50K: resolved {progress}/{len(missing_ids)}", flush=True)
        time.sleep(0.5)
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
                split = (
                    "validation"
                    if stable_bucket(f"fsd-uploader:{uploader}") == 0
                    else "train"
                )
                source_group = f"freesound-uploader:{uploader}"
            else:
                split = "validation" if stable_bucket(f"fsd:{source_id}") == 0 else "train"
                source_group = f"freesound:{source_id}"
            rows.append(SourceSelection(
                id=f"fsd-{source_id}",
                label=label,
                split=split,
                dataset="FSD50K",
                source_id=str(source_id),
                source_group=source_group,
                truth_basis=basis,
                license=str(metadata["license"]),
                attribution=f"{metadata.get('uploader', 'Freesound contributor')} · Freesound {source_id}",
                source_paths=[str(paths[source_id])],
            ))

    # Music-only and audience-only clips are hard negatives: either sound alone is
    # not sufficient evidence that the phone is at a live concert.
    for source_id in music_train[:16] + crowd_train[:16]:
        metadata = info[str(source_id)]
        rows.append(SourceSelection(
            id=f"fsd-negative-{source_id}",
            label="other",
            split="train",
            dataset="FSD50K",
            source_id=str(source_id),
            source_group=f"freesound:{source_id}",
            truth_basis="单独音乐或观众声音，仅作为音乐现场困难负样本",
            license=str(metadata["license"]),
            attribution=f"{metadata.get('uploader', 'Freesound contributor')} · Freesound {source_id}",
            source_paths=[str(paths[source_id])],
        ))

    for index_value in range(64):
        split = "validation" if index_value >= 52 else "train"
        local_index = index_value - 52 if split == "validation" else index_value
        selected_music = music_validation if split == "validation" else music_train
        selected_crowd = crowd_validation if split == "validation" else crowd_train
        music_id = selected_music[local_index % len(selected_music)]
        crowd_id = selected_crowd[(local_index * 7 + 3) % len(selected_crowd)]
        rows.append(SourceSelection(
            id=f"fsd-concert-mix-{index_value:03d}",
            label="concert",
            split=split,
            dataset="FSD50K synthetic mixture",
            source_id=f"{music_id}+{crowd_id}",
            # Music/crowd pools are split before mixing. Key by the music source so
            # the manifest validator also catches any future cross-split reuse.
            source_group=f"freesound:{music_id}",
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
                with RemoteZip(url) as archive:
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
        # Opening a remote ZIP repeatedly is much more expensive than reading
        # several members from one indexed connection. Larger batches also
        # reduce pressure on Zenodo's range-request rate limits.
        for start in range(0, len(entries), 10):
            batches.append((url, entries[start:start + 10]))

    def extract_archive_batch(work: tuple[str, list[tuple[str, str]]]) -> int:
        url, entries = work
        for attempt in range(12):
            try:
                with RemoteZip(url) as archive:
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
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
    output = root / "prepared" / selection.split / selection.label / f"{selection.id}.wav"
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
        truth_basis=selection.truth_basis,
        license=selection.license,
        attribution=selection.attribution,
        sha256=sha256_file(output),
    )


def validate_manifest(items: list[ManifestItem]) -> None:
    split_by_group: dict[str, str] = {}
    for item in items:
        previous = split_by_group.setdefault(item.source_group, item.split)
        if previous != item.split:
            raise RuntimeError(f"source leakage: {item.source_group} occurs in {previous} and {item.split}")
        if item.label not in CLASSES:
            raise RuntimeError(f"unknown label: {item.label}")
    for label in CLASSES:
        for split in ("train", "validation"):
            count = sum(item.label == label and item.split == split for item in items)
            if count == 0:
                raise RuntimeError(f"empty class/split: {label}/{split}")


def write_fixed_tests(root: Path, items: list[ManifestItem]) -> None:
    destination = root / "fixed_tests"
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for label in CLASSES:
        candidates = [
            item for item in items
            if item.label == label and item.split == "validation" and "synthetic" not in item.dataset.lower()
        ]
        if label in {"high_speed_train", "concert"}:
            licensed = [item for item in candidates if item.license in COMMERCIAL_CC]
            if len(licensed) >= 3:
                candidates = licensed
        ordered = sorted(candidates, key=lambda value: value.id)
        chosen: list[ManifestItem] = []
        seen_groups: set[str] = set()
        for item in ordered:
            if item.source_group in seen_groups:
                continue
            chosen.append(item)
            seen_groups.add(item.source_group)
            if len(chosen) == 3:
                break
        if len(chosen) < 3:
            for item in ordered:
                if item in chosen:
                    continue
                chosen.append(item)
                if len(chosen) == 3:
                    break
        for item in chosen:
            source = root / item.path
            target = destination / f"{label}-{len(rows):02d}-{source.name}"
            shutil.copy2(source, target)
            row = asdict(item)
            row["path"] = target.name
            rows.append(row)
    (destination / "manifest.json").write_text(
        json.dumps({"version": 1, "items": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    tau_dir, tut_dir, fsd_info, fsd_labels = ensure_metadata(root)
    tau_available = list_tau_mirror_paths(root)
    tau_mirror = build_tau_selections(tau_dir, tau_available)
    tau_official = [
        row for row in build_tau_selections(tau_dir)
        if row.label == "shopping_mall"
    ]
    tau = tau_mirror + tau_official
    tut = build_tut_selections(tut_dir)
    print(f"selected TAU={len(tau)} TUT={len(tut)}", flush=True)
    fsd = build_fsd_selections(root, fsd_info, fsd_labels)
    selections = tau + tut + fsd

    tau_mirror_paths = sorted({path for row in tau_mirror for path in row.source_paths})
    tau_official_paths = sorted({path for row in tau_official for path in row.source_paths})
    tut_paths = sorted({path for row in tut for path in row.source_paths})
    resolved_tau = download_tau_mirror_entries(root, tau_mirror_paths)
    resolved_tau.update(extract_remote_entries(
        root, "tau2022", tau_official_paths, TAU_ARCHIVES
    ))
    print(f"downloaded TAU entries={len(resolved_tau)}", flush=True)
    resolved_tut = extract_remote_entries(root, "tut2017", tut_paths, TUT_ARCHIVES)
    print(f"downloaded TUT entries={len(resolved_tut)}", flush=True)

    items: list[ManifestItem] = []
    for selection in selections:
        if selection.dataset == "TAU Urban Acoustic Scenes 2022 Mobile":
            resolved = [resolved_tau[path] for path in selection.source_paths]
        elif selection.dataset == "TUT Acoustic Scenes 2017 Evaluation":
            resolved = [resolved_tut[path] for path in selection.source_paths]
        else:
            resolved = [Path(path) for path in selection.source_paths]
        items.append(materialize(root, selection, resolved))

    validate_manifest(items)
    manifest = {
        "version": 1,
        "sample_rate": SAMPLE_RATE,
        "classes": CLASSES,
        "sources": {
            "tau_record": TAU_RECORD,
            "tau_transport_mirror": TAU_MIRROR_REPO,
            "tau_transport_revision": TAU_MIRROR_REVISION,
            "tut_record": TUT_RECORD,
            "fsd_repository": FSD_REPO,
            "fsd_revision": FSD_REVISION,
        },
        "items": [asdict(item) for item in items],
    }
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_fixed_tests(root, items)

    for split in ("train", "validation"):
        counts = {label: sum(item.label == label and item.split == split for item in items) for label in CLASSES}
        print(split, counts)
    print(f"manifest: {root / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
