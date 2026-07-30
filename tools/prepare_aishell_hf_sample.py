#!/usr/bin/env python3
"""Prepare a small AISHELL-1 benchmark manifest from a file-level HF mirror."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import quote

from huggingface_hub import HfApi


DEFAULT_REPO = "shenyunhang/AISHELL-1"


def load_transcripts(path: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                refs[parts[0]] = parts[1]
    return refs


def download_hf_file(repo: str, remote_path: str, output: Path) -> None:
    if output.exists() and output.stat().st_size > 0:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded_repo = quote(repo, safe="/")
    encoded_path = quote(remote_path, safe="/")
    url = f"https://huggingface.co/datasets/{encoded_repo}/resolve/main/{encoded_path}"
    tmp = output.with_suffix(output.suffix + ".part")
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "-C",
        "-",
        "-o",
        str(tmp),
        url,
    ]
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, 6):
        try:
            subprocess.run(cmd, check=True)
            last_error = None
            break
        except subprocess.CalledProcessError as exc:
            last_error = exc
            time.sleep(min(30, attempt * 3))
    if last_error is not None:
        raise last_error
    tmp.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/asr-bench/aishell100"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = args.out_dir / "wav"
    hf_dir = args.out_dir / "hf"
    wav_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.dataset_info(args.repo, files_metadata=True)
    paths = sorted(
        f.rfilename for f in info.siblings
        if f.rfilename.startswith(f"data_aishell/wav/{args.split}/") and f.rfilename.endswith(".wav")
    )
    if len(paths) < args.limit:
        raise RuntimeError(f"Only found {len(paths)} wav files for split={args.split}; need {args.limit}")
    paths = paths[:args.limit]

    transcript = hf_dir / "data_aishell/transcript/aishell_transcript_v0.8.txt"
    download_hf_file(args.repo, "data_aishell/transcript/aishell_transcript_v0.8.txt", transcript)
    refs = load_transcripts(transcript)

    manifest = args.out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as out:
        for index, remote_path in enumerate(paths, 1):
            uid = Path(remote_path).stem
            local = hf_dir / remote_path
            download_hf_file(args.repo, remote_path, local)
            target = wav_dir / f"{uid}.wav"
            if not target.exists():
                shutil.copy2(local, target)
            row = {
                "utt_id": uid,
                "dataset": "AISHELL-1",
                "split": args.split,
                "wav": str(target),
                "ref": refs.get(uid, ""),
                "source_repo": args.repo,
                "source_path": remote_path,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 20 == 0 or index == len(paths):
                print(f"prepared {index}/{len(paths)}")

    print(manifest)
    missing = sum(1 for p in paths if Path(p).stem not in refs)
    if missing:
        print(f"warning: missing transcripts for {missing} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
