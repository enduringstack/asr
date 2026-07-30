#!/usr/bin/env python3
"""Run FunASR models on the same manifest used by local sherpa benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import re
import statistics
import time
import unicodedata

import numpy as np
import soundfile as sf
from funasr import AutoModel


SAMPLE_RATE = 16000
PUNCT_RE = re.compile(r"[，。！？；：、,.!?;:《》“”‘’\"'()\[\]{}\-_/\\\s]")
SPECIAL_RE = re.compile(r"<\s*/?\s*s\s*>|<\s*unk\s*>", re.IGNORECASE)


def normalize_for_score(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = SPECIAL_RE.sub("", text)
    text = PUNCT_RE.sub("", text)
    return text.strip()


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = SPECIAL_RE.sub("", text)
    text = text.replace("▁", "")
    return re.sub(r"\s+", "", text).strip()


def edit_counts(ref: str, hyp: str) -> tuple[int, int, int]:
    n = len(ref)
    m = len(hyp)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost, sub, delete, ins = dp[i - 1][0]
        dp[i][0] = (cost + 1, sub, delete + 1, ins)
    for j in range(1, m + 1):
        cost, sub, delete, ins = dp[0][j - 1]
        dp[0][j] = (cost + 1, sub, delete, ins + 1)

    for i in range(1, n + 1):
        rc = ref[i - 1]
        for j in range(1, m + 1):
            hc = hyp[j - 1]
            best = dp[i - 1][j - 1] if rc == hc else tuple_add(dp[i - 1][j - 1], 1, 1, 0, 0)
            delete = tuple_add(dp[i - 1][j], 1, 0, 1, 0)
            insert = tuple_add(dp[i][j - 1], 1, 0, 0, 1)
            if delete[0] < best[0]:
                best = delete
            if insert[0] < best[0]:
                best = insert
            dp[i][j] = best

    _, sub, delete, ins = dp[n][m]
    return sub, delete, ins


def tuple_add(value: tuple[int, int, int, int], cost: int, sub: int, delete: int, ins: int) -> tuple[int, int, int, int]:
    return (value[0] + cost, value[1] + sub, value[2] + delete, value[3] + ins)


def read_manifest(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * p
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return values[int(idx)]
    return values[lower] * (upper - idx) + values[upper] * (idx - lower)


class FunAsrDecoder:
    def __init__(self, model_kind: str, streaming_profile: str) -> None:
        self.model_kind = model_kind
        self.streaming_profile = streaming_profile
        if model_kind == "offline-large":
            self.model_id = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            self.model = AutoModel(model=self.model_id, model_revision="v2.0.4", disable_update=True)
        elif model_kind == "streaming-online":
            self.model_id = "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online"
            self.model = AutoModel(model=self.model_id, model_revision="v2.0.4", disable_update=True)
        else:
            raise ValueError(f"unsupported model_kind: {model_kind}")

    def decode(self, wav: Path) -> str:
        if self.model_kind == "offline-large":
            result = self.model.generate(input=str(wav), batch_size_s=300)
            return clean_text(result[0].get("text", "") if result else "")
        if self.streaming_profile == "file":
            result = self.model.generate(input=str(wav), chunk_size=[0, 10, 5], encoder_chunk_look_back=4, decoder_chunk_look_back=1)
            return clean_text(result[0].get("text", "") if result else "")
        return self.decode_streaming(wav)

    def decode_streaming(self, wav: Path) -> str:
        samples, sample_rate = sf.read(str(wav), dtype="float32", always_2d=False)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"{wav} has sample_rate={sample_rate}; expected {SAMPLE_RATE}")
        if isinstance(samples, np.ndarray) and samples.ndim > 1:
            samples = np.mean(samples, axis=1)

        if self.streaming_profile == "modelscope":
            chunk_size = [5, 10, 5]
            encoder_chunk_look_back = 0
            decoder_chunk_look_back = 0
        else:
            chunk_size = [0, 10, 5]
            encoder_chunk_look_back = 4
            decoder_chunk_look_back = 1
        chunk_stride = chunk_size[1] * 960
        total_chunks = int((len(samples) - 1) / chunk_stride + 1)
        cache: dict[str, object] = {}
        pieces: list[str] = []

        for i in range(total_chunks):
            start = i * chunk_stride
            end = min(len(samples), (i + 1) * chunk_stride)
            chunk = samples[start:end]
            result = self.model.generate(
                input=chunk,
                cache=cache,
                is_final=i == total_chunks - 1,
                chunk_size=chunk_size,
                encoder_chunk_look_back=encoder_chunk_look_back,
                decoder_chunk_look_back=decoder_chunk_look_back,
            )
            text = clean_text(result[0].get("text", "") if result else "")
            if text:
                pieces.append(text)

        return clean_text("".join(pieces))


def write_report(out_dir: Path, rows: list[dict[str, object]], decoder: FunAsrDecoder, args: argparse.Namespace) -> None:
    total_ref = sum(int(r["ref_chars"]) for r in rows)
    total_err = sum(int(r["sub"]) + int(r["del"]) + int(r["ins"]) for r in rows)
    micro_cer = total_err / total_ref if total_ref else 0.0
    macro_cer = statistics.mean(float(r["cer"]) for r in rows) if rows else 0.0
    rtfs = [float(r["rtf"]) for r in rows]
    exact = sum(1 for r in rows if float(r["cer"]) == 0.0)
    high = sum(1 for r in rows if float(r["cer"]) > 0.20)
    empty = sum(1 for r in rows if not str(r["hyp_norm"]))

    with (out_dir / "report.md").open("w", encoding="utf-8") as f:
        f.write("# FunASR Benchmark Report\n\n")
        f.write(f"- Manifest: `{args.manifest}`\n")
        f.write(f"- Model kind: `{args.model_kind}`\n")
        f.write(f"- Model id: `{decoder.model_id}`\n")
        if args.model_kind == "streaming-online":
            f.write(f"- Streaming profile: `{decoder.streaming_profile}`\n")
        f.write(f"- Samples: {len(rows)}\n")
        f.write(f"- Total audio: {sum(float(r['duration_sec']) for r in rows):.1f}s\n")
        f.write(f"- Micro CER: {micro_cer:.2%}\n")
        f.write(f"- Macro CER: {macro_cer:.2%}\n")
        f.write(f"- Median CER: {statistics.median(float(r['cer']) for r in rows):.2%}\n")
        f.write(f"- Exact matches: {exact}/{len(rows)}\n")
        f.write(f"- CER > 20%: {high}/{len(rows)}\n")
        f.write(f"- Empty hypothesis rate: {empty}/{len(rows)}\n")
        f.write(f"- RTF p50/p90/p95: {percentile(rtfs, 0.50):.3f} / {percentile(rtfs, 0.90):.3f} / {percentile(rtfs, 0.95):.3f}\n")
        f.write(f"- Platform: {platform.platform()}\n\n")
        f.write("## Worst 20\n\n")
        f.write("| # | Utt | Dur | CER | Ref | Hyp |\n")
        f.write("|---:|---|---:|---:|---|---|\n")
        for i, row in enumerate(sorted(rows, key=lambda r: float(r["cer"]), reverse=True)[:20], 1):
            ref = str(row["ref_raw"]).replace("|", " ")
            hyp = str(row["hyp_raw"]).replace("|", " ")
            f.write(f"| {i} | {row['utt_id']} | {float(row['duration_sec']):.1f} | {float(row['cer']):.2%} | {ref[:80]} | {hyp[:80]} |\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-kind", choices=("offline-large", "streaming-online"), required=True)
    parser.add_argument("--streaming-profile", choices=("funasr", "modelscope", "file"), default="funasr")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    decoder = FunAsrDecoder(args.model_kind, args.streaming_profile)
    rows_in = read_manifest(args.manifest, args.limit)
    result_rows: list[dict[str, object]] = []

    for idx, item in enumerate(rows_in, 1):
        wav = Path(item["wav"])
        info = sf.info(str(wav))
        duration = float(info.frames) / float(info.samplerate)
        started = time.perf_counter()
        try:
            hyp = decoder.decode(wav)
            error = ""
        except Exception as exc:
            hyp = ""
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started

        ref_raw = str(item.get("ref", ""))
        ref_norm = normalize_for_score(ref_raw)
        hyp_norm = normalize_for_score(hyp)
        sub, delete, ins = edit_counts(ref_norm, hyp_norm)
        ref_chars = len(ref_norm)
        cer = (sub + delete + ins) / ref_chars if ref_chars else 0.0
        row = {
            "index": idx,
            "utt_id": item.get("utt_id", wav.stem),
            "dataset": item.get("dataset", ""),
            "split": item.get("split", ""),
            "wav": str(wav),
            "ref_raw": ref_raw,
            "ref_norm": ref_norm,
            "hyp_raw": hyp,
            "hyp_norm": hyp_norm,
            "duration_sec": duration,
            "decode_ms": round(elapsed * 1000, 3),
            "rtf": elapsed / duration if duration > 0 else 0.0,
            "ref_chars": ref_chars,
            "sub": sub,
            "del": delete,
            "ins": ins,
            "cer": cer,
            "empty_hyp": hyp_norm == "",
            "error": error,
        }
        result_rows.append(row)
        print(f"[{idx:03d}/{len(rows_in):03d}] {row['utt_id']} cer={cer:.2%} rtf={row['rtf']:.3f} hyp={hyp[:60]}")

    jsonl = args.out_dir / "results.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in result_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = args.out_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()) if result_rows else [])
        if result_rows:
            writer.writeheader()
            writer.writerows(result_rows)

    write_report(args.out_dir, result_rows, decoder, args)
    print(f"\nWrote {args.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
