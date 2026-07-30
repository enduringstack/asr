#!/usr/bin/env python3
"""Run a local ASR benchmark with the same sherpa-onnx model used by the app.

Manifest format: one JSON object per line with at least:
  {"utt_id": "...", "wav": "/abs/path.wav", "ref": "reference text"}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import time
import unicodedata

import numpy as np
import soundfile as sf
import sherpa_onnx


SAMPLE_RATE = 16000
DIRECT_TAIL_PADDING_SAMPLES = 16000
LONG_AUDIO_SEGMENT_SECONDS = 10.0
LONG_AUDIO_OVERLAP_SECONDS = 0.25
LONG_AUDIO_MIN_SEGMENTED_SECONDS = 20.0
LONG_AUDIO_BOUNDARY_SEARCH_SECONDS = 0.6
LONG_AUDIO_BOUNDARY_WINDOW_SECONDS = 0.16
LONG_AUDIO_MIN_SEGMENT_SECONDS = 6.0
TEXT_OVERLAP_MAX_CHARS = 32
TEXT_OVERLAP_SEARCH_CHARS = 80
TEXT_TAIL_REPLACE_MIN_OVERLAP_CHARS = 8
TEXT_TAIL_REPLACE_MAX_CHARS = 48


PUNCT_RE = re.compile(r"[，。！？；：、,.!?;:《》“”‘’\"'()\[\]{}\-_/\\\s]")
SPECIAL_RE = re.compile(r"<\s*/?\s*s\s*>|<\s*unk\s*>", re.IGNORECASE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_for_score(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = SPECIAL_RE.sub("", text)
    text = PUNCT_RE.sub("", text)
    return text.strip()


def clean_decoded_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = SPECIAL_RE.sub("", text)
    text = text.replace("▁", "")
    return re.sub(r"\s+", "", text).strip()


ReplacementRule = tuple[str, str]


def load_replacement_rules(path: Path | None) -> list[ReplacementRule]:
    """Load {from, to} pairs from a JSON file, sorted longest-from-first."""
    if path is None:
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            parsed = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    rules: list[ReplacementRule] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        src = item.get("from")
        dst = item.get("to")
        if isinstance(src, str) and isinstance(dst, str) and src:
            rules.append((src, dst))
    rules.sort(key=lambda r: len(r[0]), reverse=True)
    return rules


def apply_replacements(text: str, rules: list[ReplacementRule]) -> str:
    if not rules or not text:
        return text
    for src, dst in rules:
        if src in text:
            text = text.replace(src, dst)
    return text


def edit_counts(ref: str, hyp: str) -> tuple[int, int, int]:
    """Return substitution, deletion, insertion counts for character sequences."""
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
            if rc == hc:
                best = dp[i - 1][j - 1]
            else:
                c, s, d, ins = dp[i - 1][j - 1]
                best = (c + 1, s + 1, d, ins)

            c, s, d, ins = dp[i - 1][j]
            cand = (c + 1, s, d + 1, ins)
            if cand[0] < best[0]:
                best = cand

            c, s, d, ins = dp[i][j - 1]
            cand = (c + 1, s, d, ins + 1)
            if cand[0] < best[0]:
                best = cand

            dp[i][j] = best

    _, sub, delete, ins = dp[n][m]
    return sub, delete, ins


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path} has sample_rate={sample_rate}; expected {SAMPLE_RATE}")
    return np.asarray(samples, dtype=np.float32), sample_rate


def score_audio_window(samples: np.ndarray, center: int, window: int) -> float:
    start = max(0, center - window // 2)
    end = min(len(samples), center + window // 2)
    if end <= start:
        return float("inf")
    frame = samples[start:end]
    return float(np.sqrt(np.mean(frame * frame)))


def find_quiet_boundary(
    samples: np.ndarray,
    target: int,
    search_radius: int,
    window: int,
    min_boundary: int,
    max_boundary: int,
) -> int:
    lower = max(min_boundary, target - search_radius)
    upper = min(max_boundary, target + search_radius)
    if upper <= lower:
        return max(min_boundary, min(max_boundary, target))

    step = max(1, window // 4)
    best = lower
    best_score = float("inf")
    for center in range(lower, upper + 1, step):
        score = score_audio_window(samples, center, window)
        if score < best_score:
            best_score = score
            best = center
    return best


def build_long_audio_segments(samples: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    segment_samples = max(1, int(sample_rate * LONG_AUDIO_SEGMENT_SECONDS))
    if len(samples) <= segment_samples:
        return [(0, len(samples))]

    search_radius = max(1, int(sample_rate * LONG_AUDIO_BOUNDARY_SEARCH_SECONDS))
    boundary_window = max(1, int(sample_rate * LONG_AUDIO_BOUNDARY_WINDOW_SECONDS))
    min_segment = max(1, int(sample_rate * LONG_AUDIO_MIN_SEGMENT_SECONDS))
    boundaries = [0]

    previous = 0
    target = segment_samples
    while target < len(samples) - min_segment:
        min_boundary = min(len(samples), previous + min_segment)
        max_boundary = min(len(samples) - min_segment, target + search_radius)
        if max_boundary <= min_boundary:
            break
        boundary = find_quiet_boundary(samples, target, search_radius, boundary_window, min_boundary, max_boundary)
        if boundary <= previous:
            break
        boundaries.append(boundary)
        previous = boundary
        target = previous + segment_samples
    boundaries.append(len(samples))

    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1) if boundaries[i + 1] > boundaries[i]]


def append_text_with_overlap(previous: str, current: str) -> str:
    previous = previous or ""
    current = current or ""
    if previous == "":
        return current
    if current == "":
        return previous

    previous_tail = previous[-TEXT_OVERLAP_SEARCH_CHARS:]
    max_overlap = min(TEXT_OVERLAP_MAX_CHARS, len(previous_tail), len(current))
    for overlap in range(max_overlap, 0, -1):
        prefix = current[:overlap]
        idx = previous_tail.rfind(prefix)
        if idx >= 0:
            global_idx = len(previous) - len(previous_tail) + idx
            replace_len = len(previous) - global_idx
            if overlap >= TEXT_TAIL_REPLACE_MIN_OVERLAP_CHARS or replace_len <= TEXT_TAIL_REPLACE_MAX_CHARS:
                return previous[:global_idx] + current
            return previous + current[overlap:]
    return previous + current


class EdgeLikeDecoder:
    def __init__(self, rawfile: Path, with_punctuation: bool, precision: str,
                 replacements: list[ReplacementRule] | None = None,
                 hotwords_file: Path | None = None, hotwords_score: float = 1.5) -> None:
        self.replacements = replacements or []
        self.hotwords_file = hotwords_file
        self.hotwords_score = hotwords_score
        model_dir = rawfile / "sherpa-onnx-streaming-paraformer-bilingual-zh-en"
        if precision == "fp32":
            self.encoder = model_dir / "encoder.onnx"
            self.decoder = model_dir / "decoder.onnx"
        else:
            self.encoder = model_dir / "encoder.int8.onnx"
            self.decoder = model_dir / "decoder.int8.onnx"
        self.tokens = model_dir / "tokens.txt"
        kwargs: dict[str, object] = dict(
            tokens=str(self.tokens),
            encoder=str(self.encoder),
            decoder=str(self.decoder),
            num_threads=2,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            enable_endpoint_detection=False,
            rule1_min_trailing_silence=2.0,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20.0,
            provider="cpu",
        )
        if hotwords_file is not None:
            kwargs["hotwords_file"] = str(hotwords_file)
            kwargs["hotwords_score"] = hotwords_score
        try:
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(**kwargs)
        except TypeError as exc:
            # Paraformer factories do not accept hotwords_file/hotwords_score (and paraformer
            # only supports greedy_search, not modified_beam_search) — so hotword biasing is
            # structurally unavailable on this model. Warn loudly and proceed WITHOUT hotwords
            # so the A/B run honestly reflects "hotwords not applied" rather than silently
            # producing a misleading delta=0.
            if hotwords_file is not None:
                import sys
                print(f"[warn] streaming paraformer does not accept hotwords ({exc}); "
                      f"running WITHOUT hotwords. Hotword biasing is unavailable on this model.",
                      file=sys.stderr)
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
                **{k: v for k, v in kwargs.items() if k not in ("hotwords_file", "hotwords_score")})

        self.punctuation_model = rawfile / "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8" / "model.int8.onnx"
        self.punctuation = None
        if with_punctuation and self.punctuation_model.exists():
            model_config = sherpa_onnx.OfflinePunctuationModelConfig()
            model_config.ct_transformer = str(self.punctuation_model)
            model_config.num_threads = 1
            model_config.provider = "cpu"
            config = sherpa_onnx.OfflinePunctuationConfig()
            config.model = model_config
            self.punctuation = sherpa_onnx.OfflinePunctuation(config)

    def model_hashes(self) -> dict[str, str]:
        paths = {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "tokens": self.tokens,
        }
        if self.punctuation_model.exists():
            paths["punctuation"] = self.punctuation_model
        return {name: sha256(path) for name, path in paths.items()}

    def add_punctuation(self, text: str) -> str:
        text = clean_decoded_text(text)
        if not text:
            return text
        if self.punctuation is not None:
            text = clean_decoded_text(self.punctuation.add_punctuation(text))
        return apply_replacements(text, self.replacements)

    def decode_segment(self, samples: np.ndarray, sample_rate: int, chunk_samples: int) -> str:
        stream = self.recognizer.create_stream()
        for start in range(0, len(samples), chunk_samples):
            stream.accept_waveform(sample_rate, samples[start:start + chunk_samples])
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)
        stream.accept_waveform(sample_rate, np.zeros(DIRECT_TAIL_PADDING_SAMPLES, dtype=np.float32))
        stream.input_finished()
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        return clean_decoded_text(self.recognizer.get_result(stream))

    def decode(self, samples: np.ndarray, sample_rate: int, chunk_samples: int) -> tuple[str, str, list[tuple[int, int]]]:
        if len(samples) / sample_rate < LONG_AUDIO_MIN_SEGMENTED_SECONDS:
            raw = self.decode_segment(samples, sample_rate, chunk_samples)
            return raw, self.add_punctuation(raw), [(0, len(samples))]

        segments = build_long_audio_segments(samples, sample_rate)
        overlap = max(0, int(sample_rate * LONG_AUDIO_OVERLAP_SECONDS))
        raw = ""
        for start, end in segments:
            seg_start = max(0, start - overlap)
            seg_end = min(len(samples), end + overlap)
            if seg_end <= seg_start:
                continue
            text = self.decode_segment(samples[seg_start:seg_end], sample_rate, chunk_samples)
            raw = append_text_with_overlap(raw, text)
        return clean_decoded_text(raw), self.add_punctuation(raw), segments


class OfflineFinalDecoder:
    def __init__(
        self,
        model_dir: Path,
        rawfile: Path,
        with_punctuation: bool,
        decoder_type: str,
        sensevoice_language: str,
        sensevoice_use_itn: bool,
        replacements: list[ReplacementRule] | None = None,
        hotwords_file: Path | None = None, hotwords_score: float = 1.5,
    ) -> None:
        self.replacements = replacements or []
        self.hotwords_file = hotwords_file
        self.hotwords_score = hotwords_score
        model = model_dir / "model.int8.onnx"
        if not model.exists():
            model = model_dir / "model.onnx"
        self.model = model
        self.tokens = model_dir / "tokens.txt"
        self.decoder_type = decoder_type
        if decoder_type == "offline-paraformer":
            kwargs: dict[str, object] = dict(
                paraformer=str(self.model),
                tokens=str(self.tokens),
                num_threads=2,
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                provider="cpu",
            )
            if hotwords_file is not None:
                kwargs["hotwords_file"] = str(hotwords_file)
                kwargs["hotwords_score"] = hotwords_score
            try:
                self.recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(**kwargs)
            except TypeError as exc:
                if hotwords_file is not None:
                    import sys
                    print(f"[warn] offline paraformer does not accept hotwords ({exc}); "
                          f"running WITHOUT hotwords. Hotword biasing is unavailable on this model.",
                          file=sys.stderr)
                self.recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                    **{k: v for k, v in kwargs.items() if k not in ("hotwords_file", "hotwords_score")})
        elif decoder_type == "sensevoice":
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(self.model),
                tokens=str(self.tokens),
                num_threads=2,
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                provider="cpu",
                language=sensevoice_language,
                use_itn=sensevoice_use_itn,
            )
        else:
            raise ValueError(f"unsupported offline decoder: {decoder_type}")

        self.punctuation_model = rawfile / "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8" / "model.int8.onnx"
        self.punctuation = None
        if with_punctuation and self.punctuation_model.exists():
            model_config = sherpa_onnx.OfflinePunctuationModelConfig()
            model_config.ct_transformer = str(self.punctuation_model)
            model_config.num_threads = 1
            model_config.provider = "cpu"
            config = sherpa_onnx.OfflinePunctuationConfig()
            config.model = model_config
            self.punctuation = sherpa_onnx.OfflinePunctuation(config)

    def model_hashes(self) -> dict[str, str]:
        paths = {
            "offline_model": self.model,
            "tokens": self.tokens,
        }
        if self.punctuation_model.exists():
            paths["punctuation"] = self.punctuation_model
        return {name: sha256(path) for name, path in paths.items()}

    def add_punctuation(self, text: str) -> str:
        text = clean_decoded_text(text)
        if not text:
            return text
        if self.punctuation is not None:
            text = clean_decoded_text(self.punctuation.add_punctuation(text))
        return apply_replacements(text, self.replacements)

    def decode_segment(self, samples: np.ndarray, sample_rate: int) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self.recognizer.decode_stream(stream)
        return clean_decoded_text(stream.result.text)

    def decode(self, samples: np.ndarray, sample_rate: int, chunk_samples: int) -> tuple[str, str, list[tuple[int, int]]]:
        raw = self.decode_segment(samples, sample_rate)
        return raw, self.add_punctuation(raw), [(0, len(samples))]


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


def write_report(out_dir: Path, rows: list[dict[str, object]], decoder: EdgeLikeDecoder, args: argparse.Namespace) -> None:
    total_ref = sum(int(r["ref_chars"]) for r in rows)
    total_sub = sum(int(r["sub"]) for r in rows)
    total_del = sum(int(r["del"]) for r in rows)
    total_ins = sum(int(r["ins"]) for r in rows)
    micro_cer = (total_sub + total_del + total_ins) / total_ref if total_ref else 0.0
    macro_cer = statistics.mean(float(r["cer"]) for r in rows) if rows else 0.0
    # Post-processing CER: scored on final delivered text (post punctuation + replacements).
    total_psub = sum(int(r.get("psub", 0)) for r in rows)
    total_pdel = sum(int(r.get("pdel", 0)) for r in rows)
    total_pins = sum(int(r.get("pins", 0)) for r in rows)
    micro_cer_punct = (total_psub + total_pdel + total_pins) / total_ref if total_ref else 0.0
    macro_cer_punct = statistics.mean(float(r.get("cer_punct", r["cer"])) for r in rows) if rows else 0.0
    rtfs = [float(r["rtf"]) for r in rows]
    empty = sum(1 for r in rows if not str(r["hyp_norm"]))

    by_dataset: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)

    report = out_dir / "report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# ASR Benchmark Report\n\n")
        f.write(f"- Manifest: `{args.manifest}`\n")
        f.write(f"- Samples: {len(rows)}\n")
        f.write(f"- Total audio: {sum(float(r['duration_sec']) for r in rows):.1f}s\n")
        f.write(f"- Micro CER (raw ASR): {micro_cer:.2%}\n")
        f.write(f"- Macro CER (raw ASR): {macro_cer:.2%}\n")
        f.write(f"- Micro CER (post-processing, final text): {micro_cer_punct:.2%}\n")
        f.write(f"- Macro CER (post-processing, final text): {macro_cer_punct:.2%}\n")
        f.write(f"- Empty hypothesis rate: {empty}/{len(rows)}\n")
        f.write(f"- RTF p50/p90/p95: {percentile(rtfs, 0.50):.3f} / {percentile(rtfs, 0.90):.3f} / {percentile(rtfs, 0.95):.3f}\n")
        f.write(f"- sherpa-onnx: {getattr(sherpa_onnx, '__version__', 'unknown')}\n")
        f.write(f"- Platform: {platform.platform()}\n")
        f.write(f"- Decoder: {args.decoder}\n")
        if args.offline_model_dir:
            f.write(f"- Offline model dir: `{args.offline_model_dir}`\n")
        f.write(f"- Chunk samples: {args.chunk_samples}\n")
        f.write(f"- ASR precision: {args.precision}\n")
        f.write(f"- Punctuation enabled: {not args.no_punctuation}\n")
        f.write(f"- Replacements: {len(decoder.replacements)} entries"
                f"{f' from {args.replacements_file}' if args.replacements_file else ''}\n")
        hotwords_n = 0
        if decoder.hotwords_file is not None:
            try:
                hotwords_n = sum(1 for line in Path(decoder.hotwords_file).read_text(encoding="utf-8").splitlines() if line.strip())
            except OSError:
                hotwords_n = 0
        f.write(f"- Hotwords: {hotwords_n} entries"
                f"{f' from {decoder.hotwords_file}' if decoder.hotwords_file else ' (none)'}"
                f", score={decoder.hotwords_score}\n\n")

        f.write("## Models\n\n")
        for name, digest in decoder.model_hashes().items():
            f.write(f"- {name}: `{digest}`\n")

        f.write("\n## By Dataset\n\n")
        f.write("| Dataset | N | Duration(s) | Micro CER (raw) | Micro CER (post) | RTF p50 |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for dataset, items in sorted(by_dataset.items()):
            ref = sum(int(r["ref_chars"]) for r in items)
            err = sum(int(r["sub"]) + int(r["del"]) + int(r["ins"]) for r in items)
            perr = sum(int(r.get("psub", 0)) + int(r.get("pdel", 0)) + int(r.get("pins", 0)) for r in items)
            f.write(
                f"| {dataset} | {len(items)} | {sum(float(r['duration_sec']) for r in items):.1f} | "
                f"{(err / ref if ref else 0):.2%} | {(perr / ref if ref else 0):.2%} | "
                f"{percentile([float(r['rtf']) for r in items], 0.50):.3f} |\n"
            )

        f.write("\n## Worst 20 (by post-processing CER)\n\n")
        f.write("| # | Utt | Dataset | Dur | CER raw | CER post | Ref | Hyp (final) |\n")
        f.write("|---:|---|---|---:|---:|---:|---|---|\n")
        for i, row in enumerate(sorted(rows, key=lambda r: float(r.get("cer_punct", r["cer"])), reverse=True)[:20], 1):
            ref = str(row["ref_raw"]).replace("|", " ")
            hyp = str(row["hyp_punctuated"]).replace("|", " ")
            f.write(
                f"| {i} | {row['utt_id']} | {row['dataset']} | {float(row['duration_sec']):.1f} | "
                f"{float(row['cer']):.2%} | {float(row.get('cer_punct', row['cer'])):.2%} | {ref[:70]} | {hyp[:70]} |\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--rawfile", default=Path("/Users/cannkit/ASR/entry/src/main/resources/rawfile"), type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-samples", type=int, default=1600)
    parser.add_argument("--decoder", choices=("streaming", "offline-paraformer", "sensevoice"), default="streaming")
    parser.add_argument("--offline-model-dir", type=Path, default=None)
    parser.add_argument("--precision", choices=("int8", "fp32"), default="int8")
    parser.add_argument("--no-punctuation", action="store_true")
    parser.add_argument("--sensevoice-language", default="zh")
    parser.add_argument("--no-sensevoice-itn", action="store_true")
    parser.add_argument("--replacements-file", type=Path, default=None,
                        help="JSON array of {from, to} pairs applied after punctuation (mirrors app).")
    parser.add_argument("--hotwords-file", type=Path, default=None,
                        help="Built-in hotword list (one phrase per line) passed to sherpa paraformer.")
    parser.add_argument("--hotwords-score", type=float, default=1.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    replacements = load_replacement_rules(args.replacements_file)
    rows_in = read_manifest(args.manifest, args.limit)
    if args.decoder == "streaming":
        decoder = EdgeLikeDecoder(args.rawfile, with_punctuation=not args.no_punctuation,
                                  precision=args.precision, replacements=replacements,
                                  hotwords_file=args.hotwords_file, hotwords_score=args.hotwords_score)
    else:
        if args.offline_model_dir is None:
            raise ValueError("--offline-model-dir is required for offline decoders")
        decoder = OfflineFinalDecoder(
            args.offline_model_dir,
            args.rawfile,
            with_punctuation=not args.no_punctuation,
            decoder_type=args.decoder,
            sensevoice_language=args.sensevoice_language,
            sensevoice_use_itn=not args.no_sensevoice_itn,
            replacements=replacements,
            hotwords_file=args.hotwords_file, hotwords_score=args.hotwords_score,
        )

    result_rows: list[dict[str, object]] = []
    for idx, item in enumerate(rows_in, 1):
        wav = Path(item["wav"])
        samples, sample_rate = load_audio(wav)
        started = time.perf_counter()
        try:
            raw, punctuated, segments = decoder.decode(samples, sample_rate, args.chunk_samples)
            error = ""
        except Exception as exc:
            raw, punctuated, segments = "", "", []
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        duration = len(samples) / sample_rate
        ref_raw = str(item.get("ref", ""))
        ref_norm = normalize_for_score(ref_raw)
        hyp_norm = normalize_for_score(raw)
        sub, delete, ins = edit_counts(ref_norm, hyp_norm)
        ref_chars = len(ref_norm)
        cer = (sub + delete + ins) / ref_chars if ref_chars else 0.0
        # Post-processing CER: scored on the final delivered text (after punctuation +
        # replacement dictionary), which is what the user sees. This is the metric that
        # reflects replacements.json / ITN / punctuation improvements.
        hyp_punct_norm = normalize_for_score(punctuated)
        psub, pdel, pins = edit_counts(ref_norm, hyp_punct_norm)
        cer_punct = (psub + pdel + pins) / ref_chars if ref_chars else 0.0
        row = {
            "index": idx,
            "utt_id": item.get("utt_id", wav.stem),
            "dataset": item.get("dataset", ""),
            "split": item.get("split", ""),
            "wav": str(wav),
            "ref_raw": ref_raw,
            "ref_norm": ref_norm,
            "hyp_raw_asr": raw,
            "hyp_punctuated": punctuated,
            "hyp_norm": hyp_norm,
            "duration_sec": duration,
            "segment_mode": "direct" if len(segments) <= 1 else "10s",
            "num_segments": len(segments),
            "segment_boundaries_sec": [[round(a / sample_rate, 3), round(b / sample_rate, 3)] for a, b in segments],
            "decode_ms": round(elapsed * 1000, 3),
            "rtf": elapsed / duration if duration > 0 else 0.0,
            "ref_chars": ref_chars,
            "sub": sub,
            "del": delete,
            "ins": ins,
            "cer": cer,
            "psub": psub,
            "pdel": pdel,
            "pins": pins,
            "cer_punct": cer_punct,
            "empty_hyp": hyp_norm == "",
            "error": error,
        }
        result_rows.append(row)
        print(
            f"[{idx:03d}/{len(rows_in):03d}] {row['utt_id']} dur={duration:.1f}s "
            f"cer={cer:.2%} cer_post={cer_punct:.2%} rtf={row['rtf']:.3f} hyp={punctuated[:60]}"
        )

    jsonl = args.out_dir / "results.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in result_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = args.out_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()) if result_rows else [])
        if result_rows:
            writer.writeheader()
            for row in result_rows:
                writer.writerow(row)

    write_report(args.out_dir, result_rows, decoder, args)
    print(f"\nWrote {args.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
