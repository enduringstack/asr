#!/usr/bin/env python3
"""Build a domain-tagged AISHELL manifest for hotword / dictionary benchmarking.

Reads the AISHELL manifest produced by prepare_aishell_hf_sample.py and writes a copy
where each row is tagged with `dataset` = "domain" (its ref contains a proper-noun-ish
keyword) or "aishell" otherwise. The run_asr_benchmark.py harness already breaks CER
down by the `dataset` field, so this lets a hotword/replacement A/B run show per-bucket
CER without changing the audio.

CAVEAT: AISHELL-1 dev audio does not contain the app's hotword phrases (华为/鸿蒙/
Paraformer/...), so a CER delta of ~0 from enabling `--hotwords-file` on this set is
expected BOTH because (a) Paraformer hotword biasing is architecturally a no-op in
sherpa-onnx, and (b) the hotword phrases are simply not in the audio. To disambiguate,
add real domain audio (wav + ref rows) containing the hotword phrases to the manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Tokens that, if present in the ref, mark the utterance as a "domain" (proper-noun-bearing)
# row for the by-dataset CER breakdown. Extend freely.
DOMAIN_KEYWORDS = (
    "北京", "上海", "广州", "深圳", "杭州", "南京", "武汉", "成都", "西安", "重庆",
    "香港", "台湾", "澳门",
    "公司", "集团", "银行", "协会", "研究院", "大学", "医院", "政府", "部门",
    "华为", "阿里", "腾讯", "百度", "字节", "微软", "苹果", "谷歌",
    "人工智能", "语音识别", "互联网", "芯片",
)


def ref_norm(ref: str) -> str:
    return ref.replace(" ", "").replace("　", "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("/tmp/asr-bench/aishell100/manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/asr_aishell100/aishell100_domain.jsonl"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    domain_n = 0
    total = 0
    with args.src.open("r", encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            norm = ref_norm(str(row.get("ref", "")))
            is_domain = any(kw in norm for kw in DOMAIN_KEYWORDS)
            if is_domain:
                domain_n += 1
                row["dataset"] = "domain"
            else:
                row["dataset"] = "aishell"
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {args.out}: {domain_n}/{total} rows tagged domain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
