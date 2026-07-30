# 热词偏置实测实验（确认 Paraformer no-op）

## 目的

参照《开源ASR优化指南》"热词/上下文偏置（最快见效）"机制，在当前 sherpa-onnx Paraformer 模型上实测热词是否生效。

经 sherpa_onnx 包 CHANGELOG 核实：**热词偏置（`hotwordsFile`/`hotwordsBuf`）与 `modified_beam_search` 只对 transducer（Zipformer）和 LLM-ASR（Qwen3-ASR/FunASR-Nano）生效；对 Paraformer/SenseVoice（CIF 非自回归）预期是 no-op**。本实验用基准坐实这一预期。

## Python 环境

`sherpa-onnx` 装在 `wenet` conda 环境。所有命令用显式路径：
```
PY=/Users/cannkit/anaconda3/envs/wenet/bin/python3
```
（默认 `python3` 是 anaconda base，没装 sherpa_onnx，会 import 失败。）

## 前置：准备数据

```bash
# 1. 生成 aishell100 音频 + manifest（HuggingFace 镜像，需 curl/HF）
$PY tools/prepare_aishell_hf_sample.py --split dev --limit 100
#   产出 /tmp/asr-bench/aishell100/{manifest.jsonl,wav/}

# 2. 生成领域 manifest（按专有名词关键词打 dataset 标签，供 by-dataset CER 分桶）
$PY tools/build_domain_manifest.py
#   产出 benchmarks/asr_aishell100/aishell100_domain.jsonl
```

## 实测：有/无热词对照（streaming + offline-paraformer）

热词词表：`entry/src/main/resources/rawfile/hotwords.txt`

```bash
PY=/Users/cannkit/anaconda3/envs/wenet/bin/python3
RAW=entry/src/main/resources/rawfile
HW=$RAW/hotwords.txt
MAN=benchmarks/asr_aishell100/aishell100_domain.jsonl
PARA=$RAW/sherpa-onnx-paraformer-zh-2024-03-09

# streaming：开热词 vs 关热词
$PY tools/run_asr_benchmark.py --manifest $MAN --out-dir benchmarks/hotwords_streaming_on  \
  --decoder streaming --precision int8 --hotwords-file $HW --hotwords-score 1.5
$PY tools/run_asr_benchmark.py --manifest $MAN --out-dir benchmarks/hotwords_streaming_off \
  --decoder streaming --precision int8

# offline-paraformer：开热词 vs 关热词
$PY tools/run_asr_benchmark.py --manifest $MAN --out-dir benchmarks/hotwords_offline_on  \
  --decoder offline-paraformer --offline-model-dir $PARA --hotwords-file $HW --hotwords-score 1.5
$PY tools/run_asr_benchmark.py --manifest $MAN --out-dir benchmarks/hotwords_offline_off \
  --decoder offline-paraformer --offline-model-dir $PARA
```

对比每对 `report.md` 的 Micro CER 行（整体 + by-dataset 的 `domain`/`aishell` 两个桶）。

## 预期结果与判读

- **预期**：开/关热词的 CER delta ≈ 0。
- **判读**：delta≈0 → 与 sherpa-onnx 源码一致，坐实"Paraformer 热词不可用"。真正的"热词"收益已由后处理自定义词典（`replacements.json` + `applyReplacementDictionary`）兜底。

## 精确结论（查 sherpa-onnx 1.13.4 源码）

`OnlineRecognizer.from_paraformer` / `OfflineRecognizer.from_paraformer` 工厂**根本不接受 `hotwords_file`/`hotwords_score` 参数**——传入会 `TypeError`，benchmark 捕获后打 `[warn] ... does not accept hotwords` 并以**无热词**方式继续。因此"开热词"那次实际等同"关热词"。这比"no-op"更强：热词对 paraformer 是**结构上不可用**（paraformer 只支持 greedy_search，而热词只在 modified_beam_search 下生效）。

## 固有混淆（已消解）

原担心"aishell 音频不含热词表里的词导致测不出"——现在已无需纠结：python 工厂压根不接受热词，"开热词"那次本就回退为无热词。要真正测热词偏置，必须换 transducer（Zipformer）或 LLM-ASR（Qwen3-ASR/FunASR-Nano）模型，见 deferred。

## 后处理词典对照（真正生效的准确率收益，参考）

```bash
PY=/Users/cannkit/anaconda3/envs/wenet/bin/python3
$PY tools/run_asr_benchmark.py --manifest $MAN --out-dir benchmarks/repl_on  \
  --decoder streaming --precision int8 \
  --replacements-file entry/src/main/resources/rawfile/replacements.json
$PY tools/run_asr_benchmark.py --manifest $MAN --out-dir benchmarks/repl_off \
  --decoder streaming --precision int8
```

判据：命中替换的行 CER 下降；aishell 基线（streaming ≈ 10.44%）不回退。

## deferred（本轮不做）

要真正吃到热词偏置与 Prompt 上下文收益（文章机制 #1/#3），需换模型：
- 流式换 streaming Zipformer transducer（`hotwordsBuf`+`modified_beam_search` 生效）；
- 或离线终跳换 Qwen3-ASR / FunASR-Nano（`OfflineQwen3AsrModelConfig.hotwords` / `OfflineFunASRNanoModelConfig.hotwords` 原生支持 + prompt 上下文）。

需下载模型 + 重跑基准，单独立项。
