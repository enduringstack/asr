# ASR Model Selection Benchmark

Benchmark set: `/tmp/asr-bench/aishell100/manifest.jsonl`

Samples: 100 AISHELL-1 utterances, 492.1s total audio. CER scoring removes punctuation and spaces.

| Model | Mode | Micro CER | Macro CER | Median CER | Exact | CER > 20% | RTF p50/p90/p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| `sherpa-onnx-streaming-paraformer-bilingual-zh-en` int8 | streaming, current app model | 10.44% | 10.58% | 7.14% | 33/100 | 17/100 | 0.090 / 0.102 / 0.112 |
| `iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online` | streaming, ModelScope profile `[5,10,5]`, lookback `0/0` | 9.19% | 9.07% | 5.88% | 42/100 | 12/100 | 0.314 / 0.335 / 0.340 |
| `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | offline large | 2.85% | 2.86% | 0.00% | 72/100 | 2/100 | 0.089 / 0.122 / 0.125 |

## Decision

For product-quality final transcripts, use Paraformer-large/offline as the accuracy path. It cuts Micro CER from 10.44% to 2.85% on the same local benchmark and reduces severe failures from 17 to 2 utterances.

For live partial captions, the current sherpa int8 model is still the lightest path. FunASR streaming-online is slightly more accurate on this benchmark, but the Python CPU streaming loop is much slower because it runs per 600ms chunk. It is only worth replacing the live path if we convert and validate it on CANN/NPU.

Do not use the FunASR AutoModel realtime profile `[0,10,5]` with lookback `4/1` for this model in our benchmark harness. It produced repeated tail text in smoke tests. The ModelScope profile `[5,10,5]` with lookback `0/0` is the reliable streaming configuration here.

The current sherpa bilingual model's fp32 files did not improve accuracy on this benchmark; the previous fp32 run was 10.79% Micro CER. Staying on int8 is correct for the current app until a stronger model is integrated.

## Product Chain

Recommended launch chain:

1. Capture audio with VAD for endpointing and silence trimming.
2. Show low-latency interim text from the current int8 streaming model, or keep it as a fallback.
3. On segment end or button release, run Paraformer-large/offline as a second pass and replace the interim text with final text.
4. Run punctuation after the final ASR text.
5. Run speaker diarization/voiceprint only after stable segment text is available.

Enhancement should default to off. Turn it on only when measured noise level is high or the user explicitly enables it, because enhancement can distort clean speech and hurt ASR.

## Reports

- Current sherpa int8: `/Users/cannkit/ASR/benchmarks/asr_aishell100/report.md`
- Current sherpa fp32: `/Users/cannkit/ASR/benchmarks/asr_aishell100_fp32/report.md`
- FunASR streaming-online: `/Users/cannkit/ASR/benchmarks/funasr_streaming_online_modelscope_aishell100/report.md`
- FunASR offline-large: `/Users/cannkit/ASR/benchmarks/funasr_offline_large_aishell100/report.md`
- Benchmark runner: `/Users/cannkit/ASR/tools/run_funasr_benchmark.py`

## External Reference Points

FunASR paper reports Paraformer-large at 1.95% CER on AISHELL test, 2.85% on AISHELL-2 test_ios, and 6.97% on WenetSpeech test_meeting. It also reports ONNX int8 Paraformer-large preserving 1.95% CER with faster runtime. These are reference numbers from their setup, not guaranteed numbers for our device or our audio domain.

Sources:

- Paraformer paper: https://arxiv.org/abs/2206.08317
- FunASR paper / runtime benchmark: https://ar5iv.labs.arxiv.org/html/2305.11013
- FunASR ONNX C++ benchmark: https://github.com/modelscope/FunASR/blob/main/runtime/docs/benchmark_onnx_cpp.md
- Streaming online model card cached locally from ModelScope: `/Users/cannkit/.cache/modelscope/hub/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/README.md`
