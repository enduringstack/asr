# Streaming Paraformer Diagnosis

- Date: 2026-07-01
- Target model: `sherpa-onnx-streaming-paraformer-bilingual-zh-en`
- App path: `/Users/cannkit/ASR`
- Device smoke test: `4GHBB25508005910`

## Finding

The high CER is not primarily caused by int8 quantization, chunk size, endpoint configuration, or tail padding. It is caused by using the online/streaming first-pass Paraformer as the final transcript.

FunASR's production real-time package is designed as a collaborative chain: VAD + online ASR + punctuation, and in `2pass` mode it also uses an offline model to correct the final sentence. Our app previously used the online result as final unless a fallback condition was triggered.

## Evidence

### Existing AISHELL100 Reports

| Chain | Micro CER | Notes |
|---|---:|---|
| Sherpa streaming Paraformer int8 | 10.44% | `/Users/cannkit/ASR/benchmarks/asr_aishell100/report.md` |
| Sherpa streaming Paraformer fp32 | 10.79% | `/Users/cannkit/ASR/benchmarks/asr_aishell100_fp32/report.md` |
| FunASR online streaming | 9.19% | `/Users/cannkit/ASR/benchmarks/funasr_streaming_online_modelscope_aishell100/report.md` |
| Sherpa offline Paraformer 2024 int8 | 5.01% | `/Users/cannkit/ASR/benchmarks/sherpa_offline_paraformer_int8_20240309_aishell100/report.md` |
| FunASR PyTorch offline-large | 2.85% | `/Users/cannkit/ASR/benchmarks/funasr_offline_large_aishell100/report.md` |

### Parameter Sweep

On the first 30 AISHELL samples:

| Precision | Endpoint | Chunk | Tail | CER |
|---|---:|---:|---:|---:|
| int8 | off | 800 | 1s | 8.98% |
| int8 | off | 1600 | 1s | 8.98% |
| int8 | off | 3200 | 1s | 8.98% |
| int8 | on | 800 | 1s | 8.98% |
| int8 | off | 1600 | 2s | 8.77% |

Full AISHELL100 tail check:

| Tail Padding | Micro CER |
|---:|---:|
| 1s | 10.44% |
| 2s | 10.37% |
| 4s | 10.37% |

This rules out a simple endpoint/chunk/tail integration bug.

## Change Applied

`chooseMicFinalText()` now uses offline Paraformer as a second-pass final recognizer for press-to-talk audio:

- online streaming Paraformer remains responsible for partial display
- offline Paraformer-large produces the final text after release
- if offline second-pass fails to load or returns empty, the app falls back to the existing streaming result

Relevant file:

- `/Users/cannkit/ASR/entry/src/main/ets/workers/StreamingSherpaWorker.ets`

## Verification

- `hvigor assembleHap`: successful
- Installed HAP to `4GHBB25508005910`: successful
- Device smoke test:
  - `sherpa-asr-started`
  - `sherpa-asr-final`
  - `sherpa-asr-stopped`
  - no `BusinessError`, `ValueSerialize`, `readData`, crash, or `sherpa-asr-error`

## Product Implication

For press-to-talk:

- partial text quality is still bounded by the streaming model
- final text should now align with the offline Paraformer 2024 int8 path, whose measured AISHELL100 CER is 5.01%
- first use after app start may have extra latency because the offline model is loaded lazily

