# APP 模型清单

- 生成时间：2026-07-09 15:23:17 CST
- 工程：`/Users/cannkit/ASR`
- HAP：`/Users/cannkit/ASR/entry/build/default/outputs/default/entry-default-signed.hap`
- 已安装并启动设备：`5XM0125A10000251`

## 当前默认链路

- 按住说话：`Silero VAD -> streaming Paraformer 实时中间结果 -> 松手后 offline Paraformer 最终二次识别 -> 标点恢复`
- 导入音频：默认 `offline Paraformer -> 标点恢复`
- 长音频：按约 10 秒切段，分段追加显示，最后合并文本并加标点
- 说话人分段、分段校正、原始增强对比、声纹相关 UI：默认隐藏且关闭

## 模型表

| 模型 | 当前用途 | 大小 | 本地准确率/效果 | 实时性 |
|---|---|---:|---|---|
| `sherpa-onnx-streaming-paraformer-bilingual-zh-en` int8 | 按住说话的实时中间结果 | 226.2 MiB | AISHELL-1 100条：CER 10.44% | RTF p50/p90/p95：0.090 / 0.102 / 0.112 |
| `sherpa-onnx-paraformer-zh-2024-03-09` int8 | 松手后的最终二次识别；音频文件默认识别 | 216.9 MiB | AISHELL-1 100条：CER 5.01% | RTF p50/p90/p95：0.012 / 0.012 / 0.012 |
| `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09` | 包内候选/兜底模型，当前首页默认不走它 | 226.4 MiB | AISHELL-1 100条：CER 6.61% | RTF p50/p90/p95：0.017 / 0.019 / 0.020 |
| `sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8` | 标点恢复 | 72.0 MiB | 没有单独标点 F1 benchmark；ASR benchmark 已包含它 | 很快，耗时包含在 ASR RTF 内 |
| `silero_vad.onnx` | VAD，控制起点、静音、尾音 | 0.6 MiB | 没有单独误检/漏检 benchmark | 端侧实时，开销很小 |
| `sherpa-onnx-pyannote-segmentation-3-0` | 说话人分段，默认隐藏且关闭 | 1.5 MiB | 14条会议无人工说话人标注，只做稳定性代理 | 与声纹链路合计 RTF 约 0.147 |
| `sherpa-onnx-3dspeaker-sv-zh-cn` | 声纹/说话人 embedding，默认隐藏且关闭 | 37.8 MiB | 无正式准确率；当前阈值 0.60 | 只在声纹/说话人功能开启或恢复声纹时用 |

## 说明

- CER 是字符错误率，越低越好。
- RTF 是实时因子，越低越快。RTF 0.1 表示处理 10 秒音频约需 1 秒。
- AISHELL-1 100条 benchmark 有人工标注，因此可以计算 CER。
- `/Users/cannkit/Downloads/audios` 的 14 条会议音频没有人工标准答案，不能计算真实 CER；当前报告只能用于观察输出质量、稳定性和耗时。
- 模型大小为资源文件/目录大小，不是参数量。

