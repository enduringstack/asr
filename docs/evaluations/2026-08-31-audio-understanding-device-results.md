# 声音理解真机评测结果

日期：2026-08-31
设备：HarmonyOS 手机 `5NC0125514000056`
应用：`com.chen.myapplication`
推理后端：sherpa-onnx 1.13.3 / CPU / 4 threads

## 模型

- 基础模型：`iic/SenseVoiceSmall`
- sherpa-onnx 模型包：`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17`
- 量化：INT8
- 模型文件：239,334,411 bytes（约 228.2 MiB）
- 源模型 SHA-256：`c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51`
- 应用内模型 SHA-256：`359a0e2ac5968e52941d2e6a9b5a73174631f89066355ae5699cb82ba7b56a17`

sherpa-onnx 的 HarmonyOS 接口没有暴露 FunASR 的
`ban_emo_unk` 参数。应用内模型通过
`tools/patch_sensevoice_ban_emo_unknown.py` 将
`<|EMO_UNKNOWN|>` 的 CTC logit 屏蔽，语义与 FunASR
`ban_emo_unk=True` 一致，不改变其他类别的分数。

此前工程内的 2025-09-09 模型实际来自
`ASLP-lab/WSYue-ASR/sensevoice_small_yue`。它是粤语识别微调模型，首轮真机
评测把所有英文判为粤语、情感判为中性、事件判为语音，仅得到 3/30，已从
应用资源中移除并保存在 `dev_assets` 作为可恢复备份。

## 固定测试集

- CREMA-D：18 条，覆盖愤怒、厌恶、恐惧、高兴、中性、悲伤 6 类；每类 3 条。
- VocalSound：12 条，覆盖咳嗽、笑声、喷嚏 3 类；每类 4 条。
- 全部音频为 16 kHz、单声道、PCM16 WAV，可在 App 内逐条播放。
- 只比较数据集提供真值的目标字段。CREMA-D 不参与事件准确率，VocalSound
  不参与情感准确率。
- 处理失败仍保留在准确率分母中。

## 真机结果

| 指标 | 命中 | 准确率 |
|---|---:|---:|
| 总体 | 23 / 30 | 76.7% |
| 情感 | 14 / 18 | 77.8% |
| 声音事件 | 9 / 12 | 75.0% |
| 处理失败 | 0 / 30 | 0% |

批量评测从第一条结果到汇总日志约 12.0 秒。首条 3.9 秒 CREMA-D 音频在
手机 UI 中显示模型处理耗时 235 ms。

未匹配 7 条：

- 恐惧：`1036_TSI_FEA_XX`、`1026_IEO_FEA_HI`、`1008_TSI_FEA_XX`
- 高兴：`1065_IEO_HAP_MD`
- 喷嚏：VocalSound test rows 6、23、38

## UI 与播放验收

- 工作台可选择“声音理解”，录音结果展示语种、情感、声音事件。
- 数据集评测入口打开独立的“声音理解评测”页面。
- 页面展示真实标签、SenseVoice 输出、完整原生辅助字段、ASR 文本、耗时和
  音频时长。
- 支持全部/情感/事件/未匹配筛选、单条运行和 30 条批量运行。
- 第一条音频播放时 UI 显示“正在播放 0:01 / 0:03”，按钮切换为“停止”。
- 真机结果截图：`docs/evaluations/audio-understanding-device-result.jpeg`

## 结论

官方通用 SenseVoiceSmall 能在当前鸿蒙端侧链路上同时输出 LID、SER 和 AED；
固定跨数据集测试达到 76.7%，可以用于能力验证和产品原型，但不能把该结果
解释为生产场景的普适准确率。当前主要误差集中在 CREMA-D 的恐惧类和
VocalSound 的喷嚏类，后续如需生产级效果，应增加中文真实场景数据并按场景
重新标定或使用专用 SER/AED 模型。
