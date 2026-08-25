# ADR-0004：采用 27 MB 中英双语 CAM++ 声纹模型

## 状态

Accepted

## 背景

工程正式资源中已有约 37.76 MB 的 ERes2Net 中文模型，`dev_assets` 中另有约 68.13 MB ERes2NetV2。为兼顾跨录音环境精度、端侧速度和包体，已在 Mac M4 上使用 AISHELL-3、VoxCeleb1 和 AISHELL-4 进行同口径评测。

## 决策

采用 `speech_campplus_sv_zh_en_16k-common_advanced`：ONNX 26.97 MB、192 维 embedding、2 线程。身份匹配阈值为 `0.60`，聚类阈值为 `0.85`。模型文件 SHA-256 为 `aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2`。

正式包替换现有 38 MB 模型，不同时打包旧模型或 68 MB ERes2NetV2。旧 512 维声纹不可转换为新 192 维声纹，只保留人物资料并标记需要重新注册。

## 后果

### 正面

- 模型比当前模型小约 10.8 MB，embedding RTF 从 0.0135 降至 0.0050。
- 跨视频已知识别/未知拒识从 54%/100% 提升至 96%/100%。
- AISHELL-4 整场 DER 7.93%、RTF 0.0747，可支持会后端侧处理。

### 负面

- 旧声纹必须重新注册，不能无感迁移。
- AISHELL-4 输出 12 个簇而真值为 5 人，需要依靠“相同已注册声纹合并”减少过分拆分。

### 中性

- `0.60/0.85` 仅适用于该模型和本次协议；未来模型升级继续使用同一评测门禁。

## 备选方案

- 现有 37.76 MB ERes2Net：中文测试略高，但跨视频已知识别只有 54%，拒绝。
- 68.13 MB ERes2NetV2：更大、更慢，在当前协议下也未胜出，拒绝。

## 参考

- https://k2-fsa.github.io/sherpa/onnx/harmony-os/speaker-identification.html
- https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced
- ../evaluation/2026-08-25-speaker-model-selection.md
