# 端侧声纹模型选型记录

- 日期：2026-08-25
- 设备：Mac M4
- 场景：最多 10 名已注册人员，同时存在未知人员
- 原始结果：`/tmp/asr-voiceprint-eval-20260825/results`（临时目录，不作为长期归档）

## 最终结论

选择 `speech_campplus_sv_zh_en_16k-common_advanced`。

| 属性 | 值 |
|---|---|
| ONNX 大小 | 28,281,164 字节 / 26.97 MB |
| Embedding | 192 维 |
| 线程数 | 2 |
| 身份匹配阈值 | 0.60 |
| 说话人聚类阈值 | 0.85 |
| SHA-256 | `aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2` |
| 许可证 | Apache-2.0 |

## 对比结果

| 模型 | 大小 | 中文已知识别 / 未知拒识 | 跨视频已知识别 / 未知拒识 | Embedding RTF |
|---|---:|---:|---:|---:|
| 当前 ERes2Net | 37.76 MB | 100% / 95% | 54% / 100% | 0.0135 |
| CAM++ 中英双语 | 26.97 MB | 97.5% / 95% | 96% / 100% | 0.0050 |
| ERes2NetV2 | 68.13 MB | 95% / 90% | 94% / 98% | 0.0557 |

CAM++ 比当前模型小约 10.8 MB，推理快约 2.7 倍，跨视频和跨录音环境稳定性明显更好。ERes2NetV2 虽然公开标准榜单更强，但在当前产品协议中没有胜出。

## 评测数据与协议

- AISHELL-3：40 名中文说话人；20 人用于调参，10 名已注册测试，10 名未知测试；加入 10 dB 噪声和混响。
- VoxCeleb1：40 名跨视频说话人；注册和测试语音来自不同视频，只做本地测试，不分发数据。
- AISHELL-4：38 分 47 秒、5 名真实中文会议说话人，使用 RTTM 真值。
- 注册使用多条语音聚合；开放集测试同时包含已注册和未知人员。

单条噪声短句的已知接纳率为 42.9%，聚合三段后为 95%。因此产品必须先累计同 cluster 至少 3 段或 4 秒有效语音，再进行身份命名。

## AISHELL-4 整场结果

| 指标 | 结果 |
|---|---:|
| 音频时长 | 2327.119 秒 |
| 参考说话人数 | 5 |
| 输出簇数 | 12 |
| DER | 7.93% |
| 运行时间 | 173.79 秒 |
| RTF | 0.0747 |

系统存在过分拆分，但没有为了减少簇数采用容易合并错人的激进阈值。产品侧通过“多个 cluster 匹配到同一已注册 personId 后统一显示”修复可见身份，不直接破坏底层 diarization 边界。

## 工程落地约束

1. 正式包只包含 CAM++，删除现有 38 MB 模型，不加入 68 MB 模型。
2. `VOICEPRINT_EMBEDDING_DIM` 从 512 改为 192。
3. `VOICEPRINT_MATCH_THRESHOLD` 固定为 0.60，`DIARIZATION_CLUSTER_THRESHOLD` 改为 0.85。
4. 旧 512 维模板标记为需要重新注册，不做维度转换。
5. 阈值与模型 id、SHA 和评测版本一起保存，防止未来误用。

## 参考

- [CAM++ 模型卡](https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced)
- [3D-Speaker benchmark](https://github.com/modelscope/3D-Speaker#benchmark)
- [AISHELL-3](https://huggingface.co/datasets/AISHELL/AISHELL-3)
- [AISHELL-4 paper](https://arxiv.org/abs/2104.03603)
