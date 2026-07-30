# ASR Benchmark Report

- Manifest: `/tmp/asr_aishell100_manifest.jsonl`
- Samples: 10
- Total audio: 51.4s
- Micro CER: 3.92%
- Macro CER: 4.83%
- Empty hypothesis rate: 0/10
- RTF p50/p90/p95: 0.017 / 0.018 / 0.019
- sherpa-onnx: 1.13.3
- Platform: macOS-27.0-arm64-arm-64bit-Mach-O
- Decoder: sensevoice
- Offline model dir: `/Users/cannkit/ASR/dev_assets/asr_models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09`
- Chunk samples: 1600
- ASR precision: int8
- Punctuation enabled: True

## Models

- offline_model: `12ca1a2ae7ecf3e0019ef2822307ee0b5cadc9196569e379b4c4026f8205276d`
- tokens: `f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc`
- punctuation: `65a3fb9f5ad7bfb96bf69e0dc4481df97f6ee60513c1d94ce981ba6effd524b1`

## By Dataset

| Dataset | N | Duration(s) | Micro CER | Macro CER | RTF p50 |
|---|---:|---:|---:|---:|---:|
| AISHELL-1 | 10 | 51.4 | 3.92% | 4.83% | 0.017 |

## Worst 20

| # | Utt | Dataset | Dur | CER | Ref | Hyp |
|---:|---|---|---:|---:|---|---|
| 1 | BAC009S0724W0129 | AISHELL-1 | 3.5 | 33.33% | 其中  越秀  区  涨幅  领先 | 其中,越秀区丈夫领衔。 |
| 2 | BAC009S0724W0127 | AISHELL-1 | 5.4 | 6.25% | 但  受  穗  六条  及  二  套房  首  付  七成  的  制约 | 但受贿六条及二套房首付七成的制约。 |
| 3 | BAC009S0724W0128 | AISHELL-1 | 7.3 | 4.35% | 下半  年  楼市  能  是否  能  回到  快速  上升  通道  依然  存在  变数 | 下半年,楼市能是否能回荡快速上升,通道依然存在变数。 |
| 4 | BAC009S0724W0130 | AISHELL-1 | 7.4 | 4.35% | 天河  区  的  签约  面积  在  豪宅  交  投  增多  的  带动  下  上升  较快 | 天河区的签约面积在豪宅交头增多的,带动下上升较快。 |
| 5 | BAC009S0724W0121 | AISHELL-1 | 4.3 | 0.00% | 广州  市  房地  产中  介  协会  分析 | 广州市房地产中介协会分析。 |
| 6 | BAC009S0724W0122 | AISHELL-1 | 4.3 | 0.00% | 广州  市  房地  产中  介  协会  还  表示 | 广州市房地产中介协会还表示。 |
| 7 | BAC009S0724W0123 | AISHELL-1 | 3.2 | 0.00% | 相比  于  其他  一  线  城市 | 相比于其他一线城市。 |
| 8 | BAC009S0724W0124 | AISHELL-1 | 5.5 | 0.00% | 广州  二手  住宅  市场  表现  一直  相对  稳健 | 广州二手住宅市场表现一直相对稳健。 |
| 9 | BAC009S0724W0125 | AISHELL-1 | 3.9 | 0.00% | 而  在  股市  大幅  震荡  的  环境  下 | 而在股市大幅震荡的环境下。 |
| 10 | BAC009S0724W0126 | AISHELL-1 | 6.6 | 0.00% | 预计  第  三  季度  将  陆续  有  部分  股市  资金  重归  楼市 | 预计第三季度将陆续有部分股市资金重归楼市。 |
