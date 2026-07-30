# FunASR Benchmark Report

- Manifest: `/tmp/asr-bench/aishell100/manifest.jsonl`
- Model kind: `streaming-online`
- Model id: `iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online`
- Streaming profile: `funasr`
- Samples: 5
- Total audio: 21.2s
- Micro CER: 37.10%
- Macro CER: 34.34%
- Median CER: 25.00%
- Exact matches: 0/5
- CER > 20%: 4/5
- Empty hypothesis rate: 0/5
- RTF p50/p90/p95: 0.325 / 0.338 / 0.338
- Platform: macOS-27.0-arm64-arm-64bit

## Worst 20

| # | Utt | Dur | CER | Ref | Hyp |
|---:|---|---:|---:|---|---|
| 1 | BAC009S0724W0124 | 5.5 | 62.50% | 广州  二手  住宅  市场  表现  一直  相对  稳健 | 广州二手住宅住场的有是一是现现的的的 |
| 2 | BAC009S0724W0125 | 3.9 | 50.00% | 而  在  股市  大幅  震荡  的  环境  下 | 而在股市大大幅震降的境的增的 |
| 3 | BAC009S0724W0121 | 4.3 | 25.00% | 广州  市  房地  产中  介  协会  分析 | 广州市房地产中介协会会司公 |
| 4 | BAC009S0724W0122 | 4.3 | 23.08% | 广州  市  房地  产中  介  协会  还  表示 | 广州市房地产中介协会表示示意 |
| 5 | BAC009S0724W0123 | 3.2 | 11.11% | 相比  于  其他  一  线  城市 | 相比于其他一线城市市 |
