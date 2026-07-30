# ASR Benchmark Report

- Manifest: `/tmp/asr_aishell100_manifest.jsonl`
- Samples: 100
- Total audio: 492.1s
- Micro CER: 8.00%
- Macro CER: 9.14%
- Empty hypothesis rate: 0/100
- RTF p50/p90/p95: 0.013 / 0.015 / 0.015
- sherpa-onnx: 1.13.3
- Platform: macOS-27.0-arm64-arm-64bit-Mach-O
- Decoder: offline-paraformer
- Offline model dir: `/Users/cannkit/ASR/dev_assets/asr_models/sherpa-onnx-paraformer-zh-int8-2025-10-07`
- Chunk samples: 1600
- ASR precision: int8
- Punctuation enabled: True

## Models

- offline_model: `53813ee1d41722cc6370a571c887e6d0b391d25b8312cf714a31af85ea603812`
- tokens: `59aba8873a2ed1e122c25fee421e25f283b63290efbde85c1f01a853d83cb6e6`
- punctuation: `65a3fb9f5ad7bfb96bf69e0dc4481df97f6ee60513c1d94ce981ba6effd524b1`

## By Dataset

| Dataset | N | Duration(s) | Micro CER | Macro CER | RTF p50 |
|---|---:|---:|---:|---:|---:|
| AISHELL-1 | 100 | 492.1 | 8.00% | 9.14% | 0.013 |

## Worst 20

| # | Utt | Dataset | Dur | CER | Ref | Hyp |
|---:|---|---|---:|---:|---|---|
| 1 | BAC009S0724W0200 | AISHELL-1 | 3.5 | 66.67% | 杨  丙  卿  承认  指控  事实 | 一阳病情超人指控事实。 |
| 2 | BAC009S0724W0197 | AISHELL-1 | 3.6 | 60.00% | 多次  指令  邢  某  向  某  二人 | 多户指定刑马相貌二人。 |
| 3 | BAC009S0724W0214 | AISHELL-1 | 6.6 | 50.00% | 原  重组  委  委员  吴  建敏  资料  图 | 元重组为美元吴建民辅料图。 |
| 4 | BAC009S0724W0188 | AISHELL-1 | 3.3 | 44.44% | 二中  院  开庭  审理  此案 | 二、中院开定顺利父案。 |
| 5 | BAC009S0724W0169 | AISHELL-1 | 3.6 | 40.00% | 再有  两  宗地  块  抢闸  入市 | 占有两宗地块强诈入室。 |
| 6 | BAC009S0724W0190 | AISHELL-1 | 3.7 | 36.36% | 杨  丙  卿  将  成为  国内  因此  罪 | 杨病清将成为国内引妇罪。 |
| 7 | BAC009S0724W0129 | AISHELL-1 | 3.5 | 33.33% | 其中  越秀  区  涨幅  领先 | 其中,女越秀区丈户领先。 |
| 8 | BAC009S0724W0137 | AISHELL-1 | 4.5 | 30.77% | 部分  高端  豪宅  的  交  投  持续  回暖 | 部分高端豪宅的家头持续混乱。 |
| 9 | BAC009S0724W0154 | AISHELL-1 | 3.8 | 30.00% | 由  改善  型  换房  人士  承购 | 有改善型换房人士成构。 |
| 10 | BAC009S0724W0220 | AISHELL-1 | 3.9 | 30.00% | 吴  建敏  于  去年  去年  开始 | 误见面于去年去年开始。 |
| 11 | BAC009S0724W0202 | AISHELL-1 | 4.3 | 27.27% | 通过  事先  买入  后  抛售  获利 | 通过事先迈入后抛售活力。 |
| 12 | BAC009S0724W0135 | AISHELL-1 | 6.6 | 26.32% | 位于  榜首  的  同德  围  罗  冲  围  交易  量  与  上月  持平 | 位于榜首的同德伟罗冲伟教育亮与上月持平。 |
| 13 | BAC009S0724W0164 | AISHELL-1 | 3.0 | 25.00% | 起始  楼面  价  三千  元 | 骑士楼面价三千元。 |
| 14 | BAC009S0724W0142 | AISHELL-1 | 4.0 | 23.08% | 来自  合富  置业  的  统计  数据  显示 | 来福和福置业的统计数据显示。 |
| 15 | BAC009S0724W0166 | AISHELL-1 | 3.7 | 22.22% | 起始  楼面  价  约  一万  元 | 七十楼面价约一万元。 |
| 16 | BAC009S0724W0136 | AISHELL-1 | 6.8 | 20.00% | 而  六月  居  冠  的  天河  北成  交  宗  数  下滑  近  百分  之二 | 而六月居冠的天河北城江松树下滑近百分之二。 |
| 17 | BAC009S0724W0159 | AISHELL-1 | 4.4 | 20.00% | 合富  置业  成交  数据  统计 | 和福置业成交数据统计。 |
| 18 | BAC009S0724W0157 | AISHELL-1 | 4.6 | 18.75% | 入市  积极  性  明显  提升  的  首次  置业  人士 | 入市积极性明显提升到首富职业人士。 |
| 19 | BAC009S0724W0187 | AISHELL-1 | 6.0 | 18.75% | 原  投资  经理  借  未  公开  信息  炒股  被  公诉 | 原投副经理届未公开信息炒股本公诉。 |
| 20 | BAC009S0724W0165 | AISHELL-1 | 4.7 | 18.18% | 起拍  总价  一  点  一一  五亿  元 | 奇葩总价一点一一五亿元。 |
