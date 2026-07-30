# ASR Benchmark Report

- Manifest: `/tmp/asr_aishell100_manifest.jsonl`
- Samples: 100
- Total audio: 492.1s
- Micro CER: 6.61%
- Macro CER: 6.61%
- Empty hypothesis rate: 0/100
- RTF p50/p90/p95: 0.017 / 0.019 / 0.020
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
| AISHELL-1 | 100 | 492.1 | 6.61% | 6.61% | 0.017 |

## Worst 20

| # | Utt | Dataset | Dur | CER | Ref | Hyp |
|---:|---|---|---:|---:|---|---|
| 1 | BAC009S0724W0154 | AISHELL-1 | 3.8 | 40.00% | 由  改善  型  换房  人士  承购 | 有改善性换房人士成高。 |
| 2 | BAC009S0724W0190 | AISHELL-1 | 3.7 | 36.36% | 杨  丙  卿  将  成为  国内  因此  罪 | 杨炳清将成为国内淫付罪。 |
| 3 | BAC009S0724W0129 | AISHELL-1 | 3.5 | 33.33% | 其中  越秀  区  涨幅  领先 | 其中,越秀区丈夫领衔。 |
| 4 | BAC009S0724W0187 | AISHELL-1 | 6.0 | 31.25% | 原  投资  经理  借  未  公开  信息  炒股  被  公诉 | 语文投副经理借未公开信息考股本公诉。 |
| 5 | BAC009S0724W0136 | AISHELL-1 | 6.8 | 30.00% | 而  六月  居  冠  的  天河  北成  交  宗  数  下滑  近  百分  之二 | 而六月浇冠的天河北城江松树下滑境百分之二。 |
| 6 | BAC009S0724W0135 | AISHELL-1 | 6.6 | 26.32% | 位于  榜首  的  同德  围  罗  冲  围  交易  量  与  上月  持平 | 位于榜首的同德文卢春文交易量与上语持平。 |
| 7 | BAC009S0724W0209 | AISHELL-1 | 5.0 | 25.00% | 基金  经理  已  因此  获刑  据  了解 | 基金经理以因此互行据了解。 |
| 8 | BAC009S0724W0214 | AISHELL-1 | 6.6 | 25.00% | 原  重组  委  委员  吴  建敏  资料  图 | 原重组为未员吴建敏敷料图。 |
| 9 | BAC009S0724W0142 | AISHELL-1 | 4.0 | 23.08% | 来自  合富  置业  的  统计  数据  显示 | 来富和福置业的统计数据显示。 |
| 10 | BAC009S0724W0175 | AISHELL-1 | 6.7 | 22.22% | 该  地块  曾  于  去年  一月  挂牌  后因  故  中止  出让 | 该地款曾于去年一月挂盘,后因雇中只出让。 |
| 11 | BAC009S0724W0195 | AISHELL-1 | 5.4 | 22.22% | 杨  丙  卿  担任  中金  公司  资产  管理  部  投资  经理 | 杨炳清担任中金公司资产管理部头副经理。 |
| 12 | BAC009S0724W0200 | AISHELL-1 | 3.5 | 22.22% | 杨  丙  卿  承认  指控  事实 | 杨炳清承认指控事实。 |
| 13 | BAC009S0724W0157 | AISHELL-1 | 4.6 | 18.75% | 入市  积极  性  明显  提升  的  首次  置业  人士 | 入实积极性明显提升的首府职业人士。 |
| 14 | BAC009S0724W0176 | AISHELL-1 | 4.5 | 16.67% | 起拍  总价  较  去年  贵  了  四万  元 | 其他总价较去年贵了四万元。 |
| 15 | BAC009S0724W0212 | AISHELL-1 | 6.6 | 16.67% | 是  目前  因  老鼠  仓  被  判刑  罚  最  高  的  基金  经理 | 是目前因老虎桑被判刑法最高的基金经理。 |
| 16 | BAC009S0724W0196 | AISHELL-1 | 6.1 | 15.79% | 以  陈  笑蕊  名义  开立  并  实际  控制  个人  证券  账户 | 以陈孝蕊名义看利并实际控制个人证券账户。 |
| 17 | BAC009S0724W0137 | AISHELL-1 | 4.5 | 15.38% | 部分  高端  豪宅  的  交  投  持续  回暖 | 部分高端豪宅的交头持续混暖。 |
| 18 | BAC009S0724W0205 | AISHELL-1 | 4.9 | 15.38% | 检  方建  议  对  杨  丙  卿  处  有  期  徒刑 | 检方建议对杨北清处有期徒刑。 |
| 19 | BAC009S0724W0211 | AISHELL-1 | 4.4 | 15.38% | 韩  刚  是  业界  老鼠  仓  获刑  第  一人 | 韩钢是业界老鼠仓获型第一人。 |
| 20 | BAC009S0724W0162 | AISHELL-1 | 5.6 | 12.50% | 南沙  日前  趁  热  打铁  挂牌  出让  两  宗地  块 | 南沙日前趁热打铁瓜盆出让两宗地块。 |
