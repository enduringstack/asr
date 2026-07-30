# ASR Benchmark Report

- Manifest: `benchmarks/asr_aishell100/aishell100_domain.jsonl`
- Samples: 100
- Total audio: 492.1s
- Micro CER (raw ASR): 10.51%
- Macro CER (raw ASR): 10.73%
- Micro CER (post-processing, final text): 10.02%
- Macro CER (post-processing, final text): 10.19%
- Empty hypothesis rate: 0/100
- RTF p50/p90/p95: 0.032 / 0.034 / 0.035
- sherpa-onnx: 1.13.4
- Platform: macOS-27.0-arm64-arm-64bit
- Decoder: streaming
- Chunk samples: 1600
- ASR precision: int8
- Punctuation enabled: True
- Replacements: 10 entries from entry/src/main/resources/rawfile/replacements.json
- Hotwords: 0 entries (none), score=1.5

## Models

- encoder: `81a70226a8934e6ed92aa1d4fc486b428b5398e2f2619ed4897b7294cab90e9a`
- decoder: `f3cca9f77bb9d93c8fcbfb63ae617b6b1ee96818df3aa3b151c40658fe38594f`
- tokens: `59aba8873a2ed1e122c25fee421e25f283b63290efbde85c1f01a853d83cb6e6`
- punctuation: `65a3fb9f5ad7bfb96bf69e0dc4481df97f6ee60513c1d94ce981ba6effd524b1`

## By Dataset

| Dataset | N | Duration(s) | Micro CER (raw) | Micro CER (post) | RTF p50 |
|---|---:|---:|---:|---:|---:|
| aishell | 87 | 427.4 | 10.94% | 10.38% | 0.032 |
| domain | 13 | 64.7 | 7.73% | 7.73% | 0.032 |

## Worst 20 (by post-processing CER)

| # | Utt | Dataset | Dur | CER raw | CER post | Ref | Hyp (final) |
|---:|---|---|---:|---:|---:|---|---|
| 1 | BAC009S0724W0200 | aishell | 3.5 | 55.56% | 55.56% | 杨  丙  卿  承认  指控  事实 | 以丙丙用承认指控控是。 |
| 2 | BAC009S0724W0136 | aishell | 6.8 | 50.00% | 50.00% | 而  六月  居  冠  的  天河  北成  交  宗  数  下滑  近  百分  之二 | 而六月击贯的听河北城江中树下华进百分之。 |
| 3 | BAC009S0724W0214 | aishell | 6.6 | 50.00% | 50.00% | 原  重组  委  委员  吴  建敏  资料  图 | 语重组为违约无建敏敷料图。 |
| 4 | BAC009S0724W0180 | aishell | 4.2 | 45.45% | 45.45% | 折合  最  低  楼面  价  约  一百  元 | 之后最低楼面价,因为一百。 |
| 5 | BAC009S0724W0169 | aishell | 3.6 | 40.00% | 40.00% | 再有  两  宗地  块  抢闸  入市 | 占有两宗地块敲诈入室。 |
| 6 | BAC009S0724W0135 | aishell | 6.6 | 36.84% | 36.84% | 位于  榜首  的  同德  围  罗  冲  围  交易  量  与  上月  持平 | 位于榜首的同德伟罗冲伟,焦育亮与尚宇持平。 |
| 7 | BAC009S0724W0166 | aishell | 3.7 | 33.33% | 33.33% | 起始  楼面  价  约  一万  元 | 起始楼面价,因为一万。 |
| 8 | BAC009S0724W0176 | aishell | 4.5 | 33.33% | 33.33% | 起拍  总价  较  去年  贵  了  四万  元 | 奇葩总价叫较去年贵了四万。 |
| 9 | BAC009S0724W0154 | aishell | 3.8 | 40.00% | 30.00% | 由  改善  型  换房  人士  承购 | 有改善型换房人士乘高。 |
| 10 | BAC009S0724W0153 | aishell | 5.4 | 29.41% | 29.41% | 六月  仍有  百分  之  七  单价  在  三万  元  的  房源 | 六月仍有百分之七加加在三万元等房问员。 |
| 11 | BAC009S0724W0167 | aishell | 6.4 | 29.41% | 29.41% | 广东  自贸区  南沙  片区  首度  开拍  住宅  地块 | 广东自贸区南上聘区域首开开住宅地块。 |
| 12 | BAC009S0724W0196 | aishell | 6.1 | 26.32% | 26.32% | 以  陈  笑蕊  名义  开立  并  实际  控制  个人  证券  账户 | 以沉孝蕊名义看例,并实际控制个人证券账。 |
| 13 | BAC009S0724W0157 | aishell | 4.6 | 25.00% | 25.00% | 入市  积极  性  明显  提升  的  首次  置业  人士 | 入实积极性明显提升的首富职业人。 |
| 14 | BAC009S0724W0187 | aishell | 6.0 | 25.00% | 25.00% | 原  投资  经理  借  未  公开  信息  炒股  被  公诉 | 源头副经理借未公开信息炒股本公诉。 |
| 15 | BAC009S0724W0188 | aishell | 3.3 | 22.22% | 22.22% | 二中  院  开庭  审理  此案 | 二、中院开庭庭审理此。 |
| 16 | BAC009S0724W0217 | aishell | 3.5 | 22.22% | 22.22% | 二零一一  年  一月  一  日 | 二零零一一年一月一。 |
| 17 | BAC009S0724W0192 | aishell | 3.3 | 20.00% | 20.00% | 获刑  的  第  四  名基  金  经理 | 或型的第四名基金经理。 |
| 18 | BAC009S0724W0127 | aishell | 5.4 | 18.75% | 18.75% | 但  受  穗  六条  及  二  套房  首  付  七成  的  制约 | 但受贿六条第二套方首付七成的制约。 |
| 19 | BAC009S0724W0130 | aishell | 7.4 | 17.39% | 17.39% | 天河  区  的  签约  面积  在  豪宅  交  投  增多  的  带动  下  上升  较快 | 天河区的签约面积在豪宅交投增多的大海栋小上升较快。 |
| 20 | BAC009S0724W0181 | domain | 5.4 | 16.67% | 16.67% | 竞买  申请  人  须  在  南沙  区  注册  成立  项目  公司 | 竞买申请人需在南沙区注册成立项目目公。 |
