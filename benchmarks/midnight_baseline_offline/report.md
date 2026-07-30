# ASR Benchmark Report

- Manifest: `benchmarks/asr_aishell100/aishell100_domain.jsonl`
- Samples: 100
- Total audio: 492.1s
- Micro CER (raw ASR): 5.22%
- Macro CER (raw ASR): 5.12%
- Micro CER (post-processing, final text): 5.22%
- Macro CER (post-processing, final text): 5.12%
- Empty hypothesis rate: 0/100
- RTF p50/p90/p95: 0.013 / 0.014 / 0.014
- sherpa-onnx: 1.13.4
- Platform: macOS-27.0-arm64-arm-64bit
- Decoder: offline-paraformer
- Offline model dir: `entry/src/main/resources/rawfile/sherpa-onnx-paraformer-zh-2024-03-09`
- Chunk samples: 1600
- ASR precision: int8
- Punctuation enabled: True
- Replacements: 0 entries
- Hotwords: 0 entries (none), score=1.5

## Models

- offline_model: `90bc03034ae1bef9575f8cc798cd1519c8be8aa9e8b458a033e32017ff4d584c`
- tokens: `6c0e3b35cece259829e6cb5b8d90d13db88f61ea3a2953d11898e4b2bfd7a2e2`
- punctuation: `65a3fb9f5ad7bfb96bf69e0dc4481df97f6ee60513c1d94ce981ba6effd524b1`

## By Dataset

| Dataset | N | Duration(s) | Micro CER (raw) | Micro CER (post) | RTF p50 |
|---|---:|---:|---:|---:|---:|
| aishell | 87 | 427.4 | 5.47% | 5.47% | 0.013 |
| domain | 13 | 64.7 | 3.61% | 3.61% | 0.013 |

## Worst 20 (by post-processing CER)

| # | Utt | Dataset | Dur | CER raw | CER post | Ref | Hyp (final) |
|---:|---|---|---:|---:|---:|---|---|
| 1 | BAC009S0724W0190 | aishell | 3.7 | 36.36% | 36.36% | 杨  丙  卿  将  成为  国内  因此  罪 | 杨炳清将成为国内应腐罪。 |
| 2 | BAC009S0724W0135 | aishell | 6.6 | 31.58% | 31.58% | 位于  榜首  的  同德  围  罗  冲  围  交易  量  与  上月  持平 | 位于榜首的佟德伟卢崇伟交易量与上虞持平。 |
| 3 | BAC009S0724W0154 | aishell | 3.8 | 30.00% | 30.00% | 由  改善  型  换房  人士  承购 | 有改善型换房人士重构。 |
| 4 | BAC009S0724W0169 | aishell | 3.6 | 30.00% | 30.00% | 再有  两  宗地  块  抢闸  入市 | 占有两宗地块抢砸入室。 |
| 5 | BAC009S0724W0156 | aishell | 5.9 | 26.32% | 26.32% | 仅有  百分  之  四  的  房源  由  改善  型  换房  买家  购入 | 仍有百分之四的方圆有改善性换房买家购入。 |
| 6 | BAC009S0724W0200 | aishell | 3.5 | 22.22% | 22.22% | 杨  丙  卿  承认  指控  事实 | 杨炳清承认指控事实。 |
| 7 | BAC009S0724W0136 | aishell | 6.8 | 20.00% | 20.00% | 而  六月  居  冠  的  天河  北成  交  宗  数  下滑  近  百分  之二 | 而六月居冠的天河北城江松树下滑近百分之二。 |
| 8 | BAC009S0724W0193 | aishell | 3.3 | 20.00% | 20.00% | 拥有  五年  证券  从业  经历 | 拥有武林证券从业经历。 |
| 9 | BAC009S0724W0168 | aishell | 4.7 | 18.75% | 18.75% | 新地  王  的  诞生  迅速  搅  热  南沙  土地  市场 | 新帝网的诞生,迅速绞热南沙土地市场。 |
| 10 | BAC009S0724W0153 | aishell | 5.4 | 17.65% | 17.65% | 六月  仍有  百分  之  七  单价  在  三万  元  的  房源 | 六月仍有百分之七单价在三万元的防万元。 |
| 11 | BAC009S0724W0176 | aishell | 4.5 | 16.67% | 16.67% | 起拍  总价  较  去年  贵  了  四万  元 | 其他总价较去年贵了四万元。 |
| 12 | BAC009S0724W0209 | aishell | 5.0 | 16.67% | 16.67% | 基金  经理  已  因此  获刑  据  了解 | 基金经理已因此祸形据了解。 |
| 13 | BAC009S0724W0205 | aishell | 4.9 | 15.38% | 15.38% | 检  方建  议  对  杨  丙  卿  处  有  期  徒刑 | 检方建议对杨美清处有期徒刑。 |
| 14 | BAC009S0724W0182 | domain | 6.8 | 15.00% | 15.00% | 竞买  申请  人  或  其  所  属  集团  须  是  医药  类  上市  公司 | 竞买申请人沃奇所属集团需是医药类上市公司。 |
| 15 | BAC009S0724W0157 | aishell | 4.6 | 12.50% | 12.50% | 入市  积极  性  明显  提升  的  首次  置业  人士 | 入市积极性明显提升的首付职业人士。 |
| 16 | BAC009S0724W0175 | aishell | 6.7 | 11.11% | 11.11% | 该  地块  曾  于  去年  一月  挂牌  后因  故  中止  出让 | 该地块曾于去年一月挂牌,后因雇终止出让。 |
| 17 | BAC009S0724W0195 | domain | 5.4 | 11.11% | 11.11% | 杨  丙  卿  担任  中金  公司  资产  管理  部  投资  经理 | 杨炳清担任中金公司资产管理部投资经理。 |
| 18 | BAC009S0724W0199 | aishell | 3.5 | 11.11% | 11.11% | 非法  获利  二百  馀  万元 | 非法获利二百余万元。 |
| 19 | BAC009S0724W0217 | aishell | 3.5 | 11.11% | 11.11% | 二零一一  年  一月  一  日 | 二零零一年一月一日。 |
| 20 | BAC009S0724W0196 | aishell | 6.1 | 10.53% | 10.53% | 以  陈  笑蕊  名义  开立  并  实际  控制  个人  证券  账户 | 以陈小蕊名义开例,并实际控制个人证券账户。 |
