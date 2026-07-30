# ASR Benchmark Report

- Manifest: `/tmp/asr_downloads_14_manifest.jsonl`
- Samples: 14
- Total audio: 2520.0s
- Micro CER: 0.00%
- Macro CER: 0.00%
- Empty hypothesis rate: 0/14
- RTF p50/p90/p95: 0.036 / 0.041 / 0.042
- sherpa-onnx: 1.13.3
- Platform: macOS-27.0-arm64-arm-64bit-Mach-O
- Decoder: streaming
- Chunk samples: 1600
- ASR precision: int8
- Punctuation enabled: True

## Models

- encoder: `81a70226a8934e6ed92aa1d4fc486b428b5398e2f2619ed4897b7294cab90e9a`
- decoder: `f3cca9f77bb9d93c8fcbfb63ae617b6b1ee96818df3aa3b151c40658fe38594f`
- tokens: `59aba8873a2ed1e122c25fee421e25f283b63290efbde85c1f01a853d83cb6e6`
- punctuation: `65a3fb9f5ad7bfb96bf69e0dc4481df97f6ee60513c1d94ce981ba6effd524b1`

## By Dataset

| Dataset | N | Duration(s) | Micro CER | Macro CER | RTF p50 |
|---|---:|---:|---:|---:|---:|
| DownloadsAudios | 14 | 2520.0 | 0.00% | 0.00% | 0.036 |

## Worst 20

| # | Utt | Dataset | Dur | CER | Ref | Hyp |
|---:|---|---|---:|---:|---|---|
| 1 | chinese_meeting_room_discussion_alimeeting_R8002_M8002_MS802_3min | DownloadsAudios | 180.0 | 0.00% |  | 嗯,现在快到年年底了,咱们那个说一下咱们年会的事情吧,嗯就是咱们年会办几天比较合适,都是一天吧。一天嗯你要有正常工作的,很避免是如果你弄完了,那现在确定年会的形 |
| 2 | chinese_meeting_room_discussion_alimeeting_R8002_M8003_MS803_3min | DownloadsAudios | 180.0 | 0.00% |  | 那什么大家早上好,今天把大鸡把,but大家的召集过来呢,是想讨论一下,我们要打算在秋天的时候开一场运动会,运动会,大家都不陌生,是不是?对嗯呃看看咱们这些人总呢 |
| 3 | chinese_meeting_room_discussion_alimeeting_R8004_M8005_MS803_3min | DownloadsAudios | 180.0 | 0.00% |  | 啊,今天咱们来聊聊各自的家庭财务规划的事情啊,然后各自都聊一聊。每家对有什么固定的财务支出啊呃固定支出的法,你没有什房贷车贷可以还的啥的。对,还是我我们家没有我 |
| 4 | chinese_meeting_room_discussion_alimeeting_R8004_M8006_MS805_3min | DownloadsAudios | 180.0 | 0.00% |  | 咱们公司打算新上一款游戏,我们需要提高他的市场竞争力,来引爆它的一款游戏吧。那我们选择地把我觉得就让脚果着呢,像那个联盟呀,什盟对吃鸡啊或者者大喜欢玩的那还是那 |
| 5 | chinese_meeting_room_discussion_alimeeting_R8005_M8007_MS806_3min | DownloadsAudios | 180.0 | 0.00% |  | 啊,行,咱时间也差不多了。咱们今天咱们公司部门,咱们几个人开个那个碰头会,然后讨论一下,就是咱咱呃咱们公司的这个办公环境问题啊,我就是咱们公司成立也这么多年了, |
| 6 | chinese_meeting_room_discussion_alimeeting_R8005_M8008_MS806_3min | DownloadsAudios | 180.0 | 0.00% |  | 今天我们来开会讨论一下如何提高实体店的营业额的问题。这一年一度的双十一马上又要来了啊呃现在线上的消费啊已经占到这个国民消费的很大一个比例。那线下的这些实体店这个 |
| 7 | chinese_meeting_room_discussion_alimeeting_R8005_M8009_MS802_3min | DownloadsAudios | 180.0 | 0.00% |  | 今天咱们开个会议大到元旦了,咱们几个人作为领导来说,咱们商量商量给员工发展礼品。但是这应该预算一下,加点福利,有可点儿福利是吧?给咱们给大家干干一年了是吧?行, |
| 8 | chinese_meeting_room_discussion_alimeeting_R8006_M8012_MS803_3min | DownloadsAudios | 180.0 | 0.00% |  | 就定三分小区还剩一直有一些问题里面收到了关于机费啊,同事啊,还有一些外国人员对咱们一些讨论。今天咱们开会讨论一下这个咱们小区环境治理的问题,从带来几个违维方面面 |
| 9 | chinese_meeting_room_discussion_alimeeting_R8008_M8014_MS807_3min | DownloadsAudios | 180.0 | 0.00% |  | 好,今天我们来首先开一下我们这个共享单车的情况及问题。然后呢,我希望我们今天都趁着星期三下午这个机会,然后是在大家都有空的时候,然后我们能一起把这个问题能讨论一 |
| 10 | chinese_meeting_room_discussion_alimeeting_R8008_M8015_MS808_3min | DownloadsAudios | 180.0 | 0.00% |  | 今天导两位叫过来,主要是我们讨论一下关疫情防控及两个汇报,一下自己的工作。好的好的,您好。嗯,好好一摇因个位置后面的,然后有一些问题需要练系,还有一些嗯方面的事 |
| 11 | chinese_meeting_room_discussion_alimeeting_R8008_M8016_MS808_3min | DownloadsAudios | 180.0 | 0.00% |  | 呃,你们好,就是今天呢我召集大家过来的那个原因呢是想因为最近我们有群众在反馈,就一直在说这个我们的我们是啊嗯这个我之前我去过了解了一下,然后呢呃就是就是剧群中反 |
| 12 | chinese_meeting_room_discussion_alimeeting_R8008_M8017_MS808_3min | DownloadsAudios | 180.0 | 0.00% |  | 嗯,肖伙伴,各位好啊,是这样的,就刚刚在一起,现在的话就是说来讨论一下这一个月下来,我们在这个店里面有遇到过一些问题。那你先开始吧。嗯,这这边的问题的话,就是现 |
| 13 | chinese_meeting_room_discussion_alimeeting_R8009_M8021_MS810_3min | DownloadsAudios | 180.0 | 0.00% |  | 王老师,咱们培训机构,我现在儿童节快到了,我们得办一个活动。对,那人都都道我们们培训机构,然后来宣传宣传,那我们场地选在我们时间选择肯定选择六月一号,对吧?对啊 |
| 14 | chinese_meeting_room_discussion_alimeeting_R8009_M8022_MS810_3min | DownloadsAudios | 180.0 | 0.00% |  | 那个王老师,我们是这周六社团要招新是吧?嗯,我知道呃,那关于这个造新,我们现在开会说一下吧,我们需校做的一些准备,还有一些注意事项啊,那个我们招新地点,你觉得哪 |
