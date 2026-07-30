# Speaker Diarization Embedding A/B

- Audio files: 14
- Segmentation model: `sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx`
- Metrics are stability proxies because the 14 meeting clips do not include speaker-turn ground truth.

## Summary

| Model | Threshold | Avg speakers | Avg segments | Avg short segments | Avg secondary ratio | Avg RTF |
|---|---:|---:|---:|---:|---:|---:|
| 38M-eres2net-base | 1.20 | 1.64 | 23.50 | 9.79 | 20.60% | 0.147 |
| 68M-eres2netv2 | 1.20 | 1.21 | 17.36 | 6.29 | 7.89% | 0.257 |

## Per Audio At Current Threshold 1.20

| # | Audio | 38M speakers/segments | 68M speakers/segments | 38M short | 68M short |
|---:|---|---:|---:|---:|---:|
| 1 | chinese_meeting_room_discussion_alimeeting_R8002_M8002_MS802_3min.mp3 | 3/38 | 1/9 | 22 | 2 |
| 2 | chinese_meeting_room_discussion_alimeeting_R8002_M8003_MS803_3min.mp3 | 1/30 | 1/30 | 9 | 9 |
| 3 | chinese_meeting_room_discussion_alimeeting_R8004_M8005_MS803_3min.mp3 | 2/31 | 1/12 | 11 | 1 |
| 4 | chinese_meeting_room_discussion_alimeeting_R8004_M8006_MS805_3min.mp3 | 2/35 | 2/35 | 16 | 15 |
| 5 | chinese_meeting_room_discussion_alimeeting_R8005_M8007_MS806_3min.mp3 | 2/23 | 1/10 | 11 | 5 |
| 6 | chinese_meeting_room_discussion_alimeeting_R8005_M8008_MS806_3min.mp3 | 2/21 | 2/21 | 8 | 9 |
| 7 | chinese_meeting_room_discussion_alimeeting_R8005_M8009_MS802_3min.mp3 | 3/42 | 2/37 | 24 | 19 |
| 8 | chinese_meeting_room_discussion_alimeeting_R8006_M8012_MS803_3min.mp3 | 1/19 | 1/19 | 11 | 11 |
| 9 | chinese_meeting_room_discussion_alimeeting_R8008_M8014_MS807_3min.mp3 | 1/10 | 1/10 | 2 | 2 |
| 10 | chinese_meeting_room_discussion_alimeeting_R8008_M8015_MS808_3min.mp3 | 1/8 | 1/8 | 2 | 2 |
| 11 | chinese_meeting_room_discussion_alimeeting_R8008_M8016_MS808_3min.mp3 | 1/7 | 1/7 | 1 | 1 |
| 12 | chinese_meeting_room_discussion_alimeeting_R8008_M8017_MS808_3min.mp3 | 1/14 | 1/14 | 3 | 3 |
| 13 | chinese_meeting_room_discussion_alimeeting_R8009_M8021_MS810_3min.mp3 | 1/25 | 1/25 | 8 | 8 |
| 14 | chinese_meeting_room_discussion_alimeeting_R8009_M8022_MS810_3min.mp3 | 2/26 | 1/6 | 9 | 1 |

## Readout

- Lower average speaker count/short-segment count is not automatically better; it is a proxy for less over-splitting.
- If 68M reduces short islands at similar speaker count, it is a safer upgrade.
- If both models over-split single-speaker-looking clips, threshold tuning is more important than the embedding model alone.
