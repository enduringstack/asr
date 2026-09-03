# ASR Evaluation and On-Device Pipeline Optimization Plan

**Goal:** Separate plain Mandarin ASR quality from multi-speaker pipeline quality, then optimize the on-device recognizer without hiding diarization or overlap errors inside a single misleading CER.

**Architecture:** Keep the current 100 playable AISHELL fixtures. Add protocol-aware scoring and oracle-turn decoding first, then add predicted-diarization scoring, and only after those baselines are trustworthy run model candidates through the same device protocol. Ground-truth text is used only by the UI scorer and is never sent to the worker/model.

**Constraints:** HarmonyOS ArkTS, offline-only inference, current bundled Paraformer remains the baseline, no application uninstall/data clearing, and no device installation unless explicitly requested in the active task.

## Batch 1: Correct ASR-only evaluation

### Task 1: Protocol-aware scoring core

- Output: overlap detection, turn-level summaries, and speaker-permutation CER primitives in `AsrEvaluationTypes.ets`.
- Test: Hypium covers non-overlap/overlap classification, independent turn aggregation, and minimum-permutation speaker scoring.

### Task 2: Oracle-turn worker protocol

- Output: the ASR evaluation worker accepts timestamp-only segments and returns one raw model result and timing value per segment.
- Test: source contract checks prove reference text is never included in worker requests; ArkTS compilation validates transferable message types.

### Task 3: Protocol selector and evidence UI

- Output: the AISHELL page can run `整段基线` or the secondary `ASR 上限诊断`; oracle timestamps are never presented as the real meeting result.
- Test: Hypium plus signed HAP build. No phone installation in this batch.

## Batch 2: End-to-end multi-speaker evaluation

### Task 4: Predicted diarization evaluation

- Output: send one complete audio file to Pyannote segmentation 3.0 INT8 plus CAM++ clustering, transcribe only model-predicted turns, merge by predicted cluster, restore punctuation, and attempt registered-voiceprint matching.
- Test: calculate no-collar overlap-inclusive DER/cpCER against labels that remain outside the worker; score raw overlap detection separately from the final single-stream speaker turns.

### Task 5: Multi-speaker UI protocol

- Output: make `完整多人链路` the default for AISHELL-4/5, with truth and predicted timelines, cpCER, DER, overlap detection, merged speaker streams, punctuation output, and voiceprint decisions. Explicitly disclose that detected overlap has not yet been separated into independent waveforms.
- Test: deterministic metric tests and HAP build.

## Batch 3: Model and package optimization

### Task 6: Reproducible model A/B harness

- Output: compare current offline Paraformer with selected Zipformer INT8 candidates using identical audio, protocol, normalization and phone conditions.
- Test: record CER/cpCER, RTF, load time, peak memory and packaged size; reject any candidate that regresses clean speech beyond the agreed gate.

### Task 7: Production model decision

- Output: use the winning final recognizer and, if its measured accuracy is acceptable, replace the large live draft recognizer with the small streaming candidate.
- Test: full host tests, signed build, then device installation only after explicit approval.
