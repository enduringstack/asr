# AISHELL-1 50-Sample On-Device ASR Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a playable 50-sample Mandarin ASR ground-truth evaluation to the HarmonyOS App and run every prediction through the current on-device final transcription model.

**Architecture:** A reproducible host script selects and bundles a speaker-balanced AISHELL-1 test subset. Pure ArkTS scoring types calculate CER independently of the UI, while a dedicated page serializes raw audio requests through the existing ASR worker and renders truth, prediction, errors, timing, playback, and aggregate metrics.

**Tech Stack:** Python 3, Hugging Face file mirror, HarmonyOS ArkTS, ThreadWorker, sherpa-onnx Offline Paraformer, Hypium, AVPlayer, Git LFS.

---

### Task 1: Reproducible fixed dataset

**Files:**
- Create: `tools/prepare_aishell1_app_asr_test.py`
- Create: `entry/src/main/resources/rawfile/test/asr_aishell1_50/manifest.json`
- Create: `entry/src/main/resources/rawfile/test/asr_aishell1_50/ATTRIBUTION.md`
- Create: `entry/src/main/resources/rawfile/test/asr_aishell1_50/*.wav`
- Create: `entry/src/main/ets/common/AsrTestData.ets`

1. Implement deterministic 20-speaker / 50-unique-prompt sampling from the AISHELL-1 test split.
2. Pin the source repository revision and write provenance, SHA-256, duration, speaker, prompt and transcript metadata.
3. Download the selected WAV files and generate the typed ArkTS fixture list.
4. Verify exactly 50 files, 20 speakers, 50 unique texts and no empty transcript.

### Task 2: CER scoring with tests

**Files:**
- Create: `entry/src/main/ets/common/AsrEvaluationTypes.ets`
- Create: `entry/src/test/AsrEvaluationTypes.test.ets`
- Modify: `entry/src/test/List.test.ets`

1. Write failing tests for normalization, exact match, substitution, deletion, insertion and aggregate micro/macro CER.
2. Implement deterministic Levenshtein scoring with error counts.
3. Run `hvigorw test` and verify all new cases pass.

### Task 3: Dedicated worker protocol

**Files:**
- Modify: `entry/src/main/ets/workers/StreamingSherpaWorker.ets`

1. Add `sherpa-asr-eval-init` to load and warm the existing offline Paraformer only.
2. Add request-correlated `sherpa-asr-eval` handling for WAV bytes.
3. Return raw model text, audio duration and elapsed time; return a correlated failure without terminating the batch.

### Task 4: Evaluation UI and routing

**Files:**
- Create: `entry/src/main/ets/pages/AsrTestCenter.ets`
- Modify: `entry/src/main/resources/base/profile/main_pages.json`
- Modify: `entry/src/main/ets/common/VoiceWorkbenchTypes.ets`
- Modify: `entry/src/main/ets/pages/Index.ets`
- Modify: `entry/src/test/VoiceWorkbenchTypes.test.ets`

1. Build a serial batch runner and single-case rerun flow.
2. Add summary cards for micro-CER, macro-CER, exact rate and RTF.
3. Add truth/output comparison, edit counts, error filter and AVPlayer controls.
4. Route the ASR dataset card directly to this page and label it as 50 AISHELL-1 Chinese samples.

### Task 5: Verification and device run

**Files:**
- Create: `docs/evaluation/2026-09-03-asr-aishell1-50-device-results.md`

1. Run source invariants and Git LFS checks for all WAV files.
2. Run the complete ArkTS test suite.
3. Build the signed HAP.
4. Incrementally install to the connected phone without uninstalling or clearing data.
5. Launch the App, run all 50 items, capture visible summary and record the actual device results.
