# AISHELL-1–5 100-Group On-Device Chinese Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 20 labeled, playable Chinese groups from each of AISHELL-1 through AISHELL-5 and run their ASR inputs through the current on-device final transcription model.

**Architecture:** Reproducible host scripts select and bundle task-aware subsets with transcripts, speaker identities and, where the source provides them, speaker turns. Pure ArkTS scoring types calculate CER independently of the UI, while a dedicated page serializes raw audio requests through the existing ASR worker and renders per-dataset truth, prediction, errors, timing, playback, and aggregate metrics.

**Tech Stack:** Python 3, Hugging Face file mirror, HarmonyOS ArkTS, ThreadWorker, sherpa-onnx Offline Paraformer, Hypium, AVPlayer, Git LFS.

---

### Task 1: Reproducible AISHELL-1–5 fixed datasets

**Files:**
- Create: `tools/prepare_aishell_app_test_suite.py`
- Create: `entry/src/main/resources/rawfile/test/asr_aishell_suite/manifest.json`
- Create: `entry/src/main/resources/rawfile/test/asr_aishell_suite/ATTRIBUTION.md`
- Create: `entry/src/main/resources/rawfile/test/asr_aishell_suite/aishell{1,2,3,4,5}/*.wav`
- Create: `entry/src/main/ets/common/AsrTestData.ets`

1. Implement deterministic 20-group sampling per dataset, preserving each dataset's native task labels.
2. Pin source revisions and write provenance, SHA-256, duration, transcript, speaker and turn metadata.
3. Download the selected WAV files and generate the typed ArkTS fixture list.
4. Verify exactly 20 groups per dataset and no empty ASR reference; verify speaker/turn labels only where the source defines them.

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
3. Add dataset tabs, truth/output comparison, edit counts, error filter and AVPlayer controls.
4. Route the ASR dataset card directly to this page and label it as 100 Chinese groups from AISHELL-1–5.

### Task 5: Verification and device run

**Files:**
- Create: `docs/evaluation/2026-09-03-asr-aishell1-5-device-results.md`

1. Run source invariants and Git LFS checks for all WAV files.
2. Run the complete ArkTS test suite.
3. Build the signed HAP.
4. Incrementally install to the connected phone without uninstalling or clearing data.
5. Launch the App, run all 50 items, capture visible summary and record the actual device results.
