# Voice Intelligence Workbench Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the HarmonyOS main page with a production-grade speech workbench that exposes real ASR, punctuation, voiceprint, and diarization outputs for recording and labeled-dataset workflows.

**Architecture:** Keep the existing audio service, worker, profile stores, and evaluation pages as the only inference paths. Add a small typed presentation model, expose raw/final text and telemetry already produced inside the worker, and rebuild `Index.ets` around two modes plus a selected-model inspector.

**Tech Stack:** HarmonyOS ArkUI/ArkTS, ThreadWorker, sherpa-onnx, Hypium, HUKS-backed profile stores.

---

### Task 1: Add the typed workbench presentation contract

**Files:**
- Create: `entry/src/main/ets/common/VoiceWorkbenchTypes.ets`
- Create: `entry/src/test/VoiceWorkbenchTypes.test.ets`
- Modify: `entry/src/test/List.test.ets`

**Steps:**
1. Write tests asserting exactly four ordered model descriptors and the recording/dataset suite mapping.
2. Run `hvigorw test --no-daemon`; expect the new imports to fail before implementation.
3. Add `WorkbenchMode`, `SpeechModelId`, `SpeechModelDescriptor`, and dataset suite descriptors.
4. Add pure formatting helpers for model state, telemetry and speaker counts.
5. Run tests; expect all tests to pass.

### Task 2: Expose real per-model intermediate output

**Files:**
- Modify: `entry/src/main/ets/workers/StreamingSherpaWorker.ets`
- Modify: `entry/src/main/ets/pages/Index.ets`
- Test: `entry/src/test/TranscriptBuilder.test.ets`

**Steps:**
1. Extend worker responses with optional `rawText` while preserving `text` as final output.
2. Include raw ASR text and punctuated text in microphone, WAV and M4A completion responses without running a second fake pipeline.
3. Store the latest raw text, final text and telemetry in page state.
4. Preserve all existing message names and callers.
5. Run unit tests and an ArkTS compile.

### Task 3: Replace the main recording UI

**Files:**
- Modify: `entry/src/main/ets/pages/Index.ets`

**Steps:**
1. Add workbench mode and selected-model state.
2. Build the header, readiness status and two-mode segmented control.
3. Build a recording controller using the existing press/stop methods, duration and level state.
4. Build four accessible model cards in processing order.
5. Build the selected-model effect inspector using actual raw text, final text, voiceprint candidates, turns and telemetry.
6. Build the full result section and retain speaker editing, clearing, continuous recording, profiles and enrollment controls.
7. Check empty, loading, recording, processing and error states.

### Task 4: Integrate dataset evaluation mode

**Files:**
- Modify: `entry/src/main/ets/pages/Index.ets`
- Modify: `entry/src/main/ets/pages/SpeakerTestCenter.ets` only if navigation copy needs alignment.

**Steps:**
1. Add four dataset suite cards mapped to the same model descriptors.
2. Route ASR and punctuation to packaged/imported audio using the existing worker path.
3. Route voiceprint to `SpeakerIdentityTestCenter` and diarization to `SpeakerTestCenter`.
4. Keep original-audio playback in the existing test result pages.
5. Never present live-recording output as labeled accuracy.

### Task 5: Verify and hand off

**Files:**
- Verify: `entry/src/main/ets/pages/Index.ets`
- Verify: `entry/src/main/ets/workers/StreamingSherpaWorker.ets`

**Steps:**
1. Run `git diff --check`.
2. Run `hvigorw test --no-daemon`; expect zero failures.
3. Run `hvigorw assembleHap --no-daemon`; expect a signed HAP.
4. Do not install unless installation is explicitly requested in the current task.
5. Commit only feature files, preserving the existing uncommitted `build-profile.json5` change.
