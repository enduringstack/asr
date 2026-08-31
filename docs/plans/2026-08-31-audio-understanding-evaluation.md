# Audio Understanding Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a fifth SenseVoice audio-understanding module with bundled labeled emotion/event audio, native model metadata, playback, batch evaluation, and recording output.

**Architecture:** Keep the model as the single source of predictions by exposing sherpa-onnx `lang`, `emotion`, and `event` through the Worker. Represent test cases and result aggregation in pure ArkTS helpers, then render a dedicated evaluation page and reuse the same metadata on the existing workbench.

**Tech Stack:** HarmonyOS ArkUI/ArkTS, sherpa-onnx 1.13.3, SenseVoice INT8, Hypium unit tests, rawfile WAV assets.

---

### Task 1: Add the labeled test protocol

**Files:**
- Create: `entry/src/main/ets/common/AudioUnderstandingTypes.ets`
- Create: `entry/src/main/ets/common/AudioUnderstandingTestData.ets`
- Test: `entry/src/test/AudioUnderstandingTypes.test.ets`
- Modify: `entry/src/test/List.test.ets`

**Steps:**

1. Write failing tests for label normalization, target-field comparison, summary totals, filters, and the 30-case data manifest.
2. Run the entry unit tests and confirm the new suite fails because the helpers do not exist.
3. Implement typed test cases, label mappings, comparison, summaries, and filters without inference-by-keyword.
4. Run the entry unit tests and confirm they pass.

### Task 2: Bundle compact, licensed audio evidence

**Files:**
- Create: `entry/src/main/resources/rawfile/test/understanding/crema_d/*.wav`
- Create: `entry/src/main/resources/rawfile/test/understanding/vocalsound/*.wav`
- Create: `entry/src/main/resources/rawfile/test/understanding/ATTRIBUTION.md`

**Steps:**

1. Download 18 CREMA-D clips whose official voice vote matches the intended emotion.
2. Download 12 VocalSound test clips for Cough, Laughter, and Sneeze.
3. Validate every file as readable WAV and record size, duration, source, label, and license.
4. Confirm the bundled subset stays small relative to the existing test corpus.

### Task 3: Expose native SenseVoice metadata

**Files:**
- Modify: `entry/src/main/ets/workers/StreamingSherpaWorker.ets`

**Steps:**

1. Add an `AudioUnderstandingResult` structure and a decoder that returns `text`, `lang`, `emotion`, and `event` directly from `OfflineRecognizerResult`.
2. Add lightweight init and inference messages for the dedicated evaluation page.
3. Attach the same metadata to final microphone responses when SenseVoice runs.
4. Preserve current ASR, punctuation, diarization, and voiceprint message contracts.

### Task 4: Build the evaluation page

**Files:**
- Create: `entry/src/main/ets/pages/AudioUnderstandingTestCenter.ets`
- Modify: `entry/src/main/resources/base/profile/main_pages.json`

**Steps:**

1. Implement ready, running, completed, failed, and playback states.
2. Render the Image 2 hierarchy: summary cards, target filters, truth/prediction evidence cards, playback, and batch action.
3. Add source and license disclosures without internal development wording.
4. Ensure page exit terminates the Worker and releases playback resources.

### Task 5: Integrate the fifth workbench model

**Files:**
- Modify: `entry/src/main/ets/common/VoiceWorkbenchTypes.ets`
- Modify: `entry/src/main/ets/pages/Index.ets`
- Test: `entry/src/test/VoiceWorkbenchTypes.test.ets`

**Steps:**

1. Extend model and dataset action types with `understanding`.
2. Add the fifth model card and the labeled 30-case suite.
3. Route dataset mode to the evaluation page.
4. Show latest language, emotion, and event in recording mode.

### Task 6: Verify and deliver

**Files:**
- Modify only source, tests, plans, and new rawfile assets from the tasks above.
- Preserve: `build-profile.json5`

**Steps:**

1. Run all entry unit tests and fix only failures caused by this feature.
2. Build the signed HAP and inspect its size delta.
3. Install with `hdc install -r` only; never uninstall or clear app data.
4. Run the full 30-case batch on the connected phone, capture metrics and screenshots, and verify playback.
5. Commit explicit feature files and push the main branch, leaving the user's `build-profile.json5` untouched.
