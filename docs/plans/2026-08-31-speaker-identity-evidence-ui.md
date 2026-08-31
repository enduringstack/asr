# Speaker Identity Evidence UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign the AISHELL-3 speaker identity page so every test visibly compares dataset truth with the real CAM++ prediction and decision evidence.

**Architecture:** Keep the existing isolated evaluation pipeline and worker messages unchanged. Add pure presentation helpers for result categories and decision copy, then rebuild `SpeakerIdentityTestCenter.ets` around a summary, filters, and evidence cards using the fields already produced by `SpeakerIdentityEvaluation`.

**Tech Stack:** HarmonyOS ArkUI/ArkTS, sherpa-onnx CAM++, Hypium, RawResourcePlayer.

---

### Task 1: Add testable comparison presentation helpers

**Files:**
- Create: `entry/src/main/ets/common/SpeakerIdentityPresentation.ets`
- Create: `entry/src/test/SpeakerIdentityPresentation.test.ets`
- Modify: `entry/src/test/List.test.ets`

**Steps:**
1. Write failing tests for waiting, correct known, false reject, misidentification, correct unknown rejection and false acceptance labels.
2. Run the entry unit tests and confirm the missing presentation module fails compilation.
3. Implement typed result categories and threshold-decision copy without performing identity inference in the UI layer.
4. Run the tests and confirm all cases pass.

### Task 2: Preserve complete real prediction evidence

**Files:**
- Modify: `entry/src/main/ets/pages/SpeakerIdentityTestCenter.ets`

**Steps:**
1. Extend each UI row with accepted state, second-best score and typed comparison category.
2. Populate those fields only from `scoreSpeakerIdentityCase`.
3. Preserve error and waiting states without invented predictions or scores.

### Task 3: Rebuild the mobile evaluation page

**Files:**
- Modify: `entry/src/main/ets/pages/SpeakerIdentityTestCenter.ets`

**Steps:**
1. Replace the header with a clear title, model badge and progress state.
2. Add a compact dataset protocol card and three overall metric cards.
3. Add result filters for all, known, unknown and errors.
4. Replace each row with a truth-versus-prediction evidence card showing speaker ID, actual label, model output, score, threshold, elapsed time and explicit outcome.
5. Preserve three-query-audio playback and isolated-test privacy copy.
6. Cover loading, waiting, running, correct, incorrect and failure states.

### Task 4: Verify and deploy

**Files:**
- Verify: `entry/src/main/ets/pages/SpeakerIdentityTestCenter.ets`

**Steps:**
1. Run `git diff --check`.
2. Run all entry unit tests and expect zero failures.
3. Build the signed HAP and expect a successful ArkTS compile.
4. Install with `hdc install -r` so existing application data is preserved.
5. Open the AISHELL-3 page on the connected phone and verify truth labels, prediction empty states, playback, filters and model-ready state from the UI hierarchy and screenshots.
6. Commit only feature files and push `main`, preserving the existing local `build-profile.json5` change.
