# Fixed Audio Playback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let users play and stop every bundled fixed-test audio file directly from the speaker test UI.

**Architecture:** A page-scoped raw-resource player owns one `AVPlayer` and one rawfile descriptor at a time. `SpeakerTestCenter` observes playback state, renders per-row controls, and prevents playback and performance testing from running concurrently.

**Tech Stack:** HarmonyOS ArkTS, `@kit.MediaKit`, `@kit.LocalizationKit`, ArkUI, Hvigor/Hypium.

---

### Task 1: Add the raw-resource player

**Files:**
- Create: `entry/src/main/ets/services/audio/RawResourcePlayer.ets`

**Steps:**

1. Define playback state and callback types.
2. Open bundled audio with `ResourceManager.getRawFd()`.
3. Drive `AVPlayer` through initialized, prepared, completed, and error states.
4. Release the player before calling `closeRawFd()`.
5. Ensure switching audio stops the previous source.

### Task 2: Add per-row playback controls

**Files:**
- Modify: `entry/src/main/ets/pages/SpeakerTestCenter.ets`

**Steps:**

1. Add page playback state and bind the player callback in `aboutToAppear()`.
2. Stop and detach the player in `aboutToDisappear()`.
3. Add a play/stop button and progress to every result row.
4. Disable playback while tests run and disable the benchmark while audio plays.
5. Surface playback errors without discarding benchmark results.

### Task 3: Verify and deliver

**Files:**
- Test: `entry/src/test/*.test.ets`
- Build: `entry/build/default/outputs/default/entry-default-signed.hap`

**Steps:**

1. Run `/Applications/DevEco-Studio.app/Contents/tools/hvigor/bin/hvigorw test --no-daemon`; expect all tests to pass.
2. Run `/Applications/DevEco-Studio.app/Contents/tools/hvigor/bin/hvigorw assembleHap --no-daemon`; expect `BUILD SUCCESSFUL`.
3. Run `git diff --check`; expect no output.
4. Review the staged files and exclude `build-profile.json5`.
5. Commit and push the playback feature to `main` after verification.
