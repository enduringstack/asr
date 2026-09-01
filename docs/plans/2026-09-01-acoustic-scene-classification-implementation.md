# Acoustic Scene Classification Implementation Plan

**Goal:** Train, deploy, and verify a six-class on-device acoustic-scene model for recordings and labeled playable test audio.

**Architecture:** Compare AudioSet-pretrained EfficientAT MN04 and DyMN04 backbones on a reproducible scene corpus, deploy the selected DyMN04 as a power-spectrogram-input ONNX model, and run it through the ONNX Runtime already packaged with sherpa-onnx. Native C++ computes the exact spectral frontend and aggregates fixed single-window inference while ArkTS owns model lifecycle and evidence UI.

**Tech Stack:** Python 3.11, PyTorch/MPS, EfficientAT, ONNX Runtime 1.16.3, C++17/N-API, HarmonyOS ArkTS/ArkUI, Hypium.

---

### Task 1: Reproducible dataset builder

**Files:**
- Create: `tools/acoustic_scene/prepare_dataset.py`
- Create: `tools/acoustic_scene/dataset_manifest.schema.json`
- Create: `tools/acoustic_scene/README.md`

**Steps:**

1. Implement source metadata download, checksum validation, license filtering and source-grouped splits.
2. Select TAU/TUT scene labels and explicitly tagged FSD50K high-speed/concert evidence.
3. Materialize 16 kHz mono 10-second training windows outside the repository.
4. Emit a manifest containing source identity, original label, product label, split and license.
5. Validate that no source group occurs in more than one split and print class counts.

### Task 2: Fine-tuning and model export

**Files:**
- Create: `tools/acoustic_scene/train.py`
- Create: `tools/acoustic_scene/export_onnx.py`
- Create: `tools/acoustic_scene/evaluate.py`
- Create: `tools/acoustic_scene/model_config.json`

**Steps:**

1. Add deterministic waveform loading, crop/pad, gain, time-roll, noise and concert-mix augmentation.
2. Compare EfficientAT `mn04_as` and `dymn04_as`, replace the head with six outputs and fine-tune on MPS.
3. Select the checkpoint by validation macro accuracy and record per-class metrics/confusion matrix.
4. Export the fixed Mel projection and DyMN CNN as a fixed-one-window opset-17 ONNX graph.
5. Compare PyTorch/FP32 and experimental QDQ INT8 outputs; retain FP32 when INT8 fails the accuracy gate.
6. Write model card, labels, SHA-256 and evaluation report.

### Task 3: Native on-device classifier

**Files:**
- Create: `entry/src/main/cpp/acoustic_scene/acoustic_scene_classifier.h`
- Create: `entry/src/main/cpp/acoustic_scene/acoustic_scene_classifier.cpp`
- Create: `entry/src/main/cpp/third_party/onnxruntime_c_api.h`
- Create: `entry/src/main/cpp/third_party/onnxruntime_float16.h`
- Modify: `entry/src/main/cpp/CMakeLists.txt`
- Modify: `entry/src/main/cpp/napi_init.cpp`
- Modify: `entry/src/main/cpp/types/libentry/Index.d.ts`

**Steps:**

1. Add failing native frontend parity tests using exported Python fixtures.
2. Implement exact pre-emphasis, centered Hann STFT, power spectrum and sequential multi-window aggregation.
3. Load ONNX Runtime 1.16.3 through its C API and validate input/output shapes.
4. Expose initialize, classify and release N-API functions with typed error results.
5. Compare native and Python power spectra before enabling model inference.

### Task 4: ArkTS contracts and worker integration

**Files:**
- Create: `entry/src/main/ets/common/AcousticSceneTypes.ets`
- Create: `entry/src/main/ets/common/AcousticSceneTestData.ets`
- Modify: `entry/src/main/ets/workers/StreamingSherpaWorker.ets`
- Test: `entry/src/test/AcousticSceneTypes.test.ets`
- Modify: `entry/src/test/List.test.ets`

**Steps:**

1. Write failing tests for label mapping, unknown threshold, summaries, filters and confusion counts.
2. Implement pure typed helpers and the fixed test manifest.
3. Initialize the classifier once per Worker and add single-test and recording/file messages.
4. Return raw probabilities, confidence, window count, duration and elapsed time.
5. Ensure scene inference failure never changes ASR, punctuation, voiceprint or diarization output.

### Task 5: Scene UI and workbench integration

**Files:**
- Create: `entry/src/main/ets/pages/AcousticSceneTestCenter.ets`
- Modify: `entry/src/main/ets/common/VoiceWorkbenchTypes.ets`
- Modify: `entry/src/main/ets/pages/Index.ets`
- Modify: `entry/src/main/resources/base/profile/main_pages.json`
- Test: `entry/src/test/VoiceWorkbenchTypes.test.ets`

**Steps:**

1. Add the “场景感知” model and labeled-dataset entry.
2. Build summary, filter, truth/prediction, six-score and playback states.
3. Show the latest stable scene in recording and file modes.
4. Display low confidence as “暂未确定” while retaining candidate probabilities.
5. Release playback and Worker resources on page exit.

### Task 6: Bundle model and fixed evidence

**Files:**
- Generate locally (Git-ignored): `entry/src/main/resources/rawfile/acoustic-scene-classifier/model.onnx`
- Generate locally: `entry/src/main/resources/rawfile/acoustic-scene-classifier/MODEL_INFO.json`
- Generate locally (Git-ignored): `entry/src/main/resources/rawfile/test/acoustic_scene/*.wav`
- Generate: `entry/src/main/ets/common/AcousticSceneTestData.ets`

**Steps:**

1. Copy only the gated ONNX artifact and exact labels.
2. Select at least three held-out, playable clips per target scene.
3. Normalize audio to 16 kHz mono PCM16 without loudness-based label leakage.
4. Verify paths, duration, SHA-256, source truth and license for every clip.

### Task 7: Build and real-device acceptance

**Files:**
- Create: `docs/evaluation/2026-09-01-acoustic-scene-device-results.md`

**Steps:**

1. Run ArkTS unit tests and the Python reproducibility/parity suite.
2. Build the signed HAP while preserving the user's uncommitted signing profile.
3. Install with replacement only; never uninstall or clear application data.
4. Run the full fixed set, capture metrics, per-class failures, RTF, memory and screenshots.
5. Verify single-item playback and a live recording result on the connected phone.
6. Commit source, generated metadata, plans and evaluation report; keep restricted model/audio assets local and Git-ignored.
