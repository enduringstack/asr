# Acoustic Scene 95% Gate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and independently verify a six-class on-device acoustic-scene classifier whose held-out accuracy and macro recall both reach the 95% release gate.

**Architecture:** Expand the source-grouped corpus into train/calibration/test splits, add a hierarchical known-scene/open-set objective, and compare DyMN04 and DyMN10 students under one protocol. Keep the phone contract at six output scores by folding the binary reject head into the exported ONNX graph.

**Tech Stack:** Python 3.11, unittest, PyTorch/MPS, EfficientAT, ONNX Runtime 1.16.3, HarmonyOS C++17/N-API, ArkTS/Hypium.

---

### Task 1: Three-way source-isolated dataset protocol

**Files:**
- Create: `tools/acoustic_scene/dataset_protocol.py`
- Create: `tools/acoustic_scene/tests/test_dataset_protocol.py`
- Modify: `tools/acoustic_scene/dataset_manifest.schema.json`

**Steps:**

1. Write failing unit tests for deterministic 70/15/15 source-group assignment, round-robin source diversity and cross-split leakage rejection.
2. Run `python -m unittest discover -s tools/acoustic_scene/tests -v`; expect failures because protocol helpers do not exist.
3. Implement `split_source_group`, `round_robin_limit`, split-count summaries and leakage validation without reading audio labels from paths at inference time.
4. Update the manifest schema to version 2 and `train|calibration|test`.
5. Rerun unit tests; expect all tests to pass.
6. Commit with `test: add acoustic scene dataset protocol`.

### Task 2: Expand official scene datasets

**Files:**
- Modify: `tools/acoustic_scene/prepare_dataset.py`
- Modify: `tools/acoustic_scene/README.md`
- Test: `tools/acoustic_scene/tests/test_dataset_protocol.py`

**Steps:**

1. Add a failing metadata-only test proving TUT 2017 development, TAU target scenes and stratified negative sub-scenes produce all three splits with no shared source group.
2. Add official TUT 2017 development record `400515`, metadata/archive indexing and location grouping.
3. Replace hard-coded tiny TAU limits with per-split, per-subscene quotas and include airport, bus, metro station, park, public square, pedestrian street, traffic street and tram as hard negatives.
4. Expand license-filtered FSD50K concert, high-speed and pure-music/pure-crowd pools while grouping by uploader or original source.
5. Materialize a new corpus outside Git and validate at least 100 test examples per product class before opening test predictions.
6. Record exact source revisions, counts, licenses and SHA-256 values.
7. Commit with `feat: expand source-isolated acoustic scene corpus`.

### Task 3: Hierarchical training and larger backbones

**Files:**
- Modify: `tools/acoustic_scene/train.py`
- Create: `tools/acoustic_scene/scene_model.py`
- Create: `tools/acoustic_scene/tests/test_scene_model.py`
- Modify: `tools/acoustic_scene/model_config.json`

**Steps:**

1. Write failing tests for DyMN width/name mapping and for combining six-class logits with a known-vs-other logit into six finite scores.
2. Implement `dymn04_as` and `dymn10_as` factories and a seven-output training wrapper.
3. Train with adjusted six-class cross entropy plus binary known/other loss; select checkpoints only by calibration macro recall, with accuracy as tie-breaker.
4. Add device/noise/gain/time/frequency augmentation and deterministic seeds.
5. Train direct DyMN04, hierarchical DyMN04 and hierarchical DyMN10 candidates on the same corpus.
6. Commit with `feat: add hierarchical acoustic scene training`.

### Task 4: Calibrated and blind evaluation

**Files:**
- Modify: `tools/acoustic_scene/evaluate.py`
- Create: `tools/acoustic_scene/calibrate.py`
- Create: `tools/acoustic_scene/tests/test_metrics.py`

**Steps:**

1. Write failing tests for explicit split selection, temperature/reject calibration and deterministic stratified bootstrap confidence intervals.
2. Make evaluation require `--split`; forbid threshold search on `test`.
3. Fit temperature and reject threshold on `calibration` only and store them in a separate JSON artifact.
4. Report accuracy, macro recall, per-class recall, confusion matrix, failure rows and 95% bootstrap intervals.
5. Freeze the selected model and run the test split once; require accuracy and macro recall at least 0.95.
6. Commit with `feat: add blind acoustic scene release gate`.

### Task 5: ONNX export and endpoint gate

**Files:**
- Modify: `tools/acoustic_scene/export_onnx.py`
- Modify: `tools/acoustic_scene/bundle_app_assets.py`
- Test: `tools/acoustic_scene/tests/test_scene_model.py`

**Steps:**

1. Add an export parity test for the hierarchical wrapper's final six scores.
2. Embed temperature and known/other fusion into a static one-window opset-17 graph.
3. Compare PyTorch and ONNX outputs on calibration fixtures with maximum absolute difference below `1e-4`.
4. Measure model bytes and Mac RTF; reject candidates whose endpoint RTF exceeds 0.25.
5. Bundle only a model that passed both 95% gates and preserve all fixed-test truth metadata.
6. Commit with `feat: export gated hierarchical scene model`.

### Task 6: HarmonyOS deployment and UI acceptance

**Files:**
- Modify: `entry/src/main/resources/rawfile/acoustic-scene-classifier/MODEL_INFO.json`
- Modify: `docs/evaluation/2026-09-01-acoustic-scene-device-results.md`
- Modify only if output contract changes: `entry/src/main/cpp/acoustic_scene/acoustic_scene_classifier.cpp`

**Steps:**

1. Run Python unit tests, `py_compile`, Git whitespace checks, host C++ strict compilation and ArkTS unit tests.
2. Rebuild the signed HAP in the main worktree while preserving `build-profile.json5`.
3. Install only with `hdc install -r`; never uninstall or clear data.
4. Run the fixed playable set, one live recording and model/UI error states on the phone.
5. Capture and inspect screenshots after all content finishes loading.
6. Record device RTF, real outputs and remaining failures, then commit and push `main`.
