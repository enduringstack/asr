# Acoustic scene model pipeline

This directory builds and evaluates the six-class audio-only acoustic-scene model used by the HarmonyOS app. The corpus combines official scene labels from TAU/TUT, explicitly licensed FSD50K evidence and license-audited Radio Aporee field recordings. Source groups are deterministically separated into 70% training, 15% calibration and 15% blind-test partitions. The app may separately fuse the audio probabilities with permissioned motion speed, ASR keywords and sound-event evidence; evaluation reports must keep the audio-only and fused metrics separate.

## Environment

Use Python 3.11 and install:

```bash
python -m pip install torch torchvision torchaudio librosa soundfile numpy pandas scikit-learn onnx onnxruntime remotezip requests tqdm
```

Clone the pinned EfficientAT source outside this repository:

```bash
git clone https://github.com/fschmid56/EfficientAT.git /tmp/EfficientAT
git -C /tmp/EfficientAT checkout a425fdce92572e602a1d5634799bd9f1f2efa806
```

## Prepare, train and export

```bash
python tools/acoustic_scene/prepare_dataset.py --root /path/to/acoustic-scene-data
# Rebuild entirely from complete local source groups when audio is cached:
python tools/acoustic_scene/prepare_dataset.py \
  --root /path/to/acoustic-scene-data --cache-only
python tools/acoustic_scene/train.py \
  --manifest /path/to/acoustic-scene-data/dataset_manifest.json \
  --efficientat-root /tmp/EfficientAT \
  --output /path/to/acoustic-scene-run \
  --backbone dymn04_as
python tools/acoustic_scene/export_onnx.py \
  --checkpoint /path/to/acoustic-scene-run/best.pt \
  --efficientat-root /tmp/EfficientAT \
  --manifest /path/to/acoustic-scene-data/dataset_manifest.json \
  --output /path/to/acoustic-scene-run/export \
  --skip-int8
python tools/acoustic_scene/evaluate.py \
  --manifest /path/to/acoustic-scene-data/dataset_manifest.json \
  --model /path/to/acoustic-scene-run/export/model.fp32.onnx \
  --split calibration \
  --output /path/to/acoustic-scene-run/evaluation.json
# Open the blind test only after freezing the calibration threshold:
python tools/acoustic_scene/evaluate.py \
  --manifest /path/to/acoustic-scene-data/dataset_manifest.json \
  --model /path/to/acoustic-scene-run/export/model.fp32.onnx \
  --split test --threshold 0.52 \
  --output /path/to/acoustic-scene-run/blind-test.json
# Refresh only the playable calibration Demo while the 95% gate is closed:
python tools/acoustic_scene/bundle_app_assets.py \
  --fixed-manifest /path/to/acoustic-scene-data/fixed_tests-32k/manifest.json \
  --repo /path/to/ASR
```

`prepare_dataset.py` keeps complete corpora outside Git. The v4 research branch uses the official EfficientAT 32 kHz / 1024-point FFT frontend. The currently deployed App model still uses its legacy 16 kHz / 512-point native frontend; `bundle_app_assets.py` rejects an incompatible 32 kHz export instead of copying a model that the phone cannot run. A v4 model may be bundled only after the release gate passes and the recorder/native 32 kHz scene branch has its own parity and endpoint tests. Every Aporee source contributes adjacent 0–10, 10–20 and 20–30 second windows, and source-level evaluation averages those windows before scoring. Do not inspect test predictions until the model and calibration parameters are frozen. The generated `fixed_tests-32k` directory contains three playable 30-second **calibration** sessions per class; the blind-test set is never bundled into the App. Omit `--model` from `bundle_app_assets.py` to refresh demo audio without replacing the deployed model.

## Licensing

EfficientAT is MIT licensed. TAU/TUT recordings are marked `other-nc` by Zenodo and are restricted to local research/evaluation use. FSD50K rows are filtered to CC0 or CC BY. App fixtures are Radio Aporee recordings marked CC0, Public Domain Mark or CC BY and retain attribution in the generated metadata. Do not use TAU/TUT-derived weights as a commercial production model without replacing that data or obtaining separate permission.
