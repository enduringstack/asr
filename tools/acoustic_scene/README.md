# Acoustic scene model pipeline

This directory builds the six-class acoustic-scene model used by the HarmonyOS app. It never infers labels from ASR text. The corpus combines official scene labels from TAU/TUT with explicitly licensed FSD50K evidence for high-speed trains and live audiences. Source groups are deterministically separated into 70% training, 15% calibration and 15% blind-test partitions.

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
python tools/acoustic_scene/bundle_app_assets.py \
  --fixed-manifest /path/to/acoustic-scene-data/fixed_tests/manifest.json \
  --model /path/to/acoustic-scene-run/export/model.fp32.onnx \
  --evaluation /path/to/acoustic-scene-run/evaluation.json \
  --repo /path/to/ASR
```

`prepare_dataset.py` keeps complete corpora outside Git. Do not inspect test predictions until the model and calibration parameters are frozen. The generated `fixed_tests` directory is the only audio subset intended for the App's labeled evaluation page. The bundling command writes local, Git-ignored model/audio resources and generates the ArkTS fixture metadata.

## Licensing

EfficientAT is MIT licensed. TAU/TUT recordings are marked `other-nc` by Zenodo and are restricted to local research/evaluation use. FSD50K rows are filtered to CC0 or CC BY for the high-speed and concert evidence bundled in the App. Do not use the TAU/TUT-derived weights as a commercial production model without replacing that data or obtaining separate permission.
