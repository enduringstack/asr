# Acoustic scene model pipeline

This directory builds the six-class acoustic-scene model used by the HarmonyOS app. It never infers labels from ASR text. The default corpus combines official scene labels from TAU/TUT with explicitly licensed FSD50K evidence for high-speed trains and live audiences.

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
  --output /path/to/acoustic-scene-run/evaluation.json
python tools/acoustic_scene/bundle_app_assets.py \
  --fixed-manifest /path/to/acoustic-scene-data/fixed_tests/manifest.json \
  --model /path/to/acoustic-scene-run/export/model.fp32.onnx \
  --evaluation /path/to/acoustic-scene-run/evaluation.json \
  --repo /path/to/ASR
```

`prepare_dataset.py` keeps complete corpora outside Git. The generated `fixed_tests` directory is the only audio subset intended for the App's labeled evaluation page. The bundling command writes local, Git-ignored model/audio resources and generates the ArkTS fixture metadata.

## Licensing

EfficientAT is MIT licensed. TAU/TUT recordings are marked `other-nc` by Zenodo and are restricted to local research/evaluation use. FSD50K rows are filtered to CC0 or CC BY for the high-speed and concert evidence bundled in the App. Do not use the TAU/TUT-derived weights as a commercial production model without replacing that data or obtaining separate permission.
