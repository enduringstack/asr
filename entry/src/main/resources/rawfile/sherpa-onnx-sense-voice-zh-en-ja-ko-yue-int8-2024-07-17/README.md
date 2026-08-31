# SenseVoiceSmall INT8

This directory contains the official sherpa-onnx conversion of
`iic/SenseVoiceSmall` released as
`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17`.

Source archive:
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2

The packaged `model.int8.onnx` applies the same behavior as FunASR inference
with `ban_emo_unk=True`: the `<|EMO_UNKNOWN|>` CTC logit is masked before
argmax decoding. sherpa-onnx 1.13.3 exposes language/emotion/event results but
does not expose this inference option in its HarmonyOS API.

- Original model SHA-256:
  `c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51`
- Packaged model SHA-256:
  `359a0e2ac5968e52941d2e6a9b5a73174631f89066355ae5699cb82ba7b56a17`
- Reproducible transform: `tools/patch_sensevoice_ban_emo_unknown.py`

The 239 MB derived ONNX file is intentionally not stored in Git. The root
Hvigor plugin runs `tools/prepare_sensevoice_model.mjs` before project tasks.
When the model is absent, it downloads the official archive, verifies the
source hash, applies the deterministic patch in
`tools/sensevoice-model.patch.json`, and verifies the packaged hash before the
build continues. Run the preparation script directly to prefetch the model.
