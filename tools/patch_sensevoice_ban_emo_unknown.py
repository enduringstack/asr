#!/usr/bin/env python3
"""Bake SenseVoice's ban_emo_unk inference option into an ONNX graph.

FunASR implements ``ban_emo_unk=True`` by setting the EMO_UNKNOWN CTC logit
to negative infinity before argmax decoding. sherpa-onnx does not expose that
runtime option, so the HarmonyOS build uses this reproducible graph transform.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


MARKER_KEY = "sensevoice_ban_emo_unknown"
MASK_NAME = "sensevoice_emo_unknown_logit_mask"
UNMASKED_LOGITS_NAME = "logits_before_emo_unknown_mask"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def token_id(tokens_path: Path, token: str) -> int:
    for line in tokens_path.read_text(encoding="utf-8").splitlines():
        name, value = line.rsplit(" ", 1)
        if name == token:
            return int(value)
    raise RuntimeError(f"Token not found: {token}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    model = onnx.load(args.model)
    metadata = {item.key: item.value for item in model.metadata_props}
    if metadata.get(MARKER_KEY) == "1":
        raise RuntimeError("Model is already patched")
    if len(model.graph.output) != 1 or model.graph.output[0].name != "logits":
        raise RuntimeError("Expected one ONNX output named logits")

    vocab_size = int(metadata.get("vocab_size", "0"))
    unknown_id = token_id(args.tokens, "<|EMO_UNKNOWN|>")
    if vocab_size <= unknown_id:
        raise RuntimeError(
            f"EMO_UNKNOWN id {unknown_id} is outside vocab size {vocab_size}"
        )

    producer = None
    for node in model.graph.node:
        for index, output_name in enumerate(node.output):
            if output_name == "logits":
                if producer is not None:
                    raise RuntimeError("Multiple nodes produce logits")
                producer = (node, index)
    if producer is None:
        raise RuntimeError("No node produces logits")

    producer[0].output[producer[1]] = UNMASKED_LOGITS_NAME
    mask = np.zeros((vocab_size,), dtype=np.float32)
    mask[unknown_id] = -1.0e9
    model.graph.initializer.append(numpy_helper.from_array(mask, name=MASK_NAME))
    model.graph.node.append(
        helper.make_node(
            "Add",
            inputs=[UNMASKED_LOGITS_NAME, MASK_NAME],
            outputs=["logits"],
            name="/sensevoice/BanEmotionUnknown",
        )
    )

    marker = model.metadata_props.add()
    marker.key = MARKER_KEY
    marker.value = "1"
    source_hash = model.metadata_props.add()
    source_hash.key = "sensevoice_source_sha256"
    source_hash.value = sha256(args.model)
    unknown_token = model.metadata_props.add()
    unknown_token.key = "sensevoice_banned_token_id"
    unknown_token.value = str(unknown_id)

    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    reloaded = onnx.load(args.output, load_external_data=False)
    onnx.checker.check_model(reloaded)
    print(
        f"Patched {args.model} -> {args.output}; "
        f"EMO_UNKNOWN id={unknown_id}, output={args.output.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
