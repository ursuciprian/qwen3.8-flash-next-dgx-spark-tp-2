#!/usr/bin/env bash
# Patched PLE (n-gram embedding) loader for vllm/vllm-openai:qwen38-flash-next with the RadixArk
# checkpoint. Its PLE shards are global-scale FP8 and the checkpoint lists *.ple.* under `ignore`,
# so the stock gate never selects the FP8 embedding path and weight loading fails on
# ngram_embedding.weight_scale. Pair with VLLM_PLE_FP8_CHECKPOINT=1 in env.
# Kept for the Spark Arena entry that predates the current recipes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py
[ -f "$T" ] || { echo "mod vllm-ple-fp8: $T not in this image, skipping"; exit 0; }
cp "$T" "$T.orig"; cp "$HERE/vllm_ple_layer.py" "$T"
echo "mod vllm-ple-fp8: replaced $T"
