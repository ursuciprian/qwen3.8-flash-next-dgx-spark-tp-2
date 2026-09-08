#!/usr/bin/env bash
# Patched PLE (n-gram embedding) loader for vllm/vllm-openai:qwen38-flash-next with the RadixArk
# checkpoint: its PLE shards are global-scale FP8 and listed under `ignore`, so the stock gate never
# selects the FP8 embedding path. Pair with VLLM_PLE_FP8_CHECKPOINT=1 in env.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py
[ -f "$TARGET" ] || { echo "mod vllm-ple-fp8: $TARGET not found in this image, skipping"; exit 0; }
cp "$TARGET" "$TARGET.orig"; cp "$HERE/vllm_ple_layer.py" "$TARGET"
echo "mod vllm-ple-fp8: replaced $TARGET"
