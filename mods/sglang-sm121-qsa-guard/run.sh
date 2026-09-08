#!/usr/bin/env bash
# Replace SGLang's Qwen sparse-attention backend with the SM121 guard fix (sgl-project/sglang#36556).
# For the day-0 lmsysorg/sglang:qwen38flashnext image only. Do NOT apply on the 2026-09-03 or later
# images: they ship a dedicated SM121 kernel and this file would reinstate the path that corrupts
# contexts above ~95k tokens (sgl-project/sglang#36806).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py
[ -f "$TARGET" ] || { echo "mod sglang-sm121-qsa-guard: $TARGET not found in this image, skipping"; exit 0; }
cp "$TARGET" "$TARGET.orig"
cp "$HERE/qsa_upstream_fix.py" "$TARGET"
echo "mod sglang-sm121-qsa-guard: replaced $TARGET"
