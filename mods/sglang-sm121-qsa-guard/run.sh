#!/usr/bin/env bash
# Upstream's SM121 guard fix (sgl-project/sglang#36556) for the day-0
# lmsysorg/sglang:qwen38flashnext build, whose sparse-attention backend selects a kernel path that
# does not exist on GB10 and crashes at the first decode.
#
# Only for a recipe pinned to that day-0 digest. Do NOT use it on the 2026-09-03 or later build:
# that one ships upstream's dedicated SM121 kernel (#36845), and replacing it restores the earlier
# path, which silently corrupts context above roughly 95k tokens (#36806).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T=/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py
[ -f "$T" ] || { echo "mod sglang-sm121-qsa-guard: $T not in this image, skipping"; exit 0; }
cp "$T" "$T.orig"; cp "$HERE/qsa_upstream_fix.py" "$T"
echo "mod sglang-sm121-qsa-guard: replaced $T"
