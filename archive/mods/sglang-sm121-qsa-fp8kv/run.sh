#!/usr/bin/env bash
# Let --kv-cache-dtype fp8_e4m3 reach the SM121 QSA decode kernel on GB10.
#
# The kernel qualified for SM121 (sglang/kernels/kda_kernels/qwen38_qsa_sm121) accepts BF16 only:
# can_use_qwen38_qsa_sm121() returns False unless k.dtype == v.dtype == q.dtype == bfloat16, and
# qwen38_qsa_sm121_varlen() then raises
#   ValueError: unsupported SM121 QSA call: expected BF16 D=256, 12:1 GQA, TP1 24Q/2KV or
#   TP2 12Q/1KV, bs<=128, and selected KV<=2055
# With an fp8 KV cache the scheduler dies on the first decode, because the QSA backend allocates
# its packed gather scratch in k_buffer.dtype, so the selected keys arrive as fp8.
#
# sgl-project/sglang#38855 (merged 2026-09-10) fixed the same class of problem for the *chunk
# prefill* kernel by casting the loaded K/V tiles to the query dtype. The paged decode path in
# _forward_paged_attention is not covered by it.
#
# This does the equivalent one level up: allocate the packed scratch in the query dtype so the
# extraction kernel converts on store, and the SM121 kernel sees the BF16 it requires. Only the
# selected keys are converted (at most _MAX_SELECTED_KV=2055 rows per sequence), not the cache,
# so the KV cache keeps its fp8 footprint. Correct while the KV scales are 1.0, which is the
# default without a calibration file; with calibrated scales the extraction kernel would have to
# apply them, and this patch must not be used.
#
# Idempotent and fail-closed: both call sites must match exactly once each.
set -euo pipefail
B=/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py
[ -f "$B" ] || { echo "mod sglang-sm121-qsa-fp8kv: $B not in this image, skipping"; exit 0; }
python3 - "$B" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
MARK = "# [sm121-qsa-fp8kv]"
if MARK in src:
    print("mod sglang-sm121-qsa-fp8kv: already applied"); raise SystemExit(0)
old = """            k_buffer.shape[1],
            k_buffer.shape[2],
            k_buffer.dtype,
            k_buffer.device,
        )"""
new = """            k_buffer.shape[1],
            k_buffer.shape[2],
            q.dtype,  """ + MARK + """ SM121 QSA takes BF16 only; convert on gather
            k_buffer.device,
        )"""
n = src.count(old)
if n != 2:
    sys.exit(f"FATAL: expected 2 scratch allocation sites, found {n}; image changed, not patching")
src = src.replace(old, new)
compile(src, p, "exec")
open(p, "w").write(src)
print("mod sglang-sm121-qsa-fp8kv: patched both packed-scratch allocations to the query dtype")
PY
