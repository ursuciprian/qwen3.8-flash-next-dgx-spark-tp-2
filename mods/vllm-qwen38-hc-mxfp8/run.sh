#!/usr/bin/env bash
# mod vllm-qwen38-hc-mxfp8
#
# Lever B2 of results/kernel-pass/lmhead-hc-design.md §5: halve the largest
# dense weight stream in Qwen3.8-Flash-Next by quantizing the hyper-connection
# mixer projections and the MoE router gate to MXFP8 g32 once at load, and
# running them on b12x's block-scaled linear -- the same kernel that already
# serves this model's MXFP8 linear-attention projections (9.16 ms/step).
#
#   *_hyper_connection.input_mix_weight_down_block_inject  [336, 10240] BF16
#   *_hyper_connection.input_mix_weight_up                 [10240, 320] BF16
#       97 instances/step, 6.88 + 6.55 MB each = 1303 MB/node/step
#       8.22 ms/step measured (rank0, c1) = 158 GB/s of a 273 GB/s roofline
#   mlp.gate (MoE router)                                  [512,  2560] BF16
#       48 launches/step, 126 MB/step, 0.85 ms/step
#
#   MXFP8 g32 (values + swizzled UE8M0 scales, K padded to 128 by
#   blockscaled.pack_weight): 739 MB + 65 MB = 804 MB/step, 625 MB saved.
#   At today's measured 158 GB/s that is ~3.97 ms of a 55.03 ms step (7.2%).
#
# Weight loading is untouched: quant_method is rebound *after* create_weights,
# so the MergedColumnParallelLinear shard loaders, the 12 alignment pad rows
# and _HC_WEIGHTS_MAPPER see exactly the parameter they saw before. The
# in-tree Mxfp8OnlineLinearMethod quantizes it exactly once, in
# process_weights_after_loading -- never per step. Both HC tensors are
# TP-replicated (ReplicatedLinear / disable_tp=True) and stay replicated.
#
# NOT bit-identical: MXFP8 rounding is ~2^-8 relative per weight, so
# scripts/logits_equiv.py FAILS BY CONSTRUCTION. This arm is judged on
# scripts/gate_arm.sh (fidelity 8k-128k, hardmode >= 88, straggler probe).
# A *rising* MTP acceptance rate is a warning sign, not a win.
#
# Gated by VLLM_QWEN38_HC_MXFP8 -- default "hc,gate"; set "hc", "gate" or
# "off" in the recipe env to bisect without a rebuild or a remod.
# Activation precision follows b12x: VLLM_B12X_MXFP8_ACTIVATION_MODE
# (default via VLLM_B12X_DENSE_ACTIVATION_MODE = "auto") picks BF16
# activations (W8A16) for the M <= 8 decode regimes and the
# quantized-activation path at prefill capacity. Set it to "a16" to pin W8A16
# everywhere, including prefill.
#
# MUTUALLY EXCLUSIVE with mods/vllm-qwen38-bf16-gemv: that mod rebinds
# quant_method on the same three modules and patches the same two files. This
# mod refuses to run if its fingerprint is present.
#
# Fail-closed: mutual-exclusion check, exact SHA256 pre-image match on both
# files, dry-run first, SHA256 post-image match, AST parse, then an import
# smoke test that also re-evaluates b12x's own MXFP8 admission contract for
# every shape this mod routes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/usr/local/lib/python3.12/dist-packages
PKG="$ROOT/vllm/models/qwen3_8_flash_next"
PATCHFILE="$HERE/vllm-qwen38-hc-mxfp8.patch"

PRE_HC=34f9ff5d07dccdd1090c4b08149be805829611f40a767d3b685ccca76db8e206
PRE_MODEL=7485ac00f1b1e60132d1c85c30251fc5f291d371ff133f0f750ee95e21a45cf2
POST_HC=f3258dbf8e2e69d0aee88e5c08811d506cf36f7dfeab3b0edf50694147ae5cd2
POST_MODEL=1e1b0f84722c8050b083fe54198e91996ca81e7d8b8e4b5654247247ac699602

sha() { sha256sum "$1" | cut -d' ' -f1; }

for f in "$PKG/hyperconnection.py" "$PKG/model.py" "$PATCHFILE"; do
  [ -f "$f" ] || { echo "mod vllm-qwen38-hc-mxfp8: missing $f -- refusing"; exit 1; }
done

if grep -q "maybe_route_small_n_bf16" "$PKG/hyperconnection.py"; then
  echo "mod vllm-qwen38-hc-mxfp8: vllm-qwen38-bf16-gemv is applied -- refusing"
  echo "  the two mods rebind quant_method on the same HC/gate modules and"
  echo "  patch the same two files; pick one arm per boot"
  exit 1
fi

HC_NOW=$(sha "$PKG/hyperconnection.py")
MODEL_NOW=$(sha "$PKG/model.py")

if [ "$HC_NOW" = "$POST_HC" ] && [ "$MODEL_NOW" = "$POST_MODEL" ]; then
  echo "mod vllm-qwen38-hc-mxfp8: already applied, skipping"
  exit 0
fi

if [ "$HC_NOW" != "$PRE_HC" ] || [ "$MODEL_NOW" != "$PRE_MODEL" ]; then
  echo "mod vllm-qwen38-hc-mxfp8: pre-image mismatch -- refusing to patch"
  echo "  hyperconnection.py want $PRE_HC got $HC_NOW"
  echo "  model.py           want $PRE_MODEL got $MODEL_NOW"
  echo "  (the patch is cut against vllm 8e1f1e587f; rebuild it for this image)"
  exit 1
fi

cd "$ROOT"
patch -p1 --batch --forward --fuzz=0 --dry-run < "$PATCHFILE" >/dev/null
patch -p1 --batch --forward --fuzz=0 < "$PATCHFILE"

HC_NEW=$(sha "$PKG/hyperconnection.py")
MODEL_NEW=$(sha "$PKG/model.py")
if [ "$HC_NEW" != "$POST_HC" ] || [ "$MODEL_NEW" != "$POST_MODEL" ]; then
  echo "mod vllm-qwen38-hc-mxfp8: post-image mismatch -- patched result is not the expected bytes"
  echo "  hyperconnection.py want $POST_HC got $HC_NEW"
  echo "  model.py           want $POST_MODEL got $MODEL_NEW"
  exit 1
fi

python3 -c "
import ast
for f in ('$PKG/hyperconnection.py', '$PKG/model.py'):
    ast.parse(open(f).read())
print('ast ok')
"

python3 -c "
import vllm.models.qwen3_8_flash_next.hyperconnection as hc
import vllm.models.qwen3_8_flash_next.model  # noqa: F401
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

assert callable(hc.maybe_route_hc_mxfp8)
assert issubclass(hc.HcMxfp8LinearMethod, UnquantizedLinearMethod)
# create_weights must still be the unquantized one: the merged down+inject
# loader, its 12 pad rows and _HC_WEIGHTS_MAPPER depend on it.
assert hc.HcMxfp8LinearMethod.create_weights is UnquantizedLinearMethod.create_weights

# b12x's own MXFP8 admission contract, re-evaluated on CPU for every shape
# this mod routes. SM121 is asserted through a stand-in device identity.
from b12x.gemm.blockscaled import _tuning as bt

class _Dev:
    compute_capability = (12, 1)

shapes = {
    'hc   input_mix_weight_down_block_inject': (336, 10240),
    'hc   input_mix_weight_up':                (10240, 320),
    'hc   input_mix_weight_down (MTP mixer)':  (320, 10240),
    'gate mlp.gate':                           (512, 2560),
}
for label, (n, k) in shapes.items():
    padded = -(-k // 128) * 128
    for mode in ('a16', 'quantized'):
        q = bt.BlockscaledQuery(
            recipe='mxfp8', num_tokens=16, in_features=k,
            padded_in_features=padded, out_features=n, activation_mode=mode,
            source_contiguous=True, source_aligned=True,
            workspace_form='provided', workspace_nbytes=2_000_000_000,
        )
        bt._validate_query(q, None)
        bt._validate_config(q, bt._default_config(q, _Dev()), _Dev())
    auto = bt._default_config(
        bt.BlockscaledQuery(
            recipe='mxfp8', num_tokens=8, in_features=k,
            padded_in_features=padded, out_features=n, activation_mode='auto',
            source_contiguous=True, source_aligned=True,
            workspace_form='provided', workspace_nbytes=2_000_000_000,
        ),
        _Dev(),
    ).mode
    bf16 = n * k * 2
    print('  admits %-40s N=%-6d K=%-6d padK=%-6d auto@M8=%-9s %6.2f MB -> %6.2f MB'
          % (label, n, k, padded, auto, bf16 / 1e6, hc._mxfp8_bytes(n, k) / 1e6))

print('import smoke ok; targets =', sorted(hc._HC_MXFP8_TARGETS))
"

echo "mod vllm-qwen38-hc-mxfp8: applied (HC mixers + MoE router gate -> online MXFP8 g32 on b12x blockscaled)"
