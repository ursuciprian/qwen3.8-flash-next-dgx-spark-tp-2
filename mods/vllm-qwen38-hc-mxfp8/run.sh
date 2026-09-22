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
# "off" in the recipe env to bisect without a rebuild or a remod. The gate is
# DECLARED IN vllm/envs.py, not read through a bare os.getenv: it changes how
# many b12x plans a boot creates, b12x plan handles are a process-local
# counter that torch bakes into AOT-compiled graphs as integer constants, and
# only environment_variables entries reach envs.compile_factors() and
# therefore the torch_aot_compile cache key. Without that registration an
# "hc"-only boot loads an "hc,gate" artifact and dies with
#   ValueError: plan belongs to gemm.blockscaled_precision, not norm.hyperconnection
# (observed 2026-09-22, /tmp/la-hcq-hconly_boot.log). An invalid value now
# raises at boot instead of silently disabling the arm.
#
# Activation precision follows b12x: VLLM_B12X_MXFP8_ACTIVATION_MODE
# (default via VLLM_B12X_DENSE_ACTIVATION_MODE = "auto") picks BF16
# activations (W8A16) for the M <= 8 decode regimes and the
# quantized-activation path at prefill capacity. Set it to "a16" to pin W8A16
# everywhere, including prefill.
#
# MUTUALLY EXCLUSIVE with mods/vllm-qwen38-bf16-gemv: that mod rebinds
# quant_method on the same three modules and patches two of the same files.
# This mod refuses to run if its fingerprint is present.
#
# Fail-closed: mutual-exclusion check, exact SHA256 pre-image match on all
# three files, dry-run first, SHA256 post-image match, AST parse, then an
# import smoke test that re-evaluates b12x's own MXFP8 admission contract for
# every shape this mod routes AND asserts the compile-cache-key regression.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/usr/local/lib/python3.12/dist-packages
PKG="$ROOT/vllm/models/qwen3_8_flash_next"
ENVS="$ROOT/vllm/envs.py"
PATCHFILE="$HERE/vllm-qwen38-hc-mxfp8.patch"

PRE_ENVS=6b2122c817af4ee048533c76e35a0bc1bc4e785c29a08ea188a3ae53468b6516
PRE_HC=34f9ff5d07dccdd1090c4b08149be805829611f40a767d3b685ccca76db8e206
PRE_MODEL=7485ac00f1b1e60132d1c85c30251fc5f291d371ff133f0f750ee95e21a45cf2
POST_ENVS=e053d175ed9e9c5638ffcbdef2ad0d4ee8b0877df7399bd43e58fc40a7129188
POST_HC=f6c33fa738a4733f987fd11ffb1acdb5fa8ae143e03e5209293ad60a3c7f854c
POST_MODEL=1e1b0f84722c8050b083fe54198e91996ca81e7d8b8e4b5654247247ac699602

sha() { sha256sum "$1" | cut -d' ' -f1; }

for f in "$ENVS" "$PKG/hyperconnection.py" "$PKG/model.py" "$PATCHFILE"; do
  [ -f "$f" ] || { echo "mod vllm-qwen38-hc-mxfp8: missing $f -- refusing"; exit 1; }
done

if grep -q "maybe_route_small_n_bf16" "$PKG/hyperconnection.py"; then
  echo "mod vllm-qwen38-hc-mxfp8: vllm-qwen38-bf16-gemv is applied -- refusing"
  echo "  the two mods rebind quant_method on the same HC/gate modules and"
  echo "  patch the same files; pick one arm per boot"
  exit 1
fi

ENVS_NOW=$(sha "$ENVS")
HC_NOW=$(sha "$PKG/hyperconnection.py")
MODEL_NOW=$(sha "$PKG/model.py")

if [ "$ENVS_NOW" = "$POST_ENVS" ] && [ "$HC_NOW" = "$POST_HC" ] && [ "$MODEL_NOW" = "$POST_MODEL" ]; then
  echo "mod vllm-qwen38-hc-mxfp8: already applied, skipping"
  exit 0
fi

if [ "$ENVS_NOW" != "$PRE_ENVS" ] || [ "$HC_NOW" != "$PRE_HC" ] || [ "$MODEL_NOW" != "$PRE_MODEL" ]; then
  echo "mod vllm-qwen38-hc-mxfp8: pre-image mismatch -- refusing to patch"
  echo "  envs.py            want $PRE_ENVS got $ENVS_NOW"
  echo "  hyperconnection.py want $PRE_HC got $HC_NOW"
  echo "  model.py           want $PRE_MODEL got $MODEL_NOW"
  echo "  (the patch is cut against vllm 8e1f1e587f; rebuild it for this image)"
  exit 1
fi

cd "$ROOT"
patch -p1 --batch --forward --fuzz=0 --dry-run < "$PATCHFILE" >/dev/null
patch -p1 --batch --forward --fuzz=0 < "$PATCHFILE"

ENVS_NEW=$(sha "$ENVS")
HC_NEW=$(sha "$PKG/hyperconnection.py")
MODEL_NEW=$(sha "$PKG/model.py")
if [ "$ENVS_NEW" != "$POST_ENVS" ] || [ "$HC_NEW" != "$POST_HC" ] || [ "$MODEL_NEW" != "$POST_MODEL" ]; then
  echo "mod vllm-qwen38-hc-mxfp8: post-image mismatch -- patched result is not the expected bytes"
  echo "  envs.py            want $POST_ENVS got $ENVS_NEW"
  echo "  hyperconnection.py want $POST_HC got $HC_NEW"
  echo "  model.py           want $POST_MODEL got $MODEL_NEW"
  exit 1
fi

python3 -c "
import ast
for f in ('$ENVS', '$PKG/hyperconnection.py', '$PKG/model.py'):
    ast.parse(open(f).read())
print('ast ok')
"

python3 -c "
import os
import vllm.envs as envs
from vllm.config.utils import hash_factors

# --- regression check for the 2026-09-22 'hc'-only crash -------------------
# The gate must reach the AOT compile-cache key, or two target sets share one
# artifact and dereference each other's b12x plan handles.
factors = envs.compile_factors()
assert 'VLLM_QWEN38_HC_MXFP8' in factors, \
    'VLLM_QWEN38_HC_MXFP8 is missing from envs.compile_factors()'
keys = {}
for value in ('hc,gate', 'hc', 'gate', 'off'):
    os.environ['VLLM_QWEN38_HC_MXFP8'] = value
    keys[value] = hash_factors(envs.compile_factors())
os.environ.pop('VLLM_QWEN38_HC_MXFP8', None)
assert len(set(keys.values())) == 4, 'target sets collide on one compile-cache key: %r' % keys
print('compile-cache key ok; 4 distinct keys for hc,gate / hc / gate / off')

# A bad value must raise where the model reads it. compile_factors() only
# warns-and-skips a failing getter, which would silently drop the gate back
# out of the cache key -- so the model-side read is the one that must fail.
try:
    os.environ['VLLM_QWEN38_HC_MXFP8'] = 'hc,typo'
    envs.VLLM_QWEN38_HC_MXFP8
except ValueError:
    print('invalid gate value rejected at read time')
else:
    raise AssertionError('invalid VLLM_QWEN38_HC_MXFP8 was silently accepted')
finally:
    os.environ.pop('VLLM_QWEN38_HC_MXFP8', None)
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
