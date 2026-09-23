#!/usr/bin/env bash
# mod vllm-qwen38-bf16-gemv
#
# Route Qwen3.8-Flash-Next's two unquantized BF16 decode GEMMs through b12x's
# small-N CuTe GEMV (b12x.gemm.bf16_gemv) instead of the cuBLAS small-M
# fallback kernel cutlass_80_wmma_tensorop_bf16_16x16_128x2 (one warp per CTA,
# SM80 WMMA):
#
#   hyper_connection.input_mix_weight_down_block_inject  [336, 10240] BF16
#       97 launches/step, 39.09 us each, 3.79 ms/step   (measured, rank0, c1)
#   mlp.gate (MoE router)                                [512,  2560] BF16
#       48 launches/step, 17.82 us each, 0.85 ms/step
#
# b12x has shipped that GEMV all along; the only route to it is
# B12XFp6Config.get_quant_method, which a modelopt MIXED_PRECISION checkpoint
# never instantiates. See results/kernel-pass/dense-gemm-fusion.md.
#
# NOT bit-identical (different accumulation order). Gated by
# VLLM_QWEN38_BF16_GEMV -- default "hc,gate"; set "hc", "gate", "hcup" or "off"
# in the recipe env to bisect. Any shape the GEMV cannot serve (prefill, c8
# decode M=40 > SMALL_M_MAX=8, misaligned input) falls through to
# torch.nn.functional.linear inside the custom op, i.e. bit-identical to stock.
#
# Fail-closed: exact SHA256 pre-image match on both files, dry-run first,
# SHA256 post-image match, AST parse, then an import smoke test.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/usr/local/lib/python3.12/dist-packages
PKG="$ROOT/vllm/models/qwen3_8_flash_next"
PATCHFILE="$HERE/vllm-qwen38-bf16-gemv.patch"

PRE_HC=34f9ff5d07dccdd1090c4b08149be805829611f40a767d3b685ccca76db8e206
PRE_MODEL=7485ac00f1b1e60132d1c85c30251fc5f291d371ff133f0f750ee95e21a45cf2
POST_HC=61d600df3821d68fbb0168a5a0cd6fee536e6f2c47e335a397a66575f9bf05bd
POST_MODEL=66fbbf58cb0823b146e82cbb548b50848d52d7bd52fdcaf2183b4ad7f9f8fb11

sha() { sha256sum "$1" | cut -d' ' -f1; }

for f in "$PKG/hyperconnection.py" "$PKG/model.py" "$PATCHFILE"; do
  [ -f "$f" ] || { echo "mod vllm-qwen38-bf16-gemv: missing $f -- refusing"; exit 1; }
done

HC_NOW=$(sha "$PKG/hyperconnection.py")
MODEL_NOW=$(sha "$PKG/model.py")

if [ "$HC_NOW" = "$POST_HC" ] && [ "$MODEL_NOW" = "$POST_MODEL" ]; then
  echo "mod vllm-qwen38-bf16-gemv: already applied, skipping"
  exit 0
fi

if [ "$HC_NOW" != "$PRE_HC" ] || [ "$MODEL_NOW" != "$PRE_MODEL" ]; then
  echo "mod vllm-qwen38-bf16-gemv: pre-image mismatch -- refusing to patch"
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
  echo "mod vllm-qwen38-bf16-gemv: post-image mismatch -- patched result is not the expected bytes"
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
import torch
assert hasattr(torch.ops.vllm, 'qwen38_small_n_bf16_linear'), 'custom op not registered'
assert hc.maybe_route_small_n_bf16 is not None
import inspect
src = inspect.getsource(hc._SmallNBF16Provider.get_b12x_preparation_units)
assert 'produce=' in src, 'benchmark call has no activation producer; a b12x candidate race with B12X_AUTOTUNE=1 refuses it'
import vllm.models.qwen3_8_flash_next.model  # noqa: F401
print('import smoke ok; targets =', sorted(hc._SMALL_N_BF16_TARGETS))
"

echo "mod vllm-qwen38-bf16-gemv: applied (HC down/block-inject + MoE router gate -> b12x bf16_gemv)"
