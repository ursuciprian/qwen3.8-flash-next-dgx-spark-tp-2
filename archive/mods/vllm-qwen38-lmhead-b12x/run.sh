#!/usr/bin/env bash
# mod vllm-qwen38-lmhead-b12x
#
# Run the multi-row vocabulary projection (the MTP *verify* head) on b12x's
# small-M CuTe GEMV instead of cuBLAS's SM80 WMMA small-M fallback:
#
#   lm_head (verify, M = 1 + num_speculative_tokens = 5)
#       [124160, 2560] BF16 per node = 636 MB
#       1 launch/step, 3.60 ms/step   (measured, rank0, c1, 55.03 ms step)
#       cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8
#       176 GB/s = 65% of the 273 GB/s LPDDR5x roofline
#
# The head already takes the b12x route (use_b12x_vocab_projection is true and
# the plans are primed at boot).  b12x.gemm.bf16_vocab_projection simply has no
# kernel above M=1 -- its tuning contract admits the Triton backend only for
# max_tokens == 1, so every wider capacity resolves to backend="torch", i.e.
# torch.nn.functional.linear.  gemm.bf16_gemv's SIMT backend is the kernel for
# this shape: one CTA per output column, up to SMALL_M_MAX=8 source rows
# accumulated against a single weight load.
#
# NOT bit-identical (different fp32 summation order, ~5e-6 relative over
# K=2560).  Gate with scripts/logits_equiv.py --tol 1e-3.
# VLLM_B12X_VOCAB_GEMV=0 in the recipe env reverts without a rebuild.
# M=1, M>8, misaligned or non-contiguous hidden states, and a quantized head
# all keep the stock path and stay bit-identical to stock.
#
# Fail-closed: exact SHA256 pre-image match on both files, dry-run first,
# SHA256 post-image match, AST parse, then an import smoke test.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/usr/local/lib/python3.12/dist-packages
UTILS="$ROOT/vllm/utils/b12x.py"
LOGITS="$ROOT/vllm/model_executor/layers/logits_processor.py"
PATCHFILE="$HERE/vllm-qwen38-lmhead-b12x.patch"

PRE_UTILS=79bc7e0b5e033379717cd9df4320c4946f8f3096044964238a8e7c6fdc638bdb
PRE_LOGITS=9458a0368d7e4868d5882455a2bbfcec62f0b99979a0192e18db33b02b7173bf
POST_UTILS=eef249d208a7f22eafcd9d0a9f874afe049905bc43bc2849fe3f0a523f208022
POST_LOGITS=bb4bc0edb8b4bd4697a50c61d4ee88722aafffb4423233e6a5cf1eafc60f2c67

sha() { sha256sum "$1" | cut -d' ' -f1; }

for f in "$UTILS" "$LOGITS" "$PATCHFILE"; do
  [ -f "$f" ] || { echo "mod vllm-qwen38-lmhead-b12x: missing $f -- refusing"; exit 1; }
done

UTILS_NOW=$(sha "$UTILS")
LOGITS_NOW=$(sha "$LOGITS")

if [ "$UTILS_NOW" = "$POST_UTILS" ] && [ "$LOGITS_NOW" = "$POST_LOGITS" ]; then
  echo "mod vllm-qwen38-lmhead-b12x: already applied, skipping"
  exit 0
fi

if [ "$UTILS_NOW" != "$PRE_UTILS" ] || [ "$LOGITS_NOW" != "$PRE_LOGITS" ]; then
  echo "mod vllm-qwen38-lmhead-b12x: pre-image mismatch -- refusing to patch"
  echo "  utils/b12x.py        want $PRE_UTILS got $UTILS_NOW"
  echo "  logits_processor.py  want $PRE_LOGITS got $LOGITS_NOW"
  echo "  (the patch is cut against vllm 8e1f1e587f; rebuild it for this image)"
  exit 1
fi

cd "$ROOT"
patch -p1 --batch --forward --fuzz=0 --dry-run < "$PATCHFILE" >/dev/null
patch -p1 --batch --forward --fuzz=0 < "$PATCHFILE"

UTILS_NEW=$(sha "$UTILS")
LOGITS_NEW=$(sha "$LOGITS")
if [ "$UTILS_NEW" != "$POST_UTILS" ] || [ "$LOGITS_NEW" != "$POST_LOGITS" ]; then
  echo "mod vllm-qwen38-lmhead-b12x: post-image mismatch -- patched result is not the expected bytes"
  echo "  utils/b12x.py        want $POST_UTILS got $UTILS_NEW"
  echo "  logits_processor.py  want $POST_LOGITS got $LOGITS_NEW"
  exit 1
fi

python3 -c "
import ast
for f in ('$UTILS', '$LOGITS'):
    ast.parse(open(f).read())
print('ast ok')
"

python3 -c "
from vllm.utils.b12x import get_b12x_bf16_gemv
api = get_b12x_bf16_gemv()
assert api is not None, 'b12x.gemm.bf16_gemv did not import'
assert int(api.SMALL_M_MAX) == 8, api.SMALL_M_MAX
assert api.GemvQuery is not None and api.plan is not None and api.mm is not None
import vllm.model_executor.layers.logits_processor as lp
assert lp._b12x_vocab_gemv is not None
import inspect
src = inspect.getsource(lp.LogitsProcessor._apply_head)
assert 'gemm.bf16_gemv' in src, 'dispatch not present'
src = inspect.getsource(lp.LogitsProcessor._declare_b12x_vocab_plan)
assert 'GemvQuery' in src, 'plan declaration not present'
src = inspect.getsource(lp.LogitsProcessor._b12x_vocab_call)
assert 'produce=produce' in src, 'benchmark call has no activation producer; a b12x candidate race with B12X_AUTOTUNE=1 refuses it'
print('import smoke ok; SMALL_M_MAX =', int(api.SMALL_M_MAX),
      'enabled =', lp._VOCAB_GEMV_ENABLED)
"

echo "mod vllm-qwen38-lmhead-b12x: applied (verify vocab projection -> b12x bf16_gemv for 1 < M <= 8)"
