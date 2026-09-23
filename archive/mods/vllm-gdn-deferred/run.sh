#!/usr/bin/env bash
# vllm-gdn-deferred: drive b12x deferred GDN checkpoints from vLLM, behind
# VLLM_GDN_DEFERRED_CHECKPOINTS (default off).
#
# Why. The b12x Qwen GDN decode kernel is DRAM-bandwidth saturated at c8-c16,
# measured at 86% of the GB10's 273 GB/s. Five of the six recurrent-state
# snapshots it moves per layer-step exist only because acceptance is unknown
# until after the sampler, and four of them are thrown away. With deferred
# checkpoints the kernel keeps one base checkpoint in the running block plus a
# compact per-token record in each speculative block, and replays the accepted
# prefix as part of the state read it already performs: 6 snapshots become 2,
# 2.66x less state traffic, modelled at roughly 7% of the c8 step.
#
# The cost, and what this mod is actually for: the running block stops being
# the committed state and the speculative blocks stop being checkpoints, so
# every reader outside the decode kernel must ask b12x to materialize the
# accepted prefix first. Under align cache mode there are exactly two, both
# block-boundary state copies (postprocess_mamba_fused_kernel and
# precopy_mamba_align_fused_kernel). Each now runs its copy kernel twice: once
# with DECISION_ONLY to emit which requests cross a boundary and with what
# acceptance, then the per-layer commit, then the real copy with a zero
# temporal bias. The shared copy helper therefore takes the conv and temporal
# biases separately -- the conv half still shifts its window by the
# accepted-token bias while the temporal half copies an already-committed
# block. A third reader, request-boundary checkpointing, is refused rather
# than handled: one capture can ask for up to three distinct biases for the
# same request and a single accepted-prefix commit cannot express that.
#
# REQUIRES A MATCHING b12x. This mod alone does nothing useful: it needs the
# b12x branch feat/gdn-deferred-checkpoints, which adds
# Caps(deferred_checkpoints=...) and commit_deferred_checkpoints. No published
# wheel carries it yet, so on the current image this mod applies cleanly and
# the feature then refuses to turn on (the caps keyword is rejected by the
# installed b12x). Do not set VLLM_GDN_DEFERRED_CHECKPOINTS=1 until a wheel
# with that b12x branch exists; leaving it unset is a no-op and safe.
#
# NOT bit-identical to the shipped path by construction -- it is bit-identical
# to the *checkpoint b12x would have written*, which the b12x pytest proves on
# GPU. This arm is judged on scripts/gate_arm.sh.
#
# Fail closed: pre-image sha256 per file, exact-block match on every edit,
# nothing written unless all of them match, every touched file ast-parsed,
# post-image sha256 verified, then an import + behaviour self-check. Idempotent
# on the marker and on the post-image sha.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/usr/local/lib/python3.12/dist-packages
V="$ROOT/vllm"

FILES=(
  envs.py
  v1/worker/mamba_utils.py
  model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
)

# sha256 of each file as built from vllm.git 8e1f1e587f (this image's base).
declare -A PRE=(
  [envs.py]=6b2122c817af4ee048533c76e35a0bc1bc4e785c29a08ea188a3ae53468b6516
  [v1/worker/mamba_utils.py]=fa7b4662fbc08e3865d88ee7350d72dd7f729921e70070ebaafd29ac374eb86b
  [model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py]=e6c5b7f1dea3c16c8ad41e3c59c00bd0e922c878c32aeb6e8efd5288f166d67c
)
# ... and after this mod has applied.
declare -A POST=(
  [envs.py]=d2dc58b3e26824b39ac2e70a546e453522be78d9302df63818ce7b736e56e657
  [v1/worker/mamba_utils.py]=8f7d782e67f0c89dbbf9c8a77ab5d4264bb391b80c56cc325a5dc60117ef56e5
  [model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py]=581f7c88283d177979f897c12e74c18ee8cc749eba9ae9b930df5ce858c4f06b
)
NEW_MODULE=v1/worker/gdn_deferred_commit.py
NEW_MODULE_SHA=a742030c2253b7c8eb862c391b1eb5a459a52385aced48bfc805704a25e1fd95

for f in "${FILES[@]}"; do
  [ -f "$V/$f" ] || { echo "mod vllm-gdn-deferred: $V/$f not in this image, skipping"; exit 0; }
done

sha() { sha256sum "$1" | cut -d' ' -f1; }

applied=1
for f in "${FILES[@]}"; do
  [ "$(sha "$V/$f")" = "${POST[$f]}" ] || applied=0
done
if [ "$applied" = 1 ] && [ -f "$V/$NEW_MODULE" ]; then
  echo "mod vllm-gdn-deferred: already applied, skipping"
  exit 0
fi

if /bin/grep -q "VLLM_GDN_DEFERRED_CHECKPOINTS" "$V/envs.py"; then
  echo "mod vllm-gdn-deferred: marker already present but the post-image sha" \
       "does not match -- refusing to patch a half-applied tree" >&2
  exit 1
fi

for f in "${FILES[@]}"; do
  have=$(sha "$V/$f")
  if [ "$have" != "${PRE[$f]}" ]; then
    echo "mod vllm-gdn-deferred: pre-image mismatch -- refusing to patch" >&2
    echo "  $f want ${PRE[$f]} got $have" >&2
    echo "  (the blocks are cut against vllm 8e1f1e587f; regenerate them for this image)" >&2
    exit 1
  fi
done

install -m 0644 "$HERE/gdn_deferred_commit.py" "$V/$NEW_MODULE"
have=$(sha "$V/$NEW_MODULE")
[ "$have" = "$NEW_MODULE_SHA" ] || {
  echo "mod vllm-gdn-deferred: copied module sha $have != $NEW_MODULE_SHA" >&2
  exit 1
}

python3 "$HERE/apply.py" "$ROOT"

for f in "${FILES[@]}"; do
  have=$(sha "$V/$f")
  [ "$have" = "${POST[$f]}" ] || {
    echo "mod vllm-gdn-deferred: post-image mismatch for $f: $have != ${POST[$f]}" >&2
    exit 1
  }
done

python3 - <<'PY'
import ast

# The patched files parse, and the feature is genuinely inert while unset.
for path in (
    "/usr/local/lib/python3.12/dist-packages/vllm/envs.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gdn_deferred_commit.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/"
    "gdn/qwen_gdn_linear_attn.py",
):
    ast.parse(open(path).read())

import os

os.environ.pop("VLLM_GDN_DEFERRED_CHECKPOINTS", None)
from vllm import envs
from vllm.v1.worker import gdn_deferred_commit as deferred

assert envs.VLLM_GDN_DEFERRED_CHECKPOINTS is False, "default must be off"
assert deferred.requested() is False


class _Cache:
    mamba_cache_mode = "none"


class _Config:
    cache_config = _Cache()
    use_request_boundary_checkpoints = True
    speculative_config = None


# Fail closed: every unsupported dimension has to be named, not silently
# downgraded to the shipped checkpoint path.
reasons = deferred.refuse_reasons(_Config())
assert len(reasons) == 3, reasons
assert any("align" in r for r in reasons)
assert any("request-boundary" in r for r in reasons)
assert any("speculative" in r for r in reasons)
assert deferred.resolve(_Config()) is False, "off must stay off, not raise"

os.environ["VLLM_GDN_DEFERRED_CHECKPOINTS"] = "1"
import importlib

importlib.reload(envs)
assert envs.VLLM_GDN_DEFERRED_CHECKPOINTS is True
try:
    deferred.resolve(_Config())
except ValueError as exc:
    assert "request-boundary" in str(exc), exc
else:
    raise AssertionError("asking for an unsupported configuration must raise")
os.environ.pop("VLLM_GDN_DEFERRED_CHECKPOINTS", None)
importlib.reload(envs)

# The copy helper really does take the two biases apart.
from vllm.v1.worker import mamba_utils

names = mamba_utils._copy_mamba_state_block.fn.__code__.co_varnames
assert "conv_bias" in names and "temporal_bias" in names, names
assert "token_bias" not in names[:8], names
PY

echo "mod vllm-gdn-deferred: applied (VLLM_GDN_DEFERRED_CHECKPOINTS=1 routes" \
     "block-boundary GDN state copies through a b12x accepted-prefix commit;" \
     "unset is a no-op, and the feature needs a b12x wheel built from" \
     "feat/gdn-deferred-checkpoints)"
