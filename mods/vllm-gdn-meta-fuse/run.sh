#!/usr/bin/env bash
# mod vllm-gdn-meta-fuse
#
# Cherry-pick of local-inference-lab/vllm a18246b06 ("Fuse GDN metadata copies
# and state refresh across cache groups", integration/karmic-kraken-beta,
# 2026-09-22) onto our serving fork base 8e1f1e587f.
#
# What it does: when a second GDN cache group reuses a captured sibling's
# buffers, `GDNAttentionMetadataBuilder` used to run
#   mixed.copy_worklists_from(src)      -> 3 torch._foreach_copy_ (14 tensors)
#   mixed.refresh_state_indices(...)    -> 3 zero_() + 3-4 gather/copy_ + where
# i.e. ~10 device launches plus a host-side torch.where, every step, per extra
# group. The commit adds one Triton kernel, `_copy_worklists_and_refresh_states`,
# that does the worklist copies and the recurrent-state / checkpoint index
# refresh in a single launch, reading worklists from the *source* group so no
# cross-CTA ordering is needed, and keeps the destination pointers stable so
# CUDA-graph capture still holds.
#
# Additive and self-guarding: `copy_and_refresh_from` falls back to the old
# two-call path when `source is self` or the buffers are not CUDA, and both old
# methods stay in place. Only one call site changes (gdn_attn.py:904).
#
# KK's own measurement on this exact arch (TP2 Qwen serving, warmed prompts):
# mean verifier-step latency 41.19 -> 40.16 ms (-2.5%).
#
# Not shipped from the upstream commit: its tests/ and benchmarks/ files --
# neither directory is installed in the image, so the patch is the two vllm/
# files only. Upstream test names kept in README.md for the GPU pass.
#
# Fail-closed: exact SHA256 pre-image on both files, dry-run, SHA256
# post-image, AST parse, import smoke, and a CPU equivalence check of the
# fallback path against the old two-call sequence. Idempotent: re-running on an
# already-patched tree exits 0.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/usr/local/lib/python3.12/dist-packages
PKG="$ROOT/vllm/v1/attention/backends"
PATCHFILE="$HERE/vllm-gdn-meta-fuse.patch"

PRE_META=82811f63bf58f70bd8afd870f8decef6300331c863c664a468ad54879703b900
PRE_ATTN=dd3ccfaffce20954ab327bdd9278a660eeee2093c7b8f197429fe4dc489d60c4
POST_META=41dab35907b83562791ef76538319dd8038ee265cc489278c806f96a63203730
POST_ATTN=8efe9a7e6fcb7bf786ee78f9662034e292dd1b6bdde40b05ff237152ed668cc2

sha() { sha256sum "$1" | cut -d' ' -f1; }

for f in "$PKG/b12x_gdn_metadata.py" "$PKG/gdn_attn.py" "$PATCHFILE"; do
  [ -f "$f" ] || { echo "mod vllm-gdn-meta-fuse: missing $f -- refusing"; exit 1; }
done

META_NOW=$(sha "$PKG/b12x_gdn_metadata.py")
ATTN_NOW=$(sha "$PKG/gdn_attn.py")

if [ "$META_NOW" = "$POST_META" ] && [ "$ATTN_NOW" = "$POST_ATTN" ]; then
  echo "mod vllm-gdn-meta-fuse: already applied, skipping"
  exit 0
fi

if [ "$META_NOW" != "$PRE_META" ] || [ "$ATTN_NOW" != "$PRE_ATTN" ]; then
  echo "mod vllm-gdn-meta-fuse: pre-image mismatch -- refusing to patch"
  echo "  b12x_gdn_metadata.py want $PRE_META got $META_NOW"
  echo "  gdn_attn.py          want $PRE_ATTN got $ATTN_NOW"
  echo "  (the patch is cut against vllm 8e1f1e587f; rebuild it for this image)"
  exit 1
fi

cd "$ROOT"
patch -p1 --batch --forward --fuzz=0 --dry-run < "$PATCHFILE" >/dev/null
patch -p1 --batch --forward --fuzz=0 < "$PATCHFILE"

META_NEW=$(sha "$PKG/b12x_gdn_metadata.py")
ATTN_NEW=$(sha "$PKG/gdn_attn.py")
if [ "$META_NEW" != "$POST_META" ] || [ "$ATTN_NEW" != "$POST_ATTN" ]; then
  echo "mod vllm-gdn-meta-fuse: post-image mismatch -- patched result is not the expected bytes"
  echo "  b12x_gdn_metadata.py want $POST_META got $META_NEW"
  echo "  gdn_attn.py          want $POST_ATTN got $ATTN_NEW"
  exit 1
fi

python3 -c "
import ast
for f in ('$PKG/b12x_gdn_metadata.py', '$PKG/gdn_attn.py'):
    ast.parse(open(f).read())
print('ast ok')
"

python3 - <<'PY'
import inspect
import torch
from vllm.v1.attention.backends.b12x_gdn_metadata import (
    B12xGdnMixedMetadata,
    _copy_worklists_and_refresh_states,
)
import vllm.v1.attention.backends.gdn_attn as gdn_attn

cls = B12xGdnMixedMetadata
for name in ("copy_worklists_from", "refresh_state_indices", "copy_and_refresh_from"):
    assert callable(getattr(cls, name)), name
assert _copy_worklists_and_refresh_states is not None
# the single call site must be rewired, and the old pair must be gone from it
src = inspect.getsource(gdn_attn.GDNAttentionMetadataBuilder)
assert "copy_and_refresh_from" in src
assert "copy_worklists_from" not in src, "old call site still present in gdn_attn"

# CPU equivalence: the fallback branch of copy_and_refresh_from must produce
# exactly what copy_worklists_from + refresh_state_indices produced before.
cpu = torch.device("cpu")
kw = dict(max_tokens=64, max_seqs=8, state_columns=4, device=cpu)


def make_source():
    s = cls(**kw)
    s._num_non_spec, s._num_spec = 3, 2
    s.request_rows[:3] = torch.tensor([0, 2, 5], dtype=torch.int64)
    s.spec_request_rows[:2] = torch.tensor([1, 3], dtype=torch.int64)
    s.checkpoint_columns[:3] = torch.tensor([0, 1, 2], dtype=torch.int64)
    s.checkpoint.checkpoint_offsets[:3] = torch.tensor([7, 0, 3], dtype=torch.int32)
    s.query_start_loc[:4] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    s.token_indices[:3] = torch.tensor([9, 8, 7], dtype=torch.int64)
    s.has_initial_state[:3] = torch.tensor([True, False, True])
    return s


state_indices = torch.arange(8 * 4, dtype=torch.int32).reshape(8, 4)
block_table = torch.arange(8 * 4, dtype=torch.int32).reshape(8, 4) + 100

src_a, src_b = make_source(), make_source()
new, old = cls(**kw), cls(**kw)
new.copy_and_refresh_from(src_a, state_indices, block_table)
old.copy_worklists_from(src_b)
old.refresh_state_indices(state_indices, block_table)

for name in ("state_indices", "spec_state_indices", "token_indices",
             "request_rows", "spec_request_rows", "checkpoint_columns",
             "has_initial_state", "query_start_loc"):
    assert torch.equal(getattr(new, name), getattr(old, name)), name
assert torch.equal(new.checkpoint.state_indices, old.checkpoint.state_indices)
assert torch.equal(new.checkpoint.checkpoint_offsets, old.checkpoint.checkpoint_offsets)
# and the values are the ones the old path is specified to produce
assert new.state_indices[:3].tolist() == [0, 8, 20]
assert new.checkpoint.state_indices[:3].tolist() == [100, 0, 122]
assert new.spec_state_indices[:2].tolist() == [[4, 5, 6, 7], [12, 13, 14, 15]]
assert (new._num_non_spec, new._num_spec) == (3, 2)

# capacity mismatch must still fail closed
try:
    cls(max_tokens=64, max_seqs=9, state_columns=4, device=cpu).copy_and_refresh_from(
        src_a, state_indices, block_table
    )
except ValueError:
    pass
else:  # pragma: no cover
    raise AssertionError("capacity mismatch was not rejected")

print("import smoke ok; cpu fallback == legacy two-call path")
PY

echo "mod vllm-gdn-meta-fuse: applied (GDN cross-group metadata copy + state refresh fused into one Triton launch)"
