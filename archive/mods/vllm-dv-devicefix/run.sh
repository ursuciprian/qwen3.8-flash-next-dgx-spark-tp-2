#!/usr/bin/env bash
# Draft-vocab fixes for the dv arm (vllm-0001-mtp-draft-vocab.patch), applied to
# the installed vLLM inside the -dv image so both crashes can be fixed and
# re-tested without a rebuild. Same mechanism as mods/vllm-spec-trace: replaces
# the installed file by path (no vllm import), idempotent and fail-closed.
# Ships mtp_fixed.py directly (rather than anchor-patching an already-modified
# target) since two separate crashes each needed structural changes -- see
# patches/vllm-0001-mtp-draft-vocab.patch for the full history/rationale.
#
# Fix 1 (boot1): torch.arange(...) in register_buffer("draft_id_to_target_id", ...)
# defaulted to CPU while target_ids is already on cuda:0.
# Fix 2 (boot2): resizing the PRIMARY lm_head to draft_vocab_size made b12x's
# checkpoint router reject the shape-mismatched sliced weight load
# (NotImplementedError: checkpoint routing performed an unsupported data
# transformation). Now the primary lm_head stays full-vocab (checkpoint loads
# unmodified) and a separate draft_lm_head is populated post-load instead.
#
# Once this arm is gated and promoted, rebuild from
# ~/GEN-AI/build/patches/vllm-0001-mtp-draft-vocab.patch (already carries both
# fixes) so they ship in the image, not as a runtime mod.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MTP=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/mtp.py

[ -f "$MTP" ] || { echo "mod vllm-dv-devicefix: $MTP not in this image, skipping"; exit 0; }

if cmp -s "$HERE/mtp_fixed.py" "$MTP"; then
  echo "mod vllm-dv-devicefix: already applied, skipping"
  exit 0
fi

cp "$HERE/mtp_fixed.py" "$MTP"
python3 -c "import ast; ast.parse(open('$MTP').read())"
echo "mod vllm-dv-devicefix: applied (device fix + draft_lm_head split)"
