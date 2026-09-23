#!/usr/bin/env bash
# Mod: vllm-qwen-scratch-isolation
# Applies upstream fork commit 98cd717 ("Fix shared b12x scratch in concurrent
# Qwen projections") to the installed vLLM, rebased against this image's
# actual file contents. See README.md for the commit analysis and why
# 200b6e2/9c27ec0 are NOT included (200b6e2 needs 9c27ec0's preparation API,
# which does not exist in this image and is too large to patch cleanly).
#
# Candidate fix for the c5-c7/c9/c12 MTP straggler: gives the GDN and QSA
# input projections disjoint preallocated scratch instead of letting the
# aux-stream and main-stream projections share whatever scratch the b12x
# split-K/activation kernel picked.
#
# Idempotent: skips a file whose marker (get_b12x_projection_workspaces) is
# already present. Fail-closed: any `patch` failure aborts before AST-check;
# no half-patched file is left in a broken state because `patch` only writes
# on a clean hunk match.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P=/usr/local/lib/python3.12/dist-packages
B12X="$P/vllm/utils/b12x.py"
GDN="$P/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
QSA="$P/vllm/models/qwen3_8_flash_next/nvidia/qsa.py"

for f in "$B12X" "$GDN" "$QSA"; do
  [ -f "$f" ] || { echo "mod vllm-qwen-scratch-isolation: $f not in this image, skipping"; exit 0; }
done

if grep -q get_b12x_projection_workspaces "$B12X" \
   && grep -q get_b12x_projection_workspaces "$GDN" \
   && grep -q get_b12x_projection_workspaces "$QSA"; then
  echo "mod vllm-qwen-scratch-isolation: already patched, skipping"
  exit 0
fi

cd /
patch -p0 --forward < "$HERE/01-b12x-projection-workspaces.diff"
patch -p0 --forward < "$HERE/02-gdn-projection-isolation.diff"
patch -p0 --forward < "$HERE/03-qsa-projection-isolation.diff"

python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$B12X" "$GDN" "$QSA"
echo "mod vllm-qwen-scratch-isolation: applied (98cd717 rebased: GDN/QSA input-projection scratch isolation)"
