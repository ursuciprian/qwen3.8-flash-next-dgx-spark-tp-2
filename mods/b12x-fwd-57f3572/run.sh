#!/usr/bin/env bash
# Forward-ports b12x commit 57f3572 ("fix(moe): restore native NVFP4 A16
# autotuning") onto the KNOWN-GOOD old image
# (spark-vllm-b12x:local-20260918-a8333658, b12x @ a8333658, the exact
# parent commit of 57f3572 -- confirmed with `git rev-parse 57f3572^`).
#
# Why the old image and not the b12x-HEAD rebuild: the HEAD rebuild
# (spark-vllm-b12x:local-20260919-0f3a8cbf-b12x0f3a8cb) hangs the 2-node TP
# preparation session twice, at two different distributed-coordination
# barriers (see results/kernel-pass/b12x0f3a8cb-hang/ and .../hang2/) --
# b12x HEAD's massively restructured preparation-session protocol (837
# candidates vs. 222, new TuningCacheRequirement handshake, new
# "rank N batch M" compile-batch barriers) assumes fork-side (vLLM)
# coordinator support our pinned vllm fork (local-inference-lab/vllm
# @dev/jovian-judgement, 8e1f1e58, 0 commits behind origin all session)
# does not have. Reverting one hanging commit (06809d5) fixed the first
# barrier but exposed a second one -- unbounded scope to fix by reverting
# individual protocol commits. The old a8333658 image already boots and
# serves fine with this exact vllm fork, so instead of dragging its
# preparation-session protocol forward, this mod drags forward ONLY the
# narrowly-scoped MoE-kernel fix we actually want (57f3572), onto the
# protocol base that is already known to work.
#
# 57f3572 touches only b12x/moe/*.py (fused_moe/_impl.py, _preparation.py,
# _tuning.py, and the w4a16 kernel.py under
# b12x/moe/_shared/kernels/w4a16/) plus docs/tests -- confirmed via
# `git show --stat 57f3572`, no .cu/.so. Its _preparation.py touch is
# b12x/moe/fused_moe/_preparation.py, a different module from
# b12x/preparation/session.py (the one that hung) -- this commit does not
# touch the distributed rank-coordination protocol at all, confirmed by
# reading the diff.
#
# 57f3572^ IS a8333658 (`git rev-parse 57f3572^` == a8333658), so this is a
# clean linear forward-port with no rebase, and the diff was verified with
# `git apply --check` against the b12x a8333658 checkout before this mod
# was written.
#
# Effect: admits native W4A16 direct routes to MoE precision tuning,
# selected by default at capacities 1-8 on SM120/SM121 (our exact decode
# shapes and GPU), and bumps the moe.decode candidate contract 4->5,
# forcing a retune from an empty plan cache on first boot (expected; not a
# bug). a2b5152 and 8783519 are NOT included: 57f3572 applies cleanly on
# its own (fail-closed check below), so they are unnecessary for this arm.
#
# Fail-closed: verifies the pre-image hash of every touched file matches
# git's exact a8333658 blob before patching (not just "looks similar"), a
# dry run must succeed, and each patched file is AST-validated plus
# smoke-imported afterward.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P=/usr/local/lib/python3.12/dist-packages

declare -A EXPECTED_SHA256=(
  ["$P/b12x/moe/_shared/kernels/w4a16/kernel.py"]="b6883970fcd1dc1284e2e908c72efce8972f23a3f5dd21fda399faa087c4ed3b"
  ["$P/b12x/moe/fused_moe/_impl.py"]="d54d7a579d303d3275440fc19fe88bcf0672151f0868030f467c6acfc50edf23"
  ["$P/b12x/moe/fused_moe/_preparation.py"]="fcd0b1f8d903fe1ca6076becd96c99c4562d45f6c42b4efd1bbaf2bfb70079c8"
  ["$P/b12x/moe/fused_moe/_tuning.py"]="648d433b76c780658ad4d45e08a2b8d455b345784ccd8bba69590c4caff98b05"
)

[ -f "$P/b12x/moe/fused_moe/_preparation.py" ] || { echo "mod b12x-fwd-57f3572: b12x.moe.fused_moe not in this image, skipping"; exit 0; }

# If the contract-bump marker is already present, this was already applied
# (or merged) -- skip rather than double-patch.
if grep -q "native_direct" "$P/b12x/moe/fused_moe/_preparation.py"; then
  echo "mod b12x-fwd-57f3572: native_direct marker already present; already applied or merged, skipping"
  exit 0
fi

for f in "${!EXPECTED_SHA256[@]}"; do
  actual="$(sha256sum "$f" | awk '{print $1}')"
  if [ "$actual" != "${EXPECTED_SHA256[$f]}" ]; then
    echo "FATAL: $f does not match the expected a8333658 pre-image (this is not the build 57f3572 was diffed against; refusing to patch a file that has drifted)." >&2
    echo "  expected: ${EXPECTED_SHA256[$f]}" >&2
    echo "  actual:   $actual" >&2
    exit 1
  fi
done

if ! patch -p1 -d "$P" --dry-run < "$HERE/57f3572.diff" > /tmp/57f3572-dryrun.txt 2>&1; then
  echo "FATAL: forward-patch of b12x 57f3572 does not apply to this build:" >&2
  cat /tmp/57f3572-dryrun.txt >&2
  exit 1
fi
patch -p1 -d "$P" < "$HERE/57f3572.diff" > /dev/null
python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" \
  "$P/b12x/moe/_shared/kernels/w4a16/kernel.py" "$P/b12x/moe/fused_moe/_impl.py" \
  "$P/b12x/moe/fused_moe/_preparation.py" "$P/b12x/moe/fused_moe/_tuning.py"
python3 -c "import b12x.moe.fused_moe._preparation, b12x.moe.fused_moe._tuning, b12x.moe.fused_moe._impl" \
  2>/tmp/57f3572-import.txt \
  || { echo "FATAL: b12x MoE modules fail to import after patch:" >&2; cat /tmp/57f3572-import.txt >&2; exit 1; }
echo "mod b12x-fwd-57f3572: applied (native NVFP4 A16 MoE autotuning restored at capacities 1-8/SM120-SM121, moe.decode contract 4->5)"
