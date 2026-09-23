#!/usr/bin/env bash
# Production fix for the b12x TP=2 preparation deadlock
# (results/kernel-pass/prep-deadlock/mechanism.md). Trimmed from
# mods/b12x-startup-trace/ (diagnostic boot 1): keeps only the fail-fast
# behavior, drops the verbose ADVANCE/EXCHANGE tracing.
#
# Root cause (confirmed by trace1, see mechanism.md "RoCE spin-limit
# root cause" section): the round rendezvous in B12xPreparationCoordinator
# ._exchange() (vllm/v1/worker/b12x_startup.py) itself worked correctly --
# round 0 exchanged fine on both ranks. The actual failure is one layer
# down, in b12x's RoCE one-shot collective
# (b12x/comm/roce/roce_oneshot.py): its GPU-side spin-wait for a peer's
# flag gives up after B12X_ROCE_SPIN_LIMIT polls (default 20_000_000,
# "roughly 20s" per its own comment) and marks the runtime "poisoned".
# During a from-empty-cache retune the two ranks' local MoE candidate
# racing paces diverge (different shard sizes, different per-candidate
# compile/measure cost -- see mechanism.md), so one rank can reach the
# shared distributed.roce.0-1.collectives priming call more than ~20s
# before its peer, which is normally fine (steady-state serving calls
# both sides in lockstep) but fails during a large retune. That is fixed
# by raising B12X_ROCE_SPIN_LIMIT in the recipe env, not by this mod.
#
# This mod is the belt-and-suspenders half of the fix: even with the RoCE
# spin limit raised, the METADATA-level round rendezvous here still has no
# bound on its own Store.get() calls, so any other kind of divergence
# still hangs silently forever instead of failing with evidence. Bounding
# it to 120s (well above the RoCE spin window) turns that into a visible,
# retryable wait instead of a silent parked process -- it does not change
# the payload/decision format or any protocol behavior when both ranks are
# healthy.
#
# Fail-closed: verifies the pre-image hash of the one touched file matches
# the exact vllm fork blob this image was built from, a dry run must
# succeed, and the patched file is AST-validated plus smoke-imported.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P=/usr/local/lib/python3.12/dist-packages

TARGET="$P/vllm/v1/worker/b12x_startup.py"
EXPECTED_SHA256="7e8f18873b5ac23f316dae492e59497b18f9828303b4e3563664d293eb9e0d1b"

[ -f "$TARGET" ] || { echo "mod b12x-startup-boundedwait: $TARGET not in this image, skipping"; exit 0; }

if grep -q "b12x_startup_trace" "$TARGET"; then
  echo "mod b12x-startup-boundedwait: trace/boundedwait marker already present; already applied, skipping"
  exit 0
fi

actual="$(sha256sum "$TARGET" | awk '{print $1}')"
if [ "$actual" != "$EXPECTED_SHA256" ]; then
  echo "FATAL: $TARGET does not match the expected a8333658-image pre-image (refusing to patch a file that has drifted)." >&2
  echo "  expected: $EXPECTED_SHA256" >&2
  echo "  actual:   $actual" >&2
  exit 1
fi

if ! patch -p1 -d "$P" --dry-run < "$HERE/boundedwait.diff" > /tmp/boundedwait-dryrun.txt 2>&1; then
  echo "FATAL: b12x-startup-boundedwait patch does not apply to this build:" >&2
  cat /tmp/boundedwait-dryrun.txt >&2
  exit 1
fi
patch -p1 -d "$P" < "$HERE/boundedwait.diff" > /dev/null
python3 -c "import ast; ast.parse(open('$TARGET').read())"
python3 -c "import vllm.v1.worker.b12x_startup" 2>/tmp/boundedwait-import.txt \
  || { echo "FATAL: vllm.v1.worker.b12x_startup fails to import after patch:" >&2; cat /tmp/boundedwait-import.txt >&2; exit 1; }
echo "mod b12x-startup-boundedwait: applied (120s bounded-wait/retry on the round rendezvous Store.get() calls, fail-visible instead of silent-forever)"
