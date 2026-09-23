#!/usr/bin/env bash
# Reverts b12x commit 06809d5 ("Reconcile cached tuning selections before
# distributed preparation") to unblock 2-node TP preparation on this image
# (spark-vllm-b12x:local-20260919-0f3a8cbf-b12x0f3a8cb, b12x @ 0f3a8cb).
#
# Root cause (see results/kernel-pass/b12x0f3a8cb-hang/SUMMARY.md and
# results/kernel-pass/arms.md for the full evidence trail): 06809d5 adds a
# new TuningCacheRequirement/"ready_cache" handshake. PreparationJob._run()
# now yields this requirement before any tuning begins whenever
# len(session._tuning_ranks) > 1 (i.e. unconditionally at TP=2), and expects
# the CALLER's coordinator (vLLM's startup code, not b12x) to recognize
# state.ready_cache and answer with a cache= kwarg carrying every rank's
# selection snapshot. Our pinned vLLM fork (local-inference-lab/vllm
# @dev/jovian-judgement, 8e1f1e58, 0 commits behind origin) predates this
# b12x feature and has no code path that answers ready_cache, so
# PreparationSession.prepare()'s advance() loop spins forever re-polling an
# unresolved state: confirmed live via nvidia-smi (both GPUs 0% util/~9.6W)
# and /proc/<pid>/wchan (rank0 Worker_TP0 on-CPU spinning at wchan=0 with
# real utime/stime deltas; rank1 Worker_TP1 legitimately parked at
# wchan=wait_woken). Reproduced identically from a completely empty plan
# cache, ruling out a stale-cache explanation.
#
# Fix: revert 06809d5's four touched Python files to their pre-06809d5
# state. The commit touches ONLY Python (b12x/preparation/__init__.py,
# _cache.py, session.py, types.py -- confirmed via `git show --stat`, no
# .cu/.so), so this is a source-only fix, no kernel rebuild needed. The
# very next commit (a2b5152) also touches types.py but only adds an
# unrelated field (PreparedCall.benchmark_producers) in a different class,
# confirmed independent -- the reverse-patch applies cleanly on top of HEAD
# (verified with `git apply -R --check` against the b12x 0f3a8cb checkout
# before this mod was written).
#
# This restores the pre-06809d5 single-rank-authoritative tuning-cache
# behavior (rank 0 owns the disk cache, others just read it) -- correct
# tuning behavior, not a numerics change: no kernel, no accumulation dtype,
# no scaling touched. It gives up 06809d5's cross-rank race-avoidance
# improvement for asymmetric caches, which does not apply to our identical
# 2-node setup anyway.
#
# Fail-closed: a dry run must succeed before anything is written; the
# touched files must contain the 06809d5 marker (not already reverted, and
# not absent because a different image is running) or this is a no-op.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P=/usr/local/lib/python3.12/dist-packages
TARGET="$P/b12x/preparation/types.py"

[ -f "$TARGET" ] || { echo "mod b12x-revert-06809d5: b12x.preparation not in this image, skipping"; exit 0; }

if ! grep -q "class TuningCacheRequirement" "$TARGET"; then
  echo "mod b12x-revert-06809d5: TuningCacheRequirement absent, already reverted or not present in this build; skipping"
  exit 0
fi

if ! patch -p1 -d "$P" -R --dry-run < "$HERE/06809d5.diff" > /tmp/06809d5-revert-dryrun.txt 2>&1; then
  echo "FATAL: reverse-patch of b12x 06809d5 does not apply to this build:" >&2
  cat /tmp/06809d5-revert-dryrun.txt >&2
  exit 1
fi
patch -p1 -d "$P" -R < "$HERE/06809d5.diff" > /dev/null
python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" \
  "$P/b12x/preparation/__init__.py" "$P/b12x/preparation/_cache.py" \
  "$P/b12x/preparation/session.py" "$P/b12x/preparation/types.py"
python3 -c "import b12x.preparation" 2>/tmp/06809d5-revert-import.txt \
  || { echo "FATAL: b12x.preparation fails to import after revert:" >&2; cat /tmp/06809d5-revert-import.txt >&2; exit 1; }
echo "mod b12x-revert-06809d5: applied (pre-06809d5 tuning-cache behavior restored)"
