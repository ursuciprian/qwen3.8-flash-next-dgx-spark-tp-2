#!/usr/bin/env bash
# Diagnostic-only instrumentation for the b12x TP=2 preparation deadlock
# (results/kernel-pass/prep-deadlock/mechanism.md). Patches ONLY
# vllm/v1/worker/b12x_startup.py (the fork-side coordinator, our own file,
# Apache-2.0) in the KNOWN-GOOD old image
# (spark-vllm-b12x:local-20260918-a8333658). No b12x file is touched.
#
# What it adds, all additive/observational:
#   - one INFO log line per B12xPreparationCoordinator.advance() call:
#     rank, round, local_done, global_done, ready collectives, ready tuning
#     keys, active request names, candidates prepared/total.
#   - EXCHANGE ENTER/EXIT log lines around _exchange(): round, authority,
#     wall time waited, decision done/tuning keys.
#   - the two unbounded Store.get() calls inside _exchange() (gathering
#     per-rank round payloads, and reading the round's decision) are
#     replaced with _bounded_get(): a 120s store.wait() that logs
#     "WAIT round=N key=... rank=... 120s" and retries instead of blocking
#     silently forever. Same eventual read, same payload/decision format --
#     this does NOT change the protocol or candidate selection, it only
#     makes an existing indefinite block observable and non-silent.
#
# All trace output goes to stderr via its own "b12x_startup_trace" logger
# with its own handler (independent of vLLM's log config), so it lands in
# each rank's normal sparkrun server log.
#
# Fail-closed: verifies the pre-image hash of the one touched file matches
# the exact vllm fork blob this image was built from (confirmed via
# `docker exec ... sha256sum` against the running a8333658 image before
# writing this mod), a dry run must succeed, and the patched file is
# AST-validated plus smoke-imported afterward.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P=/usr/local/lib/python3.12/dist-packages

TARGET="$P/vllm/v1/worker/b12x_startup.py"
EXPECTED_SHA256="7e8f18873b5ac23f316dae492e59497b18f9828303b4e3563664d293eb9e0d1b"

[ -f "$TARGET" ] || { echo "mod b12x-startup-trace: $TARGET not in this image, skipping"; exit 0; }

if grep -q "b12x_startup_trace" "$TARGET"; then
  echo "mod b12x-startup-trace: trace marker already present; already applied, skipping"
  exit 0
fi

actual="$(sha256sum "$TARGET" | awk '{print $1}')"
if [ "$actual" != "$EXPECTED_SHA256" ]; then
  echo "FATAL: $TARGET does not match the expected a8333658-image pre-image (refusing to patch a file that has drifted)." >&2
  echo "  expected: $EXPECTED_SHA256" >&2
  echo "  actual:   $actual" >&2
  exit 1
fi

if ! patch -p1 -d "$P" --dry-run < "$HERE/startup-trace.diff" > /tmp/startup-trace-dryrun.txt 2>&1; then
  echo "FATAL: b12x-startup-trace patch does not apply to this build:" >&2
  cat /tmp/startup-trace-dryrun.txt >&2
  exit 1
fi
patch -p1 -d "$P" < "$HERE/startup-trace.diff" > /dev/null
python3 -c "import ast; ast.parse(open('$TARGET').read())"
python3 -c "import vllm.v1.worker.b12x_startup" 2>/tmp/startup-trace-import.txt \
  || { echo "FATAL: vllm.v1.worker.b12x_startup fails to import after patch:" >&2; cat /tmp/startup-trace-import.txt >&2; exit 1; }
echo "mod b12x-startup-trace: applied (rank-tagged coordinator tracing + 120s bounded-wait/retry on the round rendezvous Store.get() calls, diagnostic only)"
