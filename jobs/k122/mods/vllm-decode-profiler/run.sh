#!/usr/bin/env bash
# Decode-step-local profiler for phase C kernel timing (results/profiling/README.md).
# Wraps GPUModelRunner.execute_model (vllm/v1/worker/gpu/model_runner.py, the
# runner used when VLLM_USE_V2_MODEL_RUNNER=1, as this recipe sets) with a
# rank-local torch.profiler window -- no collective RPC, so it sidesteps the
# start_profile/stop_profile "collective_rpc should not be called on follower
# node" deadlock documented in results/profiling/README.md. Off by default;
# armed only by dropping a uniquely-named file into VLLM_LOCAL_PROF_TRIGGER_DIR
# or by VLLM_LOCAL_PROF_START.
# Purely additive (appended after the class body already exists in the
# module), so it never edits vLLM's own execute_model logic. Idempotent
# (skips if marker already present) and fail-closed (aborts if the file or
# class isn't there instead of silently no-op'ing on a signature change).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py
MARKER="local step profiler (mods/vllm-decode-profiler)"

[ -f "$RUNNER" ] || { echo "mod vllm-decode-profiler: $RUNNER not in this image, skipping"; exit 0; }
grep -q "^class GPUModelRunner" "$RUNNER" || { echo "mod vllm-decode-profiler: GPUModelRunner class not found, aborting" >&2; exit 1; }

if grep -q "$MARKER" "$RUNNER"; then
  echo "mod vllm-decode-profiler: already applied, skipping"
  exit 0
fi

cat "$HERE/profiler_snippet.py" >> "$RUNNER"
python3 -c "import ast; ast.parse(open('$RUNNER').read())"
echo "mod vllm-decode-profiler: applied (execute_model wrapped for rank-local decode profiling)"
