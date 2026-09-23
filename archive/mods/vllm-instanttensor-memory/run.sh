#!/usr/bin/env bash
# mod vllm-instanttensor-memory
#
# Port of eugr/spark-vllm-docker f247397d ("Instanttensor memory buffer fix",
# docker/patch_instanttensor_vllm_memory.py, 2026-09-18) to our pinned image
# spark-vllm-b12x:local-20260918-a8333658. Credit: eugr.
#
# What it does: InstantTensor sizes its weight-load staging buffer from
# `torch.cuda.mem_get_info()`. On UMA parts (GB10, GH200, Jetson) cudaMemGetInfo
# underreports free memory -- it does not count reclaimable host memory (page
# cache, buffers) -- so the loader asks for half of a number that is far below
# what the device can actually allocate. vLLM already solved this in
# `MemorySnapshot.measure()`: on an integrated GPU that is not CUDA-on-WSL it
# replaces the cudaMemGetInfo figure with `psutil.virtual_memory().available`.
# This patch makes InstantTensor's budget read that same snapshot, so the two
# agree and the WSL carve-out is honoured in one place.
#
# Our hardware takes the psutil branch (Linux GB10, integrated, not WSL), so
# this is a behaviour change here, not a no-op: the load budget grows to half of
# MemAvailable instead of half of cudaMemGetInfo free. Effect is on weight-load
# staging only -- it does not touch gpu_memory_utilization or the KV budget.
#
# `total_bytes` is dropped: it had exactly one binding and no reader.
#
# Fail-closed: exact SHA256 pre-image, dry-run, SHA256 post-image, AST parse,
# an AST check that the edit sits in `safe_open._determine_io_params`, and a
# CPU-only smoke of the MemorySnapshot surface the patch now depends on
# (including the UMA branch it exists for). Idempotent: re-running on an
# already-patched tree exits 0.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/usr/local/lib/python3.12/dist-packages
TARGET="$ROOT/instanttensor/_impl.py"
MEMUTILS="$ROOT/vllm/utils/mem_utils.py"
PATCHFILE="$HERE/vllm-instanttensor-memory.patch"

PRE=4c5dc40095b859612ff13cb013db1d5f6cfcb21ff2921413cb7cc6c5e0258cb7
POST=c6ebded826a007c9a6e0fdc3311a02077026a80e431373c998c2d825df477886

sha() { sha256sum "$1" | cut -d' ' -f1; }

for f in "$TARGET" "$MEMUTILS" "$PATCHFILE"; do
  [ -f "$f" ] || { echo "mod vllm-instanttensor-memory: missing $f -- refusing"; exit 1; }
done

NOW=$(sha "$TARGET")
if [ "$NOW" = "$POST" ]; then
  echo "mod vllm-instanttensor-memory: already applied, skipping"
  exit 0
fi
if [ "$NOW" != "$PRE" ]; then
  echo "mod vllm-instanttensor-memory: pre-image mismatch -- refusing to patch"
  echo "  instanttensor/_impl.py want $PRE got $NOW"
  echo "  (cut against spark-vllm-b12x:local-20260918-a8333658; re-cut for this image)"
  exit 1
fi

cd "$ROOT"
patch -p1 --batch --forward --fuzz=0 --dry-run < "$PATCHFILE" >/dev/null
patch -p1 --batch --forward --fuzz=0 < "$PATCHFILE"

NEW=$(sha "$TARGET")
if [ "$NEW" != "$POST" ]; then
  echo "mod vllm-instanttensor-memory: post-image mismatch -- patched result is not the expected bytes"
  echo "  want $POST got $NEW"
  exit 1
fi

TARGET="$TARGET" MEMUTILS="$MEMUTILS" python3 - <<'PY'
import ast, os

target, memutils = os.environ["TARGET"], os.environ["MEMUTILS"]
src = open(target).read()
ast.parse(src)

# the edit must live in safe_open._determine_io_params, and nothing else in the
# file may still size a budget from cudaMemGetInfo
hits = [i + 1 for i, line in enumerate(src.splitlines()) if "MemorySnapshot(device=self.device)" in line]
assert len(hits) == 1, hits
assert "torch.cuda.mem_get_info" not in src, "an unpatched mem_get_info budget remains"

line = hits[0]
scopes = [
    n.name
    for n in ast.walk(ast.parse(src))
    if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.lineno <= line <= n.end_lineno
]
assert "safe_open" in scopes and "_determine_io_params" in scopes, scopes

# the snapshot surface the patch depends on, without touching CUDA
from vllm.utils.mem_utils import MemorySnapshot

snap = MemorySnapshot(device="cuda:0", auto_measure=False)
assert isinstance(snap.free_memory, int)
measure = ast.unparse(
    next(
        m
        for n in ast.parse(open(memutils).read()).body
        if isinstance(n, ast.ClassDef) and n.name == "MemorySnapshot"
        for m in n.body
        if isinstance(m, ast.FunctionDef) and m.name == "measure"
    )
)
# the UMA carve-out is the whole point; if upstream drops it the patch is pointless
assert "is_integrated_gpu" in measure and "psutil.virtual_memory().available" in measure
assert "in_wsl" in measure, "WSL carve-out missing from MemorySnapshot.measure"
print("ast ok; edit in safe_open._determine_io_params; MemorySnapshot UMA+WSL branch present")
PY

echo "mod vllm-instanttensor-memory: applied (InstantTensor load budget reads vLLM's UMA-aware MemorySnapshot)"
