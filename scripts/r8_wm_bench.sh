#!/usr/bin/env bash
# r8 TP=1 wm + draft-GEMV microbench (opus-kernel-16, 2026-10-01). Run detached on dgx-01:
#   setsid nohup bash scripts/r8_wm_bench.sh <STATE file to wait for> > <log> 2>&1 &
# Waits until the STATE file reads DONE/FAILED/SKIPPED, then takes ~/GEN-AI/gpu-lock (flock,
# shared with other agents) and runs short containers on the r8 TP=1 image with the b12x
# exp/r8-tp1-wm tree mounted over PYTHONPATH (no image build). Never touches a server.
#   1. GPU equivalence: tests/moe/test_wm_decode.py at I=640 (TP=1) and I=320 (TP=2 regression)
#   2. bench_moe_pad.py dynamic vs wm, I=640 at M:D 1:10 5:33 8:43 10:60 20:96, I=320 5:33 20:96
#   3. bench_draft_gemv.py: MTP draft BF16 projections, cuBLAS vs b12x bf16_gemv
set -u
WAIT=${1:?usage: r8_wm_bench.sh <STATE file>}
REPO=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
B12X=${B12X:-$HOME/GEN-AI/worktrees/b12x-r8wm}
IMAGE=${IMAGE:-spark-vllm-b12x:r8tp1-b7fbaf96-95f39b28}
R=${RESULTS:-$REPO/results/r8-wm-bench}
mkdir -p "$R"; echo waiting > "$R/STATE"
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$R/run.log"; }
log "waiting for $WAIT"
until grep -qE '^(DONE|FAILED|SKIPPED)' "$WAIT" 2>/dev/null; do sleep 60; done
log "wait over: $(head -1 "$WAIT"); taking gpu-lock"
run() { # name timeout_s cmd...
  local name=$1 t=$2; shift 2
  timeout "$t" docker run --rm --name "r8wm-$name" --gpus all --ipc host --network none \
    -e CUTE_DSL_ARCH=sm_121a -e PYTHONPATH=/b12x -v "$B12X":/b12x -v "$REPO/scripts":/s:ro \
    -v "$HOME/GEN-AI/worktrees/pipwheels":/w:ro -w /b12x --entrypoint bash "$IMAGE" -c "$*" \
    > "$R/$name.txt" 2>&1
  local rc=$?; [ $rc -eq 124 ] && docker rm -f "r8wm-$name" >/dev/null 2>&1
  log "$name exit=$rc"; tail -4 "$R/$name.txt" | sed 's/^/    /' | tee -a "$R/run.log"
}
(
  flock 9
  echo running > "$R/STATE"; log "gpu-lock held; free: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  PYT="pip install -q --no-index -f /w pytest >/dev/null 2>&1; python3 -m pytest -q -p no:cacheprovider"
  run test-I640 1800 "B12X_WM_TEST_INTERMEDIATE=640 $PYT tests/moe/test_wm_decode.py"
  run test-I320 1800 "$PYT tests/moe/test_wm_decode.py"
  run moe-I640 1800 "python3 /s/bench_moe_pad.py --backends dynamic,wm --intermediate 640 --shapes 1:10,5:33,8:43,10:60,20:96"
  run moe-I320 1200 "python3 /s/bench_moe_pad.py --backends dynamic,wm --intermediate 320 --shapes 5:33,20:96"
  run gemv 1200 "python3 /s/bench_draft_gemv.py"
  echo DONE > "$R/STATE"; log DONE
) 9>"$HOME/GEN-AI/gpu-lock"
