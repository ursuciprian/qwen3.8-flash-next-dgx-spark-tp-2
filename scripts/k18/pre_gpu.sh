#!/usr/bin/env bash
# k18 (opus-kernel-18): GPU checks before the TP=1 A16 boot, run on one Spark with its server down
# (PRE_H2 of k18_tp1_driver.sh). 1) b12x A16 GPU tests in the k18 image, 2) b12x MoE microbench
# W4A4 vs W4A16 at I=640, 3) vLLM Marlin W4A16 MoE microbench in the reference image.
# Exit code = the GPU tests' exit code (non-zero skips the A16 boot on this node).
set -u
OUT=${1:?out dir}; mkdir -p "$OUT"
K=$HOME/GEN-AI/k18
IMG=${K18_IMAGE:-spark-vllm-b12x:k18a16-4e777f5e-884b4ff6}
REF=${REF_IMAGE:-qwen38-flash-dgx:iter6d-20260910}
SHAPES=${SHAPES:-5:33,10:60,20:96,40:170}
G="--rm --gpus all --ipc host --network none -e CUTE_DSL_ARCH=sm_121a -e PYTHONDONTWRITEBYTECODE=1"
echo "[$(date '+%F %T')] gpu tests"
timeout 3600 docker run $G -v "$K/pylib:/pl:ro" -v "$K/b12x-src:/src:ro" -w /src -e PYTHONPATH=/pl \
  --entrypoint python3 "$IMG" -m pytest -q -p no:cacheprovider -x -rs \
  tests/moe/test_a16_cutoff_qwen_tp1.py tests/moe/test_nvfp4_auto.py tests/moe/test_w4a16_route_pack.py \
  tests/preparation/test_stream_gate_collection.py > "$OUT/gpu-tests.txt" 2>&1
T=$?; echo "[$(date '+%F %T')] gpu tests exit=$T"; tail -25 "$OUT/gpu-tests.txt"
echo "[$(date '+%F %T')] b12x MoE bench"
timeout 3600 docker run $G -v "$K/scripts:/s:ro" --entrypoint python3 "$IMG" /s/bench_moe_a16.py \
  --intermediate 640 --shapes "$SHAPES" > "$OUT/moe-a16-I640.txt" 2>&1
echo "[$(date '+%F %T')] b12x bench exit=$?"; cat "$OUT/moe-a16-I640.txt" | grep -vE "Warning|warn" | tail -30
echo "[$(date '+%F %T')] marlin bench"
timeout 1800 docker run $G -v "$K/scripts:/s:ro" --entrypoint python3 "$REF" /s/bench_moe_marlin.py \
  --intermediate 640 --shapes "$SHAPES" > "$OUT/moe-marlin-I640.txt" 2>&1
echo "[$(date '+%F %T')] marlin bench exit=$?"; tail -8 "$OUT/moe-marlin-I640.txt"
exit $T
