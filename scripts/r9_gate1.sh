#!/usr/bin/env bash
# r9 requant gate 1 job (opus-kernel-17): waits for <STATE>, takes gpu-lock, runs bench_nvfp4_a16.py on
# dgx-01 in the b1.4 image (b1.4 serving idle beside it, as for k15's microbench). STATE in $OUT.
set -u
WAIT=${1:?usage: r9_gate1.sh <STATE file>}
REPO=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
OUT=${OUT:-$REPO/results/r9-gate1}; mkdir -p "$OUT"; echo waiting > "$OUT/STATE"
until grep -qE '^(DONE|FAILED|SKIPPED)' "$WAIT" 2>/dev/null; do sleep 60; done
exec 9>"$HOME/GEN-AI/gpu-lock"; flock 9; echo running > "$OUT/STATE"
timeout 3600 docker run --rm --name r9-gate1 --gpus all --ipc=host --entrypoint bash \
  -v "$REPO/scripts:/s:ro" -v "$OUT:/out" -e PYTHONDONTWRITEBYTECODE=1 \
  ghcr.io/ursuciprian/spark-vllm-b12x:b1.4-20261001-b7fbaf96-a7e649d8-warm \
  -c "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv; python3 -W ignore /s/bench_nvfp4_a16.py --out /out/nvfp4.jsonl" > "$OUT/bench.log" 2>&1
rc=$?; [ $rc = 0 ] && echo DONE > "$OUT/STATE" || echo "FAILED: rc=$rc" > "$OUT/STATE"
