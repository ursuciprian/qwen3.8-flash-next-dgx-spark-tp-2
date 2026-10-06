#!/usr/bin/env bash
# refit-p2 resume queue (#97, 2026-10-06): blocks on ~/GEN-AI/gpu-lock behind k55 (k55 queue pid 3081953), then runs
# job.sh --resume-cap (DAY 20261006: cap boot, capture, checks, restore of the shipped 2x) under the lock with
# GPU_LOCK_HELD=1. Starts the dashboard sidecar once the job has written a running STATE. Waits up to 12 h.
# Stop while waiting: kill -TERM <queue pid in after_k55.log>. Never pkill -f.
set -u
P=$HOME/GEN-AI/refit-p2; M=$HOME/GEN-AI/refit-metrics
export DAY=20261006
RES=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/refit-p2-$DAY; D=$HOME/.cache/huggingface/mtp-refit/p2-$DAY
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" >> "$P/after_k55.log"; }
log "queue pid $$"; echo "waiting: gpu-lock (k55), then resume at cap boot" > "$P/STATE"
flock -w 43200 -o "$HOME/GEN-AI/gpu-lock" env GPU_LOCK_HELD=1 bash -c '
  bash "$1/job.sh" --resume-cap & J=$!
  for i in $(seq 120); do grep -q "^running" "$1/STATE" && break; kill -0 $J 2>/dev/null || break; sleep 5; done
  if grep -q "^running" "$1/STATE"; then
    ( cd "$2" && setsid nohup nice -n 19 python3 -m tools.mtp_refit.p2_metrics --state "$1/STATE" --res "$3" --data "$4" \
        --out /var/lib/node_exporter/textfile_collector/mtp_refit_p2.prom >> p2_metrics.log 2>&1 < /dev/null &
      echo $! > p2_metrics.pid; echo "sidecar pid $!" >> "$1/after_k55.log" )
  fi
  wait $J' _ "$P" "$M" "$RES" "$D"; rc=$?
grep -q '^waiting: gpu-lock' "$P/STATE" && echo "FAILED: gpu-lock busy for 12 h (flock exit $rc); nothing run" > "$P/STATE"
log "job.sh / flock exit $rc; STATE: $(cat "$P/STATE")"
