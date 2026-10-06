#!/usr/bin/env bash
# k56 (#97): Thunderdome screen of the refit run1 drafter vs shipped v3d, one arm per Spark (k56/refit-01 on dgx-01,
# k56/refit-02 on dgx-02, same recipe), bake (new model path), k53's hook (T=0 and T=1 acceptance per position);
# split gate on a PROMOTE (thunderdome default), restore of the shipped 2x by thunderdome. Staged by refit-p3/job.sh
# when the offline gain was >= +2 pts/pos; refuses without k56/READY. Started only on the user's word:
#   setsid nohup bash ~/GEN-AI/k56/start.sh > ~/GEN-AI/k56/start.nohup 2>&1 < /dev/null &
# Waits up to 12 h for ~/GEN-AI/gpu-lock. Stop: kill -TERM <pid in start.log>, or the thunderdome pid. Never pkill -f.
set -u
K=$HOME/GEN-AI/k56; R=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
RES=$R/results/thunderdome-k56-$(TZ=Europe/Bucharest date +%Y%m%d)
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" >> "$K/start.log"; }
st() { echo "$*" > "$K/STATE"; log "STATE: $*"; }
log "start pid $$"
[ -f "$K/READY" ] || { st "FAILED: k56 not staged (no $K/READY); nothing run"; exit 1; }
st "waiting: gpu-lock"
flock -w 43200 -o "$HOME/GEN-AI/gpu-lock" env GPU_LOCK_HELD=1 RES="$RES" bash "$R/scripts/thunderdome.sh" "$K/refit-01" "$K/refit-02" \
  > "$K/thunderdome.nohup" 2>&1 < /dev/null; rc=$?
grep -q '^waiting: gpu-lock' "$K/STATE" && [ ! -d "$RES" ] && { st "FAILED: gpu-lock busy for 12 h; nothing run"; exit 1; }
st "$([ $rc = 0 ] && echo DONE || echo FAILED): thunderdome exit $rc, $(head -1 "$RES/STATE" 2>/dev/null) ($RES)"
