#!/usr/bin/env bash
# r3-queue (2026-10-08, #97): starts the backlog runner for the refit run3 chain (r3data r3train k56c, GRACE0=60) only
# after the jobs queued before it: b16 (b1.6 ship) and k72 (#106 DP=2 gate) both have a final STATE. If k72 never gets
# a STATE within 60 min of b16 ending, k72 was not queued after all and the chain goes. Never touches the gpu-lock
# itself; the runner does (one job at a time, exit 75 waits). Copy runs as ~/GEN-AI/backlog/r3-queue.sh on dgx-01.
# Start: setsid nohup bash ~/GEN-AI/backlog/r3-queue.sh > ~/GEN-AI/backlog/r3-queue.nohup 2>&1 < /dev/null &
# Stop:  kill -TERM <pid in r3-queue.log> while it waits.
set -u
BL=$HOME/GEN-AI/backlog
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" >> "$BL/r3-queue.log"; }
final() { head -1 "$BL/$1/STATE" 2>/dev/null | grep -qE '^(DONE|FAILED|SKIPPED|STOPPED)'; }
log "r3-queue pid $$ waiting for b16 and k72"
t0=$(date +%s)
while :; do
  [ $(( $(date +%s) - t0 )) -gt 172800 ] && { log "gave up after 48 h"; exit 1; }
  if final b16; then
    final k72 && break
    if [ ! -e "$BL/k72/STATE" ] && [ $(( $(date +%s) - $(stat -c %Y "$BL/b16/STATE") )) -gt 3600 ]; then log "k72 has no STATE 60 min after b16 ended: not queued, going"; break; fi
  fi
  sleep 60; done
log "b16: $(head -1 "$BL/b16/STATE"); k72: $(head -1 "$BL/k72/STATE" 2>/dev/null || echo none); starting the runner"
JOBS="r3data r3train k56c" GRACE0=60 setsid nohup bash "$BL/runner.sh" > "$BL/runner-r3.nohup" 2>&1 < /dev/null &
log "runner pid $! (JOBS='r3data r3train k56c' GRACE0=60)"
