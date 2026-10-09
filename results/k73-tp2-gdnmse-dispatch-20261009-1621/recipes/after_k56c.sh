#!/usr/bin/env bash
# k73 queue (2026-10-08, #123): starts the backlog runner for k73 (JOBS=k73 GRACE0=60) only after the refit run3 chain
# is over: k56c has a final STATE and the runner that runs that chain (RP, JOBS='r3data r3train k56c') has exited, so
# its end-of-chain restore is done. A chain runner that is gone without a final k56c STATE, or a STOPPED runner, needs a
# person: then this exits without starting anything. Never touches the gpu-lock itself; the runner does.
# Start: RP=<r3 runner pid> setsid nohup bash ~/GEN-AI/backlog/k73/after_k56c.sh > ~/GEN-AI/backlog/k73/after_k56c.nohup 2>&1 < /dev/null &
# Stop:  kill -TERM <pid in k73/queue.log> while it waits.
set -u
BL=$HOME/GEN-AI/backlog; RP=${RP:?r3 runner pid}
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" >> "$BL/k73/queue.log"; }
final() { head -1 "$BL/$1/STATE" 2>/dev/null | grep -qE '^(DONE|FAILED|SKIPPED|STOPPED)'; }
log "k73 queue pid $$ waiting for k56c and the r3 runner (pid $RP)"
echo "queued: after_k56c.sh pid $$ waits for the r3 chain (runner $RP: r3data r3train k56c), then runner.sh JOBS=k73 GRACE0=60" > "$BL/k73/STATE"
t0=$(date +%s)
while kill -0 "$RP" 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -gt 259200 ] && { log "gave up after 72 h"; exit 1; }
  sleep 60; done
if ! final k56c; then log "r3 runner $RP gone, k56c STATE '$(head -1 "$BL/k56c/STATE" 2>/dev/null)': not starting k73"; exit 1; fi
head -1 "$BL/STATE" | grep -q '^STOPPED' && { log "runner STATE '$(head -1 "$BL/STATE")': not starting k73"; exit 1; }
log "k56c: $(head -1 "$BL/k56c/STATE" | cut -c1-160); starting the runner"
JOBS=k73 GRACE0=60 setsid nohup bash "$BL/runner.sh" > "$BL/runner-k73.nohup" 2>&1 < /dev/null &
log "runner pid $! (JOBS=k73 GRACE0=60)"
