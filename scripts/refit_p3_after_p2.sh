#!/usr/bin/env bash
# refit-p3 queue (#97, 2026-10-06): start refit-p3/job.sh as soon as refit-p2 (job.sh --resume-cap under after_k55.sh) is
# over and passed, so the pair is not idle overnight.
#   1. wait  until no refit-p2 job.sh/after_k55.sh process is left and refit-p2/STATE is DONE/FAILED. No process for
#            30 min with STATE still running/waiting -> "skipped: refit-p2 ended without a final state". Up to 12 h.
#   2. gate  refit-p2/STATE must start "DONE: parity drafts=PASS live: t0=PASS t1=PASS"; otherwise STATE
#            "skipped: parity ..." and nothing runs (no lock, the pair untouched).
#   3. lock  flock -o ~/GEN-AI/gpu-lock (wait up to 12 h), then job.sh with GPU_LOCK_HELD=1 (job.sh re-checks the gate).
# Run:  setsid nohup bash ~/GEN-AI/refit-p3/after_p2.sh > ~/GEN-AI/refit-p3/after_p2.nohup 2>&1 < /dev/null &
# Stop: while waiting, kill -TERM <queue pid in after_p2.log>; once job.sh runs, kill -TERM its pid ("refit-p3 pid" in
#   refit-p3.log of the results dir). Never pkill -f.
set -u
P=$HOME/GEN-AI/refit-p3; P2=$HOME/GEN-AI/refit-p2
PARITY_RE='^DONE: parity drafts=PASS live: t0=PASS t1=PASS'
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" >> "$P/after_p2.log"; }
st() { echo "$*" > "$P/STATE"; log "STATE: $*"; }
p2_alive() { pgrep -f "bash $P2/job.sh" >/dev/null || pgrep -fx "bash after_k55.sh" >/dev/null \
  || pgrep -fx "bash $P2/after_k55.sh" >/dev/null; }
s2() { head -1 "$P2/STATE" 2>/dev/null; }
log "queue pid $$"
st "waiting: refit-p2 ($(s2))"; gone=; t0=$(date +%s)
while :; do
  if p2_alive; then gone=
  elif s2 | grep -qE '^(DONE|FAILED)'; then break
  else [ -z "$gone" ] && gone=$(date +%s)
    [ $(( $(date +%s) - gone )) -gt 1800 ] && { st "skipped: refit-p2 ended without a final state ('$(s2)'); nothing run"; exit 1; }; fi
  [ $(( $(date +%s) - t0 )) -gt 43200 ] && { st "FAILED: refit-p2 still running after 12 h ('$(s2)'); nothing run"; exit 1; }
  sleep 60; done
log "refit-p2 finished: $(s2)"
s2 | grep -qE "$PARITY_RE" || { st "skipped: parity not all PASS (refit-p2: $(s2)); nothing run"; exit 0; }
st "waiting: gpu-lock"
flock -w 43200 -o "$HOME/GEN-AI/gpu-lock" env GPU_LOCK_HELD=1 bash "$P/job.sh"; rc=$?
grep -q '^waiting: gpu-lock' "$P/STATE" && st "FAILED: gpu-lock busy for 12 h (flock exit $rc); nothing run"
log "job.sh / flock exit $rc; STATE: $(head -1 "$P/STATE")"
