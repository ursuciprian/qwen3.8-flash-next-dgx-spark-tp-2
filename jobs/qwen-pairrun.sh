#!/usr/bin/env bash
# qwen-pairrun.sh (2026-11-14) on dgx-01: runs Qwen backlog jobs through the old backlog runner (runner.sh, gpu-lock,
# shipped-2x restore) while holding the pair's shared lock ~/GEN-AI/ds41/pair/PAIR.LOCK (convention 2026-10-20: before
# any GPU use the lock must be absent; the owner writes one line and removes it when done; never break another's lock).
#   wait     until no DS41 chain/queue script runs (the wd4.py pattern, also matching /bin/bash and /usr/bin/bash
#            launches), PAIR.LOCK is absent, pairlock's 4-min yield after a refit release is over, and the noclobber
#            write of our line succeeds (HELD is set in the same step, so a signal cannot strand our line)
#   check    every container on both Sparks is a sparkrun container on a spark-vllm-b12x image; anything else (another
#            image, a non-sparkrun name, or a Spark that cannot be listed) = release the lock and stop, nothing touched
#   run      JOBS="$JOBS" GRACE0=60 bash runner.sh in the foreground (it restores the shipped 2x after each job)
#   end      only when the runner was started: stop every sparkrun container on both Sparks (the pair is left idle,
#            as found); then remove our lock line. The lock stays while any of our jobs' run.sh still runs (TERM to
#            this script never orphans a job).
#   exit     0 = runner DONE and every job DONE/SKIPPED/FAILED without a restore failure; 1 = runner failed or never
#            started; 3 = a job or the runner ended STOPPED / MANUAL INTERVENTION (the pair needs a person)
# Start: JOBS="k122 k123 k120" setsid nohup bash ~/GEN-AI/backlog/qwen-pairrun.sh > ~/GEN-AI/backlog/qwen-pairrun.nohup 2>&1 < /dev/null &
# Dry:   JOBS="k122 k123 k120" DRY=1 bash ~/GEN-AI/backlog/qwen-pairrun.sh   (lock holder, queue ahead, each job's --dry-run)
# Stop:  kill -TERM <run.sh pid of the running job> (its EXIT trap restores), then kill -TERM <this pid>. Never pkill -f.
# Log:   ~/GEN-AI/backlog/qwen-pairrun.log
set -u
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/GEN-AI/backlog/lib.sh"   # BL R H1 H2 (addresses live in lib.sh on dgx-01, not in tracked files)
LK=$HOME/GEN-AI/ds41/pair/PAIR.LOCK
JOBS=${JOBS:?set JOBS, e.g. JOBS="k122 k123 k120"}; HOURS=${HOURS:-12}; DRY=${DRY:-0}; HELD=0; STARTED=0; RUNNER=
plog() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$BL/qwen-pairrun.log" >&2; }
others() { pgrep -af '^(/usr)?(/bin/)?bash .*[^ ]*(chain|q-|wait|after-|pb[0-9]-)[^ ]*\.sh' | grep -v qwen-pairrun; }   # ds41 tools/wd4.py
foreign() { # containers that are not sparkrun-on-spark-vllm-b12x, both Sparks; return 2 when a Spark cannot be listed
  local a b
  a=$(docker ps --format '{{.Names}} {{.Image}}' 2>/dev/null) || return 2
  b=$(ssh -n -o ConnectTimeout=10 -o BatchMode=yes $H2 "docker ps --format '{{.Names}} {{.Image}}'" 2>/dev/null) || return 2
  printf '%s\n%s\n' "$a" "$b" | awk 'NF && !($1 ~ /^sparkrun/ && $2 ~ /(^|\/)spark-vllm-b12x:/)'; }
ours_running() { for j in $JOBS; do pgrep -f "backlog/$j/run.sh" > /dev/null && return 0; done; return 1; }
take() { ( set -o noclobber; echo "qwen $(date '+%F %T') until $(date -d "now + $HOURS hours" '+%F %H:%M'): Qwen backlog JOBS=$JOBS (runner.sh, sparkrun) pid $$" > "$LK" ) 2>/dev/null && HELD=1; }
release() {
  [ $HELD = 1 ] || return 0
  [ -n "$RUNNER" ] && kill -TERM "$RUNNER" 2>/dev/null
  while ours_running; do plog "a job's run.sh still runs; lock kept until it exits"; sleep 60; done
  if [ $STARTED = 1 ]; then
    ( cd "$R" && timeout -k 30 600 sparkrun stop --all ) >> "$BL/qwen-pairrun.log" 2>&1
    for h in $H1 $H2; do timeout -k 30 300 sparkrun stop --all --hosts $h >> "$BL/qwen-pairrun.log" 2>&1; done; fi
  grep -q " pid $$\$" "$LK" 2>/dev/null && rm -f "$LK" && plog "PAIR.LOCK released; containers left on dgx-01: $(docker ps -q | wc -l)"
  HELD=0; }

if [ "$DRY" = 1 ]; then rc=0
  echo "PAIR.LOCK: $(cat "$LK" 2>/dev/null || echo free)"; echo "ds41 queue ahead: $(others | tr '\n' ' ')"
  f=$(foreign) || { echo "FAILED: cannot list the containers of a Spark (ssh/docker)"; rc=1; }
  echo "foreign containers (not sparkrun on spark-vllm-b12x): $(echo $f)"; [ -e "$BL/HOLD" ] && echo "note: $BL/HOLD exists (the runner pauses)"
  bash -n "$BL/runner.sh" || rc=1
  grep -q GRACE0 "$BL/runner.sh" || echo "note: runner.sh ignores GRACE0 (it waits 1200 s before the first job)"
  for j in $JOBS; do s=$(head -1 "$BL/$j/STATE" 2>/dev/null); [ -n "$s" ] && echo "$j STATE: $s (DONE/SKIPPED are not rerun)"
    bash "$BL/$j/run.sh" --dry-run > "$BL/$j/dryrun.txt" 2>&1; r=$?; echo "$j dry-run exit=$r ($BL/$j/dryrun.txt)"; tail -3 "$BL/$j/dryrun.txt" | sed 's/^/  /'
    [ $r = 0 ] || rc=1; done
  echo "pairrun dry-run exit=$rc"; exit $rc; fi

trap 'plog "signal"; exit 143' TERM INT HUP
trap 'release' EXIT
plog "pid $$ JOBS=$JOBS: waiting for the ds41 queue, a free PAIR.LOCK and pairlock's yield"
n=0
until [ -z "$(others)" ] && [ ! -e "$LK" ] && [ -z "$(find "$LK.refit-released" -mmin -4 2>/dev/null)" ] && take; do
  [ $((n % 30)) = 0 ] && plog "waiting: $(others | tr '\n' ' ') lock: $(cat "$LK" 2>/dev/null)"; n=$((n + 1)); sleep 60; done
plog "PAIR.LOCK: $(cat "$LK")"
f=$(foreign) || { plog "FAILED: cannot list the containers of a Spark (ssh/docker); nothing started"; exit 1; }
[ -z "$f" ] || { plog "FAILED: foreign containers up: $(echo $f); nothing started"; exit 1; }
STARTED=1
JOBS="$JOBS" GRACE0=60 bash "$BL/runner.sh" > "$BL/runner.nohup" 2>&1 < /dev/null & RUNNER=$!
wait $RUNNER; rc=$?; RUNNER=
rs=$(head -1 "$BL/STATE" 2>/dev/null); plog "runner exit=$rc: $rs"
x=0; [ $rc = 0 ] && [ "${rs%%:*}" = DONE ] || x=1
for j in $JOBS; do s=$(head -1 "$BL/$j/STATE" 2>/dev/null); plog "$j: $s"
  case "$s" in *STOPPED*|*"MANUAL INTERVENTION"*|*"FAILED: restore"*) x=3 ;; esac; done
case "$rs" in *STOPPED*|*"MANUAL INTERVENTION"*) x=3 ;; esac
exit $x
