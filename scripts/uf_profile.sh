#!/usr/bin/env bash
# Torch profile of the dime-online v16b single-Spark stack (opus-kernel-17, 2026-10-02).
# Run detached on dgx-01:
#   setsid nohup bash scripts/uf_profile.sh <STATE file to wait for> > ~/GEN-AI/uf-prof.nohup 2>&1 < /dev/null &
# Waits for <STATE> DONE/FAILED/SKIPPED, takes ~/GEN-AI/gpu-lock, stops the TP=2 server, boots their
# stack on dgx-02 from the copy ~/GEN-AI/uf-test/v16b-prof (their v16b dir, env EXTRA adds
# --profiler-config torch, /tmp/uf-prof, max_iterations 60; their repo untouched), then profiles
# the r8-prof windows at T=0 (fresh c1, 16k c1, fresh c4; same probe offsets) via /start_profile +
# /stop_profile, breaks each trace down with scripts/step_breakdown.py. EXIT runs
# ~/GEN-AI/uf-test/99_restore.sh (removes their container, boots b1.4, checks image + pong).
set -u
WAIT=${1:?usage: uf_profile.sh <STATE file>}
export PATH="$HOME/.local/bin:$PATH"
REPO=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
H2=192.168.100.53; BASE=http://$H2:8000
RESULTS=${RESULTS:-$REPO/results/uf-prof-$(date +%Y%m%d-%H%M)}
WINDOWS=${WINDOWS:-"fresh-c1:256:1:1 d16k-c1:16384:1:4 fresh-c4:256:4:3"}
mkdir -p "$RESULTS"; LOG=$RESULTS/driver.log
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$LOG"; }
set_state() { echo "$*" > "$RESULTS/STATE"; log "STATE: $*"; }
set_state waiting; log "waiting for $WAIT"
until grep -qE '^(DONE|FAILED|SKIPPED)' "$WAIT" 2>/dev/null; do sleep 60; done
log "wait over: $(head -1 "$WAIT")"
log "taking gpu-lock"; exec 9>"$HOME/GEN-AI/gpu-lock"; flock 9; log "gpu-lock held"
FINAL="FAILED: profile"
restore() {
  set_state restore
  ssh $H2 'docker logs qwen38-flash' > "$RESULTS/serve.log" 2>&1
  if bash "$HOME/GEN-AI/uf-test/99_restore.sh" >> "$LOG" 2>&1; then set_state "$FINAL"
  else log "FAILED FAILED: restore -- MANUAL INTERVENTION"; set_state "FAILED: restore"; fi; }
trap restore EXIT
set_state running
sparkrun stop --all >> "$LOG" 2>&1
for i in $(seq 1 60); do [ -z "$(docker ps -q)" ] && [ -z "$(ssh $H2 docker ps -q)" ] && break; sleep 5; done
[ -z "$(ssh $H2 docker ps -q)" ] || { log "dgx-02 still has containers"; exit 1; }
for i in $(seq 1 24); do a=$(ssh $H2 "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo"); [ "${a:-0}" -ge 100 ] && break; sleep 5; done
log "dgx-02 MemAvailable ${a:-?} GiB; booting v16b-prof"
ssh $H2 'cd ~/GEN-AI/uf-test/v16b-prof && bash launch.sh --run' >> "$LOG" 2>&1 || { log "launch failed"; exit 1; }
s=$(date +%s)
until curl -fsS -m 5 $BASE/v1/models > /dev/null 2>&1; do
  [ -n "$(ssh $H2 'docker ps -q --filter name=^qwen38-flash$ --filter status=running')" ] || { log "their container exited"; exit 1; }
  [ $(( $(date +%s) - s )) -lt 1800 ] || { log "not ready after 30 min"; exit 1; }; sleep 15; done
log "READY after $(( $(date +%s) - s ))s"
probe() { # tag depth conc repeats offset [extra]
  local tag=$1 depth=$2 c=$3 r=$4 off=$5; shift 5
  curl -s -m 10 $BASE/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens' > "$RESULTS/pos-$tag.before"
  ( cd "$REPO" && python3 scripts/depth_decode_probe.py --base $BASE --model qwen --label "$tag" --corpus results/corpus-code.txt \
      --depth "$depth" --new 2048 --concurrency "$c" --repeats "$r" --offset "$off" --max-tokens 512 --temperature 0 \
      --out "$RESULTS/probe-$tag.json" "$@" > "$RESULTS/probe-$tag.log" 2>&1 )
  log "probe $tag exit=$?"
  curl -s -m 10 $BASE/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens' > "$RESULTS/pos-$tag.after"; }
probe warm 4096 1 1 900000
ok=0
for spec in $WINDOWS; do IFS=: read -r tag depth c k <<< "$spec"
  n0=$(ssh $H2 "docker exec qwen38-flash sh -c 'ls /tmp/uf-prof 2>/dev/null | grep -c trace'")
  probe "$tag" "$depth" "$c" 2 $((k*211111)) --trigger-cmd "curl -s -m 30 -X POST $BASE/start_profile" --trigger-repeat 1 --trigger-after-chunks 40
  curl -s -m 600 -X POST $BASE/stop_profile >> "$LOG" 2>&1
  for i in $(seq 1 60); do n=$(ssh $H2 "docker exec qwen38-flash sh -c 'ls /tmp/uf-prof 2>/dev/null | grep -c trace'"); [ "${n:-0}" -gt "${n0:-0}" ] && break; sleep 10; done
  sleep 20  # let the trace handler finish writing
  f=$(ssh $H2 "docker exec qwen38-flash sh -c 'ls -t /tmp/uf-prof | grep trace | head -1'")
  [ -n "$f" ] && [ "${n:-0}" -gt "${n0:-0}" ] || { log "WARN: no new trace for $tag"; continue; }
  mkdir -p "$RESULTS/$tag"
  ssh $H2 "docker exec qwen38-flash cat '/tmp/uf-prof/$f'" > "$RESULTS/$tag/$f"
  python3 "$REPO/scripts/step_breakdown.py" 60 "$RESULTS/$tag/$f" --json "$RESULTS/$tag/breakdown.json" > "$RESULTS/$tag/breakdown.txt" 2>&1
  log "$tag: $f $(head -2 "$RESULTS/$tag/breakdown.txt" | tail -1)"; ok=$((ok+1))
done
[ "$ok" -gt 0 ] && FINAL=DONE
log "profiled $ok windows"
exit 0
