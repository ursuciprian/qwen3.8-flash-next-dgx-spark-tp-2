#!/usr/bin/env bash
# TP=1 16k-c8 KV screen (2026-10-03): v2 control vs tp1-v2-kv8 / tp1-v2-kv10, both Sparks in parallel.
# Launch detached on dgx-01 from the repo root:
#   RESULTS=$PWD/results/tp1-kv16k-20261003 setsid nohup bash scripts/tp1_kv16k_screen.sh \
#     > results/tp1-kv16k-20261003.nohup 2>&1 < /dev/null &
# Wait phase (no GPU): until the night chain, k18, k18 screen and ab_b15 launchers have exited (cap 36 h, then
# FAILED), the heads screen STATE says DONE|FAILED (cap 12 h, missing file = keep waiting; after the cap, run
# anyway), then ~/GEN-AI/gpu-lock. The run phase re-execs itself under a hard timeout.
# Run phase, three passes, one boot per node per pass (passes are barriers):
#   p1 dgx-01 kv10 COLD, dgx-02 kv8 COLD  (runtime cache moved aside, plan seed copied = a user's first boot)
#   p2 dgx-01 ctrl,      dgx-02 ctrl
#   p3 dgx-01 kv8,       dgx-02 kv10
# Each boot: MemAvailable/Cached every 5 s; below 2 GiB the node's container is removed and the cell is
# FAILED: mem-abort. Then llama-benchy task grid d0/16k c1 c2 c4 c8 (T=1.0, 3 runs; same as tp1-v2b) with
# preemption and prefix-cache counters. EXIT: original runtime caches back, shipped TP=2 recipe booted,
# image + health + pong verified. report: python3 scripts/tp1_kv16k_report.py $RESULTS
set -u
: "${RESULTS:?}"
export PATH="$HOME/.local/bin:$PATH"
REPO=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
H1=192.168.100.62; H2=192.168.100.53; MODEL=qwen3.8-flash-next
HEADS_STATE=${HEADS_STATE:-$HOME/GEN-AI/q-r10-heads4/STATE}
RC=$HOME/.cache/sparkrun/runtime-cache/vllm/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-2d9615ab
TOK=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
HERE=$(cd "$(dirname "$0")/.." && pwd); ARMS=$HERE/archive/recipes/arms/tp1
declare -A RECIPE=([ctrl]=$HERE/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml
                   [kv8]=$ARMS/tp1-v2-kv8.yaml [kv10]=$ARMS/tp1-v2-kv10.yaml)
MIN_KB=$((2 * 1024 * 1024))
mkdir -p "$RESULTS"; LOG=$RESULTS/driver.log
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$LOG"; }
st() { echo "$*" > "$RESULTS/STATE"; log "STATE: $*"; }

if [ "${1:-}" != --run ]; then
  st waiting
  t0=$(date +%s); hcap=0
  # pgrep only reads; these are the launchers queued before this one (k19 is HELD and ignored).
  while :; do
    el=$(( $(date +%s) - t0 ))
    [ $el -gt 129600 ] && { st "FAILED: earlier launchers still alive after 36 h"; exit 1; }
    hd=0; grep -qE '^(DONE|FAILED)' "$HEADS_STATE" 2>/dev/null && hd=1
    [ $hd = 0 ] && [ $el -gt 43200 ] && [ $hcap = 0 ] && { hcap=1; log "heads STATE wait hit the 12 h cap: not waiting for it any more"; }
    if ! pgrep -f 'night_chain\.sh|run_k18(_screen)?\.sh|ab_b15\.sh' >/dev/null && { [ $hd = 1 ] || [ $hcap = 1 ]; }; then break; fi
    sleep 60; done
  log "earlier launchers gone; heads: $(head -1 "$HEADS_STATE" 2>/dev/null || echo missing)"
  log "taking gpu-lock"; exec 9>"$HOME/GEN-AI/gpu-lock"; flock 9; log "gpu-lock held"
  timeout -k 7200 21600 bash "$0" --run; rc=$?
  log "run phase exit=$rc state=$(cat "$RESULTS/STATE")"
  [ $rc -eq 124 ] && st "FAILED: run phase timeout (6 h)"
  exit $rc
fi

health() { curl -s -m 5 -o /dev/null -w '%{http_code}' "$1:8000/health"; }
wait_health() { local s=$(date +%s); until [ "$(health "$1")" = 200 ]; do
  [ -e "$3" ] && return 1; [ $(( $(date +%s) - s )) -gt "$2" ] && return 1; sleep 15; done
  log "$1 health after $(( $(date +%s) - s ))s"; }
pong() { curl -s -m 120 "$1:8000/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Reply with exactly one word: pong"}],"max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>&1; }
on() { if [ "$1" = $H1 ]; then shift; bash -c "$*"; else local h=$1; shift; ssh -n "$h" "$*"; fi; }
serve_log() { local c; c=$(on "$1" "docker ps -a --format '{{.Names}}' | grep -E '_solo|sparkrun' | head -1")
  [ -n "$c" ] && on "$1" "docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1" > "$2" 2>&1; }
rm_containers() { on "$1" "docker ps -q --filter name=sparkrun | xargs -r docker rm -f" >/dev/null 2>&1; }
counters() { curl -s -m 10 "$1:8000/metrics" | grep -E '^vllm:(num_preemptions|prefix_cache_(hits|queries)|spec_decode_num_(accepted|draft)_tokens(_per_pos)?)(_total)?' > "$2"; }
PIDS=; MOVED=
memwatch() {  # host outdir cellpid: MemAvailable/Cached every 5 s; below 2 GiB remove the container and flag
  local a c
  while kill -0 "$3" 2>/dev/null && [ ! -e "$2/END" ]; do
    read -r a c < <(on "$1" "awk '/^MemAvailable/{a=\$2}/^Cached:/{c=\$2}END{print a, c}' /proc/meminfo")
    if [ -n "$a" ]; then echo "$(date +%T) $(awk -v a="$a" -v c="$c" 'BEGIN{printf "%.2f %.2f", a/1048576, c/1048576}')" >> "$2/mem.log"
      if [ "$a" -lt $MIN_KB ] && [ ! -e "$2/ABORT" ]; then echo "MemAvailable $a kB" > "$2/ABORT"; log "$1 MEM ABORT: MemAvailable $a kB"; rm_containers "$1"; fi; fi
    sleep 5; done; }
cell() {  # host arm pass cold(0/1)
  local h=$1 arm=$2 out=$RESULTS/$2-p$3-$( [ "$1" = $H1 ] && echo dgx01 || echo dgx02 ) s
  mkdir -p "$out"; cp "${RECIPE[$arm]}" "$out/recipe.yaml"
  memwatch "$h" "$out" $BASHPID &
  s=$(date +%s)
  ( cd "$out" && flock "$RESULTS/.sparkrun.lock" sparkrun run "$out/recipe.yaml" --hosts "$h" --solo --no-follow >> "$out/sparkrun.log" 2>&1 )
  if ! wait_health "$h" $(( $4 ? 3600 : 1800 )) "$out/ABORT"; then
    serve_log "$h" "$out/serve.log"; touch "$out/END"
    echo "FAILED: $( [ -e "$out/ABORT" ] && echo mem-abort || echo boot )" > "$out/STATE"; log "$out $(cat "$out/STATE")"; return; fi
  echo "boot $(( $(date +%s) - s ))s cold=$4 min MemAvailable $(sort -k2 -n "$out/mem.log" | head -1)" > "$out/boot.txt"; log "$out $(cat "$out/boot.txt")"
  log "$out pong=$(pong "$h")"; counters "$h" "$out/counters.before"
  ( cd "$REPO" && timeout 4500 uvx --from "$HOME/GEN-AI/llama-benchy-fork" llama-benchy --base-url "http://$h:8000/v1" --model $MODEL --tokenizer "$TOK" \
      --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 16384 --concurrency 1 2 4 8 --runs 3 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$h:8000/metrics" \
      --save-result "$out/task.csv" > "$out/task_grid.log" 2>&1 ); local rc=$?
  counters "$h" "$out/counters.after"; serve_log "$h" "$out/serve.log"; touch "$out/END"
  if [ -e "$out/ABORT" ]; then echo "FAILED: mem-abort" > "$out/STATE"; else echo "DONE benchy=$rc" > "$out/STATE"; fi
  log "$out $(cat "$out/STATE"); min MemAvailable $(sort -k2 -n "$out/mem.log" | head -1)"; }
stop_all() { sparkrun stop --all >>"$LOG" 2>&1; sleep 10; rm_containers $H1; rm_containers $H2; }
cache_cold() { on "$1" "[ ! -e '$RC.kv16k-keep' ] && rm -rf '$RC.kv16k-cold' && mv '$RC' '$RC.kv16k-keep' && mkdir -p '$RC'" && MOVED="$MOVED $1" && log "$1 runtime cache moved aside"; }
FINAL="FAILED: screen"
restore() {
  for p in $PIDS; do kill "$p" 2>/dev/null; done
  st restore; stop_all
  for h in $MOVED; do on "$h" "mv '$RC' '$RC.kv16k-cold' && mv '$RC.kv16k-keep' '$RC'" && log "$h runtime cache restored (cold copy at $RC.kv16k-cold)"; done
  sparkrun run qwen3.8-flash-next-2x-dgx-spark --no-follow >>"$LOG" 2>&1
  local exp p img0 img1
  exp=$(awk '/^container:/{print $2; exit}' "$(grep -rl '^name: qwen3.8-flash-next-2x-dgx-spark$' "$HOME/.cache/sparkrun/registries" | head -1)")
  wait_health localhost 3600 /nonexistent; p=$(pong localhost)
  img0=$(docker ps --format '{{.Image}}' | grep spark-vllm); img1=$(ssh -n $H2 "docker ps --format '{{.Image}}'" | grep spark-vllm)
  log "restored pong=$p head=$img0 worker=$img1 expect=$exp"
  if [ -n "$exp" ] && [ "$img0" = "$exp" ] && [ "$img1" = "$exp" ] && [[ "$p" == *[Pp]ong* ]]; then st "$FINAL"
  else log "FAILED FAILED: restore -- MANUAL INTERVENTION"; st "FAILED: restore"; fi; }
trap restore EXIT
trap 'exit 143' TERM INT

st running; stop_all
IMG=$(awk '/^container:/{print $2}' "${RECIPE[ctrl]}")
for h in $H1 $H2; do on "$h" "docker image inspect $IMG >/dev/null 2>&1 || docker pull -q $IMG" >> "$LOG" 2>&1 || { log "$h has no $IMG"; exit 1; }; done
pass() {  # n cold armH1 armH2
  st "pass $1: dgx-01 $3, dgx-02 $4"
  [ "$2" = 1 ] && { cache_cold $H1 || exit 1; cache_cold $H2 || exit 1; }
  cell $H1 "$3" "$1" "$2" & local a=$!; cell $H2 "$4" "$1" "$2" & local b=$!; PIDS="$a $b"
  wait $a $b; PIDS=; stop_all; }
pass 1 1 kv10 kv8
pass 2 0 ctrl ctrl
pass 3 0 kv8 kv10
FINAL=DONE
python3 "$HERE/scripts/tp1_kv16k_report.py" "$RESULTS" > "$RESULTS/report.txt" 2>&1
exit 0
