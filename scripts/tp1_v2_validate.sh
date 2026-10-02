#!/usr/bin/env bash
# Single-Spark recipe v2 validation (opus-kernel-17, 2026-10-02). Run detached on dgx-01:
#   SHIPPED_IMAGE_EXPECT=<b1.4 tag> RECIPE=<abs path to the 1x recipe> RESULTS=<dir> \
#     setsid nohup bash scripts/tp1_v2_validate.sh <STATE file to wait for> > <log> 2>&1 < /dev/null &
# Two nodes in parallel, each running the same 1x recipe:
#   dgx-01: COLD boot (the model's sparkrun runtime cache moved aside, b12x seed copy disabled, so
#           every plan autotunes and every kernel compiles: the worst-case first boot), 5 s
#           MemAvailable log, then the gate: gate_arm.sh (straggler, fidelity 8k-128k, hardmode,
#           categories, decode probe, sweep) + TC-45 x5 + 128k fidelity seeds 11/13.
#   dgx-02: warm boot, llama-benchy task grid fresh/16k c1 c2 c4 c8 (T=1.0, 3 runs) and
#           bench_copy_streams.py (clients on dgx-01 over CX-7).
# EXIT: stops both, puts dgx-01's runtime cache back (the cold one is kept as <dir>.k17cold), boots
# the shipped TP=2 registry recipe and verifies image + health + pong. Holds ~/GEN-AI/gpu-lock.
set -u
WAIT=${1:?usage: tp1_v2_validate.sh <STATE file>}
: "${SHIPPED_IMAGE_EXPECT:?}" "${RECIPE:?}" "${RESULTS:?}"
export PATH="$HOME/.local/bin:$PATH"
REPO=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
H1=192.168.100.62; H2=192.168.100.53; MODEL=qwen3.8-flash-next
RC=$HOME/.cache/sparkrun/runtime-cache/vllm/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-2d9615ab
TOK=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
mkdir -p "$RESULTS"/{cold,gate,bench}; LOG=$RESULTS/driver.log
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$LOG"; }
st() { echo "$*" > "$RESULTS/STATE"; log "STATE: $*"; }
st waiting; log "waiting for $WAIT"
until grep -qE '^(DONE|FAILED|SKIPPED)' "$WAIT" 2>/dev/null; do sleep 60; done
log "wait over: $(head -1 "$WAIT")"
log "taking gpu-lock"; exec 9>"$HOME/GEN-AI/gpu-lock"; flock 9; log "gpu-lock held"
health() { curl -s -m 5 -o /dev/null -w '%{http_code}' "${1:-localhost}:8000/health"; }
wait_health() { local s=$(date +%s); until [ "$(health "$1")" = 200 ]; do
  [ $(( $(date +%s) - s )) -gt "$2" ] && return 1; sleep 15; done; log "$1 health after $(( $(date +%s) - s ))s"; }
pong() { curl -s -m 120 "${1:-localhost}:8000/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Reply with exactly one word: pong"}],"max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>&1; }
serve_log() { local c; c=$(ssh -n "$1" "docker ps -a --format '{{.Names}}' | grep -E '_solo|sparkrun' | head -1")
  [ -n "$c" ] && ssh -n "$1" "docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1" > "$2" 2>&1; }
MOVED=0; FINAL="FAILED: validate"; PIDS=
restore() {
  for p in $PIDS; do kill "$p" 2>/dev/null; done
  st restore; sparkrun stop --all >>"$LOG" 2>&1; sleep 10
  docker ps -q | xargs -r docker rm -f >/dev/null; ssh -n $H2 'docker ps -q | xargs -r docker rm -f' >/dev/null
  if [ "$MOVED" = 1 ]; then rm -rf "$RC.k17cold"; mv "$RC" "$RC.k17cold" && mv "$RC.k17keep" "$RC" && log "runtime cache restored (cold copy at $RC.k17cold)"; fi
  sparkrun run qwen3.8-flash-next-2x-dgx-spark --no-follow >>"$LOG" 2>&1
  wait_health localhost 3600; p=$(pong localhost)
  img0=$(docker ps --format '{{.Image}}' | grep spark-vllm); img1=$(ssh -n $H2 "docker ps --format '{{.Image}}'" | grep spark-vllm)
  log "restored pong=$p head=$img0 worker=$img1"
  case "$img0|$img1|$p" in *"$SHIPPED_IMAGE_EXPECT"*"|"*"$SHIPPED_IMAGE_EXPECT"*"|"*[Pp]ong*) st "$FINAL" ;;
    *) log "FAILED FAILED: restore -- MANUAL INTERVENTION"; st "FAILED: restore" ;; esac; }
trap restore EXIT
st running; sparkrun stop --all >>"$LOG" 2>&1; sleep 15
IMG=$(awk '/^container:/{print $2}' "$RECIPE")
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  # The CI warm build runs on this node's runner and refuses while :8000 serves: dispatch it now.
  st ci-window; log "waiting up to 40 min for $IMG (dispatch build-b0-warm now)"
  for i in $(seq 1 80); do docker image inspect "$IMG" >/dev/null 2>&1 && break; sleep 30; done
  docker image inspect "$IMG" >/dev/null 2>&1 || { log "image $IMG never appeared"; exit 1; }
  sleep 120; st running   # let the workflow finish its push
fi
for i in $(seq 1 40); do ssh -n $H2 "docker image inspect $IMG >/dev/null 2>&1 || docker pull -q $IMG" >> "$LOG" 2>&1 && break; sleep 30; done
ssh -n $H2 "docker image inspect $IMG >/dev/null 2>&1" || { log "dgx-02 cannot get $IMG"; exit 1; }
cp "$RECIPE" "$RESULTS/recipe.yaml"
sed 's|S=/opt/b12x-seed/preparation;|S=/nonexistent-k17-cold;|' "$RECIPE" > "$RESULTS/cold/recipe-noseed.yaml"
grep -q nonexistent-k17-cold "$RESULTS/cold/recipe-noseed.yaml" || { log "seed line not found in recipe"; exit 1; }

# ---- dgx-02: warm boot + benchmarks (background) ----
bench() {
  ( cd "$(dirname "$RECIPE")" && flock "$RESULTS/.sparkrun.lock" sparkrun run "$RECIPE" --hosts $H2 --solo --no-follow >> "$RESULTS/bench/sparkrun.log" 2>&1 )
  wait_health $H2 5400 || { log "dgx-02 boot failed"; serve_log $H2 "$RESULTS/bench/serve.log"; echo FAILED > "$RESULTS/bench/STATE"; return; }
  log "dgx-02 pong=$(pong $H2)"
  ( cd "$HOME/GEN-AI/ultrafast-bench" && python3 bench_copy_streams.py --base http://$H2:8000 --model $MODEL --min-mem-gib 4 \
      --out "$RESULTS/bench/copy-streams.json" > "$RESULTS/bench/copy-streams.log" 2>&1 ); log "copy-streams exit=$?"
  curl -s -m 10 http://$H2:8000/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens(_per_pos)?_total' > "$RESULTS/bench/pos.before"
  ( cd "$REPO" && uvx --from "$HOME/GEN-AI/llama-benchy-fork" llama-benchy --base-url http://$H2:8000/v1 --model $MODEL --tokenizer "$TOK" \
      --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 16384 --concurrency 1 2 4 8 --runs 3 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url http://$H2:8000/metrics \
      --save-result "$RESULTS/bench/task.csv" > "$RESULTS/bench/task_grid.log" 2>&1 ); log "benchy exit=$?"
  curl -s -m 10 http://$H2:8000/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens(_per_pos)?_total' > "$RESULTS/bench/pos.after"
  serve_log $H2 "$RESULTS/bench/serve.log"; echo DONE > "$RESULTS/bench/STATE"; }
bench & PIDS="$PIDS $!"

# ---- dgx-01: cold boot + gate ----
mv "$RC" "$RC.k17keep" && mkdir -p "$RC" && MOVED=1 && log "runtime cache moved aside (cold boot)"
( while true; do echo "$(date +%T) $(awk '/MemAvailable/{printf "%.2f", $2/1048576}' /proc/meminfo)"; sleep 5; done > "$RESULTS/cold/mem.log" ) & PIDS="$PIDS $!"
s=$(date +%s)
( cd "$RESULTS/cold" && flock "$RESULTS/.sparkrun.lock" sparkrun run "$RESULTS/cold/recipe-noseed.yaml" --hosts $H1 --solo --no-follow >> "$RESULTS/cold/sparkrun.log" 2>&1 )
if wait_health localhost 5400; then
  log "cold boot $(( $(date +%s) - s ))s, min MemAvailable $(sort -k2 -n "$RESULTS/cold/mem.log" | head -1)"
  serve_log $H1 "$RESULTS/cold/serve.log"
  c=$(docker ps --format '{{.Names}}' | grep -E '_solo|sparkrun' | head -1)
  docker exec "$c" sh -c "ps -eo rss,cmd | grep -c '[c]ompile'" > "$RESULTS/cold/compile-procs.txt" 2>&1
  log "cold pong=$(pong localhost)"
  st gate; ( cd "$REPO" && bash scripts/gate_arm.sh tp1-v2-k17 > "$RESULTS/gate/gate_arm.log" 2>&1 )
  cp "$REPO"/results/arms/tp1-v2-k17/* "$RESULTS/gate/" 2>/dev/null
  st tc45; tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 \
    --base-url http://localhost:8000/v1 --model $MODEL --scenarios TC-45 --trials 5 > "$RESULTS/gate/tc45.log" 2>&1
  st fid; for sd in 11 13; do ( cd "$REPO" && python3 scripts/fidelity_probe.py --base http://localhost:8000 --model $MODEL \
    --depths 128000 --seed $sd --out "$RESULTS/gate/fidelity-seed$sd.json" > "$RESULTS/gate/fidelity-seed$sd.txt" 2>&1 ); done
  serve_log $H1 "$RESULTS/gate/serve.log"; log "gate done; min MemAvailable over the whole run $(sort -k2 -n "$RESULTS/cold/mem.log" | head -1)"
  FINAL=DONE
else
  log "cold boot FAILED after $(( $(date +%s) - s ))s; min MemAvailable $(sort -k2 -n "$RESULTS/cold/mem.log" | head -1)"
  serve_log $H1 "$RESULTS/cold/serve.log"
fi
st "wait-bench"; until [ -f "$RESULTS/bench/STATE" ]; do sleep 30; done
log "bench: $(cat "$RESULTS/bench/STATE")"
exit 0
