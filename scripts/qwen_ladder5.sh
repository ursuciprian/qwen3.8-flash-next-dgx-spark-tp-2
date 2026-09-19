#!/usr/bin/env bash
# qwen_ladder.sh <recipe.yaml>...
# Fresh boot per recipe, then the two reference harnesses plus a tool-calling gate:
#   T1  llama-benchy in MiaAI-Lab's "standard spec": pp 128/4096 x tg 256/1024, depth 0/32768, c1/c4
#   TONY tonyd2wild's 40-prompt category harness, C1/2/4/6, cold prefill ladder, counting ceiling
#   T2  tool-eval-bench --short (SeraphimSerapis), thinking off
# Results under results/qwen-ladder/<NN-name>-*. One line per lane in ladder.log.
export PATH="$HOME/.local/bin:$PATH"
R="$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2"; cd "$R" || exit 1
OUT="${LADDER_OUT:-$R/results/qwen-ladder}"; mkdir -p "$OUT"; LOG="$OUT/ladder.log"
BASE=http://192.168.100.62:8000; WORKER=192.168.100.53
LANES="${LANES:-T1 TONY T2}"
say() { echo "=== $(date +%T) $*" | tee -a "$LOG"; }
spec() { curl -s -m 6 "$BASE/metrics" | grep -E '^(vllm|sglang):spec_decode_num_(accepted_tokens|drafts)_total' | awk '{s[$1]+=$2} END{for(k in s) print k, s[k]}'; }

boot() {  # boot <recipe> <name>: 0 healthy, 1 failed, 2 retry-worthy
  sparkrun stop --all >/dev/null 2>&1; sleep 5
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  ssh "$WORKER" 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null'
  sparkrun run "$1" --cluster dgx-cluster-cx7 --tp 2 --trust --no-follow > "$OUT/$2-sparkrun.log" 2>&1 \
    || { say "$2 sparkrun exit"; grep -aE "failed|Error" "$OUT/$2-sparkrun.log" | tail -2 | tee -a "$LOG"; return 1; }
  # 90x20s was not enough: the shipped image needed 29 min from a dropped page cache, and a
  # first boot of a new image also pays cold Triton/torch JIT. 180x20s = 60 min.
  for _ in $(seq 1 180); do
    [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' $BASE/health)" = 200 ] && return 0
    C=$(docker ps -q | head -1); [ -z "$C" ] && break
    # SGLang writes to /tmp/sparkrun_serve.log inside the container; docker logs only has the CUDA
    # banner, so reading it meant every failure waited out the whole window unexplained.
    docker exec "$C" sh -c 'cat /tmp/sparkrun_serve.log 2>/dev/null' > "$OUT/$2-serve.log" 2>/dev/null
    [ -s "$OUT/$2-serve.log" ] || docker logs "$C" > "$OUT/$2-serve.log" 2>&1
    grep -aqE "Scheduler hit an exception|Engine core initialization failed|WorkerProc failed to start" "$OUT/$2-serve.log" && break
    sleep 20
  done
  C=$(docker ps -q | head -1)
  if [ -n "$C" ]; then
    docker exec "$C" sh -c 'cat /tmp/sparkrun_serve.log 2>/dev/null' > "$OUT/$2-serve.log" 2>/dev/null
    [ -s "$OUT/$2-serve.log" ] || docker logs "$C" > "$OUT/$2-serve.log" 2>&1
  fi
  # the head only ever sees "1/2 clients joined"; the reason is in the worker's own log
  ssh "$WORKER" 'C=$(docker ps -q | head -1); [ -n "$C" ] && docker exec "$C" sh -c "cat /tmp/sparkrun_serve.log 2>/dev/null"' > "$OUT/$2-worker-serve.log" 2>/dev/null
  grep -aqE "Timed out after [0-9]+ seconds waiting for clients|FileNotFoundError.*No such file" "$OUT/$2-serve.log" 2>/dev/null && return 2
  return 1
}

i=0
for rec in "$@"; do
  i=$((i+1)); name=$(printf "%02d-%s" "$i" "$(basename "$rec" .yaml)"); say "$name start"
  boot "$rec" "$name"; rc=$?
  [ $rc = 2 ] && { say "$name rendezvous race, retrying once"; boot "$rec" "$name"; rc=$?; }
  if [ $rc != 0 ]; then say "$name never healthy"; grep -aE "Error|error:|ValueError|Scheduler hit an exception" "$OUT/$name-serve.log" 2>/dev/null | grep -viE "use_fast|deprecated|Ignore import error" | tail -2 | cut -c1-200 | tee -a "$LOG"; sparkrun stop --all >/dev/null 2>&1; continue; fi
  C=$(docker ps -q | head -1)
  docker exec "$C" sh -c 'cat /tmp/sparkrun_serve.log 2>/dev/null' > "$OUT/$name-serve.log" 2>/dev/null
  [ -s "$OUT/$name-serve.log" ] || docker logs "$C" > "$OUT/$name-serve.log" 2>&1
  kv=$(grep -aoE "GPU KV cache size: [0-9,]+ tokens|KV Cache is allocated[^\n]{0,60}|max_total_num_tokens=[0-9]+" "$OUT/$name-serve.log" | head -1)
  mem="avail head=$(free -g | awk '/^Mem:/{print $7}')G worker=$(ssh "$WORKER" "free -g | awk '/^Mem:/{print \$7}'")G"
  M=$(curl -s -m 6 $BASE/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
  model=$(grep -E "^model:" "$rec" | awk '{print $2}'); TOK=$(ls -d ~/.cache/huggingface/hub/models--${model//\//--}/snapshots/*/ | head -1)
  say "$name healthy | $kv | $mem | model=$M"

  if [[ " $LANES " == *" T1 "* ]]; then
  # T1: MiaAI standard spec. return_token_ids=false is required on SGLang: llama-benchy sets it
  # true by default and the server rejects it with streaming ("not supported with streaming on
  # /v1/chat/completions"). Thinking-off kwarg differs per engine, so pick both from the recipe.
  runtime=$(grep -E "^runtime:" "$rec" | awk '{print $2}')
  # llama-benchy --extra-body takes key=value / key:value pairs, NOT a JSON object: a JSON string
  # is accepted silently and sets nothing. return_token_ids=false is required on SGLang, which
  # rejects llama-benchy's default (true) whenever the request streams.
  if [ "$runtime" = "sglang" ]; then
    EXTRA_ARGS=(--extra-body return_token_ids=false --extra-body 'chat_template_kwargs={"enable_thinking":false}')
  else
    EXTRA_ARGS=(--extra-body 'chat_template_kwargs={"enable_thinking":false}')
  fi
  spec > "$OUT/$name-t1-spec0.txt"
  uvx llama-benchy@0.4.0 --base-url "$BASE/v1" --model "$M" --tokenizer "$TOK" \
    --pp 128 4096 --tg 256 1024 --depth 0 32768 --concurrency 1 4 --runs 2 \
    "${EXTRA_ARGS[@]}" \
    --save-result "$OUT/$name-t1.csv" > "$OUT/$name-t1-benchy.log" 2>&1
  if [ "$(grep -c '^| ' "$OUT/$name-t1.csv" 2>/dev/null)" -lt 2 ]; then
    say "$name T1 produced no rows"; grep -aoE "HTTP [0-9]+: [^\"]{0,120}" "$OUT/$name-t1-benchy.log" | head -1 | tee -a "$LOG"
  fi
  spec > "$OUT/$name-t1-spec1.txt"
  say "$name T1 done"; grep -E "^\| " "$OUT/$name-t1.csv" | grep -E "tg(256|1024)" | tr -s " " | cut -d"|" -f3-5 | sed 's/^/   /' | tee -a "$LOG"

  fi
  if [[ " $LANES " == *" TONY "* ]]; then
  # TONY lane
  BASE="$BASE" MODEL="$M" "$R/tools/tony-bench/run_lane_sglang.sh" "qwen_${name}" > "$OUT/$name-tony.log" 2>&1
  say "$name TONY done"; grep -E "median|x[0-9] |prefill" "$OUT/$name-tony.log" | head -10 | sed 's/^/   /' | tee -a "$LOG"

  fi
  if [[ " $LANES " == *" T2 "* ]]; then
  # T2: tool-eval-bench short
  ( cd "$OUT" && tool-eval-bench run --short --no-think --temperature 0.0 --backend vllm --timeout 300 --base-url "$BASE" --model "$M" > "$OUT/$name-t2.log" 2>&1 )
  say "$name T2 done"; grep -iE "final score|score:|passed|failed" "$OUT/$name-t2.log" | tail -3 | sed 's/^/   /' | tee -a "$LOG"

  fi
  [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' $BASE/health)" = 200 ] || say "!!! $name unhealthy after lanes"
  say "$name done"; sparkrun stop --all >/dev/null 2>&1
done
say LADDER_DONE
