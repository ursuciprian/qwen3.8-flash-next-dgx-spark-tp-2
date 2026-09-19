#!/usr/bin/env bash
# Improvement ladder for the vLLM lane (Qwen3.8-Flash-Next nvidia/NVFP4, TP=2).
# Fresh boot per arm; per arm: prefix-caching correctness gate -> benchy grid ->
# tonyd2wild harness lane -> vLLM spec counters. The control (Tony SPEED) is NOT
# rerun: it exists twice (results/arms/fn-vllm-tony-speed*, results/codex-campaigns).
# An arm that fails the gate is stopped and not benchmarked. If the first arm that
# turns prefix caching on fails the gate, the remaining arms are flipped back to
# --no-enable-prefix-caching (in place, logged) so the other axes still get measured.
#   scripts/vllm_ladder.sh fn-vllm-tony-speed-pc fn-vllm-tony-speed-pc-idx fn-vllm-imp-k2 ...
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEAD=${HEAD_IP:-192.168.100.62}; WORKER=${WORKER_IP:-192.168.100.53}; CLUSTER=${CLUSTER:-dgx-cluster-cx7}
DEPTHS=${DEPTHS:-"0 2048 4096 8192 16384 32768"}; CONC=${CONC:-"1 2 5"}
CORPUS=${CORPUS:-$REPO/codex-optimization/campaigns/20260907T154810Z/a1-c1/corpus.txt}
TOK=$(ls -d "$HOME"/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54*/ | head -1)
OUT="$REPO/results/arms/vllm-imp"; mkdir -p "$OUT"
BASE="http://$HEAD:8000"
spec_counters() { curl -s -m 6 "$BASE/metrics" | grep -E '^vllm:spec_decode_num_(accepted_tokens|drafts|draft_tokens)_total' | awk '{s[$1]+=$2} END{for(k in s) print k, s[k]}' | sort; }
accept_per_cycle() {  # accept_per_cycle before.txt after.txt
  paste <(sort "$1") <(sort "$2") | awk '{split($1,a,"{"); d[a[1]]=$4-$2} END{if(d["vllm:spec_decode_num_drafts_total"]>0) printf "accepted_drafts_per_cycle %.3f  drafts %d\n", d["vllm:spec_decode_num_accepted_tokens_total"]/d["vllm:spec_decode_num_drafts_total"], d["vllm:spec_decode_num_drafts_total"]; else print "accept n/a (no drafts)"}'
}
PC_BROKEN=0
i=0
for arm in "$@"; do
  i=$((i+1)); tag=$(printf '%02d-%s' "$i" "$arm"); R_YAML="$REPO/recipes/arms/$arm.yaml"
  [ -f "$R_YAML" ] || { echo "missing recipe $R_YAML"; continue; }
  if [ "$PC_BROKEN" = 1 ] && grep -q -- '--enable-prefix-caching' "$R_YAML"; then
    sed -i 's/--enable-prefix-caching/--no-enable-prefix-caching/' "$R_YAML"
    echo "!!! $tag: prefix caching flipped OFF (gate failed on an earlier arm)"; tag="$tag-nopc"
  fi
  echo "=== $(date +%T) $tag start"
  sparkrun stop --all >/dev/null 2>&1; sleep 5
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  ssh "$WORKER" 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null'
  slog="$OUT/$tag-serve.log"
  sparkrun run "$R_YAML" --cluster "$CLUSTER" --tp 2 > "$slog" 2>&1 &
  up=0
  for _ in $(seq 1 120); do
    [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' "$BASE/health")" = 200 ] && { up=1; break; }
    grep -aqE "launch_inference failed|Engine core initialization failed" "$slog" && break
    sleep 15
  done
  [ "$up" = 1 ] || { echo "!!! $tag never became healthy, see $slog"; grep -aE "Error|error:" "$slog" | tail -3 | cut -c1-200; sparkrun stop --all >/dev/null 2>&1; continue; }
  M=$(curl -s -m 6 "$BASE/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
  C=$(docker ps -q | head -1)
  echo "served: $M  image: $(docker inspect "$C" --format '{{.Image}}' | cut -c8-19)"
  grep -aoE 'GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|Capturing CUDA graphs[^\n]{0,80}|cudagraph_capture_sizes[^\n]{0,60}|num_speculative_tokens[^,]{0,12}' "$slog" | sort -u | head -8
  echo "mem: $(free -g | awk '/^Mem:/{print $7}')G avail head, $(ssh "$WORKER" "free -g | awk '/^Mem:/{print \$7}'")G worker"

  echo "--- $(date +%T) reuse probe (same 20k prompt x3, exclusive)"
  python3 "$REPO/scripts/reuse_probe.py" "$BASE" "$M" "$CORPUS" "$OUT/$tag-reuse.json" 2>&1 | tee "$OUT/$tag-reuse.log" | grep -E "^req|^REUSE"
  grep -aoE "CODEX_CACHE_(CONFIG|SPLIT)[^\n]{0,160}" "$slog" | head -6

  echo "--- $(date +%T) prefix gate"
  python3 "$REPO/scripts/prefix_gate.py" "$BASE" "$M" "$CORPUS" "$OUT/$tag-gate.json" > "$OUT/$tag-gate.log" 2>&1
  gate=$?; tail -1 "$OUT/$tag-gate.log"
  if [ "$gate" != 0 ]; then
    echo "!!! $tag FAILED prefix gate - not benchmarked"; grep -m3 FAIL "$OUT/$tag-gate.log" | cut -c1-240
    docker logs "$C" 2>&1 | grep -aiE "illegal memory|CUBLAS_STATUS|Traceback|EngineDead" | tail -5 | cut -c1-200 > "$OUT/$tag-gate-crash.txt"
    grep -q -- '--enable-prefix-caching' "$R_YAML" && PC_BROKEN=1
    sparkrun stop --all >/dev/null 2>&1; sleep 10; continue
  fi

  echo "--- $(date +%T) benchy grid depths=[$DEPTHS] conc=[$CONC]"
  spec_counters > "$OUT/$tag-spec-before.txt"
  uvx llama-benchy@0.4.0 --base-url "$BASE/v1" --model "$M" --tokenizer "$TOK" \
    --depth $DEPTHS --concurrency $CONC --pp 2048 --tg 128 --enable-prefix-caching \
    ${BOOK_URL:+--book-url "$BOOK_URL"} \
    --save-result "$OUT/$tag.csv" > "$OUT/$tag-benchy.log" 2>&1
  spec_counters > "$OUT/$tag-spec-after.txt"
  echo "benchy $(accept_per_cycle "$OUT/$tag-spec-before.txt" "$OUT/$tag-spec-after.txt")"
  awk -F'|' '$3 ~ /tg128|pp2048/ {gsub(/ /,"",$2); gsub(/ /,"",$3); gsub(/ /,"",$4); print "  " $2, $3, $4}' "$OUT/$tag.csv" | head -40

  if [ "${TONY_LANE:-1}" = 1 ]; then
    echo "--- $(date +%T) tony lane"
    spec_counters > "$OUT/$tag-spec-lane-before.txt"
    BASE="$BASE" MODEL="$M" "$REPO/tools/tony-bench/run_lane_sglang.sh" "vllm_${arm#fn-vllm-}" > "$OUT/$tag-lane.log" 2>&1
    spec_counters > "$OUT/$tag-spec-lane-after.txt"
    echo "lane $(accept_per_cycle "$OUT/$tag-spec-lane-before.txt" "$OUT/$tag-spec-lane-after.txt")"
    grep -E "median|x[0-9] " "$OUT/$tag-lane.log" | head -12
  fi
  [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' "$BASE/health")" = 200 ] || echo "!!! $tag server unhealthy after benchmarks"
  echo "=== $(date +%T) $tag done"
  sparkrun stop --all >/dev/null 2>&1; sleep 10
done
echo LADDER_DONE
