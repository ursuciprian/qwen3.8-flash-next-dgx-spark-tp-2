#!/usr/bin/env bash
# A/B/A ladder for Qwen3.8-Flash-Next on SGLang TP=2: boot each arm, screening grid, counters, stop, next.
# Fresh boot per arm is the point - single-run deltas on this hardware do not
# survive a repeat (2026-09-06: a +34.2% cell came back at baseline on rerun).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEAD=${HEAD_IP:-192.168.100.62}; WORKER=${WORKER_IP:-192.168.100.53}; CLUSTER=${CLUSTER:-dgx-cluster-cx7}
DEPTHS=${DEPTHS:-"0 16384"}; CONC=${CONC:-"1 2 5 10"}
# Boot gate. Decode drift between two fresh boots of the SAME recipe measured
# 10.6% median / 25% worst on 2026-09-06, and the FlashInfer autotuner is known
# to commit a slow roll for the process lifetime (research doc 3.4: identical
# launches serve at ~42 or ~33 tok/s). A probe after boot and a restart on a
# slow roll is the only cheap defence. Floor is per-recipe; 0 disables.
BOOT_TG_FLOOR=${BOOT_TG_FLOOR:-0}; BOOT_TRIES=${BOOT_TRIES:-3}
TOK=$(ls -d "$HOME"/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/*/ | head -1)
OUT="$REPO/results/arms"; mkdir -p "$OUT"
i=0
for arm in "$@"; do
  i=$((i+1)); tag=$(printf '%02d-%s' "$i" "$arm"); R_YAML="$REPO/recipes/arms/$arm.yaml"
  [ -f "$R_YAML" ] || { echo "missing recipe $R_YAML"; continue; }
  echo "=== $(date +%T) $tag start"
  ok=0
  for try in $(seq 1 "$BOOT_TRIES"); do
    sparkrun stop --all >/dev/null 2>&1; sleep 5
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
    ssh "$WORKER" 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null'
    slog="$OUT/$tag-serve.log"; [ "$try" -gt 1 ] && slog="$OUT/$tag-serve.try$try.log"
    sparkrun run "$R_YAML" --cluster "$CLUSTER" --tp 2 > "$slog" 2>&1 &
    up=0
    for _ in $(seq 1 100); do
      [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' "http://$HEAD:8000/health")" = 200 ] && { up=1; break; }
      sleep 15
    done
    [ "$up" = 1 ] || { echo "!!! $tag try $try never became healthy, see $slog"; continue; }
    M=$(curl -s -m 6 "http://$HEAD:8000/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
    [ "$BOOT_TG_FLOOR" = 0 ] && { ok=1; break; }
    # Probe: one short c1 decode, read aggregate tg128 from the table benchy writes.
    uvx llama-benchy@0.4.0 --base-url "http://$HEAD:8000/v1" --model "$M" --tokenizer "$TOK" \
      --extra-body return_token_ids=false --depth 0 --concurrency 1 --pp 512 --tg 128 \
      --save-result "$OUT/$tag-probe$try.csv" > /dev/null 2>&1
    probe=$(awk -F'|' '$3 ~ /^ *tg128( \(c1\))? *$/ {gsub(/ /,"",$4); split($4,a,"±"); print a[1]; exit}' "$OUT/$tag-probe$try.csv" 2>/dev/null)
    echo "boot probe try $try: tg128 c1 = ${probe:-n/a} (floor $BOOT_TG_FLOOR)"
    if [ -n "$probe" ] && awk -v p="$probe" -v f="$BOOT_TG_FLOOR" 'BEGIN{exit !(p+0 >= f+0)}'; then ok=1; break; fi
    echo "    slow roll, rebooting"
  done
  [ "$ok" = 1 ] || { echo "!!! $tag failed boot gate after $BOOT_TRIES tries"; sparkrun stop --all; continue; }
  echo "served model: $M"
  # Prove which image and QSA backend the arm actually booted. Arms differ by
  # image tag and bind mount, and the bare tag moves under us, so a wrong pin
  # here silently turns A into B.
  C=$(docker ps -q | head -1)
  echo "image: $(docker inspect "$C" --format '{{.Config.Image}}')  qsa_backend_sha: $(docker exec "$C" sha256sum /sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py | cut -c1-16)"
  grep -aoE 'max_total_tokens=[0-9]+|max_running_requests=[0-9]+|max_mamba_cache_size=[0-9-]+|bs=\[[0-9, ]+' "$OUT/$tag-serve.log" | sort -u | head -6
  uvx llama-benchy@0.4.0 --base-url "http://$HEAD:8000/v1" --model "$M" --tokenizer "$TOK" \
    --extra-body return_token_ids=false \
    --depth $DEPTHS --concurrency $CONC --pp 2048 --tg 128 --enable-prefix-caching \
    ${BOOK_URL:+--book-url "$BOOK_URL"} \
    --save-result "$OUT/$tag.csv" > "$OUT/$tag-benchy.log" 2>&1
  curl -s "http://$HEAD:8000/metrics" | grep -E '^sglang:spec_accept_length' > "$OUT/$tag-metrics.txt"
  awk '{print "accept_len " $NF}' "$OUT/$tag-metrics.txt" | tail -1
  grep -aoE "Using the [^\n]{0,80}QSA kernel[^\n]{0,30}|KV cache size: [0-9,]+|max_total_tokens=[0-9]+" "$OUT/$tag-serve.log" | sort -u | head -3
  if [ "${TONY_LANE:-1}" = 1 ]; then
    echo "--- $(date +%T) category lane"
    BASE="http://$HEAD:8000" MODEL="$M" "$REPO/tools/tony-bench/run_lane_sglang.sh" "sgl_${arm#fn-}" > "$OUT/$tag-lane.log" 2>&1
    grep -E "^  c[1-6]:" "$OUT/$tag-lane.log" | head -6
    curl -s "http://$HEAD:8000/metrics" | grep -E '^sglang:spec_accept_length' | awk '{print "accept_len after lane " $NF}' | tail -1
  fi
  echo "=== $(date +%T) $tag done"
  sparkrun stop --all >/dev/null 2>&1; sleep 10
done
echo LADDER_DONE
