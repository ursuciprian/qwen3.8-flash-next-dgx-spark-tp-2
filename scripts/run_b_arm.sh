#!/usr/bin/env bash
# MTP sampling arm: default-temp prose grid (0,16384 x c1,4,10), temp0 decode checks,
# and the task-shaped grid from the llama-benchy fork.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
ARM="${1:?arm label}"
REPO=/home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
OUT="$REPO/results/arms/$ARM"; mkdir -p "$OUT"
BENCHY="$REPO/results/benchy"; mkdir -p "$BENCHY"
TOK=$(ls -d "$HOME"/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/*/ | head -1)
M=qwen3.8-flash-next
HEAD=192.168.100.62

echo "== $ARM start $(date -u +%FT%TZ)"

curl -s -m 10 localhost:8000/metrics | grep -E "spec_decode_num_(accepted|draft)_tokens_total" > "$OUT/mtp_before_prose.txt"
uvx llama-benchy@0.4.0 --base-url "http://$HEAD:8000/v1" --model "$M" --tokenizer "$TOK" \
  --extra-body "return_token_ids=false" --depth 0 16384 --concurrency 1 4 10 \
  --pp 2048 --tg 128 --runs 2 --enable-prefix-caching --save-result "$BENCHY/${ARM}-prose.csv" \
  > "$BENCHY/${ARM}-prose.log" 2>&1
curl -s -m 10 localhost:8000/metrics | grep -E "spec_decode_num_(accepted|draft)_tokens_total" > "$OUT/mtp_after_prose.txt"
python3 "$REPO/scripts/mk_table2.py" "$BENCHY/${ARM}-prose.csv" > "$BENCHY/${ARM}-prose.md" 2>>"$BENCHY/${ARM}-prose.log"
echo "-- $ARM prose grid done"

echo "-- decode regression: 3x levels=1, 1x levels=8 (temp0 check)"
cd "$REPO/tools/tony-bench"
for i in 1 2 3; do
  python3 bench_sweep.py http://localhost:8000 "$M" "$ARM" --levels 1 --out "$OUT/decode_c1_run${i}.json" > "$OUT/decode_c1_run${i}.log" 2>&1
done
python3 bench_sweep.py http://localhost:8000 "$M" "$ARM" --levels 8 --out "$OUT/decode_c8.json" > "$OUT/decode_c8.log" 2>&1
cd "$REPO"
echo "-- $ARM decode checks done"

echo "-- task grid (fork, default temp)"
curl -s -m 10 localhost:8000/metrics | grep -E "spec_decode_num_(accepted|draft)_tokens_total" > "$OUT/mtp_before_task.txt"
uvx --from ~/GEN-AI/llama-benchy-fork llama-benchy --base-url "http://$HEAD:8000/v1" --model "$M" --tokenizer "$TOK" \
  --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 16384 65536 --concurrency 1 4 10 \
  --metrics-url "http://$HEAD:8000/metrics" --live --enable-prefix-caching --save-result "$BENCHY/${ARM}-task.csv" \
  > "$BENCHY/${ARM}-task.log" 2>&1
curl -s -m 10 localhost:8000/metrics | grep -E "spec_decode_num_(accepted|draft)_tokens_total" > "$OUT/mtp_after_task.txt"
python3 "$REPO/scripts/mk_table2.py" "$BENCHY/${ARM}-task.csv" > "$BENCHY/${ARM}-task.md" 2>>"$BENCHY/${ARM}-task.log"
echo "-- $ARM task grid done"

echo "== $ARM done $(date -u +%FT%TZ)"
