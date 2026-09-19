#!/usr/bin/env bash
# Gate battery: straggler, fidelity, hardmode, categories c1, decode_probe, sweep c1/c4/c8/c16.
# Run on dgx-01 inside the repo. Usage: gate.sh <label> [--needle]
set -uo pipefail
REPO=/home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
LABEL="${1:?label}"; shift || true
NEEDLE=0
[ "${1:-}" = "--needle" ] && NEEDLE=1
OUT="$REPO/results/arms/$LABEL"
mkdir -p "$OUT"
export PATH="$HOME/.local/bin:$PATH"
BASE=http://localhost:8000
MODEL=qwen3.8-flash-next
TOK=$(ls -d "$HOME"/.cache/huggingface/hub/models--*Qwen3.8-Flash-Next-NVFP4/snapshots/*/ 2>/dev/null | head -1)

echo "== $LABEL gate start $(date -u +%FT%TZ) ==" | tee "$OUT/gate.log"

echo "-- health" | tee -a "$OUT/gate.log"
curl -s -m 10 "$BASE/health" -w " HTTP:%{http_code}\n" | tee -a "$OUT/gate.log"

echo "-- straggler (batches 5 6 7 8 12 16)" | tee -a "$OUT/gate.log"
python3 /tmp/straggler.py 5 6 7 8 12 16 > "$OUT/straggler.log" 2>&1
tail -20 "$OUT/straggler.log" | tee -a "$OUT/gate.log"

echo "-- fidelity_probe (8k/32k/64k/128k)" | tee -a "$OUT/gate.log"
python3 "$REPO/scripts/fidelity_probe.py" --base "$BASE" --model "$MODEL" \
  --depths 8000,32000,64000,128000 --out "$OUT/fidelity.json" > "$OUT/fidelity_probe.txt" 2>&1
tail -20 "$OUT/fidelity_probe.txt" | tee -a "$OUT/gate.log"

echo "-- tool-eval-bench hardmode (full scenarios, thinking on, matches baseline methodology)" | tee -a "$OUT/gate.log"
tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 \
  --base-url "$BASE" --model "$MODEL" > "$OUT/hardmode.log" 2>&1
grep -E "Final Score|Quality:|Deployability" "$OUT/hardmode.log" | tee -a "$OUT/gate.log"

echo "-- bench_categories c1" | tee -a "$OUT/gate.log"
( cd "$REPO" && python3 tools/tony-bench/bench_categories.py "$BASE" "$MODEL" "$LABEL" \
  --thinking off --concurrency 1 > "$OUT/categories_c1.log" 2>&1 )
mv "$REPO/results/categories_${LABEL}_off_c1.json" "$OUT/" 2>/dev/null
tail -10 "$OUT/categories_c1.log" | tee -a "$OUT/gate.log"

echo "-- decode_probe" | tee -a "$OUT/gate.log"
python3 "$REPO/scripts/decode_probe.py" "$MODEL" 3 > "$OUT/decode_probe.txt" 2>&1
cat "$OUT/decode_probe.txt" | tee -a "$OUT/gate.log"

echo "-- bench_sweep c1/c4/c8/c16" | tee -a "$OUT/gate.log"
( cd "$REPO/tools/tony-bench" && python3 bench_sweep.py "$BASE" "$MODEL" "$LABEL" \
  --levels 1,4,8,16 --out "$OUT/sweep.json" > "$OUT/sweep.log" 2>&1 )
tail -10 "$OUT/sweep.log" | tee -a "$OUT/gate.log"

if [ "$NEEDLE" = "1" ]; then
  echo "-- needle_ladder 8k/32k/64k" | tee -a "$OUT/gate.log"
  python3 "$REPO/scripts/needle_ladder.py" --base "$BASE" --model "$MODEL" \
    --depths 8000,32000,64000 --out "$OUT/needle.json" > "$OUT/needle.log" 2>&1
  tail -10 "$OUT/needle.log" | tee -a "$OUT/gate.log"
fi

echo "-- MTP acceptance metrics" | tee -a "$OUT/gate.log"
curl -s -m 10 "$BASE/metrics" | grep -i "spec_decode_num_accepted_tokens_per_pos\|spec_decode_num_accepted_tokens_total\|spec_decode_num_drafts_total" > "$OUT/mtp_metrics.txt" 2>&1
cat "$OUT/mtp_metrics.txt" | tee -a "$OUT/gate.log"

echo "== $LABEL gate done $(date -u +%FT%TZ) ==" | tee -a "$OUT/gate.log"
