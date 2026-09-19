#!/usr/bin/env bash
# Drive llama-benchy against an already-loading sparkrun server.
# run.sh exited on a false-positive "container exited during load", but sparkrun
# kept loading, so the serve process is fine and only the bench step was lost.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO=/home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
HEAD=192.168.100.62; PORT=8000
OUT="$REPO/results/retest-bigkv-g8"; mkdir -p "$OUT"
TOK=$(ls -d "$HOME"/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/*/ | head -1)

echo "== waiting for server ready (up to 30 min) =="
ok=0
for i in $(seq 1 180); do
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://$HEAD:$PORT/health" || true)
  if [ "$code" = 200 ]; then ok=1; echo "SERVER READY after $((i*10))s"; break; fi
  # sparkrun died -> stop waiting instead of burning the full 30 min
  pgrep -f '[s]parkrun run' >/dev/null || { echo "FAILED: sparkrun process gone"; break; }
  sleep 10
done
[ "$ok" = 1 ] || { echo "FAILED: server never became ready"; exit 1; }

echo "== KV / mamba sizing =="
grep -aoE 'max_mamba_cache_size=[0-9-]+|KV cache size: [0-9,]+|max_running_requests=[0-9]+|#tokens: [0-9]+' \
  "$REPO/results/sglang-serve.log" | tail -5

echo "== benchmarking: depths 0 16384 32768 65536 x conc 1 2 5 =="
uvx llama-benchy@0.4.0 --base-url "http://$HEAD:$PORT/v1" \
  --model qwen3.8-flash-next --tokenizer "$TOK" \
  --extra-body return_token_ids=false \
  --depth 0 16384 32768 65536 --pp 2048 --tg 128 --enable-prefix-caching \
  --concurrency 1 2 5 \
  --save-result "$OUT/bigkv-g8-c4096-retest.csv"
rc=$?
echo "BENCHY_EXIT=$rc"
curl -s -m 6 "http://$HEAD:$PORT/metrics" | grep -iE 'spec.*accept|accept_len' | head -5
echo "BENCH_DONE"
