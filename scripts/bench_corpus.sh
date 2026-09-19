#!/usr/bin/env bash
# Run the benchy grid against an already-serving endpoint with a chosen corpus.
# Exists because every benchy number before 2026-09-07 used llama-benchy's default
# book (Sherlock Holmes) without anyone noticing that was the workload. Prose is
# the least predictable text, so it is the speculation floor. This makes corpus
# an explicit, logged choice.
#
#   scripts/bench_corpus.sh <label> <book-url> [DEPTHS="0 2048 ..."] [CONC="1 2 5"]
#   scripts/bench_corpus.sh code-btree https://raw.githubusercontent.com/sqlite/sqlite/master/src/btree.c
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
LABEL="${1:?label}"; BOOK="${2:?book url}"
DEPTHS="${DEPTHS:-0 2048 4096 8192 16384 32768}"; CONC="${CONC:-1 2 5}"
HEAD="${HEAD_IP:-192.168.100.62}"; REPO="$(cd "$(dirname "$0")/.." && pwd)"
TOK=$(ls -d "$HOME"/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/*/ | head -1)
OUT="$REPO/results/corpus"; mkdir -p "$OUT"
M=$(curl -s -m 6 "http://$HEAD:8000/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])') || { echo "no server"; exit 1; }
echo "== $(date +%T) corpus=$LABEL book=$BOOK model=$M"
echo "   image: $(docker inspect "$(docker ps -q | head -1)" --format '{{.Config.Image}}' 2>/dev/null)"
curl -s "http://$HEAD:8000/metrics" | grep -E '^sglang:spec_(accept_length|num_draft_tokens)\{' | sed -E 's/\{[^}]*\}//' | sed 's/^/   before: /'
uvx llama-benchy@0.4.0 --base-url "http://$HEAD:8000/v1" --model "$M" --tokenizer "$TOK" \
  --extra-body return_token_ids=false --book-url "$BOOK" \
  --depth $DEPTHS --concurrency $CONC --pp 2048 --tg 128 --enable-prefix-caching \
  --save-result "$OUT/$LABEL.csv" > "$OUT/$LABEL-benchy.log" 2>&1
grep -a "Total tokens available" "$OUT/$LABEL-benchy.log" | sed 's/^/   /'
curl -s "http://$HEAD:8000/metrics" | grep -E '^sglang:spec_accept_length\{' | sed -E 's/\{[^}]*\}//' | sed 's/^/   after:  /'
echo "== $(date +%T) done -> $OUT/$LABEL.csv"
