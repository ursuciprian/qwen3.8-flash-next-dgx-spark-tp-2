#!/usr/bin/env bash
# r11 POST (run by the driver before restore): merged/ pass links, logits diffs, screen report.
R=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2; RESULTS=${RESULTS:?}; D=$RESULTS/screen/dgx01
mkdir -p "$RESULTS/merged" && cd "$RESULTS/merged" && for p in ../screen/dgx0?/*-p?; do ln -sfn "$p" .; done
cd "$R" || exit 1
for x in "tp1-v2c-p1/logits-a tp1-v2c-p1/logits-b" "tp1-keepalive-p1/logits-a tp1-keepalive-p1/logits-b" \
         "tp1-v2c-p1/logits-a tp1-keepalive-p1/logits-a" "tp1-v2c-p1/logits-b tp1-keepalive-p1/logits-b"; do
  set -- $x; echo "== $1 vs $2"; python3 scripts/logits_equiv.py diff "$D/$1.json" "$D/$2.json"; done > "$RESULTS/logits-diff.txt" 2>&1
python3 scripts/r3_screen_report.py "$RESULTS/merged" tp1-keepalive --base tp1-v2c > "$RESULTS/report-r11.txt" 2>&1
