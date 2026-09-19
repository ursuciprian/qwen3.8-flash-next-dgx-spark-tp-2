#!/usr/bin/env bash
# Gate on the 27B all-in grid finishing, then run the Flash-Next A/B/A ladder.
set -x
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
mkdir -p results/arms
until grep -q ALLIN_GRID_DONE /home/nvidia/GEN-AI/qwen3.8-27b-dgx-spark/results/allin-grid/benchy.log 2>/dev/null; do sleep 120; done
sleep 30; sparkrun stop --all; sleep 10
exec scripts/ab_ladder.sh fn-control fn-capfit fn-capfit-mtp2 fn-capfit-mtp4 fn-capfit-ep fn-control > results/arms/ladder-fn1.log 2>&1
