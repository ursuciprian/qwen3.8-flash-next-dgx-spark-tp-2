#!/usr/bin/env bash
# After the current arm finishes, replace the running ladder with the next arm list.
#   vllm_ladder_handoff.sh "<done-marker>" <archive-name> arm1 arm2 ...
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
MARK="$1"; ARCHIVE="$2"; shift 2
LOG=results/arms/vllm-imp/ladder.log
until grep -q "$MARK" "$LOG"; do sleep 20; done
kill $(pgrep -f "bash scripts/vllm_ladder.sh") 2>/dev/null; sleep 3
sparkrun stop --all >/dev/null 2>&1; sleep 10
mv "$LOG" "results/arms/vllm-imp/$ARCHIVE"
exec bash scripts/vllm_ladder.sh "$@" > "$LOG" 2>&1
