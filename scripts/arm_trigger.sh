#!/usr/bin/env bash
# Arms the decode-step profiler on both nodes for one label. Each call uses a
# brand-new filename (label + epoch nanoseconds) -- see profiler_snippet.py's
# comment for why a fixed, reused filename is not reliably observed by the
# long-lived worker process.
# Usage: arm_trigger.sh <label>
set -euo pipefail
LABEL="$1"
DIR=".cache/sparkrun/runtime-cache/vllm/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-ca6f25af/prof"
NAME="go-${LABEL}-$(date +%s%N)"
mkdir -p "$HOME/$DIR"
echo -n "$LABEL" > "$HOME/$DIR/$NAME"
ssh 192.168.100.53 "mkdir -p ~/$DIR && echo -n '$LABEL' > ~/$DIR/$NAME"
echo "armed label=$LABEL on both nodes (file=$NAME)"
