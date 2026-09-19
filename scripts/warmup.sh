#!/usr/bin/env bash
# 20 short requests to get past CUDA-graph/autotune warmup before arming the profiler.
set -euo pipefail
URL="http://localhost:8000/v1/chat/completions"
for i in $(seq 1 20); do
  curl -s -o /dev/null -m 60 -X POST "$URL" -H 'Content-Type: application/json' -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role":"user","content":"Say OK."}],
    "max_tokens": 8,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }' &
done
wait
echo "warmup done"
