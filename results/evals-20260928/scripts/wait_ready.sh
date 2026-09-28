#!/bin/bash
# wait up to 60 min for health 200 + pong; prints image of running container
for i in $(seq 1 360); do
  if [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/health)" = 200 ]; then
    out=$(curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen3.8-flash-next\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word pong.\"}],\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
    if echo "$out" | grep -qi pong; then echo "READY after $((i*10))s image=$(docker ps --format "{{.Image}}" | grep spark-vllm)"; exit 0; fi
  fi
  sleep 10
done
echo "TIMEOUT"; exit 1
