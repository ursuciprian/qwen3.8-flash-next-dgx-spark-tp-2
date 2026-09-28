#!/bin/bash
# usage: run_lmeval.sh <build> <eval> <tasks> [extra lm_eval args...]
set -u
B=$1; E=$2; TASKS=$3; shift 3
R=~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/evals-20260928
OUT=$R/$E/$B; mkdir -p $OUT
C=${CONC:-8}
snap(){ curl -s localhost:8000/metrics | grep -E "^vllm:(generation_tokens_total|prompt_tokens_total|request_success_total)" > $OUT/metrics_$1.txt; }
CMD=(~/GEN-AI/evals/venv/bin/lm_eval run --model local-chat-completions
  --model_args "model=qwen3.8-flash-next,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=$C,max_retries=5,timeout=7200,tokenized_requests=False"
  --tasks "$TASKS" --include_path ~/GEN-AI/evals/tasks --apply_chat_template --fewshot_as_multiturn
  --output_path $OUT --log_samples --seed 1234 "$@")
printf "%q " "${CMD[@]}" > $OUT/command.txt; echo >> $OUT/command.txt
curl -s localhost:8000/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)[\"data\"][0][\"root\"])" > $OUT/served_model.txt
docker ps --format "{{.Image}}" | grep spark-vllm >> $OUT/served_model.txt
snap before; T0=$(date +%s); echo "start $(date -Is)" > $OUT/timing.txt
HF_DATASETS_TRUST_REMOTE_CODE=1 "${CMD[@]}"; RC=$?
T1=$(date +%s); snap after; echo "end $(date -Is) rc=$RC wall_s=$((T1-T0)) conc=$C" >> $OUT/timing.txt
