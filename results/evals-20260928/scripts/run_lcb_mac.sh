#!/bin/bash
# usage: run_lcb.sh <build> [extra args]   env: MAXTOK START END CONC   (runs on the Mac; dataset needs RAM)
set -u
B=$1; shift
API=${API:-http://<head-ip>:8000}
R=$HOME/work/GEN-AI/evals-lcb/results/$B; mkdir -p $R; cd $HOME/work/GEN-AI/evals-lcb/lcb; rm -rf output/Qwen3.8-Flash-Next-NVFP4-local
snap(){ curl -s -m 10 $API/metrics | grep -E "^vllm:(generation_tokens_total|prompt_tokens_total|request_success_total)" > $R/metrics_$1.txt; }
CMD=(venv/bin/python -m lcb_runner.runner.main --model qwen3.8-flash-next --scenario codegeneration
  --release_version release_v6 --start_date ${START:-2025-02-01} --end_date ${END:-2025-05-31} --n 1 --codegen_n 1 --temperature 1.0 --top_p 0.95
  --max_tokens ${MAXTOK:-32768} --multiprocess ${CONC:-8} --openai_timeout 7200 --evaluate --num_process_evaluate 8 --timeout 6 "$@")
printf "%q " "${CMD[@]}" > $R/command.txt; echo >> $R/command.txt
ssh dgx-01 'docker ps --format "{{.Image}}"' | grep spark-vllm > $R/served_image.txt
snap before; T0=$(date +%s); echo "start $(date +%FT%T%z)" > $R/timing.txt
PYTHONPATH=. OPENAI_KEY=local OPENAI_BASE_URL=$API/v1 HF_DATASETS_TRUST_REMOTE_CODE=1 "${CMD[@]}"; RC=$?
T1=$(date +%s); snap after; echo "end $(date +%FT%T%z) rc=$RC wall_s=$((T1-T0))" >> $R/timing.txt
cp -r output/Qwen3.8-Flash-Next-NVFP4-local $R/output
