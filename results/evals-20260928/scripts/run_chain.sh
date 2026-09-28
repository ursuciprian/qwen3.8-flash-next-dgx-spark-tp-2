#!/bin/bash
# usage: run_chain.sh <build>   MMLU-Pro 2000 subset -> GSM8K full -> IFEval full, lm-eval at c8
B=$1; cd ~/GEN-AI/evals
./run_lmeval.sh $B mmlu_pro mmlupt --samples /home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/evals-20260928/mmlu_pro/mmlu_pro_subset_2000_seed20260928.json
./run_lmeval.sh $B gsm8k gsm8k_think
./run_lmeval.sh $B ifeval ifeval_think
echo CHAIN_DONE $B
