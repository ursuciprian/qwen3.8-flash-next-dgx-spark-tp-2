#!/usr/bin/env bash
# Detached ladder launcher. Runs ON the node. The caller's ssh returns in <2s.
#
# Why this exists: on 2026-09-05 07:34 a ladder was launched as
#   ssh dgx-01 '... nohup setsid scripts/ab_ladder.sh ... & sleep 6; ps ...'
# from a client with a 60s command timeout. The client killed its ssh; the
# session teardown reached `sparkrun run` (a child that had not fully detached
# from the session before spawning its own SSH fan-out to the worker), and
# sparkrun's exit path evicted the cluster it had just booted: docker `kill`
# exit 137 on both nodes within 1s, running.json + jobs/ rewritten at the same
# second. Looked exactly like a memguard trip or a node fault. It was neither.
#
# Rule: never put the long-running thing in the same ssh as anything that
# waits. Write the invocation here, run it under setsid with all fds closed,
# and return.
set -uo pipefail
cd /home/nvidia/GEN-AI/qwen3.8-27b-dgx-spark
OUT_SUB=${1:?out_sub}; shift
DEPTHS_V=${DEPTHS:-"0 16384"}; CONC_V=${CONC:-"1 2 5 10"}
# ab_ladder.sh also reads ARMDIR/TOKREPO/EXTRA (SGLang needs all three: its
# recipes live elsewhere, it uses the RadixArk tokenizer, and benchy 400s every
# request without --extra-body return_token_ids=false). Forward them or an
# SGLang run silently benchmarks with vLLM defaults.
ARMDIR_V=${ARMDIR:-recipes/vllm/arms}; TOKREPO_V=${TOKREPO:-models--unsloth--Qwen3.8-27B-NVFP4}; EXTRA_V=${EXTRA:-}
mkdir -p "results/$OUT_SUB"
setsid nohup env OUT_SUB="$OUT_SUB" DEPTHS="$DEPTHS_V" CONC="$CONC_V" \
  ARMDIR="$ARMDIR_V" TOKREPO="$TOKREPO_V" EXTRA="$EXTRA_V" \
  bash scripts/ab_ladder.sh "$@" > "results/$OUT_SUB/chain.log" 2>&1 < /dev/null &
disown
# Callers: this script returns fast because the child owns no inherited fd.
# When you launch ANY other detached job over ssh, the pattern is
#   ssh host 'setsid nohup CMD > LOG 2>&1 < /dev/null & disown'
# with ALL THREE redirects - a missing one keeps ssh attached and a client
# timeout then kills the job (2026-09-05 07:34, see docs/FIXES.md).
sleep 1
echo "launched pid=$! out=results/$OUT_SUB/chain.log arms: $*"
