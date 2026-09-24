#!/usr/bin/env bash
# Complete-QSA transaction latency at decode shapes, three b12x trees.
# WRITTEN, NOT LAUNCHED. Single GPU on dgx-01; run only with the server
# stopped (sparkrun stop --all) and nothing else queued. Journal tag
# opus:16k-regression.
#
#   shipped  a8333658                  in the shipped image (its own toolchain)
#   master   e6b6b93e (candidate)      in the candidate image
#   fix      feat/candidate-16k-fix    in the candidate image, PYTHONPATH over
#            the installed b12x; correctness tests gate its timing.
#
# Rows 1..80 packed decode rows (c1 verify = 5 rows, c16 = 80), contexts up to
# 128K; each row is one request at that context (benchmark_qsa.py
# "throughput" kind), FP8 KV, interleaved cache as vLLM binds it. Output:
# $OUT/<tree>.json (+ .log); compare with the summary lines each run prints.
set -euo pipefail
B12X_GIT=$HOME/GEN-AI/build/b12x
SRC=$HOME/GEN-AI/qsa-bench
OUT=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/qsa-microbench-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)
SHIPPED_IMG=ghcr.io/ursuciprian/spark-vllm-b12x:b0-20260918-a8333658-warm
CAND_IMG=spark-vllm-b12x:candidate-b12x-e6b6b93e-vllm-869138f2
ROWS=1,2,5,10,20,40,80
CONTEXTS=2048,8192,18432,65536,131072
mkdir -p "$SRC" "$OUT"

[ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health)" = 200 ] \
  && { echo "server is up; stop it first (sparkrun stop --all)"; exit 1; }

git -C "$B12X_GIT" fetch -q fork feat/candidate-16k-fix
for spec in shipped:a8333658 master:e6b6b93e fix:FETCH_HEAD; do
  name=${spec%%:*} ref=${spec#*:}
  rm -rf "${SRC:?}/$name"; git -C "$B12X_GIT" worktree prune
  git -C "$B12X_GIT" worktree add -q --detach "$SRC/$name" "$ref"
  git -C "$SRC/$name" log --oneline -1 > "$OUT/$name.commit"
done

run() {  # run <image> <tree> <pythonpath-override:0|1> <command>
  local pp=""; [ "$3" = 1 ] && pp="/src/$2"
  docker run --rm --gpus all --ipc=host --network host \
    -v "$SRC:/src:ro" -v "$OUT:/out" \
    -e CUTE_DSL_ARCH=sm_121a -e XDG_CACHE_HOME=/tmp/xdg -e PYTHONPATH="$pp" \
    -w "/src/$2" --entrypoint bash "$1" -c "$4"
}

bench="python3 benchmarks/benchmark_qsa.py --profiles tp2 --rows $ROWS --contexts $CONTEXTS \
  --kv-cache-dtype fp8_e4m3 --main-cache-layout interleaved --graph-replays 200"
nvidia-smi --query-gpu=uuid,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv > "$OUT/gpu.txt"

# Correctness gate for the fix: exact winners, graph replay, frozen resolution.
run "$CAND_IMG" fix 1 "pip install -q --target /tmp/pt pytest >/dev/null 2>&1; \
  PYTHONPATH=/src/fix:/tmp/pt python3 -m pytest -q -x tests/attention/test_qsa_stable_selection.py \
  tests/attention/test_qsa_program_keys.py tests/attention/test_qsa_contract.py" \
  > "$OUT/fix-tests.log" 2>&1 || { echo "fix tests failed, see $OUT/fix-tests.log"; exit 1; }

# Interleave arms so drift (clocks, thermals) hits all three.
for round in 1 2; do
  run "$SHIPPED_IMG" shipped 1 "$bench --output /out/shipped-r$round.json" > "$OUT/shipped-r$round.log" 2>&1
  run "$CAND_IMG" master 0 "$bench --output /out/master-r$round.json" > "$OUT/master-r$round.log" 2>&1
  run "$CAND_IMG" fix 1 "$bench --output /out/fix-r$round.json" > "$OUT/fix-r$round.log" 2>&1
done
nvidia-smi --query-gpu=uuid,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv >> "$OUT/gpu.txt"
echo "done: $OUT"
