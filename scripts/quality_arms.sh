#!/usr/bin/env bash
# Model-quality comparison chain for Qwen3.8-Flash-Next on the DGX Spark pair.
# Arms: A0 (eugr-agents baseline), A1 (BF16 KV), R0 (RadixArk ckpt, fp8 KV, w/ boot fallbacks),
# R1 (RadixArk ckpt, BF16 KV, w/ boot fallbacks), N1 (nvidia ckpt, BF16 KV — only if R0+R1 both fail).
# Runs detached (setsid nohup) on dgx-01 so it survives ssh/agent teardown.
export PATH="$HOME/.local/bin:$PATH"
Q=~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
OUT=$Q/results/quality-arms
L=~/glm-sgl/quality_arms.log
BASE=http://localhost:8000
M=local-inference-lab/Qwen3.8-Flash-Next-NVFP4   # served-model-name kept identical across all arms
TOK=$(ls -d ~/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/*/ | head -1)
mkdir -p "$OUT" ~/glm-sgl
cd "$Q"

say() { echo "=== $(date +%T) $*" | tee -a "$L"; }

# real 1-token completion check (not just /health — earlier boots died 3 min after health)
completion_ok() {
  code=$(curl -s -m 20 -o /tmp/qa_completion.json -w '%{http_code}' "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"reply with the single word OK\"}],\"max_tokens\":4,\"temperature\":0}")
  [ "$code" = "200" ] && grep -q '"content"' /tmp/qa_completion.json
}

# gen_variant: derive a fallback recipe by dropping/altering one flag from a base recipe
gen_variant() {
  base=$1; out=$2; mode=$3
  case "$mode" in
    nolf) grep -v -- '--load-format b12x' "$base" > "$out" ;;
    nolb) grep -v -- '--linear-backend b12x' "$base" > "$out" ;;
    mmixed) sed 's/--quantization modelopt \\/--quantization modelopt_mixed \\/' "$base" > "$out" ;;
  esac
  sed -i "s/^name: .*/name: $(basename "$out" .yaml)/" "$out"
}

# boot <recipe_yaml_path> <arm_dir> -> 0 on success (health + real completion + 3min survive check)
boot() {
  rec=$1; arm=$2
  mkdir -p "$OUT/$arm"
  sparkrun stop --all >/dev/null 2>&1; sleep 5
  bash ~/glm-sgl/dropcache.sh >/dev/null 2>&1 || true
  say "$arm: boot $(basename "$rec")"
  ( sparkrun run "$rec" --cluster dgx-cluster-cx7 --tp 2 --trust --no-follow > "$OUT/$arm/sparkrun.log" 2>&1 ) || true
  ok=1
  for _ in $(seq 1 90); do   # up to 90*20s = 30 min
    if [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' $BASE/health)" = 200 ] && completion_ok; then
      ok=0; break
    fi
    if [ -z "$(docker ps -q)" ]; then sleep 20; [ -z "$(docker ps -q)" ] && break; fi
    C=$(docker ps -q | head -1)
    [ -n "$C" ] && docker exec "$C" grep -aqE "EngineCore.*(died|failed)|RuntimeError|ValueError|Traceback" /tmp/sparkrun_serve.log 2>/dev/null && { sleep 15; break; }
    sleep 20
  done
  C=$(docker ps -q | head -1)
  [ -n "$C" ] && docker exec "$C" cat /tmp/sparkrun_serve.log > "$OUT/$arm/serve.log" 2>/dev/null
  if [ $ok -ne 0 ]; then
    say "$arm: boot FAILED | $(grep -aE 'RuntimeError|ValueError|Error:|Traceback' "$OUT/$arm/serve.log" 2>/dev/null | grep -viE 'deprecat|warn' | tail -1 | cut -c1-220)"
    return 1
  fi
  say "$arm: healthy, verified with real completion"
  grep -a -iE "Model loading took|GPU KV cache size|Maximum concurrency" "$OUT/$arm/serve.log" | tail -6 | cut -c1-200 | tee -a "$L"
  say "$arm: waiting 180s idle (no ssh sessions) to check RemoveIPC survival"
  sleep 180
  if completion_ok; then
    echo survived > "$OUT/$arm/survive.txt"; say "$arm: survived 3min idle check"
  else
    echo died > "$OUT/$arm/survive.txt"; say "$arm: DIED during 3min idle check"; return 1
  fi
  return 0
}

# boot_with_fallbacks <base_recipe_path> <arm_dir> — R0/R1 fallback order per task spec
boot_with_fallbacks() {
  base=$1; arm=$2
  if boot "$base" "$arm"; then
    echo "$base" > "$OUT/$arm/final-recipe.txt"; return 0
  fi
  for mode in nolf nolb mmixed; do
    var="$Q/recipes/eugr/${arm}-${mode}.yaml"
    gen_variant "$base" "$var" "$mode"
    say "$arm: fallback attempt $mode -> $(basename "$var")"
    if boot "$var" "$arm"; then
      echo "$var" > "$OUT/$arm/final-recipe.txt"; return 0
    fi
  done
  return 1
}

# battery <arm_dir> <extra_temp: 0|1>
battery() {
  arm=$1; extra_t10=$2
  say "$arm: fidelity T0.6"
  python3 $Q/scripts/fidelity_probe.py --base $BASE --model "$M" --depths 8000,32000,64000,128000 \
    --k 5 --trials 4 --thinking on --temperature 0.6 --concurrency 4 \
    --out "$OUT/$arm/fid-t06.json" > "$OUT/$arm/fid-t06.log" 2>&1
  say "$arm: fidelity T0.6 done | $(tail -4 "$OUT/$arm/fid-t06.log" | tr '\n' ' ' | cut -c1-300)"
  if [ "$extra_t10" = "1" ]; then
    say "$arm: fidelity T1.0 (A0 only)"
    python3 $Q/scripts/fidelity_probe.py --base $BASE --model "$M" --depths 8000,32000,64000,128000 \
      --k 5 --trials 4 --thinking on --temperature 1.0 --concurrency 4 \
      --out "$OUT/$arm/fid-t10.json" > "$OUT/$arm/fid-t10.log" 2>&1
    say "$arm: fidelity T1.0 done | $(tail -4 "$OUT/$arm/fid-t10.log" | tr '\n' ' ' | cut -c1-300)"
  fi
  say "$arm: decode probe (code/structured/counting/prose x3)"
  sed "s#http://192.168.100.62:8000#$BASE#" $Q/scripts/decode_probe.py > /tmp/decode_probe_qa.py
  python3 /tmp/decode_probe_qa.py "$M" 3 > "$OUT/$arm/decode-lanes.log" 2>&1
  cat "$OUT/$arm/decode-lanes.log" | tail -10 | cut -c1-200 | tee -a "$L"
  say "$arm: tool-eval hardmode (T1.0, thinking medium, 32 turns)"
  ( cd "$OUT/$arm" && tool-eval-bench run --hardmode --seed 42 --parallel 1 --max-turns 32 \
      --backend vllm --timeout 600 --base-url $BASE --model "$M" \
      --backend-kwargs '{"chat_template_kwargs": {"thinking": true, "reasoning_effort":"medium"},"temperature": 1.0,"top_p":0.95}' \
      > "$OUT/$arm/hardmode.log" 2>&1 )
  say "$arm: hardmode $(grep -aE 'Score: *[0-9]+ */ *100|passed|Responsiveness' "$OUT/$arm/hardmode.log" | tr -s ' ' | tr '\n' ' ' | cut -c1-240)"
}

### A0 smoke test first ###
say "A0: smoke test boot"
if boot recipes/eugr/eugr-agents.yaml A0; then
  say "A0: smoke fidelity depth 8000 trials 1"
  python3 $Q/scripts/fidelity_probe.py --base $BASE --model "$M" --depths 8000 \
    --k 5 --trials 1 --thinking on --temperature 0.6 --concurrency 4 \
    --out "$OUT/A0/fid-smoke.json" > "$OUT/A0/fid-smoke.log" 2>&1
  say "A0: smoke result | $(tail -3 "$OUT/A0/fid-smoke.log" | tr '\n' ' ' | cut -c1-300)"
  echo SMOKE_READY_FOR_REVIEW >> "$L"

  ### full A0 battery (extra T1.0) ###
  battery A0 1
else
  say "A0: boot FAILED — aborting chain"
  say QUALITY_ARMS_DONE
  exit 1
fi

### A1 ###
if boot recipes/eugr/eugr-agents-kvbf16.yaml A1; then
  echo recipes/eugr/eugr-agents-kvbf16.yaml > "$OUT/A1/final-recipe.txt"
  battery A1 0
else
  say "A1: boot FAILED, skipping battery"
fi

### R0 ###
r0_ok=1
if boot_with_fallbacks recipes/eugr/eugr-agents-radixark.yaml R0; then
  r0_ok=0
  battery R0 0
else
  say "R0: all fallbacks failed"
fi

### R1 ###
r1_ok=1
if boot_with_fallbacks recipes/eugr/eugr-agents-radixark-kvbf16.yaml R1; then
  r1_ok=0
  battery R1 0
else
  say "R1: all fallbacks failed"
fi

### N1 (only if both R0 and R1 failed) ###
if [ $r0_ok -ne 0 ] && [ $r1_ok -ne 0 ]; then
  say "N1: R0 and R1 both failed, trying nvidia checkpoint fallback"
  if boot recipes/eugr/eugr-agents-nvidia-kvbf16.yaml N1; then
    echo recipes/eugr/eugr-agents-nvidia-kvbf16.yaml > "$OUT/N1/final-recipe.txt"
    battery N1 0
  else
    say "N1: boot FAILED too"
  fi
fi

say QUALITY_ARMS_DONE
