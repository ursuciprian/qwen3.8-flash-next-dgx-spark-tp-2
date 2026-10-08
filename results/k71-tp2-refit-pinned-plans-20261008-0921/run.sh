#!/usr/bin/env bash
# k71 (2026-10-08, #115): k70 again with the plan confound removed. k70's arm bake measured 33 b12x
# gemm.blockscaled_precision plans the b1.4 image seed lacks; the control serves the same 33 keys from its long-lived
# runtime cache, and 8 of them select a different tile/split. stage.sh pins the control's selections into the arm's
# runtime cache on both Sparks (pin_plans.py), so arm and control differ only in the 24 refit mtp.* tensors.
#   preflight  stage.sh (k70 stage.sh + pin_plans.py, idempotent) + tools
#   screen     as k70 (ctl, arm, arm, ctl; same probes, logits, verdict rules, TD_ACC_RISE_OK=1), no bake (the k70
#              arm cache is warm), llama-benchy with 4 runs per cell instead of 2 (k70's arm tg512 c1 spread was
#              +-7 t/s in one boot)
#   gate       only on PROMOTE, as k70
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; comment.md + RESULT for the poster (#115). lib.sh restores the 2x.
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k71; K=$HOME/GEN-AI/backlog/k71; K70=$HOME/GEN-AI/backlog/k70
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k71-tp2-refit-pinned-plans-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
ARM=$K70/arm/k70.yaml; CTL=qwen3.8-flash-next-2x-dgx-spark; MODEL=qwen3.8-flash-next
SNAP=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
CORPUS=$R/results/corpus-code.txt; BENCHY_SRC=$G/llama-benchy-fork
GEN="List the numbers from 1 to 300 separated by commas. Output only the numbers, nothing else, no commentary."
# as scripts/thunderdome.sh: tag:depth:concurrency:reps:corpus offset:timeout s
CELLS="fresh-c1:256:1:2:104729:300 fresh-c4:256:4:2:314187:400 fresh-c8:256:8:2:418916:500
d16k-c4:16384:4:2:1361477:700 d16k-c8:16384:8:1:1466206:480 count-c8:0:8:2:0:300"

checks() {
  local ok=0 t
  bash "$K/stage.sh" || { echo "stage.sh failed"; ok=1; }
  python3 "$K70/k61.py" --selftest > /dev/null || { echo "k61.py selftest failed"; ok=1; }
  python3 "$R/scripts/thunderdome_report.py" --selftest > /dev/null 2>&1 || { echo "thunderdome_report.py selftest failed"; ok=1; }
  for t in sparkrun uvx tool-eval-bench python3 curl; do command -v $t > /dev/null || { echo "missing $t"; ok=1; }; done
  for t in scripts/depth_decode_probe.py scripts/logits_equiv.py scripts/thunderdome_report.py scripts/paired_decode_ab.py \
           scripts/fidelity_probe.py scripts/straggler_probe.py results/corpus-code.txt; do [ -e "$R/$t" ] || { echo "missing $R/$t"; ok=1; }; done
  [ -d "$BENCHY_SRC" ] && [ -d "$SNAP" ] || { echo "missing llama-benchy fork or tokenizer snapshot"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  rc=0; checks || rc=1
  others_busy && echo "other GPU jobs: busy (the runner keeps waiting)" || echo "other GPU jobs: clear"
  echo "arm  $ARM ($(sed -n 's/^container: //p' "$ARM"))"; echo "ctl  registry recipe $CTL (b1.4, $SHIPPED_TAG_DEFAULT)"
  echo "plan ctl, arm, arm, ctl (~20-23 min each: boot ~3 min, probes ~10 min, logits ~1 min, benchy ~6 min)"
  echo "     -> report -> on PROMOTE a gate boot (~50-60 min) -> restore 2x + pong (~5 min)"
  echo "estimate ~1 h 25 min - 1 h 40 min without the gate, ~2 h 30 min with it"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is queued or running: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; : > "$K/comment.md"; S=$RES/screen; mkdir -p "$S"
st "running: preflight"
checks > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: k71 preflight ($RES/preflight.txt)"; exit 1; }
cp "$ARM" "$RES/k70.yaml"; cp "$K"/stage.sh "$K"/run.sh "$K"/pin_plans.py "$RES/"
{ echo "k71 $(TZ=Europe/Bucharest date '+%F %T %Z'): refit run1 drafter on the 2x b1.4 vs b1.4, control plans pinned (#115)"
  grep -E '^(seed|recipe):|ok \(|dgx-0[12]: ' "$RES/preflight.txt"; } > "$RES/k71.txt"

keep_logs() { # dir
  local c; c=$(n0); [ -n "$c" ] || c=$(docker ps -a --format '{{.Names}}' | grep node_0 | head -1); [ -n "$c" ] || return 0
  docker exec "$c" cat /tmp/sparkrun_serve.log > "$1/serve.log" 2>&1 || docker logs "$c" > "$1/serve.log" 2>&1
  grep -E "Directly load AOT|saved AOT|Dynamo bytecode|b12x .* ready" "$1/serve.log" | cut -c1-300 > "$1/aot.txt"
  docker exec "$c" sh -c "env | grep -E '^(VLLM|B12X)_' | sort" > "$1/env.txt" 2>&1
  { docker inspect --format '{{.Config.Image}}' "$c"; ssh -n $H2 "docker ps --format '{{.Image}}' --filter name=sparkrun"; } > "$1/image.txt" 2>&1; }
tboot() { # recipe dir cold_ok -> 0 up, 1 failed, 2 cold
  local rec=$1 d=$2 s m n
  stop_all; log "boot $(basename "$rec") -> $d"
  if [ "$rec" = "$CTL" ]; then ( cd "$R" && timeout -k 60 900 sparkrun run "$CTL" --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  else cp "$rec" "$d/recipe.yaml"; ( cd "$(dirname "$rec")" && timeout -k 60 900 sparkrun run "$(basename "$rec")" --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-; fi
  s=$(date +%s)
  while :; do
    n=$(n0)
    m=$([ -n "$n" ] && docker exec "$n" grep -m1 -oE 'Directly load AOT compilation|Dynamo bytecode transform time' /tmp/sparkrun_serve.log 2>/dev/null)
    [ "$m" = "Dynamo bytecode transform time" ] && [ "$3" != 1 ] && { log "$(basename "$d"): COLD (backbone compiling), stopped"; return 2; }
    [ "$(health localhost)" = 200 ] && { echo "boot $(( $(date +%s) - s ))s ${m:-no AOT line}" > "$d/boot.txt"; log "$(basename "$d"): up, $(cat "$d/boot.txt")"; return 0; }
    if [ $(( $(date +%s) - s )) -gt 180 ] && { [ -z "$n" ] || docker exec "$n" grep -qE 'Worker failed with error|EngineCore failed to start|Engine core initialization failed' /tmp/sparkrun_serve.log 2>/dev/null; }; then
      log "$(basename "$d"): boot FAILED after $(( $(date +%s) - s ))s"; return 1; fi
    [ $(( $(date +%s) - s )) -ge $([ "$3" = 1 ] && echo 5400 || echo 1800) ] && { log "$(basename "$d"): health timeout"; return 1; }
    sleep 10; done; }
pos() { curl -s -m 10 "localhost:8000/metrics" | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens(_per_pos)?_total'; }
probe() { # dir tag depth conc reps offset timeout (as thunderdome.sh, against the TP2 head)
  local d=$1 tag=$2 depth=$3 c=$4 r=$5 off=$6 t=$7 rc s
  if [ "${tag#count}" != "$tag" ]; then set -- --prompt "$GEN" --max-tokens 320
  else set -- --depth "$depth" --new 2048 --offset "$off" --max-tokens 512; fi
  pos > "$d/pos-$tag.before"; s=$(date +%s)
  ( cd "$R" && child timeout -k 30 "$t" python3 scripts/depth_decode_probe.py --base "http://$H1:8000" --label "$tag" --corpus "$CORPUS" \
      --concurrency "$c" --repeats "$r" --temperature 0 --out "$d/probe-$tag.json" "$@" > "$d/probe-$tag.log" 2>&1 ); rc=$?
  echo "$rc $(( $(date +%s) - s )) $t" > "$d/time-$tag.txt"; pos > "$d/pos-$tag.after"
  [ $rc -eq 0 ] || rm -f "$d/probe-$tag.json"; log "$(basename "$d"): $tag exit=$rc"; }
run_pass() { # dir
  local d=$1 spec tag depth c r off t x
  probe "$d" warm 4096 1 1 900000 300
  for spec in $CELLS; do IFS=: read -r tag depth c r off t <<< "$spec"; probe "$d" "$tag" "$depth" "$c" "$r" "$off" "$t"; done
  for x in a b; do ( cd "$R" && timeout 300 python3 scripts/logits_equiv.py capture --base-url "http://$H1:8000" --out "$d/logits-$x.json" > "$d/logits-$x.log" 2>&1 ); done
  ( cd "$R" && child timeout -k 30 1200 uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$H1:8000/v1" --model $MODEL \
      --tokenizer "$SNAP" --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 --concurrency 1 4 8 --runs 4 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$H1:8000/metrics" \
      --save-result "$d/task.csv" > "$d/benchy.log" 2>&1 ); log "$(basename "$d"): benchy exit=$?"
  [ "$(health localhost)" = 200 ] || { echo "server gone after the pass" > "$d/DIED"; log "$(basename "$d"): server DIED"; }; }


for item in ctl:1 arm:1 arm:2 ctl:2; do
  who=${item%:*}; k=${item#*:}; d=$S/$who-p$k; rm -rf "$d"; mkdir -p "$d"; st "running: screen $who-p$k"
  # arm boots may recompile the drafter graph (k70 arm-p1 did): cold allowed, boot.txt records it
  if [ $who = arm ]; then tboot "$ARM" "$d" 1; else tboot "$CTL" "$d" 0; fi; rc=$?
  if [ $rc -ne 0 ]; then keep_logs "$d"; [ $rc -eq 2 ] && echo cold > "$d/COLD" || echo failed > "$d/FAILED"; stop_all; break; fi
  run_pass "$d"; keep_logs "$d"; stop_all
  if [ "$item" = arm:1 ]; then TD_ACC_RISE_OK=1 python3 "$R/scripts/thunderdome_report.py" "$S" --acc-only > "$S/acc-p1.txt" 2>&1
    [ $? -eq 3 ] && { log "acceptance dropped past 0.03 in pass 1, pass 2 skipped"; break; }; fi; done
TD_ACC_RISE_OK=1 python3 "$R/scripts/thunderdome_report.py" "$S" > "$S/verdict.txt" 2>&1
V=$(sed -n 's/^VERDICT=//p' "$S/verdict.txt"); V=${V:-ERROR}
tg4=$(python3 - "$S" <<'PY'
import glob, os, re, statistics, sys
row = re.compile(r"^\|\s*[^|]+\|\s*tg512 \(c4\)\s*\|\s*([\d.]+)")
v = {w: [float(m.group(1)) for f in sorted(glob.glob(os.path.join(sys.argv[1], f"{w}-p[0-9]/task.csv"))) for m in map(row.match, open(f)) if m] for w in ("ctl", "arm")}
c, a = v["ctl"], v["arm"]
print(f"benchy tg512 (c4): control {'/'.join(f'{x:.1f}' for x in c)} arm {'/'.join(f'{x:.1f}' for x in a)} t/s"
      + (f", arm vs control {100 * (statistics.mean(a) / statistics.mean(c) - 1):+.2f}%, control boot-to-boot {100 * abs(c[1] / c[0] - 1):.2f}%" if len(c) >= 2 and a else " (incomplete)"))
PY
)
{ echo; echo "== screen (Thunderdome rules on the pair, ctl/arm/arm/ctl): VERDICT=$V"; cat "$S/verdict.txt"; echo "$tg4"; } >> "$RES/k71.txt"
log "screen: $V; $tg4"

GV="not run (screen $V)"
if [ "$V" = PROMOTE ]; then
  st "running: gate"; g=$RES/gate; mkdir -p "$g"
  if tboot "$ARM" "$g" 1; then
    ( cd "$R" && child timeout 900 python3 scripts/straggler_probe.py 5 6 7 8 12 16 > "$g/straggler.log" 2>&1 )
    child timeout 7200 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://localhost:8000 --model $MODEL > "$g/hardmode.log" 2>&1
    child timeout 3600 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://localhost:8000/v1 --model $MODEL --scenarios TC-45 --trials 5 > "$g/tc45.log" 2>&1
    ( cd "$R" && child timeout 5400 python3 scripts/fidelity_probe.py --base http://localhost:8000 --model $MODEL --depths 8000,32000,64000,128000 --out "$g/fidelity.json" > "$g/fidelity_probe.txt" 2>&1 )
    for sd in 11 13; do ( cd "$R" && child timeout 2700 python3 scripts/fidelity_probe.py --base http://localhost:8000 --model $MODEL --depths 128000 --seed $sd --out "$g/fidelity-seed$sd.json" > "$g/fidelity-seed$sd.txt" 2>&1 ); done
    keep_logs "$g"; stop_all
    t=$(mktemp -d); ln -s "$g" "$t/cand1"; ln -s "$g" "$t/cand2"
    GV=$(python3 -c "import sys; sys.path.insert(0, sys.argv[1]); import k61; b, hm = k61.gate(sys.argv[2]); print('PASS' if not b else 'FAIL (' + '; '.join(b) + ')')" "$K70" "$t" 2>&1); rm -rf "$t"
    { echo; echo "== gate (arm, one TP2 boot): $GV"; grep -hE "Quality:" "$g/hardmode.log"; grep -hE "Pass\^5|Score:" "$g/tc45.log" | head -2
      grep -h "^depth" "$g/fidelity_probe.txt" "$g"/fidelity-seed1?.txt; tail -6 "$g/straggler.log"; } >> "$RES/k71.txt"
  else keep_logs "$g"; stop_all; GV="FAIL (gate boot failed)"; echo "== gate: boot failed" >> "$RES/k71.txt"; fi
  log "gate: $GV"; fi
for l in dgx01 dgx02; do echo "guard $l: min MemAvailable $(awk '$2 ~ /^[0-9]+$/' "$RES/guard-$l.log" | sort -k2 -n | head -1 | awk '{printf "%.2f GiB", $2/1024}')"; done >> "$RES/k71.txt"
FINAL="DONE: k71 TP2 refit drafter, control plans pinned, screen=$V gate=${GV%% (*} ($RES/k71.txt)"
acc=$(sed -n '/^== acceptance/,/^==/p' "$S/verdict.txt" | grep -E '^ +pos ')
cells=$(sed -n '/^== cells/,/^==/p' "$S/verdict.txt" | grep -E '^ +(probe|wall|benchy) ')
{ echo "k71 result: the k70 arm (refit run 1 drafter on the 2x b1.4) rerun with the control's b12x plan selections pinned into its cache, so the only difference from b1.4 is the 24 retrained mtp.* tensors. Boots in the order b1.4, arm, arm, b1.4 on the pair, Thunderdome rules (noise = control boot-to-boot, 1% floor), llama-benchy with 4 runs per cell."
  echo; echo "Screen verdict: $V. Gate: $GV."
  echo; echo "Acceptance per draft position (T=0 probe cells, pooled, b1.4 then arm):"; echo '```'; echo "$acc"; echo '```'
  echo; echo "Cells (arm vs b1.4, noise band):"; echo '```'; echo "$cells"; echo "$tg4"; echo '```'
  echo; case $V in
    PROMOTE) [ "${GV%% *}" = PASS ] && echo "The arm wins on the 2x and passed the TP2 gate. I plan to publish it as the next 2x recipe." \
               || echo "The arm wins on speed but the gate did not pass ($GV), so the 2x stays on b1.4 for now." ;;
    KILL) echo "The arm still loses on the 2x with identical plans, so the c1 loss is not the plan confound. The 2x stays on b1.4." ;;
    *) echo "No clear result on the 2x ($V), so the 2x stays on b1.4 and the gate was not run." ;; esac
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
exit 0
