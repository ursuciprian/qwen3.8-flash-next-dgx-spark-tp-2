#!/usr/bin/env bash
# Thunderdome (#83): one-hour paired screen for single-Spark (TP=1) arms. Two arms enter, one arm leaves.
# Doc: docs/THUNDERDOME.md. Runs on dgx-01; dgx-02 is driven over ssh.
#
# screen (default): each Spark screens its own arm against a control on that Spark. Four fresh warm boots
#   per Spark in the order control, arm, arm, control (pass 1 = boots 1+2, pass 2 = boots 3+4), so node
#   bias and linear drift (PLE page-cache residency grows boot to boot) cancel. Per boot: warm-up request,
#   paired temperature-0 probes (fresh c1/c4/c8, d16k c4, count c8: 2 reps; d16k c8: 1 rep cut at 480 s and
#   judged on wall time, since the 6 GiB KV pool churns there; same prompts in every boot), logits_equiv.py
#   capture twice, quick llama-benchy (pp2048 c1, tg512 c1/c8; T=1, 2 runs).
#   thunderdome_report.py writes <node dir>/verdict.txt, last line VERDICT=KILL|PROMOTE|INCONCLUSIVE.
#   A Spark whose acceptance already moved past 0.03 after pass 1 skips pass 2 (KILL).
#   PROMOTE arms then get the split gate (unless GATE=0).
# gate: arm on both Sparks; dgx-01 stragglers + hardmode + TC-45 x5, dgx-02 fidelity 8k-128k + 128k
#   seeds 11/13, in parallel. Writes summary.txt; the gate is summarized, not judged.
# bake: cold boot of each recipe on both Sparks at once (fills both compile caches), then stop.
#
# Usage:
#   ARM_A=<arm.yaml> ARM_B=<arm.yaml> RES=<dir> bash scripts/thunderdome.sh [--dry-run]
#   RES=<dir> bash scripts/thunderdome.sh gate <recipe.yaml> [--dry-run]     -> $RES/gate-<name>
#   RES=<dir> bash scripts/thunderdome.sh bake <recipe.yaml>... [--dry-run]  -> $RES/bake-<name>
# Env:
#   ARM_A, ARM_B       arm recipes screened on dgx-01 / dgx-02 (one may be empty)
#   CONTROL            control recipe for both Sparks (default: the v3b recipe); CONTROL_A / CONTROL_B per Spark
#   RES                results dir (required); OUT_A / OUT_B node dirs (default $RES/dgx01, $RES/dgx02)
#   GATE=0             no gate after a PROMOTE
#   GPU_LOCK_HELD=1    the caller holds ~/GEN-AI/gpu-lock; otherwise this script takes it (waits until free)
#   NO_RESTORE=1 or CHAIN_NO_RESTORE=1   leave the pair idle on exit instead of booting the 2x recipe
#   SHIPPED_TAG        image tag the 2x recipe serves (restore check; default: the 2x image at start)
# Warm images only: refuses (exit 2, before touching the pair) when a recipe is not TP=1, or its image is
# missing on a Spark that boots it, or has no baked compile seed (/opt/b12x-seed/torch_compile_cache in the
# image history). A boot whose backbone still compiles (AOT key not in the cache) is stopped at once and
# that Spark's verdict is INCONCLUSIVE (cold): run `thunderdome.sh bake <recipe>` once, then screen again.
# Exit: 0 done, 1 failed, 2 refused. STATE: $RES/STATE. Log: $RES/thunderdome.log.
# Stop: kill -TERM <pid> (logged at start). Never pkill -f.
set -u
export PATH="$HOME/.local/bin:$PATH"
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
H1=192.168.100.62; H2=192.168.100.53; MODEL=qwen3.8-flash-next
SNAP=${SNAP:-$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd}
CORPUS=${CORPUS:-$R/results/corpus-code.txt}
BENCHY_SRC=${BENCHY_SRC:-$HOME/GEN-AI/llama-benchy-fork}
LOCKF=$HOME/GEN-AI/gpu-lock
V3B=$R/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-v3b.yaml
[ -f "$V3B" ] || V3B=$R/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml
GEN="List the numbers from 1 to 300 separated by commas. Output only the numbers, nothing else, no commentary."
ORDER="ctl:1 arm:1 arm:2 ctl:2"
# tag:depth:concurrency:reps:corpus offset:timeout s:estimate s. Offsets as r8_tp1_driver.sh, identical in every
# boot. Estimates from the 2026-10-02/03 TP1 ABBA driver logs (v2/v3a, both Sparks); dry-run plan only.
# d16k-c8 on the 6 GiB pool: one rep took 134-441 s when it finished, and ~40% of runs hit the 900 s request
# timeout; hence 1 rep, cut at 480 s, judged on wall time (the cut counts as the time of an unfinished run).
CELLS="fresh-c1:256:1:2:104729:300:45 fresh-c4:256:4:2:314187:400:85 fresh-c8:256:8:2:418916:500:120
d16k-c4:16384:4:2:1361477:700:150 d16k-c8:16384:8:1:1466206:480:330 count-c8:0:8:2:0:300:20"
EST_BOOT=180 EST_WARM=25 EST_LOGITS=30 EST_BENCHY=120 EST_LOGS=15 EST_GATE_BOOT=200 EST_GATE01=1500 EST_GATE02=1100 EST_RESTORE=240

DRY=; ARGS=()
for a in "$@"; do if [ "$a" = --dry-run ]; then DRY=1; else ARGS+=("$a"); fi; done
set -- ${ARGS[@]+"${ARGS[@]}"}
MODE=screen
case "${1:-}" in screen|gate|bake) MODE=$1; shift ;; "") ;; *) echo "usage: see the header of $0" >&2; exit 2 ;; esac
abs() { case $1 in ""|/*) echo "$1" ;; *) echo "$PWD/$1" ;; esac; }   # recipes are booted from their own dir
P=(); for a in "$@"; do P+=("$(abs "$a")"); done; set -- ${P[@]+"${P[@]}"}
[ -n "${RES:-}" ] || { echo "set RES to a results dir" >&2; exit 2; }; RES=$(abs "$RES")
OUT_A=$(abs "${OUT_A:-$RES/dgx01}"); OUT_B=$(abs "${OUT_B:-$RES/dgx02}")
ARM_A=$(abs "${ARM_A:-}"); ARM_B=$(abs "${ARM_B:-}"); GATE=${GATE:-1}
CTL_A=$(abs "${CONTROL_A:-${CONTROL:-$V3B}}"); CTL_B=$(abs "${CONTROL_B:-${CONTROL:-$V3B}}")
LOG=$RES/thunderdome.log

log() { local m; m="[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*"; if [ -n "$DRY" ]; then echo "$m"; else echo "$m" >> "$LOG"; fi; }
st() { echo "$*" > "$RES/STATE"; log "STATE: $*"; }
on() { local h=$1; shift; if [ "$h" = "$H1" ]; then bash -c "$*" < /dev/null; else ssh -n -o ConnectTimeout=10 "$h" "$*"; fi; }
label() { [ "$1" = "$H1" ] && echo dgx01 || echo dgx02; }
name() { basename "$1" .yaml | sed 's/^qwen3\.8-flash-next-1x-dgx-spark-//'; }
image() { sed -n 's/^container: *//p' "$1" | head -1 | tr -d "\"' "; }
health() { curl -s -m 5 -o /dev/null -w '%{http_code}' "$1:8000/health"; }
pong() { curl -s -m 120 "$1:8000/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Reply with exactly one word: pong"}],"max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>&1; }
# the two Spark loops share sparkrun state; fd 9 (gpu-lock) never reaches sparkrun or the containers
sr() { flock "$RES/.sparkrun.lock" timeout --foreground 900 sparkrun "$@" 9>&-; }
solo() { on "$1" "docker ps --format '{{.Names}}' | grep -E '_solo|sparkrun' | head -1"; }
cx() { local h=$1 c; c=$(solo "$h"); shift; [ -n "$c" ] && on "$h" "docker exec $c sh -c $(printf %q "$*")"; }
busy() { [ -n "$(on "$1" "docker ps -q --filter name=sparkrun")" ]; }
stop_host() { busy "$1" || return 0; sr stop --all --hosts "$1" >> "$LOG" 2>&1; local i
  for i in $(seq 1 36); do busy "$1" || return 0; sleep 5; done
  on "$1" "docker ps -q --filter name=sparkrun | xargs -r docker rm -f" >> "$LOG" 2>&1; }
stop_all() { { busy $H1 || busy $H2; } || return 0; sr stop --all >> "$LOG" 2>&1; stop_host $H1; stop_host $H2; }
wait_health() { local s; s=$(date +%s); until [ "$(health "$1")" = 200 ]; do
  [ $(( $(date +%s) - s )) -gt "$2" ] && return 1; sleep 15; done; }
MEMLOOP='while true; do echo "$(date +%T) $(awk '"'"'/^MemAvailable/{printf "%.2f", $2/1048576}'"'"' /proc/meminfo)"; sleep 5; done'
memlog() { if [ "$1" = "$H1" ]; then bash -c "$MEMLOOP" > "$2" 2>&1 < /dev/null 9>&- &
  else ssh -n "$1" "$MEMLOOP" > "$2" 2>&1 9>&- & fi; echo $!; }
minmem() { sort -k2 -n "$1" 2>/dev/null | head -1; }
tree() { local c; echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
kill_tree() { kill -TERM $(tree "$1") 2>/dev/null; }   # whole tree at once, so nothing new is spawned in between

keep_logs() { # host dir: serve log, AOT lines, env, image (also after a failed boot)
  local h=$1 d=$2 c; c=$(on "$h" "docker ps -a --format '{{.Names}}' | grep -E '_solo|sparkrun' | head -1")
  [ -n "$c" ] || return 0
  on "$h" "docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1" > "$d/serve.log" 2>&1
  grep -E "Directly load AOT|saved AOT|Dynamo bytecode" "$d/serve.log" | cut -c1-300 > "$d/aot.txt"
  on "$h" "docker exec $c sh -c \"env | grep -E '^(VLLM|B12X)_' | sort\"" > "$d/env.txt" 2>&1
  on "$h" "docker inspect --format '{{.Config.Image}}' $c" > "$d/image.txt" 2>&1; }

boot() { # host recipe dir -> 0 up and warm, 1 failed, 2 cold (backbone compiling; ALLOW_COLD=1 accepts it)
  local h=$1 rec=$2 d=$3 s m n
  stop_host "$h"; cp "$rec" "$d/recipe.yaml"
  log "$(label "$h"): boot $(name "$rec") ($(image "$rec")) -> $d"
  ( cd "$(dirname "$rec")" && sr run "$rec" --hosts "$h" --solo --no-follow ) >> "$d/sparkrun.log" 2>&1
  s=$(date +%s)
  while true; do
    # warm boots load the backbone AOT first; a cold one starts with a Dynamo transform of the backbone
    m=$(cx "$h" "grep -m1 -oE 'Directly load AOT compilation|Dynamo bytecode transform time' /tmp/sparkrun_serve.log" 2>/dev/null)
    if [ "$m" = "Dynamo bytecode transform time" ] && [ -z "${ALLOW_COLD:-}" ]; then
      log "$(label "$h"): $(name "$rec") COLD: backbone compiling, boot stopped (run: thunderdome.sh bake $rec)"; return 2; fi
    if [ "$(health "$h")" = 200 ]; then
      echo "boot $(( $(date +%s) - s ))s ${m:-no AOT line}" > "$d/boot.txt"; log "$(label "$h"): up, $(cat "$d/boot.txt")"; return 0; fi
    n=$(solo "$h")
    if [ $(( $(date +%s) - s )) -gt 120 ] && { [ -z "$n" ] || cx "$h" "grep -qE 'Worker failed with error|EngineCore failed to start|Engine core initialization failed' /tmp/sparkrun_serve.log"; }; then
      log "$(label "$h"): boot FAILED after $(( $(date +%s) - s ))s"; return 1; fi
    [ $(( $(date +%s) - s )) -ge "${BOOT_MAX:-1800}" ] && { log "$(label "$h"): health timeout"; return 1; }
    sleep 10; done; }

pos() { curl -s -m 10 "$1:8000/metrics" | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens(_per_pos)?_total'; }
probe() { # host dir tag depth conc reps offset timeout: one depth_decode_probe.py cell, T=0
  local h=$1 d=$2 tag=$3 depth=$4 c=$5 r=$6 off=$7 t=$8 rc s
  if [ "${tag#count}" != "$tag" ]; then set -- --prompt "$GEN" --max-tokens 320
  else set -- --depth "$depth" --new 2048 --offset "$off" --max-tokens 512; fi
  pos "$h" > "$d/pos-$tag.before"; s=$(date +%s)
  ( cd "$R" && timeout -k 30 "$t" python3 scripts/depth_decode_probe.py --base "http://$h:8000" --label "$tag" --corpus "$CORPUS" \
      --concurrency "$c" --repeats "$r" --temperature 0 --out "$d/probe-$tag.json" "$@" > "$d/probe-$tag.log" 2>&1 ); rc=$?
  echo "$rc $(( $(date +%s) - s )) $t" > "$d/time-$tag.txt"; pos "$h" > "$d/pos-$tag.after"
  [ $rc -eq 0 ] || rm -f "$d/probe-$tag.json"   # a cut-short cell counts as missing
  log "$(label "$h")/$(basename "$d"): $tag exit=$rc"; }
benchy() { # host dir: pp2048 tg512 at c1 and c8, depth 0, T=1 (as the driver grid), 2 runs
  local h=$1 d=$2 rc
  ( cd "$R" && timeout -k 30 900 uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$h:8000/v1" --model $MODEL \
      --tokenizer "$SNAP" --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 --concurrency 1 8 --runs 2 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$h:8000/metrics" \
      --save-result "$d/task.csv" > "$d/benchy.log" 2>&1 ); rc=$?
  log "$(label "$h")/$(basename "$d"): benchy exit=$rc"; }
run_pass() { # host dir: everything measured on one boot
  local h=$1 d=$2 mp spec tag depth c r off t e x
  mp=$(memlog "$h" "$d/mem.log")
  probe "$h" "$d" warm 4096 1 1 900000 300
  for spec in $CELLS; do IFS=: read -r tag depth c r off t e <<< "$spec"; probe "$h" "$d" "$tag" "$depth" "$c" "$r" "$off" "$t"; done
  for x in a b; do ( cd "$R" && timeout 300 python3 scripts/logits_equiv.py capture --base-url "http://$h:8000" \
      --out "$d/logits-$x.json" > "$d/logits-$x.log" 2>&1 ); done
  benchy "$h" "$d"
  kill "$mp" 2>/dev/null; }

node_run() { # host control arm outdir: ABBA on one Spark, then its verdict
  local h=$1 ctl=$2 arm=$3 out=$4 item who k d rc
  trap 'exit 143' TERM
  mkdir -p "$out"; echo "arm $(name "$arm") $arm" > "$out/arms.txt"; echo "ctl $(name "$ctl") $ctl" >> "$out/arms.txt"
  for item in $ORDER; do
    who=${item%:*}; k=${item#*:}; d=$out/$who-p$k; rm -rf "$d"; mkdir -p "$d"
    if [ "$who" = arm ]; then boot "$h" "$arm" "$d"; else boot "$h" "$ctl" "$d"; fi; rc=$?
    if [ $rc -ne 0 ]; then
      keep_logs "$h" "$d"; if [ $rc -eq 2 ]; then echo cold > "$d/COLD"; else echo failed > "$d/FAILED"; fi
      stop_host "$h"; break; fi
    run_pass "$h" "$d"; keep_logs "$h" "$d"; stop_host "$h"
    if [ "$item" = arm:1 ]; then python3 "$R/scripts/thunderdome_report.py" "$out" --acc-only > "$out/acc-p1.txt" 2>&1
      [ $? -eq 3 ] && { log "$(label "$h"): acceptance moved past 0.03 in pass 1, pass 2 skipped"; break; }; fi
  done
  python3 "$R/scripts/thunderdome_report.py" "$out" > "$out/verdict.txt" 2>&1
  log "$(label "$h"): $(name "$arm") $(tail -1 "$out/verdict.txt") | $(grep -E '^(reason|better):' "$out/verdict.txt" | head -4 | tr '\n' ' ')"; }

gate() { # recipe outdir: split gate on both Sparks
  local rec=$1 G=$2 n1 n2 r1 r2 m1 m2 p1 p2 sd
  mkdir -p "$G/dgx01" "$G/dgx02"; cp "$rec" "$G/recipe.yaml"
  boot $H1 "$rec" "$G/dgx01" 9>&- & n1=$!; boot $H2 "$rec" "$G/dgx02" 9>&- & n2=$!; BG="$BG $n1 $n2"
  wait $n1; r1=$?; wait $n2; r2=$?
  if [ $r1 -ne 0 ] || [ $r2 -ne 0 ]; then keep_logs $H1 "$G/dgx01"; keep_logs $H2 "$G/dgx02"; stop_host $H1; stop_host $H2
    echo "gate boot failed (dgx01 rc=$r1, dgx02 rc=$r2; 2 = cold)" > "$G/summary.txt"; log "gate $(name "$rec"): boot failed"; return 1; fi
  m1=$(memlog $H1 "$G/dgx01/mem.log"); m2=$(memlog $H2 "$G/dgx02/mem.log")
  ( cd "$R" && timeout 900 python3 scripts/straggler_probe.py 5 6 7 8 12 16 > "$G/dgx01/straggler.log" 2>&1   # localhost = dgx-01
    timeout 7200 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 \
      --base-url http://localhost:8000 --model $MODEL > "$G/dgx01/hardmode.log" 2>&1
    timeout 3600 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 \
      --base-url http://localhost:8000/v1 --model $MODEL --scenarios TC-45 --trials 5 > "$G/dgx01/tc45.log" 2>&1 ) 9>&- & p1=$!
  ( cd "$R" && timeout 5400 python3 scripts/fidelity_probe.py --base "http://$H2:8000" --model $MODEL \
      --depths 8000,32000,64000,128000 --out "$G/dgx02/fidelity.json" > "$G/dgx02/fidelity.txt" 2>&1
    for sd in 11 13; do timeout 2700 python3 scripts/fidelity_probe.py --base "http://$H2:8000" --model $MODEL \
      --depths 128000 --seed $sd --out "$G/dgx02/fidelity-seed$sd.json" > "$G/dgx02/fidelity-seed$sd.txt" 2>&1; done ) 9>&- & p2=$!
  BG="$BG $m1 $m2 $p1 $p2"; wait $p1; wait $p2; kill $m1 $m2 2>/dev/null
  keep_logs $H1 "$G/dgx01"; keep_logs $H2 "$G/dgx02"; stop_host $H1; stop_host $H2
  { echo "gate $(name "$rec") $(image "$rec")"
    grep -hE "Quality:" "$G/dgx01/hardmode.log"; grep -hE "Pass\^5|Score:" "$G/dgx01/tc45.log" | head -2
    grep -h "^depth" "$G/dgx02/fidelity.txt" "$G"/dgx02/fidelity-seed1?.txt
    tail -6 "$G/dgx01/straggler.log"
    echo "min MemAvailable dgx01 $(minmem "$G/dgx01/mem.log") dgx02 $(minmem "$G/dgx02/mem.log")"; } > "$G/summary.txt" 2>&1
  log "gate $(name "$rec"): $(tr '\n' ';' < "$G/summary.txt" | cut -c1-400)"; }

# ---- 2x restore (k31 pattern) ----
E=
is_shipped() { [ "$(health localhost)" = 200 ] && docker ps --format '{{.Image}}' | grep -q ":$E\$" \
  && on $H2 "docker ps --format '{{.Image}}'" | grep -q ":$E\$"; }
restore() {
  if is_shipped && pong localhost | grep -qi pong; then log "restore: 2x already serving"; return 0; fi
  stop_all; timeout -k 60 3600 sparkrun run qwen3.8-flash-next-2x-dgx-spark --no-follow >> "$LOG" 2>&1 9>&-
  wait_health localhost 3600; local p; p=$(pong localhost)
  if is_shipped && echo "$p" | grep -qi pong; then log "restore: 2x verified ($E), pong=$p"; return 0; fi
  log "FAILED: restore -- MANUAL INTERVENTION (pong=$p)"; return 1; }

# ---- preflight: read-only, before the lock and before anything stops ----
NPROB=0
refuse() { echo "REFUSED: $*"; NPROB=$((NPROB + 1)); }
check() { # host recipe seed(1|0)
  local h=$1 rec=$2 img
  [ -f "$rec" ] || { refuse "recipe $rec not found"; return; }
  grep -Eq '^ *tensor_parallel: *1 *$' "$rec" || { refuse "$(name "$rec"): not a TP=1 recipe (tensor_parallel: 1)"; return; }
  img=$(image "$rec"); [ -n "$img" ] || { refuse "$(name "$rec"): no container line"; return; }
  on "$h" "docker image inspect $img > /dev/null 2>&1" || { refuse "$(label "$h"): image $img ($(name "$rec")) missing"; return; }
  if [ "$3" = 1 ] && ! on "$h" "docker history --no-trunc --format '{{.CreatedBy}}' $img" 2>/dev/null | grep -q '/opt/b12x-seed/torch_compile_cache'; then
    refuse "$(label "$h"): image $img ($(name "$rec")) has no baked compile seed: warm images only"; return; fi
  echo "ok   $(label "$h"): $(name "$rec") $img$([ "$3" = 1 ] && echo ', compile seed baked')"; }
preflight() {
  local t; for t in sparkrun python3 curl flock uvx; do command -v $t > /dev/null || refuse "$t not on PATH"; done
  ip -o -4 addr show 2>/dev/null | grep -q "inet $H1/" || refuse "run this on dgx-01 ($H1)"
  [ "${GPU_LOCK_HELD:-}" = 1 ] || [ -e "$LOCKF" ] || refuse "$LOCKF missing"
  case $MODE in
  screen)
    [ -n "$ARM_A$ARM_B" ] || refuse "set ARM_A and/or ARM_B"
    [ -s "$CORPUS" ] || refuse "corpus $CORPUS missing"; [ -d "$SNAP" ] || refuse "tokenizer snapshot $SNAP missing"
    [ "$GATE" = 0 ] || command -v tool-eval-bench > /dev/null || refuse "tool-eval-bench not on PATH (or GATE=0)"
    if [ -n "$ARM_A" ]; then check $H1 "$CTL_A" 1; check $H1 "$ARM_A" 1; [ "$GATE" = 0 ] || check $H2 "$ARM_A" 1; fi
    if [ -n "$ARM_B" ]; then check $H2 "$CTL_B" 1; check $H2 "$ARM_B" 1; [ "$GATE" = 0 ] || check $H1 "$ARM_B" 1; fi ;;
  gate) [ $# -eq 1 ] || refuse "gate takes one recipe"; command -v tool-eval-bench > /dev/null || refuse "tool-eval-bench not on PATH"
    for t in "$@"; do check $H1 "$t" 1; check $H2 "$t" 1; done ;;
  bake) [ $# -ge 1 ] || refuse "bake takes recipes"; for t in "$@"; do check $H1 "$t" 0; check $H2 "$t" 0; done ;;
  esac; }

plan() { # dry run: what would run, and how long
  local per=$((EST_BOOT + EST_WARM + EST_LOGITS + EST_BENCHY + EST_LOGS)) cells="" spec tag depth c r off t e n=0 g
  for spec in $CELLS; do IFS=: read -r tag depth c r off t e <<< "$spec"; per=$((per + e)); cells="$cells $tag(${r}x c$c ~${e}s)"; done
  g=$((EST_GATE_BOOT + (EST_GATE01 > EST_GATE02 ? EST_GATE01 : EST_GATE02)))
  echo "== Thunderdome $MODE, dry run: nothing is locked, stopped or booted"
  echo "repo      $R"; echo "results   $RES"
  echo "lock      $([ "${GPU_LOCK_HELD:-}" = 1 ] && echo "held by the caller (GPU_LOCK_HELD=1)" || echo "takes $LOCKF, waits until free")"
  echo "on exit   $([ -n "${NO_RESTORE:-}${CHAIN_NO_RESTORE:-}" ] && echo "pair left idle (NO_RESTORE/CHAIN_NO_RESTORE)" || echo "boots qwen3.8-flash-next-2x-dgx-spark and checks image + pong (~$((EST_RESTORE / 60)) min)")"
  case $MODE in
  screen)
    [ -n "$ARM_A" ] && { echo "dgx01     arm $(name "$ARM_A") vs ctl $(name "$CTL_A"), boots: $ORDER -> $OUT_A"; n=1; }
    [ -n "$ARM_B" ] && { echo "dgx02     arm $(name "$ARM_B") vs ctl $(name "$CTL_B"), boots: $ORDER -> $OUT_B"; n=$((n + 1)); }
    echo "per boot  boot ~${EST_BOOT}s, warm-up ~${EST_WARM}s,$cells, logits a/b ~${EST_LOGITS}s, benchy pp2048 c1 + tg512 c1/c8 x2 ~${EST_BENCHY}s, logs ~${EST_LOGS}s"
    echo "          = ~$((per / 60)) min per boot, 4 boots per Spark, Sparks in parallel"
    echo "screen    ~$((4 * per / 60 + 1)) min wall for $n arm(s)$([ "$GATE" = 0 ] && echo ", no gate (GATE=0)" || echo "; + ~$((g / 60)) min split gate per PROMOTE arm (sequential)")"
    echo "verdict   <node dir>/verdict.txt: KILL (cell < -2x noise, acceptance shift > 0.03, arm boot failed) /"
    echo "          PROMOTE (no cell < -1x noise, >= 1 cell > +1x noise, all cells measured) / INCONCLUSIVE"
    for spec in $CELLS; do IFS=: read -r tag depth c r off t e <<< "$spec"; [ "$tag" = d16k-c8 ] && break; done
    echo "example   timeout $t python3 scripts/depth_decode_probe.py --base http://$H1:8000 --label $tag --corpus $CORPUS \\"
    echo "            --concurrency $c --repeats $r --temperature 0 --depth $depth --new 2048 --offset $off --max-tokens 512 --out <boot dir>/probe-$tag.json" ;;
  gate) echo "gate      $(name "$1"): boot on both Sparks ~$((EST_GATE_BOOT / 60)) min; dgx01 stragglers + hardmode + TC-45 x5 ~$((EST_GATE01 / 60)) min || dgx02 fidelity 8k-128k + 128k seeds 11/13 ~$((EST_GATE02 / 60)) min"
    echo "          = ~$((g / 60)) min -> $RES/gate-$(name "$1")/summary.txt" ;;
  bake) echo "bake      $*: cold boot on both Sparks at once, up to 90 min each (typ. 6-7 min), then stop" ;;
  esac; }

echo "== preflight"
preflight "$@"
[ $NPROB -eq 0 ] || { echo "$NPROB problem(s): nothing started"; exit 2; }
[ -n "$DRY" ] && { plan "$@"; exit 0; }

# ---- run ----
mkdir -p "$RES"
BG=; OWNS=0; FINAL="FAILED: $MODE"
cleanup() {
  trap '' TERM INT HUP
  local p; for p in $BG; do kill_tree "$p"; done
  [ "$OWNS" = 1 ] || { st "$FINAL (stopped before taking the pair)"; return; }
  if [ -n "${NO_RESTORE:-}${CHAIN_NO_RESTORE:-}" ]; then stop_host $H1; stop_host $H2; log "restore: skipped, pair left idle"
  else restore || FINAL="FAILED: restore -- $FINAL"; fi
  st "$FINAL"; }
trap cleanup EXIT
trap 'log "signal: stopping"; FINAL="FAILED: stopped by signal during $MODE"; exit 143' TERM INT HUP
log "==== thunderdome $MODE pid $$ pgid $(ps -o pgid= $$ | tr -d ' ') ARM_A=$ARM_A ARM_B=$ARM_B CTL_A=$CTL_A CTL_B=$CTL_B $*"
if [ "${GPU_LOCK_HELD:-}" = 1 ]; then log "gpu-lock held by the caller"
else st "waiting: gpu-lock"; exec 9> "$LOCKF"; flock 9 & BG="$BG $!"; wait $! || exit 1; log "gpu-lock taken"; fi
E=${SHIPPED_TAG:-$(docker ps --format '{{.Names}} {{.Image}}' | awk '/node_0/ && /spark-vllm-b12x/ {print $2; exit}' | sed 's|.*:||')}
{ [ -n "$E" ] && on $H2 "docker ps --format '{{.Image}}'" | grep -q ":$E\$"; } || E=${SHIPPED_TAG:-b1.4-20261001-b7fbaf96-a7e649d8-warm}
log "2x image tag for the restore check: $E"
OWNS=1; stop_all

case $MODE in
bake)
  for rec in "$@"; do d=$RES/bake-$(name "$rec"); mkdir -p "$d/dgx01" "$d/dgx02"; st "running: bake $(name "$rec")"
    ALLOW_COLD=1 BOOT_MAX=5400 boot $H1 "$rec" "$d/dgx01" 9>&- & n1=$!; ALLOW_COLD=1 BOOT_MAX=5400 boot $H2 "$rec" "$d/dgx02" 9>&- & n2=$!
    BG="$BG $n1 $n2"; wait $n1; r1=$?; wait $n2; r2=$?
    for h in $H1 $H2; do keep_logs $h "$d/$(label $h)"; stop_host $h; done
    log "bake $(name "$rec"): dgx01 rc=$r1 dgx02 rc=$r2; AOT saved: $(cat "$d"/dgx0?/aot.txt | grep -c 'saved AOT')"
    [ $r1 -eq 0 ] && [ $r2 -eq 0 ] || { FINAL="FAILED: bake $(name "$rec") (dgx01 rc=$r1, dgx02 rc=$r2)"; exit 1; }; done
  FINAL="DONE: baked $*" ;;
gate)
  st "running: gate $(name "$1")"; gate "$1" "$RES/gate-$(name "$1")" || exit 1
  FINAL="DONE: gate $(name "$1") ran ($RES/gate-$(name "$1")/summary.txt)" ;;
screen)
  st "running: screen"; PA=; PB=
  [ -n "$ARM_A" ] && { node_run $H1 "$CTL_A" "$ARM_A" "$OUT_A" 9>&- & PA=$!; }
  [ -n "$ARM_B" ] && { node_run $H2 "$CTL_B" "$ARM_B" "$OUT_B" 9>&- & PB=$!; }
  BG="$BG $PA $PB"; [ -n "$PA" ] && wait $PA; [ -n "$PB" ] && wait $PB
  SUM=
  for x in A B; do arm=ARM_$x; out=OUT_$x; [ -n "${!arm}" ] || continue
    v=$(sed -n 's/^VERDICT=//p' "${!out}/verdict.txt" 2>/dev/null); v=${v:-ERROR}; SUM="$SUM $(name "${!arm}")=$v"
    if [ "$v" = PROMOTE ] && [ "$GATE" != 0 ]; then st "running: gate $(name "${!arm}")"
      gate "${!arm}" "${!out}/gate" && SUM="$SUM(gate ran)" || SUM="$SUM(gate boot failed)"; fi; done
  echo "$SUM" > "$RES/verdicts.txt"; FINAL="DONE:$SUM" ;;
esac
exit 0
