#!/usr/bin/env bash
# Single-Spark v3a chain (opus-v3, 2026-10-03). v3a = TP1 v2 + VLLM_PLE_MMAP_KEEPALIVE_MS=50 on image
# spark-vllm-b12x:v3a-5bf24021-7fa812b3 (vLLM exp/v3-tp1 7fa812b3, b12x 5bf24021).
# Run detached on dgx-01:
#   K19PID=<k19 flint_job pid> setsid nohup bash ~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/scripts/v3a_chain.sh \
#     > ~/GEN-AI/k21v3/chain.nohup 2>&1 < /dev/null &
# a. waits for ~/GEN-AI/k19/STATE = DONE|FAILED|SKIPPED (or the k19 job pid $K19PID gone; cap 24 h), then takes
#    ~/GEN-AI/gpu-lock and holds it to the end. Every stage runs under a hard timeout (this script re-execs
#    itself per stage); chain.log prints each stage's pgid (stop: kill -TERM -- -<launcher pgid>).
# b. bake: dgx-01 cold boot (runtime cache moved aside, plan seed kept = same plans as v2); its torch AOT cache
#    and plan files are baked into $IMG on both nodes; dgx-02 meanwhile runs the v2 benchy reference grid.
#    Then a warm boot of the v3a recipe on both nodes (AOT seed check).
# c. ab: r8_tp1_driver.sh ABBA v3a vs tp1-v2 (T=0), dgx-01 p1-2, dgx-02 p3-4, report-tp1-v3a.txt.
# d. kv10: ABBA tp1-v3a-kv10 vs tp1-v3a, BENCHY=1 (paired probes + llama-benchy every boot).
# e. gate: dgx-01 v3a gate_arm.sh (straggler, fidelity 8k-128k, hardmode, ...) + TC-45 x5 + 128k seeds 11/13;
#    dgx-02 v3a benchy grid.
# f. gdbake / gd / gdgate: v3a + gdnmse (local snapshot f4..03). Bake as b with the TP=1 f4..03 plan file kept
#    (into $IMGGD); ABBA vs tp1-v3a with BENCHY=1 (pp2048, tg512, 16k ctx_pp/ctx_tg); gate as e, with the
#    128k seeds 11/13 on dgx-02 in parallel.
# g. r9: TP=2 r9-pin rescreen (launch_r9.sh, image r9-85c4f24d-79f3d7c5) under the held lock.
# The drivers restore the shipped 2x recipe at their end; EXIT here restores it again unless it already serves.
# STATE: $RES/STATE (chain), $RES/{ab,kv10,gd}/STATE (drivers), $R/results/r9-pin-20261003/STATE.
set -u
export PATH="$HOME/.local/bin:$PATH"
R=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2; K=$HOME/GEN-AI/k21v3
RES=${RES:-$R/results/v3a-tp1-20261003}
AD=$HOME/GEN-AI/r4-recipes/archive/recipes/qwen3.8-flash-next
H1=192.168.100.62; H2=192.168.100.53; MODEL=qwen3.8-flash-next
IMG0=spark-vllm-b12x:v3a-5bf24021-7fa812b3; IMG=$IMG0-warm; IMGGD=$IMG0-gd-warm
E=b1.4-20261001-b7fbaf96-a7e649d8-warm
RCD=$HOME/.cache/sparkrun/runtime-cache/vllm
RC=$RCD/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-2d9615ab
RCGD=$RCD/f400000000000000000000000000000000000003-b481aa11
GDPLAN=8bb5a5b317a3975df599d900e295b12b6ee7225bd8eff6f0de3adc67fd55202e.json   # TP=1 f4..03 namespace
SNAP=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
V2=$R/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml
V3=$R/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-v3a.yaml
KV=$R/archive/recipes/arms/tp1/tp1-v3a-kv10.yaml
GD=$R/archive/recipes/arms/tp1/tp1-v3a-gdnmse.yaml
mkdir -p "$RES"
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" >> "$RES/chain.log"; }
st() { echo "$*" > "$RES/STATE"; log "STATE: $*"; }
on() { if [ "$1" = $H1 ]; then shift; bash -c "$*" < /dev/null; else local h=$1; shift; ssh -n "$h" "$*"; fi; }
sr() { flock "$RES/.sparkrun.lock" sparkrun "$@"; }
health() { curl -s -m 5 -o /dev/null -w '%{http_code}' "$1:8000/health"; }
wait_health() { local s=$(date +%s); until [ "$(health "$1")" = 200 ]; do
  [ $(( $(date +%s) - s )) -gt "$2" ] && return 1; sleep 15; done; log "$1 health after $(( $(date +%s) - s ))s"; }
pong() { curl -s -m 120 "$1:8000/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Reply with exactly one word: pong"}],"max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>&1; }
serve_log() { local c; c=$(on "$1" "docker ps -a --format '{{.Names}}' | grep -E '_solo|sparkrun' | head -1")
  [ -n "$c" ] && on "$1" "docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1" > "$2" 2>&1; }
stop_host() { sr stop --all --hosts "$1" >> "$RES/chain.log" 2>&1; sleep 5; on "$1" "docker ps -q --filter name=sparkrun | xargs -r docker rm -f" >/dev/null 2>&1; }
stop_all() { sr stop --all >> "$RES/chain.log" 2>&1; sleep 10; stop_host $H1; stop_host $H2; }
MEMLOOP='while true; do echo "$(date +%T) $(awk '"'"'/^MemAvailable/{printf "%.2f", $2/1048576}'"'"' /proc/meminfo)"; sleep 5; done'
# samplers close fd 9 so an orphaned one can never hold gpu-lock
memlog() { if [ "$1" = $H1 ]; then bash -c "$MEMLOOP" > "$2" 2>&1 < /dev/null 9>&- & else ssh -n "$1" "$MEMLOOP" > "$2" 2>&1 9>&- & fi; echo $!; }
minmem() { sort -k2 -n "$1" | head -1; }
boot() { # host recipe outdir maxwait: boot solo, keep serve log on failure
  mkdir -p "$3"; cp "$2" "$3/recipe.yaml"; local s=$(date +%s)
  ( cd "$3" && sr run "$3/recipe.yaml" --hosts "$1" --solo --no-follow >> "$3/sparkrun.log" 2>&1 )
  if wait_health "$1" "$4"; then echo "boot $(( $(date +%s) - s ))s" > "$3/boot.txt"; log "$3 $(cat "$3/boot.txt") pong=$(pong "$1")"; return 0; fi
  serve_log "$1" "$3/serve.log"; log "$3 boot FAILED after $(( $(date +%s) - s ))s"; return 1; }
aot() { grep -E "Directly load AOT|saved AOT|Compil(ing|ation) .*took" "$1/serve.log" | cut -c1-300 > "$1/aot.txt"; }
benchy() { # host outdir
  curl -s -m 10 "$1:8000/metrics" | grep -E '^vllm:(num_preemptions|prefix_cache_(hits|queries))(_total)?' > "$2/counters.before"
  ( cd "$R" && timeout 4500 uvx --from "$HOME/GEN-AI/llama-benchy-fork" llama-benchy --base-url "http://$1:8000/v1" --model $MODEL --tokenizer "$SNAP" \
      --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 16384 --concurrency 1 2 4 8 --runs 3 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$1:8000/metrics" \
      --save-result "$2/task.csv" > "$2/task_grid.log" 2>&1 ); local rc=$?
  curl -s -m 10 "$1:8000/metrics" | grep -E '^vllm:(num_preemptions|prefix_cache_(hits|queries))(_total)?' > "$2/counters.after"
  serve_log "$1" "$2/serve.log"; log "$2 benchy exit=$rc"; }
bench_cell() { # host recipe outdir (run with &; own traps: async subshells reset them)
  local mp=; trap 'kill $mp 2>/dev/null' EXIT; trap 'exit 143' TERM
  mkdir -p "$3"; mp=$(memlog "$1" "$3/mem.log"); boot "$1" "$2" "$3" 1800 && benchy "$1" "$3"; stop_host "$1"; }
fid_cell() { # host recipe outdir: boot, 128k fidelity seeds 11/13
  local mp=; trap 'kill $mp 2>/dev/null' EXIT; trap 'exit 143' TERM
  mkdir -p "$3"; mp=$(memlog "$1" "$3/mem.log"); boot "$1" "$2" "$3" 1800 || return 1
  for sd in 11 13; do ( cd "$R" && python3 scripts/fidelity_probe.py --base "http://$1:8000" --model $MODEL \
    --depths 128000 --seed $sd --out "$3/fidelity-seed$sd.json" > "$3/fidelity-seed$sd.txt" 2>&1 ); done
  serve_log "$1" "$3/serve.log"; stop_host "$1"; }
arm_recipe() { sed "s|^name: .*|name: qwen3.8-flash-next-1x-dgx-spark-$2|" "$1" > "$AD/qwen3.8-flash-next-1x-dgx-spark-$2.yaml"; }
BACK=; B=; MP=
back_rc() { [ -n "$BACK" ] && [ -e "$BACK.v3a-keep" ] && { rm -rf "$BACK.v3a-cold"; mv "$BACK" "$BACK.v3a-cold" && mv "$BACK.v3a-keep" "$BACK" \
  && log "runtime cache $BACK restored (cold copy at $BACK.v3a-cold)"; }; BACK=; }
cold_bake() { # tag runtime-cache recipe base-image out-image [plan file to keep]
  local tag=$1 rc=$2 rec=$3 base=$4 out=$5 plan=${6:-} D=$RES/$1 X=$K/img-$1 mp
  mkdir -p "$D/cold"; sed "s|^container: .*|container: $base|" "$rec" > "$D/cold/prewarm.yaml"
  [ -e "$rc.v3a-keep" ] && { log "$tag: stale $rc.v3a-keep, restore it by hand"; return 1; }
  [ -z "$plan" ] || [ -s "$rc/b12x/compile/preparation/$plan" ] || { log "$tag: plan $plan missing in $rc"; return 1; }
  mv "$rc" "$rc.v3a-keep" && BACK=$rc && mkdir -p "$rc/b12x/compile/preparation" || return 1
  [ -n "$plan" ] && { cp "$rc.v3a-keep/b12x/compile/preparation/$plan" "$rc/b12x/compile/preparation/" || return 1; }
  log "$tag: dgx-01 runtime cache moved aside (cold boot, kept plan: ${plan:-none})"
  mp=$(memlog $H1 "$D/cold/mem.log"); MP="$MP $mp"
  boot $H1 "$D/cold/prewarm.yaml" "$D/cold" 5400 || return 1
  kill "$mp"; log "$tag: cold min MemAvailable $(minmem "$D/cold/mem.log")"
  serve_log $H1 "$D/cold/serve.log"; aot "$D/cold"; stop_host $H1
  rm -rf "$X"; mkdir -p "$X/preparation"; cp -a "$rc/vllm/torch_compile_cache" "$X/" || return 1
  if [ -n "$plan" ]; then cp "$rc/b12x/compile/preparation/$plan" "$X/preparation/"; else cp "$rc"/b12x/compile/preparation/*.json "$X/preparation/"; fi || return 1
  ( cd "$X/preparation" && md5sum *.json ) > "$D/cold/preparation-baked.md5"
  docker run --rm --entrypoint sh "$base" -c 'cd /opt/b12x-seed/preparation && md5sum *.json' > "$D/cold/preparation-base.md5" 2>&1
  [ -n "$plan" ] || diff -q "$D/cold/preparation-base.md5" "$D/cold/preparation-baked.md5" >/dev/null || log "$tag: WARN baked plan files differ from the $base seed (see $D/cold/preparation-*.md5)"
  printf '%s\n' "# $tag (opus-v3, 2026-10-03): torch AOT compile cache + b12x plan files of a cold boot on dgx-01;" \
    "# the recipe copies both into the runtime cache when absent." \
    "FROM $base" "COPY preparation/ /opt/b12x-seed/preparation/" "COPY torch_compile_cache/ /opt/b12x-seed/torch_compile_cache/" \
    "RUN chmod -R a+rX /opt/b12x-seed" > "$X/Dockerfile"
  log "$tag: context $(du -sh "$X" | cut -f1)"
  docker build -q -t "$out" "$X" >> "$RES/chain.log" 2>&1 || { log "$tag: build failed on dgx-01"; return 1; }
  rsync -a --delete "$X/" "$H2:$X/" && ssh -n $H2 "docker build -q -t $out $X" >> "$RES/chain.log" 2>&1 || { log "$tag: build failed on dgx-02"; return 1; }
  log "$tag: built $out on both nodes"; back_rc; }
warm_check() { # tag recipe: warm boot on both nodes in parallel (AOT seed check)
  local h n
  for h in $H1 $H2; do n=$([ $h = $H1 ] && echo dgx01 || echo dgx02)
    ( boot $h "$2" "$RES/$1/warm-$n" 1800 && { serve_log $h "$RES/$1/warm-$n/serve.log"; aot "$RES/$1/warm-$n"; }; stop_host $h ) & done; wait
  [ -s "$RES/$1/warm-dgx01/boot.txt" ] && [ -s "$RES/$1/warm-dgx02/boot.txt" ] || { log "$1: warm boot failed"; return 1; }; }

if [ -n "${1:-}" ]; then
  trap 'exit 143' TERM INT
  trap 'back_rc; kill $B $MP 2>/dev/null' EXIT
fi
case "${1:-}" in
bake)
  stop_all
  for h in $H1 $H2; do on $h "docker image inspect $IMG0 >/dev/null" || { log "$IMG0 missing on $h"; exit 1; }; done
  bench_cell $H2 "$V2" "$RES/bench/v2-dgx02" & B=$!
  cold_bake bake "$RC" "$V3" "$IMG0" "$IMG" || exit 1
  wait $B; B=
  warm_check bake "$V3" || exit 1
  exit 0 ;;
gdbake)
  stop_all
  cold_bake gdbake "$RCGD" "$GD" "$IMG" "$IMGGD" "$GDPLAN" || exit 1
  warm_check gdbake "$GD" || exit 1
  exit 0 ;;
ab|kv10|gd)
  arm_recipe "$V3" tp1-v3a; arm_recipe "$KV" tp1-v3a-kv10; arm_recipe "$GD" tp1-v3a-gdnmse
  export SHIPPED_IMAGE_EXPECT=$E RESULTS=$RES/$1
  case $1 in ab) A=tp1-v3a; C=tp1-v2 ;; kv10) A=tp1-v3a-kv10; C=tp1-v3a; export BENCHY=1 ;; gd) A=tp1-v3a-gdnmse; C=tp1-v3a; export BENCHY=1 ;; esac
  export CONTROL_ARM=$C CANARY_ARMS=$A SEQ_H1="$C:1 $A:1 $A:2 $C:2" SEQ_H2="$A:3 $C:3 $C:4 $A:4"
  export POST="mkdir -p $RESULTS/merged && cd $RESULTS/merged && for p in ../screen/dgx0?/*-p?; do ln -sfn \$p .; done; cd $R && python3 scripts/r3_screen_report.py $RESULTS/merged $A --base $C > $RESULTS/report-$A.txt 2>&1"
  trap - EXIT
  exec bash "$R/scripts/r8_tp1_driver.sh" ;;
gate|gdgate)
  stop_all; G=$RES/$1; mkdir -p "$G"
  if [ $1 = gate ]; then REC=$V3; LAB=tp1-v3a-gate; bench_cell $H2 "$V3" "$RES/bench/v3a-dgx02" & B=$!
  else REC=$GD; LAB=tp1-v3a-gdnmse-gate; fid_cell $H2 "$GD" "$G/dgx02" & B=$!; fi
  mp=$(memlog $H1 "$G/mem.log"); MP="$MP $mp"
  boot $H1 "$REC" "$G" 1800 || exit 1
  ( cd "$R" && bash scripts/gate_arm.sh $LAB > "$G/gate_arm.log" 2>&1 ); cp "$R"/results/arms/$LAB/* "$G/" 2>/dev/null
  tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 \
    --base-url http://localhost:8000/v1 --model $MODEL --scenarios TC-45 --trials 5 > "$G/tc45.log" 2>&1
  [ $1 = gate ] && for sd in 11 13; do ( cd "$R" && python3 scripts/fidelity_probe.py --base http://localhost:8000 --model $MODEL \
    --depths 128000 --seed $sd --out "$G/fidelity-seed$sd.json" > "$G/fidelity-seed$sd.txt" 2>&1 ); done
  serve_log $H1 "$G/serve.log"; kill "$mp"; stop_host $H1; wait $B; B=
  { grep -E "Quality:" "$G/hardmode.log"; grep -E "Pass\^5|Score:" "$G/tc45.log" | head -2
    grep -h "^depth" "$G/fidelity_probe.txt" "$G"/fidelity-seed1?.txt "$G"/dgx02/fidelity-seed1?.txt 2>/dev/null
    tail -5 "$G/straggler.log"; echo "min MemAvailable $(minmem "$G/mem.log")"; } > "$G/summary.txt" 2>&1
  exit 0 ;;
r9)
  trap - EXIT; echo DONE > "$RES/go-r9"
  cd "$R" && GPU_LOCK_HELD=1 SHIPPED_IMAGE_EXPECT=$E R9_TAG=r9-85c4f24d-79f3d7c5 RESULTS=$R/results/r9-pin-20261003 \
    exec bash scripts/launch_r9.sh "$RES/go-r9" ;;
esac

# ---- launcher: wait, lock, stages, restore ----
st "waiting: k19 STATE DONE"; t0=$(date +%s)
until grep -qE '^(DONE|FAILED|SKIPPED)' "$HOME/GEN-AI/k19/STATE" 2>/dev/null; do
  [ -n "${K19PID:-}" ] && ! kill -0 "$K19PID" 2>/dev/null && { log "k19 job $K19PID gone"; break; }
  [ $(( $(date +%s) - t0 )) -gt 86400 ] && { st "FAILED: k19 not finished after 24 h"; exit 1; }
  sleep 60; done
log "k19: $(cat "$HOME/GEN-AI/k19/STATE")"; st "waiting: gpu-lock"
exec 9>"$HOME/GEN-AI/gpu-lock"; flock 9; log "gpu-lock held, launcher pid $$ pgid $(ps -o pgid= $$ | tr -d ' ')"
is_shipped() { [ "$(health localhost)" = 200 ] && docker ps --format '{{.Image}}' | grep -q "$E" && ssh -n $H2 "docker ps --format '{{.Image}}'" | grep -q "$E"; }
restore() {
  if is_shipped; then log "restore: shipped 2x already serving"; return; fi
  stop_all; sparkrun run qwen3.8-flash-next-2x-dgx-spark --no-follow >> "$RES/chain.log" 2>&1; wait_health localhost 3600
  is_shipped && log "restore: shipped 2x verified, pong=$(pong localhost)" || { log "FAILED FAILED: restore -- MANUAL INTERVENTION"; st "FAILED: restore"; }; }
FINAL="FAILED: chain"; SP=
trap 'restore; grep -q "^FAILED: restore" "$RES/STATE" || st "$FINAL"' EXIT
trap '[ -n "$SP" ] && { kill -TERM -- -$SP 2>/dev/null; wait $SP; }; exit 143' TERM INT
stage() { # name timeout_s; timeout makes its own process group (pgid = its pid)
  st "$1"; timeout -k 900 "$2" bash "$0" "$1" & SP=$!; log "stage $1 pgid $SP"; wait $SP; local rc=$?; SP=
  log "stage $1 exit=$rc"; [ -f "$RES/$1/STATE" ] && log "$1 STATE: $(cat "$RES/$1/STATE")"; return $rc; }
stage bake 9000 || exit 1
stage ab 18000
stage kv10 23400
stage gate 10800
if stage gdbake 7200; then stage gd 23400; stage gdgate 10800; else log "gdbake failed: gd and gdgate skipped"; fi
stage r9 23400; log "r9 STATE: $(cat "$R/results/r9-pin-20261003/STATE" 2>/dev/null)"
FINAL=DONE
exit 0
