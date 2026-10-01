#!/usr/bin/env bash
# r8 TP=2 driver (2026-10-01, opus-kernel-11). Copy of r7_driver.sh with: recipe_of resolves
# r8-* arms (scripts/r8_make_recipes.py tp2, generated from the promoted b1.4 recipe); the
# acceptance canary runs after pass 1 of every arm in CANARY_ARMS (default r8-fp4scale r8-d3
# r8-d3g) against r8-off p1 and drops only that arm; SHIPPED_IMAGE_EXPECT has no default
# (set it to the b1.4 tag the registry recipe serves). Launch via scripts/launch_r8.sh.
# r7 notes (opus-kernel-10, kept):
#   recipe_of resolves r7-* arms; the dvocab canary runs after pass 1 of every r7-dvocab* arm
#   against r7-off p1 and drops only that arm from the remaining stages (r6 stopped the driver);
#   WAIT_FOR_STATE_FILE also waits for the owning run.sh (WAIT_FOR_DIR cwd) to exit, so a
#   transient FAILED-* written before that script's EXIT restore does not start us early.
#   screen:<arm>:<pass>[:<temp>] sets the probe temperature (default 0).
#   Boot failure keeps the full serve log of both ranks; a failed non-control arm is dropped,
#   only an r7-off boot failure stops the driver. WAIT_FOR_STATE_FILES (space separated)
#   waits for every listed STATE file.
#   gpu:<name>       server down; run $GPU_<name> in the b1.2 image with ~/GEN-AI/r5/b12x mounted as /src (PYTHONPATH)
#   build:<name>     build.sh image from VLLM_REF_<name>/B12X_REF_<name>, tag spark-vllm-b12x:<TAG_<name>> on both nodes
# STAGES (space separated), each optional:
#   prof            b1.2 + vllm-decode-profiler, windows fresh/16k c1,c2,c4 + counting c1,c4
#   dbg:<arm>       boot arm, short acceptance probes, grab mtpq-debug log lines from both ranks
#   screen:<arm>:<pass>[:<temp>]  paired screening probes (code fresh/16k + counting at c1 c2 c4 c8 c16)
# Arms resolve to recipes/.../-2x-dgx-spark.yaml (base) or archive/.../-2x-dgx-spark-r4-<arm>.yaml.
# EXIT trap restores the registry recipe (b1.2) and verifies image + env + health + pong.
# Stop: kill -KILL -- -<pgid> while WAITING; kill -TERM -- -<pgid> once it owns the pair.
set -u
export PATH="$HOME/.local/bin:$PATH"
REPO=/home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
RECIPES=/home/nvidia/GEN-AI/r4-recipes
WORKER_IP=192.168.100.53
AD=archive/recipes/qwen3.8-flash-next
REGISTRY_RECIPE=qwen3.8-flash-next-2x-dgx-spark
SHIPPED_IMAGE_EXPECT=${SHIPPED_IMAGE_EXPECT:?set to the tag the registry recipe serves (b1.4)}
CANARY_ARMS=${CANARY_ARMS:-"r8-fp4scale r8-d3 r8-d3g"}
RCDIR=$HOME/.cache/sparkrun/runtime-cache/vllm/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-2d9615ab
CODE_CORPUS=$REPO/results/corpus-code.txt
PROSE_CORPUS=$REPO/results/corpus-prose.txt
GEN="List the numbers from 1 to 300 separated by commas. Output only the numbers, nothing else, no commentary."
STAGES=${STAGES:?STAGES}
RESULTS=${RESULTS:?RESULTS}
mkdir -p "$RESULTS"
LOG=$RESULTS/driver.log
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$LOG"; }
set_state() { echo "$*" > "$RESULTS/STATE"; log "STATE: $*"; }
recipe_of() { case $1 in
  base) echo recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml ;;
  r5-*|r6-*|r7-*|r8-*) echo "$AD/qwen3.8-flash-next-2x-dgx-spark-$1.yaml" ;;
  *) echo "$AD/qwen3.8-flash-next-2x-dgx-spark-r4-$1.yaml" ;; esac; }

OWNS=0; RESTORED=0
health() { curl -s -m5 -o /dev/null -w '%{http_code}' localhost:8000/health; }
wait_health() { local s=$(date +%s); while true; do
  [ "$(health)" = 200 ] && { log "health 200 after $(( $(date +%s)-s ))s"; return 0; }
  [ $(( $(date +%s)-s )) -ge "$1" ] && { log "health timeout"; return 1; }; sleep 15; done; }
stop_all() { sparkrun stop --all >>"$LOG" 2>&1; local i; for i in $(seq 1 60); do
  [ -z "$(docker ps -q)" ] && [ -z "$(ssh $WORKER_IP docker ps -q)" ] && return 0; sleep 5; done
  docker ps -q | xargs -r docker rm -f; ssh $WORKER_IP 'docker ps -q | xargs -r docker rm -f'; }
n0() { docker ps --format '{{.Names}}' | grep node_0 | head -1; }
pong() { curl -s -m 120 localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Reply with exactly one word: pong"}],"max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>/dev/null; }
is_shipped() { [ "$(health)" = 200 ] && docker ps --format '{{.Image}}' | grep -q "$SHIPPED_IMAGE_EXPECT" \
  && ssh $WORKER_IP "docker ps --format '{{.Image}}'" | grep -q "$SHIPPED_IMAGE_EXPECT" \
  && docker exec "$(n0)" sh -c 'env' | grep -qx 'VLLM_QWEN38_HC_MXFP8=hc' \
  && ! docker exec "$(n0)" sh -c 'env' | grep -qE 'VLLM_LOCAL_PROF|VLLM_QWEN3_8_FLASH_NEXT_OVERLAP' \
  && docker exec "$(n0)" sh -c "ps aux | grep '[v]llm serve'" | grep -q '"num_speculative_tokens":4' \
  && ! docker exec "$(n0)" sh -c "ps aux | grep '[v]llm serve'" | grep -q -- '--enforce-eager' \
  && ! docker exec "$(n0)" sh -c 'env' | grep -qE 'VLLM_DPROBE_OUT|VLLM_R4_' \
  && ! docker exec "$(n0)" sh -c 'grep -rlE "mtpq-debug|moe-dprobe|local step profiler|r4-w8a16" /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next /usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/router /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py 2>/dev/null' | grep -q .; }
restore() {
  [ "$OWNS" = 1 ] || { log "restore: skipped, never owned the pair"; return; }
  [ "$RESTORED" = 1 ] && return; RESTORED=1
  set_state restore
  if ! is_shipped; then stop_all; sparkrun run "$REGISTRY_RECIPE" --no-follow >>"$LOG" 2>&1; wait_health 3600; fi
  if is_shipped; then log "restore: shipped verified, pong=$(pong)"; set_state DONE
  else log "FAILED FAILED: shipped restore -- MANUAL INTERVENTION"; set_state "FAILED: restore"; fi
}
trap restore EXIT

boot() { # recipe_rel; fails fast when the engine dies during startup
  stop_all; log "boot $1"; ( cd "$RECIPES" && sparkrun run "$1" --no-follow >>"$LOG" 2>&1 )
  local s=$(date +%s) n
  while true; do
    [ "$(health)" = 200 ] && { log "health 200 after $(( $(date +%s)-s ))s"; return 0; }
    n=$(n0)
    if [ $(( $(date +%s)-s )) -gt 120 ] && { [ -z "$n" ] || docker exec "$n" grep -qE "Worker failed with error|EngineCore failed to start|Engine core initialization failed" /tmp/sparkrun_serve.log 2>/dev/null; }; then
      [ -n "$n" ] && docker exec "$n" sh -c "grep -m5 -E 'Error|error' /tmp/sparkrun_serve.log" >> "$LOG" 2>&1
      local t=$(date +%s) w; w=$(ssh $WORKER_IP "docker ps -a --format '{{.Names}}'" | grep node_1 | head -1)
      [ -n "$n" ] && docker exec "$n" cat /tmp/sparkrun_serve.log > "$RESULTS/bootfail-$t-node0.log" 2>&1
      [ -n "$w" ] && ssh $WORKER_IP "docker exec $w cat /tmp/sparkrun_serve.log" > "$RESULTS/bootfail-$t-node1.log" 2>&1
      log "boot FAILED (engine died) after $(( $(date +%s)-s ))s: $1"; return 1; fi
    [ $(( $(date +%s)-s )) -ge 5400 ] && { log "health timeout"; return 1; }; sleep 15; done; }
warm() { local i; for i in 1 2 3; do curl -s -m 600 localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":16}' > /dev/null; done; }
record() { # dir
  mkdir -p "$1"; local a b; a=$(n0); b=$(ssh $WORKER_IP "docker ps --format '{{.Names}}'" | grep node_1 | head -1)
  { docker inspect --format '{{.Config.Image}}' "$a"; ssh $WORKER_IP "docker inspect --format '{{.Config.Image}}' $b"; } > "$1/image.txt"
  docker exec "$a" sh -c "env | grep -E '^(VLLM|B12X)_' | sort" > "$1/env.txt" 2>&1
  docker exec "$a" sh -c "ps aux | grep '[v]llm serve'" > "$1/cmdline.txt" 2>&1
  docker exec "$a" sh -c "grep -E 'Quantizing LM head|snapshots/|cached|measured|Traceback|num_speculative|HC MXFP8' /tmp/sparkrun_serve.log | head -80" > "$1/boot_grep.txt" 2>&1
  { echo "== node0"; docker exec "$a" sh -c "grep -o 'snapshots/[0-9a-f]\{8\}' /tmp/sparkrun_serve.log | sort | uniq -c"
    echo "== node1"; ssh $WORKER_IP "docker exec $b sh -c \"grep -o 'snapshots/[0-9a-f]\\{8\\}' /tmp/sparkrun_serve.log | sort | uniq -c\""; } > "$1/snapshots.txt" 2>&1
}
serve_logs() { # dir: full serve logs of both ranks
  local a b; a=$(n0); b=$(ssh $WORKER_IP "docker ps --format '{{.Names}}'" | grep node_1 | head -1)
  docker exec "$a" cat /tmp/sparkrun_serve.log > "$1/serve-node0.log" 2>&1
  ssh $WORKER_IP "docker exec $b cat /tmp/sparkrun_serve.log" > "$1/serve-node1.log" 2>&1; }
pos() { curl -s -m 10 localhost:8000/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens(_per_pos)?_total' ; }
probe() { # dir tag corpus|count depth conc repeats offset [extra...]
  local d=$1 tag=$2 src=$3 depth=$4 c=$5 r=$6 off=$7; shift 7
  pos > "$d/pos-$tag.before"
  if [ "$src" = count ]; then
    ( cd "$REPO" && python3 scripts/depth_decode_probe.py --label "$tag" --prompt "$GEN" --concurrency "$c" --repeats "$r" \
        --max-tokens 320 --temperature "${PROBE_TEMP:-0}" --out "$d/probe-$tag.json" "$@" > "$d/probe-$tag.log" 2>&1 )
  else
    ( cd "$REPO" && python3 scripts/depth_decode_probe.py --label "$tag" --corpus "$src" --depth "$depth" --new 2048 \
        --concurrency "$c" --repeats "$r" --offset "$off" --max-tokens 512 --temperature "${PROBE_TEMP:-0}" \
        --out "$d/probe-$tag.json" "$@" > "$d/probe-$tag.log" 2>&1 )
  fi
  local rc=$?; pos > "$d/pos-$tag.after"; log "$(basename "$d"): probe $tag exit=$rc $(acc "$d" "$tag")"
}
acc() { python3 - "$1/pos-$2.before" "$1/pos-$2.after" <<'PY' 2>/dev/null
import re,sys
def rd(f):
    d={}
    for l in open(f):
        m=re.match(r'(\S+?)\{(.*)\}\s+([\d.e+]+)',l)
        if not m: continue
        p=re.search(r'position="(\d+)"',m.group(2))
        d[(m.group(1),p.group(1) if p else None)]=float(m.group(3))
    return d
a,b=rd(sys.argv[1]),rd(sys.argv[2]); g=lambda k:b.get(k,0)-a.get(k,0)
A='vllm:spec_decode_num_accepted_tokens_per_pos_total'; D='vllm:spec_decode_num_draft_tokens_per_pos_total'
dr=g((D,'0')) or 1
print("acc/pos "+"/".join("%.2f"%(g((A,str(i)))/dr) for i in range(8) if (A,str(i)) in b))
PY
}

trig() { echo "n=go-\$(date +%s%N); echo $1 > $RCDIR/prof-trigger/\$n; ssh $WORKER_IP \"echo $1 > $RCDIR/prof-trigger/\$n\""; }
collect() { # label
  local i; for i in $(seq 1 24); do [ -f "$RCDIR/prof/$1/rank0.json" ] && break; sleep 5; done; sleep 10
  local out=$RESULTS/prof${PROF_ARM:+-$PROF_ARM}/$1; mkdir -p "$out"
 cp "$RCDIR/prof/$1/rank0.json" "$out/" 2>>"$LOG" || log "WARN: no rank0 trace $1"
  scp -q "$WORKER_IP:$RCDIR/prof/$1/rank1.json" "$out/" 2>>"$LOG" || log "WARN: no rank1 trace $1"
  for r in 0 1; do [ -f "$out/rank$r.json" ] && python3 "$REPO/scripts/step_breakdown.py" 60 "$out/rank$r.json" \
      --json "$out/breakdown-rank$r.json" > "$out/breakdown-rank$r.txt" 2>&1; done
  gzip -f "$out"/rank*.json
}
stage_prof() { # [arm] (default prof)
  local arm=${1:-prof}; PROF_ARM=$arm
  set_state "prof:$arm"; OWNS=1
  mkdir -p "$RCDIR/prof-trigger"; ssh $WORKER_IP "mkdir -p $RCDIR/prof-trigger"
  rm -f "$RCDIR"/prof-trigger/*; ssh $WORKER_IP "rm -f $RCDIR/prof-trigger/*"
  rm -rf "$RCDIR"/prof/prof-*; ssh $WORKER_IP "rm -rf $RCDIR/prof/prof-*"
  boot "$(recipe_of "$arm")" || { log "prof boot failed"; return 1; }
  local d=$RESULTS/prof-$arm; mkdir -p "$d"; record "$d/boot"; warm
  docker exec "$(n0)" grep -q 'local step profiler (mods/vllm-decode-profiler)' /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py \
    || { log "profiler mod not applied"; return 1; }
  probe "$d" warm "$CODE_CORPUS" 4096 1 1 900000
  local spec k=0 tag depth c
  for spec in ${PROF_WINDOWS:-fresh-c1:256:1 fresh-c2:256:2 fresh-c4:256:4 d16k-c1:16384:1 d16k-c2:16384:2 d16k-c4:16384:4}; do
    IFS=: read -r tag depth c <<< "$spec"; k=$((k+1))
    probe "$d" "$tag" "$CODE_CORPUS" "$depth" "$c" 2 $((k*211111)) --trigger-cmd "$(trig "prof-$tag")" --trigger-repeat 1 --trigger-after-chunks 40
    collect "prof-$tag"
  done
  [ -n "${PROF_WINDOWS:-}" ] && return 0
  probe "$d" count-c1 count 0 1 2 0 --trigger-cmd "$(trig prof-count-c1)" --trigger-repeat 1 --trigger-after-chunks 40; collect prof-count-c1
  probe "$d" count-c4 count 0 4 2 0 --trigger-cmd "$(trig prof-count-c4)" --trigger-repeat 1 --trigger-after-chunks 40; collect prof-count-c4
}
stage_dbg() { # arm
  local d=$RESULTS/dbg-$1; mkdir -p "$d"; set_state "dbg:$1"; OWNS=1
  boot "$(recipe_of "$1")" || { log "dbg $1 boot failed"; echo fail > "$d/FAILED"; return 1; }
  record "$d"; warm
  probe "$d" count-c1 count 0 1 3 0
  probe "$d" fresh-c1 "$CODE_CORPUS" 256 1 2 104729
  probe "$d" count-c4 count 0 4 1 0
  serve_logs "$d"; grep -h "mtpq-debug" "$d"/serve-node*.log | sort | uniq -c | sort -rn | head -5 >> "$LOG"
}
stage_screen() { # arm pass [temp]
  local arm=$1 d=$RESULTS/screen/$1-p$2 k=0 c r spec PROBE_TEMP=${3:-0}
  set_state "screen:$arm-p$2 temp=$PROBE_TEMP"; OWNS=1
  boot "$(recipe_of "$arm")" || { log "$arm boot failed"; mkdir -p "$d"; echo fail > "$d/FAILED"; return 1; }
  record "$d"; warm
  probe "$d" warm "$CODE_CORPUS" 4096 1 1 900000
  for spec in 1:8 2:4 4:3 8:1 16:1; do IFS=: read -r c r <<< "$spec"; k=$((k+1))
    probe "$d" "fresh-c$c" "$CODE_CORPUS" 256 "$c" "$r" $((k*104729))
    probe "$d" "d16k-c$c" "$CODE_CORPUS" 16384 "$c" "$r" $(((k+10)*104729))
    probe "$d" "count-c$c" count 0 "$c" "$r" 0; done
  log "=== screen $arm p$2 done ==="
}

stage_dprobe() { # eager boot, routed-expert D histogram per M
  local d=$RESULTS/dprobe; mkdir -p "$d"; set_state dprobe; OWNS=1
  rm -f "$RCDIR/dprobe.json"
  boot "$(recipe_of dprobe)" || { log "dprobe boot failed"; return 1; }
  record "$d"; warm
  probe "$d" fresh-c1 "$CODE_CORPUS" 256 1 2 104729
  probe "$d" fresh-c4 "$CODE_CORPUS" 256 4 1 314187
  probe "$d" d16k-c4 "$CODE_CORPUS" 16384 4 1 1256748
  probe "$d" count-c4 count 0 4 1 0
  probe "$d" fresh-c8 "$CODE_CORPUS" 256 8 1 418916
  probe "$d" fresh-c16 "$CODE_CORPUS" 256 16 1 523645
  sleep 5; cp "$RCDIR/dprobe.json" "$d/" 2>>"$LOG"; log "dprobe: $(head -c 600 "$d/dprobe.json")"
}

stage_bench() { # name: server down, run $BENCH_<name> (a command) in the b1.2 image with the GPU
  local d=$RESULTS/bench-$1; mkdir -p "$d"; set_state "bench:$1"; OWNS=1; stop_all
  local var="BENCH_$1"; log "bench $1: ${!var}"
  timeout 3600 docker run --rm --gpus all --ipc host -v /home/nvidia/GEN-AI/r4-kernels:/k -w /k \
    ghcr.io/ursuciprian/spark-vllm-b12x:$SHIPPED_IMAGE_EXPECT bash -c "${!var}" > "$d/bench.log" 2>&1
  log "bench $1 exit=$? $(tail -1 "$d/bench.log" | cut -c1-200)"
}


stage_gpu() { # name
  local d=$RESULTS/gpu-$1; mkdir -p "$d"; set_state "gpu:$1"; OWNS=1; stop_all
  local var="GPU_$1"; log "gpu $1: ${!var}"
  timeout ${GPU_TIMEOUT:-5400} docker run --rm --gpus all --ipc host -e CUTE_DSL_ARCH=sm_121a -e B12X_COMPILE_WORKERS=2 \
    -v /home/nvidia/GEN-AI/r5/b12x:/src -w /src -e PYTHONPATH=/src \
    ghcr.io/ursuciprian/spark-vllm-b12x:$SHIPPED_IMAGE_EXPECT bash -c "${!var}" > "$d/run.log" 2>&1
  log "gpu $1 exit=$? $(tail -1 "$d/run.log" | cut -c1-200)"
}
BUILD_DIR=/home/nvidia/GEN-AI/gdndef-build/spark-vllm-b12x
BUILD_ROOT=$HOME/GEN-AI/build
stage_build() { # name
  local name=$1 vv="VLLM_REF_$1" bv="B12X_REF_$1" tv="TAG_$1" ok=0 attempt built
  local tag="spark-vllm-b12x:${!tv}"
  set_state "build:$name"; OWNS=1; stop_all
  if docker image inspect "$tag" >/dev/null 2>&1 && ssh $WORKER_IP "docker image inspect $tag >/dev/null 2>&1"; then log "build $name: $tag present on both nodes"; return 0; fi
  for attempt in 1 2; do
    ( cd "$BUILD_DIR" && CONFIRM_BUILD=1 VLLM_REF="${!vv}" B12X_REF="${!bv}" DISTRIBUTE_TO="$WORKER_IP" \
        IMAGE_TAG_SUFFIX="-$name" BUILD_ROOT="$BUILD_ROOT" ./build.sh > "$RESULTS/build-$name-$attempt.log" 2>&1 ) && { ok=1; break; }
    log "build $name attempt $attempt failed"
  done
  [ "$ok" = 1 ] || { log "build $name FAILED"; echo fail > "$RESULTS/BUILD_FAILED_$name"; return 1; }
  built=$(grep -oE "spark-vllm-b12x:local-[0-9]+-[0-9a-f]+-$name" "$RESULTS/build-$name-$attempt.log" | tail -1)
  [ -n "$built" ] || { log "build $name: no tag in log"; echo fail > "$RESULTS/BUILD_FAILED_$name"; return 1; }
  docker tag "$built" "$tag" && ssh $WORKER_IP "docker tag '$built' '$tag'" || { log "tag $name failed"; return 1; }
  log "build $name: $built -> $tag vllm=$(docker run --rm "$tag" python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null) wm=$(docker run --rm "$tag" python3 -c 'import b12x.moe._shared.kernels.wm_geometry as m; print(m.STAGES)' 2>&1 | tail -1)"
}

canary() { # arm: <arm> p1 vs r8-off p1, draft position 0 acceptance per cell: fail if any cell < 0.7x off
  python3 - "$RESULTS/screen/r8-off-p1" "$RESULTS/screen/$1-p1" <<'PY' >> "$LOG" 2>&1
import re,sys,glob,os
def acc0(d,tag):
    def rd(f):
        v={}
        for l in open(f):
            m=re.match(r'vllm:spec_decode_num_(accepted|draft)_tokens_per_pos_total\{.*position="0"\} ([0-9.e+]+)',l)
            if m: v[m.group(1)]=float(m.group(2))
        return v
    a,b=rd(f"{d}/pos-{tag}.before"),rd(f"{d}/pos-{tag}.after")
    dr=b.get("draft",0)-a.get("draft",0)
    return (b.get("accepted",0)-a.get("accepted",0))/dr if dr else None
bad=0
for f in sorted(glob.glob(sys.argv[1]+"/pos-*.after")):
    tag=os.path.basename(f)[4:-6]
    if tag=="warm": continue
    o,x=acc0(sys.argv[1],tag),acc0(sys.argv[2],tag)
    if o is None or x is None: continue
    print(f"canary {os.path.basename(sys.argv[2])} {tag}: off {o:.2f} arm {x:.2f}")
    if x < 0.7*o: bad+=1
print("canary bad cells", bad); sys.exit(1 if bad else 0)
PY
}
log "driver start pid=$$ pgid=$(ps -o pgid= $$ | tr -d ' ') STAGES=$STAGES"
set_state waiting
owner_alive() { local p; for p in $(pgrep -f 'run.sh'); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "${WAIT_FOR_DIR:-}" ] && return 0; done; return 1; }
if [ -n "${WAIT_FOR_STATE_FILE:-}" ]; then
  until grep -qE '^(DONE|FAILED)' "$WAIT_FOR_STATE_FILE" 2>/dev/null && ! owner_alive; do sleep 60; done
  log "wait over: $(cat "$WAIT_FOR_STATE_FILE")"
fi
states_done() { local f; for f in ${WAIT_FOR_STATE_FILES:-}; do grep -qE '^(DONE|FAILED)' "$f" 2>/dev/null || return 1; done; }
if [ -n "${WAIT_FOR_STATE_FILES:-}" ]; then
  # settle 5 min: a FAILED-* can be written before its script's EXIT restore runs
  until states_done; do sleep 60; done; sleep 300
  until states_done; do sleep 60; done
  for f in $WAIT_FOR_STATE_FILES; do log "wait over: $f = $(cat "$f")"; done
fi
SKIP=" "
for st in $STAGES; do
  IFS=: read -r kind a b c <<< "$st"
  case $kind in
    prof) stage_prof "$a" ;;
    dbg) stage_dbg "$a" ;;
    dprobe) stage_dprobe ;;
    bench) stage_bench "$a" ;;
    gpu) stage_gpu "$a" ;;
    build) stage_build "$a" || { log "build failed: stopping"; break; } ;;
    screen) case $SKIP in *" $a "*) log "skip $st (canary)"; continue ;; esac
            stage_screen "$a" "$b" "$c"
            if [ -f "$RESULTS/screen/$a-p$b/FAILED" ]; then
              [ "$a" = r8-off ] && { log "screen $a boot failed: stopping"; break; }
              log "screen $a boot failed: dropping $a"; SKIP="$SKIP$a "; continue; fi
            if [[ " $CANARY_ARMS " == *" $a "* ]] && [ "$b" = 1 ] && ! canary "$a"; then
              log "CANARY: $a acceptance collapsed vs r8-off p1: dropping $a"; echo "$a" >> "$RESULTS/CANARY_FAILED"; SKIP="$SKIP$a "; fi ;;
    *) log "unknown stage $st" ;;
  esac
done
restore
exit 0
