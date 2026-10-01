#!/usr/bin/env bash
# r8 TP=1 driver (2026-10-01, opus-kernel-11). One TP=1 server per Spark, both Sparks in
# parallel, each running its own same-node sequence (ABBA against tp1-off, e.g.
# "tp1-off:1 tp1-ple-willneed:1 tp1-ple-willneed:2 tp1-off:2"). Recipes are
# archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-<arm>.yaml from
# scripts/r8_make_recipes.py tp1. Start it through scripts/launch_r8_tp1.sh or
# scripts/r8_profile.sh (both wait on a STATE file first).
#
# Sequence items, per node:
#   <arm>:<pass>[:<temp>]   screening pass: boot, warm, paired probes fresh/16k/count at
#                           c1 c2 c4 c8 (same offsets on both nodes and in every pass)
#   prof:<arm>              profile pass: boot, 60-step decode-profiler windows
#                           (PROF_WINDOWS), host counters around each window
# Env:
#   RESULTS (required); SEQ_H1 (dgx-01) / SEQ_H2 (dgx-02), either may be empty
#   PAGECACHE_H1 / PAGECACHE_H2  keep (default) | evict | touch: PLE table page-cache
#                state set by scripts/ple_pagecache.py before every boot on that node
#   SHIPPED_IMAGE_EXPECT (required): tag of the shipped TP=2 registry recipe (restore check)
#   CANARY_ARMS (default "tp1-dv98"): after pass 1, pos-0 acceptance of every temp-0 cell
#                vs tp1-off p1 on the same node; < 0.7x drops the arm on that node
#   tp1-ple-cpuhash is dropped on a node whose serve log shows "PLE cpu-hash MISMATCH".
# Every pass keeps the full serve log, the PLE reader stats lines, a 5 s MemAvailable/Cached
# log and the env/cmdline. A failed boot drops that arm on that node (tp1-off: the node stops).
# EXIT restores the registry TP=2 recipe and verifies image + env + health + pong.
# Stop: kill -TERM -- -<pgid> (printed in driver.log). Never pkill -f.
set -u
export PATH="$HOME/.local/bin:$PATH"
REPO=/home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
RECIPES=/home/nvidia/GEN-AI/r4-recipes
AD=archive/recipes/qwen3.8-flash-next
H1=192.168.100.62; H2=192.168.100.53; WORKER_IP=$H2
REGISTRY_RECIPE=qwen3.8-flash-next-2x-dgx-spark
SHIPPED_IMAGE_EXPECT=${SHIPPED_IMAGE_EXPECT:?set to the tag the registry recipe serves}
RCDIR=$HOME/.cache/sparkrun/runtime-cache/vllm/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-2d9615ab
SNAP=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
CODE_CORPUS=$REPO/results/corpus-code.txt
GEN="List the numbers from 1 to 300 separated by commas. Output only the numbers, nothing else, no commentary."
CANARY_ARMS=${CANARY_ARMS:-"tp1-dv98"}
PROF_WINDOWS=${PROF_WINDOWS:-"fresh-c1:256:1:1 d16k-c1:16384:1:4 fresh-c4:256:4:3 d16k-c4:16384:4:6 count-c1:0:1:0"}
RESULTS=${RESULTS:?RESULTS}
mkdir -p "$RESULTS"
LOG=$RESULTS/driver.log
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$LOG"; }
set_state() { echo "$*" > "$RESULTS/STATE"; log "STATE: $*"; }
on() { local h=$1; shift; if [ "$h" = "$H1" ]; then bash -c "$*"; else ssh "$h" "$*"; fi; }
label() { [ "$1" = "$H1" ] && echo dgx01 || echo dgx02; }

# ---- TP=2 restore (as in r7_driver.sh) ----
OWNS=0; RESTORED=0
health() { curl -s -m5 -o /dev/null -w '%{http_code}' "${1:-localhost}:8000/health"; }
stop_all() { sparkrun stop --all >>"$LOG" 2>&1; local i; for i in $(seq 1 60); do
  [ -z "$(docker ps -q)" ] && [ -z "$(ssh $WORKER_IP docker ps -q)" ] && return 0; sleep 5; done
  docker ps -q | xargs -r docker rm -f; ssh $WORKER_IP 'docker ps -q | xargs -r docker rm -f'; }
n0() { docker ps --format '{{.Names}}' | grep node_0 | head -1; }
pong() { curl -s -m 120 "${1:-localhost}:8000/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Reply with exactly one word: pong"}],"max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>/dev/null; }
is_shipped() { [ "$(health)" = 200 ] && docker ps --format '{{.Image}}' | grep -q "$SHIPPED_IMAGE_EXPECT" \
  && ssh $WORKER_IP "docker ps --format '{{.Image}}'" | grep -q "$SHIPPED_IMAGE_EXPECT" \
  && docker exec "$(n0)" sh -c 'env' | grep -qx 'VLLM_QWEN38_HC_MXFP8=hc' \
  && ! docker exec "$(n0)" sh -c 'env' | grep -qE 'VLLM_LOCAL_PROF|VLLM_PLE_MMAP|VLLM_MALLOC_TRIM' \
  && docker exec "$(n0)" sh -c "ps aux | grep '[v]llm serve'" | grep -q -- '--tensor-parallel-size 2' \
  && ! docker exec "$(n0)" sh -c 'grep -rl "local step profiler" /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py 2>/dev/null' | grep -q .; }
cleanup_jobs() { local p; for p in $(jobs -p) $(cat "$RESULTS/.bg.pids" 2>/dev/null); do kill "$p" 2>/dev/null; done; }
restore() {
  cleanup_jobs
  [ "$OWNS" = 1 ] || { log "restore: skipped, never owned the pair"; return; }
  [ "$RESTORED" = 1 ] && return; RESTORED=1
  set_state restore
  if ! is_shipped; then stop_all; sparkrun run "$REGISTRY_RECIPE" --no-follow >>"$LOG" 2>&1
    local s=$(date +%s); until [ "$(health)" = 200 ] || [ $(( $(date +%s)-s )) -ge 3600 ]; do sleep 15; done; fi
  if is_shipped; then log "restore: shipped verified, pong=$(pong)"; set_state DONE
  else log "FAILED FAILED: shipped restore -- MANUAL INTERVENTION"; set_state "FAILED: restore"; fi
}
trap restore EXIT

# ---- per-node TP=1 helpers ----
solo() { on "$1" "docker ps --format '{{.Names}}' | grep -E '_solo|sparkrun' | head -1"; }
cx() { local h=$1 c; c=$(solo "$h"); shift; [ -n "$c" ] && on "$h" "docker exec $c sh -c $(printf %q "$*")"; }
stop_host() { sparkrun stop --all --hosts "$1" >>"$LOG" 2>&1; local i; for i in $(seq 1 60); do
  [ -z "$(on "$1" 'docker ps -q')" ] && return 0; sleep 5; done; on "$1" 'docker ps -q | xargs -r docker rm -f'; }
pagecache() { # host
  local mode; [ "$1" = "$H1" ] && mode=${PAGECACHE_H1:-keep} || mode=${PAGECACHE_H2:-keep}
  [ "$mode" = keep ] && mode=resident
  on "$1" "python3 $REPO/scripts/ple_pagecache.py $mode $SNAP" 2>&1 | tee -a "$LOG" | tail -1; }
MEMLOOP='while true; do echo "$(date +%T) $(awk '"'"'/^MemAvailable|^Cached/{printf "%s %.2f ", $1, $2/1048576}'"'"' /proc/meminfo)"; sleep 5; done'
memlog() { # host file: 5 s MemAvailable/Cached sampler in the background, prints its pid
  if [ "$1" = "$H1" ]; then bash -c "$MEMLOOP" > "$2" 2>&1 < /dev/null &
  else ssh "$1" "$MEMLOOP" > "$2" 2>&1 < /dev/null & fi
  echo $! | tee -a "$RESULTS/.bg.pids"; }
minmem() { awk '{for(i=1;i<=NF;i++) if($i=="MemAvailable:") print $(i+1)}' "$1" | sort -n | head -1; }
boot_arm() { # host recipe dir: fails fast when the engine dies during startup
  local h=$1 rec=$2 d=$3 s n
  stop_host "$h"; log "$(label "$h"): pagecache $(pagecache "$h")"
  log "$(label "$h"): boot $rec"; ( cd "$RECIPES" && sparkrun run "$rec" --hosts "$h" --solo --no-follow >>"$d/sparkrun.log" 2>&1 )
  s=$(date +%s)
  while true; do
    [ "$(health "$h")" = 200 ] && { log "$(label "$h"): health 200 after $(( $(date +%s)-s ))s"; return 0; }
    n=$(solo "$h")
    if [ $(( $(date +%s)-s )) -gt 120 ] && { [ -z "$n" ] || cx "$h" "grep -qE 'Worker failed with error|EngineCore failed to start|Engine core initialization failed' /tmp/sparkrun_serve.log"; }; then
      log "$(label "$h"): boot FAILED after $(( $(date +%s)-s ))s: $rec"; return 1; fi
    [ $(( $(date +%s)-s )) -ge 5400 ] && { log "$(label "$h"): health timeout"; return 1; }; sleep 15; done; }
keep_logs() { # host dir: full serve log + reader stats + env/cmdline, also after a failure
  local h=$1 d=$2 c; c=$(on "$h" "docker ps -a --format '{{.Names}}' | grep -E '_solo|sparkrun' | head -1")
  [ -n "$c" ] || { log "$(label "$h"): no container for logs"; return; }
  on "$h" "docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1" > "$d/serve.log" 2>&1
  grep -E "PLE reader|PLE prewarm|PLE page-cache reader|PLE cpu-hash|malloc_trim" "$d/serve.log" > "$d/ple_reader.txt"
  on "$h" "docker exec $c sh -c \"env | grep -E '^(VLLM|B12X)_' | sort\"" > "$d/env.txt" 2>&1
  on "$h" "docker exec $c sh -c \"ps aux | grep '[v]llm serve'\"" > "$d/cmdline.txt" 2>&1
  on "$h" "docker inspect --format '{{.Config.Image}}' $c" > "$d/image.txt" 2>&1; }
pos() { curl -s -m 10 "$1:8000/metrics" | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens(_per_pos)?_total'; }
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
probe() { # host dir tag corpus|count depth conc repeats offset [extra...]
  local h=$1 d=$2 tag=$3 src=$4 depth=$5 c=$6 r=$7 off=$8; shift 8
  pos "$h" > "$d/pos-$tag.before"
  if [ "$src" = count ]; then
    ( cd "$REPO" && python3 scripts/depth_decode_probe.py --base "http://$h:8000" --label "$tag" --prompt "$GEN" \
        --concurrency "$c" --repeats "$r" --max-tokens 320 --temperature "${PROBE_TEMP:-0}" \
        --out "$d/probe-$tag.json" "$@" > "$d/probe-$tag.log" 2>&1 )
  else
    ( cd "$REPO" && python3 scripts/depth_decode_probe.py --base "http://$h:8000" --label "$tag" --corpus "$src" \
        --depth "$depth" --new 2048 --concurrency "$c" --repeats "$r" --offset "$off" --max-tokens 512 \
        --temperature "${PROBE_TEMP:-0}" --out "$d/probe-$tag.json" "$@" > "$d/probe-$tag.log" 2>&1 )
  fi
  local rc=$?; pos "$h" > "$d/pos-$tag.after"; log "$(label "$h")/$(basename "$d"): probe $tag exit=$rc $(acc "$d" "$tag")"; }
warm() { local i; for i in 1 2 3; do curl -s -m 600 "$1:8000/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":16}' > /dev/null; done; }
counters() { # host: one JSON line of host + EngineCore counters
  local h=$1 c; c=$(solo "$h")
  on "$h" "python3 - $c" <<'PY'
import json, re, subprocess, sys, time
out = {"t": time.time(), "nvme_reads": 0}
for line in open("/proc/diskstats"):
    f = line.split()
    if re.fullmatch(r"nvme\d+n\d+", f[2]): out["nvme_reads"] += int(f[3])
for line in open("/proc/meminfo"):
    k, v = line.split(":", 1)
    if k in ("MemAvailable", "Cached"): out[k] = int(v.split()[0]) * 1024
code = ("import os\nfor p in os.listdir('/proc'):\n if p.isdigit():\n  try:\n"
        "   c=open(f'/proc/{p}/cmdline','rb').read()\n  except OSError: continue\n"
        "  if b'EngineCore' in c: print(open(f'/proc/{p}/stat').read().rsplit(')',1)[1].split()[9]); break")
try:
    out["enginecore_majflt"] = int(subprocess.run(["docker", "exec", sys.argv[1], "python3", "-c", code],
                                                  capture_output=True, text=True, timeout=30).stdout.split()[0])
except Exception:
    out["enginecore_majflt"] = None
print(json.dumps(out))
PY
}
trig() { # host label: trigger command for depth_decode_probe --trigger-cmd
  if [ "$1" = "$H1" ]; then echo "echo $2 > $RCDIR/prof-trigger/go-\$(date +%s%N)"
  else echo "ssh $1 \"echo $2 > $RCDIR/prof-trigger/go-\$(date +%s%N)\""; fi; }
collect() { # host label dir
  local h=$1 lab=$2 out=$3/$2 i; mkdir -p "$out"
  for i in $(seq 1 24); do on "$h" "test -f $RCDIR/prof/$lab/rank0.json" && break; sleep 5; done; sleep 10
  if [ "$h" = "$H1" ]; then cp "$RCDIR/prof/$lab/rank0.json" "$out/" 2>>"$LOG"; else scp -q "$h:$RCDIR/prof/$lab/rank0.json" "$out/" 2>>"$LOG"; fi \
    || { log "WARN: no trace $lab on $(label "$h")"; return; }
  python3 "$REPO/scripts/step_breakdown.py" 60 "$out/rank0.json" --json "$out/breakdown-rank0.json" > "$out/breakdown-rank0.txt" 2>&1
  gzip -f "$out/rank0.json"; }
pass_prof() { # host arm dir
  local h=$1 arm=$2 d=$3 spec tag depth c k
  cx "$h" "grep -q 'local step profiler (mods/vllm-decode-profiler)' /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py" \
    || { log "$(label "$h"): profiler mod not applied"; return 1; }
  probe "$h" "$d" warm "$CODE_CORPUS" 4096 1 1 900000
  for spec in $PROF_WINDOWS; do IFS=: read -r tag depth c k <<< "$spec"
    counters "$h" > "$d/counters-$tag.before"
    if [ "$tag" = "${tag#count}" ]; then
      probe "$h" "$d" "$tag" "$CODE_CORPUS" "$depth" "$c" 2 $((k*211111)) --trigger-cmd "$(trig "$h" "prof-$tag")" --trigger-repeat 1 --trigger-after-chunks 40
    else
      # ~65 steps per counting request: start early so the window stays inside it
      probe "$h" "$d" "$tag" count 0 "$c" 2 0 --trigger-cmd "$(trig "$h" "prof-$tag")" --trigger-repeat 1 --trigger-after-chunks 5
    fi
    counters "$h" > "$d/counters-$tag.after"
    collect "$h" "prof-$tag" "$d"
  done; }
pass_screen() { # host dir
  local h=$1 d=$2 k=0 c r spec
  probe "$h" "$d" warm "$CODE_CORPUS" 4096 1 1 900000
  for spec in 1:8 2:4 4:3 8:1; do IFS=: read -r c r <<< "$spec"; k=$((k+1))
    probe "$h" "$d" "fresh-c$c" "$CODE_CORPUS" 256 "$c" "$r" $((k*104729))
    probe "$h" "$d" "d16k-c$c" "$CODE_CORPUS" 16384 "$c" "$r" $(((k+10)*104729))
    probe "$h" "$d" "count-c$c" count 0 "$c" "$r" 0; done; }
canary() { # control-dir arm-dir
  python3 - "$1" "$2" <<'PY' >> "$LOG" 2>&1
import glob, os, re, sys
def acc0(d, tag):
    def rd(f):
        v = {}
        for l in open(f):
            m = re.match(r'vllm:spec_decode_num_(accepted|draft)_tokens_per_pos_total\{.*position="0"\} ([0-9.e+]+)', l)
            if m: v[m.group(1)] = float(m.group(2))
        return v
    a, b = rd(f"{d}/pos-{tag}.before"), rd(f"{d}/pos-{tag}.after")
    dr = b.get("draft", 0) - a.get("draft", 0)
    return (b.get("accepted", 0) - a.get("accepted", 0)) / dr if dr else None
bad = 0
for f in sorted(glob.glob(sys.argv[1] + "/pos-*.after")):
    tag = os.path.basename(f)[4:-6]
    if tag == "warm": continue
    o, x = acc0(sys.argv[1], tag), acc0(sys.argv[2], tag)
    if o is None or x is None: continue
    print(f"canary {sys.argv[2]} {tag}: off {o:.2f} arm {x:.2f}")
    bad += x < 0.7 * o
print("canary bad cells", bad); sys.exit(1 if bad else 0)
PY
}
node_run() { # host "seq": one node's sequence, sequential
  local h=$1 seq=$2 node; node=$(label "$h"); local skip=" " item arm pass temp d mp rec
  for item in $seq; do
    IFS=: read -r arm pass temp <<< "$item"
    if [ "$arm" = prof ]; then rec=$AD/qwen3.8-flash-next-1x-dgx-spark-$pass.yaml; d=$RESULTS/prof/$node
    else rec=$AD/qwen3.8-flash-next-1x-dgx-spark-$arm.yaml; d=$RESULTS/screen/$node/$arm-p$pass; fi
    case $skip in *" $arm "*) log "$node: skip $item (dropped)"; continue ;; esac
    [ -f "$RECIPES/$rec" ] || { log "$node: missing $rec, skip $item"; continue; }
    mkdir -p "$d"; echo "$item" > "$d/item.txt"
    # stale triggers would fire during warm-up and leave a trace under a window's label
    [ "$arm" = prof ] && on "$h" "mkdir -p $RCDIR/prof-trigger; rm -f $RCDIR/prof-trigger/*; rm -rf $RCDIR/prof/prof-*"
    mp=$(memlog "$h" "$d/mem.log")
    if ! boot_arm "$h" "$rec" "$d"; then
      keep_logs "$h" "$d"; kill "$mp" 2>/dev/null; echo fail > "$d/FAILED"; stop_host "$h"
      [ "$arm" = tp1-off ] && { log "$node: control boot failed: node stops"; return 1; }
      skip="$skip$arm "; continue; fi
    warm "$h"
    if [ "$arm" = prof ]; then pass_prof "$h" "$pass" "$d"; else PROBE_TEMP=${temp:-0} pass_screen "$h" "$d"; fi
    keep_logs "$h" "$d"; kill "$mp" 2>/dev/null
    log "$node: $item done, min MemAvailable $(minmem "$d/mem.log") GiB"
    if [ "$arm" = tp1-ple-cpuhash ] && grep -q "PLE cpu-hash MISMATCH" "$d/serve.log"; then
      log "$node: CANARY tp1-ple-cpuhash ids differ from the GPU hash: dropping"; echo "$node $arm" >> "$RESULTS/CANARY_FAILED"; skip="$skip$arm "; fi
    if [[ " $CANARY_ARMS " == *" $arm "* ]] && [ "$pass" = 1 ] && [ -z "$temp" ] \
        && ! canary "$RESULTS/screen/$node/tp1-off-p1" "$d"; then
      log "$node: CANARY $arm acceptance collapsed vs tp1-off p1: dropping"; echo "$node $arm" >> "$RESULTS/CANARY_FAILED"; skip="$skip$arm "; fi
    stop_host "$h"
  done; }

log "driver start pid=$$ pgid=$(ps -o pgid= $$ | tr -d ' ') SEQ_H1=${SEQ_H1:-} SEQ_H2=${SEQ_H2:-}"
set_state running; OWNS=1
stop_all
P1=; P2=
[ -n "${SEQ_H1:-}" ] && { node_run "$H1" "$SEQ_H1" & P1=$!; }
[ -n "${SEQ_H2:-}" ] && { node_run "$H2" "$SEQ_H2" & P2=$!; }
[ -n "$P1" ] && wait "$P1"; [ -n "$P2" ] && wait "$P2"
log "both nodes done"
if [ -n "${POST:-}" ]; then set_state post; log "post: $POST"; bash -c "$POST" >> "$LOG" 2>&1; log "post exit=$?"; fi
restore
exit 0
