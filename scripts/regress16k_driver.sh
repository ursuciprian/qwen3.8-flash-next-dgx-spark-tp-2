#!/usr/bin/env bash
# 16k low-concurrency regression, GPU phase: shipped vs candidate-off.
# WRITTEN, NOT LAUNCHED. Needs the pair to itself (no other driver queued or
# running); an EXIT trap reboots the shipped recipe. Journal tag opus:16k-regression.
#
# Per arm: boot a copy of the arm's recipe with mods/vllm-decode-profiler and
# a trigger dir, then run scripts/depth_decode_probe.py:
#   d2k c1   (control, 1 profile window)
#   d16k c1  x8 (1 profile window, mid-decode of repeat 2)
#   d16k c2  x4
#   d64k c1  x3 (1 profile window; tests the QSA stable-selection scan that
#            grows with context, see results/regress16k-README in the report)
# Per window: rank0+rank1 chrome traces -> prof_summary.py; at the end
# prof_compare.py prints per-kernel us/step deltas candoff vs shipped.
#
#   setsid nohup bash scripts/regress16k_driver.sh >/tmp/regress16k.nohup 2>&1 &
set -u
export PATH="$HOME/.local/bin:$PATH"
REPO=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
PORTABLE=$HOME/GEN-AI/portable-test
CAND_RECIPES=$HOME/GEN-AI/candidate-recipes
WORKER_IP=192.168.100.53
RCDIR=$HOME/.cache/sparkrun/runtime-cache/vllm/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-2d9615ab
SHIPPED_RECIPE_REL=recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml
OFF_RECIPE_REL=archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-candidate-off.yaml
PROF_STEPS=60
OUT=$REPO/results/regress16k-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)
mkdir -p "$OUT"
LOG=$OUT/driver.log
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$LOG"; }

wait_health() {
  local t0=$SECONDS
  while (( SECONDS - t0 < ${1:-5400} )); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health)" = 200 ] && return 0
    sleep 10
  done
  return 1
}

stop_all() { sparkrun stop --all >>"$LOG" 2>&1; sleep 20; }

boot() {  # boot <dir> <recipe_rel>
  stop_all
  log "boot $2 (dir=$1)"
  ( cd "$1" && sparkrun run "$2" --no-follow >>"$LOG" 2>&1 )
  wait_health 5400 || { log "FATAL: health timeout for $2"; exit 1; }
}

restore_shipped() {
  log "restore: shipped"
  boot "$PORTABLE" "$SHIPPED_RECIPE_REL" && log "restore: shipped healthy"
}
trap restore_shipped EXIT

# Copy of an arm's recipe plus the profiler mod and trigger dir, placed where
# mods/ resolves (archive/recipes/qwen3.8-flash-next/mods -> archive/mods).
make_prof_recipe() {  # make_prof_recipe <src_yaml> <arm> -> prints rel path
  local rel=archive/recipes/qwen3.8-flash-next/regress16k-$2.yaml
  python3 - "$1" "$CAND_RECIPES/$rel" "$2" "$PROF_STEPS" <<'PY'
import re, sys
src, dst, arm, steps = sys.argv[1:]
text = open(src).read()
assert "\nmods:" not in text, "recipe already has mods; merge by hand"
text = re.sub(r"^name: .*$", f"name: regress16k-{arm}", text, count=1, flags=re.M)
text = text.replace("\ndefaults:", "\nmods:\n  - vllm-decode-profiler\n\ndefaults:", 1)
text = text.replace("\nenv:\n", "\nenv:\n  VLLM_LOCAL_PROF_TRIGGER_DIR: \"/cache/runtime/prof-trigger\"\n"
                    f"  VLLM_LOCAL_PROF_STEPS: \"{steps}\"\n", 1)
open(dst, "w").write(text)
PY
  echo "$rel"
}

trigger_cmd() {  # a unique filename per window; its content is the trace label
  local label=$1
  echo "n=go-\$(date +%s%N); echo $label > $RCDIR/prof-trigger/\$n; ssh $WORKER_IP \"echo $label > $RCDIR/prof-trigger/\$n\""
}

collect() {  # collect <label>
  sleep 45   # PROF_STEPS decode steps + chrome-trace export
  mkdir -p "$OUT/$1"
  cp "$RCDIR/prof/$1/rank0.json" "$OUT/$1/" 2>>"$LOG" || log "WARN: no rank0 trace for $1"
  scp -q "$WORKER_IP:$RCDIR/prof/$1/rank1.json" "$OUT/$1/" 2>>"$LOG" || log "WARN: no rank1 trace for $1"
  python3 "$REPO/scripts/prof_summary.py" "$1" "$PROF_STEPS" "$OUT/$1"/rank*.json > "$OUT/$1/summary.txt" 2>&1
  gzip -f "$OUT/$1"/rank*.json
}

probe() {  # probe <arm> <tag> <extra args...>
  local arm=$1 tag=$2; shift 2
  log "$arm: probe $tag $*"
  ( cd "$REPO" && python3 scripts/depth_decode_probe.py --label "$arm-$tag" \
      --out "$OUT/$arm-$tag.json" "$@" ) > "$OUT/$arm-$tag.log" 2>&1 || log "WARN: probe $arm-$tag failed"
}

run_arm() {  # run_arm <arm> <dir> <recipe_rel>
  local arm=$1 rel
  rel=$(make_prof_recipe "$2/$3" "$arm")
  mkdir -p "$RCDIR/prof-trigger"; ssh "$WORKER_IP" "mkdir -p $RCDIR/prof-trigger"
  rm -f "$RCDIR"/prof-trigger/*; ssh "$WORKER_IP" "rm -f $RCDIR/prof-trigger/*"
  boot "$CAND_RECIPES" "$rel"
  local n0 n1
  n0=$(docker ps --format '{{.Names}}' | grep node_0 | head -1)
  n1=$(ssh "$WORKER_IP" "docker ps --format '{{.Names}}'" | grep node_1 | head -1)
  local marker='local step profiler (mods/vllm-decode-profiler)' runner=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py
  docker exec "$n0" grep -q "$marker" "$runner" && ssh "$WORKER_IP" "docker exec $n1 grep -q '$marker' $runner" \
    || { log "FATAL: $arm profiler mod not applied on both ranks"; exit 1; }
  docker inspect --format '{{.Config.Image}}' "$n0" > "$OUT/$arm-image.txt"
  probe "$arm" warm --depth 6000 --new 64 --repeats 1
  probe "$arm" d2k-c1 --depth 2048 --new 64 --repeats 4 --trigger-cmd "$(trigger_cmd "$arm-d2k-c1")"
  collect "$arm-d2k-c1"
  probe "$arm" d16k-c1 --depth 16384 --new 2048 --repeats 8 --trigger-cmd "$(trigger_cmd "$arm-d16k-c1")"
  collect "$arm-d16k-c1"
  probe "$arm" d16k-c2 --depth 16384 --new 2048 --repeats 4 --concurrency 2
  probe "$arm" d64k-c1 --depth 65536 --new 2048 --repeats 3 --trigger-cmd "$(trigger_cmd "$arm-d64k-c1")"
  collect "$arm-d64k-c1"
}

log "start OUT=$OUT"
run_arm shipped "$PORTABLE" "$SHIPPED_RECIPE_REL"
run_arm candoff "$CAND_RECIPES" "$OFF_RECIPE_REL"
for w in d2k-c1 d16k-c1 d64k-c1; do
  python3 "$REPO/scripts/prof_compare.py" "$OUT/shipped-$w" "$OUT/candoff-$w" "$PROF_STEPS" > "$OUT/compare-$w.txt" 2>&1
done
log "done; compare-*.txt and *-d16k-*.json in $OUT"
