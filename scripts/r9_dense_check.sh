#!/usr/bin/env bash
# r9 dense NVFP4 first-boot numerics (opus-kernel-17, 2026-10-02). Run detached on dgx-01:
#   SHIPPED_IMAGE_EXPECT=<b1.4 tag> RESULTS=<dir> setsid nohup bash scripts/r9_dense_check.sh <STATE file> > <log> 2>&1 < /dev/null &
# Same b1.4 image and recipe, only the checkpoint snapshot differs (f4000...01 = shipped 7c4f1bc1 with
# GDN qkv/z/out + attention q/k/v/o as W4A16_NVFP4, scripts/r9_requant_dense.py). Two TP=2 boots:
#   base = the b1.4 registry recipe, dense = the same recipe on the f400 snapshot. Each boot: logits
#   capture x2 (scripts/logits_equiv.py; a/b = self-noise), T=0 paired probes fresh c1 / 16k c1 /
#   count c1 / fresh c4 with per-position acceptance, then on dense only fidelity 32k + 128k (seed 7).
# Canary: dense pos-0 acceptance < 0.7x base on any cell -> skip fidelity, mark CANARY.
# EXIT restores the registry recipe and verifies image + health + pong. Holds ~/GEN-AI/gpu-lock.
set -u
WAIT=${1:?usage: r9_dense_check.sh <STATE file>}
: "${SHIPPED_IMAGE_EXPECT:?}" "${RESULTS:?}"
export PATH="$HOME/.local/bin:$PATH"
REPO=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2; H2=192.168.100.53
SNAP=f400000000000000000000000000000000000001; SHIP=7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
CODE=$REPO/results/corpus-code.txt
GEN="List the numbers from 1 to 300 separated by commas. Output only the numbers, nothing else, no commentary."
mkdir -p "$RESULTS"; LOG=$RESULTS/driver.log
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$LOG"; }
st() { echo "$*" > "$RESULTS/STATE"; log "STATE: $*"; }
st waiting; log "waiting for $WAIT"
until grep -qE '^(DONE|FAILED|SKIPPED)' "$WAIT" 2>/dev/null; do sleep 60; done
log "wait over: $(head -1 "$WAIT")"
log "taking gpu-lock"; exec 9>"$HOME/GEN-AI/gpu-lock"; flock 9; log "gpu-lock held"
health() { curl -s -m 5 -o /dev/null -w '%{http_code}' localhost:8000/health; }
wait_health() { local s=$(date +%s); until [ "$(health)" = 200 ]; do [ $(( $(date +%s) - s )) -gt 5400 ] && return 1; sleep 15; done; log "health after $(( $(date +%s) - s ))s"; }
pong() { curl -s -m 120 localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Reply with exactly one word: pong"}],"max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>&1; }
stop_all() { sparkrun stop --all >>"$LOG" 2>&1; sleep 10; docker ps -q | xargs -r docker rm -f >/dev/null; ssh -n $H2 'docker ps -q | xargs -r docker rm -f' >/dev/null; }
FINAL="FAILED: check"
restore() { st restore; stop_all; sparkrun run qwen3.8-flash-next-2x-dgx-spark --no-follow >>"$LOG" 2>&1; wait_health; p=$(pong)
  img=$(docker ps --format '{{.Image}}' | grep spark-vllm); log "restored pong=$p image=$img"
  case "$img|$p" in *"$SHIPPED_IMAGE_EXPECT"*"|"*[Pp]ong*) st "$FINAL" ;; *) log "FAILED FAILED: restore -- MANUAL INTERVENTION"; st "FAILED: restore" ;; esac; }
trap restore EXIT
BASE=$(find "$HOME/.cache/sparkrun/registries" -path '*recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml' | head -1)
grep -q "^container: .*$SHIPPED_IMAGE_EXPECT" "$BASE" || { log "registry recipe does not serve b1.4"; exit 1; }
cp "$BASE" "$RESULTS/base.yaml"
sed -e "s/^name: .*/name: qwen3.8-flash-next-2x-dgx-spark-r9-dense/" -e "s/$SHIP/$SNAP/g" "$BASE" > "$RESULTS/dense.yaml"
[ "$(grep -c "$SNAP" "$RESULTS/dense.yaml")" -ge 2 ] || { log "revision swap failed"; exit 1; }
pos() { curl -s -m 10 localhost:8000/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens_per_pos_total'; }
probe() { # dir tag src depth conc reps offset
  local d=$1 tag=$2 src=$3 depth=$4 c=$5 r=$6 off=$7; pos > "$d/pos-$tag.before"
  if [ "$src" = count ]; then ( cd "$REPO" && python3 scripts/depth_decode_probe.py --label "$tag" --prompt "$GEN" --concurrency "$c" \
      --repeats "$r" --max-tokens 320 --temperature 0 --out "$d/probe-$tag.json" > "$d/probe-$tag.log" 2>&1 )
  else ( cd "$REPO" && python3 scripts/depth_decode_probe.py --label "$tag" --corpus "$src" --depth "$depth" --new 2048 --concurrency "$c" \
      --repeats "$r" --offset "$off" --max-tokens 512 --temperature 0 --out "$d/probe-$tag.json" > "$d/probe-$tag.log" 2>&1 ); fi
  pos > "$d/pos-$tag.after"; log "$(basename "$d"): probe $tag exit=$?"; }
acc0() { python3 - "$1/pos-$2.before" "$1/pos-$2.after" <<'PY'
import re,sys
def rd(f):
    v={}
    for l in open(f):
        m=re.match(r'vllm:spec_decode_num_(accepted|draft)_tokens_per_pos_total\{.*position="(\d)"\} ([0-9.e+]+)',l)
        if m: v[(m.group(1),m.group(2))]=float(m.group(3))
    return v
a,b=rd(sys.argv[1]),rd(sys.argv[2]); g=lambda k:b.get(k,0)-a.get(k,0)
print(" ".join("%.3f"%(g(("accepted",str(i)))/(g(("draft","0")) or 1)) for i in range(4)))
PY
}
run_boot() { # name recipe
  local d=$RESULTS/$1; mkdir -p "$d"; st "boot:$1"; stop_all
  ( cd "$RESULTS" && sparkrun run "$2" --no-follow >> "$d/sparkrun.log" 2>&1 )
  wait_health || { log "$1 boot failed"; c=$(docker ps -a --format '{{.Names}}' | grep node_0 | head -1); docker logs "$c" > "$d/serve.log" 2>&1; return 1; }
  log "$1 pong=$(pong)"; st "probe:$1"
  ( cd "$REPO" && python3 scripts/logits_equiv.py capture --out "$d/logits-a.json" > "$d/logits.log" 2>&1 \
      && python3 scripts/logits_equiv.py capture --out "$d/logits-b.json" >> "$d/logits.log" 2>&1 )
  probe "$d" warm "$CODE" 4096 1 1 900000
  probe "$d" fresh-c1 "$CODE" 256 1 8 104729; probe "$d" d16k-c1 "$CODE" 16384 1 8 1151909
  probe "$d" count-c1 count 0 1 8 0; probe "$d" fresh-c4 "$CODE" 256 4 3 314187
  c=$(docker ps --format '{{.Names}}' | grep node_0 | head -1); docker exec "$c" cat /tmp/sparkrun_serve.log > "$d/serve.log" 2>&1 || docker logs "$c" > "$d/serve.log" 2>&1
  for t in fresh-c1 d16k-c1 count-c1 fresh-c4; do log "$1 $t acc/pos $(acc0 "$d" "$t")"; done; }
st running
run_boot base "$RESULTS/base.yaml" || exit 1
run_boot dense "$RESULTS/dense.yaml" || { FINAL="FAILED: dense boot"; exit 1; }
( cd "$REPO" && for p in a b; do python3 scripts/logits_equiv.py diff "$RESULTS/base/logits-$p.json" "$RESULTS/dense/logits-$p.json"; done
  python3 scripts/logits_equiv.py diff "$RESULTS/base/logits-a.json" "$RESULTS/base/logits-b.json" ) > "$RESULTS/logits-diff.txt" 2>&1
bad=0; for t in fresh-c1 d16k-c1 count-c1 fresh-c4; do
  b=$(acc0 "$RESULTS/base" $t | cut -d' ' -f1); x=$(acc0 "$RESULTS/dense" $t | cut -d' ' -f1)
  python3 -c "import sys; sys.exit(0 if $x >= 0.7*$b else 1)" || bad=1; done
if [ $bad = 1 ]; then log "CANARY: dense pos-0 acceptance < 0.7x base"; FINAL="DONE: CANARY"; exit 0; fi
st fidelity; ( cd "$REPO" && python3 scripts/fidelity_probe.py --base http://localhost:8000 --model qwen3.8-flash-next \
  --depths 32000,128000 --seed 7 --out "$RESULTS/dense/fidelity.json" > "$RESULTS/dense/fidelity.txt" 2>&1 )
log "fidelity: $(tail -3 "$RESULTS/dense/fidelity.txt" | tr '\n' ' ')"
FINAL=DONE; exit 0
