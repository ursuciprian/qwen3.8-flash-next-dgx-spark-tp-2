#!/usr/bin/env bash
# b16 (2026-10-08, #115, opus-ship-b16): ship b1.6 = the 2x b1.4 + the retrained MTP drafter (k71 PROMOTE + gate PASS).
#   image   P1 = b1.4 warm + docker/mtp-refit (24 refit tensors + overlay.py); CI build-b0-warm on top (recipe_ref
#           feat/b1.6-refit-drafter-2x: seed 15b63901 for the overlay path) pushes TAG. STATE ci-open: the agent
#           triggers CI from the Mac and writes b16/ci-done ("run <id> ok|failed <digest>"), cap 2 h.
#   check   first-boot check of TAG with the branch recipe (plain sparkrun run): overlay dir and the 15b63901 plan file
#           removed from every runtime cache on both Sparks first, so the boot builds the overlay and restores the
#           plan from the image seed. Pass = b12x 0 measured on both nodes, overlay sha ok, pong, fresh-c4 T=0
#           acceptance per position closer to k71's arm than to its control; then llama-benchy pp2048/tg512 d0 c1/c8.
#   end     pass: b1.6 stays up (E = TAG, so restore is a no-op); fail: lib.sh restores b1.4 with a pong.
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=b16; K=$HOME/GEN-AI/backlog/b16
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/b1.6-20261008}
source "$HOME/GEN-AI/backlog/lib.sh"
P1=spark-vllm-b12x:b16-base-b7fbaf96-a7e649d8-mtp16c9bd54
TAG=ghcr.io/ursuciprian/spark-vllm-b12x:b1.6-20261008-b7fbaf96-a7e649d8-warm
REC=$K/recipe/qwen3.8-flash-next-2x-dgx-spark.yaml
SEED=15b639016c932b03d623d76be56a320a94946895d5522c30e3e1d8829572c729
RC=$HOME/.cache/sparkrun/runtime-cache/vllm
SNAP=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
K71=$R/results/k71-tp2-refit-pinned-plans-20261008-0921/screen
CORPUS=$R/results/corpus-code.txt; BENCHY_SRC=$G/llama-benchy-fork; MODEL=qwen3.8-flash-next

pos() { curl -s -m 10 "localhost:8000/metrics" | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens(_per_pos)?_total'; }
acc() { python3 - "$1" "$2" <<'EOF'
import re, sys
def rd(p):
    c = {"accepted": {}, "draft": {}}
    for l in open(p):
        m = re.match(r'vllm:spec_decode_num_(accepted|draft)_tokens_per_pos_total\{.*position="(\d+)"\} ([0-9.e+]+)', l)
        if m:
            k = int(m[2]); c[m[1]][k] = c[m[1]].get(k, 0) + float(m[3])
    return c["accepted"], c["draft"]
(a0, d0), (a1, d1) = rd(sys.argv[1]), rd(sys.argv[2])
print(" ".join(f"{(a1[k] - a0.get(k, 0)) / max(d1[k] - d0.get(k, 0), 1):.3f}" for k in sorted(d1)))
EOF
}
clean_host() { # remove the overlay and its plan file from every runtime cache (first-boot conditions)
  local c="rm -rf $RC/*/mtp-refit; rm -f $RC/*/b12x/compile/preparation/$SEED.json $RC/*/b12x/compile/preparation/$SEED.lock"
  bash -c "$c"; ssh -n -o ConnectTimeout=10 $H2 "$c"; }

if [ "${1:-}" = --dry-run ]; then
  rc=0; ok() { if eval "$2" >/dev/null 2>&1; then echo "ok   $1"; else echo "FAIL $1"; rc=1; fi; }
  ok "syntax" "bash -n $0"
  ok "recipe container $TAG" "grep -qx 'container: $TAG' $REC"
  ok "recipe serves the overlay" "grep -q 'python3 /opt/mtp-refit/overlay.py build || exit 1' $REC && grep -q 'vllm serve /cache/runtime/mtp-refit/' $REC"
  ok "base image $P1 (overlay + patch)" "docker run --rm --network none --entrypoint ls $P1 /opt/mtp-refit/overlay.py /opt/mtp-refit/mtp-refit-16c9bd54.safetensors"
  ok "7c4f1bc1 snapshot on dgx-02" "ssh -n -o ConnectTimeout=10 $H2 test -s $SNAP/model-00034-of-00036.safetensors"
  ok "k71 fresh-c4 pos files" "[ -s $K71/arm-p1/pos-fresh-c4.after ] && [ -s $K71/ctl-p1/pos-fresh-c4.after ]"
  ok "tools" "command -v sparkrun && command -v uvx && [ -d $BENCHY_SRC ] && [ -s $CORPUS ]"
  echo "k71 fresh-c4 acc: arm p1 $(acc $K71/arm-p1/pos-fresh-c4.before $K71/arm-p1/pos-fresh-c4.after) | ctl p1 $(acc $K71/ctl-p1/pos-fresh-c4.before $K71/ctl-p1/pos-fresh-c4.after)"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock; mkdir -p "$RES"; job_begin
st "running: stop 2x"; stop_all
clean_host; log "overlay dirs and $SEED plan files removed on both Sparks"

# ---- CI window ----
rm -f "$K/ci-done"; echo "$P1 $TAG" > "$K/ci-request"; st "ci-open"
s=$(date +%s); until [ -f "$K/ci-done" ] || [ $(( $(date +%s) - s )) -gt 7200 ]; do sleep 20; done
log "ci window closed: $(cat "$K/ci-done" 2>/dev/null || echo timeout)"; rm -f "$K/ci-request"; cp "$K/ci-done" "$RES/ci-done.txt" 2>/dev/null
grep -q ' ok ' "$K/ci-done" 2>/dev/null || { FINAL="FAILED: CI ($(cat "$K/ci-done" 2>/dev/null || echo timeout))"; exit 1; }
docker run --rm --network none --entrypoint ls "$TAG" /opt/b12x-seed/preparation/$SEED.json /opt/mtp-refit/overlay.py > "$RES/image-check.txt" 2>&1 \
  || { FINAL="FAILED: $TAG lacks the seed or the overlay"; exit 1; }
ssh -n -o ConnectTimeout=10 $H2 "docker pull -q $TAG" >> "$RES/$J.log" 2>&1 || log "dgx-02 pull failed; sparkrun copies the image from the head"

# ---- first-boot check ----
st "running: check boot"; D=$RES/check; mkdir -p "$D"; cp "$REC" "$D/recipe.yaml"; clean_host
( cd "$D" && timeout -k 60 900 sparkrun run recipe.yaml --no-follow ) >> "$D/sparkrun.log" 2>&1 < /dev/null 9>&-
s=$(date +%s); up=0
while :; do
  n=$(n0); [ "$(health localhost)" = 200 ] && { up=1; break; }
  if [ $(( $(date +%s) - s )) -gt 180 ] && { [ -z "$n" ] || docker exec "$n" grep -qE 'Worker failed with error|EngineCore failed to start|Engine core initialization failed|^overlay: .*(expected|has no)' /tmp/sparkrun_serve.log 2>/dev/null; }; then break; fi
  [ $(( $(date +%s) - s )) -ge 5400 ] && break; sleep 10; done
echo "boot $(( $(date +%s) - s ))s up=$up" > "$D/boot.txt"; log "check boot: $(cat "$D/boot.txt")"
c=$(docker ps -a --format '{{.Names}}' | grep node_0 | head -1)
docker exec "$c" cat /tmp/sparkrun_serve.log > "$D/serve-node0.log" 2>&1 || docker logs "$c" > "$D/serve-node0.log" 2>&1
ssh -n $H2 'c=$(docker ps -a --format "{{.Names}}" | grep node_1 | head -1); docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1' > "$D/serve-node1.log" 2>&1
[ $up = 1 ] || { FINAL="FAILED: check boot ($(cat "$D/boot.txt"))"; exit 1; }
P=$(pong localhost)
{ echo "image $(docker inspect --format '{{.Config.Image}}' "$c") on node 0; node 1: $(ssh -n $H2 "docker ps --format '{{.Image}}' --filter name=sparkrun")"
  echo "digest $(docker image inspect "$TAG" --format '{{join .RepoDigests " "}}')"
  echo "$(cat "$D/boot.txt"), pong=$P"
  for f in node0 node1; do
    echo "== $f"; grep -E '^overlay: ' "$D/serve-$f.log"
    grep -m1 -oE "model='[^']*'" "$D/serve-$f.log"
    echo "b12x lines with measured > 0: $(grep -E 'b12x .* [0-9]+ measured' "$D/serve-$f.log" | grep -cvE ' 0 measured')"
    echo "max measured: $(grep -oE '[0-9]+ measured' "$D/serve-$f.log" | sort -n | tail -1)"
    grep -E 'b12x (planning|priming|compiling) [^:]*: [0-9]+/[0-9]+ ready' "$D/serve-$f.log" | sed 's/^.*b12x /b12x /' | sort | uniq | tail -8; done
  for h in localhost $H2; do echo "$h plan file: $(ssh -n -o ConnectTimeout=10 $h "ls $RC/*/b12x/compile/preparation/$SEED.json" 2>/dev/null) records $(ssh -n -o ConnectTimeout=10 $h "python3 -c 'import json,glob; print(len(json.load(open(glob.glob(\"$RC/*/b12x/compile/preparation/$SEED.json\")[0]))[\"records\"]))'" 2>/dev/null)"; done
} > "$D/check.txt" 2>&1
# image gate (#115): scripts/check_seed.py = rank-0 log 0 measured + worker plan file still at the seed record count
scp -q $H2:$(ssh -n $H2 "ls $RC/*/b12x/compile/preparation/$SEED.json" | head -1) "$D/plan-node1.json"
python3 "$R/scripts/check_seed.py" "$D/serve-node0.log" "$D/plan-node1.json:616" $(ls $RC/*/b12x/compile/preparation/$SEED.json | head -1):616 > "$D/plans-check.txt" 2>&1 && M=2 || M=0; OV=$(grep -cE '^overlay: (built|.* ready)' "$D/check.txt")
pos > "$D/pos-fresh-c4.before"
( cd "$R" && child timeout -k 30 400 python3 scripts/depth_decode_probe.py --base "http://$H1:8000" --label fresh-c4 --corpus "$CORPUS" \
    --concurrency 4 --repeats 2 --temperature 0 --depth 256 --new 2048 --offset 314187 --max-tokens 512 \
    --out "$D/probe-fresh-c4.json" > "$D/probe-fresh-c4.log" 2>&1 )
pos > "$D/pos-fresh-c4.after"
A=$(acc "$D/pos-fresh-c4.before" "$D/pos-fresh-c4.after")
KA=$(acc $K71/arm-p1/pos-fresh-c4.before $K71/arm-p1/pos-fresh-c4.after); KC=$(acc $K71/ctl-p1/pos-fresh-c4.before $K71/ctl-p1/pos-fresh-c4.after)
DR=$(python3 -c "import sys; a,k,c=(list(map(float,s.split())) for s in sys.argv[1:4]); print('REFIT' if all(abs(x-y)<abs(x-z) for x,y,z in list(zip(a,k,c))[:4]) else 'NOT-REFIT')" "$A" "$KA" "$KC" 2>/dev/null || echo UNKNOWN)
{ echo "fresh-c4 T=0 acceptance per position: this boot $A"; echo "k71 arm p1: $KA"; echo "k71 control (b1.4) p1: $KC"; echo "drafter $DR"; } > "$D/drafter-check.txt"
log "measured-0 nodes $M/2, overlay $OV/2, pong=$P, drafter $DR ($A)"
mkdir -p "$RES/bench"
( cd "$R" && child timeout -k 30 1500 uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$H1:8000/v1" --model $MODEL \
    --tokenizer "$SNAP" --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 --concurrency 1 8 --runs 4 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$H1:8000/metrics" \
    --save-result "$RES/bench/task.csv" > "$RES/bench/benchy.log" 2>&1 ); log "benchy exit=$?"
[ "$(health localhost)" = 200 ] || { FINAL="FAILED: server died during the check bench"; exit 1; }
echo "$RES" > "$K/RESULT"
if [ "$M" = 2 ] && [ "$OV" = 2 ] && echo "$P" | grep -qi pong && [ "$DR" = REFIT ]; then
  E=${TAG##*:}; FINAL="DONE: b1.6 check PASS: 0 measured on both nodes, overlay ok, drafter $DR (fresh-c4 acc $A), $TAG serving"
else FINAL="FAILED: b1.6 check (measured-0 nodes $M/2, overlay $OV/2, pong=$P, drafter $DR)"; fi
exit 0
