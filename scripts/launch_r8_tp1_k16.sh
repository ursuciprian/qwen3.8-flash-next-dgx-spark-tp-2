#!/usr/bin/env bash
# r8 TP=1 kernel arms (opus-kernel-16, 2026-10-01). PREPARED, launch only on the orchestrator's go:
#   SHIPPED_IMAGE_EXPECT=b1.4-20261001-b7fbaf96-a7e649d8-warm setsid nohup \
#     bash scripts/launch_r8_tp1_k16.sh <STATE file to wait for> > ~/GEN-AI/r8-tp1-k16.nohup 2>&1 < /dev/null &
# 1. waits for <STATE file> DONE/FAILED/SKIPPED, then takes ~/GEN-AI/gpu-lock for the whole run;
# 2. builds one image (vLLM feat/r8-mtp-gemv + b12x exp/r8-tp1-wm) if absent on either node;
# 3. writes only the k16 arms (r8_make_recipes.py --only), all on that image, control tp1-willneed
#    (= the r8 PLE winner tp1-ple-willneed; every knob below is off in it);
# 4. runs scripts/r8_tp1_driver.sh, ABBA per node, canary on every arm (pos-0 acc < 0.7x control).
# Arms: tp1-gemv (dgx-01), tp1-gemv-mtp (dgx-02), ~35 min per pass -> ~2.3 h. tp1-wm2m8 is
# written too but not in the default sequences (I=640 microbench: wm = dynamic at M=5).
# Refuses to start unless results/r8-wm-bench/gemv-test.txt (job2, GPU op test) passed.
set -u
WAIT=${1:?usage: launch_r8_tp1_k16.sh <STATE file to wait for>}
: "${SHIPPED_IMAGE_EXPECT:?set to the tag the registry recipe serves (b1.4)}"
export SHIPPED_IMAGE_EXPECT PATH="$HOME/.local/bin:$PATH"
REPO=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
RECIPES=$HOME/GEN-AI/r4-recipes; AD=$RECIPES/archive/recipes/qwen3.8-flash-next
H2=192.168.100.53
VLLM_REF=${VLLM_REF:-feat/r8-mtp-gemv}; B12X_REF=${B12X_REF:-exp/r8-tp1-wm}
TP1_TAG=${TP1_TAG:-r8tp1k16-5bf24021-71e4f478}
IMAGE=spark-vllm-b12x:$TP1_TAG
ARMS=tp1-willneed,tp1-gemv-mtp,tp1-gemv,tp1-wm2m8
export RESULTS=${RESULTS:-$REPO/results/r8-tp1-k16-$(date +%Y%m%d-%H%M)}
export CONTROL_ARM=tp1-willneed CANARY_ARMS="tp1-gemv-mtp tp1-gemv tp1-wm2m8"
export SEQ_H1=${SEQ_H1-"tp1-willneed:1 tp1-gemv:1 tp1-gemv:2 tp1-willneed:2"}
export SEQ_H2=${SEQ_H2-"tp1-willneed:1 tp1-gemv-mtp:1 tp1-gemv-mtp:2 tp1-willneed:2"}
mkdir -p "$RESULTS"; L=$RESULTS/launch.log
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$L"; }
echo waiting > "$RESULTS/STATE"
log "waiting for $WAIT"
until grep -qE '^(DONE|FAILED|SKIPPED)' "$WAIT" 2>/dev/null; do sleep 60; done
log "wait over: $(head -1 "$WAIT")"
B=$REPO/results/r8-wm-bench
until grep -q DONE "$B/job2.STATE" 2>/dev/null; do sleep 60; done
grep -q " passed" "$B/gemv-test.txt" && ! grep -qiE "failed|error" "$B/gemv-test.txt" \
  || { log "b12x GEMV op GPU test did not pass ($B/gemv-test.txt)"; echo "FAILED: gemv test" > "$RESULTS/STATE"; exit 1; }
log "taking gpu-lock"
exec 9>"$HOME/GEN-AI/gpu-lock"; flock 9; log "gpu-lock held"
if docker image inspect "$IMAGE" >/dev/null 2>&1 && ssh $H2 "docker image inspect $IMAGE >/dev/null 2>&1"; then
  log "image $IMAGE present on both nodes"
else
  echo build > "$RESULTS/STATE"
  ( cd "$HOME/GEN-AI/gdndef-build/spark-vllm-b12x" && CONFIRM_BUILD=1 VLLM_REF="$VLLM_REF" B12X_REF="$B12X_REF" \
      DISTRIBUTE_TO="$H2" IMAGE_TAG_SUFFIX=-r8tp1k16 BUILD_ROOT="$HOME/GEN-AI/build" ./build.sh > "$RESULTS/build.log" 2>&1 ) \
    || { log "build FAILED (see build.log)"; echo "FAILED: build" > "$RESULTS/STATE"; exit 1; }
  built=$(grep -oE "spark-vllm-b12x:local-[0-9]+-[0-9a-f]+-r8tp1k16" "$RESULTS/build.log" | tail -1)
  [ -n "$built" ] && docker tag "$built" "$IMAGE" && ssh $H2 "docker tag '$built' '$IMAGE'" \
    || { log "tagging $built failed"; echo "FAILED: build tag" > "$RESULTS/STATE"; exit 1; }
  log "built $built -> $IMAGE"
fi
P=/usr/local/lib/python3.12/dist-packages
docker run --rm --entrypoint grep "$IMAGE" -q VLLM_QWEN38_B12X_GEMV $P/vllm/envs.py \
  && docker run --rm --entrypoint grep "$IMAGE" -q RING_CHOICES $P/b12x/moe/_shared/kernels/wm_geometry.py \
  || { log "image $IMAGE lacks the k16 code"; echo "FAILED: image" > "$RESULTS/STATE"; exit 1; }
( cd "$RECIPES" && python3 "$REPO/scripts/r8_make_recipes.py" tp1 --image "$IMAGE" --only "$ARMS" \
    --base "$AD/qwen3.8-flash-next-1x-dgx-spark-tp1-safe.yaml" --out "$AD" ) >> "$L" 2>&1 \
  || { log "recipe generation failed"; echo "FAILED: recipes" > "$RESULTS/STATE"; exit 1; }
for a in ${ARMS//,/ }; do
  sparkrun recipe validate "$AD/qwen3.8-flash-next-1x-dgx-spark-$a.yaml" >> "$L" 2>&1 \
    || { log "validate failed: $a"; echo "FAILED: validate" > "$RESULTS/STATE"; exit 1; }
done
scp -q "$REPO/scripts/ple_pagecache.py" "$H2:$REPO/scripts/ple_pagecache.py" || { log "scp ple_pagecache.py failed"; exit 1; }
log "start driver: RESULTS=$RESULTS SEQ_H1=$SEQ_H1 SEQ_H2=$SEQ_H2"
cd "$REPO" && bash scripts/r8_tp1_driver.sh
log "driver exit=$?"
