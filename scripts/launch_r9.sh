#!/usr/bin/env bash
# r9 TP=2 small-arm screen (opus-kernel-17, 2026-10-02). Run detached on dgx-01:
#   SHIPPED_IMAGE_EXPECT=<b1.4 tag> R9_TAG=<tag> setsid nohup bash scripts/launch_r9.sh <STATE file> > <log> 2>&1 < /dev/null &
# 1. waits for <STATE file> DONE/FAILED/SKIPPED;
# 2. builds the r9 image (VLLM_REF / B12X_REF, default feat/r9-pin on both forks) if it is not on
#    both nodes, while the shipped b1.4 keeps serving; checks the image carries the pin code;
# 3. writes the r9 arms (r8_make_recipes.py tp2 --only $ARMS --image) from the b1.4 registry recipe;
# 4. takes ~/GEN-AI/gpu-lock and runs scripts/r8_driver.sh (ABBA vs r9-off = same image, knobs off).
set -u
WAIT=${1:?usage: launch_r9.sh <STATE file to wait for>}
: "${SHIPPED_IMAGE_EXPECT:?set to the tag the registry recipe serves (b1.4)}"
: "${R9_TAG:?image tag, e.g. r9-85c4f24d-72e0b8e1}"
export SHIPPED_IMAGE_EXPECT PATH="$HOME/.local/bin:$PATH"
REPO=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
RECIPES=$HOME/GEN-AI/r4-recipes; AD=$RECIPES/archive/recipes/qwen3.8-flash-next
H2=192.168.100.53
VLLM_REF=${VLLM_REF:-feat/r9-pin}; B12X_REF=${B12X_REF:-feat/r9-pin}
IMAGE=spark-vllm-b12x:$R9_TAG
ARMS=${ARMS:-r9-off,r9-pin}
export RESULTS=${RESULTS:-$REPO/results/r9-screen-$(date +%Y%m%d-%H%M)}
export CONTROL_ARM=r9-off CANARY_ARMS=${CANARY_ARMS:-"r9-pin r9-gemv"}
STAGES=${STAGES:-"screen:r9-off:1 screen:r9-pin:1 screen:r9-pin:2 screen:r9-off:2"}
mkdir -p "$RESULTS"; L=$RESULTS/launch.log
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$L"; }
echo waiting > "$RESULTS/STATE"; log "waiting for $WAIT"
until grep -qE '^(DONE|FAILED|SKIPPED)' "$WAIT" 2>/dev/null; do sleep 60; done
log "wait over: $(head -1 "$WAIT")"
if docker image inspect "$IMAGE" >/dev/null 2>&1 && ssh $H2 "docker image inspect $IMAGE >/dev/null 2>&1"; then
  log "image $IMAGE present on both nodes"
else
  echo build > "$RESULTS/STATE"
  ( cd "$HOME/GEN-AI/gdndef-build/spark-vllm-b12x" && CONFIRM_BUILD=1 VLLM_REF="$VLLM_REF" B12X_REF="$B12X_REF" \
      DISTRIBUTE_TO="$H2" IMAGE_TAG_SUFFIX=-r9 BUILD_ROOT="$HOME/GEN-AI/build" ./build.sh > "$RESULTS/build.log" 2>&1 ) \
    || { log "build FAILED (see build.log)"; echo "FAILED: build" > "$RESULTS/STATE"; exit 1; }
  built=$(grep -oE "spark-vllm-b12x:local-[0-9]+-[0-9a-f]+-r9" "$RESULTS/build.log" | tail -1)
  [ -n "$built" ] && docker tag "$built" "$IMAGE" && ssh $H2 "docker tag '$built' '$IMAGE'" \
    || { log "tagging $built failed"; echo "FAILED: build tag" > "$RESULTS/STATE"; exit 1; }
  log "built $built -> $IMAGE"
fi
P=/usr/local/lib/python3.12/dist-packages
docker run --rm --entrypoint grep "$IMAGE" -q VLLM_B12X_BLOCKSCALED_PIN $P/vllm/envs.py \
  && docker run --rm --entrypoint grep "$IMAGE" -q "overrides=None" $P/b12x/gemm/blockscaled/_preparation.py \
  || { log "image $IMAGE lacks the r9 pin code"; echo "FAILED: image" > "$RESULTS/STATE"; exit 1; }
BASE=$(find "$HOME/.cache/sparkrun/registries" -path '*recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml' | head -1)
grep -q "^container: .*$SHIPPED_IMAGE_EXPECT" "$BASE" \
  || { log "registry recipe $BASE does not serve $SHIPPED_IMAGE_EXPECT"; echo "FAILED: base" > "$RESULTS/STATE"; exit 1; }
cp "$BASE" "$RESULTS/base-recipe.yaml"
python3 "$REPO/scripts/r8_make_recipes.py" tp2 --base "$BASE" --out "$AD" --only "$ARMS" --image "$IMAGE" >> "$L" 2>&1 \
  || { log "recipe generation failed"; echo "FAILED: recipes" > "$RESULTS/STATE"; exit 1; }
for a in ${ARMS//,/ }; do
  sparkrun recipe validate "$AD/qwen3.8-flash-next-2x-dgx-spark-$a.yaml" >> "$L" 2>&1 \
    || { log "validate failed: $a"; echo "FAILED: validate" > "$RESULTS/STATE"; exit 1; }
done
# GPU_LOCK_HELD=1: the caller already holds ~/GEN-AI/gpu-lock (fd inherited); a second flock would deadlock.
if [ -n "${GPU_LOCK_HELD:-}" ]; then log "gpu-lock held by the caller"
else log "taking gpu-lock"; exec 9>"$HOME/GEN-AI/gpu-lock"; flock 9; log "gpu-lock held"; fi
export STAGES
log "start driver: RESULTS=$RESULTS STAGES=$STAGES"
cd "$REPO" && exec bash scripts/r8_driver.sh
