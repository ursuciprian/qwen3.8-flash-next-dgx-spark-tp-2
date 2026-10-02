#!/usr/bin/env bash
# r8 TP=1 launcher (opus-kernel-11, 2026-10-01). Run detached on dgx-01:
#   SHIPPED_IMAGE_EXPECT=<b1.4 tag> setsid nohup bash scripts/launch_r8_tp1.sh <STATE file> > <log> 2>&1 &
# 1. waits until <STATE file> reads DONE (FAILED*: exits without touching the pair);
# 2. builds the TP=1 screening image if it is not on both nodes (build.sh, VLLM_REF/B12X_REF,
#    tag spark-vllm-b12x:$TP1_TAG), while the shipped server keeps serving;
# 3. regenerates the tp1-* recipes (scripts/r8_make_recipes.py tp1) from the tp1-safe base
#    with that image, validates them, checks the r7 98k draft-vocab ids on both nodes;
# 4. runs scripts/r8_tp1_driver.sh (SEQ_H1/SEQ_H2, PAGECACHE_*, POST pass through).
# Defaults: the PLE reader arms on dgx-01, the host/spec arms on dgx-02, each ABBA vs tp1-off.
# ARMS=<comma list>: generate/validate only these arms. ANY_STATE=1: start after DONE or FAILED
# (the driver stops every server first). The whole run holds ~/GEN-AI/gpu-lock (flock).
set -u
WAIT=${1:?usage: launch_r8_tp1.sh <STATE file to wait for>}
: "${SHIPPED_IMAGE_EXPECT:?set to the tag the registry recipe serves (b1.4)}"
export SHIPPED_IMAGE_EXPECT PATH="$HOME/.local/bin:$PATH"
REPO=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
RECIPES=$HOME/GEN-AI/r4-recipes; AD=$RECIPES/archive/recipes/qwen3.8-flash-next
H2=192.168.100.53
RC=$HOME/.cache/sparkrun/runtime-cache/vllm/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-2d9615ab
VLLM_REF=${VLLM_REF:-exp/r8-tp1-screen}; B12X_REF=${B12X_REF:-feat/b1.1}
TP1_TAG=${TP1_TAG:?set e.g. r8tp1-b7fbaf96-<vllm sha8> (the vLLM tip of VLLM_REF)}
IMAGE=spark-vllm-b12x:$TP1_TAG
export RESULTS=${RESULTS:-$REPO/results/r8-tp1-$(date +%Y%m%d-%H%M)}
export SEQ_H1=${SEQ_H1-"tp1-off:1 tp1-ple-willneed:1 tp1-ple-par:1 tp1-ple-par:2 tp1-ple-willneed:2 tp1-off:2"}
export SEQ_H2=${SEQ_H2-"tp1-off:1 tp1-trim:1 tp1-d3:1 tp1-d3:2 tp1-trim:2 tp1-off:2"}
mkdir -p "$RESULTS"; L=$RESULTS/launch.log
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$L"; }
echo waiting > "$RESULTS/STATE"
log "waiting for $WAIT (DONE)"
until grep -qE '^(DONE|FAILED|SKIPPED)' "$WAIT" 2>/dev/null; do sleep 60; done
{ grep -q '^DONE' "$WAIT" || [ "${ANY_STATE:-0}" = 1 ]; } || { log "wait state: $(cat "$WAIT"): not starting"; echo "SKIPPED: $(head -1 "$WAIT")" > "$RESULTS/STATE"; exit 1; }
log "wait over: $(cat "$WAIT")"
log "taking gpu-lock"; exec 9>"$HOME/GEN-AI/gpu-lock"; flock 9; log "gpu-lock held"
if docker image inspect "$IMAGE" >/dev/null 2>&1 && ssh $H2 "docker image inspect $IMAGE >/dev/null 2>&1"; then
  log "image $IMAGE present on both nodes"
else
  echo build > "$RESULTS/STATE"
  ( cd "$HOME/GEN-AI/gdndef-build/spark-vllm-b12x" && CONFIRM_BUILD=1 VLLM_REF="$VLLM_REF" B12X_REF="$B12X_REF" \
      DISTRIBUTE_TO="$H2" IMAGE_TAG_SUFFIX=-r8tp1 BUILD_ROOT="$HOME/GEN-AI/build" ./build.sh > "$RESULTS/build.log" 2>&1 ) \
    || { log "build FAILED (see build.log)"; echo "FAILED: build" > "$RESULTS/STATE"; exit 1; }
  built=$(grep -oE "spark-vllm-b12x:local-[0-9]+-[0-9a-f]+-r8tp1" "$RESULTS/build.log" | tail -1)
  [ -n "$built" ] && docker tag "$built" "$IMAGE" && ssh $H2 "docker tag '$built' '$IMAGE'" \
    || { log "tagging $built failed"; echo "FAILED: build tag" > "$RESULTS/STATE"; exit 1; }
  log "built $built -> $IMAGE"
fi
docker run --rm --entrypoint grep "$IMAGE" -q "def reader_knobs" /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/ple_mmap.py \
  || { log "image $IMAGE lacks the r8 PLE reader"; echo "FAILED: image" > "$RESULTS/STATE"; exit 1; }
( cd "$RECIPES" && python3 "$REPO/scripts/r8_make_recipes.py" tp1 --image "$IMAGE" \
    --base "$AD/qwen3.8-flash-next-1x-dgx-spark-tp1-safe.yaml" --out "$AD" ${ARMS:+--only "$ARMS"} ) >> "$L" 2>&1 \
  || { log "recipe generation failed"; echo "FAILED: recipes" > "$RESULTS/STATE"; exit 1; }
for f in "$AD"/qwen3.8-flash-next-1x-dgx-spark-tp1-*.yaml; do
  case $f in *-tp1-safe.yaml|*-tp1-dvocab.yaml) continue ;; esac
  [ -n "${ARMS:-}" ] && [[ ",$ARMS," != *",$(basename "$f" .yaml | sed s/qwen3.8-flash-next-1x-dgx-spark-//),"* ]] && continue
  sparkrun recipe validate "$f" >> "$L" 2>&1 || { log "validate failed: $f"; echo "FAILED: validate" > "$RESULTS/STATE"; exit 1; }
done
SUM="396cdfa7872aad2b9b19f8f9bebd11df461e4a0ef30f2d5b850202658eaf56b8  ids-v2-K98304.txt.gz
fc40970533741fc14217879a5f5385897f4126bd1d7ce23ed0268236b4b8cf7b  ids-v2-K131072.txt.gz"
if [[ "$SEQ_H1 $SEQ_H2" == *tp1-dv* ]]; then
  (cd "$RC/r7" && echo "$SUM" | sha256sum -c --quiet) && ssh $H2 "cd $RC/r7 && echo '$SUM' | sha256sum -c --quiet" \
    || { log "r7 draft-vocab ids missing on a node"; echo "FAILED: ids" > "$RESULTS/STATE"; exit 1; }
fi
scp -q "$REPO/scripts/ple_pagecache.py" "$H2:$REPO/scripts/ple_pagecache.py" || { log "scp ple_pagecache.py failed"; exit 1; }
log "start driver: RESULTS=$RESULTS SEQ_H1=$SEQ_H1 SEQ_H2=$SEQ_H2"
cd "$REPO" && exec bash scripts/r8_tp1_driver.sh
