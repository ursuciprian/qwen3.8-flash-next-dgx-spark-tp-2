#!/bin/bash
# Build-only waiter for the rebuilt r7 image (opus-kernel-10, 2026-09-30). No stop, no boot.
# Waits for uf-test/queue and tp1-lane STATE = DONE|FAILED* (+5 min settle), then build.sh
# vLLM exp/r7-screen + b12x exp/r7-a16 and tags spark-vllm-b12x:r7-3d5040e8-fef484ad on both nodes.
set -u
D=$HOME/GEN-AI/r7; ST=$D/build-STATE; LOG=$D/build-r7b.log; WORKER_IP=192.168.100.53
TAG=spark-vllm-b12x:r7-3d5040e8-fef484ad
st() { echo "$*" > "$ST"; echo "[$(date '+%F %T %Z')] STATE: $*" >> "$LOG"; }
done_all() { grep -qE '^(DONE|FAILED)' $HOME/GEN-AI/uf-test/queue/STATE 2>/dev/null && grep -qE '^(DONE|FAILED)' $HOME/GEN-AI/tp1-lane/STATE 2>/dev/null; }
st waiting
until done_all; do sleep 60; done; sleep 300; until done_all; do sleep 60; done
if docker image inspect $TAG >/dev/null 2>&1 && ssh $WORKER_IP "docker image inspect $TAG >/dev/null 2>&1"; then st DONE present; exit 0; fi
st building
for attempt in 1 2; do
  ( cd $HOME/GEN-AI/gdndef-build/spark-vllm-b12x && CONFIRM_BUILD=1 VLLM_REF=exp/r7-screen B12X_REF=exp/r7-a16 \
      DISTRIBUTE_TO=$WORKER_IP IMAGE_TAG_SUFFIX=-r7 BUILD_ROOT=$HOME/GEN-AI/build ./build.sh > $D/build-r7b-$attempt.log 2>&1 ) || continue
  built=$(grep -oE "spark-vllm-b12x:local-[0-9]+-[0-9a-f]+-r7" $D/build-r7b-$attempt.log | tail -1)
  [ -n "$built" ] && docker tag "$built" $TAG && ssh $WORKER_IP "docker tag '$built' $TAG" && { st "DONE $built"; exit 0; }
done
st "FAILED build"
