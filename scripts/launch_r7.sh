#!/bin/bash
# r7 screening (opus-kernel-10, 2026-09-30): one build (vLLM exp/r7-screen = shipped-b1.3-20260929
# 7344a9976 + exp/r7-a16 920bed9f + feat/r6-dvocab 511d0fc0 -> fef484ad; b12x exp/r7-a16 ecf2e2a7 =
# b7fbaf96 + upstream b4b12bcf), then same-image screening vs r7-off:
#   temp 0 ABBA x2: dvocab98k, dvocab128k (dvocab v2 id lists), a16-8, a16-24 (b12x A16 cutoff);
#   temp 1.0: dvocab98k, dvocab128k.
# Waits (detached) until ~/GEN-AI/tp1-gate/STATE reads DONE or FAILED* and its run.sh has exited.
# EXIT restores the registry recipe (b1.3). Run on dgx-01:
#   SHIPPED_IMAGE_EXPECT=b1.3-20260929-b7fbaf96-7344a997-warm bash ~/GEN-AI/r7/launch_r7.sh
set -u
: "${SHIPPED_IMAGE_EXPECT:?set to the tag the registry recipe serves (b1.3)}"
export SHIPPED_IMAGE_EXPECT
WORKER_IP=192.168.100.53
RC=$HOME/.cache/sparkrun/runtime-cache/vllm/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-2d9615ab
SRC=$HOME/GEN-AI/r4-recipes/archive/mods/r7-dvocab
SUMS="396cdfa7872aad2b9b19f8f9bebd11df461e4a0ef30f2d5b850202658eaf56b8  ids-v2-K98304.txt.gz
fc40970533741fc14217879a5f5385897f4126bd1d7ce23ed0268236b4b8cf7b  ids-v2-K131072.txt.gz"
# Stage + verify the id lists in the runtime cache on both nodes (sparkrun mounts $RC at /cache/runtime).
(cd "$SRC" && echo "$SUMS" | sha256sum -c --quiet) || exit 1
mkdir -p "$RC/r7" && cp "$SRC"/ids-v2-K98304.txt.gz "$SRC"/ids-v2-K131072.txt.gz "$RC/r7/" || exit 1
(cd "$RC/r7" && echo "$SUMS" | sha256sum -c --quiet) || exit 1
ssh $WORKER_IP "mkdir -p $RC/r7" && scp -q "$RC"/r7/ids-v2-K98304.txt.gz "$RC"/r7/ids-v2-K131072.txt.gz "$WORKER_IP:$RC/r7/" || exit 1
ssh $WORKER_IP "cd $RC/r7 && echo '$SUMS' | sha256sum -c --quiet" || exit 1
R=~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/r7-screen-$(date +%Y%m%d-%H%M); mkdir -p $R
export RESULTS=$R WAIT_FOR_STATE_FILE=$HOME/GEN-AI/tp1-gate/STATE WAIT_FOR_DIR=$HOME/GEN-AI/tp1-gate
export STAGES="${STAGES:-build:r7 screen:r7-off:1 screen:r7-dvocab98k:1 screen:r7-dvocab128k:1 screen:r7-a16-8:1 screen:r7-a16-24:1 screen:r7-a16-24:2 screen:r7-a16-8:2 screen:r7-dvocab128k:2 screen:r7-dvocab98k:2 screen:r7-off:2 screen:r7-off:3:1.0 screen:r7-dvocab98k:3:1.0 screen:r7-dvocab128k:3:1.0 screen:r7-dvocab128k:4:1.0 screen:r7-dvocab98k:4:1.0 screen:r7-off:4:1.0}"
export VLLM_REF_r7=exp/r7-screen B12X_REF_r7=exp/r7-a16 TAG_r7=r7-ecf2e2a7-fef484ad
cd ~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
setsid nohup bash scripts/r7_driver.sh > $R/nohup.log 2>&1 < /dev/null & disown
echo "$R pid=$!"
