#!/bin/bash
# r7 resume (opus-kernel-10, 2026-09-30): continues results/r7-screen-20260930-1654 after the
# r7-a16-8 p1 boot died in b12x moe.decode preparation. b12x exp/r7-a16 3d5040e8 adds upstream
# 8783519a (prepared W4A16 route-pack launches pass constexprs) + 57f35723 (native NVFP4 A16
# direct launches retained; contract kept at 4). Non-A16 arms run unchanged kernels; vLLM is the
# same fef484ad. Passes 1 of off/d98/d128 (old image r7-ecf2e2a7-fef484ad) stay in place.
# r7-off p2 runs first so the A16 arms have a same-image control.
# Waits for uf-test/queue, tp1-lane and r7/build-STATE (build_r7b.sh) = DONE|FAILED* (+5 min settle).
# EXIT restores the registry recipe (b1.3). Do not start before the orchestrator says so:
#   SHIPPED_IMAGE_EXPECT=b1.3-20260929-b7fbaf96-7344a997-warm bash ~/GEN-AI/r7/launch_r7_resume.sh
set -u
: "${SHIPPED_IMAGE_EXPECT:?set to the tag the registry recipe serves (b1.3)}"
export SHIPPED_IMAGE_EXPECT
WORKER_IP=192.168.100.53
RC=$HOME/.cache/sparkrun/runtime-cache/vllm/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-2d9615ab
SUMS="396cdfa7872aad2b9b19f8f9bebd11df461e4a0ef30f2d5b850202658eaf56b8  ids-v2-K98304.txt.gz
fc40970533741fc14217879a5f5385897f4126bd1d7ce23ed0268236b4b8cf7b  ids-v2-K131072.txt.gz"
(cd "$RC/r7" && echo "$SUMS" | sha256sum -c --quiet) || exit 1
ssh $WORKER_IP "cd $RC/r7 && echo '$SUMS' | sha256sum -c --quiet" || exit 1
AD=$HOME/GEN-AI/r4-recipes/archive/recipes/qwen3.8-flash-next
TAG=r7-3d5040e8-fef484ad
grep -L "container: spark-vllm-b12x:$TAG" $AD/qwen3.8-flash-next-2x-dgx-spark-r7-*.yaml | grep . && { echo "recipes not on $TAG"; exit 1; }
R=~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/r7-screen-20260930-1654
[ -d $R/screen/r7-a16-8-p1 ] && mv $R/screen/r7-a16-8-p1 $R/screen/r7-a16-8-p1.bootfail-ecf2e2a7
export RESULTS=$R WAIT_FOR_STATE_FILES="$HOME/GEN-AI/uf-test/queue/STATE $HOME/GEN-AI/tp1-lane/STATE $HOME/GEN-AI/r7/build-STATE"
export STAGES="${STAGES:-build:r7 screen:r7-off:2 screen:r7-block:2 screen:r7-a16-8:1 screen:r7-a16-24:1 screen:r7-a16-24:2 screen:r7-a16-8:2 screen:r7-dvocab128k:2 screen:r7-dvocab98k:2 screen:r7-block:1 screen:r7-off:3:1.0 screen:r7-block:3:1.0 screen:r7-dvocab98k:3:1.0 screen:r7-dvocab128k:3:1.0 screen:r7-dvocab128k:4:1.0 screen:r7-dvocab98k:4:1.0 screen:r7-block:4:1.0 screen:r7-off:4:1.0}"
export VLLM_REF_r7=exp/r7-screen B12X_REF_r7=exp/r7-a16 TAG_r7=$TAG
cd ~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
setsid nohup bash scripts/r7_driver.sh > $R/nohup-resume.log 2>&1 < /dev/null & disown
echo "$R pid=$!"
