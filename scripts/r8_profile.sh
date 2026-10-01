#!/usr/bin/env bash
# r8 Phase 1 TP=1 profile (opus-kernel-11, 2026-10-01). Run detached on dgx-01:
#   SHIPPED_IMAGE_EXPECT=<b1.4 tag> TP1_TAG=<r8 TP=1 image tag> \
#     setsid nohup bash scripts/r8_profile.sh <STATE file> > <log> 2>&1 &
# Both Sparks at once, TP=1, recipe tp1-prof (= tp1-off + mods/vllm-decode-profiler, reader
# stats every 100 reads): dgx-01 with the PLE table read into the page cache first (warm),
# dgx-02 with it evicted (cold). Windows of 60 steps at temperature 0, same prompts and offsets
# as the TP=2 profile (results/r5-p2d-wm2prof): fresh c1, 16k c1, fresh c4, 16k c4, count c1;
# host counters (NVMe reads, MemAvailable, Cached, EngineCore major faults) around each.
# Then, servers down: the TP=2 reference traces re-broken-down with the gap split, the b12x MoE
# M=5 vs M=8 microbench at TP=1 and TP=2 widths, and results/<run>/profile_table.md.
# ~1 h. EXIT (in the driver) restores the shipped TP=2 recipe.
set -u
WAIT=${1:?usage: r8_profile.sh <STATE file to wait for>}
REPO=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
export RESULTS=${RESULTS:-$REPO/results/r8-profile-$(date +%Y%m%d-%H%M)}
export SEQ_H1="prof:tp1-prof" SEQ_H2="prof:tp1-prof" PAGECACHE_H1=touch PAGECACHE_H2=evict
TP2REF=$REPO/results/r5-p2d-wm2prof/prof-r5-wm2offprof
IMAGE=spark-vllm-b12x:${TP1_TAG:?set the r8 TP=1 image tag}
export POST="for w in fresh-c1 d16k-c1 fresh-c4 d16k-c4; do mkdir -p $RESULTS/tp2-ref/prof-\$w; \
python3 $REPO/scripts/step_breakdown.py 60 $TP2REF/prof-\$w/rank0.json.gz --json $RESULTS/tp2-ref/prof-\$w/breakdown-rank0.json \
> $RESULTS/tp2-ref/prof-\$w/breakdown-rank0.txt 2>&1; done; \
for i in 640 320; do timeout 1800 docker run --rm --gpus all --ipc host -e CUTE_DSL_ARCH=sm_121a -v $REPO/scripts:/s:ro \
--entrypoint python3 $IMAGE /s/bench_moe_pad.py --intermediate \$i > $RESULTS/moe-pad-I\$i.txt 2>&1; done; \
python3 $REPO/scripts/r8_profile_table.py $RESULTS --tp2 $RESULTS/tp2-ref > $RESULTS/profile_table.md 2>&1"
exec bash "$REPO/scripts/launch_r8_tp1.sh" "$WAIT"
