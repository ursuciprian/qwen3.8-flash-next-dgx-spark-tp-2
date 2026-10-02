#!/usr/bin/env bash
# k18 (opus-kernel-18, 2026-10-02) on dgx-01, inserted into the k17 night chain between stage 2
# (TP=1 v2 vs gdnmse) and stage 3 (r9-pin): ~/GEN-AI/k17/go-pin is held as a directory, so
# launch_r9.sh keeps waiting until this job writes DONE into it (EXIT trap, also on failure).
#   dgx-01: TP=1 v2 profile (fresh c1/c4/c8, 16k c4), then tp1-a16off pass 1 (A16 control)
#   dgx-02: GPU tests + MoE microbench (pre_gpu.sh), then tp1-a16-40 pass 1 = A16 boot test
# Run: setsid nohup bash ~/GEN-AI/k18/run_k18.sh > ~/GEN-AI/k18/run_k18.nohup 2>&1 < /dev/null &
set -u
K17=$HOME/GEN-AI/k17; K=$HOME/GEN-AI/k18
R=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
export PATH=$HOME/.local/bin:$PATH
log() { echo "[$(date '+%F %T %Z')] $*" >> "$K/k18.log"; }
release() { [ -d "$K17/go-pin" ] && rmdir "$K17/go-pin"; echo DONE > "$K17/go-pin"; log "go-pin released (r9-pin may start)"; }
trap release EXIT
log "waiting for night chain stage 2 to exit"
until grep -q "stage 2 exit=" "$K17/night_chain.log"; do sleep 60; done
log "stage 2 over: $(grep 'stage 2 exit=' "$K17/night_chain.log" | tail -1)"
exec 9>"$HOME/GEN-AI/gpu-lock"; flock 9; log "gpu-lock held"
export RESULTS=$R/results/k18-tp1-prof-a16-20261002
export SHIPPED_IMAGE_EXPECT=b1.4-20261001-b7fbaf96-a7e649d8-warm
export CONTROL_ARM=tp1-a16off CANARY_ARMS=tp1-a16-40
export SEQ_H1="prof:tp1-v2-prof tp1-a16off:1" SEQ_H2="tp1-a16-40:1"
export PRE_H2="bash $K/pre_gpu.sh $K/out-pre"
export PROF_WINDOWS="fresh-c1:256:1:1 fresh-c4:256:4:3 fresh-c8:256:8:5 d16k-c4:16384:4:6"
timeout 16200 bash "$R/scripts/k18_tp1_driver.sh" >> "$K/k18.log" 2>&1
log "driver exit=$? state=$(cat "$RESULTS/STATE")"
scp -q -r 192.168.100.53:GEN-AI/k18/out-pre "$RESULTS/" 2>>"$K/k18.log"
python3 "$R/scripts/r8_profile_table.py" "$RESULTS" > "$RESULTS/profile_table.md" 2>&1
log "done"
