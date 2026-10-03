#!/usr/bin/env bash
# k18b (opus-kernel-18, 2026-10-03): after the k17 night chain (r9-pin), on the k18 image with the
# W4A16 exact direct-capacity fix (b12x 4e777f5e). The first k18 run skipped the A16 boot because
# the A16 GPU test failed at M=5 (direct launch compiled for m=8); fixed in ea7df8c8.
#   1) dgx-02: GPU tests + MoE microbench (pre_gpu.sh), then tp1-a16-40 pass 1 (boot test)
#      dgx-01: tp1-a16off pass 1, tp1-a16-24 pass 1 (control + second cutoff, canary vs control)
#   2) boot_check.py (acceptance per position >= 0.9x control, every T=0 cell); if it passes,
#      the screen: dgx-01 T=0 passes 1-2, dgx-02 T=1.0 passes 3-4, ABBA over the three arms.
# Run: setsid nohup bash ~/GEN-AI/k18/run_k18b.sh > ~/GEN-AI/k18/run_k18b.nohup 2>&1 < /dev/null &
set -u
K17=$HOME/GEN-AI/k17; K=$HOME/GEN-AI/k18
R=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
export PATH=$HOME/.local/bin:$PATH
log() { echo "[$(date '+%F %T %Z')] $*" >> "$K/k18.log"; }
log "k18b: waiting for night chain stage 3"
until grep -q "stage 3 exit=" "$K17/night_chain.log"; do sleep 60; done
exec 9>"$HOME/GEN-AI/gpu-lock"; flock 9; log "k18b: gpu-lock held"
export SHIPPED_IMAGE_EXPECT=b1.4-20261001-b7fbaf96-a7e649d8-warm CONTROL_ARM=tp1-a16off
B=$R/results/k18b-tp1-a16-boot-20261003
rm -rf "$K/out-pre"
RESULTS=$B CANARY_ARMS="tp1-a16-24" SEQ_H1="tp1-a16off:1 tp1-a16-24:1" SEQ_H2="tp1-a16-40:1" \
  PRE_H2="bash $K/pre_gpu.sh $K/out-pre" timeout 14400 bash "$R/scripts/k18_tp1_driver.sh" >> "$K/k18.log" 2>&1
log "k18b: boot driver exit=$? state=$(cat "$B/STATE")"
scp -q -r 192.168.100.53:GEN-AI/k18/out-pre "$B/" 2>>"$K/k18.log"
if ! python3 "$K/boot_check.py" "$B/screen/dgx01/tp1-a16off-p1" "$B/screen/dgx02/tp1-a16-40-p1" > "$B/boot_check.txt" 2>&1; then
  log "k18b: screen SKIPPED, A16 boot test failed: $(tail -1 "$B/boot_check.txt")"; exit 1; fi
log "k18b: boot test $(tail -1 "$B/boot_check.txt"); screening"
export RESULTS=$R/results/k18-tp1-a16-screen-20261003 CANARY_ARMS="tp1-a16-24 tp1-a16-40"
export SEQ_H1="tp1-a16off:1 tp1-a16-24:1 tp1-a16-40:1 tp1-a16-40:2 tp1-a16-24:2 tp1-a16off:2"
export SEQ_H2="tp1-a16-24:3:1.0 tp1-a16-40:3:1.0 tp1-a16off:3:1.0 tp1-a16off:4:1.0 tp1-a16-40:4:1.0 tp1-a16-24:4:1.0"
timeout 25200 bash "$R/scripts/k18_tp1_driver.sh" >> "$K/k18.log" 2>&1
log "k18b: screen driver exit=$? state=$(cat "$RESULTS/STATE")"
