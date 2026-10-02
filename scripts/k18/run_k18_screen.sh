#!/usr/bin/env bash
# k18 (opus-kernel-18): TP=1 A16 MoE screening, after the k17 night chain (r9-pin) and only if the
# k18 A16 boot test passed (boot_check.py: acceptance per position >= 0.9x tp1-a16off in every
# temperature-0 cell). Same image for all arms (k18), control tp1-a16off = shipped v2 + knob off.
#   dgx-01: temperature 0, passes 1-2 ABBA; dgx-02: temperature 1.0, passes 3-4 ABBA (benchy coding is T=1.0)
# Run: setsid nohup bash ~/GEN-AI/k18/run_k18_screen.sh > ~/GEN-AI/k18/run_k18_screen.nohup 2>&1 < /dev/null &
set -u
K17=$HOME/GEN-AI/k17; K=$HOME/GEN-AI/k18
R=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
B=$R/results/k18-tp1-prof-a16-20261002
export PATH=$HOME/.local/bin:$PATH
log() { echo "[$(date '+%F %T %Z')] $*" >> "$K/k18.log"; }
log "screen: waiting for night chain stage 3 and k18 boot test"
until grep -q "stage 3 exit=" "$K17/night_chain.log" && grep -q "^\[.*\] done$" "$K/k18.log"; do sleep 60; done
if ! python3 "$K/boot_check.py" "$B/screen/dgx01/tp1-a16off-p1" "$B/screen/dgx02/tp1-a16-40-p1" > "$K/boot_check.txt" 2>&1; then
  log "screen: SKIPPED, A16 boot test failed: $(tail -1 "$K/boot_check.txt")"; exit 1; fi
log "screen: boot test $(tail -1 "$K/boot_check.txt")"
exec 9>"$HOME/GEN-AI/gpu-lock"; flock 9; log "screen: gpu-lock held"
export RESULTS=$R/results/k18-tp1-a16-screen-20261003
export SHIPPED_IMAGE_EXPECT=b1.4-20261001-b7fbaf96-a7e649d8-warm
export CONTROL_ARM=tp1-a16off CANARY_ARMS="tp1-a16-24 tp1-a16-40"
export SEQ_H1="tp1-a16off:1 tp1-a16-24:1 tp1-a16-40:1 tp1-a16-40:2 tp1-a16-24:2 tp1-a16off:2"
export SEQ_H2="tp1-a16-24:3:1.0 tp1-a16-40:3:1.0 tp1-a16off:3:1.0 tp1-a16off:4:1.0 tp1-a16-40:4:1.0 tp1-a16-24:4:1.0"
timeout 25200 bash "$R/scripts/k18_tp1_driver.sh" >> "$K/k18.log" 2>&1
log "screen: driver exit=$? state=$(cat "$RESULTS/STATE")"
