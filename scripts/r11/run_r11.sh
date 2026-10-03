#!/usr/bin/env bash
# r11 TP=1 host-gap screen (opus-gaps, 2026-10-03). k18 profile: one ~10.5 ms host gap per decode
# step at c4/c8 (PLE ids sync -> page-cache gather -> _nvfp4_lookup_kernel), 0.4-1.5 ms at c1.
# Cause: NVMe APST; the drive sleeps when a step is longer than the 100 ms primary timeout.
# Arm tp1-keepalive (VLLM_PLE_MMAP_KEEPALIVE_MS=50, vLLM exp/r11-tp1-gaps 1e41784f) vs tp1-v2c.
#   0) waits for "r9-pin exit=" in ~/GEN-AI/k17/r9pin_late.log, then takes ~/GEN-AI/gpu-lock
#   1) PRE (servers down): NVMe O_DIRECT read latency vs idle gap on each node (pre-dgx0N.log)
#   2) screen, all T=0, ABBA: dgx-01 passes 1-2, dgx-02 passes 3-4; logits a/b after dgx-01 pass 1
#   3) POST: logits diffs (self vs cross), r3_screen_report over merged/
# Driver EXIT restores the shipped 2x recipe; this script retries the restore if STATE != DONE.
# Run: setsid nohup bash ~/GEN-AI/r11/run_r11.sh > ~/GEN-AI/r11/run_r11.nohup 2>&1 < /dev/null &
set -u
K=$HOME/GEN-AI/r11; K17=$HOME/GEN-AI/k17
R=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
export PATH=$HOME/.local/bin:$PATH
export RESULTS=$R/results/r11-tp1-gaps-20261003
IMG=spark-vllm-b12x:r11-5bf24021-1e41784f-warm
SNAP=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
log() { echo "[$(date '+%F %T %Z')] r11: $*" >> "$K/r11.log"; }
mkdir -p "$RESULTS"; echo "queued: waiting for r9-pin exit" > "$RESULTS/STATE"
log "waiting for r9-pin exit= in $K17/r9pin_late.log"
until grep -q "r9-pin exit=" "$K17/r9pin_late.log" 2>/dev/null; do sleep 120; done
exec 9>"$HOME/GEN-AI/gpu-lock"; flock 9; log "gpu-lock held"
for h in 192.168.100.62 192.168.100.53; do
  ssh -n "$h" "docker image inspect $IMG >/dev/null" || { log "image $IMG missing on $h"; echo "FAILED: image missing on $h" > "$RESULTS/STATE"; exit 1; }; done
python3 "$K/r8_make_recipes.py" tp1v2 --base "$K17/qwen3.8-flash-next-1x-dgx-spark.yaml" --image "$IMG" \
  --only tp1-v2c,tp1-keepalive --out "$HOME/GEN-AI/r4-recipes/archive/recipes/qwen3.8-flash-next" >> "$K/r11.log" 2>&1 \
  || { log "recipe generation failed"; echo "FAILED: recipes" > "$RESULTS/STATE"; exit 1; }
F=$(readlink -f "$(ls "$SNAP"/*.safetensors | head -1)")
PRE="timeout 300 python3 $R/scripts/nvme_idle_latency.py $F || true"
POST="bash $K/post_r11.sh"
export SHIPPED_IMAGE_EXPECT=b1.4-20261001-b7fbaf96-a7e649d8-warm CONTROL_ARM=tp1-v2c CANARY_ARMS=tp1-keepalive
export PRE_H1="$PRE" PRE_H2="$PRE" POST
export SEQ_H1="tp1-v2c:1 tp1-keepalive:1 tp1-keepalive:2 tp1-v2c:2"
export SEQ_H2="tp1-keepalive:3 tp1-v2c:3 tp1-v2c:4 tp1-keepalive:4"
timeout -k 4500 25200 bash "$R/scripts/r11_tp1_driver.sh" >> "$K/r11.log" 2>&1
rc=$?; log "driver exit=$rc state=$(cat "$RESULTS/STATE")"
if ! grep -q "^DONE" "$RESULTS/STATE"; then
  log "STATE not DONE: restoring the shipped 2x recipe"
  grep -q "^FAILED" "$RESULTS/STATE" || echo "FAILED: driver exit=$rc" > "$RESULTS/STATE"
  sparkrun stop --all >> "$K/r11.log" 2>&1
  timeout 3600 sparkrun run qwen3.8-flash-next-2x-dgx-spark --no-follow >> "$K/r11.log" 2>&1
  s=$(date +%s); until [ "$(curl -s -m5 -o /dev/null -w '%{http_code}' localhost:8000/health)" = 200 ] \
    || [ $(( $(date +%s)-s )) -ge 3600 ]; do sleep 15; done
  log "fallback restore health=$(curl -s -m5 -o /dev/null -w '%{http_code}' localhost:8000/health)"
fi
log "exit"
