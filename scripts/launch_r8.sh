#!/usr/bin/env bash
# r8 TP=2 screening launcher (opus-kernel-11, 2026-10-01). Run detached on dgx-01:
#   SHIPPED_IMAGE_EXPECT=<b1.4 tag> setsid nohup bash scripts/launch_r8.sh <STATE file> > <log> 2>&1 &
# Waits until <STATE file> reads DONE (FAILED*: exits, pair untouched), then generates the r8-*
# recipes from the promoted registry recipe (must serve SHIPPED_IMAGE_EXPECT; r8-block is not
# generated when b1.4 already uses block verification), validates them, drops stages whose
# recipe is missing and runs scripts/r8_driver.sh (ABBA vs r8-off, canary on CANARY_ARMS).
# Default STAGES: fp4scale and per-batch d3, ABBA at temp 0 and again at temp 1.0 (12 boots).
# Generated but not scheduled: r8-cg (no-op on uniform decode: FULL graphs already run
# round_up(capture, 5) = exact 5/10/20 at c1/c2/c4), r8-d3g (global 3 drafts, c1 loss
# expected), r8-block (r7-block was rejected on 2026-10-01), r8-prof (prepend prof:r8-prof
# for the b1.4 eager-tail re-profile, docs/r8-eager.md).
set -u
WAIT=${1:?usage: launch_r8.sh <STATE file to wait for>}
: "${SHIPPED_IMAGE_EXPECT:?set to the tag the registry recipe serves (b1.4)}"
export SHIPPED_IMAGE_EXPECT PATH="$HOME/.local/bin:$PATH"
REPO=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
RECIPES=$HOME/GEN-AI/r4-recipes; AD=$RECIPES/archive/recipes/qwen3.8-flash-next
export RESULTS=${RESULTS:-$REPO/results/r8-screen-$(date +%Y%m%d-%H%M)}
STAGES=${STAGES:-"screen:r8-off:1 screen:r8-fp4scale:1 screen:r8-d3:1 screen:r8-d3:2 screen:r8-fp4scale:2 screen:r8-off:2 \
screen:r8-off:3:1.0 screen:r8-d3:3:1.0 screen:r8-fp4scale:3:1.0 screen:r8-fp4scale:4:1.0 screen:r8-d3:4:1.0 screen:r8-off:4:1.0"}
mkdir -p "$RESULTS"; L=$RESULTS/launch.log
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$L"; }
echo waiting > "$RESULTS/STATE"; log "waiting for $WAIT (DONE)"
until grep -qE '^(DONE|FAILED)' "$WAIT" 2>/dev/null; do sleep 60; done
grep -q '^DONE' "$WAIT" || { log "wait state: $(cat "$WAIT"): not starting"; echo "SKIPPED: $(head -1 "$WAIT")" > "$RESULTS/STATE"; exit 1; }
BASE=$(find "$HOME/.cache/sparkrun/registries" -path '*recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml' | head -1)
grep -q "^container: .*$SHIPPED_IMAGE_EXPECT" "$BASE" \
  || { log "registry recipe $BASE does not serve $SHIPPED_IMAGE_EXPECT (registry update pending?)"; echo "FAILED: base" > "$RESULTS/STATE"; exit 1; }
cp "$BASE" "$RESULTS/base-recipe.yaml"
python3 "$REPO/scripts/r8_make_recipes.py" tp2 --base "$BASE" --out "$AD" >> "$L" 2>&1 \
  || { log "recipe generation failed"; echo "FAILED: recipes" > "$RESULTS/STATE"; exit 1; }
KEEP=
for st in $STAGES; do
  IFS=: read -r kind arm _ <<< "$st"
  if [ "$kind" = screen ] && [ ! -f "$AD/qwen3.8-flash-next-2x-dgx-spark-$arm.yaml" ]; then log "no recipe for $arm: drop $st"; continue; fi
  KEEP="$KEEP $st"
done
for f in "$AD"/qwen3.8-flash-next-2x-dgx-spark-r8-*.yaml; do
  sparkrun recipe validate "$f" >> "$L" 2>&1 || { log "validate failed: $f"; echo "FAILED: validate" > "$RESULTS/STATE"; exit 1; }
done
export STAGES="${KEEP# }"
log "start driver: RESULTS=$RESULTS STAGES=$STAGES"
cd "$REPO" && exec bash scripts/r8_driver.sh
