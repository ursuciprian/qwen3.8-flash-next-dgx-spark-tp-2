#!/usr/bin/env bash
# r9-dense TP=2 screen (opus-kernel-17, 2026-10-02): waits for the r9_dense_check STATE and starts only if
# that check is sane: STATE DONE (not CANARY/FAILED), fidelity 20/20 at 32k and 128k, and dense acceptance
# per position within 0.03 of base on fresh-c1 and d16k-c1 (T=0, single boot). Arms on the b1.4 image:
# r8-off (registry recipe) vs r9-dense (same recipe, f400 snapshot), ABBA 2 passes at T=0 + 2 at T=1.0.
set -u
WAIT=${1:?usage: launch_r9_dense.sh <r9_dense_check STATE>}
: "${SHIPPED_IMAGE_EXPECT:?}"
export SHIPPED_IMAGE_EXPECT PATH="$HOME/.local/bin:$PATH"
REPO=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2; AD=$HOME/GEN-AI/r4-recipes/archive/recipes/qwen3.8-flash-next
CHK=$(dirname "$WAIT")
export RESULTS=${RESULTS:-$REPO/results/r9-dense-screen-$(date +%Y%m%d-%H%M)}
export CONTROL_ARM=r8-off CANARY_ARMS="r9-dense"
STAGES=${STAGES:-"screen:r8-off:1 screen:r9-dense:1 screen:r9-dense:2 screen:r8-off:2 screen:r8-off:3:1.0 screen:r9-dense:3:1.0 screen:r9-dense:4:1.0 screen:r8-off:4:1.0"}
mkdir -p "$RESULTS"; L=$RESULTS/launch.log
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$L"; }
echo waiting > "$RESULTS/STATE"; log "waiting for $WAIT"
until grep -qE '^(DONE|FAILED|SKIPPED)' "$WAIT" 2>/dev/null; do sleep 60; done
log "wait over: $(cat "$WAIT")"
sane=$(python3 - "$CHK" <<'PY'
import json, re, sys
d = sys.argv[1]
def acc(sub, tag):
    def rd(f):
        v = {}
        for l in open(f):
            m = re.match(r'vllm:spec_decode_num_(accepted|draft)_tokens_per_pos_total\{.*position="(\d)"\} ([0-9.e+]+)', l)
            if m: v[(m.group(1), m.group(2))] = float(m.group(3))
        return v
    a, b = rd(f"{d}/{sub}/pos-{tag}.before"), rd(f"{d}/{sub}/pos-{tag}.after")
    g = lambda k: b.get(k, 0) - a.get(k, 0); dr = g(("draft", "0")) or 1
    return [g(("accepted", str(i))) / dr for i in range(4)]
try:
    if open(f"{d}/STATE").read().strip() != "DONE": print("no: state"); sys.exit()
    txt = open(f"{d}/dense/fidelity.txt").read()
    if len(re.findall(r"20/20", txt)) < 2: print("no: fidelity"); sys.exit()
    for t in ("fresh-c1", "d16k-c1"):
        if any(abs(x - y) > 0.03 for x, y in zip(acc("base", t), acc("dense", t))): print(f"no: acceptance {t}"); sys.exit()
    print("yes")
except Exception as e:
    print(f"no: {e}")
PY
)
log "dense check sane: $sane"
[ "$sane" = yes ] || { echo "SKIPPED: dense check $sane" > "$RESULTS/STATE"; exit 0; }
BASE=$(find "$HOME/.cache/sparkrun/registries" -path '*recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml' | head -1)
grep -q "^container: .*$SHIPPED_IMAGE_EXPECT" "$BASE" || { log "registry recipe not b1.4"; echo "FAILED: base" > "$RESULTS/STATE"; exit 1; }
cp "$BASE" "$RESULTS/base-recipe.yaml"
python3 "$REPO/scripts/r8_make_recipes.py" tp2 --base "$BASE" --out "$AD" --only r8-off,r9-dense >> "$L" 2>&1 \
  || { log "recipe generation failed"; echo "FAILED: recipes" > "$RESULTS/STATE"; exit 1; }
for a in r8-off r9-dense; do sparkrun recipe validate "$AD/qwen3.8-flash-next-2x-dgx-spark-$a.yaml" >> "$L" 2>&1 \
  || { log "validate failed: $a"; echo "FAILED: validate" > "$RESULTS/STATE"; exit 1; }; done
log "taking gpu-lock"; exec 9>"$HOME/GEN-AI/gpu-lock"; flock 9; log "gpu-lock held"
export STAGES; log "start driver: STAGES=$STAGES"
cd "$REPO" && exec bash scripts/r8_driver.sh
