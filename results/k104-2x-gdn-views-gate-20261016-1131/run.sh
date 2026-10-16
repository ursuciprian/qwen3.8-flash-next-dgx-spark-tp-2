#!/usr/bin/env bash
# k104-2x-gdn-views-gate (2026-10-15, tp-2): the release quality gate on the 2x ship candidate with k101 only
# (VLLM_GDN_PREFILL_VIEWS=1 on the 2x v2.1.0, TP=2 A/B WIN in k101s: cold TTFT -2.2 to -2.5%).
#   arm      the k101s 2x arm as it ran: local image spark-vllm-b12x:k101-gv-10d0acc8-d21d7ade (2x v2.1.0 image + the
#            six 10d0acc8 files, same Id on both Sparks), recipe = registry 2x v2.1.0 + own runtime cache
#            /cache/runtime/k101 (baked by k101s) + the knob (tp2.sh mkarm, regenerated here). No push: the gate boots
#            the local tag. Fail-closed: d21d7ade's nvfp4/b12x.py in the arm image is byte-identical to the shipped
#            image's, and /opt/k101-overlay names 10d0acc8.
#   gate     refuses (SKIPPED) unless k101s STATE says 2x WIN.
#   gate     one TP=2 arm boot (k92 gate, k61.gate rules): stragglers c5-c16 0 preemptions, tool-use hardmode >= 88,
#            TC-45 x5 = 100, fidelity 8k/32k/64k/128k + 128k seeds 11/13 all 20/20; the views line on both ranks.
#   not here the packaging check of a release image (warm build, fresh-user boot): it belongs to the release build.
# ETA ~1 h 45 min (boot ~10, gate ~90) + restore ~6 min.
# Contract (backlog/runner.sh): both GPU locks; exit 75 while busy; STATE DONE/SKIPPED/FAILED; comment.md + RESULT.
NEEDS="dgx01 dgx02"
set -u
J=k104; K=$HOME/GEN-AI/backlog/k104
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k104-2x-gdn-views-gate-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"   # x, hn, MODEL
IMG2=ghcr.io/ursuciprian/spark-vllm-b12x:k94-20261013-21e0b201-d21d7ade-warm   # 2x v2.1.0
ARM2=spark-vllm-b12x:k101-gv-10d0acc8-d21d7ade; VLLM_SHA=10d0acc85fb450472230558eda063c9059c20234
REV=03f4a0570496bbe741189f4a2a7d611679e85406
CRC=$HOME/.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923
ARC=$CRC/k101; ACACHE=/cache/runtime/k101; K70=$BL/k70
REGS=$HOME/.cache/sparkrun/registries; REG2=qwen38-flashnext; N2=qwen3.8-flash-next-2x-dgx-spark; RD=$K/rec
B12XPY=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/nvfp4/b12x.py
VIEWS='GDN prefill views: in-order'
PRIV=$(cat "$BL/k98/private.pat" 2>/dev/null); [ -n "$PRIV" ] || { echo "missing $BL/k98/private.pat" >&2; exit 2; }
on2() { ssh -n -o ConnectTimeout=10 $H2 "$@"; }
hx() { if [ "$1" = localhost ]; then bash -c "$2"; else on2 "$2"; fi; }
img() { sed -n 's/^container: *//p' "$1" | head -1 | tr -d "\"' "; }
fld() { sed -n "s/^$2: *//p" "$1" | head -1 | tr -d "\"' "; }
mkarm() { # ctl.yaml arm.yaml: own name, image, runtime cache, knob; 2 lines changed, 4 added (k101s/tp2.sh)
  python3 - "$1" "$2" "$ACACHE" "$ARM2" <<'PY'
import sys
from collections import Counter
src, dst, cache, image = sys.argv[1:5]
lines = open(src).read().split("\n")
def one(pred, what):
    i = [k for k, l in enumerate(lines) if pred(l)]
    if len(i) != 1:
        sys.exit(f"mkarm: need exactly one {what}, found {len(i)}")
    return i[0]
n, c, e = one(lambda l: l.startswith("name: "), "name line"), one(lambda l: l.startswith("container: "), "container line"), one(lambda l: l == "env:", "env: block")
if any(l.strip().startswith(("XDG_CACHE_HOME:", "VLLM_CACHE_ROOT:", "VLLM_GDN_PREFILL_VIEWS")) for l in lines):
    sys.exit("mkarm: the control already sets one of the arm's variables")
add = ["  # k101 arm: GDN prefill views, own runtime cache", f'  XDG_CACHE_HOME: "{cache}"',
       f'  VLLM_CACHE_ROOT: "{cache}/vllm"', '  VLLM_GDN_PREFILL_VIEWS: "1"']
out = lines[:]
out[n] = "name: qwen3.8-flash-next-2x-k101-gv"
out[c] = f"container: {image}"
out[e + 1:e + 1] = add
a, b = Counter(lines), Counter(out)
if sum((a - b).values()) != 2 or sum((b - a).values()) != 2 + len(add):
    sys.exit("mkarm: unexpected diff")
open(dst, "w").write("\n".join(out))
PY
}
mkrecipes() {
  local f2; mkdir -p "$RD"
  f2=$(grep -rlx --include='*.yaml' "name: $N2" "$(readlink -f "$REGS/$REG2")/recipes" 2>/dev/null | head -1)
  [ -s "$f2" ] || { echo "2x recipe not resolvable from the registry ($REG2/$N2)"; return 1; }
  cp "$f2" "$RD/ctl-2x.yaml"
  [ "$(img "$RD/ctl-2x.yaml")" = "$IMG2" ] || { echo "registry 2x image is not $IMG2"; return 1; }
  [ "$(fld "$RD/ctl-2x.yaml" model_revision)" = "$REV" ] || { echo "registry 2x revision is not $REV"; return 1; }
  sed -n 1p "$RD/ctl-2x.yaml" | grep -q '^# Release: v2.1.0' || { echo "registry 2x recipe is not v2.1.0"; return 1; }
  mkarm "$RD/ctl-2x.yaml" "$RD/arm-2x.yaml"; }
checks() {
  local ok=0 f t h id=
  for f in "$K70/k61.py" "$R/scripts/straggler_probe.py" "$R/scripts/fidelity_probe.py"; do [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  for t in tool-eval-bench sparkrun python3; do command -v $t > /dev/null || { echo "missing $t"; ok=1; }; done
  mkrecipes || ok=1
  for h in localhost $H2; do
    hx $h "test -d $ARC/vllm && test -d $ARC/b12x/compile/preparation" || { echo "$h: arm cache $ARC (k101s bake) missing"; ok=1; }
    t=$(hx $h "docker image inspect -f '{{.Id}}' $ARM2" 2>/dev/null) || { echo "$h: $ARM2 missing"; ok=1; continue; }
    [ -z "$id" ] && id=$t; [ "$t" = "$id" ] || { echo "$ARM2 Id differs between the Sparks"; ok=1; }
    [ "$(hx $h "docker run --rm --network none --entrypoint sha256sum $IMG2 $B12XPY" | cut -d' ' -f1)" = \
      "$(hx $h "docker run --rm --network none --entrypoint sha256sum $ARM2 $B12XPY" | cut -d' ' -f1)" ] \
      || { echo "$h: d21d7ade's nvfp4/b12x.py differs between $IMG2 and $ARM2"; ok=1; }
    hx $h "docker run --rm --network none --entrypoint cat $ARM2 /opt/k101-overlay" 2>/dev/null | grep -q "^$VLLM_SHA " \
      || { echo "$h: $ARM2 /opt/k101-overlay does not name $VLLM_SHA"; ok=1; }; done
  [ -s "$RD/arm-2x.yaml" ] && { echo "arm diff:"; diff "$RD/ctl-2x.yaml" "$RD/arm-2x.yaml" | sed 's/^/   /'; }
  return $ok; }
win() { head -1 "$BL/k101s/STATE" 2>/dev/null | grep -q '^DONE: k101 RULE=WIN.*; 2x WIN'; }
if [ "${1:-}" = --dry-run ]; then checks; rc=$?
  win && echo "gate: k101s 2x WIN" || { echo "k101s is not DONE with 2x WIN: the job would end SKIPPED"; }
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"; echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
win || { echo "SKIPPED: k104, k101s is not DONE with 2x WIN ($(head -1 "$BL/k101s/STATE" | cut -c1-80))" > "$K/STATE"; exit 0; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
timeout 300 sparkrun registry update $REG2 >> "$RES/$J.log" 2>&1 < /dev/null 9>&-
checks > "$RES/checks.txt" 2>&1 || { FINAL="FAILED: k104 checks ($RES/checks.txt)"; exit 1; }
cp "$K/run.sh" "$RD"/*.yaml "$RES/"
keep_logs2() { # dir (before stop_all), k99
  local c; c=$(n0); [ -n "$c" ] || c=$(docker ps -a --format '{{.Names}}' | grep node_0 | head -1); [ -n "$c" ] || return 0
  docker exec "$c" cat /tmp/sparkrun_serve.log > "$1/serve.log" 2>&1 || docker logs "$c" > "$1/serve.log" 2>&1
  on2 'c=$(docker ps -a --format "{{.Names}}" | grep node_1 | head -1); [ -n "$c" ] && { docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1; }' > "$1/serve-node1.log" 2>&1
  docker exec "$c" sh -c "env | grep -E '^(VLLM|B12X|XDG)' | sort" > "$1/env.txt" 2>&1; }
tboot() { # recipe dir -> 0 up (cold allowed, cap 5400 s), 1 failed (k101s/tp2.sh)
  local rec=$1 d=$2 s m n
  stop_all; cp "$rec" "$d/recipe.yaml"
  ( cd "$(dirname "$rec")" && timeout -k 60 900 sparkrun run "$(basename "$rec")" --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  s=$(date +%s)
  while :; do
    n=$(n0)
    m=$([ -n "$n" ] && docker exec "$n" grep -m1 -oE 'Directly load AOT compilation|Dynamo bytecode transform time' /tmp/sparkrun_serve.log 2>/dev/null)
    [ "$(health localhost)" = 200 ] && { echo "boot $(( $(date +%s) - s ))s ${m:-no AOT line}; pong $(pong localhost)" > "$d/boot.txt"; return 0; }
    if [ $(( $(date +%s) - s )) -gt 180 ] && { [ -z "$n" ] || docker exec "$n" grep -qE 'Worker failed with error|EngineCore failed to start|Engine core initialization failed' /tmp/sparkrun_serve.log 2>/dev/null; }; then return 1; fi
    [ $(( $(date +%s) - s )) -ge 5400 ] && return 1
    sleep 10; done; }
measured() { grep -ohE '\b[0-9]+ measured' "$1" 2>/dev/null | awk '{s+=$1} END {print s+0}'; }
G=$RES/gate; mkdir -p "$G"
st "running: gate boot (2x arm, TP=2)"
tboot "$RD/arm-2x.yaml" "$G" || { keep_logs2 "$G"; stop_all; FINAL="FAILED: k104 gate boot ($G)"; exit 1; }
st "running: gate (stragglers, hardmode, TC-45, fidelity 8k-128k + 128k seeds 11/13)"
( cd "$R" && child timeout 900 python3 scripts/straggler_probe.py 5 6 7 8 12 16 > "$G/straggler.log" 2>&1 )
child timeout 7200 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://localhost:8000 --model $MODEL > "$G/hardmode.log" 2>&1
child timeout 3600 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://localhost:8000/v1 --model $MODEL --scenarios TC-45 --trials 5 > "$G/tc45.log" 2>&1
( cd "$R" && child timeout 5400 python3 scripts/fidelity_probe.py --base http://localhost:8000 --model $MODEL --depths 8000,32000,64000,128000 --out "$G/fidelity.json" > "$G/fidelity_probe.txt" 2>&1 )
for sd in 11 13; do ( cd "$R" && child timeout 2700 python3 scripts/fidelity_probe.py --base http://localhost:8000 --model $MODEL --depths 128000 --seed $sd --out "$G/fidelity-seed$sd.json" > "$G/fidelity-seed$sd.txt" 2>&1 ); done
keep_logs2 "$G"; stop_all
F=; grep -q "$VIEWS" "$G/serve.log" && grep -q "$VIEWS" "$G/serve-node1.log" || F="views line missing on a rank"
FACTS="views lines rank 0 $(grep -c "$VIEWS" "$G/serve.log") / rank 1 $(grep -c "$VIEWS" "$G/serve-node1.log"); rank 0 plans measured $(measured "$G/serve.log"); Traceback $(grep -c Traceback "$G/serve.log")/$(grep -c Traceback "$G/serve-node1.log"); $(cat "$G/boot.txt")"
t=$(mktemp -d); ln -s "$G" "$t/cand1"; ln -s "$G" "$t/cand2"
GV=$(python3 -c "import sys; sys.path.insert(0, sys.argv[1]); import k61; b, hm = k61.gate(sys.argv[2]); print('PASS' if not b else 'FAIL (' + '; '.join(b) + ')')" "$K70" "$t" 2>&1); rm -rf "$t"
{ echo "k104 $(TZ=Europe/Bucharest date '+%F %T %Z'): release quality gate on the 2x v2.1.0 + VLLM_GDN_PREFILL_VIEWS=1 ($ARM2), one TP=2 boot"
  echo "GATE=$GV"; echo "$FACTS"; grep -hE "Quality:" "$G/hardmode.log"; grep -hE "Pass\^5|Score:" "$G/tc45.log" | head -2
  grep -h "^depth" "$G/fidelity_probe.txt" "$G"/fidelity-seed1?.txt; tail -8 "$G/straggler.log"; } > "$RES/k104.txt" 2>&1
mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
{ echo "k104 result: I ran the release quality gate of v2.1.0 on the 2x with GDN prefill views (\`VLLM_GDN_PREFILL_VIEWS=1\`, vLLM \`${VLLM_SHA:0:10}\` files on the v2.1.0 image), one TP=2 boot. Its TP=2 A/B against v2.1.0 already passed (cold TTFT 2.2 to 2.5% lower)."
  echo; echo "Gate: \`${GV}\`.${F:+ $F.}"
  echo; echo '```'; head -c 30000 "$RES/k104.txt"; echo '```'
  echo; echo "Before a release the image still needs the warm build and a fresh-user boot check. Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"
if grep -qiE "$PRIV" "$K/comment.md" || grep -qE '192\.168\.|dgx-0[0-9]\.local' "$K/comment.md"; then
  mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k104 comment.md matched the private or LAN pattern (not posted; $RES)"; exit 1; fi
cp "$K/comment.md" "$RES/comment.md"
[ -n "$F" ] && { FINAL="FAILED: k104 $F; gate ${GV%% (*} ($RES/k104.txt)"; exit 1; }
FINAL="DONE: k104 gate=$GV ($RES/k104.txt)"
exit 0
