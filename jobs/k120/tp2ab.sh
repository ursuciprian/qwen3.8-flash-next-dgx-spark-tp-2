#!/usr/bin/env bash
# 2x TP=2 A/B harness for k120 (2026-11-14): copy of k107fix/tp2ab.sh (itself a copy of k100b/tp2ab.sh); changes:
# llama-benchy also runs tg512 at c16 (the rule covers c1..c16, RULE.md), its timeout 1800 -> 2400 s, the prebake
# boot must answer the pong probe (from k114/tp2ab.sh), and (review 2026-11-14):
#   - the arm recipe drops the torch AOT seed line of the control recipe (in the arm image /opt/b12x-seed holds the
#     control's AOT artifact), so the arm compiles its own AOT in the bake boots; every boot logs the AOT line of
#     both ranks and a warm boot that compiles on either rank counts as cold
#   - the prebake boot runs mods/b12x-a16-marker: both ranks must log a BF16-path call on the HC down-projection
#     shape (n <= 512, k >= 8192, >= 1264 rows) during the cold 16K prompt, else PATH=NOT-ENGAGED, RULE2=EMPTY and no
#     screen boot runs
#   - identity across boots judges short prompts only ($K/ident.py); plans measured are counted on both ranks
#   - when the screen leaves a ship or flat outcome open (RULE2 WIN/FLAT, IDENTC OK, IDENT PASS/UNSTABLE) one more arm
#     boot runs the release quality gate of k104 (k61.gate: stragglers, hardmode, TC-45, fidelity) -> GATE=
# The arm comes from arm.conf (J and K are set by run.sh; K is the frozen job copy $RES/job during a run).
#   control  the 2x v2.2.0 release recipe $BL/k104s/recipe.yaml as k104s built and checked it (code sha256), image $IMG2
#            named explicitly (digest bba1374d, same Id on both Sparks). Never the registry.
#   arm      control + arm.conf: own name, ARM_IMG when set, the ARM_ENV lines, a runtime cache of its own under
#            /cache/runtime/$ARM_NAME (XDG_CACHE_HOME there; VLLM_CACHE_ROOT a fresh $ARM_NAME/vllm) and no AOT seed
#   stage    arm cache on both Sparks: no-clobber copies of the control's b12x/triton/... dirs (plan selections and
#            compiled kernels), never a vLLM compile root
#   bake     prebake with a 6 GiB KV pool + the marker mod (<= 3 tries, cold compile allowed) + pong + one cold 16K
#            prompt + marker check; then one full-pool boot (cold allowed, compiles the arm's AOT)
#   screen   ctl arm arm ctl ctl arm arm ctl (4 boots per side), warm only: warm-up 16K+64K, cold TTFT 16K/64K/128K x3,
#            fresh-c8 T=0 acceptance per draft position, fresh-c1 T=0 probe (4 requests, paired with the same pass of the
#            other side), llama-benchy pp2048 + tg512 c1/c4/c8/c16 x3. IDENTC=1 adds ident.py (c1, 8 prompts) and
#            identc.py (16 prompts at once, 3 rounds) before the speed cells.
#   judge    report2x.py: RULE2=WIN (a MAIN cell better beyond noise, nothing worse) | FLAT | LOSS | EMPTY
#   gate     see above (one arm boot, ~1 h 40 min)
# Writes $RES2/$J-tp2.txt (lines PONG= PATH= RULE2= IDENT= IDENTC= GATE=). Holds no lock of its own: run.sh holds both
# GPU locks.
# Usage: called by run.sh | J=<job> K=<job dir> bash tp2ab.sh --dry-run
set -u
: "${J:?set J}" "${K:?set K}"
source "$HOME/GEN-AI/backlog/lib.sh"
source "$K/arm.conf"   # ARM_NAME ARM_IMG ARM_ENV MAIN IDENTC
RES=${RES2:-${RES:-}}; [ -z "$RES" ] || mkdir -p "$RES"
IMG2=ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm   # 2x v2.2.0 (k104s)
IMG2_ID=sha256:9c05231a900dda1c301b410b90dfaf0611b1804ba4ed3d2f39c534f98e50a840
IMG2_DIGEST=sha256:bba1374d1026bf3eba13d68102a0001d66e8b39c35c582937094359cc91ed2e2
CTL_SRC=$BL/k104s/recipe.yaml
RECCODE=ef181a0eb043d4e08c73c86330049ade36c925553406e0560d30a3ba51c9d431   # k104s: sha256 of the recipe without comment lines
CTL_ROOT=/cache/runtime/vllm-2x-21e0b201-10d0acc8
REV=03f4a0570496bbe741189f4a2a7d611679e85406
CRC=$HOME/.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923   # the 2x's cache (k99)
ARC=$CRC/$ARM_NAME; ACACHE=/cache/runtime/$ARM_NAME; KV_PRE=6442450944
COLD=$BL/k101/cold.py; IDENT=$K/ident.py; CORPUS=$R/results/corpus-code.txt; BENCHY_SRC=$G/llama-benchy-fork; MODEL=qwen3.8-flash-next
K61=$BL/k61/k61.py; MARKMOD=$K/mods/b12x-a16-marker
SNAP=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
ORDER="ctl:1 arm:1 arm:2 ctl:2 ctl:3 arm:3 arm:4 ctl:4"; LENS=16384,65536,131072; REPS=3; RD=$RES/recipes
AOTRE='Directly load AOT compilation|Dynamo bytecode transform time'
on2() { ssh -n -o ConnectTimeout=10 $H2 "$@"; }
hx() { if [ "$1" = localhost ]; then bash -c "$2"; else on2 "$2"; fi; }
img() { sed -n 's/^container: *//p' "$1" | head -1 | tr -d "\"' "; }
fld() { sed -n "s/^$2: *//p" "$1" | head -1 | tr -d "\"' "; }

mkarm() { # ctl.yaml arm.yaml: own name, image, runtime cache, ARM_ENV, no AOT seed line; every edit counted
  python3 - "$1" "$2" "$ACACHE" "${ARM_IMG:-}" "$CTL_ROOT" "$J" "$ARM_ENV" <<'PY'
import sys
from collections import Counter
src, dst, cache, image, root, job, envs = sys.argv[1:8]
lines = open(src).read().split("\n")
def one(pred, what):
    i = [k for k, l in enumerate(lines) if pred(l)]
    if len(i) != 1:
        sys.exit(f"mkarm: need exactly one {what}, found {len(i)}")
    return i[0]
n = one(lambda l: l.startswith("name: "), "name line")
c = one(lambda l: l.startswith("container: "), "container line")
e = one(lambda l: l == "env:", "env: block")
r = one(lambda l: l == f'  VLLM_CACHE_ROOT: "{root}"', "control VLLM_CACHE_ROOT line")
s = one(lambda l: l.startswith("  A=/opt/b12x-seed/torch_compile_cache;"), "torch AOT seed line of the command")
pairs = [p.split("=", 1) for p in envs.split()]
keys = ["XDG_CACHE_HOME"] + [k for k, _ in pairs]
if any(l.strip().split(":", 1)[0] in keys for l in lines if l.startswith("  ") and ":" in l):
    sys.exit("mkarm: the control already sets one of the arm's variables")
out = lines[:]
out[n] = f"name: qwen3.8-flash-next-2x-{job}"
if image:
    out[c] = f"container: {image}"
out[r] = f'  VLLM_CACHE_ROOT: "{cache}/vllm"'
del out[s]   # s > e: the command block follows env, so the env insert below keeps its index
add = [f"  # {job} arm: own runtime cache, no AOT seed (compiles its own)" + ("" if not pairs else ", " + " ".join(envs.split())),
       f'  XDG_CACHE_HOME: "{cache}"'] + [f'  {k}: "{v}"' for k, v in pairs]
assert s > e
out[e + 1:e + 1] = add
a, b = Counter(lines), Counter(out)
changed = 2 + (1 if image else 0)
if sum((a - b).values()) != changed + 1 or sum((b - a).values()) != changed + len(add):
    sys.exit("mkarm: unexpected diff")
if any("/opt/b12x-seed/torch_compile_cache" in l for l in out if not l.lstrip().startswith("#")):
    sys.exit("mkarm: an AOT seed reference is left in the arm recipe")
open(dst, "w").write("\n".join(out))
PY
}
mkprebake() { # arm.yaml prebake.yaml: 6 GiB KV pool + the marker mod; 1 line changed, 3 added (k100b's sed edit, counted)
  python3 - "$1" "$2" "$KV_PRE" <<'PY'
import sys
from collections import Counter
src, dst, kv = sys.argv[1:4]
lines = open(src).read().split("\n")
g = [i for i, l in enumerate(lines) if l == "    --gpu-memory-utilization {gpu_memory_utilization}"]   # last serve line
rt = [i for i, l in enumerate(lines) if l == "runtime: vllm"]
if len(g) != 1 or len(rt) != 1 or any(l.startswith("mods:") for l in lines) or g[0] < rt[0]:
    sys.exit(f"mkprebake: need one gpu-memory-utilization line ({len(g)}), one runtime line ({len(rt)}) and no mods block")
out = lines[:]
out[g[0]] = lines[g[0]] + " \\"
out[g[0] + 1:g[0] + 1] = [f"    --kv-cache-memory-bytes {kv}"]
out[rt[0] + 1:rt[0] + 1] = ["mods:", "  - b12x-a16-marker"]
a, b = Counter(lines), Counter(out)
if sum((a - b).values()) != 1 or sum((b - a).values()) != 4:
    sys.exit("mkprebake: unexpected diff")
open(dst, "w").write("\n".join(out))
PY
}
recok() { [ "$(grep -vE '^[[:space:]]*#' "$CTL_SRC" | sha256sum | cut -c1-64)" = "$RECCODE" ] && [ "$(img "$CTL_SRC")" = "$IMG2" ] \
  && [ "$(fld "$CTL_SRC" model_revision)" = "$REV" ] && grep -qx "  VLLM_CACHE_ROOT: \"$CTL_ROOT\"" "$CTL_SRC"; }
mkrecipes() {
  mkdir -p "$RD"
  recok || { echo "control recipe $CTL_SRC is not the k104s v2.2.0 recipe (code sha256, image, revision, cache root)"; return 1; }
  cp "$CTL_SRC" "$RD/ctl-2x.yaml"
  mkarm "$RD/ctl-2x.yaml" "$RD/arm-2x.yaml" || return 1
  mkprebake "$RD/arm-2x.yaml" "$RD/prebake-2x.yaml" || return 1
  mkdir -p "$RD/mods" && rm -rf "$RD/mods/b12x-a16-marker" && cp -r "$MARKMOD" "$RD/mods/"; }   # sparkrun: <recipe dir>/mods/<name>
imgid() { hx $1 "docker image inspect -f '{{.Id}}' $2" 2>/dev/null; }
checks() {
  local ok=0 f t h
  for f in "$COLD" "$K/report2x.py" "$IDENT" "$CORPUS" "$R/scripts/depth_decode_probe.py" "$R/scripts/thunderdome_report.py" \
           "$R/scripts/paired_decode_ab.py" "$R/scripts/straggler_probe.py" "$R/scripts/fidelity_probe.py" "$K61" "$CTL_SRC" \
           "$MARKMOD/run.sh" "$MARKMOD/marker_snippet.py"; do [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  [ -d "$BENCHY_SRC" ] && [ -d "$SNAP" ] || { echo "missing llama-benchy fork or tokenizer snapshot"; ok=1; }
  for t in sparkrun uvx python3 curl tool-eval-bench; do command -v $t > /dev/null || { echo "missing $t"; ok=1; }; done
  bash -n "$MARKMOD/run.sh" || { echo "syntax: $MARKMOD/run.sh"; ok=1; }
  python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$MARKMOD/marker_snippet.py" || { echo "marker_snippet.py does not parse"; ok=1; }
  python3 -c "import sys; sys.path.insert(0, sys.argv[1]); import k61; assert callable(k61.gate)" "$(dirname "$K61")" || { echo "k61.gate not importable"; ok=1; }
  TD_SCRIPTS=$R/scripts python3 "$K/report2x.py" selftest > /dev/null || { echo "report2x.py selftest failed"; ok=1; }
  if [ "${IDENTC:-0}" = 1 ]; then
    python3 "$K/identc.py" selftest > /dev/null || { echo "identc.py selftest failed"; ok=1; }
    python3 "$IDENT" selftest > /dev/null || { echo "ident.py selftest failed"; ok=1; }; fi
  mkrecipes || ok=1
  for h in localhost $H2; do
    [ "$(imgid $h $IMG2)" = "$IMG2_ID" ] || { echo "$h: $IMG2 missing or not Id $IMG2_ID"; ok=1; }
    hx $h "test -d $CRC/b12x/compile/preparation && test -d $CRC/${CTL_ROOT#/cache/runtime/}" || { echo "$h: 2x v2.2.0 runtime cache under $CRC missing"; ok=1; }; done
  hx localhost "docker image inspect -f '{{join .RepoDigests \" \"}}' $IMG2" | grep -q "$IMG2_DIGEST" || { echo "$IMG2 digest is not $IMG2_DIGEST"; ok=1; }
  [ -s "$RD/arm-2x.yaml" ] && { echo "arm diff:"; diff "$RD/ctl-2x.yaml" "$RD/arm-2x.yaml" | sed 's/^/   /'; }
  [ -s "$RD/prebake-2x.yaml" ] && { echo "prebake diff (vs arm):"; diff "$RD/arm-2x.yaml" "$RD/prebake-2x.yaml" | sed 's/^/   /'; }
  return $ok; }
if [ "${1:-}" = --dry-run ]; then RES=$(mktemp -d); RD=$RES/recipes; checks; rc=$?; rm -rf "$RES"; echo "tp2ab dry-run exit=$rc"; exit $rc; fi
[ "${GPU_LOCK_HELD:-}" = 1 ] || { echo "refused: run.sh calls this under both GPU locks" >&2; exit 2; }
OUT=$RES/$J-tp2.txt; echo "$J 2x $(TZ=Europe/Bucharest date '+%F %T %Z'): arm ${ARM_IMG:-$IMG2} ${ARM_ENV:-} vs 2x v2.2.0 ($IMG2)" > "$OUT"
fail() { echo "$*" >> "$OUT"; echo "RULE2=FAILED" >> "$OUT"; stop_all; exit 1; }
checks > "$RES/checks.txt" 2>&1 || fail "checks failed ($RES/checks.txt)"
if [ -n "${ARM_IMG:-}" ]; then [ "$(imgid localhost "$ARM_IMG")" = "$(imgid $H2 "$ARM_IMG")" ] && [ -n "$(imgid localhost "$ARM_IMG")" ] \
  || fail "arm image $ARM_IMG missing or different on the two Sparks"; fi
stop_all
for h in localhost $H2; do
  hx $h "set -e; mkdir -p $ARC; for d in b12x triton inductor flashinfer cute_dsl flash_attn_cute_dsl torch_extensions numba; do
           [ -d $CRC/\$d ] && cp -an $CRC/\$d $ARC/; done; du -sh $ARC | cut -f1" > "$RES/stage-$h.txt" 2>&1 || fail "stage $h failed"; done

keep_logs2() { # dir (before stop_all), k99; plus both ranks' env and image
  local c; c=$(n0); [ -n "$c" ] || c=$(docker ps -a --format '{{.Names}}' | grep node_0 | head -1); [ -n "$c" ] || return 0
  docker exec "$c" cat /tmp/sparkrun_serve.log > "$1/serve.log" 2>&1 || docker logs "$c" > "$1/serve.log" 2>&1
  on2 'c=$(docker ps -a --format "{{.Names}}" | grep node_1 | head -1); [ -n "$c" ] && { docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1; }' > "$1/serve-node1.log" 2>&1
  docker exec "$c" sh -c "env | grep -E '^(VLLM|B12X|XDG|NCCL)' | sort" > "$1/env.txt" 2>&1
  on2 'c=$(docker ps --format "{{.Names}}" | grep node_1 | head -1); [ -n "$c" ] && docker exec $c sh -c "env | grep -E \"^(VLLM|B12X|XDG|NCCL)\" | sort"' > "$1/env-node1.txt" 2>&1
  { docker inspect -f '{{.Config.Image}} {{.Image}}' "$c"; on2 'c=$(docker ps -a --format "{{.Names}}" | grep node_1 | head -1); [ -n "$c" ] && docker inspect -f "{{.Config.Image}} {{.Image}}" $c'; } > "$1/image.txt" 2>&1; }
aot() { # host -> first AOT load / compile line of that rank's serve log (rank 0 = localhost, rank 1 = dgx-02)
  if [ "$1" = localhost ]; then local c; c=$(n0); [ -n "$c" ] && docker exec "$c" grep -m1 -E "$AOTRE" /tmp/sparkrun_serve.log 2>/dev/null
  else on2 "c=\$(docker ps --format '{{.Names}}' | grep node_1 | head -1); [ -n \"\$c\" ] && docker exec \$c grep -m1 -E '$AOTRE' /tmp/sparkrun_serve.log" 2>/dev/null; fi; }
tboot() { # recipe dir cold_ok (k99): 0 up, 1 failed, 2 compiled on a rank although warm was required
  local rec=$1 d=$2 s m0 m1 n cap; cap=$([ "$3" = 1 ] && echo 5400 || echo 1800)
  stop_all; cp "$rec" "$d/recipe.yaml"
  ( cd "$(dirname "$rec")" && timeout -k 60 900 sparkrun run "$(basename "$rec")" --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  s=$(date +%s)
  while :; do
    n=$(n0); m0=$(aot localhost); m1=$(aot $H2)
    if [ "$3" != 1 ] && grep -q 'Dynamo bytecode transform time' <<< "$m0$m1"; then
      echo "cold: rank0 ${m0:0:200} | rank1 ${m1:0:200}" > "$d/boot.txt"; return 2; fi
    [ "$(health localhost)" = 200 ] && { echo "boot $(( $(date +%s) - s ))s; AOT rank0: ${m0:0:200}; AOT rank1: ${m1:0:200}" > "$d/boot.txt"
      [ -n "$m0" ] && [ -n "$m1" ] || echo "note: no AOT line on a rank" >> "$d/boot.txt"; return 0; }
    if [ $(( $(date +%s) - s )) -gt 180 ] && { [ -z "$n" ] || docker exec "$n" grep -qE 'Worker failed with error|EngineCore failed to start|Engine core initialization failed' /tmp/sparkrun_serve.log 2>/dev/null; }; then return 1; fi
    [ $(( $(date +%s) - s )) -ge $cap ] && return 1
    sleep 10; done; }
pos() { curl -s -m 10 "localhost:8000/metrics" | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens(_per_pos)?_total'; }
measured() { grep -ohE '\b[0-9]+ measured' "$1" 2>/dev/null | awk '{s+=$1} END {print s+0}'; }
probe() { # dir tag conc reps offset timeout: one depth_decode_probe.py cell, T=0 (thunderdome cells)
  ( cd "$R" && timeout -k 30 "$6" python3 scripts/depth_decode_probe.py --base "http://$H1:8000" --label "$2" --corpus "$CORPUS" \
      --concurrency "$3" --repeats "$4" --temperature 0 --out "$1/probe-$2.json" --depth 256 --new 2048 --offset "$5" --max-tokens 512 \
      > "$1/probe-$2.log" 2>&1 ) || rm -f "$1/probe-$2.json"; }
d=; ok=
for t in 1 2 3; do d=$RES/bake/prebake-try$t; mkdir -p "$d"
  if tboot "$RD/prebake-2x.yaml" "$d" 1; then
    p=$(pong localhost); echo "pong: $p" >> "$d/boot.txt"   # k114: fail fast if the arm's numerics are broken
    echo "$p" | grep -qi pong || { keep_logs2 "$d"; echo "PONG=FAIL" >> "$OUT"; fail "prebake boot answered '$p' to the pong probe (arm numerics broken?)"; }
    echo "PONG=OK" >> "$OUT"
    timeout -k 30 600 python3 "$COLD" run --base http://localhost:8000 --corpus "$CORPUS" --lens 16384 --reps 1 --seed ${J}b --cskip 3600000 \
      --out "$d/bake16k.json" > "$d/bake16k.txt" 2>&1
    keep_logs2 "$d"; stop_all; ok=1; break; fi
  keep_logs2 "$d"; stop_all; done
[ -n "$ok" ] || fail "prebake: 3 tries failed ($RES/bake; a 'code-generation snapshot differs' line in serve.log means seeded plan records of another codegen)"
python3 "$K/report2x.py" marker "$d/serve.log" "$d/serve-node1.log" > "$RES/marker.txt" 2>&1
PATHV=$(sed -n 's/^PATH=//p' "$RES/marker.txt" | tail -1); PATHV=${PATHV:-NOT-ENGAGED}
{ echo "BF16 path marker (prebake, cold 16K, mods/b12x-a16-marker):"; grep -v '^PATH=' "$RES/marker.txt"; echo "PATH=$PATHV"; } >> "$OUT"
[ "$PATHV" = ENGAGED ] || { echo "the BF16 path did not run on both ranks: no screen boot (RULE.md 1)" >> "$OUT"; echo "RULE2=EMPTY" >> "$OUT"; stop_all; exit 0; }
d=$RES/bake/arm-full; mkdir -p "$d"
tboot "$RD/arm-2x.yaml" "$d" 1 || { keep_logs2 "$d"; fail "bake full-pool boot failed"; }
keep_logs2 "$d"; stop_all
echo "bake: prebake $(cat "$RES"/bake/prebake-try*/boot.txt 2>/dev/null | tr '\n' ' ' | tail -c 600); full $(cat "$d/boot.txt" | tr '\n' ' '); plans measured rank 0 $(measured "$d/serve.log"), rank 1 $(measured "$d/serve-node1.log")" >> "$OUT"

S=$RES/2x; mkdir -p "$S"
for item in $ORDER; do who=${item%:*}; k=${item#*:}; d=$S/$who-p$k; mkdir -p "$d"
  rec=$RD/ctl-2x.yaml; [ $who = arm ] && rec=$RD/arm-2x.yaml
  tboot "$rec" "$d" 0; rc=$?
  if [ $rc -ne 0 ]; then keep_logs2 "$d"; [ $rc -eq 2 ] && echo cold > "$d/COLD" || echo failed > "$d/FAILED"; stop_all; continue; fi
  ( timeout -k 30 900 python3 "$COLD" run --base http://localhost:8000 --corpus "$CORPUS" --lens 16384,65536 --reps 1 --seed ${J}w --cskip 3000000 --out "$d/warm.json" > "$d/warm.txt" 2>&1 )
  if [ "${IDENTC:-0}" = 1 ]; then mkdir -p "$d/hook"
    ( timeout -k 30 900 python3 "$IDENT" run --base http://localhost:8000 --corpus "$CORPUS" --out "$d/hook/ident.json" > "$d/hook/ident.txt" 2>&1 )
    ( timeout -k 30 900 python3 "$K/identc.py" run --base http://localhost:8000 --corpus "$CORPUS" --out "$d/identc.json" > "$d/identc.txt" 2>&1 ); fi
  ( timeout -k 30 2700 python3 "$COLD" run --base http://localhost:8000 --corpus "$CORPUS" --lens $LENS --reps $REPS --seed $J --cskip 0 --out "$d/cold.json" > "$d/cold.txt" 2>&1 )
  pos > "$d/pos-c8.before"
  probe "$d" fresh-c8 8 2 418916 500
  pos > "$d/pos-c8.after"
  probe "$d" fresh-c1 1 4 104729 400
  ( cd "$R" && timeout -k 30 2400 uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$H1:8000/v1" --model $MODEL \
      --tokenizer "$SNAP" --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 --concurrency 1 4 8 16 --runs 3 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$H1:8000/metrics" \
      --save-result "$d/task.csv" > "$d/task.log" 2>&1 )
  [ "$(health localhost)" = 200 ] || echo "server gone after the pass" > "$d/DIED"
  keep_logs2 "$d"; stop_all
  m=$(( $(measured "$d/serve.log") + $(measured "$d/serve-node1.log") ))
  [ "$m" = 0 ] || { echo "ranks 0+1 measured $m b12x plans" > "$d/COLD"; }; done
REPORT_MAIN="$MAIN" REPORT_ENV="${ARM_ENV:-}" REPORT_ARM_IMG="${ARM_IMG:-$IMG2}" REPORT_CTL_IMG="$IMG2" TD_SCRIPTS=$R/scripts \
  python3 "$K/report2x.py" "$S" $REPS > "$RES/judge-2x.txt" 2>&1
{ echo "screen ($ORDER):"; cat "$RES/judge-2x.txt"; } >> "$OUT"
T2=$(sed -n 's/^RULE2=//p' "$RES/judge-2x.txt" | tail -1); [ -n "$T2" ] || { T2=EMPTY; echo "RULE2=EMPTY" >> "$OUT"; }
IDV=none; ICV=none
if [ "${IDENTC:-0}" = 1 ]; then
  python3 "$IDENT" report "$S" > "$RES/ident.txt" 2>&1; python3 "$K/identc.py" report "$S" > "$RES/identc.txt" 2>&1
  IDV=$(sed -n 's/^IDENT=//p' "$RES/ident.txt" | tail -1); ICV=$(sed -n 's/^IDENTC=//p' "$RES/identc.txt" | tail -1)
  { echo; echo "== identity ($IDENT, c1, cross-boot; short prompts judged)"; cat "$RES/ident.txt"
    echo "== identity under load (identc.py, 16 at once, 3 rounds per boot)"; cat "$RES/identc.txt"; } >> "$OUT"; fi
case "$T2:${IDV:-none}:${ICV:-none}" in
  WIN:PASS:OK|WIN:UNSTABLE:OK|FLAT:PASS:OK|FLAT:UNSTABLE:OK) ;;
  *) echo "GATE=NOT-RUN (RULE2 $T2, IDENT ${IDV:-none}, IDENTC ${ICV:-none}: the verdict does not depend on it)" >> "$OUT"; exit 0 ;; esac
# release quality gate of k104 on the arm (k61.gate rules): stragglers c5-c16 0 preemptions, tool-use hardmode >= 88,
# TC-45 x5 = 100, fidelity 8k/32k/64k/128k + 128k seeds 11/13 all 20/20
GD=$RES/gate; mkdir -p "$GD"
if ! tboot "$RD/arm-2x.yaml" "$GD" 1; then keep_logs2 "$GD"; stop_all; echo "GATE=NOT-RUN (gate boot failed, $GD)" >> "$OUT"; exit 0; fi
( cd "$R" && timeout 900 python3 scripts/straggler_probe.py 5 6 7 8 12 16 > "$GD/straggler.log" 2>&1 )
timeout 7200 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://localhost:8000 --model $MODEL > "$GD/hardmode.log" 2>&1
timeout 3600 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://localhost:8000/v1 --model $MODEL --scenarios TC-45 --trials 5 > "$GD/tc45.log" 2>&1
( cd "$R" && timeout 5400 python3 scripts/fidelity_probe.py --base http://localhost:8000 --model $MODEL --depths 8000,32000,64000,128000 --out "$GD/fidelity.json" > "$GD/fidelity_probe.txt" 2>&1 )
for sd in 11 13; do ( cd "$R" && timeout 2700 python3 scripts/fidelity_probe.py --base http://localhost:8000 --model $MODEL --depths 128000 --seed $sd --out "$GD/fidelity-seed$sd.json" > "$GD/fidelity-seed$sd.txt" 2>&1 ); done
keep_logs2 "$GD"; stop_all
t=$(mktemp -d); ln -s "$GD" "$t/cand1"; ln -s "$GD" "$t/cand2"
GV=$(python3 -c "import sys; sys.path.insert(0, sys.argv[1]); import k61; b, hm = k61.gate(sys.argv[2]); print('PASS' if not b else 'FAIL (' + '; '.join(b) + ')')" "$(dirname "$K61")" "$t" 2>&1 | tail -1); rm -rf "$t"
case "$GV" in PASS|FAIL*) ;; *) GV="NOT-RUN (k61.gate error: $GV)" ;; esac
{ echo "== release quality gate on the arm (k104 rules, k61.gate): $(cat "$GD/boot.txt" | tr '\n' ' ')"
  grep -hE "Quality:" "$GD/hardmode.log"; grep -hE "Pass\^5|Score:" "$GD/tc45.log" | head -2
  grep -h "^depth" "$GD/fidelity_probe.txt" "$GD"/fidelity-seed1?.txt; tail -8 "$GD/straggler.log"
  echo "GATE=$GV"; } >> "$OUT" 2>&1
exit 0
