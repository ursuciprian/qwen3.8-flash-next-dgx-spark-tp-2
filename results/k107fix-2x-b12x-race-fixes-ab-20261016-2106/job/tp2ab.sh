#!/usr/bin/env bash
# 2x TP=2 A/B harness (2026-10-16) for k107fix, k107 and k108; copy of k100b/tp2ab.sh. The same file sits in each job
# dir; the arm comes from that job's arm.conf (J and K are set by run.sh).
#   control  the 2x v2.2.0 release recipe $BL/k104s/recipe.yaml as k104s built and checked it (code sha256), image $IMG2
#            named explicitly (digest bba1374d, same Id on both Sparks). Never the registry: it still points at v2.1.0
#            until the release switch.
#   arm      control + arm.conf: own name, ARM_IMG when set, the ARM_ENV lines, and a runtime cache of its own under
#            /cache/runtime/$ARM_NAME (XDG_CACHE_HOME there; VLLM_CACHE_ROOT a fresh $ARM_NAME/vllm in place of the
#            control's root, so no AOT graph of the control is ever loaded by the arm)
#   stage    arm cache on both Sparks: no-clobber copies of the control's b12x/triton/... dirs (plan selections and
#            compiled kernels), never a vLLM compile root
#   bake     prebake with a 6 GiB KV pool (<= 3 tries, cold compile allowed) + one cold 16K prompt; then one full-pool
#            boot (cold allowed)
#   screen   ctl arm arm ctl ctl arm arm ctl (4 boots per side), warm only: warm-up 16K+64K, cold TTFT 16K/64K/128K x3,
#            fresh-c8 T=0 acceptance per draft position, fresh-c1 T=0 probe (4 requests, paired with the same pass of the
#            other side), llama-benchy pp2048 + tg512 c1/c4/c8 x3. IDENTC=1 adds k101/ident.py (c1, 8 prompts) and
#            identc.py (16 prompts at once, 3 rounds) before the speed cells.
#   judge    report2x.py: RULE2=WIN (a MAIN cell better beyond noise, nothing worse) | FLAT | LOSS | EMPTY
# Writes $RES2/$J-tp2.txt (last RULE2= line). Holds no lock of its own: run.sh holds both GPU locks.
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
COLD=$BL/k101/cold.py; IDENT=$BL/k101/ident.py; CORPUS=$R/results/corpus-code.txt; BENCHY_SRC=$G/llama-benchy-fork; MODEL=qwen3.8-flash-next
SNAP=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
ORDER="ctl:1 arm:1 arm:2 ctl:2 ctl:3 arm:3 arm:4 ctl:4"; LENS=16384,65536,131072; REPS=3; RD=$RES/recipes
on2() { ssh -n -o ConnectTimeout=10 $H2 "$@"; }
hx() { if [ "$1" = localhost ]; then bash -c "$2"; else on2 "$2"; fi; }
img() { sed -n 's/^container: *//p' "$1" | head -1 | tr -d "\"' "; }
fld() { sed -n "s/^$2: *//p" "$1" | head -1 | tr -d "\"' "; }

mkarm() { # ctl.yaml arm.yaml: own name, image, runtime cache, ARM_ENV; every edit counted
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
pairs = [p.split("=", 1) for p in envs.split()]
keys = ["XDG_CACHE_HOME"] + [k for k, _ in pairs]
if any(l.strip().split(":", 1)[0] in keys for l in lines if l.startswith("  ") and ":" in l):
    sys.exit("mkarm: the control already sets one of the arm's variables")
out = lines[:]
out[n] = f"name: qwen3.8-flash-next-2x-{job}"
if image:
    out[c] = f"container: {image}"
out[r] = f'  VLLM_CACHE_ROOT: "{cache}/vllm"'
add = [f"  # {job} arm: own runtime cache" + ("" if not pairs else ", " + " ".join(envs.split())),
       f'  XDG_CACHE_HOME: "{cache}"'] + [f'  {k}: "{v}"' for k, v in pairs]
out[e + 1:e + 1] = add
a, b = Counter(lines), Counter(out)
changed = 2 + (1 if image else 0)
if sum((a - b).values()) != changed or sum((b - a).values()) != changed + len(add):
    sys.exit("mkarm: unexpected diff")
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
  sed -e 's|^    --gpu-memory-utilization {gpu_memory_utilization}$|    --gpu-memory-utilization {gpu_memory_utilization} \\\n    --kv-cache-memory-bytes '"$KV_PRE"'|' "$RD/arm-2x.yaml" > "$RD/prebake-2x.yaml"
  [ "$(diff "$RD/arm-2x.yaml" "$RD/prebake-2x.yaml" | grep -c '^>')" = 2 ] && grep -qx "    --kv-cache-memory-bytes $KV_PRE" "$RD/prebake-2x.yaml" \
    || { echo "prebake-2x.yaml edit did not apply"; return 1; }; }
imgid() { hx $1 "docker image inspect -f '{{.Id}}' $2" 2>/dev/null; }
checks() {
  local ok=0 f t h
  for f in "$COLD" "$K/report2x.py" "$CORPUS" "$R/scripts/depth_decode_probe.py" "$R/scripts/thunderdome_report.py" \
           "$R/scripts/paired_decode_ab.py" "$CTL_SRC"; do [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  [ -d "$BENCHY_SRC" ] && [ -d "$SNAP" ] || { echo "missing llama-benchy fork or tokenizer snapshot"; ok=1; }
  for t in sparkrun uvx python3 curl; do command -v $t > /dev/null || { echo "missing $t"; ok=1; }; done
  python3 "$K/report2x.py" selftest > /dev/null || { echo "report2x.py selftest failed"; ok=1; }
  if [ "${IDENTC:-0}" = 1 ]; then
    python3 "$K/identc.py" selftest > /dev/null || { echo "identc.py selftest failed"; ok=1; }
    python3 "$IDENT" selftest > /dev/null || { echo "ident.py selftest failed"; ok=1; }; fi
  mkrecipes || ok=1
  for h in localhost $H2; do
    [ "$(imgid $h $IMG2)" = "$IMG2_ID" ] || { echo "$h: $IMG2 missing or not Id $IMG2_ID"; ok=1; }
    hx $h "test -d $CRC/b12x/compile/preparation && test -d $CRC/${CTL_ROOT#/cache/runtime/}" || { echo "$h: 2x v2.2.0 runtime cache under $CRC missing"; ok=1; }; done
  hx localhost "docker image inspect -f '{{join .RepoDigests \" \"}}' $IMG2" | grep -q "$IMG2_DIGEST" || { echo "$IMG2 digest is not $IMG2_DIGEST"; ok=1; }
  [ -s "$RD/arm-2x.yaml" ] && { echo "arm diff:"; diff "$RD/ctl-2x.yaml" "$RD/arm-2x.yaml" | sed 's/^/   /'; }
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
tboot() { # recipe dir cold_ok (k99)
  local rec=$1 d=$2 s m n cap; cap=$([ "$3" = 1 ] && echo 5400 || echo 1800)
  stop_all; cp "$rec" "$d/recipe.yaml"
  ( cd "$(dirname "$rec")" && timeout -k 60 900 sparkrun run "$(basename "$rec")" --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  s=$(date +%s)
  while :; do
    n=$(n0)
    m=$([ -n "$n" ] && docker exec "$n" grep -m1 -oE 'Directly load AOT compilation|Dynamo bytecode transform time' /tmp/sparkrun_serve.log 2>/dev/null)
    [ "$m" = "Dynamo bytecode transform time" ] && [ "$3" != 1 ] && return 2
    [ "$(health localhost)" = 200 ] && { echo "boot $(( $(date +%s) - s ))s ${m:-no AOT line}" > "$d/boot.txt"; return 0; }
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
    timeout -k 30 600 python3 "$COLD" run --base http://localhost:8000 --corpus "$CORPUS" --lens 16384 --reps 1 --seed ${J}b --cskip 3600000 \
      --out "$d/bake16k.json" > "$d/bake16k.txt" 2>&1
    keep_logs2 "$d"; stop_all; ok=1; break; fi
  keep_logs2 "$d"; stop_all; done
[ -n "$ok" ] || fail "prebake: 3 tries failed ($RES/bake; a 'code-generation snapshot differs' line in serve.log means seeded plan records of another codegen)"
d=$RES/bake/arm-full; mkdir -p "$d"
tboot "$RD/arm-2x.yaml" "$d" 1 || { keep_logs2 "$d"; fail "bake full-pool boot failed"; }
keep_logs2 "$d"; stop_all
echo "bake: prebake $(cat "$RES"/bake/prebake-try*/boot.txt 2>/dev/null | tail -1); full $(cat "$d/boot.txt"); rank 0 measured $(measured "$d/serve.log")" >> "$OUT"

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
  ( cd "$R" && timeout -k 30 1800 uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$H1:8000/v1" --model $MODEL \
      --tokenizer "$SNAP" --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 --concurrency 1 4 8 --runs 3 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$H1:8000/metrics" \
      --save-result "$d/task.csv" > "$d/task.log" 2>&1 )
  [ "$(health localhost)" = 200 ] || echo "server gone after the pass" > "$d/DIED"
  keep_logs2 "$d"; stop_all
  [ "$(measured "$d/serve.log")" = 0 ] || { echo "rank 0 measured $(measured "$d/serve.log") b12x plans" > "$d/COLD"; }; done
REPORT_MAIN="$MAIN" REPORT_ENV="${ARM_ENV:-}" REPORT_ARM_IMG="${ARM_IMG:-$IMG2}" REPORT_CTL_IMG="$IMG2" python3 "$K/report2x.py" "$S" $REPS > "$RES/judge-2x.txt" 2>&1
{ echo "screen ($ORDER):"; cat "$RES/judge-2x.txt"; } >> "$OUT"
[ "${IDENTC:-0}" = 1 ] && { echo; echo "== identity (k101/ident.py, c1, cross-boot)"; python3 "$IDENT" report "$S" 2>&1
  echo "== identity under load (identc.py, 16 at once, 3 rounds per boot)"; python3 "$K/identc.py" report "$S" 2>&1; } >> "$OUT"
grep -q '^RULE2=' "$RES/judge-2x.txt" || echo "RULE2=EMPTY" >> "$OUT"
exit 0
