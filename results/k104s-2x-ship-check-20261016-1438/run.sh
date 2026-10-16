#!/usr/bin/env bash
# k104s-2x-ship-check (2026-10-16, tp-2 #176): release build and fresh-user check boot of 2x v2.2.0 = 2x v2.1.0 with
# GDN prefill views (VLLM_GDN_PREFILL_VIEWS=1; k101s TP=2 RULE2=WIN, k104 GATE=PASS).
#   image   BASE = the tested image $TESTED (k101s/k104, same Id on both Sparks: the v2.1.0 image + six files of vLLM
#           10d0acc8) + one layer: the f509586e AOT artifacts the k101s bake compiled for those files (rank_0_0 from
#           dgx-01's k101 cache, rank_1_0 from dgx-02's; v2.1.0's artifacts under the same key fail with "Source code has
#           changed" and recompile). Plan 225bf665 and AOT key 295b0fe4 are already in the tested image, unchanged.
#           Then STATE ci-open: the Mac poster runs CI build-b0-warm on the dgx-01 runner (base BASE, tag $TAG, recipe
#           ref main) and writes ci-done (cap 2 h), as for v2.1.0 (k94). 2x-v2.2.0 is added after a pass (retag-image).
#   bytes   fail-closed: TAG's layers start with every layer of TESTED; the vLLM and b12x python trees, the six overlay
#           files, nvfp4/b12x.py and /opt/k101-overlay match TESTED; the seed's model files match the k101 caches.
#           dgx-02 pulls TAG from ghcr (anonymous) and must get the same Id as dgx-01.
#   check   plain `sparkrun run` of the release branch recipe ($REC, sha256 checked; the registry follows main, so a
#           by-name registry boot can only happen after the merge) with the real runtime caches: plan 225bf665 and the
#           new compile cache root $ROOT are moved out first on both Sparks, so the image has to supply them.
#           pass = AOT loaded on both ranks from $ROOT (no recompile, no Dynamo transform, nothing saved), 0 b12x plans
#           measured on rank 0, 682 records on both ranks, views line on both ranks, pong, /v1/models, revision +
#           VLLM_GDN_PREFILL_VIEWS=1 on rank 0, T=0 fresh-c8 acceptance within 0.03 of the k101s arm, llama-benchy
#           pp2048 + tg512 c1/c8 (3 runs) not below the k101s arm means by more than the k101s bands.
#   recheck written before any boot (k94b rule): if a benchy cell misses its band on boot 1, one more boot; then PASS
#           needs the 2-boot mean of every cell inside its band and no boot below the lowest k101s arm boot of that cell.
#   end     moved items go back unless the boot recreated them; lib.sh restores the shipped 2x (v2.1.0 from the
#           registry). Promotion happens outside this job after a pass.
# ETA ~1 h (build ~2, CI ~10-30, boot ~6, checks ~15) + restore ~6 min. Contract: both locks; exit 75 if busy;
# STATE DONE/FAILED; comment.md + RESULT.
NEEDS="dgx01 dgx02"
set -u
J=k104s; K=$HOME/GEN-AI/backlog/k104s
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k104s-2x-ship-check-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"   # BENCHY_SRC, TOK, MODEL
TESTED=spark-vllm-b12x:k101-gv-10d0acc8-d21d7ade; TESTED_ID=sha256:d7da780422fd7c70f76118046e1a2f6dcba1bf926ccdbf337c8c53a73e0f9950
BASE=spark-vllm-b12x:k104s-base-21e0b201-d21d7ade-10d0acc8
TAG=ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm
REV=03f4a0570496bbe741189f4a2a7d611679e85406; REC=$K/recipe.yaml; VLLM_SHA=10d0acc85fb450472230558eda063c9059c20234
RECCODE=ef181a0eb043d4e08c73c86330049ade36c925553406e0560d30a3ba51c9d431   # sha256 of the recipe without comment lines (release branch at staging)
ROOT=vllm-2x-21e0b201-10d0acc8
PLAN=225bf66521da31897db0719256f6a56cc38d43f87dd4f85f1f52549540986fe2.json; NREC=682; PLANSHA=112234f73838e908
KEY=f509586e7e4d7ee777946c1a37707d0f8702a0822867b3bf8a7e80d6f41ca1d5
M0=074743a4d2566986bbe440247499e8a7973cb162bd234c1eb3cd1b1b986e9a82   # rank_0_0/model, dgx-01 k101 cache
M1=b0a1254a8af4d4898f40d2fe9e46cdc749d903d9a02f3d5cb1f7d0a667a91d12   # rank_1_0/model, dgx-02 k101 cache
RC=$HOME/.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923
AOTSRC=$RC/k101/vllm/torch_compile_cache/torch_aot_compile/$KEY
SEED=/opt/b12x-seed/torch_compile_cache/torch_aot_compile/$KEY
DP=/usr/local/lib/python3.12/dist-packages; B12XPY=$DP/vllm/model_executor/kernels/linear/nvfp4/b12x.py
PATCHED=$BL/k101/overlay/patched.sha256
VIEWS='GDN prefill views: in-order'
CORPUS=$R/results/corpus-code.txt
# k101s TP=2 arm (results/k101s-1x-gdn-prefill-views-ab-20261015-1600/tp2): mean, band %, lowest of the 4 arm boots
REF='pp2048 (c1):2892.4:4.2:2877.8;tg512 (c1):84.9:15.5:79.4;tg512 (c8):195.2:11.7:186.5'
ACC_REF="0.850 0.704 0.576 0.470"
on2() { ssh -n -o ConnectTimeout=10 $H2 "$@"; }
x2() { if [ "$1" = localhost ]; then bash -c "$2"; else on2 "$2"; fi; }
recok() { [ "$(grep -vE '^[[:space:]]*#' "$REC" | sha256sum | cut -c1-64)" = "$RECCODE" ] && grep -qx "container: $TAG" "$REC" \
  && grep -qx "model_revision: $REV" "$REC" && grep -q -- "--revision $REV" "$REC" && grep -qx '  VLLM_GDN_PREFILL_VIEWS: "1"' "$REC" \
  && grep -qx "  VLLM_CACHE_ROOT: \"/cache/runtime/$ROOT\"" "$REC" && grep -q 'cp -rn "$A/." "$VLLM_CACHE_ROOT/torch_compile_cache/"' "$REC" \
  && sed -n 1p "$REC" | grep -q '^# Release: v2.2.0'; }
pytree() { docker run --rm --network none --entrypoint bash "$1" -c "cd $DP && find vllm b12x -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64"; }
filesha() { docker run --rm --network none --entrypoint sha256sum "$1" "$2" | cut -c1-64; }
srcok() { [ "$(sha256sum "$AOTSRC/rank_0_0/model" 2>/dev/null | cut -c1-64)" = $M0 ] \
  && [ "$(on2 "sha256sum $AOTSRC/rank_1_0/model" 2>/dev/null | cut -c1-64)" = $M1 ]; }
if [ "${1:-}" = --dry-run ]; then rc=0
  recok && echo "ok recipe v2.2.0 (code sha256 $RECCODE)" || { echo "recipe check failed"; rc=1; }
  for h in localhost $H2; do [ "$(x2 $h "docker image inspect -f '{{.Id}}' $TESTED" 2>/dev/null)" = $TESTED_ID ] \
    && echo "ok $TESTED on $h" || { echo "$TESTED missing or wrong Id on $h"; rc=1; }; done
  srcok && echo "ok k101 AOT artifacts (rank_0_0 dgx-01, rank_1_0 dgx-02)" || { echo "k101 AOT artifacts missing or changed"; rc=1; }
  [ -s "$PATCHED" ] || { echo "missing $PATCHED"; rc=1; }
  for f in check_seed.py depth_decode_probe.py; do [ -s "$R/scripts/$f" ] || { echo "missing $f"; rc=1; }; done
  [ -d "$BENCHY_SRC" ] && [ -d "$TOK" ] && [ -s "$CORPUS" ] || { echo "missing llama-benchy fork, tokenizer or corpus"; rc=1; }
  docker image inspect $TAG > /dev/null 2>&1 && echo "note: $TAG already on dgx-01 (a rerun builds and pushes it again)"
  echo "dry-run exit=$rc"; exit $rc; fi
need_lock
others_busy && { echo "another GPU job is active" >&2; exit 75; }
mkdir -p "$RES"; echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"; job_begin
MOVED=
unmove() { [ -n "$MOVED" ] || return 0; local h
  for h in localhost $H2; do x2 $h "cd $RC; [ -e b12x/compile/preparation/$PLAN ] || { [ -e $K/moved/$PLAN ] && mv $K/moved/$PLAN b12x/compile/preparation/; }
    [ -e $ROOT ] || { [ -e $K/moved/$ROOT ] && mv $K/moved/$ROOT .; }; true"; done; MOVED=; }
trap 'kill $G1 $G2 2>/dev/null; unmove; restore || { sleep 60; restore; } || FINAL="FAILED: restore -- $FINAL"; st "$FINAL"' EXIT
fail() { FINAL="FAILED: k104s $*"; exit 1; }
recok || fail "recipe check"
cp "$K/run.sh" "$REC" "$RES/"
st "running: stop 2x"; stop_all

st "running: build the local base $BASE"
B=$RES/build; mkdir -p "$B"; I=$K/img; rm -rf "$I"; mkdir -p "$I/aot/$KEY/rank_0_0" "$I/aot/$KEY/rank_1_0"
srcok || fail "k101 AOT artifacts changed since staging"
cp "$AOTSRC/rank_0_0/model" "$I/aot/$KEY/rank_0_0/model" && scp -q -o ConnectTimeout=10 "$H2:$AOTSRC/rank_1_0/model" "$I/aot/$KEY/rank_1_0/model" \
  || fail "copy of the AOT artifacts"
chmod 755 "$I/aot" "$I/aot/$KEY" "$I/aot/$KEY"/rank_*; chmod 644 "$I/aot/$KEY"/rank_*/model
( cd "$I/aot/$KEY" && sha256sum rank_0_0/model rank_1_0/model ) > "$B/aot.sha256"
grep -q "^$M0  rank_0_0/model$" "$B/aot.sha256" && grep -q "^$M1  rank_1_0/model$" "$B/aot.sha256" || fail "staged AOT sha256 ($B/aot.sha256)"
cat > "$I/Dockerfile" <<EOF
# k104s (tp-2 #176, 2026-10-16): the k101s/k104 test image ($TESTED, Id ${TESTED_ID:7:12}) + the f509586e AOT artifacts of
# both ranks compiled for its vLLM files (k101s bake). Everything else in /opt/b12x-seed is the v2.1.0 seed.
FROM $TESTED
COPY aot/ /opt/b12x-seed/torch_compile_cache/torch_aot_compile/
EOF
cp "$I/Dockerfile" "$B/"
DOCKER_BUILDKIT=1 timeout 1200 docker build -t "$BASE" "$I" > "$B/build.log" 2>&1 9>&- || fail "docker build ($B/build.log)"

layers() { docker image inspect -f '{{json .RootFS.Layers}}' "$1"; }
prefix() { python3 -c 'import json,sys; a,b=json.loads(sys.argv[1]),json.loads(sys.argv[2]); sys.exit(0 if b[:len(a)]==a and len(b)>len(a) else 1)' "$(layers "$1")" "$(layers "$2")"; }
seedok() { [ "$(filesha "$1" $SEED/rank_0_0/model)" = $M0 ] && [ "$(filesha "$1" $SEED/rank_1_0/model)" = $M1 ] \
  && [ "$(filesha "$1" /opt/b12x-seed/preparation/$PLAN | cut -c1-16)" = $PLANSHA ]; }
prefix "$TESTED" "$BASE" && seedok "$BASE" || fail "local base check (layers or seed)"

st "ci-open"   # the Mac poster reads this exact line, triggers build-b0-warm and writes ci-done
rm -f "$K/ci-done"; echo "$BASE $TAG" > "$K/ci-request"; log "ci window: $BASE -> $TAG"
s=$(date +%s); until [ -f "$K/ci-done" ] || [ $(( $(date +%s) - s )) -gt 7200 ]; do sleep 20; done
log "ci window closed: $(cat "$K/ci-done" 2>/dev/null || echo timeout)"; rm -f "$K/ci-request"
cp "$K/ci-done" "$B/ci-done.txt" 2>/dev/null
grep -q ' ok ' "$K/ci-done" 2>/dev/null || fail "CI build-b0-warm: $(cat "$K/ci-done" 2>/dev/null || echo 'no ci-done in 2 h')"
DIG=$(docker image inspect -f '{{join .RepoDigests " "}}' "$TAG" 2>/dev/null | tr ' ' '\n' | grep '^ghcr.io/ursuciprian/spark-vllm-b12x@' | head -1)
[ -n "$DIG" ] || fail "no ghcr RepoDigest on $TAG after the CI run"
ID1=$(docker image inspect -f '{{.Id}}' "$TAG")

st "running: byte checks of $TAG against $TESTED"
VB=$B/bytes.txt; : > "$VB"; bad=
prefix "$TESTED" "$TAG" && echo "layers: all $(layers "$TESTED" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)))') layers of $TESTED are the first layers of $TAG" >> "$VB" || bad="$bad layers;"
t1=$(pytree "$TESTED"); t2=$(pytree "$TAG"); echo "vllm+b12x python tree sha256: tested $t1, release $t2" >> "$VB"; [ -n "$t1" ] && [ "$t1" = "$t2" ] || bad="$bad pytree;"
while read -r h f; do r=$(filesha "$TAG" "$DP/$f"); echo "$f: $r (patched.sha256 $h)" >> "$VB"; [ "$r" = "$h" ] || bad="$bad $f;"; done < "$PATCHED"
[ "$(filesha "$TESTED" $B12XPY)" = "$(filesha "$TAG" $B12XPY)" ] && echo "nvfp4/b12x.py: same as tested" >> "$VB" || bad="$bad b12x.py;"
docker run --rm --network none --entrypoint cat "$TAG" /opt/k101-overlay | tee -a "$VB" | grep -q "^$VLLM_SHA " || bad="$bad overlay marker;"
seedok "$TAG" && echo "seed: f509586e rank_0_0 ${M0:0:12}, rank_1_0 ${M1:0:12}, plan ${PLAN:0:8} $PLANSHA" >> "$VB" || bad="$bad seed;"
[ -z "$bad" ] || fail "byte checks:$bad ($VB)"
st "running: pull $TAG on dgx-02 from ghcr"
on2 "docker pull -q $TAG" > "$B/pull-dgx02.txt" 2>&1 || fail "pull on dgx-02 ($B/pull-dgx02.txt)"
ID2=$(on2 "docker image inspect -f '{{.Id}}' $TAG"); [ "$ID1" = "$ID2" ] || fail "Id differs: dgx-01 $ID1, dgx-02 $ID2"
echo "digest $DIG; Id $ID1 on both Sparks" >> "$VB"

st "running: move the plan file and the $ROOT compile cache root out of the runtime caches"
for h in localhost $H2; do x2 $h "mkdir -p $K/moved; cd $RC; [ -e b12x/compile/preparation/$PLAN ] && mv b12x/compile/preparation/$PLAN $K/moved/ && echo moved plan
  [ -e $ROOT ] && mv $ROOT $K/moved/ && echo moved $ROOT; true"; done > "$RES/cleared.txt" 2>&1; MOVED=1

pos() { curl -s -m 10 "localhost:8000/metrics" | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens(_per_pos)?_total'; }
boot() { # n: boot the release recipe, collect facts into $D (= $RES/boot$n)
  D=$RES/boot$1; mkdir -p "$D"; local s=$(date +%s); up=
  ( cd "$K" && timeout -k 60 1800 sparkrun run "$REC" --no-follow ) > "$D/sparkrun.log" 2>&1 < /dev/null 9>&-
  while [ $(( $(date +%s) - s )) -lt 1800 ]; do [ "$(health localhost)" = 200 ] && { up=$(( $(date +%s) - s )); break; }
    [ $(( $(date +%s) - s )) -gt 240 ] && [ -z "$(n0)" ] && break; sleep 5; done
  c=$(n0); [ -n "$c" ] || c=$(docker ps -a --format '{{.Names}}' | grep node_0 | head -1)
  docker exec "$c" cat /tmp/sparkrun_serve.log > "$D/serve.log" 2>&1 || docker logs "$c" > "$D/serve.log" 2>&1
  on2 'c=$(docker ps -a --format "{{.Names}}" | grep node_1 | head -1); [ -n "$c" ] && { docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1; }' > "$D/serve-node1.log" 2>&1
  [ -n "$up" ] || return 1
  docker exec "$c" sh -c "ps aux | grep '[v]llm serve'" > "$D/cmdline.txt" 2>&1
  docker exec "$c" env | grep -E '^(VLLM_GDN_PREFILL_VIEWS|VLLM_CACHE_ROOT|XDG_CACHE_HOME)=' | sort > "$D/env.txt" 2>&1
  { docker inspect --format '{{.Config.Image}} {{.Image}}' "$c"; on2 "c=\$(docker ps --format '{{.Names}}' --filter name=sparkrun | head -1); docker inspect --format '{{.Config.Image}} {{.Image}}' \$c"; } > "$D/image.txt" 2>&1
  P=$(pong localhost); curl -s -m 20 localhost:8000/v1/models > "$D/models.json"
  echo "boot $1: health after $up s" > "$D/boot.txt"; return 0; }
warm() { # D -> WARM or COLD/...
  local f w=WARM; for f in serve.log serve-node1.log; do
    grep -q "Directly load AOT compilation from path /cache/runtime/$ROOT/" "$D/$f" \
      && ! grep -qE 'Compiling model again|saved AOT compiled function|Dynamo bytecode transform' "$D/$f" || w=COLD; done; echo $w; }
benchy() { ( cd "$R" && child timeout -k 30 1800 uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$H1:8000/v1" --model $MODEL \
    --tokenizer "$TOK" --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 --concurrency 1 8 --runs 3 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$H1:8000/metrics" \
    --save-result "$D/task.csv" > "$D/task.log" 2>&1 ); }
judge() { # task.csv files -> OK|BAD | text; mean of the boots given vs the k101s arm
  python3 - "$REF" "$@" <<'PY'
import sys
ref = {}
for p in sys.argv[1].split(";"):
    name, m, band, lo = p.rsplit(":", 3); ref[name] = (float(m), float(band), float(lo))
runs = []
for path in sys.argv[2:]:
    got = {}
    for line in open(path):
        f = [x.strip() for x in line.split("|")]
        if len(f) > 3 and f[2] in ref:
            try: got[f[2]] = float(f[3].split()[0])
            except ValueError: pass
    runs.append(got)
out, bad = [], []
for name, (m, band, lo) in ref.items():
    vals = [r[name] for r in runs if name in r]
    if len(vals) != len(runs): bad.append(f"{name} missing"); continue
    v = sum(vals) / len(vals); d = (v / m - 1) * 100
    out.append(f"{name} {'/'.join(f'{x:.1f}' for x in vals)} -> {v:.1f} ({d:+.1f}% vs k101s arm {m}, band {band}%)")
    if d < -band: bad.append(f"{name} below band")
    if len(runs) > 1 and min(vals) < lo - 0.05: bad.append(f"{name} boot below the lowest arm boot {lo}")
print(("OK" if not bad else "BAD (" + "; ".join(bad) + ")") + " | " + "; ".join(out))
PY
}

st "running: check boot 1"
boot 1 || fail "check boot: no /health in 1800 s ($D)"
B1=$D; P1=$P; up1=$up; WARM=$(warm)
nr() { x2 $1 "python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))[\"records\"]))' $RC/b12x/compile/preparation/$PLAN" 2>/dev/null; }
N1=$(nr localhost); N2=$(nr $H2)
SEEDCHK=$(python3 "$R/scripts/check_seed.py" "$D/serve.log" 2>&1 | tail -2 | tr '\n' ' ')
MODELS=$(python3 -c 'import json,sys; print(" ".join(m["id"] for m in json.load(open(sys.argv[1]))["data"]))' "$D/models.json" 2>/dev/null)
st "running: fresh-c8 acceptance probe + llama-benchy (boot 1)"
( cd "$R" && timeout -k 30 300 python3 scripts/depth_decode_probe.py --base "http://$H1:8000" --label warm --corpus "$CORPUS" --concurrency 1 --repeats 1 --temperature 0 --out "$D/probe-warm.json" --depth 4096 --new 2048 --offset 900000 --max-tokens 512 > "$D/probe-warm.log" 2>&1 )
pos > "$D/pos.before"
( cd "$R" && timeout -k 30 500 python3 scripts/depth_decode_probe.py --base "http://$H1:8000" --label fresh-c8 --corpus "$CORPUS" --concurrency 8 --repeats 2 --temperature 0 --out "$D/probe-fresh-c8.json" --depth 256 --new 2048 --offset 418916 --max-tokens 512 > "$D/probe-fresh-c8.log" 2>&1 )
pos > "$D/pos.after"
ACC=$(python3 - "$D/pos.before" "$D/pos.after" <<'PY'
import re, sys
def rd(p):
    c = {"accepted": {}, "draft": {}}
    for l in open(p):
        m = re.match(r'vllm:spec_decode_num_(accepted|draft)_tokens_per_pos_total\{.*position="(\d+)"\} ([0-9.e+]+)', l)
        if m: c[m[1]][int(m[2])] = c[m[1]].get(int(m[2]), 0) + float(m[3])
    return c
a, b = rd(sys.argv[1]), rd(sys.argv[2])
print(" ".join(f"{(b['accepted'].get(k,0)-a['accepted'].get(k,0))/max(b['draft'][k]-a['draft'].get(k,0),1):.3f}" for k in sorted(b["draft"])))
PY
)
benchy; BV=$(judge "$B1/task.csv")
{ echo "image: $(tr '\n' ' ' < "$D/image.txt")"; echo "boot 1: health after $up1 s; pong: $P1; /v1/models: $MODELS"
  for f in serve.log serve-node1.log; do echo "$f: Directly load AOT $(grep -c 'Directly load AOT compilation' "$D/$f") (from $ROOT $(grep -c "Directly load AOT compilation from path /cache/runtime/$ROOT/" "$D/$f")), compiled again $(grep -c 'Compiling model again' "$D/$f"), saved AOT $(grep -c 'saved AOT compiled function' "$D/$f"), Dynamo bytecode transform $(grep -c 'Dynamo bytecode transform' "$D/$f"), views lines $(grep -c "$VIEWS" "$D/$f"), MXFP8 copies $(grep -c 'MXFP8 copy serves rows >= 41 ' "$D/$f"), Traceback $(grep -c Traceback "$D/$f")"; done
  echo "$SEEDCHK"; echo "plan $PLAN: rank 0 ${N1:-missing} records, rank 1 ${N2:-missing} records (seed $NREC)"
  echo "revision: $(grep -oE -- "--revision [0-9a-f]+" "$D/cmdline.txt" | head -1); env: $(tr '\n' ' ' < "$D/env.txt")"
  sed 's/^/cleared before boot: /' "$RES/cleared.txt"; } > "$D/facts.txt"
V=PASS; why=
[ "$WARM" = WARM ] || { V=FAIL; why="$why boot $WARM;"; }
echo "$SEEDCHK" | grep -q 'seed check PASS' && [ "$N1" = $NREC ] && [ "$N2" = $NREC ] || { V=FAIL; why="$why plans measured or records;"; }
grep -q "$VIEWS" "$D/serve.log" && grep -q "$VIEWS" "$D/serve-node1.log" || { V=FAIL; why="$why views line missing on a rank;"; }
echo "$P1" | grep -qi pong || { V=FAIL; why="$why no pong;"; }
echo " $MODELS " | grep -q ' qwen3.8-flash-next ' || { V=FAIL; why="$why /v1/models;"; }
grep -q -- "--revision $REV" "$D/cmdline.txt" && grep -qx 'VLLM_GDN_PREFILL_VIEWS=1' "$D/env.txt" && grep -qx "VLLM_CACHE_ROOT=/cache/runtime/$ROOT" "$D/env.txt" || { V=FAIL; why="$why cmdline or env;"; }
python3 -c 'import sys; a=[float(x) for x in sys.argv[1].split()]; r=[float(x) for x in sys.argv[2].split()]; sys.exit(0 if len(a)>=4 and all(abs(x-y)<=0.03 for x,y in zip(a,r)) else 1)' "$ACC" "$ACC_REF" || { V=FAIL; why="$why acceptance $ACC;"; }
BOOTS="1"; BJ=$BV
if [ "$V" = PASS ] && [ "${BV%% *}" != OK ]; then   # pre-registered recheck (header): one more boot, judge the 2-boot mean
  log "boot 1 benchy outside a band ($BV): recheck boot"; stop_all
  st "running: recheck boot 2"
  if boot 2; then w2=$(warm); B2=$D; benchy; BJ=$(judge "$B1/task.csv" "$B2/task.csv"); BOOTS="1 2"
    echo "$P" | grep -qi pong || { V=FAIL; why="$why boot 2 no pong;"; }
    grep -q "$VIEWS" "$D/serve.log" && grep -q "$VIEWS" "$D/serve-node1.log" || { V=FAIL; why="$why boot 2 views line;"; }
    echo "boot 2: health after $up s, AOT $w2" >> "$B1/facts.txt"
  else V=FAIL; why="$why recheck boot did not come up;"; fi
fi
[ "${BJ%% *}" = OK ] || { V=FAIL; why="$why ${BJ%% |*};"; }
docker exec "$(n0)" cat /tmp/sparkrun_serve.log > "$D/serve-end.log" 2>&1; stop_all
{ echo "k104s $(TZ=Europe/Bucharest date '+%F %T %Z'): 2x v2.2.0 build and fresh-user check boot, image $TAG: $V"
  echo "digest: ${DIG#*@}"; cat "$B/bytes.txt"; cat "$B1/facts.txt"
  echo "fresh-c8 acceptance per position (T=0): $ACC (k101s arm $ACC_REF)"; echo "benchy (boots $BOOTS): $BJ"; } > "$RES/k104s.txt"
FINAL="DONE: k104s $V boot=$WARM ${up1}s digest=${DIG#*@} acc=$ACC ($RES/k104s.txt)${why:+ --$why}"
{ echo "k104s-2x-ship-check result: I built the 2x v2.2.0 release image and booted it the way a new user would, with a plain \`sparkrun run\` of the release recipe (2x v2.1.0 plus \`VLLM_GDN_PREFILL_VIEWS=1\` and a compile cache root of its own)."
  echo; echo "- Image \`$TAG\`, digest \`${DIG#*@}\`. It is the image the A/B and the gate ran on, plus one layer with the compile cache of both ranks for the new GDN files. Every layer of the tested image is in it unchanged, and the vLLM and b12x code is byte-identical to the tested image. The second Spark pulled it from ghcr and got the same image."
  echo "- Before the boot I moved the plan file and the new compile cache root out of the runtime caches on both Sparks, so the image had to supply them. Up in $up1 s, $WARM on both ranks (compile cache loaded from the image, no compile), $(echo "$SEEDCHK" | grep -oE 'seed check (PASS|FAIL)'), plan records $N1 / $N2. Both ranks log the GDN views path."
  echo "- Pong and \`/v1/models\` answer; rank 0 serves \`--revision ${REV:0:8}\`."
  echo "- T=0 fresh-c8 acceptance per draft position: $ACC (A/B arm: $ACC_REF)."
  echo "- llama-benchy (T=1, 3 runs, boots $BOOTS) against the A/B arm means: ${BJ#*| }."
  echo; echo "Verdict: $V.${why:+ Reasons:$why} The quality gate is k104 (same image code, weights, plans and flag)."
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xe2\x80\x93/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"; cp "$K/comment.md" "$RES/comment.md"
exit 0
