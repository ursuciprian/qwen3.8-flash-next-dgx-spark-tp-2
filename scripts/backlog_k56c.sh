#!/usr/bin/env bash
# k56c (2026-10-08, #97): Thunderdome screen of refit run3 (backlog/r3train: more data) against refit run1, the
# drafter shipped on the single Spark. k56b (run2 vs run1) with run3. Runs after backlog/r3train.
#   skip    unless r3train's STATE is DONE and run3's offline gain over run1 (refit-r3/eval-refit-run3.json vs
#           refit-p3/eval-refit-run1.json, mean T=0 per-position acceptance, category all, positions 1-6,
#           refit-p3/job.sh gain()) is >= +1.00 pt: STATE "SKIPPED: ...", the pair is not touched
#   stage   (as k56b) splice refit-r3/runs/run3 into snapshot f4..63 (= f4..03 + the spliced shard) on both Sparks;
#           b12x seed re-keyed to f4..63 (k48/remap_seed.py, from v3d's seed); thin image k56c-refit-run3-<sha8> =
#           v3d warm + seed on both Sparks; arm recipe = the k56 run1 recipe with model f4..63, that image and its own
#           VLLM_CACHE_ROOT
#   control the k56 run1 recipe frozen as k56b/ctl-run1.yaml (f4..61, spark-vllm-b12x:k56-refit-run1-c92ac62d, cache
#           root vllm-k56-refit), copied to k56c/ctl-run1.yaml, each arm's "base:" (= its control)
#   screen  thunderdome.sh k56c/run3-01 (dgx-01) k56c/run3-02 (dgx-02), bake on both, hook k53/hook.sh,
#           TD_ACC_RISE_OK=1 (#105: an acceptance rise is reported, drops stay strict), split gate on a PROMOTE
#           (thunderdome default), thunderdome restores the 2x
# A run3 PROMOTE is not published by this job (report back first).
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/SKIPPED/FAILED; comment.md + RESULT for the poster; 2x restored on exit.
# Usage: bash run.sh --dry-run (passes before run3 has run) | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh
set -u
J=k56c; K=$HOME/GEN-AI/backlog/k56c; P=$HOME/GEN-AI/refit-p3; W=$HOME/GEN-AI/refit-r3
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/thunderdome-k56c-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
V3D=ghcr.io/ursuciprian/spark-vllm-b12x:tp1-v3d-20261005-21e0b201-5dad364d-warm
SEED=8bb5a5b317a3975df599d900e295b12b6ee7225bd8eff6f0de3adc67fd55202e.json   # v3d's b12x seed in $V3D (f4..03 key)
F403=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/f400000000000000000000000000000000000003
F461=${F403%03}61; F463=${F403%03}63
CIMG=spark-vllm-b12x:k56-refit-run1-c92ac62d; CTL=$K/ctl-run1.yaml; RF=$W/runs/run3/mtp_refit.safetensors
GAIN_MIN=0.01; ARMS="run3-01 run3-02"; R3=$W/eval-refit-run3.json

gain() { # run1.json run3.json -> "mean d1 .. d6" (T=0, category all, positions 1-6; refit-p3/job.sh gain())
  python3 - "$1" "$2" <<'EOF'
import json, sys
b, r = (json.load(open(f))["all"]["t0"]["per_position"] for f in sys.argv[1:3])
d = [y - x for x, y in zip(b, r) if x is not None and y is not None]
assert len(d) == 6, (b, r)
print(f"{sum(d) / len(d):.4f}", *(f"{x:+.4f}" for x in d))
EOF
}
on2() { ssh -n -o ConnectTimeout=10 $H2 "$@"; }
preflight() { # before staging: inputs on both Sparks
  local ok=0
  for f in "$P/src/tools/mtp_refit/splice.py" "$G/k48/remap_seed.py" "$G/k53/hook.sh" "$TD" "$CTL" "$P/eval-refit-run1.json"; do [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  [ "$(gain "$P/eval-shipped.json" "$P/eval-refit-run1.json" | cut -d' ' -f1)" = 0.0688 ] || { echo "gain() does not reproduce run1's +0.0688 over shipped"; ok=1; }
  grep -qx "model: $F461" "$CTL" && grep -qx "container: $CIMG" "$CTL" && grep -q 'VLLM_CACHE_ROOT: "/cache/runtime/vllm-k56-refit"' "$CTL" \
    || { echo "$CTL is not the k56 run1 recipe"; ok=1; }
  local h; for h in localhost $H2; do
    x() { if [ $h = localhost ]; then bash -c "$1"; else on2 "$1"; fi; }
    x "docker image inspect $V3D >/dev/null 2>&1 && docker image inspect $CIMG >/dev/null 2>&1" || { echo "$h: $V3D or $CIMG missing"; ok=1; }
    x "test -s $F461/SPLICE.json && test -d $F403" || { echo "$h: f4..61 or f4..03 missing"; ok=1; }; done
  docker run --rm --entrypoint test "$V3D" -f /opt/b12x-seed/preparation/$SEED || { echo "seed $SEED not in $V3D"; ok=1; }
  return $ok; }
stage() { # run as "( stage )": f4..63 + seed + image on both Sparks, arm recipe + arm dirs, thunderdome dry-run
  set -e
  mkdir -p "$K/seed" "$K/img"; rm -f "$K"/seed/*.json
  ssh -n $H2 "rm -rf $F463"; rm -rf "$F463"   # always from f4..03 and this run3 (a stale f4..63 would be judged as run3)
  python3 "$P/src/tools/mtp_refit/splice.py" "$F403" "$RF" "$F463" > "$RES/splice-dgx01.json"
  ssh -n $H2 "mkdir -p $W/runs/run3 $P/src/tools/mtp_refit"
  scp -q "$RF" "$H2:$RF"; scp -q "$P/src/tools/mtp_refit/splice.py" "$H2:$P/src/tools/mtp_refit/splice.py"
  ssh -n $H2 "python3 $P/src/tools/mtp_refit/splice.py $F403 $RF $F463" > "$RES/splice-dgx02.json"
  cmp <(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["shards"])' "$F463/SPLICE.json") \
      <(ssh -n $H2 "python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[\"shards\"])' $F463/SPLICE.json")
  docker run --rm --entrypoint cat "$V3D" /opt/b12x-seed/preparation/$SEED > "$K/seed-src.json"
  docker run --rm --entrypoint python3 -v "$K:/k" -v "$G/k48:/k48:ro" "$V3D" /k48/remap_seed.py /k/seed-src.json "$F463" /k/seed > "$RES/remap-seed.txt"
  [ "$(ls "$K"/seed/*.json | wc -l)" = 1 ]
  local tag; tag=k56c-refit-run3-$(sha256sum "$RF" | cut -c1-8); AIMG=spark-vllm-b12x:$tag
  { echo "# k56c (#97): shipped v3d warm image + the b12x seed re-keyed to the refit run3 snapshot path f4..63 (k48/remap_seed.py"
    echo "# from the image's v3d seed $SEED). Nothing else changes: the refit MTP tensors live in the snapshot."
    echo "FROM $V3D"; echo "COPY seed/*.json /opt/b12x-seed/preparation/"; } > "$K/img/Dockerfile"
  rm -rf "$K/img/seed"; cp -r "$K/seed" "$K/img/seed"
  docker build -q -t "$AIMG" "$K/img" > "$RES/build-dgx01.txt" 2>&1
  ssh -n $H2 "mkdir -p $K"; rsync -a --delete "$K/img/" "$H2:$K/img/"
  ssh -n $H2 "docker build -q -t $AIMG $K/img" > "$RES/build-dgx02.txt" 2>&1
  sed -e "1i # k56c arm (#97 refit run3, $(TZ=Europe/Bucharest date +%F)): the k56 run1 recipe with the run3 MTP dense tensors (snapshot f4..63 = f4..03 + spliced shard), image $AIMG (v3d + re-keyed seed)" \
      -e "s|^name: .*|name: qwen3.8-flash-next-1x-dgx-spark-k56c-run3|" -e "s|^container: .*|container: $AIMG|" \
      -e "s|^model: $F461\$|model: $F463|" -e "s|/cache/runtime/vllm-k56-refit\"|/cache/runtime/vllm-k56c-run3\"|" "$CTL" > "$K/k56c-run3.yaml"
  [ "$(diff "$CTL" "$K/k56c-run3.yaml" | grep -c '^>')" = 5 ] && grep -qx "model: $F463" "$K/k56c-run3.yaml" && grep -qx "container: $AIMG" "$K/k56c-run3.yaml" \
    && grep -q '/cache/runtime/vllm-k56c-run3"' "$K/k56c-run3.yaml" || { echo "k56c recipe edits did not apply"; exit 1; }
  for a in $ARMS; do mkdir -p "$K/$a"; cp "$K/k56c-run3.yaml" "$K/$a/k56c-run3.yaml"
    { echo "# k56c arm (#97 refit run3) on $([ $a = run3-01 ] && echo dgx-01 || echo dgx-02) vs refit run1 (shipped 2026-10-07): run3 MTP dense tensors"
      echo "# ($GPTS pts/pos offline T=0 over run1). Own VLLM_CACHE_ROOT + bake (new model path); seed re-keyed in the image."
      echo "name: $a"; echo "recipe: k56c-run3.yaml"; echo "base: $CTL"; echo "bake: yes"; echo "hook: $G/k53/hook.sh"; } > "$K/$a/thunderdome.arm"; done
  env RES="$RES/dry" GATE=0 CHAIN_NO_RESTORE=1 TD_ACC_RISE_OK=1 bash "$TD" "$K/run3-01" "$K/run3-02" --dry-run > "$RES/td-dry-run.txt" 2>&1
  if grep -q REFUSED "$RES/td-dry-run.txt"; then echo "thunderdome dry-run dropped an arm"; exit 1; fi
  echo "$AIMG" > "$K/IMAGE"; }

if [ "${1:-}" = --dry-run ]; then
  preflight; rc=$?
  s=$(head -1 "$BL/r3train/STATE" 2>/dev/null)
  if [ -s "$R3" ] && [ "${s%%:*}" = DONE ]; then GL=$(gain "$P/eval-refit-run1.json" "$R3")
    echo "run3 offline gain over run1: $(awk -v g=${GL%% *} 'BEGIN{printf "%+.2f", g*100}') pts/pos -> $(awk -v g=${GL%% *} -v m=$GAIN_MIN 'BEGIN{print (g >= m ? "screen" : "skip")}')"
  else echo "run3 not done yet (STATE '${s:-none}'): the job decides screen/skip when it runs"; fi
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "estimate on a qualifying gain: stage ~10 min, screen ~1.6 h (thunderdome plan: ~89 min incl. ~14 min bake, + hook), gate ~35 min on a PROMOTE, restore ~5 min"
  echo "dry-run exit=$rc"; exit $rc; fi
need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
mkdir -p "$RES"; echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
skip() { st "SKIPPED: $1"; { echo "k56c (run3 vs run1 live screen) did not run: $1."; echo; echo "The shipped single-Spark drafter stays refit run 1."; } > "$K/comment.md"; exit 0; }
s=$(head -1 "$BL/r3train/STATE" 2>/dev/null)
[ "${s%%:*}" = DONE ] && [ -s "$R3" ] && [ -r "$RF" ] || skip "refit run3 did not finish (r3train STATE '${s:-none}')"
GL=$(gain "$P/eval-refit-run1.json" "$R3") || skip "gain computation failed"
G0=${GL%% *}; GPTS=$(awk -v g=$G0 'BEGIN{printf "%+.2f", g*100}')
awk -v g=$G0 -v m=$GAIN_MIN 'BEGIN{exit !(g < m)}' && skip "run3 gain $GPTS pts vs run1 < +1.00 (offline T=0 mean per position, [${GL#* }])"

job_begin
echo "k56c $(TZ=Europe/Bucharest date '+%F %T %Z'): refit run3 ($(cat $W/WINNER 2>/dev/null)) vs refit run1 (shipped single-Spark drafter), offline T=0 gain $GPTS pts/pos [${GL#* }]" > "$RES/k56c.txt"
st "running: preflight"
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
st "running: stage"
( stage ) > "$RES/stage.txt" 2>&1; [ $? = 0 ] || { FINAL="FAILED: staging ($RES/stage.txt)"; exit 1; }
st "running: screen"
child env RES="$RES" GPU_LOCK_HELD=1 TD_ACC_RISE_OK=1 timeout -k 600 32400 bash "$TD" "$K/run3-01" "$K/run3-02" > "$RES/thunderdome.nohup" 2>&1 < /dev/null; rc=$?
V=$(head -1 "$RES/STATE" 2>/dev/null)
log "thunderdome exit=$rc STATE: $V"
{ echo "image $(cat "$K/IMAGE"), snapshot f4..63 on both Sparks, control $CIMG (f4..61)"; echo "thunderdome: exit $rc, $V"
  for a in $ARMS; do echo; echo "== $a ($([ $a = run3-01 ] && echo dgx-01 || echo dgx-02))"; sed -n '/^== acceptance/,$p' "$RES/$a/verdict.txt" 2>/dev/null | grep -vE '^(  self|  cross)'
    [ -f "$RES/$a/gate/summary.txt" ] && { echo "-- gate"; cat "$RES/$a/gate/summary.txt"; }; done; } >> "$RES/k56c.txt"
[ $rc = 0 ] || { FINAL="FAILED: thunderdome exit $rc ($V; $RES)"; exit 1; }
FINAL="DONE: k56c ${V#DONE: } ($RES/k56c.txt)"
{ echo "k56c result: refit run 3 ($(cat $W/WINNER 2>/dev/null), trained on the p2 data plus the r3 data) against refit run 1, the drafter I shipped on the single Spark, one arm per Spark, full Thunderdome screen with \`TD_ACC_RISE_OK=1\` and the split gate on a PROMOTE."
  echo; echo "Offline T=0 gain of run 3 over run 1: $GPTS pts per position. Screen: \`$V\`."
  echo; echo "This job publishes nothing: a PROMOTE with the gate passing goes to review before any upload."
  echo; echo '```'; sed -n '2,$p' "$RES/k56c.txt" | head -c 50000; echo '```'
  echo; echo "Raw data on dgx-01: \`$RES\`."; } > "$K/comment.md"
