#!/usr/bin/env bash
# k73 (2026-10-08, #123): GDN-MSE with the M-dispatch on the 2x (TP=2) vs the shipped b1.6.
#   arms     c26 / c41: GDN-MSE @ 16c9bd54 (retrained drafter included), GDN in_proj_qkvz/out_proj NVFP4 below the cutoff,
#            MXFP8 (7c4f1bc1, per-rank slices, vLLM d21d7ade) at or above it. c26 keeps c5 decode (25 rows) on NVFP4,
#            c41 keeps up to c8 (40 rows). Optional phase 2: the better cutoff + VLLM_GDN_COMPACT_RECORDS=1 +
#            VLLM_GDN_SHARED_PREFILL_STAGING=1 (c26x / c41x) against it.
#   plans    pinned (k70 lesson): the arm plan file starts from b1.6's 616 records re-keyed (stage.sh), the bake
#            measures only the arm-only GDN plans once, pin_shared.py re-checks every shared key against b1.6, and both
#            Sparks get the same file. Every screen boot must measure 0 plans (scripts/check_seed.py on the rank-0 log
#            + both plan files at the frozen record count), else the boot gets a COLD marker and does not count.
#   bake     per arm a cold and a warm boot (the MTP AOT key differs cold vs warm, hfship); then the warm image =
#            P1 + the frozen plan + every new torch AOT key of both nodes (the 2x image never carried AOT before),
#            built on both Sparks; the screen and the gate run on it.
#   screen   Thunderdome rules on the pair (k71): boots ctl, c26, c41, c41, c26, ctl; per arm thunderdome_report.py
#            on {ctl-p1, ctl-p2, arm-p1, arm-p2}; winner = best verdict (PROMOTE > INCONCLUSIVE > KILL), then the
#            higher mean cell delta. Acceptance is two-sided (no TD_ACC_RISE_OK: both sides run the same drafter).
#   phase 2  only when the winner is PROMOTE and K73_COMPACT != 0: bake c<cut>x, then boots W, X, X, W with W as control
#   gate     the final arm only when its verdict is PROMOTE: one TP2 boot, stragglers c5-c16, hardmode, TC-45 x5,
#            fidelity 8k-128k (seed 7) + 128k seeds 11/13 (k61.py gate rule)
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; comment.md + RESULT for the poster (#123). lib.sh restores the 2x.
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k73; K=$HOME/GEN-AI/backlog/k73; K70=$HOME/GEN-AI/backlog/k70
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k73-tp2-gdnmse-dispatch-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
CTL=qwen3.8-flash-next-2x-dgx-spark; MODEL=qwen3.8-flash-next
WARM=spark-vllm-b12x:k73-gdnmse-tp2-21e0b201-d21d7ade-warm
BASE=ghcr.io/ursuciprian/spark-vllm-b12x:tp1-v3e-hf-20261008-21e0b201-5dad364d-warm
RC=$HOME/.cache/sparkrun/runtime-cache/vllm
CTLF=$RC/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-2d9615ab/b12x/compile/preparation/15b639016c932b03d623d76be56a320a94946895d5522c30e3e1d8829572c729.json
SNAP=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
CORPUS=$R/results/corpus-code.txt; BENCHY_SRC=$G/llama-benchy-fork
GEN="List the numbers from 1 to 300 separated by commas. Output only the numbers, nothing else, no commentary."
# as scripts/thunderdome.sh: tag:depth:concurrency:reps:corpus offset:timeout s
CELLS="fresh-c1:256:1:2:104729:300 fresh-c4:256:4:2:314187:400 fresh-c8:256:8:2:418916:500
d16k-c4:16384:4:2:1361477:700 d16k-c8:16384:8:1:1466206:480 count-c8:0:8:2:0:300"

checks() {
  local ok=0 t
  bash "$K/stage.sh" || { echo "stage.sh failed"; ok=1; }
  python3 "$K70/k61.py" --selftest > /dev/null || { echo "k61.py selftest failed"; ok=1; }
  python3 "$K/pin_shared.py" --selftest > /dev/null || { echo "pin_shared.py selftest failed"; ok=1; }
  python3 "$R/scripts/check_seed.py" --selftest > /dev/null || { echo "check_seed.py selftest failed"; ok=1; }
  python3 "$R/scripts/thunderdome_report.py" --selftest > /dev/null 2>&1 || { echo "thunderdome_report.py selftest failed"; ok=1; }
  for t in sparkrun uvx tool-eval-bench python3 curl rsync; do command -v $t > /dev/null || { echo "missing $t"; ok=1; }; done
  for t in scripts/depth_decode_probe.py scripts/logits_equiv.py scripts/thunderdome_report.py scripts/paired_decode_ab.py \
           scripts/fidelity_probe.py scripts/straggler_probe.py scripts/check_seed.py results/corpus-code.txt; do [ -e "$R/$t" ] || { echo "missing $R/$t"; ok=1; }; done
  [ -d "$BENCHY_SRC" ] && [ -d "$SNAP" ] && [ -s "$CTLF" ] || { echo "missing llama-benchy fork, tokenizer snapshot or control plan file"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  rc=0; checks || rc=1
  others_busy && echo "other GPU jobs: busy (the runner keeps waiting)" || echo "other GPU jobs: clear"
  echo "arms $K/arm/k73-{c26,c41}.yaml (+ c26x/c41x in phase 2) on $(cat "$K/IMAGE" 2>/dev/null), then $WARM"
  echo "ctl  registry recipe $CTL (b1.6, $SHIPPED_TAG_DEFAULT)"
  echo "plan bake c26 + c41 (cold + warm each, ~35-45 min) -> warm image -> ctl, c26, c41, c41, c26, ctl (~22 min each)"
  echo "     -> [winner PROMOTE: bake c<cut>x ~20 min, W, X, X, W ~90 min] -> gate on PROMOTE (~60 min) -> restore (~5 min)"
  echo "estimate ~3 h 15 min without phase 2 and gate, ~5 h 45 min with both"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is queued or running: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; : > "$K/comment.md"; mkdir -p "$RES/recipes"
st "running: preflight"
K73_SYNC=1 checks > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: k73 preflight ($RES/preflight.txt)"; exit 1; }
P1=$(cat "$K/IMAGE"); SEED=$(cat "$K/SEED"); IMG=$P1; RC0=; RC1=
cp "$CTLF" "$RES/ctl-plan.json"
cp "$K"/stage.sh "$K"/run.sh "$K"/pin_shared.py "$K"/after_k56c.sh "$K"/arm/k73-*.yaml "$RES/recipes/"
{ echo "k73 $(TZ=Europe/Bucharest date '+%F %T %Z'): GDN-MSE @ 16c9bd54 + M-dispatch at TP=2 vs b1.6 (#123)"
  echo "vLLM d21d7ade (ursuciprian/vllm feat/k73-mxfp8-large-m-tp), b12x 21e0b201, base $BASE"
  grep -E '^(image|cputest|seed|recipes):|GDN-MSE @ 16c9bd54 complete' "$RES/preflight.txt"; } > "$RES/k73.txt"

# ---------------- helpers (k71 pattern + node 1 + plan checks) ----------------
mkrec() { # arm image -> recipe path with the container swapped
  local f=$RES/recipes/run-k73-$1.yaml; sed "s|^container: .*|container: $2|" "$K/arm/k73-$1.yaml" > "$f"; echo "$f"; }
relrc0() { docker inspect "$(n0)" --format '{{range .Mounts}}{{if eq .Destination "/cache/runtime"}}{{.Source}}{{end}}{{end}}' 2>/dev/null; }
relrc1() { ssh -n -o ConnectTimeout=10 $H2 'c=$(docker ps --format "{{.Names}}" | grep node_1 | head -1); [ -n "$c" ] && docker inspect "$c" --format "{{range .Mounts}}{{if eq .Destination \"/cache/runtime\"}}{{.Source}}{{end}}{{end}}"' 2>/dev/null; }
keep_logs() { # dir (before stop_all)
  local c; c=$(n0); [ -n "$c" ] || c=$(docker ps -a --format '{{.Names}}' | grep node_0 | head -1); [ -n "$c" ] || return 0
  docker exec "$c" cat /tmp/sparkrun_serve.log > "$1/serve.log" 2>&1 || docker logs "$c" > "$1/serve.log" 2>&1
  ssh -n $H2 'c=$(docker ps -a --format "{{.Names}}" | grep node_1 | head -1); [ -n "$c" ] && { docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1; }' > "$1/serve-node1.log" 2>&1
  grep -E "Directly load AOT|saved AOT|Dynamo bytecode|b12x .* ready|MXFP8 copy serves" "$1/serve.log" | cut -c1-300 > "$1/aot.txt"
  echo "copies node0 $(grep -c 'MXFP8 copy serves rows' "$1/serve.log") node1 $(grep -c 'MXFP8 copy serves rows' "$1/serve-node1.log")" > "$1/mxfp8-copies.txt"
  docker exec "$c" sh -c "env | grep -E '^(VLLM|B12X)_' | sort" > "$1/env.txt" 2>&1
  { echo "$(relrc0)"; echo "$(relrc1)"; } > "$1/runtime-cache.txt"
  { docker inspect --format '{{.Config.Image}}' "$c"; ssh -n $H2 "docker ps --format '{{.Image}}' --filter name=sparkrun"; } > "$1/image.txt" 2>&1; }
tboot() { # recipe dir cold_ok -> 0 up, 1 failed, 2 cold
  local rec=$1 d=$2 s m n
  stop_all; log "boot $(basename "$rec") -> $d"
  if [ "$rec" = "$CTL" ]; then ( cd "$R" && timeout -k 60 900 sparkrun run "$CTL" --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  else cp "$rec" "$d/recipe.yaml"; ( cd "$(dirname "$rec")" && timeout -k 60 900 sparkrun run "$(basename "$rec")" --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-; fi
  s=$(date +%s)
  while :; do
    n=$(n0)
    m=$([ -n "$n" ] && docker exec "$n" grep -m1 -oE 'Directly load AOT compilation|Dynamo bytecode transform time' /tmp/sparkrun_serve.log 2>/dev/null)
    [ "$m" = "Dynamo bytecode transform time" ] && [ "$3" != 1 ] && { log "$(basename "$d"): COLD (backbone compiling), stopped"; return 2; }
    [ "$(health localhost)" = 200 ] && { echo "boot $(( $(date +%s) - s ))s ${m:-no AOT line}" > "$d/boot.txt"; log "$(basename "$d"): up, $(cat "$d/boot.txt")"; return 0; }
    if [ $(( $(date +%s) - s )) -gt 180 ] && { [ -z "$n" ] || docker exec "$n" grep -qE 'Worker failed with error|EngineCore failed to start|Engine core initialization failed' /tmp/sparkrun_serve.log 2>/dev/null; }; then
      log "$(basename "$d"): boot FAILED after $(( $(date +%s) - s ))s"; return 1; fi
    [ $(( $(date +%s) - s )) -ge $([ "$3" = 1 ] && echo 5400 || echo 1800) ] && { log "$(basename "$d"): health timeout"; return 1; }
    sleep 10; done; }
plans_ok() { # dir plan-file count -> check_seed.py on the rank-0 log and both nodes' plan file (written to the dir)
  local f=$2 n=$3 r0 r1; r0=$(sed -n 1p "$1/runtime-cache.txt"); r1=$(sed -n 2p "$1/runtime-cache.txt")
  scp -q "$H2:$r1/b12x/compile/preparation/$f" "$1/plan-node1.json" 2>/dev/null || echo '{"records": {}}' > "$1/plan-node1.json"
  python3 "$R/scripts/check_seed.py" "$1/serve.log" "$r0/b12x/compile/preparation/$f:$n" "$1/plan-node1.json:$n" > "$1/plans-check.txt" 2>&1; }
pos() { curl -s -m 10 "localhost:8000/metrics" | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens(_per_pos)?_total'; }
probe() { # dir tag depth conc reps offset timeout (as thunderdome.sh, against the TP2 head)
  local d=$1 tag=$2 depth=$3 c=$4 r=$5 off=$6 t=$7 rc s
  if [ "${tag#count}" != "$tag" ]; then set -- --prompt "$GEN" --max-tokens 320
  else set -- --depth "$depth" --new 2048 --offset "$off" --max-tokens 512; fi
  pos > "$d/pos-$tag.before"; s=$(date +%s)
  ( cd "$R" && child timeout -k 30 "$t" python3 scripts/depth_decode_probe.py --base "http://$H1:8000" --label "$tag" --corpus "$CORPUS" \
      --concurrency "$c" --repeats "$r" --temperature 0 --out "$d/probe-$tag.json" "$@" > "$d/probe-$tag.log" 2>&1 ); rc=$?
  echo "$rc $(( $(date +%s) - s )) $t" > "$d/time-$tag.txt"; pos > "$d/pos-$tag.after"
  [ $rc -eq 0 ] || rm -f "$d/probe-$tag.json"; log "$(basename "$d"): $tag exit=$rc"; }
run_pass() { # dir
  local d=$1 spec tag depth c r off t x
  probe "$d" warm 4096 1 1 900000 300
  for spec in $CELLS; do IFS=: read -r tag depth c r off t <<< "$spec"; probe "$d" "$tag" "$depth" "$c" "$r" "$off" "$t"; done
  for x in a b; do ( cd "$R" && timeout 300 python3 scripts/logits_equiv.py capture --base-url "http://$H1:8000" --out "$d/logits-$x.json" > "$d/logits-$x.log" 2>&1 ); done
  ( cd "$R" && child timeout -k 30 1200 uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$H1:8000/v1" --model $MODEL \
      --tokenizer "$SNAP" --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 --concurrency 1 4 8 --runs 4 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$H1:8000/metrics" \
      --save-result "$d/task.csv" > "$d/benchy.log" 2>&1 ); log "$(basename "$d"): benchy exit=$?"
  [ "$(health localhost)" = 200 ] || { echo "server gone after the pass" > "$d/DIED"; log "$(basename "$d"): server DIED"; }; }
screen_boot() { # dir who(ctl|arm name) -> boots, checks plans, runs the pass; 1 when the boot failed
  local d=$1 who=$2 rc
  rm -rf "$d"; mkdir -p "$d"; st "running: screen $(basename "$d")"; reset_plans
  if [ "$who" = ctl ]; then tboot "$CTL" "$d" 0; else tboot "$(mkrec "$who" "$IMG")" "$d" 1; fi; rc=$?
  if [ $rc -ne 0 ]; then keep_logs "$d"; [ $rc -eq 2 ] && echo cold > "$d/COLD" || echo failed > "$d/FAILED"; stop_all; return 1; fi
  keep_logs "$d"
  if [ "$who" = ctl ]; then
    if ! plans_ok "$d" "$(basename "$CTLF")" 616; then echo "measured plans: $(tail -2 "$d/plans-check.txt" | tr '\n' ' ')" > "$d/COLD"
      log "$(basename "$d"): control measured plans, pass does not count"; stop_all; return 0; fi
  elif ! plans_ok "$d" "$SEED" "$NPLAN"; then echo "measured plans: $(tail -2 "$d/plans-check.txt" | tr '\n' ' ')" > "$d/COLD"
    log "$(basename "$d"): plans not pinned (measured), pass does not count"; stop_all; return 0; fi
  run_pass "$d"; keep_logs "$d"; stop_all; return 0; }
reset_plans() { # put the frozen arm plan file and the control's original plan file back on both nodes (a boot that
  # measured would otherwise spoil every later boot: one runtime cache dir per model@revision serves all arms)
  [ -n "${RC0:-}" ] && [ -s "$RES/img-warm/preparation/$SEED" ] && { cp "$RES/img-warm/preparation/$SEED" "$RC0/b12x/compile/preparation/$SEED"
    scp -q "$RES/img-warm/preparation/$SEED" "$H2:$RC1/b12x/compile/preparation/$SEED" || log "reset_plans: arm plan copy to node 1 failed"; }
  cp "$RES/ctl-plan.json" "$CTLF"; scp -q "$RES/ctl-plan.json" "$H2:$CTLF" || log "reset_plans: control plan copy to node 1 failed"; }
mkrep() { # report dir, then pass-K boot dirs as ctl-pK arm-pK pairs (symlinks)
  local o=$1 k=1; shift; rm -rf "$o"; mkdir -p "$o"
  while [ $# -ge 2 ]; do [ -e "$1" ] && ln -s "$1" "$o/ctl-p$k"; [ -e "$2" ] && ln -s "$2" "$o/arm-p$k"; k=$((k + 1)); shift 2; done; }
report() { # report dir ctl-p1 arm-p1 ctl-p2 arm-p2 -> verdict.txt, prints VERDICT
  local o=$1 v; mkrep "$@"
  python3 "$R/scripts/thunderdome_report.py" "$o" > "$o/verdict.txt" 2>&1; v=$(sed -n 's/^VERDICT=//p' "$o/verdict.txt"); echo "${v:-ERROR}"; }
score() { # verdict.txt -> "<rank> <mean cell delta>" (PROMOTE 2, INCONCLUSIVE 1, else 0)
  python3 - "$1" <<'PY'
import re, sys
t = open(sys.argv[1]).read(); v = (re.findall(r"^VERDICT=(\w+)", t, re.M) or ["ERROR"])[-1]
d = [float(x) for x in re.findall(r"^  (?:probe|wall|benchy)\s+.+?\s([+-]\d+\.\d+)%\s+noise", t, re.M)]
print({"PROMOTE": 2, "INCONCLUSIVE": 1}.get(v, 0), round(sum(d) / len(d), 3) if d else -999)
PY
}

# ---------------- bake: cold + warm boot per arm, pinned plan on both nodes, warm image ----------------
NPLAN=0; AOTDIR=$RES/img-warm/torch_aot_compile; mkdir -p "$AOTDIR" "$RES/img-warm/preparation"
docker run --rm --network none --entrypoint ls "$BASE" /opt/b12x-seed/torch_compile_cache/torch_aot_compile > "$RES/base-aot-keys.txt" 2>/dev/null
aotkeys() { grep -ohE '(saved AOT compiled function to|Directly load AOT compilation from path) \S*torch_aot_compile/[0-9a-f]{64}' "$@" 2>/dev/null | sed 's|.*/||' | sort -u; }
newaot() { # host runtime-cache epoch log -> AOT keys saved by the boot (log lines + dirs created since epoch), base image keys excluded
  { aotkeys "$4"; ssh -n -o ConnectTimeout=10 $1 "find $2/vllm/torch_compile_cache/torch_aot_compile -mindepth 1 -maxdepth 1 -type d -newermt @$3 -printf '%f\n'" 2>/dev/null; } \
    | grep -E '^[0-9a-f]{64}$' | sort -u | grep -vxFf "$RES/base-aot-keys.txt"; }
bake() { # arm -> 0 ok
  local a=$1 p d r0 r1 k n=0
  for p in cold warm; do
    d=$RES/bake/$a-$p; rm -rf "$d"; mkdir -p "$d"; st "running: bake $a $p"; t0=$(( $(date +%s) - 5 ))
    tboot "$(mkrec "$a" "$IMG")" "$d" 1 || { keep_logs "$d"; stop_all; log "bake $a $p: boot failed"; return 1; }
    keep_logs "$d"; stop_all
    r0=$(sed -n 1p "$d/runtime-cache.txt"); r1=$(sed -n 2p "$d/runtime-cache.txt")
    [ -n "$r0" ] && [ -n "$r1" ] && [ -s "$r0/b12x/compile/preparation/$SEED" ] || { log "bake $a $p: no runtime cache / plan file ($r0 | $r1)"; return 1; }
    if [ $p = cold ]; then
      grep -q 'MXFP8 copy serves rows' "$d/serve.log" || { log "bake $a: no 'MXFP8 copy serves rows' line in the rank-0 log, dispatch not engaged"; return 1; }
      python3 "$K/pin_shared.py" "$r0/b12x/compile/preparation/$SEED" "$RES/ctl-plan.json" > "$d/pin.txt" || { log "bake $a: pin_shared.py failed"; return 1; }
      sed "s/^/bake $a: /" "$d/pin.txt" >> "$RES/k73.txt"
      python3 "$R/scripts/check_seed.py" "$d/serve.log" > "$d/plans-cold.txt" 2>&1; sed -n 1p "$d/plans-cold.txt" | sed "s/^/bake $a cold: /" >> "$RES/k73.txt"
      scp -q "$r0/b12x/compile/preparation/$SEED" "$H2:$r1/b12x/compile/preparation/$SEED" || { log "bake $a: plan copy to node 1 failed"; return 1; }
      NPLAN=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['records']))" "$r0/b12x/compile/preparation/$SEED")
      cp "$r0/b12x/compile/preparation/$SEED" "$RES/img-warm/preparation/$SEED"; chmod a+r "$RES/img-warm/preparation/$SEED"; RC0=$r0; RC1=$r1
      # b12x 21e0b201 must reuse b1.6's (b7fbaf96) request keys: if the digests differed, the cold boot measured every
      # plan anew and the arm-only count jumps to ~600 instead of the GDN plans only
      [ $((NPLAN - 616)) -le 200 ] || { log "bake $a: $((NPLAN - 616)) arm-only plans (> 200): b1.6's keys were not reused, pinning impossible"; return 1; }
    else
      plans_ok "$d" "$SEED" "$NPLAN" || { log "bake $a warm: plans still measured after the pin: $(tail -2 "$d/plans-check.txt" | tr '\n' ' ')"; return 1; }
    fi
    # new AOT keys of this boot on each node (log lines, base image keys excluded)
    for k in $(newaot localhost "$r0" $t0 "$d/serve.log"); do
      mkdir -p "$AOTDIR/$k"; cp -a "$r0/vllm/torch_compile_cache/torch_aot_compile/$k/." "$AOTDIR/$k/" && n=$((n + 1)); done
    for k in $(newaot $H2 "$r1" $t0 "$d/serve-node1.log"); do
      mkdir -p "$AOTDIR/$k"; rsync -a "$H2:$r1/vllm/torch_compile_cache/torch_aot_compile/$k/" "$AOTDIR/$k/" && n=$((n + 1)); done
    log "bake $a $p: $(cat "$d/boot.txt"), $(cat "$d/mxfp8-copies.txt"), plan $NPLAN records"
  done
  echo "bake $a: cold $(cut -d' ' -f2 "$RES/bake/$a-cold/boot.txt"), warm $(cut -d' ' -f2 "$RES/bake/$a-warm/boot.txt"), $(cat "$RES/bake/$a-warm/mxfp8-copies.txt"), $n new AOT dirs, plans $(tail -1 "$RES/bake/$a-warm/plans-check.txt")" >> "$RES/k73.txt"
  [ "$(grep -c . <(ls "$AOTDIR"))" -gt 0 ] || { log "bake $a: no AOT keys found in the logs"; return 1; }; }
warm_image() {
  local k bad=
  for k in "$AOTDIR"/*; do [ -d "$k/rank_0_0" ] && [ -d "$k/rank_1_0" ] || bad="$bad $(basename "$k")"; done
  [ -z "$bad" ] || { log "warm image: AOT keys without both rank_0_0 and rank_1_0:$bad"; return 1; }
  chmod -R a+rX "$RES/img-warm"
  cat > "$RES/img-warm/Dockerfile" <<EOF
# k73 (#123): $P1 + the frozen TP=2 plan file ($SEED, $NPLAN records: b1.6 records for every shared key + the bake's
# GDN plans) and the torch AOT compile cache of the bake boots on both nodes ($(ls "$AOTDIR" | wc -l) keys).
FROM $P1
COPY preparation/ /opt/b12x-seed/preparation/
COPY torch_aot_compile/ /opt/b12x-seed/torch_compile_cache/torch_aot_compile/
EOF
  child nice -n 19 docker build -q -t "$WARM" "$RES/img-warm" >> "$RES/$J.log" 2>&1 9>&- || return 1
  # one build, loaded on node 1: same image Id on both, so sparkrun does not ship it inside the 900 s boot timeout
  docker save "$WARM" | ssh $H2 "docker load -q" >> "$RES/$J.log" 2>&1 || return 1
  [ "$(docker image inspect -f '{{.Id}}' "$WARM")" = "$(ssh -n $H2 "docker image inspect -f '{{.Id}}' $WARM")" ] || { log "warm image Id differs on node 1"; return 1; }
  IMG=$WARM; log "warm image $WARM: $(ls "$AOTDIR" | wc -l) AOT keys, plan $NPLAN records (both Sparks)"; }

st "running: stop 2x"; stop_all
for a in c26 c41; do bake $a || { FINAL="FAILED: k73 bake $a ($RES/bake)"; exit 1; }; done
warm_image || { FINAL="FAILED: k73 warm image build"; exit 1; }
echo "warm image $WARM: $(ls "$AOTDIR" | wc -l) AOT keys, plan $SEED $NPLAN records" >> "$RES/k73.txt"

# ---------------- screen: ctl, c26, c41, c41, c26, ctl ----------------
S=$RES/screen; mkdir -p "$S"; ARMS="c26 c41"
screen_boot "$S/ctl-p1" ctl
for a in $ARMS; do screen_boot "$S/$a-p1" $a; done
for a in c26 c41; do  # acceptance after pass 1: an arm already past 0.03 skips pass 2
  mkrep "$S/rep-$a-p1" "$S/ctl-p1" "$S/$a-p1"
  python3 "$R/scripts/thunderdome_report.py" "$S/rep-$a-p1" --acc-only > "$S/acc-$a-p1.txt" 2>&1
  [ $? -eq 3 ] && { ARMS=$(echo $ARMS | tr ' ' '\n' | grep -vx $a | tr '\n' ' '); log "$a: acceptance moved past 0.03 in pass 1, pass 2 skipped"; }; rm -rf "$S/rep-$a-p1"; done
for a in $(echo $ARMS | tr ' ' '\n' | tac); do screen_boot "$S/$a-p2" $a; done
screen_boot "$S/ctl-p2" ctl
best=; bs="-1 -999"
for a in c26 c41; do
  v=$(report "$S/rep-$a" "$S/ctl-p1" "$S/$a-p1" "$S/ctl-p2" "$S/$a-p2"); sc=$(score "$S/rep-$a/verdict.txt"); eval "V_$a=$v"
  { echo; echo "== screen $a vs b1.6 (Thunderdome rules on the pair): VERDICT=$v (score $sc)"; cat "$S/rep-$a/verdict.txt"; } >> "$RES/k73.txt"
  log "screen $a: $v ($sc)"
  if python3 -c "import sys; a=list(map(float,sys.argv[1].split())); b=list(map(float,sys.argv[2].split())); sys.exit(0 if a > b else 1)" "$sc" "$bs"; then best=$a; bs=$sc; fi
done
W=${best:-c26}; eval "VW=\$V_$W"; FIN=$W; VF=$VW
echo "winner of the cutoff screen: $W ($VW)" >> "$RES/k73.txt"

# ---------------- phase 2: compact records + shared prefill staging on the winner ----------------
VX="not run"
if [ "$VW" = PROMOTE ] && [ "${K73_COMPACT:-1}" != 0 ]; then
  X=${W}x; S2=$RES/compact; mkdir -p "$S2"
  if bake "$X" && warm_image; then
    screen_boot "$S2/w-p1" "$W"; screen_boot "$S2/x-p1" "$X"; screen_boot "$S2/x-p2" "$X"; screen_boot "$S2/w-p2" "$W"
    VX=$(report "$S2/rep" "$S2/w-p1" "$S2/x-p1" "$S2/w-p2" "$S2/x-p2")
    { echo; echo "== phase 2: $X vs $W (control = $W): VERDICT=$VX"; cat "$S2/rep/verdict.txt"; } >> "$RES/k73.txt"
    [ "$VX" = PROMOTE ] && { FIN=$X; VF=PROMOTE; }
  else VX="FAILED (bake/image)"; echo "== phase 2: $X bake or image failed" >> "$RES/k73.txt"; fi
  log "phase 2 $X vs $W: $VX"; fi
echo "final arm: $FIN" >> "$RES/k73.txt"

# ---------------- gate on the final arm ----------------
gate() { # arm -> sets GV
  local arm=$1 g=$RES/gate-$1 t
  st "running: gate $arm"; rm -rf "$g"; mkdir -p "$g"; reset_plans
  if tboot "$(mkrec "$arm" "$IMG")" "$g" 1; then
    keep_logs "$g"; plans_ok "$g" "$SEED" "$NPLAN" || log "gate boot measured plans (plans-check.txt)"
    ( cd "$R" && child timeout 900 python3 scripts/straggler_probe.py 5 6 7 8 12 16 > "$g/straggler.log" 2>&1 )
    child timeout 7200 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://localhost:8000 --model $MODEL > "$g/hardmode.log" 2>&1
    child timeout 3600 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://localhost:8000/v1 --model $MODEL --scenarios TC-45 --trials 5 > "$g/tc45.log" 2>&1
    ( cd "$R" && child timeout 5400 python3 scripts/fidelity_probe.py --base http://localhost:8000 --model $MODEL --depths 8000,32000,64000,128000 --out "$g/fidelity.json" > "$g/fidelity_probe.txt" 2>&1 )
    for sd in 11 13; do ( cd "$R" && child timeout 2700 python3 scripts/fidelity_probe.py --base http://localhost:8000 --model $MODEL --depths 128000 --seed $sd --out "$g/fidelity-seed$sd.json" > "$g/fidelity-seed$sd.txt" 2>&1 ); done
    keep_logs "$g"; stop_all
    t=$(mktemp -d); ln -s "$g" "$t/cand1"; ln -s "$g" "$t/cand2"
    GV=$(python3 -c "import sys; sys.path.insert(0, sys.argv[1]); import k61; b, hm = k61.gate(sys.argv[2]); print('PASS' if not b else 'FAIL (' + '; '.join(b) + ')')" "$K70" "$t" 2>&1); rm -rf "$t"
    { echo; echo "== gate ($arm, one TP2 boot): $GV"; grep -hE "Quality:" "$g/hardmode.log"; grep -hE "Pass\^5|Score:" "$g/tc45.log" | head -2
      grep -h "^depth" "$g/fidelity_probe.txt" "$g"/fidelity-seed1?.txt; tail -6 "$g/straggler.log"; } >> "$RES/k73.txt"
  else keep_logs "$g"; stop_all; GV="FAIL (gate boot failed)"; echo "== gate ($arm): boot failed" >> "$RES/k73.txt"; fi
  log "gate $arm: $GV"; }
GV="not run (screen $VW)"
if [ "$VF" = PROMOTE ]; then
  gate "$FIN"
  # the compact arm failed the gate: the cutoff winner already beat b1.6, so it gets its own gate
  if [ "$FIN" != "$W" ] && [ "${GV%% *}" != PASS ]; then echo "$FIN failed the gate ($GV), gating $W" >> "$RES/k73.txt"; FIN=$W; gate "$W"; fi
fi
for l in dgx01 dgx02; do echo "guard $l: min MemAvailable $(awk '$2 ~ /^[0-9]+$/' "$RES/guard-$l.log" | sort -k2 -n | head -1 | awk '{printf "%.2f GiB", $2/1024}')"; done >> "$RES/k73.txt"
FINAL="DONE: k73 TP2 GDN-MSE dispatch c26=$V_c26 c41=$V_c41 compact=$VX final=$FIN gate=${GV%% (*} ($RES/k73.txt)"
cells() { sed -n '/^== cells/,/^==/p' "$1" | grep -E '^ +(probe|wall|benchy) '; }
acc() { sed -n '/^== acceptance/,/^==/p' "$1" | grep -E '^ +pos '; }
{ echo "k73 result: GDN-MSE @ 16c9bd54 with the M-dispatch on the 2x (vLLM d21d7ade, per-rank MXFP8 copies), cutoffs 26 and 41 against the shipped b1.6. Boots b1.6, c26, c41, c41, c26, b1.6 on the pair, Thunderdome rules (noise = control boot-to-boot, 1% floor), plans pinned to b1.6's selections for every shared key (0 measured on every counted boot), warm image $WARM with the torch AOT cache of the bake boots."
  for a in c26 c41; do echo; eval "echo \"Cutoff ${a#c} vs b1.6: \$V_$a\""; echo '```'; acc "$S/rep-$a/verdict.txt"; cells "$S/rep-$a/verdict.txt"; echo '```'; done
  [ "$VX" = "not run" ] || { echo; echo "Phase 2, ${W}x (compact GDN records + shared prefill staging) vs $W: $VX"; [ -s "$RES/compact/rep/verdict.txt" ] && { echo '```'; cells "$RES/compact/rep/verdict.txt"; echo '```'; }; }
  echo; echo "Final arm: $FIN. Gate: $GV."
  echo; case "$VF:${GV%% *}" in
    PROMOTE:PASS) echo "The dispatch wins on the 2x and $FIN passed the TP=2 gate, including 128k seeds 7/11/13. I plan to ship it as the next 2x build." ;;
    PROMOTE:*) echo "$FIN wins on speed but the gate did not pass ($GV), so the 2x stays on b1.6." ;;
    *) echo "No cutoff beat b1.6 under these rules, so the 2x stays on b1.6 and the gate was not run." ;; esac
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
exit 0
