#!/usr/bin/env bash
# k92-2x-d3-drafter-ab (2026-10-12, tp-2 #137): MTP drafter D3 on the 2x against the shipped 2x v2.0.0 (D1).
# D3 is ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE @ 03f4a057, which differs from 16c9bd54 (v2.0.0) only in the 26
# dense mtp.* tensors of model-00034. No new kernels, no new image: the arm is the shipped recipe with the revision bumped.
#   recipes  control = the shipped 2x recipe from the sparkrun registry clone as is; arm = the same file with
#            model_revision and --revision set to 03f4a057, its own name, and its own runtime cache (XDG_CACHE_HOME +
#            VLLM_CACHE_ROOT under /cache/runtime/k92-d3): sparkrun keys the runtime cache by model@revision, and
#            03f4a057 maps to the 1x v2.2.0 cache, which this job must not touch.
#   stage    (run only; CPU) on both Sparks: the arm cache dir gets a no-clobber copy of the control cache's b12x,
#            triton, inductor, flashinfer, cute_dsl, torch_extensions and numba dirs, plus the control's TP=2 plan
#            6fdfaa4c (682 records, identical on both Sparks) re-keyed to the 03f4a057 snapshot path (k48/remap_seed.py
#            in the shipped image), so the arm runs the control's plan selections and the A/B measures the drafter.
#   bake     the torch AOT key holds the model path, so the arm compiles once: prebake with a 6 GiB KV pool (<= 3 tries,
#            memory guard), then one full-pool arm boot. After the bake: no plan file other than the re-keyed one may
#            appear in the arm cache (k79 identity check) and the re-keyed plan must still hold 682 records on both ranks.
#   screen   TP=2 on the pair, boots ctl arm arm ctl arm ctl ctl arm (4 per side): warm-up, T=0 probes fresh c1/c4/c8 +
#            d16k-c4 (MTP acceptance per position), logits x2, llama-benchy pp2048 + tg512 c1/c4/c8/c16 (4 runs).
#            Every counted boot: warm (AOT loaded), 0 b12x plans measured on rank 0 and 682 records on both ranks.
#   judge    job-local thunderdome_report.py (probe cells present here, no wall cell, benchy c1..c16) with
#            TD_ACC_RISE_OK=1 (drafter arm) + pool.py post checks + KV pool within 3% of the control's + boot-level Welch
#            CI. RULE=PASS (>= 1 cell better beyond noise, none worse) | FLAT | FAIL.
#   gate     only on RULE=PASS (#137 win rule): one arm boot, stragglers c5-c16, tool-use hardmode, TC-45 x5, fidelity
#            8k-128k + 128k seeds 11/13 (k61.gate: hardmode >= 88, fidelity 20/20 everywhere, TC-45 100, 0 preemptions).
# Nothing ships from this job; a pass is the evidence for a 2x v2.1.0 (MINOR, drafter only), decided afterwards.
# ETA ~4 h 30 min with the gate (stage ~3 min, bake ~25-45 min, 8 boots x ~20 min, gate ~75 min) + restore ~6 min.
# Contract (backlog/runner.sh): caller holds both GPU locks; exit 75 untouched while another job is queued/running;
# STATE first line DONE/FAILED; comment.md + RESULT for the poster; the shipped 2x is restored on exit (lib.sh).
# Usage: bash run.sh --dry-run | through the runner (JOBS=k92). Stop: kill -TERM <pid>.
NEEDS="dgx01 dgx02"
set -u
J=k92; K=$HOME/GEN-AI/backlog/k92; K70=$HOME/GEN-AI/backlog/k70; K83=$HOME/GEN-AI/backlog/k83
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k92-2x-d3-drafter-ab-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
CREV=16c9bd54788d12390838a65ce4a4ecda97fa5f1d; AREV=03f4a0570496bbe741189f4a2a7d611679e85406
ACACHE=/cache/runtime/k92-d3; KV_PRE=6442450944; NREC=682
RCV=$HOME/.cache/sparkrun/runtime-cache/vllm
CRC=$RCV/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-22c0994b     # model_key of the control (@16c9bd54)
ARC=$RCV/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923/k92-d3   # arm cache on the host (/cache/runtime = ..-5d63a923)
CPLAN=6fdfaa4cdd6c827f17390e8fa20bc97a36d7be3200a2d97ceb73f7b0577c050e.json
APATH=/cache/huggingface/hub/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE/snapshots/$AREV
MODEL=qwen3.8-flash-next; GATE=${GATE:-1}
REGS=$HOME/.cache/sparkrun/registries; REG2=qwen38-flashnext; N2=qwen3.8-flash-next-2x-dgx-spark
HF=$HOME/.cache/huggingface/hub
SNAP=$HF/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
CORPUS=$R/results/corpus-code.txt; BENCHY_SRC=$G/llama-benchy-fork
GEN="List the numbers from 1 to 300 separated by commas. Output only the numbers, nothing else, no commentary."
CELLS="fresh-c1:256:1:2:104729:300 fresh-c4:256:4:2:314187:400 fresh-c8:256:8:2:418916:500 d16k-c4:16384:4:2:1361477:700"
ORDER="ctl:1 arm:1 arm:2 ctl:2 arm:3 ctl:3 ctl:4 arm:4"
on2() { ssh -n -o ConnectTimeout=10 $H2 "$@"; }
hx() { if [ "$1" = localhost ]; then bash -c "$2"; else on2 "$2"; fi; }
recf() { grep -rlx --include='*.yaml' "name: $2" "$(readlink -f "$REGS/$1")/recipes" 2>/dev/null | head -1; }
img() { sed -n 's/^container: *//p' "$1" | head -1 | tr -d "\"' "; }
fld() { sed -n "s/^$2: *//p" "$1" | head -1 | tr -d "\"' "; }
# The arm runs the shipped image, so lib.sh's is_shipped alone would take a running arm for the shipped 2x and the
# EXIT-trap restore would leave D3 serving. Here the shipped 2x must also serve revision 16c9bd54 on rank 0.
is_shipped() { [ "$(health localhost)" = 200 ] && docker ps --format '{{.Image}}' | grep -q ":$E\$" \
  && ssh -n -o ConnectTimeout=10 $H2 "docker ps --format '{{.Image}}'" | grep -q ":$E\$" \
  && docker exec "$(n0)" sh -c "ps aux | grep '[v]llm serve'" | grep -- '--tensor-parallel-size 2' | grep -q -- "--revision $CREV"; }

mkarm() { # ctl.yaml arm.yaml: revision x2, name, own cache; exactly 3 lines changed, 3 added
  python3 - "$1" "$2" "$CREV" "$AREV" "$ACACHE" <<'PY'
import sys
from collections import Counter
src, dst, crev, arev, cache = sys.argv[1:6]
lines = open(src).read().split("\n")
def one(pred, what):
    i = [k for k, l in enumerate(lines) if pred(l)]
    if len(i) != 1:
        sys.exit(f"mkarm: need exactly one {what}, found {len(i)}")
    return i[0]
out = lines[:]
r = one(lambda l: l == f"model_revision: {crev}", "model_revision line")
v = one(lambda l: l.strip() == f"--revision {crev} \\", "--revision line")
n = one(lambda l: l.startswith("name: "), "name line")
e = one(lambda l: l == "env:", "env: block")
if any(l.strip().startswith(("XDG_CACHE_HOME:", "VLLM_CACHE_ROOT:")) for l in lines):
    sys.exit("mkarm: the control already sets a cache env")
out[r] = f"model_revision: {arev}"
out[v] = out[v].replace(crev, arev)
out[n] = "name: qwen3.8-flash-next-2x-k92-d3"
out[e + 1:e + 1] = ["  # k92 arm: own runtime cache (03f4a057 keys to the 1x v2.2.0 cache dir; keep that one untouched)",
                    f'  XDG_CACHE_HOME: "{cache}"', f'  VLLM_CACHE_ROOT: "{cache}/vllm"']
a, b = Counter(lines), Counter(out)
if sum((a - b).values()) != 3 or sum((b - a).values()) != 6 or crev in "\n".join(out):
    sys.exit("mkarm: unexpected diff")
open(dst, "w").write("\n".join(out))
PY
}
mkrecipes() { # dir -> ctl-2x.yaml + arm-2x.yaml + prebake-2x.yaml
  local d=$1 f2; mkdir -p "$d"; f2=$(recf $REG2 $N2)
  [ -s "$f2" ] || { echo "2x recipe not resolvable from the registry ($REG2/$N2)"; return 1; }
  cp "$f2" "$d/ctl-2x.yaml"
  [ "$(img "$d/ctl-2x.yaml" | sed 's|.*:||')" = "$SHIPPED_TAG_DEFAULT" ] \
    || { echo "registry 2x tag $(img "$d/ctl-2x.yaml" | sed 's|.*:||') != lib.sh SHIPPED_TAG_DEFAULT $SHIPPED_TAG_DEFAULT"; return 1; }
  [ "$(fld "$d/ctl-2x.yaml" model_revision)" = "$CREV" ] || { echo "registry 2x revision is not $CREV"; return 1; }
  mkarm "$d/ctl-2x.yaml" "$d/arm-2x.yaml" || return 1
  sed -e 's|^    --gpu-memory-utilization {gpu_memory_utilization}$|    --gpu-memory-utilization {gpu_memory_utilization} \\\n    --kv-cache-memory-bytes '"$KV_PRE"'|' "$d/arm-2x.yaml" > "$d/prebake-2x.yaml"
  [ "$(diff "$d/arm-2x.yaml" "$d/prebake-2x.yaml" | grep -c '^>')" = 2 ] && grep -qx "    --kv-cache-memory-bytes $KV_PRE" "$d/prebake-2x.yaml" \
    || { echo "prebake-2x.yaml edit did not apply"; return 1; }
  echo "2x $f2 @ $(git -C "$(readlink -f "$REGS/$REG2")" rev-parse --short HEAD) $(head -1 "$f2")" > "$d/sources.txt"; }
mkreport() { # job-local report: the probes of this job, no wall cell, benchy c1..c16
  sed -e 's/^PROBES = ("fresh-c1", "fresh-c4", "fresh-c8", "d16k-c4", "count-c8")$/PROBES = ("fresh-c1", "fresh-c4", "fresh-c8", "d16k-c4")/' \
      -e 's/^WALL = ("d16k-c8",)$/WALL = ()/' \
      -e 's/^BENCHY = ("pp2048 (c1)", "tg512 (c1)", "tg512 (c8)")$/BENCHY = ("pp2048 (c1)", "tg512 (c1)", "tg512 (c4)", "tg512 (c8)", "tg512 (c16)")/' \
      "$R/scripts/thunderdome_report.py" > "$K/td_report_k92.py"
  [ "$(diff "$R/scripts/thunderdome_report.py" "$K/td_report_k92.py" | grep -c '^>')" = 3 ] && python3 -m py_compile "$K/td_report_k92.py" \
    || { echo "td_report_k92.py edits did not apply"; return 1; }
  ln -sf "$R/scripts/paired_decode_ab.py" "$K/paired_decode_ab.py"; }
nrec() { python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["records"]))' "$1" 2>/dev/null || echo 0; }
preflight() {
  local ok=0 h f t mdir
  for f in "$K/pool.py" "$G/k48/remap_seed.py" "$R/scripts/thunderdome_report.py" "$R/scripts/depth_decode_probe.py" "$R/scripts/logits_equiv.py" \
           "$R/scripts/paired_decode_ab.py" "$R/scripts/straggler_probe.py" "$R/scripts/fidelity_probe.py" "$K70/k61.py" "$CORPUS"; do
    [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  [ -d "$BENCHY_SRC" ] && [ -d "$SNAP" ] || { echo "missing llama-benchy fork or tokenizer snapshot"; ok=1; }
  for t in sparkrun uvx tool-eval-bench python3 curl; do command -v $t > /dev/null || { echo "missing $t"; ok=1; }; done
  python3 "$K/pool.py" --selftest > /dev/null || { echo "pool.py selftest failed"; ok=1; }
  python3 "$K70/k61.py" --selftest > /dev/null || { echo "k61.py selftest failed"; ok=1; }
  python3 "$R/scripts/thunderdome_report.py" --selftest > /dev/null 2>&1 || { echo "thunderdome_report.py selftest failed"; ok=1; }
  mkrecipes "$RD" || ok=1
  mkreport || ok=1
  if [ -s "$RD/arm-2x.yaml" ]; then
    for h in localhost $H2; do
      hx $h "docker image inspect $(img "$RD/ctl-2x.yaml") >/dev/null 2>&1" || { echo "$h: $(img "$RD/ctl-2x.yaml") missing"; ok=1; }
      for t in $CREV $AREV; do hx $h "test -s $HF/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE/snapshots/$t/config.json && test -s $HF/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE/snapshots/$t/model-00034-of-00036.safetensors" \
        || { echo "$h: snapshot $t missing"; ok=1; }; done
      hx $h "test -s $SNAP/model-00035-of-00036.safetensors" || { echo "$h: 7c4f1bc1 shard 35 missing"; ok=1; }
      t=$(hx $h "python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))[\"records\"]))' $CRC/b12x/compile/preparation/$CPLAN" 2>/dev/null)
      [ "$t" = $NREC ] || { echo "$h: control plan $CPLAN has ${t:-no} records, not $NREC"; ok=1; }; done
    [ "$(hx localhost "md5sum < $CRC/b12x/compile/preparation/$CPLAN")" = "$(hx $H2 "md5sum < $CRC/b12x/compile/preparation/$CPLAN")" ] \
      || { echo "control plan $CPLAN differs between the Sparks"; ok=1; }
    echo "ok   ctl-2x: $(img "$RD/ctl-2x.yaml"), $(fld "$RD/ctl-2x.yaml" model)@$(fld "$RD/ctl-2x.yaml" model_revision | cut -c1-8)"
    echo "arm diff:"; diff "$RD/ctl-2x.yaml" "$RD/arm-2x.yaml" | sed 's/^/   /'; fi
  return $ok; }
remap() { # outdir -> the control plan re-keyed to the arm path (CPU, shipped image, no network); prints the file name
  local T; T=$(mktemp -d); cp "$CRC/b12x/compile/preparation/$CPLAN" "$T/ctl.json"
  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp --entrypoint python3 -v "$T:/t" -v "$G/k48:/k48:ro" \
    "$(img "$RD/ctl-2x.yaml")" /k48/remap_seed.py /t/ctl.json "$APATH" /t/seed > "$T/remap.txt" 2>&1
  local n; n=$(ls "$T"/seed/*.json 2>/dev/null | wc -l)
  [ "$n" = 1 ] && [ "$(nrec "$(ls "$T"/seed/*.json)")" = $NREC ] || { cat "$T/remap.txt" >&2; rm -rf "$T"; return 1; }
  mkdir -p "$1"; cp "$T"/seed/*.json "$1/"; cat "$T/remap.txt" > "$1/remap.txt"; basename "$(ls "$T"/seed/*.json)"; rm -rf "$T"; }

if [ "${1:-}" = --dry-run ]; then
  RD=$K/rec-dry; rm -rf "$RD"; preflight; rc=$?
  if [ $rc = 0 ]; then D=$(mktemp -d)
    if s=$(remap "$D"); then echo "ok   remap: $(cat "$D/remap.txt")"; else echo "remap failed"; rc=1; fi; rm -rf "$D"; fi
  echo "plan: registry update; stage the arm cache k92-d3 on both Sparks (no-clobber copy of the control cache dirs + the"
  echo "      re-keyed plan); prebake the arm with a 6 GiB KV pool (<= 3 tries) + one full-pool arm boot; identity check;"
  echo "      screen $ORDER on the pair (~20 min per boot: probes fresh c1/c4/c8 + d16k-c4 T=0, logits x2,"
  echo "      llama-benchy pp2048 + tg512 c1/c4/c8/c16 x4); judge (TD_ACC_RISE_OK=1); gate on RULE=PASS (GATE=$GATE, ~75 min); restore 2x"
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

# ---------------- 2x helpers (k83 pattern) ----------------
keep_logs2() { # dir (before stop_all)
  local c; c=$(n0); [ -n "$c" ] || c=$(docker ps -a --format '{{.Names}}' | grep node_0 | head -1); [ -n "$c" ] || return 0
  docker exec "$c" cat /tmp/sparkrun_serve.log > "$1/serve.log" 2>&1 || docker logs "$c" > "$1/serve.log" 2>&1
  on2 'c=$(docker ps -a --format "{{.Names}}" | grep node_1 | head -1); [ -n "$c" ] && { docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1; }' > "$1/serve-node1.log" 2>&1
  grep -E "Directly load AOT|saved AOT|Dynamo bytecode" "$1/serve.log" | cut -c1-300 > "$1/aot.txt"
  docker exec "$c" sh -c "env | grep -E '^(VLLM|B12X|XDG)_' | sort" > "$1/env.txt" 2>&1
  docker exec "$c" sh -c "ps aux | grep '[v]llm serve'" | grep -oE -- '--revision [0-9a-f]+' | head -1 > "$1/revision.txt" 2>&1
  { docker inspect --format '{{.Config.Image}}' "$c"; on2 "docker ps --format '{{.Image}}' --filter name=sparkrun"; } > "$1/image.txt" 2>&1; }
tboot() { # recipe dir cold_ok -> 0 up, 1 failed, 2 cold (when cold is not allowed)
  local rec=$1 d=$2 s m n cap; cap=$([ "$3" = 1 ] && echo 5400 || echo 1800)
  stop_all; log "boot $(basename "$rec") -> $d"; cp "$rec" "$d/recipe.yaml"
  ( cd "$(dirname "$rec")" && timeout -k 60 900 sparkrun run "$(basename "$rec")" --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  s=$(date +%s)
  while :; do
    n=$(n0)
    m=$([ -n "$n" ] && docker exec "$n" grep -m1 -oE 'Directly load AOT compilation|Dynamo bytecode transform time' /tmp/sparkrun_serve.log 2>/dev/null)
    [ "$m" = "Dynamo bytecode transform time" ] && [ "$3" != 1 ] && { log "$(basename "$d"): COLD (backbone compiling), stopped"; return 2; }
    [ "$(health localhost)" = 200 ] && { echo "boot $(( $(date +%s) - s ))s ${m:-no AOT line}" > "$d/boot.txt"; log "$(basename "$d"): up, $(cat "$d/boot.txt")"; return 0; }
    if [ $(( $(date +%s) - s )) -gt 180 ] && { [ -z "$n" ] || docker exec "$n" grep -qE 'Worker failed with error|EngineCore failed to start|Engine core initialization failed' /tmp/sparkrun_serve.log 2>/dev/null; }; then
      log "$(basename "$d"): boot FAILED after $(( $(date +%s) - s ))s"; return 1; fi
    [ $(( $(date +%s) - s )) -ge $cap ] && { log "$(basename "$d"): health timeout ($cap s)"; return 1; }
    sleep 10; done; }
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
  ( cd "$R" && child timeout -k 30 1800 uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$H1:8000/v1" --model $MODEL \
      --tokenizer "$SNAP" --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 --concurrency 1 4 8 16 --runs 4 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$H1:8000/metrics" \
      --save-result "$d/task.csv" > "$d/task.log" 2>&1 ); log "$(basename "$d"): benchy exit=$?"
  [ "$(health localhost)" = 200 ] || { echo "server gone after the pass" > "$d/DIED"; log "$(basename "$d"): server DIED"; }; }
memlog2() { awk -v a="$2" -v b="$3" '$1 >= a && $1 <= b && $2 ~ /^[0-9]+$/ { printf "%s %.2f\n", $1, $2 / 1024 }' "$RES/guard-dgx01.log" "$RES/guard-dgx02.log" > "$1/mem.log"; }
measured() { grep -ohE '\b[0-9]+ measured' "$1" 2>/dev/null | awk '{s+=$1} END {print s+0}'; }
plans_ok() { # dir: rank 0 measured nothing, the re-keyed plan holds NREC records on both ranks, no other new plan file
  local h n o bad=
  for h in localhost $H2; do
    n=$(hx $h "python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))[\"records\"]))' $ARC/b12x/compile/preparation/$APLAN" 2>/dev/null)
    o=$(hx $h "ls $ARC/b12x/compile/preparation/ | grep '\.json\$'" | sort | comm -13 "$K/plans-before.txt" - | grep -vx "$APLAN" | tr '\n' ' ')
    echo "$h: $APLAN ${n:-no} records${o:+, new plan files: $o}"
    [ "$n" = $NREC ] && [ -z "$o" ] || bad=1; done > "$1/plans.txt"
  echo "rank 0 measured: $(measured "$1/serve.log")" >> "$1/plans.txt"
  [ -z "$bad" ] && [ "$(measured "$1/serve.log")" = 0 ]; }
regard() {
  kill -0 $G1 2>/dev/null || { bash -c "$GUARD" >> "$RES/guard-dgx01.log" 2>&1 < /dev/null 9>&- & G1=$!; log "memory guard dgx-01 restarted"; }
  kill -0 $G2 2>/dev/null || { ssh -n $H2 "$GUARD" >> "$RES/guard-dgx02.log" 2>&1 9>&- & G2=$!; log "memory guard dgx-02 restarted"; }; }
stage() { # arm cache on both Sparks: no-clobber copies of the control cache dirs + the re-keyed plan
  local h; APLAN=$(remap "$RES/seed") || return 1; echo "$APLAN" > "$K/APLAN"; log "remap: $(cat "$RES/seed/remap.txt")"
  on2 "mkdir -p $K"; scp -q "$RES/seed/$APLAN" "$H2:$K/$APLAN" || return 1; cp "$RES/seed/$APLAN" "$K/$APLAN"
  for h in localhost $H2; do
    hx $h "set -e; mkdir -p $ARC; for d in b12x triton inductor flashinfer cute_dsl flash_attn_cute_dsl torch_extensions numba; do
             [ -d $CRC/\$d ] && cp -an $CRC/\$d $ARC/; done
           mkdir -p $ARC/b12x/compile/preparation; cp -n $K/$APLAN $ARC/b12x/compile/preparation/
           cmp -s $K/$APLAN $ARC/b12x/compile/preparation/$APLAN; du -sh $ARC | cut -f1" > "$RES/stage-$h.txt" 2>&1 || { log "stage $h failed"; return 1; }
  done
  [ "$(on2 "ls $ARC/b12x/compile/preparation/" | sort)" = "$(ls "$ARC/b12x/compile/preparation/" | sort)" ] || { log "stage: plan file sets differ between the Sparks"; return 1; }
  # baseline for plans_ok: what the arm cache holds now plus what the recipe's first command line copies from the image seed
  { ls "$ARC/b12x/compile/preparation/"; docker run --rm --network none --entrypoint ls "$(img "$RD/ctl-2x.yaml")" /opt/b12x-seed/preparation; } \
    | grep '\.json$' | grep -vx "$APLAN" | sort -u > "$K/plans-before.txt"; }
bake() {
  local t d ok=
  for t in 1 2 3; do
    d=$RES/bake/prebake-try$t; rm -rf "$d"; mkdir -p "$d"; st "running: prebake try $t (arm, 6 GiB KV pool)"; regard
    if tboot "$RD/prebake-2x.yaml" "$d" 1; then keep_logs2 "$d"; stop_all; ok=1
      log "prebake try $t: $(cat "$d/boot.txt"), $(measured "$d/serve.log") plans measured"; break; fi
    keep_logs2 "$d"; stop_all; log "prebake try $t failed"
  done
  [ -n "$ok" ] || { log "prebake: 3 tries failed"; return 1; }
  d=$RES/bake/arm-full; rm -rf "$d"; mkdir -p "$d"; st "running: bake arm full pool"; regard
  tboot "$RD/arm-2x.yaml" "$d" 1 || { keep_logs2 "$d"; stop_all; log "bake full-pool boot failed"; return 1; }
  keep_logs2 "$d"; stop_all; regard
  plans_ok "$d"; log "bake full pool: $(cat "$d/boot.txt"); $(tr '\n' ' ' < "$d/plans.txt")"
  for t in "$RES"/bake/*/; do plans_ok "$t" || { log "bake: plan identity check failed ($t/plans.txt)"; return 1; }; done; }
boot2() { # dir recipe: one counted boot (warm only)
  local d=$1 rc t0; rm -rf "$d"; mkdir -p "$d"; st "running: screen $(basename "$d")"; t0=$(date +%s)
  tboot "$2" "$d" 0; rc=$?
  if [ $rc -ne 0 ]; then keep_logs2 "$d"; [ $rc -eq 2 ] && echo cold > "$d/COLD" || echo failed > "$d/FAILED"; stop_all; return 0; fi
  keep_logs2 "$d"
  case $(basename "$d") in arm*) plans_ok "$d" ;; *) [ "$(measured "$d/serve.log")" = 0 ] ;; esac \
    || { echo "plans: $(tr '\n' ' ' < "$d/plans.txt" 2>/dev/null) measured $(measured "$d/serve.log")" > "$d/COLD"; stop_all
         FINAL="FAILED: k92 $(basename "$d") measured b12x plans or lost the re-keyed plan ($RES)"; exit 1; }
  run_pass "$d"; keep_logs2 "$d"; memlog2 "$d" "$t0" "$(date +%s)"; stop_all; }
judge() { # node dir -> report, post checks per boot, Welch CI, RULE=PASS|FLAT|FAIL
  TD_ACC_RISE_OK=1 PYTHONPATH="$R/scripts" python3 - "$1" "$K/td_report_k92.py" "$K" "$K70" <<'PY'
import glob, os, subprocess, sys
d, rep, k, k70 = sys.argv[1:5]
sys.path[:0] = [k, k70]
from pool import post, rule, welch, benchy
from k61 import tracebacks
r = subprocess.run([sys.executable, rep, d], capture_output=True, text=True)
print(r.stdout.rstrip())
if r.returncode:
    print("report stderr:", r.stderr.strip()[-1500:])
print("== boot-level Welch 95% CI, benchy cells (reported; the rule uses the noise bands above)")
boots = {w: sorted(glob.glob(os.path.join(d, f"{w}-p[0-9]"))) for w in ("ctl", "arm")}
for cell in ("pp2048 (c1)", "tg512 (c1)", "tg512 (c4)", "tg512 (c8)", "tg512 (c16)"):
    c = [v for v in (benchy(p).get(cell) for p in boots["ctl"]) if v]
    a = [v for v in (benchy(p).get(cell) for p in boots["arm"]) if v]
    if len(c) < 2 or len(a) < 2:
        print(f"  {cell:12s} too few boots ({len(c)}/{len(a)})"); continue
    dd, lo, hi = welch(c, a)
    print(f"  {cell:12s} {dd:+6.2f}%  CI [{lo:+.2f},{hi:+.2f}]  control {' '.join(f'{x:.1f}' for x in c)} | arm {' '.join(f'{x:.1f}' for x in a)}")
print("== post checks per boot")
problems, kvs = [], {}
for p in boots["ctl"] + boots["arm"]:
    bad, kv = post(p)
    if "Traceback in serve.log" in bad:   # TritonBundler cubin warnings are not errors (k61)
        bad.remove("Traceback in serve.log")
        tb = tracebacks(open(os.path.join(p, "serve.log"), errors="replace").read())
        if tb:
            bad.append(f"{len(tb)} traceback(s) in serve.log")
    kvs[os.path.basename(p)] = int(kv.replace(",", "")) if kv else None
    rev = open(os.path.join(p, "revision.txt")).read().strip() if os.path.exists(os.path.join(p, "revision.txt")) else "?"
    print(f"  {os.path.basename(p)}: {rev} kv {kv} {'; '.join(bad) or 'ok'}")
    problems += [f"{os.path.basename(p)}: {b}" for b in bad]
ctl = [v for n, v in kvs.items() if n.startswith("ctl") and v]
for n, v in kvs.items():
    if n.startswith("arm") and (not ctl or not v or abs(v / (sum(ctl) / len(ctl)) - 1) > 0.03):
        problems.append(f"KV pool {n} {v} vs control {ctl} (> 3%)")
if "VERDICT=" not in r.stdout:
    problems.append("no verdict from the report")
v, worse, better = rule(r.stdout, problems)
for x in problems:
    print("post:", x)
print("worse:", "; ".join(worse) or "none")
print("better:", "; ".join(better) or "none")
print(f"RULE={v}")
PY
}
gate() { # -> GV, GSUM
  local g=$RES/gate t
  st "running: gate"; rm -rf "$g"; mkdir -p "$g"
  if tboot "$RD/arm-2x.yaml" "$g" 0; then
    keep_logs2 "$g"
    plans_ok "$g" || { stop_all; FINAL="FAILED: k92 gate boot measured b12x plans or lost the re-keyed plan ($RES)"; exit 1; }
    ( cd "$R" && child timeout 900 python3 scripts/straggler_probe.py 5 6 7 8 12 16 > "$g/straggler.log" 2>&1 )
    child timeout 7200 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://localhost:8000 --model $MODEL > "$g/hardmode.log" 2>&1
    child timeout 3600 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://localhost:8000/v1 --model $MODEL --scenarios TC-45 --trials 5 > "$g/tc45.log" 2>&1
    ( cd "$R" && child timeout 5400 python3 scripts/fidelity_probe.py --base http://localhost:8000 --model $MODEL --depths 8000,32000,64000,128000 --out "$g/fidelity.json" > "$g/fidelity_probe.txt" 2>&1 )
    for sd in 11 13; do ( cd "$R" && child timeout 2700 python3 scripts/fidelity_probe.py --base http://localhost:8000 --model $MODEL --depths 128000 --seed $sd --out "$g/fidelity-seed$sd.json" > "$g/fidelity-seed$sd.txt" 2>&1 ); done
    keep_logs2 "$g"; stop_all
    t=$(mktemp -d); ln -s "$g" "$t/cand1"; ln -s "$g" "$t/cand2"
    GV=$(python3 -c "import sys; sys.path.insert(0, sys.argv[1]); import k61; b, hm = k61.gate(sys.argv[2]); print('PASS' if not b else 'FAIL (' + '; '.join(b) + ')')" "$K70" "$t" 2>&1); rm -rf "$t"
    GSUM=$(python3 - "$g" <<'PY'
import re, sys
g = sys.argv[1]
rd = lambda f: open(f"{g}/{f}", errors="replace").read() if __import__("os").path.exists(f"{g}/{f}") else ""
hm = re.findall(r"Quality:\s*(\d+)/100", rd("hardmode.log"))
fid = re.findall(r"^depth\s+(\d+) .*?exact\s+(\d+)", rd("fidelity_probe.txt") + rd("fidelity-seed11.txt") + rd("fidelity-seed13.txt"), re.M)
rounds = {c: (per, p) for c, per, p in re.findall(r"^c=(\d+) round wall [\d.]+s \| per request[^:]*:(.*)\n\s+preemptions \+?(\d+)", rd("straggler.log"), re.M)}
st = {}
for c in ("8", "12", "16"):
    if c not in rounds:
        st[c] = None; continue
    t = sorted(float(x) for x in re.findall(r"#\d+:([\d.]+)s", rounds[c][0]))
    st[c] = f"preempt {rounds[c][1]}, max/median {t[-1] / t[len(t) // 2]:.2f}" if t else None
ok = all(v and v.startswith("preempt 0,") and float(v.rsplit(" ", 1)[1]) < 1.5 for v in st.values())
print(f"hardmode {hm[-1] if hm else 'missing'} ({'ok' if hm and int(hm[-1]) >= 88 else 'below 88'}); "
      f"long-context retrieval {' '.join(f'{d}:{e}/20' for d, e in fid) or 'missing'} ({'ok' if len(fid) == 6 and all(int(e) == 20 for _, e in fid) else 'not 20/20 everywhere'}); "
      f"stragglers c8/c12/c16 {st} ({'none' if ok else 'check straggler.log'})")
PY
)
    { echo; echo "== gate (arm, one TP2 boot): $GV"; echo "$GSUM"; grep -hE "Quality:" "$g/hardmode.log"; grep -hE "Pass\^5|Score:" "$g/tc45.log" | head -2
      grep -h "^depth" "$g/fidelity_probe.txt" "$g"/fidelity-seed1?.txt; tail -6 "$g/straggler.log"; } >> "$RES/k92.txt"
  else keep_logs2 "$g"; stop_all; GV="FAIL (gate boot failed)"; GSUM="gate boot failed"; echo "== gate: boot failed" >> "$RES/k92.txt"; fi
  log "gate: $GV; $GSUM"; }

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
mkdir -p "$RES"; echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
job_begin
RD=$RES/recipes
echo "k92 $(TZ=Europe/Bucharest date '+%F %T %Z'): MTP drafter D3 (@${AREV:0:8}) vs the shipped 2x v2.0.0 (D1, @${CREV:0:8}), same image" > "$RES/k92.txt"
st "running: registry update"
timeout 300 sparkrun registry update $REG2 >> "$RES/$J.log" 2>&1 < /dev/null 9>&-
st "running: preflight"
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
cat "$RD/sources.txt" >> "$RES/k92.txt"; cp "$K/run.sh" "$K/td_report_k92.py" "$K/pool.py" "$RD/"
st "running: stage arm cache"
stage || { FINAL="FAILED: k92 stage ($RES/stage-*.txt)"; exit 1; }
echo "re-keyed plan: $(cat "$RES/seed/remap.txt")" >> "$RES/k92.txt"

st "running: stop 2x"; stop_all
bake || { FINAL="FAILED: k92 bake ($RES/bake)"; exit 1; }
{ echo "bake:"; for b in "$RES"/bake/*/; do echo "  $(basename "$b"): $(cat "$b/boot.txt" 2>/dev/null || echo failed); $(tr '\n' ' ' < "$b/plans.txt" 2>/dev/null)"; done; } >> "$RES/k92.txt"

S=$RES/2x; mkdir -p "$S"
for item in $ORDER; do who=${item%:*}; k=${item#*:}
  if [ $who = arm ]; then boot2 "$S/arm-p$k" "$RD/arm-2x.yaml"; else boot2 "$S/ctl-p$k" "$RD/ctl-2x.yaml"; fi; done
st "running: judge"
judge "$S" > "$RES/judge-2x.txt" 2>&1
RU=$(sed -n 's/^RULE=//p' "$RES/judge-2x.txt" | tail -1)
{ echo; echo "== screen ($ORDER): RULE=${RU:-none}"; grep -vE '^  (self|cross) ' "$RES/judge-2x.txt"; } >> "$RES/k92.txt"
GV="not run (RULE=${RU:-none}; the gate runs only on PASS)"; GSUM=
[ "$RU" = PASS ] && { [ "$GATE" = 0 ] && GV="not run (GATE=0)" || gate; }
OUT=FAILS; case "$RU:${GV%% *}" in PASS:PASS) OUT=PASSES ;; esac
for l in dgx01 dgx02; do echo "guard $l: min MemAvailable $(awk '$2 ~ /^[0-9]+$/' "$RES/guard-$l.log" | sort -k2 -n | head -1 | awk '{printf "%.2f GiB", $2/1024}')"; done >> "$RES/k92.txt"
FINAL="DONE: k92 RULE=${RU:-none} gate=${GV%% (*} -> D3 on the 2x $OUT ($RES/k92.txt)"
cells() { sed -n '/^== cells/,/^==/p' "$1" | grep -E '^ +(probe|wall|benchy) ' | grep -v ' missing$'; }
acc() { sed -n '/^== acceptance/,/^==/p' "$1" | grep -E '^ +pos '; }
welchl() { sed -n '/^== boot-level Welch/,/^==/p' "$1" | grep -E '^  [pt]'; }
{ echo "k92-2x-d3-drafter-ab result: I ran drafter D3 (checkpoint revision \`${AREV:0:8}\`) against the shipped 2x v2.0.0 (D1, \`${CREV:0:8}\`) on the same image and the same plan selections (the control's TP=2 plan re-keyed to the D3 snapshot path, 0 plans measured on every counted boot). TP=2 on the pair, 4 boots per side in the order $ORDER. T=0 probes fresh c1/c4/c8 and d16k-c4, llama-benchy pp2048 + tg512 c1/c4/c8/c16."
  echo; echo "Screen rule: \`${RU:-none}\` (PASS: at least one cell better beyond its noise band and no cell from c1 to c16 worse beyond it). Gate: $GV.${GSUM:+ $GSUM.}"
  echo; echo "MTP acceptance per draft position (T=0 probe cells, pooled, D1 then D3):"; echo '```'; acc "$RES/judge-2x.txt"; echo '```'
  echo; echo "Cells (D3 vs D1, noise band) and the boot-level Welch 95% CI of the benchy cells:"
  echo '```'; cells "$RES/judge-2x.txt"; welchl "$RES/judge-2x.txt"; grep -E '^(post|worse|better):' "$RES/judge-2x.txt"; echo '```'
  echo; case $OUT in
    PASSES) echo "D3 meets my rule on the 2x and passed the gate. This is the evidence for a 2x v2.1.0 (drafter only); nothing ships from this job." ;;
    *) echo "D3 does not meet my rule on the 2x in this run, so the 2x stays on v2.0.0 with D1." ;; esac
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
# public-text guard: the comment is posted raw
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"
grep -qE '192\.168\.|dgx-0[0-9]\.local' "$K/comment.md" && { mkdir -p "$K/private"; mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k92 comment.md held a LAN address (not posted; $RES)"; exit 1; }
cp "$K/comment.md" "$RES/comment.md"
exit 0
