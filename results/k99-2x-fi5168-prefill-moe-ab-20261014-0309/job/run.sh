#!/usr/bin/env bash
# k99-2x-fi5168-prefill-moe-ab (2026-10-13, tp-2): FlashInfer 5c4058bd (PR #5168) SM12x fused MoE for large prefill
# chunks on the 2x, b12x for decode and small chunks, against the shipped 2x v2.1.0 (same image, same weights, same
# plan selections). k98: FI5168 passes accuracy and contract vs b12x at I320; layer-5 real routing x0.98/1.01/1.18/1.22
# at M 1264/3024/6048/9072 (TP2). The 2x's attention block is 2864 tokens, so at budget 8192 its large prefill chunk is
# 5728 rows (2 x 2864; 16K ~ 5728+5728+2864+2064), never 6048: the mod's default threshold 6048 would route nothing,
# so the arm sets MIN=4096 (all 5728 chunks to FlashInfer, every <= 2864 tail and every decode step stays on b12x).
#   mod      mod/sitecustomize.py + mod/k99fi.py on PYTHONPATH from the arm's runtime cache (image untouched): registers
#            VLLM_K99_FI_MOE_SRC / VLLM_K99_FI_MOE_MIN_TOKENS in vllm/envs.py (compile key), and B12xExperts.apply sends
#            calls of >= MIN rows outside graph capture to FlashInfer over views of b12x's prepared storage. The image's
#            flashinfer 0.7.0 (sampler) and cutlass-dsl 4.7.0 stay; the 5c4058bd tree is swapped in around each call.
#   P0 op    (dgx-01, 2x stopped) opcheck.py in a throwaway container of the shipped image: the mod's call on
#            b12x-prepared I320 layers 5/47 with the k97 capture, FlashInfer before b12x (flip order), storage bytes =
#            k98's FI inputs, numerics and speed vs k98 (M 2864/5728/6048/9072), memory FlashInfer keeps, sampler before
#            and after. FAIL -> one diagnostic rerun with cutlass-dsl 4.7.1 (k98 wheels), then FAILED before any boot.
#   stage    arm cache /cache/runtime/k99 (host: <2x runtime cache>/k99) on both Sparks: no-clobber copies of the
#            control cache dirs (b12x incl. plan 225bf665, triton, inductor, flashinfer, ...), the mod, the 5c4058bd
#            tree (no .git). FLASHINFER_CACHE_DIR/WORKSPACE_BASE point there too (FI JIT never lands in the control's).
#   bake     the env vars are compile factors, so the arm compiles once: prebake 6 GiB KV pool (<= 3 tries), then one
#            full-pool boot; both ranks must log K99MOD + K99FI calls (the profile run at 8192 rows takes the FI path),
#            0 b12x plans measured, plan 225bf665 still 682 records, no new plan file.
#   screen   TP=2 on the pair, boots $ORDER (2 per side, interleaved), warm only: warm-up cold 16K + 64K (own text);
#            cold TTFT 16K/64K/128K x3 (k97/cold.py, same prompts every boot: seed k99, corpus offset 0, nonce prefix);
#            T=0 fresh-c8 MTP acceptance per position; llama-benchy pp2048 + tg512 c1/c4/c8 x3 (decode steps stay on
#            b12x; a step that prefills 2+ of the 2048-token prompts at once can reach 4096 rows and take FlashInfer).
#            Every distinct row count FlashInfer served is logged once per process and listed in k99.txt.
#   judge    k99_report.py: RULE=PASS (64K or 128K TTFT better beyond boot noise, nothing worse, acceptance within
#            0.03, >= 1000 FI calls on both ranks of every arm boot, none on the control) | FLAT | FAIL | EMPTY (a counted boot
#            measured nothing: FAILED, never DONE). KV pool delta = the FlashInfer path's memory cost, reported.
#   gate     on PASS or FLAT (GATE=1): one arm boot, stragglers c5-c16, hardmode >= 88, TC-45 x5 = 100, fidelity
#            8k/32k/64k/128k + 128k seeds 11/13 all 20/20 (k61.gate). 128K fidelity runs ~22 FlashInfer chunks per prompt.
# Nothing ships from this job. ETA ~4 h: op check ~15 min, stage ~5, bake ~25-45, 4 boots x ~25, gate ~75, restore ~6.
# Contract (backlog/runner.sh): both Spark locks (GPU_LOCK_HELD=1); exit 75 untouched while another job is busy; STATE
# first line DONE/FAILED (lib.sh EXIT trap restores the shipped 2x); comment.md + RESULT for the poster.
# Usage: bash run.sh --dry-run | JOBS=k99 GRACE0=60 setsid nohup bash ~/GEN-AI/backlog/runner.sh > ~/GEN-AI/backlog/runner-k99.nohup 2>&1 < /dev/null &
# Stop: kill -TERM <run.sh pid> (see k99.log). Knobs: K99_MIN (4096), GATE (1).
NEEDS="dgx01 dgx02"
set -u
J=k99; K=$HOME/GEN-AI/backlog/k99
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k99-2x-fi5168-prefill-moe-ab-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
REV=03f4a0570496bbe741189f4a2a7d611679e85406; NREC=682; IMG=ghcr.io/ursuciprian/spark-vllm-b12x:$SHIPPED_TAG_DEFAULT
PLAN=225bf66521da31897db0719256f6a56cc38d43f87dd4f85f1f52549540986fe2.json
CRC=$HOME/.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923   # the 2x's cache (@03f4a057)
ARC=$CRC/k99; ACACHE=/cache/runtime/k99; KV_PRE=6442450944; MIN=${K99_MIN:-4096}; GATE=${GATE:-1}
FISHA=5c4058bd9b75967de947971d7f28d36bd600d888; FISRC=$BL/k98/fi-src; K98=$BL/k98; CAPD=$BL/k97/capture; COLD=$BL/k97/cold.py
HF=$HOME/.cache/huggingface/hub; HFM=$HF/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE; CKPT=$HFM/snapshots/$REV
SNAP=$HF/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
MODEL=qwen3.8-flash-next; CORPUS=$R/results/corpus-code.txt; BENCHY_SRC=$G/llama-benchy-fork; K70=$BL/k70
REGS=$HOME/.cache/sparkrun/registries; REG2=qwen38-flashnext; N2=qwen3.8-flash-next-2x-dgx-spark
ORDER="ctl:1 arm:1 arm:2 ctl:2"; LENS=16384,65536,131072; REPS=3
PRIV=$K98/private.pat
on2() { ssh -n -o ConnectTimeout=10 $H2 "$@"; }
hx() { if [ "$1" = localhost ]; then bash -c "$2"; else on2 "$2"; fi; }
img() { sed -n 's/^container: *//p' "$1" | head -1 | tr -d "\"' "; }
fld() { sed -n "s/^$2: *//p" "$1" | head -1 | tr -d "\"' "; }
nrec() { python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["records"]))' "$1" 2>/dev/null || echo 0; }
# The arm runs the shipped image and revision, so lib.sh's is_shipped alone would take a running arm for the shipped
# 2x and the EXIT-trap restore would leave it serving: the shipped 2x also has no K99 variable in its environment.
is_shipped() { [ "$(health localhost)" = 200 ] && docker ps --format '{{.Image}}' | grep -q ":$E\$" \
  && ssh -n -o ConnectTimeout=10 $H2 "docker ps --format '{{.Image}}'" | grep -q ":$E\$" \
  && docker exec "$(n0)" sh -c "ps aux | grep '[v]llm serve'" | grep -- '--tensor-parallel-size 2' | grep -q -- "--revision $REV" \
  && ! docker exec "$(n0)" env | grep -q '^VLLM_K99'; }

mkarm() { # ctl.yaml arm.yaml: own name, own runtime cache, the mod; 1 line changed, 8 added
  python3 - "$1" "$2" "$ACACHE" "$MIN" <<'PY'
import sys
from collections import Counter
src, dst, cache, mn = sys.argv[1:5]
lines = open(src).read().split("\n")
def one(pred, what):
    i = [k for k, l in enumerate(lines) if pred(l)]
    if len(i) != 1:
        sys.exit(f"mkarm: need exactly one {what}, found {len(i)}")
    return i[0]
n, e = one(lambda l: l.startswith("name: "), "name line"), one(lambda l: l == "env:", "env: block")
if any(l.strip().startswith(("XDG_CACHE_HOME:", "VLLM_CACHE_ROOT:", "PYTHONPATH:", "FLASHINFER_", "VLLM_K99")) for l in lines):
    sys.exit("mkarm: the control already sets one of the arm's variables")
add = ["  # k99 arm: FlashInfer 5c4058bd fused MoE for prefill calls >= MIN rows (mod on PYTHONPATH), own runtime cache",
       f'  XDG_CACHE_HOME: "{cache}"', f'  VLLM_CACHE_ROOT: "{cache}/vllm"', f'  FLASHINFER_CACHE_DIR: "{cache}/flashinfer"',
       f'  FLASHINFER_WORKSPACE_BASE: "{cache}/flashinfer"', f'  PYTHONPATH: "{cache}/mod"',
       f'  VLLM_K99_FI_MOE_SRC: "{cache}/fi-src"', f'  VLLM_K99_FI_MOE_MIN_TOKENS: "{mn}"']
out = lines[:]
out[n] = "name: qwen3.8-flash-next-2x-k99-fi5168"
out[e + 1:e + 1] = add
a, b = Counter(lines), Counter(out)
if sum((a - b).values()) != 1 or sum((b - a).values()) != 1 + len(add):
    sys.exit("mkarm: unexpected diff")
open(dst, "w").write("\n".join(out))
PY
}
mkrecipes() { # dir -> ctl-2x.yaml (registry, as shipped) + arm-2x.yaml + prebake-2x.yaml
  local d=$1 f2; mkdir -p "$d"
  f2=$(grep -rlx --include='*.yaml' "name: $N2" "$(readlink -f "$REGS/$REG2")/recipes" 2>/dev/null | head -1)
  [ -s "$f2" ] || { echo "2x recipe not resolvable from the registry ($REG2/$N2)"; return 1; }
  cp "$f2" "$d/ctl-2x.yaml"
  [ "$(img "$d/ctl-2x.yaml" | sed 's|.*:||')" = "$SHIPPED_TAG_DEFAULT" ] || { echo "registry 2x tag is not $SHIPPED_TAG_DEFAULT"; return 1; }
  [ "$(fld "$d/ctl-2x.yaml" model_revision)" = "$REV" ] || { echo "registry 2x revision is not $REV"; return 1; }
  sed -n 1p "$d/ctl-2x.yaml" | grep -q '^# Release: v2.1.0' || { echo "registry 2x recipe is not v2.1.0"; return 1; }
  mkarm "$d/ctl-2x.yaml" "$d/arm-2x.yaml" || return 1
  sed -e 's|^    --gpu-memory-utilization {gpu_memory_utilization}$|    --gpu-memory-utilization {gpu_memory_utilization} \\\n    --kv-cache-memory-bytes '"$KV_PRE"'|' "$d/arm-2x.yaml" > "$d/prebake-2x.yaml"
  [ "$(diff "$d/arm-2x.yaml" "$d/prebake-2x.yaml" | grep -c '^>')" = 2 ] && grep -qx "    --kv-cache-memory-bytes $KV_PRE" "$d/prebake-2x.yaml" \
    || { echo "prebake-2x.yaml edit did not apply"; return 1; }
  echo "2x $f2 @ $(git -C "$(readlink -f "$REGS/$REG2")" rev-parse --short HEAD) $(head -1 "$f2")" > "$d/sources.txt"; }
cpucheck() { # CPU only, no GPU, no network: the mod against the shipped image's source + the FlashInfer coexistence
  docker run --rm --network none --entrypoint python3 -e PYTHONPATH=/k99mod -e VLLM_K99_FI_MOE_SRC=/fi -e VLLM_K99_FI_MOE_MIN_TOKENS=$MIN \
    -v "$K/cpucheck.py:/cpucheck.py:ro" -v "$K/mod:/k99mod:ro" -v "$FISRC:/fi:ro" "$IMG" /cpucheck.py 2>&1 \
    | grep -v 'warnings.warn\|UserWarning'; }
host_checks() { # no containers
  local ok=0 f t h
  for f in "$K/mod/sitecustomize.py" "$K/mod/k99fi.py" "$K/opcheck.py" "$K/cpucheck.py" "$K/k99_report.py" "$K98/moe98.py" "$COLD" \
           "$K70/k61.py" "$CORPUS" "$PRIV" "$CKPT/model.safetensors.index.json" "$FISRC/flashinfer/fused_moe/cute_dsl/b12x_moe.py" \
           "$R/scripts/depth_decode_probe.py" "$R/scripts/straggler_probe.py" "$R/scripts/fidelity_probe.py"; do
    [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  [ -d "$BENCHY_SRC" ] && [ -d "$SNAP" ] || { echo "missing llama-benchy fork or tokenizer snapshot"; ok=1; }
  for t in sparkrun uvx tool-eval-bench python3 curl rsync; do command -v $t > /dev/null || { echo "missing $t"; ok=1; }; done
  [ "$(git -C "$FISRC" rev-parse HEAD 2>/dev/null)" = $FISHA ] || { echo "fi-src HEAD is not $FISHA"; ok=1; }
  [ -z "$(git -C "$FISRC" status --porcelain --untracked-files=no 2>/dev/null)" ] || { echo "fi-src has local changes"; ok=1; }
  [ "$(ls "$CAPD" 2>/dev/null | grep -c '^moe5\.')" -ge 1 ] && [ "$(ls "$CAPD" | grep -c '^moe47\.')" -ge 1 ] || { echo "no k97 capture for layers 5/47 in $CAPD"; ok=1; }
  for f in "$K"/*.py "$K"/mod/*.py; do python3 -m py_compile "$f" || { echo "$f does not compile"; ok=1; }; done
  for t in "$K/k99_report.py" "$K/opcheck.py" "$COLD"; do python3 "$t" selftest > /dev/null || { echo "$(basename "$t") selftest failed"; ok=1; }; done
  python3 "$K70/k61.py" --selftest > /dev/null || { echo "k61.py selftest failed"; ok=1; }
  mkrecipes "$RD" || ok=1
  for h in localhost $H2; do
    hx $h "docker image inspect $IMG > /dev/null 2>&1" || { echo "$h: $IMG missing"; ok=1; }
    hx $h "test -s $CKPT/config.json" || { echo "$h: snapshot $REV missing"; ok=1; }
    t=$(hx $h "python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))[\"records\"]))' $CRC/b12x/compile/preparation/$PLAN" 2>/dev/null)
    [ "$t" = $NREC ] || { echo "$h: control plan $PLAN has ${t:-no} records, not $NREC"; ok=1; }; done
  [ "$(md5sum < "$CRC/b12x/compile/preparation/$PLAN")" = "$(on2 "md5sum < $CRC/b12x/compile/preparation/$PLAN")" ] \
    || { echo "control plan $PLAN differs between the Sparks"; ok=1; }
  for f in "$K"/run.sh "$K"/*.py "$K"/mod/*.py; do grep -qiEf "$PRIV" "$f" && { echo "private pattern in $f"; ok=1; }; done
  [ -s "$RD/arm-2x.yaml" ] && { echo "arm diff:"; diff "$RD/ctl-2x.yaml" "$RD/arm-2x.yaml" | sed 's/^/   /'; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  RD=$K/rec-dry; mkdir -p "$RD"; host_checks; rc=$?
  cpucheck > "$K/cpucheck-dry.txt"; tail -1 "$K/cpucheck-dry.txt" | grep -qx 'CPUCHECK ok' && sed 's/^/cpucheck: /' "$K/cpucheck-dry.txt" \
    || { echo "cpucheck failed ($K/cpucheck-dry.txt)"; rc=1; }
  echo "plan: stop 2x; P0 op check on dgx-01 (FlashInfer 5c4058bd via the mod vs b12x, I320, M 2864/5728/6048/9072);"
  echo "      stage arm cache $ARC on both Sparks; bake (prebake 6 GiB <= 3 tries + full pool); screen $ORDER"
  echo "      (warm-up 16K+64K, cold 16K/64K/128K x$REPS, fresh-c8 acceptance, benchy pp2048 + tg512 c1/c4/c8 x3);"
  echo "      judge; gate on PASS/FLAT (GATE=$GATE); restore 2x. Arm threshold MIN=$MIN rows."
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  sed -n 's/^# Nothing ships from this job. ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

# ---------------- 2x helpers (k92 pattern) ----------------
keep_logs2() { # dir (before stop_all)
  local c; c=$(n0); [ -n "$c" ] || c=$(docker ps -a --format '{{.Names}}' | grep node_0 | head -1); [ -n "$c" ] || return 0
  docker exec "$c" cat /tmp/sparkrun_serve.log > "$1/serve.log" 2>&1 || docker logs "$c" > "$1/serve.log" 2>&1
  on2 'c=$(docker ps -a --format "{{.Names}}" | grep node_1 | head -1); [ -n "$c" ] && { docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1; }' > "$1/serve-node1.log" 2>&1
  grep -E "Directly load AOT|saved AOT|Dynamo bytecode" "$1/serve.log" | cut -c1-300 > "$1/aot.txt"
  docker exec "$c" sh -c "env | grep -E '^(VLLM|B12X|XDG|FLASHINFER|PYTHONPATH)' | sort" > "$1/env.txt" 2>&1
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
measured() { grep -ohE '\b[0-9]+ measured' "$1" 2>/dev/null | awk '{s+=$1} END {print s+0}'; }
fipath() { # dir: the mod is active and FlashInfer ran on both ranks
  grep -q 'K99MOD active' "$1/serve.log" && grep -q 'K99MOD active' "$1/serve-node1.log" \
    && grep -q 'K99FI calls=' "$1/serve.log" && grep -q 'K99FI calls=' "$1/serve-node1.log"; }
plans_ok() { # dir: rank 0 measured nothing, plan 225bf665 holds NREC records on both ranks, no other new plan file
  local h n o bad=
  for h in localhost $H2; do
    n=$(hx $h "python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))[\"records\"]))' $ARC/b12x/compile/preparation/$PLAN" 2>/dev/null)
    o=$(hx $h "ls $ARC/b12x/compile/preparation/ | grep '\.json\$'" | sort | comm -13 "$K/plans-before.txt" - | grep -vx "$PLAN" | tr '\n' ' ')
    echo "$h: $PLAN ${n:-no} records${o:+, new plan files: $o}"
    [ "$n" = $NREC ] && [ -z "$o" ] || bad=1; done > "$1/plans.txt"
  echo "rank 0 measured: $(measured "$1/serve.log")" >> "$1/plans.txt"
  [ -z "$bad" ] && [ "$(measured "$1/serve.log")" = 0 ]; }
regard() {
  kill -0 $G1 2>/dev/null || { bash -c "$GUARD" >> "$RES/guard-dgx01.log" 2>&1 < /dev/null 9>&- & G1=$!; log "memory guard dgx-01 restarted"; }
  kill -0 $G2 2>/dev/null || { ssh -n $H2 "$GUARD" >> "$RES/guard-dgx02.log" 2>&1 9>&- & G2=$!; log "memory guard dgx-02 restarted"; }; }
opcheck() { # outdir [dsl471]: P0 on dgx-01 (2x stopped), throwaway container of the shipped image
  local o=$1 c=(python3 /k99/opcheck.py run /out --ckpt "$CKPT" --cap-dir /cap --fi-src /fi) w=()
  mkdir -p "$o"; docker rm -f k99-opcheck > /dev/null 2>&1
  [ "${2:-}" = dsl471 ] && { w=(-v "$K98/wheels:/wheels:ro"); c=(bash -c "pip install -q --no-index --no-deps /wheels/*.whl 2>&1 | tail -2; exec ${c[*]}"); }
  child timeout -k 30 2400 docker run --rm --init --name k99-opcheck --gpus all --ipc=host --network none \
    -e CUTE_DSL_ARCH=sm_121a -e XDG_CACHE_HOME=/tmp/cache -e FLASHINFER_CACHE_DIR=/tmp/cache/flashinfer -e B12X_COMPILE_WORKERS=2 \
    -e B12X_DYNAMIC_TILE_MN=32x128 -e PYTHONUNBUFFERED=1 -v "$K:/k99:ro" -v "$K98:/k98:ro" -v "$FISRC:/fi:ro" -v "$CAPD:/cap:ro" \
    -v "$HFM:$HFM:ro" -v "$o:/out" "${w[@]}" --entrypoint "${c[0]}" "$IMG" "${c[@]:1}" > "$o/opcheck.log" 2>&1
  docker rm -f k99-opcheck > /dev/null 2>&1
  grep -m1 '^OPCHECK ' "$o/opcheck.log" || echo "OPCHECK FAIL: no verdict (exit; $o/opcheck.log)"; }
stage() { # arm cache on both Sparks: control cache dirs (no clobber), the mod, the 5c4058bd tree
  local h
  for h in localhost $H2; do
    hx $h "set -e; mkdir -p $ARC; for d in b12x triton inductor flashinfer cute_dsl flash_attn_cute_dsl torch_extensions numba; do
             [ -d $CRC/\$d ] && cp -an $CRC/\$d $ARC/; done; du -sh $ARC | cut -f1" > "$RES/stage-$h.txt" 2>&1 || { log "stage $h failed"; return 1; }
  done
  rsync -a "$K/mod/" "$ARC/mod/" && rsync -a --exclude .git "$FISRC/" "$ARC/fi-src/" \
    && rsync -a -e "ssh -o ConnectTimeout=10" "$K/mod/" "$H2:$ARC/mod/" && rsync -a --exclude .git -e "ssh -o ConnectTimeout=10" "$FISRC/" "$H2:$ARC/fi-src/" \
    || { log "stage: rsync of the mod / FlashInfer tree failed"; return 1; }
  local a b; a=$(cd "$ARC" && find mod fi-src/flashinfer -type f -name '*.py' | sort | xargs md5sum | md5sum)
  b=$(on2 "cd $ARC && find mod fi-src/flashinfer -type f -name '*.py' | sort | xargs md5sum | md5sum")
  [ "$a" = "$b" ] || { log "stage: mod / FlashInfer tree differ between the Sparks"; return 1; }
  [ "$(on2 "ls $ARC/b12x/compile/preparation/" | grep '\.json$' | sort)" = "$(ls "$ARC/b12x/compile/preparation/" | grep '\.json$' | sort)" ] || { log "stage: plan file sets differ between the Sparks"; return 1; }   # .json only: a stale .lock on one Spark failed the 2026-10-13 run
  [ "$(nrec "$ARC/b12x/compile/preparation/$PLAN")" = $NREC ] || { log "stage: arm cache plan $PLAN is not $NREC records"; return 1; }
  { ls "$ARC/b12x/compile/preparation/"; docker run --rm --network none --entrypoint ls "$IMG" /opt/b12x-seed/preparation; } \
    | grep '\.json$' | grep -vx "$PLAN" | sort -u > "$K/plans-before.txt"; }
bake() {
  local t d ok=
  for t in 1 2 3; do
    d=$RES/bake/prebake-try$t; mkdir -p "$d"; st "running: prebake try $t (arm, 6 GiB KV pool)"; regard
    if tboot "$RD/prebake-2x.yaml" "$d" 1; then   # one cold 16K prompt: its 5728-row chunks must take the FlashInfer path
      timeout -k 30 600 python3 "$COLD" run --base http://localhost:8000 --corpus "$CORPUS" --lens 16384 --reps 1 \
        --seed k99b --cskip 3600000 --out "$d/bake16k.json" > "$d/bake16k.txt" 2>&1
      keep_logs2 "$d"; stop_all; ok=1
      log "prebake try $t: $(cat "$d/boot.txt"), $(measured "$d/serve.log") plans measured"; break; fi
    keep_logs2 "$d"; stop_all; log "prebake try $t failed"
  done
  [ -n "$ok" ] || { log "prebake: 3 tries failed"; return 1; }
  grep -qx "FLASHINFER_CACHE_DIR=$ACACHE/flashinfer" "$d/env.txt" && grep -qx "VLLM_K99_FI_MOE_MIN_TOKENS=$MIN" "$d/env.txt" \
    || { log "bake: arm env not as written ($d/env.txt)"; return 1; }
  fipath "$d" || { log "bake: the mod did not take the FlashInfer path on both ranks ($d)"; return 1; }
  d=$RES/bake/arm-full; mkdir -p "$d"; st "running: bake arm full pool"; regard
  tboot "$RD/arm-2x.yaml" "$d" 1 || { keep_logs2 "$d"; stop_all; log "bake full-pool boot failed"; return 1; }
  keep_logs2 "$d"; stop_all; regard
  plans_ok "$d"; log "bake full pool: $(cat "$d/boot.txt"); $(tr '\n' ' ' < "$d/plans.txt")"
  for t in "$RES"/bake/*/; do [ -s "$t/boot.txt" ] || continue; plans_ok "$t" || { log "bake: plan identity check failed ($t/plans.txt)"; return 1; }; done; }
run_pass() { # dir: warm-up, cold TTFT cells, acceptance, benchy
  local d=$1
  ( child timeout -k 30 900 python3 "$COLD" run --base http://localhost:8000 --corpus "$CORPUS" --lens 16384,65536 --reps 1 \
      --seed k99w --cskip 3000000 --out "$d/warm.json" > "$d/warm.txt" 2>&1 )
  ( child timeout -k 30 2700 python3 "$COLD" run --base http://localhost:8000 --corpus "$CORPUS" --lens $LENS --reps $REPS \
      --seed k99 --cskip 0 --out "$d/cold.json" > "$d/cold.txt" 2>&1 ); log "$(basename "$d"): cold exit=$?"
  pos > "$d/pos-c8.before"
  ( cd "$R" && child timeout -k 30 500 python3 scripts/depth_decode_probe.py --base "http://$H1:8000" --label fresh-c8 --corpus "$CORPUS" \
      --concurrency 8 --repeats 2 --temperature 0 --out "$d/probe-fresh-c8.json" --depth 256 --new 2048 --offset 418916 --max-tokens 512 > "$d/probe-fresh-c8.log" 2>&1 )
  pos > "$d/pos-c8.after"
  ( cd "$R" && child timeout -k 30 1800 uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$H1:8000/v1" --model $MODEL \
      --tokenizer "$SNAP" --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 --concurrency 1 4 8 --runs 3 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$H1:8000/metrics" \
      --save-result "$d/task.csv" > "$d/task.log" 2>&1 ); log "$(basename "$d"): benchy exit=$?"
  [ "$(health localhost)" = 200 ] || { echo "server gone after the pass" > "$d/DIED"; log "$(basename "$d"): server DIED"; }; }
boot2() { # dir recipe: one counted boot (warm only)
  local d=$1 rc; mkdir -p "$d"; st "running: screen $(basename "$d")"
  tboot "$2" "$d" 0; rc=$?
  if [ $rc -ne 0 ]; then keep_logs2 "$d"; [ $rc -eq 2 ] && echo cold > "$d/COLD" || echo failed > "$d/FAILED"; stop_all; return 0; fi
  keep_logs2 "$d"
  case $(basename "$d") in arm*) plans_ok "$d" ;; *) [ "$(measured "$d/serve.log")" = 0 ] ;; esac \
    || { echo "plans: $(tr '\n' ' ' < "$d/plans.txt" 2>/dev/null) measured $(measured "$d/serve.log")" > "$d/COLD"; stop_all
         FINAL="FAILED: k99 $(basename "$d") measured b12x plans ($RES)"; exit 1; }
  run_pass "$d"; keep_logs2 "$d"; stop_all; }
gate() { # -> GV, GSUM (k92 gate on one arm boot)
  local g=$RES/gate t
  st "running: gate"; mkdir -p "$g"
  if tboot "$RD/arm-2x.yaml" "$g" 0; then
    keep_logs2 "$g"
    plans_ok "$g" || { stop_all; FINAL="FAILED: k99 gate boot measured b12x plans ($RES)"; exit 1; }
    ( cd "$R" && child timeout 900 python3 scripts/straggler_probe.py 5 6 7 8 12 16 > "$g/straggler.log" 2>&1 )
    child timeout 7200 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://localhost:8000 --model $MODEL > "$g/hardmode.log" 2>&1
    child timeout 3600 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://localhost:8000/v1 --model $MODEL --scenarios TC-45 --trials 5 > "$g/tc45.log" 2>&1
    ( cd "$R" && child timeout 5400 python3 scripts/fidelity_probe.py --base http://localhost:8000 --model $MODEL --depths 8000,32000,64000,128000 --out "$g/fidelity.json" > "$g/fidelity_probe.txt" 2>&1 )
    for sd in 11 13; do ( cd "$R" && child timeout 2700 python3 scripts/fidelity_probe.py --base http://localhost:8000 --model $MODEL --depths 128000 --seed $sd --out "$g/fidelity-seed$sd.json" > "$g/fidelity-seed$sd.txt" 2>&1 ); done
    keep_logs2 "$g"; stop_all
    { fipath "$g" && grep -q 'K99FI calls=1000 ' "$g/serve.log" && grep -q 'K99FI calls=1000 ' "$g/serve-node1.log"; } \
      || echo "gate boot: fewer than 1000 FlashInfer calls logged on a rank (boot-time calls only?)" >> "$g/notes.txt"
    t=$(mktemp -d); ln -s "$g" "$t/cand1"; ln -s "$g" "$t/cand2"
    GV=$(python3 -c "import sys; sys.path.insert(0, sys.argv[1]); import k61; b, hm = k61.gate(sys.argv[2]); print('PASS' if not b else 'FAIL (' + '; '.join(b) + ')')" "$K70" "$t" 2>&1); rm -f "$t/cand1" "$t/cand2"; rmdir "$t"
    [ -s "$g/notes.txt" ] && GV="FAIL (gate boot did not take the FlashInfer path)"
    GSUM=$(python3 - "$g" <<'PY'
import os, re, sys
g = sys.argv[1]
rd = lambda f: open(f"{g}/{f}", errors="replace").read() if os.path.exists(f"{g}/{f}") else ""
hm = re.findall(r"Quality:\s*(\d+)/100", rd("hardmode.log"))
tc = re.findall(r"Score:\s*([0-9.]+)", rd("tc45.log"))
fid = re.findall(r"^depth\s+(\d+) .*?exact\s+(\d+)", rd("fidelity_probe.txt") + rd("fidelity-seed11.txt") + rd("fidelity-seed13.txt"), re.M)
pre = re.findall(r"preemptions\s*\+?(\d+)", rd("straggler.log"))
print(f"hardmode {hm[-1] if hm else 'missing'}; TC-45 {tc[-1] if tc else 'missing'}; long-context retrieval "
      f"{' '.join(f'{d}:{e}/20' for d, e in fid) or 'missing'}; straggler preemptions {' '.join(pre) or 'missing'}")
PY
)
  else keep_logs2 "$g"; stop_all; GV="FAIL (gate boot failed)"; GSUM="gate boot failed"; fi
  { echo; echo "== gate (arm, one TP=2 boot): $GV"; echo "$GSUM"; } >> "$RES/k99.txt"; log "gate: $GV; $GSUM"; }

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
mkdir -p "$RES"; echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
job_begin
eval "$(trap -p EXIT | sed "s/^trap -- '/trap -- 'docker rm -f k99-opcheck > \/dev\/null 2>\&1; /")"   # the op-check container goes on any exit
RD=$RES/recipes
echo "k99 $(TZ=Europe/Bucharest date '+%F %T %Z'): FlashInfer 5c4058bd fused MoE for prefill calls >= $MIN rows vs the shipped 2x v2.1.0 (same image)" > "$RES/k99.txt"
st "running: registry update"
timeout 300 sparkrun registry update $REG2 >> "$RES/$J.log" 2>&1 < /dev/null 9>&-
st "running: preflight"
{ host_checks && cpucheck > "$RES/cpucheck.txt" && tail -1 "$RES/cpucheck.txt" | grep -qx 'CPUCHECK ok'; } > "$RES/preflight.txt" 2>&1 \
  || { FINAL="FAILED: preflight ($RES/preflight.txt, $RES/cpucheck.txt)"; exit 1; }
mkdir -p "$RES/job/mod"; cp "$K"/run.sh "$K"/*.py "$RES/job/"; cp "$K"/mod/*.py "$RES/job/mod/"; cat "$RD/sources.txt" >> "$RES/k99.txt"

st "running: stop 2x"; stop_all
st "running: P0 op check (dgx-01, FlashInfer via the mod vs b12x, I320)"
OPV=$(opcheck "$RES/opcheck"); log "op check: $OPV"
sed -n '/^layer      M/,$p' "$RES/opcheck/opcheck.log" | sed 's/^/  /' > "$RES/opcheck/summary.txt"
{ echo; echo "== P0 op check (shipped image as is: cutlass-dsl 4.7.0, flashinfer 0.7.0 + the 5c4058bd tree through the mod)"
  grep -E '^(sampler|5c4058bd|layer [0-9]+ (bytes|inputs))' "$RES/opcheck/opcheck.log"; cat "$RES/opcheck/summary.txt"; } >> "$RES/k99.txt"
if [ "${OPV#OPCHECK PASS}" = "$OPV" ]; then
  st "running: P0 diagnostic rerun with cutlass-dsl 4.7.1"
  OPV2=$(opcheck "$RES/opcheck-dsl471" dsl471); log "op check with cutlass-dsl 4.7.1: $OPV2"
  echo "diagnostic rerun with cutlass-dsl 4.7.1 (k98 wheels): $OPV2" >> "$RES/k99.txt"
  FINAL="FAILED: k99 op check: ${OPV#OPCHECK } | with cutlass-dsl 4.7.1: ${OPV2#OPCHECK } ($RES/k99.txt)"; exit 1; fi

st "running: stage arm cache"
stage || { FINAL="FAILED: k99 stage ($RES/stage-*.txt, $J.log)"; exit 1; }
bake || { FINAL="FAILED: k99 bake ($RES/bake)"; exit 1; }
{ echo; echo "bake:"; for b in "$RES"/bake/*/; do echo "  $(basename "$b"): $(cat "$b/boot.txt" 2>/dev/null || echo failed); $(tr '\n' ' ' < "$b/plans.txt" 2>/dev/null)"; done; } >> "$RES/k99.txt"

S=$RES/2x; mkdir -p "$S"
for item in $ORDER; do who=${item%:*}; k=${item#*:}
  if [ $who = arm ]; then boot2 "$S/arm-p$k" "$RD/arm-2x.yaml"; else boot2 "$S/ctl-p$k" "$RD/ctl-2x.yaml"; fi; done
st "running: judge"
python3 "$K/k99_report.py" "$S" $REPS > "$RES/judge-2x.txt" 2>&1
RU=$(sed -n 's/^RULE=//p' "$RES/judge-2x.txt" | tail -1)
{ echo; echo "== screen ($ORDER): RULE=${RU:-none}"; cat "$RES/judge-2x.txt"; } >> "$RES/k99.txt"
GV="not run (RULE=${RU:-none}; the gate runs on PASS or FLAT)"; GSUM=
case "$RU" in PASS|FLAT) [ "$GATE" = 0 ] && GV="not run (GATE=0)" || gate ;; esac
for l in dgx01 dgx02; do echo "guard $l: min MemAvailable $(awk '$2 ~ /^[0-9]+$/' "$RES/guard-$l.log" | sort -k2 -n | head -1 | awk '{printf "%.2f GiB", $2/1024}')"; done >> "$RES/k99.txt"
TT=$(grep -E '^  (16|64|128)K: ctl' "$RES/judge-2x.txt" | sed -E 's/^  ([0-9]+K): .*-> ([-+][0-9.]+%) \(([-+][0-9.]+ s).*\) ([a-zA-Z]+)$/\1 \2 \4/' | tr '\n' ' ')
KVC=$(sed -n 's/^  KV pool arm vs ctl: //p' "$RES/judge-2x.txt")
FIM=$(cat "$S"/arm-p*/serve.log 2>/dev/null | grep -oE 'K99FI M=[0-9]+' | cut -d= -f2 | sort -nu | tr '\n' ' ')
FIMS=$(echo $FIM | awk '{ if (NF) printf "%d distinct row counts from %d to %d", NF, $1, $NF; else printf "none" }')
echo "FlashInfer row counts on the arm boots (rank 0, distinct): $FIMS: $FIM" >> "$RES/k99.txt"
case "$RU" in
  EMPTY|"") FINAL="FAILED: k99 zero-measurement guard: RULE=${RU:-none} ($RES/judge-2x.txt)" ;;
  *) FINAL="DONE: k99 RULE=$RU gate=${GV%% (*} ttft ${TT% } kv ${KVC%% (*} ($RES/k99.txt)" ;; esac

# comment for the poster (public: first person, no LAN addresses)
{ echo "k99-2x-fi5168-prefill-moe-ab result: I routed the 2x's large prefill chunks to FlashInfer's SM12x fused MoE at \`${FISHA:0:10}\` (merge of FlashInfer PR 5168) and kept b12x for decode and for chunks below $MIN rows, on the shipped 2x v2.1.0 image with the same weights and plan selections. The 2x prefills a long prompt in 5,728-row chunks (2 x its 2,864-token attention block) at budget 8192, so every full chunk takes the FlashInfer path; decode steps (at most 80 rows) stay on b12x. On the arm boots FlashInfer served calls of $FIMS (rank 0). FlashInfer reads b12x's prepared expert storage in place; the image's own flashinfer stays in use for sampling. The operator probe behind this (k98, I320 rows) is on ursuciprian/qwen3.8-flash-next-1x-dgx-spark#27."
  echo; echo "Op check before any boot (one Spark, routed-MoE layers 5 and 47 at the I320 rank shard, real routing from a cold 16K prompt): \`$OPV\`."
  echo '```'; cat "$RES/opcheck/summary.txt"; echo '```'
  echo; echo "Screen on the pair, boots in the order $ORDER (2 per side), cold prompts identical on every boot: \`RULE=${RU:-none}\`. Gate on one arm boot: $GV.${GSUM:+ $GSUM.}"
  echo '```'; grep -vE '^(fail|empty|better):' "$RES/judge-2x.txt" | grep -v '^RULE='; echo '```'
  echo; case "$RU" in
    PASS) if [ "${GV%% *}" = PASS ]; then echo "The FlashInfer prefill path makes long cold prompts faster on the 2x beyond boot noise, with decode and MTP acceptance unchanged. The KV pool line above is its memory cost. Nothing ships from this job."; else echo "The FlashInfer prefill path makes long cold prompts faster on the 2x beyond boot noise, but the quality gate did not pass, so the 2x stays on b12x for prefill."; fi ;;
    FLAT) echo "The FlashInfer prefill path does not move cold TTFT beyond boot noise on the 2x in this run, so the 2x stays on b12x for prefill." ;;
    *) echo "This run does not support the FlashInfer prefill path on the 2x (see the lines above), so the 2x stays on b12x." ;; esac
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"
if grep -qE '192\.168\.|dgx-0[0-9]\.local' "$K/comment.md" || grep -qiEf "$PRIV" "$K/comment.md"; then
  mkdir -p "$K/private"; mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k99 comment.md held a LAN address or a private pattern (not posted; $RES)"; exit 1; fi
cp "$K/comment.md" "$RES/comment.md"
exit 0
