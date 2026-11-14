# k122/k123 profile helpers (2026-11-14), sourced after lib.sh by k122/run.sh and k123/run.sh (J, K, RES set).
# Pieces reused, not rewritten:
#   mods/vllm-decode-profiler  copy of tp-2 archive/mods/vllm-decode-profiler (origin/main) plus per-step markers
#                              (record_function local_prof_step_<n> inside a window only): rank-local torch.profiler
#                              around GPUModelRunner.execute_model, armed by a file in VLLM_LOCAL_PROF_TRIGGER_DIR,
#                              VLLM_LOCAL_PROF_STEPS steps, trace /cache/runtime/prof/<label>/rank<tp rank>.json.
#                              No collective RPC: the built-in --profiler-config path still has the follower-node
#                              assert in vLLM 10d0acc8 (multiproc_executor.py collective_rpc), so it stays off TP=2.
#   trig/collect/probe         r8_driver.sh stage_prof (tp-2 feat/r8 scripts): trigger file on both ranks mid-decode
#                              via depth_decode_probe.py --trigger-cmd, traces copied per rank once their size settles,
#                              step_breakdown.py; meas counts b12x plan measurements on the running ranks
#   mkprof                     tp2ab.sh mkarm style: every recipe edit counted, the shipped recipe file is only read
# The profiler boots use their own runtime cache roots ($CRC/<own>: copies of the shipped b12x/triton/... dirs and of
# the shipped VLLM_CACHE_ROOT), so no shipped cache dir is written. Traces land in $CRC/prof/$J-* (not a cache).
CRC=$HOME/.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923
REV=03f4a0570496bbe741189f4a2a7d611679e85406
IMG2=ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm    # 2x v2.2.0
IMG1=ghcr.io/ursuciprian/spark-vllm-b12x:k106s-20261017-a1ad1b69-5dad364d-10d0acc8-warm    # 1x v2.4.0
CTL_ROOT2=/cache/runtime/vllm-2x-21e0b201-10d0acc8; CTL_ROOT1=/cache/runtime/vllm-b12x-a1ad1b69-10d0acc8
REGF1=$HOME/.cache/sparkrun/registries/qwen38-flashnext-1x/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml
REGF2=$(grep -rlx --include='*.yaml' "name: qwen3.8-flash-next-2x-dgx-spark" "$(readlink -f "$HOME/.cache/sparkrun/registries/qwen38-flashnext")/recipes" 2>/dev/null | head -1)
PROBE=$R/scripts/depth_decode_probe.py; CORPUS=$R/results/corpus-code.txt; COLD=$BL/k101/cold.py
MOD=$BL/k122/mods/vllm-decode-profiler; MARK='local step profiler (mods/vllm-decode-profiler)'
MR=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py
CACHE_DIRS="b12x triton inductor flashinfer cute_dsl flash_attn_cute_dsl torch_extensions numba"
on2() { ssh -n -o ConnectTimeout=10 $H2 "$@"; }
hx() { if [ "$1" = localhost ]; then bash -c "$2"; else on2 "$2"; fi; }
c0() { docker ps --filter name=sparkrun --format '{{.Names}}' | grep -v node_1 | head -1; }
regs() { timeout 300 sparkrun registry update qwen38-flashnext > /dev/null 2>&1 < /dev/null 9>&-
  timeout 300 sparkrun registry update qwen38-flashnext-1x > /dev/null 2>&1 < /dev/null 9>&-; true; }

mkprof() { # src dst name own ctl_root img steps: shipped recipe -> profiler recipe (own cache, mod, trigger dir)
  python3 - "$@" <<'PY'
import sys
from collections import Counter
src, dst, name, own, root, img, steps = sys.argv[1:8]
lines = open(src).read().split("\n")
def one(pred, what):
    i = [k for k, l in enumerate(lines) if pred(l)]
    if len(i) != 1:
        sys.exit(f"mkprof: need exactly one {what} in {src}, found {len(i)}")
    return i[0]
n = one(lambda l: l.startswith("name: "), "name line")
one(lambda l: l == f"container: {img}", f"container {img}")
one(lambda l: l == "model_revision: 03f4a0570496bbe741189f4a2a7d611679e85406", "model_revision")
rt = one(lambda l: l == "runtime: vllm", "runtime line")
e = one(lambda l: l == "env:", "env: block")
r = one(lambda l: l == f'  VLLM_CACHE_ROOT: "{root}"', "shipped VLLM_CACHE_ROOT line")
one(lambda l: l == '  VLLM_USE_V2_MODEL_RUNNER: "1"', "VLLM_USE_V2_MODEL_RUNNER (the mod wraps the V2 runner)")
if any(l.startswith("mods:") or "XDG_CACHE_HOME:" in l or "VLLM_LOCAL_PROF" in l for l in lines):
    sys.exit("mkprof: the shipped recipe already has mods, XDG_CACHE_HOME or a profiler variable")
out = lines[:]
out[n] = f"name: {name}"
out[r] = f'  VLLM_CACHE_ROOT: "/cache/runtime/{own}/vllm"'
add = [f'  XDG_CACHE_HOME: "/cache/runtime/{own}"',
       f'  VLLM_LOCAL_PROF_TRIGGER_DIR: "/cache/runtime/prof-trigger-{own}"',
       f'  VLLM_LOCAL_PROF_STEPS: "{steps}"']
out[e + 1:e + 1] = add
out[rt + 1:rt + 1] = ["mods:", "  - vllm-decode-profiler"]
a, b = Counter(lines), Counter(out)
if sum((a - b).values()) != 2 or sum((b - a).values()) != 2 + len(add) + 2:
    sys.exit("mkprof: unexpected diff")
open(dst, "w").write("\n".join(out))
PY
}
mkrd() { # recipe dir with the mod beside the recipes (sparkrun resolves <recipe_dir>/mods/<name>)
  mkdir -p "$RD/mods" && rm -rf "$RD/mods/vllm-decode-profiler" && cp -r "$MOD" "$RD/mods/"; }

stage() { # host own ctl_root: own cache root from copies of the shipped dirs, fresh trigger dir, old traces of $J gone
  hx $1 "set -e; mkdir -p $CRC/$2; for d in $CACHE_DIRS; do [ -d $CRC/\$d ] && cp -an $CRC/\$d $CRC/$2/; done
    [ -d $CRC/$2/vllm ] || cp -a $CRC/${3#/cache/runtime/} $CRC/$2/vllm
    rm -rf $CRC/prof-trigger-$2; mkdir -p $CRC/prof-trigger-$2; rm -rf $CRC/prof/$J-*; du -sh $CRC/$2 | cut -f1"; }

pboot() { # recipe dir cap solo(0|1): sparkrun run only; 0 = health 200, 1 = failed / timed out
  local rec=$1 d=$2 s n
  stop_all; mkdir -p "$d"; cp "$rec" "$d/recipe.yaml"
  if [ "$4" = 1 ]; then ( cd "$(dirname "$rec")" && timeout -k 60 900 sparkrun run "$(basename "$rec")" --hosts $H1 --solo --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  else ( cd "$(dirname "$rec")" && timeout -k 60 900 sparkrun run "$(basename "$rec")" --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-; fi
  s=$(date +%s)
  while :; do
    [ "$(health localhost)" = 200 ] && { echo "boot $(( $(date +%s) - s ))s $(docker exec "$(c0)" grep -m1 -oE 'Directly load AOT compilation|Dynamo bytecode transform time' /tmp/sparkrun_serve.log 2>/dev/null)" > "$d/boot.txt"; return 0; }
    n=$(c0)
    if [ $(( $(date +%s) - s )) -gt 180 ] && { [ -z "$n" ] || docker exec "$n" grep -qE 'Worker failed with error|EngineCore failed to start|Engine core initialization failed' /tmp/sparkrun_serve.log 2>/dev/null; }; then return 1; fi
    [ $(( $(date +%s) - s )) -ge "$3" ] && return 1
    sleep 10; done; }
modok() { # tp: the mod applied on every rank
  docker exec "$(c0)" grep -q "$MARK" $MR || return 1
  [ "$1" = 1 ] || on2 "c=\$(docker ps --format '{{.Names}}' | grep node_1 | head -1); docker exec \$c grep -q '$MARK' $MR"; }
logs() { # dir tp
  docker exec "$(c0)" cat /tmp/sparkrun_serve.log > "$1/serve.log" 2>&1
  [ "$2" = 1 ] || on2 'c=$(docker ps -a --format "{{.Names}}" | grep node_1 | head -1); [ -n "$c" ] && docker exec $c cat /tmp/sparkrun_serve.log' > "$1/serve-node1.log" 2>&1; }

trig() { # own label tp [stamp file] -> shell command that arms the profiler on every rank (one new file name per arm);
  # with a stamp file it also writes the arming epoch there (window.py). Rank 1 (ssh) is armed first, then rank 0:
  # rank 1's window no longer opens after rank 0's profiler has slowed the step; the ranks still differ by the ssh
  # latency (a few steps), which the per-rank step markers make visible.
  local c="n=go-\$(date +%s%N); ${4:+date +%s.%N > $4; }"
  [ "$3" = 1 ] && echo "${c}echo $2 > $CRC/prof-trigger-$1/\$n" \
    || echo "${c}ssh -n -o ConnectTimeout=10 $H2 \"echo $2 > $CRC/prof-trigger-$1/\$n\"; echo $2 > $CRC/prof-trigger-$1/\$n"; }
pos() { curl -s -m 10 localhost:8000/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens(_per_pos)?_total'; }
probe() { # dir tag depth conc offset [extra...]: r8_driver.sh probe, T=0, 2 repeats, 512 new tokens
  local d=$1 tag=$2 depth=$3 c=$4 off=$5; shift 5
  pos > "$d/pos-$tag.before"
  timeout -k 30 900 python3 "$PROBE" --base http://localhost:8000 --label "$tag" --corpus "$CORPUS" --depth "$depth" --new 2048 \
    --concurrency "$c" --repeats 2 --offset "$off" --max-tokens 512 --temperature 0 --out "$d/probe-$tag.json" "$@" > "$d/probe-$tag.log" 2>&1
  local rc=$?; pos > "$d/pos-$tag.after"; return $rc; }
fsize() { # host file -> size in bytes, 0 when missing
  if [ "$1" = localhost ]; then stat -c %s "$2" 2>/dev/null || echo 0; else on2 "stat -c %s $2 2>/dev/null || echo 0"; fi; }
settled() { # host file timeout_s -> 0 once the file has the same nonzero size on two polls 3 s apart
  # (export_chrome_trace writes in place: a size check alone can copy a half-written trace)
  local s0=-1 s1 i
  for i in $(seq 1 $(( $3 / 3 ))); do s1=$(fsize $1 "$2"); [ "$s1" != 0 ] && [ "$s1" = "$s0" ] && return 0; s0=$s1; sleep 3; done
  return 1; }
collect() { # label outdir tp [steps for step_breakdown, empty = none] -> prints the count of usable results
  # (breakdown-rank*.json with steps, else rank*.json.gz); t_end = mtime of the rank-0 trace (window end + export)
  local i r f=$CRC/prof/$1; mkdir -p "$2"
  if settled localhost "$f/rank0.json" 180; then cp "$f/rank0.json" "$2/" 2>> "$2/collect.log"; date -r "$f/rank0.json" +%s.%N > "$2/t_end"
  else echo "no settled rank0 trace $1" >> "$2/collect.log"; fi
  if [ "$3" != 1 ]; then settled $H2 "$f/rank1.json" 120 && scp -q "$H2:$f/rank1.json" "$2/" 2>> "$2/collect.log" \
    || echo "no settled rank1 trace $1" >> "$2/collect.log"; fi
  if [ -n "${4:-}" ]; then for r in 0 1; do [ -f "$2/rank$r.json" ] && python3 "$BL/k122/step_breakdown.py" "$4" "$2/rank$r.json" \
      --json "$2/breakdown-rank$r.json" > "$2/breakdown-rank$r.txt" 2>&1 || rm -f "$2/breakdown-rank$r.json"; done; fi
  gzip -f "$2"/rank*.json 2>/dev/null
  if [ -n "${4:-}" ]; then find "$2" -name 'breakdown-rank*.json' -size +0 | wc -l; else ls "$2"/rank*.json.gz 2>/dev/null | wc -l; fi; }
meas() { # tp -> b12x plans measured so far ("<n> measured" summed over the serve logs of the running ranks)
  local a b=0
  a=$(docker exec "$(c0)" cat /tmp/sparkrun_serve.log 2>/dev/null | grep -ohE '\b[0-9]+ measured' | awk '{s+=$1} END {print s+0}')
  [ "$1" = 1 ] || b=$(on2 "c=\$(docker ps --format '{{.Names}}' | grep node_1 | head -1); [ -n \"\$c\" ] && docker exec \$c cat /tmp/sparkrun_serve.log" 2>/dev/null \
    | grep -ohE '\b[0-9]+ measured' | awk '{s+=$1} END {print s+0}')
  echo $(( ${a:-0} + ${b:-0} )); }

ancheck() { # CPU check of the decode analysis on a synthetic trace (step_breakdown.py + r8_profile_table.py)
  local t; t=$(mktemp -d)
  python3 - "$t" <<'PY' || { rm -rf "$t"; return 1; }
import json, os, sys
t = sys.argv[1]
ev = []
for s in range(60):
    b = s * 1000
    ev += [{"ph": "X", "cat": "kernel", "name": "b12xmoe_dynamic", "ts": b, "dur": 300, "args": {"graph id": 7}},
           {"ph": "X", "cat": "kernel", "name": "blockscaled_mxfp8", "ts": b + 300, "dur": 200, "args": {"graph id": 7}},
           {"ph": "X", "cat": "kernel", "name": "mtp_draft_gemm", "ts": b + 520, "dur": 100, "args": {"graph id": 9}},
           {"ph": "X", "cat": "kernel", "name": "argmax_kernel", "ts": b + 650, "dur": 50, "args": {}}]
for w in ("tp2", "prof/dgx01"):
    d = os.path.join(t, w, "prof-fresh-c1"); os.makedirs(d)
    json.dump({"traceEvents": ev}, open(os.path.join(d, "rank0.json"), "w"))
PY
  local ok=0 w
  for w in tp2 prof/dgx01; do python3 "$BL/k122/step_breakdown.py" 60 "$t/$w/prof-fresh-c1/rank0.json" --json "$t/$w/prof-fresh-c1/breakdown-rank0.json" > "$t/bd.txt" 2>&1 \
    && grep -q 'NVFP4 MoE' "$t/bd.txt" || { echo "step_breakdown.py failed on the synthetic trace"; cat "$t/bd.txt"; ok=1; }; done
  python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); assert abs(j["step_ms"]-1.0)<0.01 and abs(j["summary_us"]["NVFP4 MoE"]-300)<1, j' \
    "$t/tp2/prof-fresh-c1/breakdown-rank0.json" || { echo "step_breakdown.py numbers off on the synthetic trace"; ok=1; }
  python3 "$BL/k122/r8_profile_table.py" "$t" --tp2 "$t/tp2" > "$t/table.md" 2>&1 && grep -q '| NVFP4 MoE | 0.30 | 0.30 |' "$t/table.md" \
    || { echo "r8_profile_table.py failed on the synthetic layout"; cat "$t/table.md"; ok=1; }
  rm -rf "$t"; return $ok; }
