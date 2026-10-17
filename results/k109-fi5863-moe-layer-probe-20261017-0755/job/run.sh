#!/usr/bin/env bash
# k109-fi5863-moe-layer-probe (2026-10-16, tp-2 #174): microbench of one routed-MoE layer, no serving. Our pinned b12x
# (the 2x v2.2.0 image as is: b12x 21e0b201, CuTe DSL 4.7.0) against FlashInfer main's SM12x CuTe DSL MoE backends:
# W4A4 (flashinfer-ai/flashinfer#5863, SM12xNvfp4Config, merged 10-16 as 397b9607) and W4A16 (#5735,
# SM12xNvfp4Bf16Config, 0be3eb28), FlashInfer main 38e6b894 pre-cloned in $K/fi-src. Geometries I640 (1x) and I320
# (2x TP shard), E512 H2560 top-10, M 1/5/10/20/40/2048/3024/6048/8192; real skewed routing from the k97 capture (a
# cold 16K prompt's layer 5/47 routed-MoE inputs, ids and weights) plus uniform; CUDA graph replays with a cold L2;
# kept GPU memory after the whole sweep (k99 lost 15% of the KV pool to FlashInfer scratch). Harness: k98's (moe98.py
# -> moe109.py, see its docstring for the variants, the scale contract and the serving rule, fixed before the run).
#   P0  b12x container (2x v2.2.0 image as is, --network none, <= 45 min): b12x-m32 (anchor), b12x-auto, b12x-wm;
#       saves the outputs (rows <= 256) to $K/num for the FlashInfer contract check.
#   P1  FlashInfer container (same image, throwaway, <= 45 min): staged nvidia-cutlass-dsl 4.7.1 wheels installed, the
#       FlashInfer source on PYTHONPATH; numerics first (vs FP32 and vs the b12x output), then timing and memory.
#   Report: numerics gate, anchor (tp1 vs k81), speed per M and routing, memory x48, serving rule per backend and
#   geometry. A serving arm follows only on GO (>= 5%); none is queued from here.
# ETA ~1 h 40 min: stop ~1 + b12x <= 45 + FlashInfer <= 45 (JIT of the CuTe DSL kernels per bucket included) + report.
# Output: $RES/p0 (b12x), $RES/p1 (FlashInfer), k109.txt; comment.md (tp-2 #174) + RESULT here. Measure only.
# Contract (backlog/runner.sh): dgx01 lock only (GPU_LOCK_HELD=1); exit 75 untouched while another job is busy; STATE
# first line DONE/FAILED (lib.sh EXIT trap; a single-Spark job leaves the 2x restore to the runner).
# Usage: bash run.sh --dry-run | via the runner. Stop: kill -TERM <pid>.
NEEDS="dgx01"
set -u
J=k109; K=$HOME/GEN-AI/backlog/k109
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k109-fi5863-moe-layer-probe-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
IMG=ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm   # 2x v2.2.0, b12x 21e0b201
IMG_ID=sha256:9c05231a900dda1c301b410b90dfaf0611b1804ba4ed3d2f39c534f98e50a840
HFM=$HOME/.cache/huggingface/hub/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE
CKPT=$HFM/snapshots/03f4a0570496bbe741189f4a2a7d611679e85406   # expert weights are the same in every GDN-MSE revision
FISHA=38e6b8944fe8b6497d793ef21fbaadd797ea39d8; FI_NEED="397b9607c83cf201f3c44684c555c3698869b6bd 0be3eb28ca78b3e6d8a9d125eab09a12e96940aa"
CAPD=$BL/k97/capture; NUMD=$K/num
PRIV=$(cat "$BL/k98/private.pat" 2>/dev/null)
[ -n "$PRIV" ] || { echo "missing $BL/k98/private.pat" >&2; exit 2; }

drun() { # b12x|fi gpu|cpu, moe109 args (k98)
  local kind=$1 g=(); [ "$2" = gpu ] && g=(--gpus all); shift 2
  local c=(); [ -d "$CAPD" ] && c=(-v "$CAPD:/cap:ro")
  local common=(--rm --init --name k109-$kind "${g[@]}" --ipc=host -e CUTE_DSL_ARCH=sm_121a -e XDG_CACHE_HOME=/tmp/cache
    -e B12X_COMPILE_WORKERS=2 -e PYTHONUNBUFFERED=1 -v "$K:/k109:ro" -v "$HFM:$HFM:ro" -v "$NUMD:/num" "${c[@]}" ${OUT:+-v "$OUT:/out"})
  if [ $kind = b12x ]; then docker run "${common[@]}" --network none --entrypoint python3 "$IMG" /k109/moe109.py "$@"
  else docker run "${common[@]}" --network none -v "$K/fi-src:/src:ro" -v "$K/wheels:/wheels:ro" --entrypoint bash "$IMG" /k109/fi_inner.sh "$@"; fi; }
fill() { # template out KEY=file...: {KEY} -> file contents (stdlib)
  python3 - "$@" <<'PY'
import sys
t = open(sys.argv[1]).read()
for kv in sys.argv[3:]:
    k, f = kv.split("=", 1)
    t = t.replace("{" + k + "}", open(f).read().rstrip("\n"))
open(sys.argv[2], "w").write(t)
PY
}
host_checks() { # no containers: safe while other jobs run
  local ok=0 f n c
  for f in "$K/moe109.py" "$K/fi_inner.sh" "$K/comment.tmpl" "$CKPT/model.safetensors.index.json" \
           "$K/fi-src/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_nvfp4_w4a4.py" "$K/fi-src/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_nvfp4_w4a16.py"; do
    [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  [ "$(git -C "$K/fi-src" rev-parse HEAD 2>/dev/null)" = $FISHA ] || { echo "fi-src HEAD is not $FISHA"; ok=1; }
  for c in $FI_NEED; do git -C "$K/fi-src" merge-base --is-ancestor $c HEAD 2>/dev/null || { echo "fi-src lacks $c"; ok=1; }; done
  [ -z "$(git -C "$K/fi-src" status --porcelain --untracked-files=no 2>/dev/null)" ] || { echo "fi-src has local changes"; ok=1; }
  n=$(ls "$K"/wheels/nvidia_cutlass_dsl*-4.7.1-*.whl 2>/dev/null | wc -l); [ "$n" = 4 ] || { echo "expected 4 staged cutlass-dsl 4.7.1 wheels, found $n"; ok=1; }
  python3 -m py_compile "$K/moe109.py" || { echo "moe109.py does not compile"; ok=1; }
  bash -n "$K/fi_inner.sh" || { echo "fi_inner.sh syntax"; ok=1; }
  python3 "$K/moe109.py" selftest > /dev/null || { echo "moe109.py selftest failed"; ok=1; }
  [ "$(docker image inspect -f '{{.Id}}' "$IMG" 2>/dev/null)" = "$IMG_ID" ] || { echo "$IMG missing on dgx-01 or not Id $IMG_ID"; ok=1; }
  for f in "$K"/run.sh "$K"/*.py "$K"/fi_inner.sh "$K"/comment.tmpl; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; ok=1; }; done
  return $ok; }
preflight() { # under the lock: host checks plus the in-image CPU checks of both containers
  local ok=0 k; host_checks || ok=1; mkdir -p "$NUMD"
  for k in b12x fi; do
    docker rm -f k109-$k > /dev/null 2>&1   # a stale container from an earlier killed run
    timeout 600 env OUT= bash -c "$(declare -f drun); K=$K HFM=$HFM IMG=$IMG CAPD=$CAPD NUMD=$NUMD; drun $k cpu check --ckpt $CKPT $([ $k = fi ] && echo --fi)" \
      > "$K/check-$k.txt" 2>&1 && grep -qx 'check ok' "$K/check-$k.txt" || { echo "in-image CPU check ($k) failed ($K/check-$k.txt)"; ok=1; }
  done
  grep -q 'nvidia-cutlass-dsl 4.7.1' "$K/check-fi.txt" && grep -q 'flashinfer from /src/' "$K/check-fi.txt" \
    || { echo "FlashInfer container did not get cutlass-dsl 4.7.1 + the $FISHA source ($K/check-fi.txt)"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  host_checks; rc=$?
  [ -d "$CAPD" ] && echo "routing: k97 capture ($(ls "$CAPD" | grep -c '^moe') MoE chunk files) + uniform" || echo "routing: no k97 capture, uniform only"
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "plan: stop dgx-01 serving; b12x container (b12x-m32, b12x-auto, b12x-wm); FlashInfer $FISHA container (fi-w4a4, fi-w4a16); tp1 I640 + tp2 I320; report"
  echo "note: in-image CPU checks of both containers run in the preflight under the lock (no container now)"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
eval "$(trap -p EXIT | sed "s/^trap -- '/trap -- 'docker rm -f k109-b12x k109-fi > \/dev\/null 2>\&1; /")"   # probe containers go on any exit
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
st "running: preflight"
rm -rf "$NUMD"; preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
mkdir -p "$RES/job" "$RES/p0" "$RES/p1"; cp "$K"/run.sh "$K"/moe109.py "$K"/fi_inner.sh "$K"/comment.tmpl "$K"/check-*.txt "$RES/job/"
st "running: P0 stop dgx-01, b12x MoE layer probe"
stop_all
OUT=$RES/p0
child timeout -k 30 2760 bash -c "$(declare -f drun); K=$K HFM=$HFM IMG=$IMG CAPD=$CAPD NUMD=$NUMD OUT=$OUT; drun b12x gpu b12x /out --ckpt $CKPT --num /num $([ -d "$CAPD" ] && echo --cap-dir /cap)" \
  > "$RES/p0/probe.log" 2>&1; rc0=$?
docker rm -f k109-b12x > /dev/null 2>&1; log "P0 b12x exit=$rc0"
st "running: P1 FlashInfer $FISHA MoE layer probe"
OUT=$RES/p1
child timeout -k 30 2760 bash -c "$(declare -f drun); K=$K HFM=$HFM IMG=$IMG CAPD=$CAPD NUMD=$NUMD OUT=$OUT; drun fi gpu fi /out --ckpt $CKPT --num /num $([ -d "$CAPD" ] && echo --cap-dir /cap)" \
  > "$RES/p1/probe.log" 2>&1; rc1=$?
docker rm -f k109-fi > /dev/null 2>&1; log "P1 FlashInfer exit=$rc1"
st "running: report"
mkdir -p "$RES/all"; cp "$RES"/p0/*.tsv "$RES"/p1/*.tsv "$RES/all/" 2>/dev/null
python3 "$K/moe109.py" report "$RES/all" > "$RES/k109.txt" 2>&1
V=$(sed -n 's/^VERDICT //p' "$RES/k109.txt")
echo "$V (b12x exit $rc0, FlashInfer exit $rc1)" > "$RES/verdict.txt"
fill "$K/comment.tmpl" "$K/comment.md" VERDICT="$RES/verdict.txt" REPORT="$RES/k109.txt"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"
if grep -qiE "$PRIV" "$K/comment.md" || grep -qE '192\.168\.|dgx-0[0-9]\.local' "$K/comment.md"; then
  mkdir -p "$K/private"; mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k109 comment.md matched the private or LAN pattern (not posted; $RES)"; exit 1; fi
cp "$K/comment.md" "$RES/comment.md"
# run_kind exits 0 even when a variant crashed: a required variant without rows is a failed run, not a NO
MISS=; for n in b12x-m32 fi-w4a4 fi-w4a16; do [ -s "$RES/all/$n.tsv" ] || MISS="$MISS $n"; done
OPT=; for n in b12x-auto b12x-wm; do [ -s "$RES/all/$n.tsv" ] || OPT="$OPT $n"; done
[ -z "$OPT" ] || echo "optional variants without rows:$OPT (see $RES/p0/<variant>.log)" >> "$RES/k109.txt"
if [ -n "$MISS" ]; then FINAL="FAILED: k109 required variants without rows:$MISS (logs in $RES/p0 and $RES/p1)"
elif [ $rc0 = 0 ] && [ $rc1 = 0 ] && grep -q '^VERDICT ' "$RES/k109.txt"; then FINAL="DONE: k109 $(echo "$V" | cut -c1-300)${OPT:+; no rows:$OPT} ($RES/k109.txt)"
else FINAL="FAILED: k109 b12x exit $rc0, FlashInfer exit $rc1 ($RES/k109.txt)"; fi
exit 0
