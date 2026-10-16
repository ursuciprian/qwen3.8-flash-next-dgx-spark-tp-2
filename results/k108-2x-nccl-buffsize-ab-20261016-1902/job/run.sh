#!/usr/bin/env bash
# k108-2x-nccl-buffsize-ab (2026-10-16, tp-2 #176): NCCL_BUFFSIZE=1048576 and NCCL_MAX_CTAS=8 on the 2x v2.2.0 against
# v2.2.0, env only, same image. Seen in another stack's Spark launcher (no numbers there).
# Path: the recipe sets VLLM_ENABLE_ROCE_ALLREDUCE=1 and VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB. vLLM 10d0acc8
# (cuda_communicator.all_reduce) sends an all-reduce to b12x RoCE only when should_custom_ar() accepts its size and
# falls back to NCCL above the limit ("NCCL remains the fallback above 2MB" in serve.log). One all-reduce of the 2560
# hidden in BF16 is 5120 B per row, so calls above 409 rows go to NCCL: every pp2048 prefill all-reduce (2048 rows,
# 10.5 MB) and every chunk of a cold 16K prompt (up to 8192 rows, 42 MB). Decode (c8 x 5 rows = 205 KB) stays on RoCE,
# so the tg cells are guards. sparkrun's vllm_distributed runtime sets neither variable, so the control runs the NCCL
# defaults. Neither is a vLLM env, so the compile key does not change; the arm still gets a cache root of its own.
#   screen   tp2ab.sh: stage, bake, ctl arm arm ctl ctl arm arm ctl, cold TTFT 16K/64K/128K x3, fresh-c8 acceptance,
#            fresh-c1 paired, llama-benchy pp2048 + tg512 c1/c4/c8 x3; both ranks' env checked on every boot.
#   judge    RULE2: WIN = pp2048 (c1) or cold 16K better beyond noise and nothing worse (bands + boot CI guard,
#            acceptance within 0.03) | FLAT | LOSS | EMPTY.
# ETA ~2 h 40 min (stage+bake ~15, warm AOT from the image seed, 8 boots x ~16 min) + restore ~6 min. Nothing ships.
# Contract (backlog/runner.sh): both GPU locks; exit 75 while busy; STATE DONE/SKIPPED/FAILED; comment.md + RESULT.
NEEDS="dgx01 dgx02"
set -u
J=k108; K=$HOME/GEN-AI/backlog/k108; export J K
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k108-2x-nccl-buffsize-ab-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
source "$K/arm.conf"
V22=ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm
PRIV=$(cat "$BL/k98/private.pat" 2>/dev/null); [ -n "$PRIV" ] || { echo "missing $BL/k98/private.pat" >&2; exit 2; }
REGF=$(grep -rlx --include='*.yaml' "name: qwen3.8-flash-next-2x-dgx-spark" "$(readlink -f "$HOME/.cache/sparkrun/registries/qwen38-flashnext")/recipes" 2>/dev/null | head -1)
shipped_ok() { [ "$(sed -n 's/^container: *//p' "$REGF" 2>/dev/null | tr -d "\"' ")" = "ghcr.io/ursuciprian/spark-vllm-b12x:$SHIPPED_TAG_DEFAULT" ]; }
if [ "${1:-}" = --dry-run ]; then rc=0
  timeout 300 sparkrun registry update qwen38-flashnext > /dev/null 2>&1 < /dev/null || echo "note: 2x registry update failed (the run retries)"
  for f in "$K"/tp2ab.sh "$K"/arm.conf; do bash -n "$f" || { echo "syntax: $f"; rc=1; }; done
  [ -z "$ARM_IMG" ] || { echo "k108 is env only: ARM_IMG must be empty"; rc=1; }
  grep -qx '  VLLM_ROCE_ALLREDUCE_MAX_SIZE: "2MB"' "$BL/k104s/recipe.yaml" && grep -qx '  VLLM_ENABLE_ROCE_ALLREDUCE: "1"' "$BL/k104s/recipe.yaml" \
    && echo "control: RoCE all-reduce on, max 2MB (prefill all-reduces go through NCCL)" || { echo "control RoCE all-reduce settings changed"; rc=1; }
  grep -qE '^ *NCCL_(BUFFSIZE|MAX_CTAS):' "$BL/k104s/recipe.yaml" && { echo "control already sets an NCCL arm variable"; rc=1; }
  bash "$K/tp2ab.sh" --dry-run 2>&1 | sed 's/^/tp2ab: /'; [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  echo "lib.sh shipped 2x tag: $SHIPPED_TAG_DEFAULT; registry 2x container: $(sed -n 's/^container: *//p' "$REGF" 2>/dev/null)"
  shipped_ok || echo "note: they disagree (release switch half done?): the job refuses to start until lib.sh and the registry agree"
  for f in "$K"/*.sh "$K"/*.py "$K"/arm.conf; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; rc=1; }; done
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"; echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
timeout 300 sparkrun registry update qwen38-flashnext > /dev/null 2>&1 < /dev/null 9>&-   # shipped_ok reads the local registry cache
shipped_ok || { echo "FAILED: k108 not started: lib.sh restores $SHIPPED_TAG_DEFAULT but the registry 2x runs $(sed -n 's/^container: *//p' "$REGF" 2>/dev/null); finish the release switch" > "$K/STATE"; exit 0; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"; mkdir -p "$RES/job"
cp "$K"/run.sh "$K"/tp2ab.sh "$K"/report2x.py "$K"/arm.conf "$RES/job/"
st "running: TP=2 A/B (stage, bake, ctl arm arm ctl ctl arm arm ctl)"
child env RES2="$RES/tp2" GPU_LOCK_HELD=1 bash "$K/tp2ab.sh" > "$RES/tp2.nohup" 2>&1 < /dev/null
T2=$(sed -n 's/^RULE2=//p' "$RES/tp2/$J-tp2.txt" 2>/dev/null | tail -1); T2=${T2:-FAILED}
RA=$(grep -l 'RoCEnante all-reduce is live' "$RES"/tp2/2x/*-p?/serve.log 2>/dev/null | wc -l)
{ echo "k108 $(TZ=Europe/Bucharest date '+%F %T %Z'): NCCL_BUFFSIZE=1048576 NCCL_MAX_CTAS=8 on the 2x v2.2.0 ($V22) vs the same image without them, TP=2, 4 boots per side"
  echo "RULE2=$T2"; echo "RoCE all-reduce live line in $RA of 8 screen boots (decode stays on RoCE, prefill above 2MB on NCCL)"
  echo; cat "$RES/tp2/$J-tp2.txt" 2>/dev/null
  grep -h ABORT "$RES"/guard-dgx0?.log 2>/dev/null | sed 's/^/memory guard: /'; } > "$RES/k108.txt"
mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
{ echo "k108 result: I tested two NCCL settings on the 2x, \`NCCL_BUFFSIZE=1048576\` and \`NCCL_MAX_CTAS=8\`, against the 2x v2.2.0 on the same image, TP=2, 4 boots per side in the order ctl arm arm ctl ctl arm arm ctl. Only prefill all-reduces go through NCCL on this stack (above the 2MB RoCE limit, so every pp2048 and cold-prompt chunk); decode stays on RoCE, so its cells are guards. Measure only; nothing ships from this job."
  echo; echo "\`RULE2=$T2\`. Main cells: pp2048 at c1 and cold 16K TTFT."
  case $T2 in
    WIN) echo "Prefill is faster beyond boot noise and no TTFT, decode or acceptance cell is worse." ;;
    LOSS) echo "A cell is worse beyond noise or acceptance moved, so the settings do not meet my rule." ;;
    FLAT) echo "Nothing is worse beyond noise, but neither pp2048 nor cold 16K is faster beyond noise in this run." ;;
    *) echo "Some measurements are missing, so this run gives no verdict." ;; esac
  echo; echo '```'; head -c 55000 "$RES/k108.txt"; echo '```'
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"
if grep -qiE "$PRIV" "$K/comment.md" || grep -qE '192\.168\.|dgx-0[0-9]\.local' "$K/comment.md"; then
  mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k108 comment.md matched the private or LAN pattern (not posted; $RES)"; exit 1; fi
cp "$K/comment.md" "$RES/comment.md"
case $T2 in WIN|FLAT|LOSS) FINAL="DONE: k108 RULE2=$T2 ($RES/k108.txt)" ;; *) FINAL="FAILED: k108 RULE2=$T2 ($RES/k108.txt)" ;; esac
exit 0
