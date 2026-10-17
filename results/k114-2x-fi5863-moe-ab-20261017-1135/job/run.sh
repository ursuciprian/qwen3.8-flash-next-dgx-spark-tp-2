#!/usr/bin/env bash
# k114-2x-fi5863-moe-ab (2026-10-17, tp-2 #176): FlashInfer's SM12x CuTe DSL W4A4 MoE (flashinfer-ai/flashinfer#5863) for
# the trunk's NVFP4 routed experts in place of b12x dynamic A4, on the 2x against the shipped 2x v2.2.0. k109 (layer
# probe at the TP=2 shape I320, real routing) measured decode x1.133 and prefill x1.262 for the routed-MoE layer, accuracy
# vs FP32 at b12x's level (0.1145 vs 0.1138 rel L2), but its outputs differ from b12x's by about each one's own error
# (independent activation rounding), so the numerics change: MTP acceptance per draft position is the canary here and
# the full quality gate (k114g) is required before anything ships.
#   arm image  the k104s image + b12x d717da5a (21e0b201 + CuTe DSL pin 4.7.0, same b12x code and DSL as the control)
#            + vLLM 0aefd5ee (10d0acc8 + VLLM_FI_SM12X_W4A4_MOE declared in envs.py + experts/flashinfer_sm12x_w4a4_moe.py,
#            trunk layers only, the MTP drafter MoE stays on b12x) + FlashInfer 38e6b894 (the commit k109 measured) as the
#            side package flashinfer_k114: the image's FlashInfer 0.7.0 (sampler, jit-cache) and b12x stay untouched.
#            build-b12x-overlay with flashinfer_ref (spark-vllm-b12x PR #22), see BUILD.txt.
#   pull     waits <= 90 min for the image on ghcr, pulls it on both Sparks, same Id; checks the three commits, the
#            envs.py declaration, the backend file, CuTe DSL 4.7.0, that the image's FlashInfer is still 0.7.0, and
#            imgcheck.py (CPU): the backend imports flashinfer_k114 without loading FlashInfer's embedded b12x tree.
#   screen   tp2ab.sh: stage, bake, ctl arm arm ctl ctl arm arm ctl, cold TTFT 16K/64K/128K x3 (boot-level CI guard),
#            fresh-c8 acceptance per draft position, fresh-c1 paired, llama-benchy pp2048 + tg512 c1/c4/c8 x3.
#   judge    RULE2 (fixed before the run, report2x.py): WIN = a main cell (tg512 c1/c4/c8, fresh-c1, pp2048, cold
#            16K/64K/128K) better beyond noise and nothing worse (bands + CI guard), acceptance per position within
#            0.03, every arm boot logs the FlashInfer MoE line with 48 layers on rank 0 and rank 1 (else EMPTY; a
#            control boot with it is LOSS). KV rule: an arm KV pool (mean tokens) more than 3% below the control's is
#            a LOSS unless all three tg512 cells are better beyond their bands with a mean change >= +10%.
#   gate     nothing ships from here. RULE2=WIN lets k114g (the release quality gate on the arm) run next.
# ETA ~3 h 10 min (image wait 0-90, pull ~10, stage+bake ~35 incl. the cold AOT compile and the first FlashInfer kernel
# compiles, 8 boots x ~17 min) + restore ~6 min.
# Contract (backlog/runner.sh): both GPU locks; exit 75 while busy; STATE DONE/SKIPPED/FAILED; comment.md + RESULT.
NEEDS="dgx01 dgx02"
set -u
J=k114; K=$HOME/GEN-AI/backlog/k114; export J K
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k114-2x-fi5863-moe-ab-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"   # x, hn
source "$K/arm.conf"
B12X_SHA=d717da5a1902c96b8283a4e1f22243e13a8e1791; VLLM_SHA=0aefd5eef6aa96aa6b1a33655ec2539ad4ab6c2e
FI_SHA=38e6b8944fe8b6497d793ef21fbaadd797ea39d8
V22=ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm
PRIV=$(cat "$BL/k98/private.pat" 2>/dev/null); [ -n "$PRIV" ] || { echo "missing $BL/k98/private.pat" >&2; exit 2; }
REGF=$(grep -rlx --include='*.yaml' "name: qwen3.8-flash-next-2x-dgx-spark" "$(readlink -f "$HOME/.cache/sparkrun/registries/qwen38-flashnext")/recipes" 2>/dev/null | head -1)
shipped_ok() { [ "$(sed -n 's/^container: *//p' "$REGF" 2>/dev/null | tr -d "\"' ")" = "ghcr.io/ursuciprian/spark-vllm-b12x:$SHIPPED_TAG_DEFAULT" ]; }
ghcr_wait() { local t0=$(date +%s); until timeout 120 docker manifest inspect "$ARM_IMG" > /dev/null 2>&1; do
  [ $(( $(date +%s) - t0 )) -gt 5400 ] && return 1; sleep 60; done; }
inimg() { docker run --rm --network none --entrypoint sh "$ARM_IMG" -c "$1" 2>/dev/null; }
img_ok() { local S=/usr/local/lib/python3.12/dist-packages
  [ "$(inimg 'cat /workspace/b12x-source-commit')" = "$B12X_SHA" ] && [ "$(inimg 'cat /workspace/vllm-overlay-commit')" = "$VLLM_SHA" ] \
    && [ "$(inimg 'cat /workspace/flashinfer-k114-commit')" = "$FI_SHA" ] \
    && inimg "grep -q '\"VLLM_FI_SM12X_W4A4_MOE\": env_with_choices' $S/vllm/envs.py" \
    && inimg "test -s $S/vllm/model_executor/layers/fused_moe/experts/flashinfer_sm12x_w4a4_moe.py && test -s $S/flashinfer_k114/fused_moe/cute_dsl/blackwell_sm12x/moe_nvfp4_w4a4.py" \
    && inimg "ls -d $S/nvidia_cutlass_dsl-4.7.0.dist-info $S/flashinfer_python-0.7.0.dist-info" > /dev/null \
    && docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v "$K/imgcheck.py:/imgcheck.py:ro" --entrypoint python3 "$ARM_IMG" /imgcheck.py \
         > "$RES/imgcheck.txt" 2>&1 && grep -q '^k114 image check ok' "$RES/imgcheck.txt"; }
if [ "${1:-}" = --dry-run ]; then rc=0
  timeout 300 sparkrun registry update qwen38-flashnext > /dev/null 2>&1 < /dev/null || echo "note: 2x registry update failed (the run retries)"
  for f in "$K"/tp2ab.sh "$K"/arm.conf; do bash -n "$f" || { echo "syntax: $f"; rc=1; }; done
  python3 -m py_compile "$K/imgcheck.py" || { echo "imgcheck.py does not compile"; rc=1; }
  bash "$K/tp2ab.sh" --dry-run 2>&1 | sed 's/^/tp2ab: /'; [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  timeout 120 docker manifest inspect "$ARM_IMG" > /dev/null 2>&1 && echo "arm image on ghcr: $ARM_IMG" \
    || echo "note: $ARM_IMG not on ghcr yet: trigger the build in $K/BUILD.txt first (the job waits <= 90 min, then fails)"
  echo "lib.sh shipped 2x tag: $SHIPPED_TAG_DEFAULT; registry 2x container: $(sed -n 's/^container: *//p' "$REGF" 2>/dev/null)"
  shipped_ok || echo "note: they disagree (release switch half done?): the job refuses to start until lib.sh and the registry agree"
  for f in "$K"/*.sh "$K"/*.py "$K"/arm.conf; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; rc=1; }; done
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"; echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
timeout 300 sparkrun registry update qwen38-flashnext > /dev/null 2>&1 < /dev/null 9>&-   # shipped_ok reads the local registry cache
shipped_ok || { echo "FAILED: k114 not started: lib.sh restores $SHIPPED_TAG_DEFAULT but the registry 2x runs $(sed -n 's/^container: *//p' "$REGF" 2>/dev/null); finish the release switch" > "$K/STATE"; exit 0; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"; mkdir -p "$RES/job"
cp "$K"/run.sh "$K"/tp2ab.sh "$K"/report2x.py "$K"/arm.conf "$K"/imgcheck.py "$RES/job/"
st "running: pull arm image on both Sparks"
docker image inspect "$ARM_IMG" > /dev/null 2>&1 || { ghcr_wait && child timeout 3600 docker pull -q "$ARM_IMG" >> "$RES/pull.txt" 2>&1 9>&-; } \
  || { FINAL="FAILED: pull $ARM_IMG on dgx01 (not on ghcr after 90 min or pull error; was the build in BUILD.txt triggered? $RES/pull.txt)"; exit 1; }
x $H2 "docker image inspect $ARM_IMG > /dev/null 2>&1" || child timeout 3600 ssh -n -o ConnectTimeout=10 $H2 "docker pull -q $ARM_IMG" \
  >> "$RES/pull.txt" 2>&1 9>&- || { FINAL="FAILED: pull $ARM_IMG on dgx02 ($RES/pull.txt)"; exit 1; }
[ "$(x $H1 "docker image inspect -f '{{.Id}}' $ARM_IMG")" = "$(x $H2 "docker image inspect -f '{{.Id}}' $ARM_IMG")" ] || { FINAL="FAILED: arm image Id differs between Sparks"; exit 1; }
img_ok || { FINAL="FAILED: $ARM_IMG does not carry b12x $B12X_SHA + vLLM $VLLM_SHA (VLLM_FI_SM12X_W4A4_MOE declared) + flashinfer_k114 $FI_SHA + CuTe DSL 4.7.0 + FlashInfer 0.7.0"; exit 1; }
st "running: TP=2 A/B (stage, bake, ctl arm arm ctl ctl arm arm ctl)"
child env RES2="$RES/tp2" GPU_LOCK_HELD=1 bash "$K/tp2ab.sh" > "$RES/tp2.nohup" 2>&1 < /dev/null
T2=$(sed -n 's/^RULE2=//p' "$RES/tp2/$J-tp2.txt" 2>/dev/null | tail -1); T2=${T2:-FAILED}
{ echo "k114 $(TZ=Europe/Bucharest date '+%F %T %Z'): FlashInfer SM12x W4A4 routed MoE (#5863) on the 2x ($ARM_IMG: vLLM ${VLLM_SHA:0:8}, flashinfer_k114 ${FI_SHA:0:8}, b12x ${B12X_SHA:0:8}, VLLM_FI_SM12X_W4A4_MOE=1) vs 2x v2.2.0 ($V22), TP=2, 4 boots per side"
  echo "RULE2=$T2"; echo; cat "$RES/tp2/$J-tp2.txt" 2>/dev/null
  grep -h ABORT "$RES"/guard-dgx0?.log 2>/dev/null | sed 's/^/memory guard: /'; } > "$RES/k114.txt"
mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
{ echo "k114 result: I tested FlashInfer's SM12x CuTe DSL W4A4 MoE (flashinfer-ai/flashinfer#5863) for the routed experts in place of b12x on the 2x against the 2x v2.2.0, TP=2, 4 boots per side in the order ctl arm arm ctl ctl arm arm ctl. The arm image is the v2.2.0 image with vLLM \`${VLLM_SHA:0:10}\` (\`VLLM_FI_SM12X_W4A4_MOE=1\`, trunk layers only; the MTP drafter stays on b12x) and FlashInfer \`${FI_SHA:0:10}\` installed next to the image's FlashInfer. The k109 layer probe measured x1.13 decode and x1.26 prefill for the MoE layer at this shape; the outputs differ from b12x's by about each one's own rounding error, so acceptance per draft position is the canary and the release gate runs only after a WIN. Measure only; nothing ships from this job."
  echo; echo "\`RULE2=$T2\`. Main cells: tg512 at c1, c4 and c8, the fresh-c1 paired probe, pp2048 and cold TTFT at 16K, 64K and 128K. KV rule: a KV pool more than 3% smaller than the control's is a loss unless all three tg512 cells are better with a mean gain of at least 10%."
  case $T2 in
    WIN) echo "A main cell is faster beyond boot noise, no TTFT, decode or acceptance cell is worse and the KV pool rule holds. The release quality gate (k114g) runs next." ;;
    LOSS) echo "A cell is worse beyond noise, acceptance moved or the KV pool shrank beyond the rule, so the backend does not meet my rule." ;;
    FLAT) echo "Nothing is worse beyond noise, but no decode cell is faster beyond noise in this run." ;;
    *) echo "Some measurements are missing, so this run gives no verdict." ;; esac
  echo; echo '```'; head -c 55000 "$RES/k114.txt"; echo '```'
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"
if grep -qiE "$PRIV" "$K/comment.md" || grep -qE '192\.168\.|dgx-0[0-9]\.local' "$K/comment.md"; then
  mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k114 comment.md matched the private or LAN pattern (not posted; $RES)"; exit 1; fi
cp "$K/comment.md" "$RES/comment.md"
case $T2 in WIN|FLAT|LOSS) FINAL="DONE: k114 RULE2=$T2 ($RES/k114.txt)" ;; *) FINAL="FAILED: k114 RULE2=$T2 ($RES/k114.txt)" ;; esac
exit 0
