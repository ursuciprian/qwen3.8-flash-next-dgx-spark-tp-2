#!/usr/bin/env bash
# k107-2x-mhc-pdl-ab (2026-10-16, tp-2 #176): B12X_MHC_PDL=1 (programmatic dependent launch for the b12x MHC, the
# hyper-connection kernels) on the 2x v2.2.0 against v2.2.0. Seen in a published Qwen3.8-Flash-Next recipe of another
# stack (no numbers there); b12x defaults it to 0 and our recipe leaves it unset.
# Why an image and not only the env: b12x reads B12X_MHC_PDL with os.getenv at import, the MHC compile keys and
# selection records carry it (codegen snapshot), and torch AOT graphs hold b12x plan handles. vllm/envs.py at d21d7ade
# and 10d0acc8 does not declare it, so it never reaches envs.compile_factors(); and the recipe command seeds every
# VLLM_CACHE_ROOT from the image's /opt/b12x-seed/torch_compile_cache, so a fresh cache root alone would still load the
# pdl=0 AOT seed. The arm image therefore carries the envs.py declaration (ursuciprian/vllm exp/k107-mhc-pdl-env
# 648bd37e = 10d0acc8 + one env_with_choices entry) and b12x d717da5a (21e0b201 + CuTe DSL pin 4.7.0, same b12x code as
# the control): build-b12x-overlay on the k104s image (triggered by hand, see BUILD.txt). The arm also gets a fresh
# VLLM_CACHE_ROOT of its own (tp2ab.sh), and the overlay drops the baked AOT seed.
#   pull     waits <= 90 min for the image on ghcr, pulls it on both Sparks, same Id; checks both commits, the envs.py
#            declaration and CuTe DSL 4.7.0.
#   screen   tp2ab.sh: stage, bake (the MHC selections re-measure in the prebake because the codegen snapshot changed),
#            ctl arm arm ctl ctl arm arm ctl, cold TTFT 16K/64K/128K x3 (guard), fresh-c8 acceptance, fresh-c1 paired,
#            llama-benchy pp2048 + tg512 c1/c4/c8 x3.
#   judge    RULE2: WIN = tg512 c1/c4/c8 or fresh-c1 better beyond noise and nothing worse (bands + boot CI guard,
#            acceptance within 0.03) | FLAT | LOSS | EMPTY.
# ETA ~3 h (image wait 0-90, pull ~10, stage+bake ~35 with a cold AOT compile, 8 boots x ~16 min) + restore ~6 min.
# Nothing ships from this job. Contract (backlog/runner.sh): both GPU locks; exit 75 while busy; STATE DONE/SKIPPED/
# FAILED; comment.md + RESULT.
NEEDS="dgx01 dgx02"
set -u
J=k107; K=$HOME/GEN-AI/backlog/k107; export J K
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k107-2x-mhc-pdl-ab-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"   # x, hn
source "$K/arm.conf"
B12X_SHA=d717da5a1902c96b8283a4e1f22243e13a8e1791; VLLM_SHA=648bd37e25b70b87b5b55e36cefa1ead3d186642
V22=ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm
PRIV=$(cat "$BL/k98/private.pat" 2>/dev/null); [ -n "$PRIV" ] || { echo "missing $BL/k98/private.pat" >&2; exit 2; }
REGF=$(grep -rlx --include='*.yaml' "name: qwen3.8-flash-next-2x-dgx-spark" "$(readlink -f "$HOME/.cache/sparkrun/registries/qwen38-flashnext")/recipes" 2>/dev/null | head -1)
shipped_ok() { [ "$(sed -n 's/^container: *//p' "$REGF" 2>/dev/null | tr -d "\"' ")" = "ghcr.io/ursuciprian/spark-vllm-b12x:$SHIPPED_TAG_DEFAULT" ]; }
ghcr_wait() { local t0=$(date +%s); until timeout 120 docker manifest inspect "$ARM_IMG" > /dev/null 2>&1; do
  [ $(( $(date +%s) - t0 )) -gt 5400 ] && return 1; sleep 60; done; }
inimg() { docker run --rm --network none --entrypoint sh "$ARM_IMG" -c "$1" 2>/dev/null; }
img_ok() { local S=/usr/local/lib/python3.12/dist-packages
  [ "$(inimg 'cat /workspace/b12x-source-commit')" = "$B12X_SHA" ] && [ "$(inimg 'cat /workspace/vllm-overlay-commit')" = "$VLLM_SHA" ] \
    && inimg "grep -q 'env_with_choices(\"B12X_MHC_PDL\"' $S/vllm/envs.py" && inimg "ls -d $S/nvidia_cutlass_dsl-4.7.0.dist-info" > /dev/null; }
if [ "${1:-}" = --dry-run ]; then rc=0
  timeout 300 sparkrun registry update qwen38-flashnext > /dev/null 2>&1 < /dev/null || echo "note: 2x registry update failed (the run retries)"
  for f in "$K"/tp2ab.sh "$K"/arm.conf; do bash -n "$f" || { echo "syntax: $f"; rc=1; }; done
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
shipped_ok || { echo "FAILED: k107 not started: lib.sh restores $SHIPPED_TAG_DEFAULT but the registry 2x runs $(sed -n 's/^container: *//p' "$REGF" 2>/dev/null); finish the release switch" > "$K/STATE"; exit 0; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"; mkdir -p "$RES/job"
cp "$K"/run.sh "$K"/tp2ab.sh "$K"/report2x.py "$K"/arm.conf "$RES/job/"
st "running: pull arm image on both Sparks"
docker image inspect "$ARM_IMG" > /dev/null 2>&1 || { ghcr_wait && child timeout 3600 docker pull -q "$ARM_IMG" >> "$RES/pull.txt" 2>&1 9>&-; } \
  || { FINAL="FAILED: pull $ARM_IMG on dgx01 (not on ghcr after 90 min or pull error; was the build in BUILD.txt triggered? $RES/pull.txt)"; exit 1; }
x $H2 "docker image inspect $ARM_IMG > /dev/null 2>&1" || child timeout 3600 ssh -n -o ConnectTimeout=10 $H2 "docker pull -q $ARM_IMG" \
  >> "$RES/pull.txt" 2>&1 9>&- || { FINAL="FAILED: pull $ARM_IMG on dgx02 ($RES/pull.txt)"; exit 1; }
[ "$(x $H1 "docker image inspect -f '{{.Id}}' $ARM_IMG")" = "$(x $H2 "docker image inspect -f '{{.Id}}' $ARM_IMG")" ] || { FINAL="FAILED: arm image Id differs between Sparks"; exit 1; }
img_ok || { FINAL="FAILED: $ARM_IMG does not carry b12x $B12X_SHA + vLLM $VLLM_SHA (B12X_MHC_PDL declared) + CuTe DSL 4.7.0"; exit 1; }
st "running: TP=2 A/B (stage, bake, ctl arm arm ctl ctl arm arm ctl)"
child env RES2="$RES/tp2" GPU_LOCK_HELD=1 bash "$K/tp2ab.sh" > "$RES/tp2.nohup" 2>&1 < /dev/null
T2=$(sed -n 's/^RULE2=//p' "$RES/tp2/$J-tp2.txt" 2>/dev/null | tail -1); T2=${T2:-FAILED}
{ echo "k107 $(TZ=Europe/Bucharest date '+%F %T %Z'): B12X_MHC_PDL=1 on the 2x v2.2.0 ($ARM_IMG: vLLM ${VLLM_SHA:0:8} declares it, b12x ${B12X_SHA:0:8}) vs 2x v2.2.0 ($V22), TP=2, 4 boots per side"
  echo "RULE2=$T2"; echo; cat "$RES/tp2/$J-tp2.txt" 2>/dev/null
  grep -h ABORT "$RES"/guard-dgx0?.log 2>/dev/null | sed 's/^/memory guard: /'; } > "$RES/k107.txt"
mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
{ echo "k107 result: I tested programmatic dependent launch for the b12x hyper-connection kernels (\`B12X_MHC_PDL=1\`) on the 2x against the 2x v2.2.0, TP=2, 4 boots per side in the order ctl arm arm ctl ctl arm arm ctl. The arm image is the v2.2.0 image with one vLLM change, the declaration of \`B12X_MHC_PDL\` in \`vllm/envs.py\` (\`${VLLM_SHA:0:10}\`), so the setting reaches the compile cache key; the arm also has a compile cache root of its own. Measure only; nothing ships from this job."
  echo; echo "\`RULE2=$T2\`. Main cells: tg512 at c1, c4 and c8 and the fresh-c1 paired probe."
  case $T2 in
    WIN) echo "A decode cell is faster beyond boot noise and no TTFT, decode or acceptance cell is worse." ;;
    LOSS) echo "A cell is worse beyond noise or acceptance moved, so the setting does not meet my rule." ;;
    FLAT) echo "Nothing is worse beyond noise, but no decode cell is faster beyond noise in this run." ;;
    *) echo "Some measurements are missing, so this run gives no verdict." ;; esac
  echo; echo '```'; head -c 55000 "$RES/k107.txt"; echo '```'
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"
if grep -qiE "$PRIV" "$K/comment.md" || grep -qE '192\.168\.|dgx-0[0-9]\.local' "$K/comment.md"; then
  mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k107 comment.md matched the private or LAN pattern (not posted; $RES)"; exit 1; fi
cp "$K/comment.md" "$RES/comment.md"
case $T2 in WIN|FLAT|LOSS) FINAL="DONE: k107 RULE2=$T2 ($RES/k107.txt)" ;; *) FINAL="FAILED: k107 RULE2=$T2 ($RES/k107.txt)" ;; esac
exit 0
