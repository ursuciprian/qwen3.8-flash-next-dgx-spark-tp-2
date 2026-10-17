#!/usr/bin/env bash
# k107fix-2x-b12x-race-fixes-ab (2026-10-16, tp-2 #176): two b12x correctness fixes from the b12x code that moved into
# FlashInfer, on the 2x v2.2.0, against v2.2.0 itself. They are on local-inference-lab/flashinfer main; flashinfer-ai
# main (38e6b894, 10-16) has neither yet. Our fork did not have them.
#   c9b29c91  GDN decode: sync_threads at the top of each persistent work item. In our 21e0b201 the ungrouped request
#             path already has a CTA barrier when deferred checkpoints are on (the replay barrier before the shared
#             q/k rewrite), but the grouped multi-token path (MTP verify with more than one sequence) rewrites shared
#             q/k, decay and beta with no barrier after the previous work item. Upstream saw graph-replay output
#             mismatches of 0.9-1.2x the output scale on GB10 without it.
#   34eb648c  dynamic MoE: fence_view_async_shared before each of the four TMA refills of a released stage (FC1 gate,
#             fused and separate FC1 up, FC2). Our dynamic.py had none (CuTe DSL 4.7.0 in the image has the call).
# Arm: ursuciprian/b12x exp/k107fix-gdn-sync-moe-fence 3657d13a = 21e0b201 + 668b95d6 (c9b29c91, applied as is) +
# 49abfb30 (34eb648c, the four fences placed by hand, race-test hunk left out: no tests/race here) + 3657d13a (pin
# CuTe DSL 4.7.0: the fork pins 4.6.2, so an overlay would otherwise move the arm off the control's 4.7.0). Image:
# build-b12x-overlay on the k104s image, no vLLM overlay, tag in $K/arm.conf (triggered by hand, see BUILD.txt).
#   pull     waits <= 90 min for the image on ghcr, pulls it on both Sparks, same Id; checks the b12x commit, the two
#            fixes in the installed files and CuTe DSL 4.7.0 in the image.
#   tests    GPU tests of the branch on dgx-01 (GDN decode, deferred checkpoints, three NVFP4 MoE suites incl. the
#            dynamic backend), in the control image and then the arm image; a test that passes on the control and
#            fails on the arm stops the job; failures on both, skips and timeouts are recorded only.
#   screen   tp2ab.sh (control = k104s recipe and image by name): stage, bake, ctl arm arm ctl ctl arm arm ctl, cold
#            TTFT, fresh-c8 acceptance, fresh-c1 paired, benchy, plus ident.py (c1 across boots) and identc.py (16 at
#            once, 3 rounds, stability inside each boot).
#   judge    SHIP-CANDIDATE: tests PASS, RULE2 FLAT or WIN (nothing worse, acceptance within 0.03), IDENT PASS or
#            UNSTABLE (no control-stable prompt to judge), IDENTC OK. HOLD: anything else that measured. Speed is not the
#            point; a WIN is reported but not required.
# ETA ~4 h (image wait 0-90, pull ~10, tests ~60 on both images, stage+bake ~35 with a cold b12x compile, 8 boots x ~17 min)
# + restore ~6 min. Nothing ships from this job; the 1x line (a1ad1b69) lacks both fixes too and needs its own arm.
# Contract (backlog/runner.sh): both GPU locks; exit 75 while busy; STATE DONE/SKIPPED/FAILED; comment.md + RESULT.
NEEDS="dgx01 dgx02"
set -u
J=k107fix; K=$HOME/GEN-AI/backlog/k107fix; export J K
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k107fix-2x-b12x-race-fixes-ab-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"   # x, hn
source "$K/arm.conf"
B12X_SHA=3657d13abc5a6aa729805ed5a3554aba9503a0a8
V22=ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm
PYT="pytest==8.4.2 pluggy==1.6.0 iniconfig==2.1.0"
TESTS="tests/sequence/test_gdn_decode.py tests/sequence/test_gdn_deferred_checkpoints.py tests/moe/test_nvfp4_split_backend.py tests/moe/test_nvfp4_swapped_token_tiles.py tests/moe/test_sparse_moe_qwen_shapes.py"
PRIV=$(cat "$BL/k98/private.pat" 2>/dev/null); [ -n "$PRIV" ] || { echo "missing $BL/k98/private.pat" >&2; exit 2; }
REGF=$(grep -rlx --include='*.yaml' "name: qwen3.8-flash-next-2x-dgx-spark" "$(readlink -f "$HOME/.cache/sparkrun/registries/qwen38-flashnext")/recipes" 2>/dev/null | head -1)
shipped_ok() { [ "$(sed -n 's/^container: *//p' "$REGF" 2>/dev/null | tr -d "\"' ")" = "ghcr.io/ursuciprian/spark-vllm-b12x:$SHIPPED_TAG_DEFAULT" ]; }
ghcr_wait() { local t0=$(date +%s); until timeout 120 docker manifest inspect "$ARM_IMG" > /dev/null 2>&1; do
  [ $(( $(date +%s) - t0 )) -gt 5400 ] && return 1; sleep 60; done; }
inimg() { docker run --rm --network none --entrypoint sh "$ARM_IMG" -c "$1" 2>/dev/null; }
img_ok() { # the b12x commit, both fixes in the installed files, CuTe DSL 4.7.0
  local S=/usr/local/lib/python3.12/dist-packages
  [ "$(inimg 'cat /workspace/b12x-source-commit')" = "$B12X_SHA" ] \
    && [ "$(inimg "grep -c 'cute.arch.fence_view_async_shared()' $S/b12x/moe/_shared/kernels/dynamic.py")" = 4 ] \
    && inimg "grep -q 'wait until every warp is done with the last one' $S/b12x/sequence/gdn_decode/_cute_kernels.py" \
    && inimg "ls -d $S/nvidia_cutlass_dsl-4.7.0.dist-info" > /dev/null; }
pytest_in() { # image cmd: pytest from pip into /tmp/pyl inside the image, the branch's tests at /b (k100b, CI layout)
  docker run --rm --init --name k107fix-tests --gpus all --ipc=host --entrypoint bash -e CUTE_DSL_ARCH=sm_121a \
    -e XDG_CACHE_HOME=/tmp/cache -e B12X_COMPILE_WORKERS=2 -v "$K/b12x-src:/b:ro" "$1" -c "
    timeout 300 python3 -m pip install -q --break-system-packages --target /tmp/pyl --no-deps $PYT || exit 3
    cd /tmp && PYTHONPATH=/tmp/pyl $2" 9>&-; }
outcomes() { # pytest -rA log -> "<test id> PASSED|FAILED|ERROR|SKIPPED" lines, sorted
  sed -nE 's/^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) ([^ ]+).*/\2 \1/p' "$1" | sort -u; }
if [ "${1:-}" = --dry-run ]; then rc=0
  timeout 300 sparkrun registry update qwen38-flashnext > /dev/null 2>&1 < /dev/null || echo "note: 2x registry update failed (the run retries)"
  for f in $TESTS tests/conftest.py; do [ -s "$K/b12x-src/$f" ] || { echo "missing $K/b12x-src/$f"; rc=1; }; done
  for f in "$K"/tp2ab.sh "$K"/arm.conf; do bash -n "$f" || { echo "syntax: $f"; rc=1; }; done
  bash "$K/tp2ab.sh" --dry-run 2>&1 | sed 's/^/tp2ab: /'; [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  timeout 120 docker manifest inspect "$ARM_IMG" > /dev/null 2>&1 && echo "arm image on ghcr: $ARM_IMG" \
    || echo "note: $ARM_IMG not on ghcr yet: trigger the build in $K/BUILD.txt first (the job waits <= 90 min, then fails)"
  echo "lib.sh shipped 2x tag: $SHIPPED_TAG_DEFAULT; registry 2x container: $(sed -n 's/^container: *//p' "$REGF" 2>/dev/null)"
  shipped_ok || echo "note: they disagree (release switch half done?): the job refuses to start until lib.sh and the registry agree"
  [ "$SHIPPED_TAG_DEFAULT" = "${V22#*:}" ] || echo "note: lib.sh still restores $SHIPPED_TAG_DEFAULT, not v2.2.0 (fine before the release switch)"
  for f in "$K"/*.sh "$K"/*.py "$K"/arm.conf; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; rc=1; }; done
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"; echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
timeout 300 sparkrun registry update qwen38-flashnext > /dev/null 2>&1 < /dev/null 9>&-   # shipped_ok reads the local registry cache
shipped_ok || { echo "FAILED: k107fix not started: lib.sh restores $SHIPPED_TAG_DEFAULT but the registry 2x runs $(sed -n 's/^container: *//p' "$REGF" 2>/dev/null); finish the release switch" > "$K/STATE"; exit 0; }
job_begin
eval "$(trap -p EXIT | sed "s/^trap -- '/trap -- 'docker rm -f k107fix-tests > \/dev\/null 2>\&1; /")"
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"; mkdir -p "$RES/job"
cp "$K"/run.sh "$K"/tp2ab.sh "$K"/report2x.py "$K"/identc.py "$K"/arm.conf "$RES/job/"
st "running: pull arm image on both Sparks"
docker image inspect "$ARM_IMG" > /dev/null 2>&1 || { ghcr_wait && child timeout 3600 docker pull -q "$ARM_IMG" >> "$RES/pull.txt" 2>&1 9>&-; } \
  || { FINAL="FAILED: pull $ARM_IMG on dgx01 (not on ghcr after 90 min or pull error; was the build in BUILD.txt triggered? $RES/pull.txt)"; exit 1; }
x $H2 "docker image inspect $ARM_IMG > /dev/null 2>&1" || child timeout 3600 ssh -n -o ConnectTimeout=10 $H2 "docker pull -q $ARM_IMG" \
  >> "$RES/pull.txt" 2>&1 9>&- || { FINAL="FAILED: pull $ARM_IMG on dgx02 ($RES/pull.txt)"; exit 1; }
[ "$(x $H1 "docker image inspect -f '{{.Id}}' $ARM_IMG")" = "$(x $H2 "docker image inspect -f '{{.Id}}' $ARM_IMG")" ] || { FINAL="FAILED: arm image Id differs between Sparks"; exit 1; }
img_ok || { FINAL="FAILED: $ARM_IMG does not carry b12x $B12X_SHA with both fixes and CuTe DSL 4.7.0"; exit 1; }
stop_all
st "running: GPU tests of the branch, control image then arm image (dgx-01)"
# Gate: a test that passes on the control image (v2.2.0, b12x 21e0b201: the same test files) and fails or errors on
# the arm image stops the job before any boot. Failures present on both, skips and timeouts are recorded, not gated.
TR=; REG=; for t in $TESTS; do b=$(basename "$t" .py)
  for side in ctl arm; do img=$V22; [ $side = arm ] && img=$ARM_IMG
    child pytest_in "$img" "timeout -k 10 900 python3 -m pytest -q -p no:cacheprovider --import-mode=append -rA /b/$t" > "$RES/tests-$side-$b.txt" 2>&1
    echo $? > "$RES/tests-$side-$b.rc"; docker rm -f k107fix-tests > /dev/null 2>&1
    outcomes "$RES/tests-$side-$b.txt" > "$RES/tests-$side-$b.outcomes"; done
  r=$(join "$RES/tests-ctl-$b.outcomes" "$RES/tests-arm-$b.outcomes" | awk '$2 == "PASSED" && ($3 == "FAILED" || $3 == "ERROR") {print $1}')
  [ -n "$r" ] && REG="$REG $r"
  TR="$TR $b=ctl:exit$(cat "$RES/tests-ctl-$b.rc")/$(grep -c ' PASSED$' "$RES/tests-ctl-$b.outcomes")p,arm:exit$(cat "$RES/tests-arm-$b.rc")/$(grep -c ' PASSED$' "$RES/tests-arm-$b.outcomes")p/$(grep -cE ' (FAILED|ERROR)$' "$RES/tests-arm-$b.outcomes")f"; done
echo "GPU tests (exit/passed/failed per side; 124 = timeout):$TR; regressions (pass on control, fail on arm):${REG:- none}" > "$RES/tests.txt"
[ -z "$REG" ] || { FINAL="FAILED: k107fix GPU test regressions on the arm:$REG ($RES/tests-*.txt)"; exit 1; }
st "running: TP=2 A/B (stage, bake, ctl arm arm ctl ctl arm arm ctl, identity probes)"
child env RES2="$RES/tp2" GPU_LOCK_HELD=1 bash "$K/tp2ab.sh" > "$RES/tp2.nohup" 2>&1 < /dev/null
T2=$(sed -n 's/^RULE2=//p' "$RES/tp2/$J-tp2.txt" 2>/dev/null | tail -1); T2=${T2:-FAILED}
ID=$(sed -n 's/^IDENT=//p' "$RES/tp2/$J-tp2.txt" 2>/dev/null | tail -1); IC=$(sed -n 's/^IDENTC=//p' "$RES/tp2/$J-tp2.txt" 2>/dev/null | tail -1)
case "$T2:${ID:-none}:${IC:-none}" in
  FLAT:PASS:OK|FLAT:UNSTABLE:OK|WIN:PASS:OK|WIN:UNSTABLE:OK) V=SHIP-CANDIDATE ;;
  EMPTY:*|FAILED:*|*:EMPTY:*|*:none:*|*:*:EMPTY|*:*:none) V=EMPTY ;;
  *) V=HOLD ;; esac
{ echo "k107fix $(TZ=Europe/Bucharest date '+%F %T %Z'): b12x ${B12X_SHA:0:8} (GDN decode CTA barrier + MoE TMA fence) on the 2x v2.2.0 ($ARM_IMG) vs 2x v2.2.0 ($V22), TP=2, 4 boots per side"
  echo "VERDICT=$V RULE2=$T2 IDENT=${ID:-none} IDENTC=${IC:-none}"; echo; echo "== GPU tests in the arm image (dgx-01)"; cat "$RES/tests.txt"
  echo; cat "$RES/tp2/$J-tp2.txt" 2>/dev/null
  grep -h ABORT "$RES"/guard-dgx0?.log 2>/dev/null | sed 's/^/memory guard: /'; } > "$RES/k107fix.txt"
mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
{ echo "k107fix result: I tested two b12x correctness fixes on the 2x against the 2x v2.2.0 itself, TP=2, 4 boots per side in the order ctl arm arm ctl ctl arm arm ctl. The fixes are a CTA barrier at the top of each persistent GDN decode work item (the grouped multi-token path of our b12x had none) and a shared-memory fence before each TMA refill in the dynamic MoE kernel. Both come from the b12x code that moved into FlashInfer (the copy that carries them is a FlashInfer fork; FlashInfer main does not have them yet). The arm image is the v2.2.0 image with b12x \`${B12X_SHA:0:8}\` and nothing else changed (same CuTe DSL 4.7.0). Measure only; nothing ships from this job."
  echo; echo "\`VERDICT=$V\` (\`RULE2=$T2\`, identity \`IDENT=${ID:-none}\`, identity under load \`IDENTC=${IC:-none}\`). GPU tests:\`$TR\`."
  case $V in
    SHIP-CANDIDATE) echo "No TTFT, decode or acceptance cell is worse beyond noise, the GPU tests pass and the completions are as stable as on v2.2.0 or more. The fixes can go into the next 2x release; the 1x b12x needs the same two commits." ;;
    HOLD) echo "Something moved that a barrier and a fence should not move, so I hold the fixes until I know why." ;;
    *) echo "Some measurements are missing, so this run gives no verdict." ;; esac
  echo; echo '```'; head -c 55000 "$RES/k107fix.txt"; echo '```'
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"
if grep -qiE "$PRIV" "$K/comment.md" || grep -qE '192\.168\.|dgx-0[0-9]\.local' "$K/comment.md"; then
  mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k107fix comment.md matched the private or LAN pattern (not posted; $RES)"; exit 1; fi
cp "$K/comment.md" "$RES/comment.md"
case $V in SHIP-CANDIDATE|HOLD) FINAL="DONE: k107fix VERDICT=$V RULE2=$T2 IDENT=${ID:-none} IDENTC=${IC:-none} ($RES/k107fix.txt)" ;;
  *) FINAL="FAILED: k107fix VERDICT=$V RULE2=$T2 IDENT=${ID:-none} IDENTC=${IC:-none} ($RES/k107fix.txt)" ;; esac
exit 0
