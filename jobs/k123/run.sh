#!/usr/bin/env bash
# k123-2x-prefill-profile (2026-11-14, staged while the Sparks are off): per-op GPU split of one cold 16K prefill on
# the shipped 2x v2.2.0, per rank. None exists for the 2x (k81 and k64 were single-Spark).
# Mechanism: mods/vllm-decode-profiler (rank-local), not --profiler-config: /start_profile goes through collective_rpc,
# which still asserts "collective_rpc should not be called on follower node" in vLLM 10d0acc8 and hung the TP=2 pair
# in 2026-09. The mod is armed by a trigger file written on both ranks while the server is idle, so its window starts
# at the first prefill chunk of the next request. Window = STEPS (8) execute_model calls, each marked in the trace
# (local_prof_step_<n>): a cold 16K at max_num_batched_tokens 8192 takes 2 to 4 of them, and the flush request after
# it (64 tokens, ignore_eos, >= 13 steps with MTP) fills the rest and makes the profiler export (it exports at the
# start of the step after the window). Analysis: k123an.py = k81an.py (the k81 1x prefill op classes) with one step per
# marker; the 16K's steps are the leading run of heavy steps (>= 10% of the heaviest), the flush steps are counted and
# left out; coverage = GPU busy of the 16K steps / the unprofiled TTFT, flagged below 0.85 (a missed chunk) and above
# 1.10 (foreign work in the split).
#   recipe  shipped 2x registry recipe + profiler edit (prof.sh mkprof: own cache root k123-2x, mod, trigger, steps 8)
#   boot    sparkrun run (both Sparks), cold allowed (cap 90 min)
#   probes  warm-up 2K + 16K, cold 16K x3 unprofiled (cold-ttft.json), arm both ranks, cold 16K x1 (cold-prof.json),
#           flush (64 tokens, ignore_eos), collect rank0 (dgx-01) + rank1 (dgx-02) traces once their size settles
# ETA ~25 min warm (stop 1, stage 3, boot ~5, probes ~3, export + copy ~2, report 1, restore ~6); ~1 h 30 if it recompiles.
# Output: $RES/prof/rank{0,1}.json.gz, k123.txt; comment.md + RESULT here. Traces stay on dgx-01.
# Contract (backlog/runner.sh): both GPU locks (GPU_LOCK_HELD=1); exit 75 untouched while another job is busy; STATE
# first line DONE/FAILED; EXIT trap restores the shipped 2x (lib.sh job_begin). Usage: bash run.sh --dry-run
NEEDS="dgx01 dgx02"
set -u
J=k123; K=$HOME/GEN-AI/backlog/k123; export J K
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k123-2x-prefill-profile-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k122/prof.sh"
RD=$K/rec; OWN=k123-2x; LAB=$J-$OWN-16k; STEPS=8
PRIV=$(cat "$BL/k98/private.pat" 2>/dev/null); [ -n "$PRIV" ] || { echo "missing $BL/k98/private.pat" >&2; exit 2; }

preflight() {
  local ok=0 f h
  for f in "$COLD" "$CORPUS" "$K/k81an.py" "$K/k123an.py" "$MOD/run.sh"; do [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  python3 "$COLD" selftest > /dev/null || { echo "cold.py selftest failed"; ok=1; }
  python3 "$K/k123an.py" selftest > /dev/null || { echo "k123an.py selftest failed"; ok=1; }
  mkdir -p "$RD" && mkrd || ok=1
  [ -s "$REGF2" ] && mkprof "$REGF2" "$RD/prof2x.yaml" qwen3.8-flash-next-2x-k123-prof $OWN "$CTL_ROOT2" "$IMG2" $STEPS \
    && { echo "recipe edit:"; diff "$REGF2" "$RD/prof2x.yaml" | sed 's/^/   /'; } || { echo "2x profiler recipe not built (registry '$REGF2')"; ok=1; }
  for h in localhost $H2; do
    hx $h "docker image inspect $IMG2 > /dev/null 2>&1" || { echo "$h: $IMG2 missing"; ok=1; }
    hx $h "test -d $CRC/b12x && test -d $CRC/${CTL_ROOT2#/cache/runtime/}" || { echo "$h: 2x v2.2.0 runtime cache under $CRC missing"; ok=1; }; done
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  regs; preflight; rc=$?
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"; echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"; mkdir -p "$RES/job" "$RES/prof"
st "running: preflight"
regs; preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
cp "$K"/run.sh "$K"/*.py "$BL/k122/prof.sh" "$RD"/*.yaml "$RES/job/"
stop_all
st "running: stage own runtime cache $OWN (both Sparks)"
for h in localhost $H2; do stage $h $OWN "$CTL_ROOT2" > "$RES/stage-$h.txt" 2>&1 || { FINAL="FAILED: stage $h ($RES/stage-$h.txt)"; exit 1; }; done
st "running: profiler boot"
pboot "$RD/prof2x.yaml" "$RES/boot" 5400 0 || { logs "$RES/boot" 2; FINAL="FAILED: k123 profiler boot ($RES/boot)"; exit 1; }
modok 2 || { logs "$RES/boot" 2; FINAL="FAILED: k123 profiler mod not applied on both ranks"; exit 1; }
cr() { timeout -k 30 900 python3 "$COLD" run --base http://localhost:8000 --corpus "$CORPUS" --lens "$1" --reps "$2" --seed "$J-$3" --cskip "$4" --out "$RES/cold-$3.json"; }
st "running: warm-up, 3 unprofiled cold 16K, 1 profiled cold 16K"
cr 2048 1 warm 0 > "$RES/cold-warm.txt" 2>&1; cr 16384 1 w16 10000 > "$RES/cold-w16.txt" 2>&1
cr 16384 3 ttft 80000 > "$RES/cold-ttft.txt" 2>&1
[ -z "$(ls -A "$CRC/prof-trigger-$OWN" 2>/dev/null)" ] && [ -z "$(on2 "ls -A $CRC/prof-trigger-$OWN" 2>/dev/null)" ] \
  || { logs "$RES/boot" 2; FINAL="FAILED: k123 trigger dir not empty before arming (dgx-01 or dgx-02)"; exit 1; }
bash -c "$(trig $OWN $LAB 2)"; sleep 2
cr 16384 1 prof 300000 > "$RES/cold-prof.txt" 2>&1
flush() { curl -s -m 300 localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","prompt":"Count from one to one hundred:","max_tokens":64,"ignore_eos":true,"temperature":0}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"]["completion_tokens"])' 2>&1; }
echo "flush completion tokens: $(flush)" > "$RES/flush.txt"   # >= 13 more steps: the 8-step window always closes
n=$(collect $LAB "$RES/prof" 2); log "traces: $n"
logs "$RES/boot" 2; stop_all

st "running: report"
{ echo "k123 $(TZ=Europe/Bucharest date '+%F %T %Z'): 2x v2.2.0 ($IMG2), one cold 16K prompt, rank-local profiler window of $STEPS marked steps (16K + flush), per rank; $(cat "$RES/flush.txt")"
  python3 "$K/k123an.py" report "$RES" 2>&1; echo; cat "$RES/boot/boot.txt" 2>/dev/null
  grep -h ABORT "$RES"/guard-dgx0?.log 2>/dev/null | sed 's/^/memory guard: /'; } > "$RES/k123.txt"
rm -f "$RES"/prof/rank?.json "$RES"/prof/rank?-k123.json   # plain copies made by the report; the .gz traces stay
mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
{ echo "k123 result: per-op GPU split of one cold 16K prefill on the shipped 2x v2.2.0, both ranks (rank-local torch profiler). Measure only."
  echo; echo '```'; head -c 55000 "$RES/k123.txt"; echo '```'
  echo; echo "Traces stay on dgx-01. Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' "$K/comment.md"
if grep -qiE "$PRIV" "$K/comment.md" || grep -qE '192\.168\.|dgx-0[0-9]\.local' "$K/comment.md"; then
  mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k123 comment.md matched the private or LAN pattern ($RES)"; exit 1; fi
cp "$K/comment.md" "$RES/comment.md"
COV=$(sed -n 's/^COVERAGE=//p' "$RES/k123.txt" | tail -1); CL=$(grep '^coverage rank' "$RES/k123.txt" | sed 's/ (.*//' | tr '\n' ';')
if [ "$n" = 2 ] && [ "$COV" = OK ]; then FINAL="DONE: k123 2 traces, $CL ($RES/k123.txt)"
else FINAL="FAILED: k123 $n of 2 traces, COVERAGE=${COV:-n/a} $CL ($RES/k123.txt)"; fi
exit 0
