#!/usr/bin/env bash
# k122-decode-profile (2026-11-14, staged while the Sparks are off): fresh decode step profile, ms per step split by op
# class, per rank, at c1/c4/c8 for the shipped 2x v2.2.0 (TP=2, both ranks) and the shipped 1x v2.4.0 (dgx-01, solo).
# No decode profile exists for v2.0 or later (last: b1.2 2x JOURNAL 481, k18 TP=1 v2 JOURNAL 795-799).
# Same method as those runs: mods/vllm-decode-profiler (rank-local torch.profiler, 60-step windows, armed by a trigger
# file on every rank 40 stream chunks into the second repeat of a depth_decode_probe.py cell), step_breakdown.py per
# rank, r8_profile_table.py for the 1x vs 2x table (rank 0). Windows = k18's: fresh c1 / c4 / c8 (depth 256, T=0,
# 512 new tokens, code corpus, offsets k*211111 with k 1/3/5). Measure only, nothing ships, no A/B.
#   2x   shipped registry recipe + profiler edit (prof.sh mkprof: own cache root k122-2x, mod, trigger dir, steps 60)
#   1x   shipped 1x registry recipe + the same edit (own root k122-1x), sparkrun run --solo on dgx-01; dgx-02 idle
# Review 2026-11-14: per window a /metrics sampler (window.py) reports STEADY (c requests running, none waiting, no
# prompt tokens) or MIXED over [trigger, trace written]; the inf decode gap of the unprofiled repeat 0 vs repeat 1 and
# the trace's wall per step are reported side by side (profiler overhead); b12x plan measurements during a window's
# probe are flagged (MEASURED); traces are copied once their size settles; DONE needs 9 usable breakdowns.
# ETA ~45 min warm (stop 1, stage 3, 2x boot ~5 + warm-up 2 + 3 windows ~10, 1x boot ~5 + 2 + ~10, report 1,
# restore ~6); up to ~2 h if a profiler boot recompiles (cap 90 min per boot, cold allowed).
# Output: $RES/2x/prof-<w>/ and $RES/prof/dgx01/prof-<w>/ (rank*.json.gz, breakdown-rank*.txt/json, probe and
# acceptance counters), profile_table.md, k122.txt; comment.md + RESULT here. Traces stay on dgx-01.
# Contract (backlog/runner.sh): both GPU locks (GPU_LOCK_HELD=1); exit 75 untouched while another job is busy; STATE
# first line DONE/FAILED; EXIT trap restores the shipped 2x (lib.sh job_begin). Usage: bash run.sh --dry-run
NEEDS="dgx01 dgx02"
set -u
J=k122; K=$HOME/GEN-AI/backlog/k122; export J K
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k122-decode-profile-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$K/prof.sh"
RD=$K/rec; STEPS=60; WINDOWS="fresh-c1:1:1 fresh-c4:4:3 fresh-c8:8:5"
PRIV=$(cat "$BL/k98/private.pat" 2>/dev/null); [ -n "$PRIV" ] || { echo "missing $BL/k98/private.pat" >&2; exit 2; }

recipes() {
  mkdir -p "$RD" && mkrd || return 1
  [ -s "$REGF2" ] && [ -s "$REGF1" ] || { echo "registry recipes missing (2x '$REGF2', 1x $REGF1)"; return 1; }
  mkprof "$REGF2" "$RD/prof2x.yaml" qwen3.8-flash-next-2x-k122-prof k122-2x "$CTL_ROOT2" "$IMG2" $STEPS || return 1
  mkprof "$REGF1" "$RD/prof1x.yaml" qwen3.8-flash-next-1x-k122-prof k122-1x "$CTL_ROOT1" "$IMG1" $STEPS || return 1; }
preflight() {
  local ok=0 f h
  for f in "$PROBE" "$CORPUS" "$K/step_breakdown.py" "$K/r8_profile_table.py" "$K/window.py" "$MOD/run.sh" "$MOD/profiler_snippet.py"; do
    [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  python3 "$K/window.py" selftest > /dev/null || { echo "window.py selftest failed"; ok=1; }
  python3 "$PROBE" --help 2>/dev/null | grep -q -- --trigger-after-chunks || { echo "$PROBE has no --trigger-after-chunks"; ok=1; }
  bash -n "$MOD/run.sh" || { echo "syntax: $MOD/run.sh"; ok=1; }
  python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$MOD/profiler_snippet.py" || { echo "profiler_snippet.py does not parse"; ok=1; }
  ancheck || ok=1
  recipes || ok=1
  for f in "$RD"/prof*.yaml; do [ -s "$f" ] && { echo "recipe edit $(basename "$f"):"; diff "$(grep -q 1x <<< "$f" && echo "$REGF1" || echo "$REGF2")" "$f" | sed 's/^/   /'; }; done
  for h in localhost $H2; do
    hx $h "docker image inspect $IMG2 > /dev/null 2>&1" || { echo "$h: $IMG2 missing"; ok=1; }
    hx $h "test -d $CRC/b12x && test -d $CRC/${CTL_ROOT2#/cache/runtime/}" || { echo "$h: 2x v2.2.0 runtime cache under $CRC missing"; ok=1; }; done
  docker image inspect $IMG1 > /dev/null 2>&1 || { echo "dgx01: $IMG1 missing"; ok=1; }
  test -d "$CRC/${CTL_ROOT1#/cache/runtime/}" || { echo "dgx01: 1x v2.4.0 VLLM_CACHE_ROOT under $CRC missing"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  regs; preflight; rc=$?
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"; echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"; mkdir -p "$RES/job"
st "running: preflight"
regs; preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
cp "$K"/run.sh "$K"/prof.sh "$K"/*.py "$RD"/*.yaml "$RES/job/"
stop_all
st "running: stage own runtime caches (k122-2x both Sparks, k122-1x dgx-01)"
for h in localhost $H2; do stage $h k122-2x "$CTL_ROOT2" > "$RES/stage-2x-$h.txt" 2>&1 || { FINAL="FAILED: stage $h ($RES/stage-2x-$h.txt)"; exit 1; }; done
stage localhost k122-1x "$CTL_ROOT1" > "$RES/stage-1x.txt" 2>&1 || { FINAL="FAILED: stage 1x ($RES/stage-1x.txt)"; exit 1; }

windows() { # tp own outbase: warm-up, then the three windows (each with a /metrics sampler and a plan-measure count)
  local tp=$1 own=$2 ob=$3 spec tag c k n rc m0 m1 wp
  probe "$ob" warm 4096 1 900000
  for spec in $WINDOWS; do IFS=: read -r tag c k <<< "$spec"
    m0=$(meas $tp)
    python3 "$K/window.py" sample --base http://localhost:8000 --out "$ob/winmon-$tag.jsonl" > /dev/null 2>&1 & wp=$!
    probe "$ob" "$tag" 256 "$c" $((k * 211111)) --trigger-cmd "$(trig $own $J-$own-$tag $tp "$ob/trig-$tag.t")" --trigger-repeat 1 --trigger-after-chunks 40; rc=$?
    kill $wp 2>/dev/null; wait $wp 2>/dev/null
    n=$(collect "$J-$own-$tag" "$ob/prof-$tag" $tp $STEPS); m1=$(meas $tp)
    [ "$m1" -gt "$m0" ] && echo "ranks measured +$((m1 - m0)) b12x plans during the probe that holds the window" > "$ob/prof-$tag/MEASURED"
    log "$own $tag: probe rc $rc, breakdowns $n, plans measured +$((m1 - m0))"; done; }
side() { # tp own recipe outbase solo
  local d=$4
  st "running: $2 profile boot"
  if ! pboot "$3" "$d/boot" 5400 "$5"; then logs "$d/boot" $1; echo "boot failed" > "$d/FAILED"; log "$2: boot failed"; stop_all; return 1; fi
  modok $1 || { logs "$d/boot" $1; echo "profiler mod not applied on every rank" > "$d/FAILED"; log "$2: mod not applied"; stop_all; return 1; }
  st "running: $2 windows $WINDOWS"
  windows $1 $2 "$d"; logs "$d/boot" $1; stop_all; }
mkdir -p "$RES/2x" "$RES/prof/dgx01"
side 2 k122-2x "$RD/prof2x.yaml" "$RES/2x" 0
side 1 k122-1x "$RD/prof1x.yaml" "$RES/prof/dgx01" 1

st "running: report"
python3 "$K/r8_profile_table.py" "$RES" --tp2 "$RES/2x" 2>&1 | sed -e 's/^# r8 Phase 1 profile/# k122 decode profile (1x v2.4.0 dgx-01 = column "TP1 warm"; dgx-02 not used, "TP1 cold" = n\/a)/' > "$RES/profile_table.md"
{ echo "k122 $(TZ=Europe/Bucharest date '+%F %T %Z'): decode step profile, 60-step windows, T=0, 2x v2.2.0 ($IMG2) and 1x v2.4.0 ($IMG1)"
  for s in 2x prof/dgx01; do for d in "$RES/$s"/prof-*/; do [ -d "$d" ] || continue
    for r in 0 1; do f=$d/breakdown-rank$r.txt; [ -s "$f" ] || continue
      echo; echo "== $s $(basename "$d") rank $r"; sed -n '1,2p' "$f"; sed -n '/^summary categories/,/^$/p' "$f"; done
    python3 - "$d"/../pos-"$(basename "$d" | sed 's/^prof-//')".before "$d"/../pos-"$(basename "$d" | sed 's/^prof-//')".after <<'PY' 2>/dev/null
import re, sys
def rd(f):
    d = {}
    for l in open(f):
        m = re.match(r'(\S+?)\{(.*)\}\s+([\d.e+]+)', l)
        if m:
            p = re.search(r'position="(\d+)"', m.group(2)); d[(m.group(1), p.group(1) if p else None)] = float(m.group(3))
    return d
a, b = rd(sys.argv[1]), rd(sys.argv[2]); g = lambda k: b.get(k, 0) - a.get(k, 0)
A = 'vllm:spec_decode_num_accepted_tokens_per_pos_total'; D = 'vllm:spec_decode_num_draft_tokens_per_pos_total'
dr = g((D, '0')) or 1
print("  MTP acceptance per position over the probe: " + " ".join("%.3f" % (g((A, str(i))) / dr) for i in range(8) if (A, str(i)) in b))
PY
    tag=$(basename "$d" | sed 's/^prof-//'); c=$(sed -n "s/.*$tag:\([0-9]*\):.*/\1/p" <<< "${WINDOWS// /$'\n'}")
    python3 "$K/window.py" report "$d/.." "$tag" "${c:-1}" 2>&1
  done; done
  echo; cat "$RES/profile_table.md"
  for d in "$RES/2x" "$RES/prof/dgx01"; do [ -e "$d/FAILED" ] && echo "$d: $(cat "$d/FAILED")"; done
  grep -h ABORT "$RES"/guard-dgx0?.log 2>/dev/null | sed 's/^/memory guard: /'; } > "$RES/k122.txt"
mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
n=$(find "$RES" -name 'breakdown-rank*.json' -size +0 | wc -l)   # usable windows, not just copied files
ns=$(grep -c 'steady state: STEADY' "$RES/k122.txt"); nm=$(find "$RES" -name MEASURED | wc -l)
{ echo "k122 result: decode step profile of the shipped 2x v2.2.0 (both ranks) and 1x v2.4.0 at c1, c4 and c8 (60-step windows, T=0, fresh context, rank-local torch profiler). Measure only."
  echo; echo '```'; head -c 55000 "$RES/k122.txt"; echo '```'
  echo; echo "Traces stay on dgx-01. Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' "$K/comment.md"
if grep -qiE "$PRIV" "$K/comment.md" || grep -qE '192\.168\.|dgx-0[0-9]\.local' "$K/comment.md"; then
  mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k122 comment.md matched the private or LAN pattern ($RES)"; exit 1; fi
cp "$K/comment.md" "$RES/comment.md"
[ "$n" -ge 9 ] && FINAL="DONE: k122 $n of 9 breakdowns (2x 3 windows x 2 ranks, 1x 3), $ns of 6 windows steady, $nm with plan measurement ($RES/k122.txt)" \
  || FINAL="FAILED: k122 only $n of 9 breakdowns, $ns of 6 windows steady ($RES/k122.txt)"
exit 0
