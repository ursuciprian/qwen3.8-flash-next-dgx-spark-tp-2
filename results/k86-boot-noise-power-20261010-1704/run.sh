#!/usr/bin/env bash
# k86-boot-noise-power (2026-10-10, tp-2 #155): noise between boots of the shipped builds and GPU power,
# on the cells of k84 so stock and shipped sit side by side. Measure only; nothing built or shipped.
#   setups   1x v2.2.0 (recipe-1x-v2.2.0.yaml, 1x repo main) on both Sparks at once, 3 boots (6 boots in all);
#            then 2x v2.0.0 (recipe-2x-v2.0.0.yaml, tp-2 repo main, = lib.sh SHIPPED_TAG_DEFAULT), 3 boots; the last
#            2x boot stays up, so the runner's restore finds the shipped 2x serving
#   cells    per boot cred.sh std_cells without the server-sampling coding cells: 120 s idle; llama-benchy
#            pp2048/tg512 task mode 3 runs, c1 and c8, at T=1.0 top-p 0.95 top-k 20 thinking on and at T=0 thinking
#            off; coding-36 T=0 thinking off at c1 and c8, 3 passes
#   power    nvidia-smi power.draw at 1 Hz on both Sparks for the whole job (GPU power only, not wall power); per cell
#            mean W and tok/s/W, idle = 120 s with the server up; 2x adds both Sparks
#   report   cred.py: per setup and cell the mean of the boots, sd between boots, CV, mean sd within a boot; with
#            --vs the newest k84 results, the stock setups on the same cells
# ETA ~3 h 10 min: 1x 3 x ~36 min (boot 5, idle 2, benchy 6, coding 21, stop 2), 2x 3 x ~26 min.
# Output: $RES/<unit>/ benchy + coding json/log, recipe.yaml, serve log; units.tsv, marks.tsv, power-dgx0N.csv,
# cells.csv, boots.csv, cred.txt; comment.md here + RESULT.
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; EXIT trap stops the power loggers and restores the shipped 2x (lib.sh).
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k86; K=$HOME/GEN-AI/backlog/k86
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k86-boot-noise-power-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"
R1X=$K/recipe-1x-v2.2.0.yaml; R2X=$K/recipe-2x-v2.0.0.yaml; NB=3; PASSES=3
img() { sed -n 's/^container: *//p' "$1" | head -1; }

checks() {
  local ok=0 h f t
  python3 "$CRED/cred.py" --selftest > /dev/null && python3 "$CODING" --selftest > /dev/null || { echo "cred.py/coding.py selftest failed"; ok=1; }
  [ "$(sed -n 1p "$R1X")" = "# Release: v2.2.0" ] && echo "1x recipe v2.2.0: $(img "$R1X")" || { echo "$R1X is not 1x v2.2.0"; ok=1; }
  [ "$(sed -n 1p "$R2X")" = "# Release: v2.0.0 (k73-2x-gdnmse-dispatch)" ] && [ "$(img "$R2X" | sed 's|.*:||')" = "$SHIPPED_TAG_DEFAULT" ] \
    && echo "2x recipe v2.0.0: $(img "$R2X") (= lib.sh SHIPPED_TAG_DEFAULT)" || { echo "$R2X is not 2x v2.0.0 with the shipped tag"; ok=1; }
  for h in $H1 $H2; do for f in "$R1X" "$R2X"; do
    x $h "docker image inspect $(img "$f") > /dev/null 2>&1" && echo "$(hn $h): $(img "$f" | sed 's|.*:||') present" || { echo "$(hn $h): $(img "$f") MISSING"; ok=1; }; done
    x $h "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits" | grep -qE '^[0-9.]+' && echo "$(hn $h): power.draw readable" || { echo "$(hn $h): power.draw unreadable"; ok=1; }; done
  [ -s "$TOK/tokenizer.json" ] || { echo "tokenizer missing"; ok=1; }
  t=$(cd "$R" && timeout 300 uvx --from "$BENCHY_SRC" llama-benchy --version 2>/dev/null | tail -1)
  [ -n "$t" ] && echo "benchy: $t" || { echo "llama-benchy fork not runnable"; ok=1; }
  mkdir -p "$(dirname "$RES")" && [ -w "$(dirname "$RES")" ] && echo "results dir writable: $(dirname "$RES")" || { echo "results dir not writable"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  checks; rc=$?
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "plan: 1x v2.2.0 x $NB boots on both Sparks at once, then 2x v2.0.0 x $NB boots; per boot idle 120 s, tgdef-c1/c8, tgt0-c1/c8, coding t0-nothink c1/c8 x $PASSES passes; power 1 Hz"
  echo "compare with: $(cat "$BL/k84/RESULT" 2>/dev/null || echo 'k84 not run yet (read at report time)')"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
trap 'pw_stop; kill $G1 $G2 2>/dev/null; restore || { sleep 60; restore; } || FINAL="FAILED: restore -- $FINAL"; st "$FINAL"' EXIT
echo "$RES" > "$K/RESULT"; : > "$K/comment.md"
st "running: preflight"
checks > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
cp "$K/run.sh" "$R1X" "$R2X" "$CRED/cred.sh" "$CRED/cred.py" "$CODING" "$RES/"
pw_start; nb1=0; nb2=0

for b in $(seq 1 $NB); do
  st "running: 1x v2.2.0 boot $b/$NB on both Sparks"; stop_all
  if pair 1x-b$b-dgx01 "$R1X" 1x-b$b-dgx02 "$R1X" 2400; then nb1=$((nb1 + 1))
    unit 1x-b$b-dgx01 1x-v2.2.0 $b dgx01 "$R1X"; unit 1x-b$b-dgx02 1x-v2.2.0 $b dgx02 "$R1X"
    std_cells 1x-b$b-dgx01 $H1 $PASSES nodefault & a=$!; std_cells 1x-b$b-dgx02 $H2 $PASSES nodefault & c=$!
    CH="$a $c"; wait $a; wait $c; CH=
  else log "1x boot $b failed (serve logs in 1x-b$b-*)"; fi; done
for b in $(seq 1 $NB); do
  st "running: 2x v2.0.0 boot $b/$NB"; stop_all
  if boot 2x-b$b "$R2X" 1800 2x; then nb2=$((nb2 + 1)); unit 2x-b$b 2x-v2.0.0 $b dgx01+dgx02 "$R2X"; std_cells 2x-b$b $H1 $PASSES nodefault
  else log "2x boot $b failed (serve log in 2x-b$b)"; fi; done
pw_stop

st "running: report"
VS=$(cat "$BL/k84/RESULT" 2>/dev/null); [ -d "$VS" ] && [ -s "$VS/units.tsv" ] || VS=
python3 "$CRED/cred.py" report "$RES" ${VS:+--vs "$VS"} > "$RES/report.log" 2>&1 || { FINAL="FAILED: report, no rows ($RES/report.log)"; exit 1; }
{ echo "Boot-to-boot noise and GPU power of the shipped builds (k86-boot-noise-power): 1x v2.2.0 booted $nb1 times on each Spark (both at once) and 2x v2.0.0 booted $nb2 times, the same cells at every boot."
  echo; echo "Cells: llama-benchy pp2048/tg512 task mode, 3 runs per boot, c1 and c8, at T=1.0 top-p 0.95 top-k 20 with thinking on (\`tgdef\`) and at T=0 with thinking off (\`tgt0\`); the 36-prompt coding probe at T=0 with thinking off, c1 and c8, $PASSES passes per boot. \`sd boots\` is the standard deviation of the per-boot means; \`sd within\` is the mean standard deviation of the runs inside one boot. 1x rows are per Spark (3 boots each) and pooled (6 boots)."
  echo; echo "Power: nvidia-smi power.draw sampled at 1 Hz on both Sparks for the whole job, averaged over each cell's window. This is GPU power as nvidia-smi reports it for the GB10, not wall power (no CPU, memory, NIC or PSU loss). 2x adds both Sparks. Idle = 120 s with the server up and no requests. tok/s/W = the cell's aggregate output tok/s over that mean."
  echo; echo '```'; cat "$RES/cred.txt"; echo '```'
  [ -n "$VS" ] && { echo; echo "The last table puts the stock baseline from k84 (\`$(basename "$VS")\`) on the same cells next to these boots."; }
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
cp "$K/comment.md" "$RES/comment.md"
FINAL="DONE: k86 1x $nb1/$NB boots per Spark, 2x $nb2/$NB boots ($RES/cred.txt)"
exit 0
