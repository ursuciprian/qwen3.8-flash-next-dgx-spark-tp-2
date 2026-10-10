#!/usr/bin/env bash
# k76 (2026-10-09, tp-2 #128, 1x repo #14 "Capability matrix for the shipped builds"): measure only, nothing built or shipped.
# Setups, in this order (two boots in all):
#   1x    the default 1x recipe of the 1x registry (@qwen38-flashnext-1x/qwen3.8-flash-next-1x-dgx-spark), solo on dgx-01
#         AND dgx-02 at once: two independent samples, the matrix runs on both in parallel
#   dp2   the same two 1x servers behind tools/dp2/pa_router.py (fetched from the tp-2 repo's main at run time, k72 way:
#         dgx-01:8100), no reboot
#   2x    the default 2x recipe of the tp-2 registry (@qwen38-flashnext/qwen3.8-flash-next-2x-dgx-spark), TP=2; it is
#         the shipped 2x, so it stays up at the end and lib.sh restore finds it serving
# Builds are read at run time: `sparkrun registry update` on both registries, then the recipe whose `name:` matches is
# taken from the registry clone (path, registry commit, `# Release: vX.Y.Z (old name ...)` line 1, image tag, digest
# comment). Preflight refuses when the registry 2x tag differs from lib.sh SHIPPED_TAG_DEFAULT (a half-done promotion;
# the runner's post-job restore would stop the pair). Per setup the manifest records image ID/RepoDigests on each Spark,
# vLLM version/commit (serve log), b12x commit (image tag convention), checkpoint repo@revision, served snapshot paths,
# drafter, KV pool tokens, max_num_seqs, AOT/plans, sampling (recipe + the checkpoint's generation_config.json).
# Cells (k76.py PLAN; llama-benchy fork, task mode, --no-force-length, prefix caching, 3 runs, server sampling, thinking on):
#   pp sweep   pp 512/2K/8K/16K/32K/64K/128K, tg 32, c1, cold prompt                      1x, 2x
#   tg depth 0 pp 2048, tg 512, c1/2/4/8 (+16 on 2x and dp2); tg 128 at c1 only                1x, dp2, 2x
#   16K        tg 512, c1/4/8 (+16 on 2x)                                                   1x, 2x
#   64K        tg 512, c1/4                                                                 1x, 2x
#   128K       tg 512, c1/4 (1x pool fits ~7.5 requests at 128K)                            1x, 2x
#   T=0        tg 512 depth 0, c1/c8 (diagnostic, capability-t0.csv only)                   1x, dp2, 2x
#   ITL        llm-inference-bench (llama-benchy has no ITL percentiles), 30 s sustained decode window per cell:
#              1x 0/16K/64K x c1/4/8, 2x 0/16K/64K x c1/4/8/16, dp2 depth 0 x c1/4/8/16
#   coding-36  k55's 36 prompts (12 Python, 8 C++, 8 Rust, 8 Go; up to 768 tokens out; coding.py imports
#              ~/GEN-AI/k55/coding_probe.py and sends the prompts N at a time), modes "T=0, thinking off" and
#              "server defaults, thinking on", 3 passes per cell: 1x and 2x c1/8, dp2 c8 (no c4)     1x, dp2, 2x
#   copy-heavy k46/tools/bench_copy_streams.py (as k58): 1,500 tokens out, effort low, shared cached prefix, window
#              tok/s while all streams decode, its own 3 rounds after a warm-up, 1/4/8 streams         1x, 2x
#   TTFT p50/p90/p99 under concurrency: llama-benchy per-request e2e TTFT, pooled over the 3 runs.
#   Not run: 1x c16 (max_num_seqs 8), dp2 depth > 0 (the router keys on the first user message; benchy's context-load
#   turn is "." so context and measured turn land on different replicas), dp2 pp sweep and coding c1 (same path as
#   1x), dp2 copy (the bench reads /metrics, the router has none), 16K c2, 64K c2/c8, 128K c2/c8/c16, tg128 c2+.
# ETA (ETA.txt has the arithmetic): 1x 126.8 min (both Sparks at once) + dp2 20.9 + 2x 103.7 + transitions 16
# = ~4 h 27 min.
# Output: $RES/{1x-dgx01,1x-dgx02,dp2,2x}/ raw benchy/lib json+log, recipe.json, serve-excerpt.txt (serve.log stays on
# dgx-01), image-dgx0N.json; manifest.json, capability.csv, capability-t0.csv, k76.txt; comment.md here + RESULT.
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; EXIT trap kills the router and restores the shipped 2x (lib.sh).
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k76; K=$HOME/GEN-AI/backlog/k76
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k76-capability-matrix-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
MODEL=qwen3.8-flash-next; RPORT=8100; BENCHY_SRC=$G/llama-benchy-fork; C=$G/k31; PY="python3 $K/k76.py"
COPY=$G/k46/tools/bench_copy_streams.py
REGS=$HOME/.cache/sparkrun/registries; REG1=qwen38-flashnext-1x; REG2=qwen38-flashnext
N1=qwen3.8-flash-next-1x-dgx-spark; N2=qwen3.8-flash-next-2x-dgx-spark
GH2=https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2

x() { if [ "$1" = $H1 ]; then bash -c "$2"; else ssh -n -o ConnectTimeout=10 $H2 "$2"; fi; }
hn() { [ "$1" = $H1 ] && echo dgx01 || echo dgx02; }
recf() { grep -rlx --include='*.yaml' "name: $2" "$(readlink -f "$REGS/$1")/recipes" 2>/dev/null | head -1; }
img() { sed -n 's/^container: *//p' "$1" | head -1; }
tok() { local m r; m=$(sed -n 's/^model: *//p' "$1" | head -1); r=$(sed -n 's/^model_revision: *//p' "$1" | head -1)
  echo "$HOME/.cache/huggingface/hub/models--${m//\//--}/snapshots/$r"; }
kvof() { grep -m1 -oE 'GPU KV cache size: [0-9,]+' "$1" 2>/dev/null | tr -dc 0-9; }

checks() { # sets F1 F2 TOK RCOMMIT; writes $K/pa_router.py
  local ok=0 f h i t
  $PY --selftest || { echo "k76.py selftest failed"; ok=1; }
  python3 "$K/coding.py" --selftest || { echo "coding.py selftest failed (~/GEN-AI/k55/coding_probe.py)"; ok=1; }
  python3 "$COPY" --help 2>/dev/null | grep -q -- --skip-idle-check && echo "copy bench: $COPY" || { echo "copy bench $COPY missing"; ok=1; }
  for t in sparkrun uvx python3 curl git docker; do command -v $t > /dev/null || { echo "missing $t"; ok=1; }; done
  python3 -c 'import yaml, json, csv, statistics' || { echo "python3 lacks yaml/json/csv/statistics (CSV writing)"; ok=1; }
  F1=$(recf $REG1 $N1); F2=$(recf $REG2 $N2)
  for f in "$F1" "$F2"; do
    [ -s "$f" ] || { echo "recipe not resolvable from the registries ($REG1/$N1, $REG2/$N2)"; ok=1; continue; }
    i=$(img "$f"); echo "recipe $f @ $(git -C "$(dirname "$f")" rev-parse --short HEAD): $($PY recipe "$f" --short)"
    for h in $H1 $H2; do x $h "docker image inspect $i > /dev/null 2>&1" && echo "  $(hn $h): image present" || { echo "  $(hn $h): image $i MISSING"; ok=1; }; done
    [ -s "$(tok "$f")/tokenizer.json" ] || { echo "  tokenizer snapshot $(tok "$f") missing on dgx-01"; ok=1; }; done
  [ -s "$F2" ] && { [ "$(img "$F2" | sed 's|.*:||')" = "$SHIPPED_TAG_DEFAULT" ] && echo "registry 2x tag = lib.sh SHIPPED_TAG_DEFAULT ($SHIPPED_TAG_DEFAULT)" \
    || { echo "registry 2x tag $(img "$F2" | sed 's|.*:||') != lib.sh SHIPPED_TAG_DEFAULT $SHIPPED_TAG_DEFAULT: finish the promotion in lib.sh first"; ok=1; }; }
  TOK=$(tok "$F1")
  RCOMMIT=$(git ls-remote $GH2 refs/heads/main 2>/dev/null | cut -f1)
  { [ -n "$RCOMMIT" ] && curl -fsSL -m 60 "https://raw.githubusercontent.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/$RCOMMIT/tools/dp2/pa_router.py" -o "$K/pa_router.py" \
    && python3 "$K/pa_router.py" --selftest > /dev/null 2>&1; } && echo "router tools/dp2/pa_router.py @ ${RCOMMIT:0:8}: selftest OK" \
    || { echo "router tools/dp2/pa_router.py @ ${RCOMMIT:-?}: fetch or selftest failed"; ok=1; }
  t=$(cd "$R" && timeout 300 uvx --from "$BENCHY_SRC" llama-benchy --version 2>/dev/null | tail -1)
  [ -n "$t" ] && echo "benchy: $t ($BENCHY_SRC @ $(git -C "$BENCHY_SRC" rev-parse --short HEAD))" || { echo "llama-benchy fork not runnable"; ok=1; }
  t=$(sed -n 's/^VERSION = "\(.*\)"/\1/p' "$C/lib/llm_decode_bench.py" 2>/dev/null)
  [ -x "$C/venv/bin/python" ] && [ -n "$t" ] && echo "llm-inference-bench: $t ($C/lib @ $(cat "$C/lib/COMMIT" 2>/dev/null))" || { echo "llm-inference-bench missing"; ok=1; }
  mkdir -p "$(dirname "$RES")" && [ -w "$(dirname "$RES")" ] && echo "results dir writable: $(dirname "$RES")" || { echo "results dir not writable"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  checks; rc=$?
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  for s in 1x dp2 2x; do echo "plan $s:"; $PY plan $s | cut -f1,3 | sed 's/^/  /;s/--runs 3 --prompt-mode task --no-force-length --enable-prefix-caching --format json //'
    echo "  itl: $($PY plan $s --lib)"; echo "  coding (mode conc, 3 passes): $($PY plan-coding $s | paste -sd, -)"
    echo "  copy streams: $($PY plan-copy $s)"; done
  sed -n 's/^Total: /ETA: /p' "$K/ETA.txt"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
RP=; trap 'kill $G1 $G2 $RP 2>/dev/null; restore || { sleep 60; restore; } || FINAL="FAILED: restore -- $FINAL"; st "$FINAL"' EXIT
echo "$RES" > "$K/RESULT"; : > "$K/comment.md"
ph() { echo "$1 $2 $(date +%s) $(minmem "$RES/guard-dgx01.log" "$2" $(date +%s)) $(minmem "$RES/guard-dgx02.log" "$2" $(date +%s))" >> "$RES/phases.txt"; }

st "running: preflight"; t=$(date +%s)
for r in $REG1 $REG2; do timeout 300 sparkrun registry update $r >> "$RES/$J.log" 2>&1 < /dev/null 9>&-; done
checks > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
cp "$F1" "$RES/recipe-1x.yaml"; cp "$F2" "$RES/recipe-2x.yaml"; cp "$K/pa_router.py" "$K/k76.py" "$K/coding.py" "$K/run.sh" "$K/ETA.txt" "$RES/"
log "1x $F1 ($(img "$F1")), 2x $F2 ($(img "$F2")), router @ $RCOMMIT"

bench() { # unit url metrics-url name timeout args...
  local u=$RES/$1 url=$2 m=$3 n=$4 t=$5; shift 5
  ( cd "$R" && timeout -k 30 "$t" uvx --from "$BENCHY_SRC" llama-benchy --base-url "$url/v1" --model $MODEL --tokenizer "$TOK" \
      ${m:+--metrics-url "$m"} --save-result "$u/$n.json" "$@" > "$u/$n.log" 2>&1 < /dev/null 9>&- ); log "$1 $n exit=$?"; }
matrix() { # setup unit host port kv-budget
  local s=$1 u=$2 h=$3 p=$4 n t a m=; mkdir -p "$RES/$u"; [ "$s" = dp2 ] || m="http://$h:$p/metrics"
  while IFS=$'\t' read -r n t a; do bench "$u" "http://$h:$p" "$m" "$n" "$t" $a; done < <($PY plan "$s")
  ( cd "$RES/$u" && LLM_BENCH_NO_UPDATE_CHECK=1 timeout --foreground -k 30 2400 "$C/venv/bin/python" "$C/lib/llm_decode_bench.py" \
      --host "$h" --port "$p" --model $MODEL --display-mode plain --no-resume --skip-prefill --duration 30 ${5:+--kv-budget $5} \
      --output "$RES/$u/itl.json" $($PY plan "$s" --lib) > "$RES/$u/itl.log" 2>&1 < /dev/null 9>&- ); log "$u itl exit=$?"
  while read -r n t; do timeout -k 30 3600 python3 "$K/coding.py" "http://$h:$p" $n $t 3 "$RES/$u/coding-$n-c$t.json" \
      > "$RES/$u/coding-$n-c$t.log" 2>&1 < /dev/null 9>&-; log "$u coding $n c$t exit=$?"; done < <($PY plan-coding "$s")
  n=$($PY plan-copy "$s"); [ -n "$n" ] || return 0
  ( cd "$RES/$u" && timeout -k 30 1800 python3 "$COPY" --base "http://$h:$p" --model $MODEL --min-mem-gib 4 --skip-idle-check \
      --streams $n --out "$RES/$u/copy.json" > "$RES/$u/copy.log" 2>&1 < /dev/null 9>&- ); log "$u copy exit=$?"; }
facts() { # host unit recipe container-pattern
  local h=$1 u=$RES/$2 c; mkdir -p "$u"
  c=$(x $h "docker ps --format '{{.Names}}' | grep -E '$4' | head -1")
  x $h "docker exec $c cat /tmp/sparkrun_serve.log" > "$u/serve.log" 2>/dev/null
  $PY serve "$u/serve.log" > "$u/serve-excerpt.txt" 2>&1; $PY recipe "$3" > "$u/recipe.json"
  for g in $H1 $H2; do [ "$2" = 2x ] || [ $g = $h ] && x $g "docker image inspect $(img "$3")" > "$u/image-$(hn $g).json" 2>/dev/null; done
  p=$(pong $h); echo "pong $p" >> "$u/boot.txt"; echo "$p" | grep -qi pong; }
up_wait() { # timeout hosts...
  local to=$1 s=$(date +%s) h ok; shift
  while :; do ok=1; for h in "$@"; do [ "$(health $h)" = 200 ] || ok=0; done; [ $ok = 1 ] && break
    [ $(( $(date +%s) - s )) -gt $to ] && return 1; sleep 15; done
  log "healthy after $(( $(date +%s) - s )) s: $*"; }
ph preflight $t

st "running: stop 2x, boot 1x on both Sparks"; t=$(date +%s)
stop_all
for h in $H1 $H2; do ( cd "$K" && timeout -k 30 900 sparkrun run "@$REG1/$N1" --hosts $h --solo --no-follow ) >> "$RES/$J.log" 2>&1 < /dev/null 9>&-; done
up_wait 3600 $H1 $H2 || { FINAL="FAILED: 1x boot (health $(health $H1)/$(health $H2) after 1 h)"; exit 1; }
for h in $H1 $H2; do facts $h 1x-$(hn $h) "$F1" '_solo|sparkrun' || { FINAL="FAILED: 1x on $(hn $h) no pong"; exit 1; }; done
KV1=$(kvof "$RES/1x-dgx01/serve.log"); KV2=$(kvof "$RES/1x-dgx02/serve.log"); ph boot-1x $t

st "running: 1x matrix on both Sparks (~127 min)"; t=$(date +%s)
matrix 1x 1x-dgx01 $H1 8000 "$KV1" & a=$!; matrix 1x 1x-dgx02 $H2 8000 "$KV2" & b=$!; CH="$a $b"; wait $a; wait $b; CH=
ph 1x $t

if [ "$(health $H1)" = 200 ] && [ "$(health $H2)" = 200 ]; then
  st "running: DP=2 matrix (~21 min)"; t=$(date +%s); mkdir -p "$RES/dp2"
  echo "tools/dp2/pa_router.py @ $RCOMMIT ($GH2), dgx-01:$RPORT -> http://$H1:8000,http://$H2:8000" > "$RES/dp2/router.txt"
  python3 "$RES/pa_router.py" --port $RPORT --backends http://$H1:8000,http://$H2:8000 > "$RES/dp2/router.log" 2>&1 < /dev/null 9>&- & RP=$!
  sleep 3
  if [ "$(health localhost $RPORT)" = 200 ]; then matrix dp2 dp2 localhost $RPORT $(( ${KV1:-0} + ${KV2:-0} ))
    curl -s -m 5 localhost:$RPORT/router/stats > "$RES/dp2/router-stats.json"; else log "router not healthy: DP=2 skipped"; fi
  kill $RP 2>/dev/null; RP=; ph dp2 $t
else log "a 1x server is down after the 1x matrix: DP=2 skipped"; fi

st "running: stop 1x, boot 2x"; t=$(date +%s)
stop_all
( cd "$R" && timeout -k 60 3600 sparkrun run "@$REG2/$N2" --no-follow ) >> "$RES/$J.log" 2>&1 < /dev/null 9>&-
up_wait 1800 localhost || { FINAL="FAILED: 2x boot (health $(health localhost) after 30 min)"; exit 1; }
facts $H1 2x "$F2" node_0 || { FINAL="FAILED: 2x no pong"; exit 1; }
is_shipped || log "WARNING: the 2x up is not lib.sh's shipped tag $E (restore will reboot it)"
ph boot-2x $t

st "running: 2x matrix (~104 min)"; t=$(date +%s)
CH=; matrix 2x 2x $H1 8000 "$(kvof "$RES/2x/serve.log")"; ph 2x $t

st "running: report"
$PY report "$RES" > "$RES/report.log" 2>&1 || { FINAL="FAILED: report, no rows ($RES/report.log)"; exit 1; }
n=$(( $(wc -l < "$RES/capability.csv") - 1 )); miss=$(sed -n '/^Planned but not measured:/,$p' "$RES/k76.txt" | grep -vc -e '^Planned' -e '^  none$')
{ echo "I measured the shipped builds on one grid: the 1x recipe on each Spark (two independent samples), DP=2 (those two 1x servers behind \`tools/dp2/pa_router.py\`), and the 2x TP=2 recipe. Builds were read from the registries at run time; \`manifest.json\` has the recipe commit, release line, image ID and digest, vLLM version, checkpoint revision, drafter and KV pool per setup."
  echo; echo "llama-benchy (task mode, server default sampling, thinking on, 3 runs per cell) for prefill, decode at depth 0/16K/64K/128K and TTFT percentiles; llm-inference-bench for inter-token latency percentiles (one 30 s window per cell); the 36-prompt coding corpus from k55 at c1 and c8 (dp2 c8) in both modes, 3 passes; the copy-heavy cell at 1/4/8 streams (1x, 2x). The \`workload\` column says which. Main rows are mean ± sd of 3 runs; 1x main rows pool both Sparks (2 Sparks x 3 runs) and the per-Spark rows carry the suffixes \`_dgx01\`/\`_dgx02\`; \`*_max\` rows are the max of those runs. T=0 diagnostic rows are in \`capability-t0.csv\` only."
  echo; echo "$n rows in \`capability.csv\` (the format of \`docs/data/capability.csv\`); $miss planned cells not measured (listed in \`k76.txt\`). Same data for ursuciprian/qwen3.8-flash-next-1x-dgx-spark#14."
  echo; echo '```'; head -c 50000 "$RES/k76.txt"; echo '```'; echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
cp "$K/comment.md" "$RES/comment.md"
FINAL="DONE: k76 $n capability rows, $miss planned cells not measured ($RES/k76.txt)"
exit 0
