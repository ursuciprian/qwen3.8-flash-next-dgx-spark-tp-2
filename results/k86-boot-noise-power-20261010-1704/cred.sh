# ~/GEN-AI/backlog/k84/cred.sh (2026-10-10, tp-2 #155): helpers shared by k84 (stock baseline), k85
# (quality: KLD + evals) and k86 (boot noise + power). Sourced after lib.sh; the caller sets J, RES and K.
#   pw_start / pw_stop   nvidia-smi power.draw, clocks.sm, temperature at 1 Hz on both Sparks -> $RES/power-dgx0N.csv
#                        ("epoch,W,MHz,C"). GPU power as nvidia-smi reports it for the GB10: not wall power.
#   bench <unit> <host> <name> <timeout> <args...>   llama-benchy, pp2048 tg512 depth 0, task mode, 3 runs (k76 d0 args)
#   coding <unit> <host> <mode> <conc> <passes>      k76/coding.py: k55's 36 prompts, up to 768 tokens out
#   idle <unit> <s>      server up, no requests (power baseline)
#   std_cells <unit> <host> <passes> [nodefault]     the shared cell list (below)
#   boot <unit> <recipe.yaml> <timeout s> <host|2x>  sparkrun run --no-follow, waits for /health and a pong
#   unit <unit> <setup> <boot> <hosts> <recipe>      one line in $RES/units.tsv (read by cred.py)
# Every cell appends "unit<TAB>cell<TAB>start<TAB>end<TAB>rc" to $RES/marks.tsv; cred.py joins marks with the power logs.
BENCHY_SRC=$G/llama-benchy-fork; MODEL=qwen3.8-flash-next; CRED=$BL/k84; CODING=$BL/k76/coding.py
TOK=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
SAMP="--temperature 1.0 --top-p 0.95 --top-k 20"
NOTHINK='{"enable_thinking": false}'
x() { if [ "$1" = $H1 ]; then bash -c "$2"; else ssh -n -o ConnectTimeout=10 $H2 "$2"; fi; }
hn() { [ "$1" = $H1 ] && echo dgx01 || echo dgx02; }

PWQ='nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu --format=csv,noheader,nounits -lms 1000 | while IFS= read -r l; do echo "$EPOCHREALTIME,$l"; done'
pw_start() {
  setsid bash -c "$PWQ" > "$RES/power-dgx01.csv" 2>&1 < /dev/null 9>&- & PW1=$!
  ssh -n $H2 "setsid bash -c '$PWQ' > /tmp/$J-power.csv 2>&1 < /dev/null & echo \$! > /tmp/$J-power.pid" 9>&-; }
pw_stop() {
  [ -n "${PW1:-}" ] && kill -- -"$PW1" 2>/dev/null; PW1=
  ssh -n -o ConnectTimeout=10 $H2 "p=\$(cat /tmp/$J-power.pid 2>/dev/null) && [ -n \"\$p\" ] && kill -- -\$p 2>/dev/null; rm -f /tmp/$J-power.pid" 9>&-
  ssh -n -o ConnectTimeout=10 $H2 "cat /tmp/$J-power.csv" > "$RES/power-dgx02.csv" 2>/dev/null 9>&-; }

mark() { printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$EPOCHREALTIME" "$4" >> "$RES/marks.tsv"; }
unit() { printf '%s\t%s\t%s\t%s\t%s\n' "$@" >> "$RES/units.tsv"; }
bench() { # unit host name timeout args...
  local u=$1 h=$2 n=$3 t=$4 s=$EPOCHREALTIME rc; shift 4; mkdir -p "$RES/$u"
  ( cd "$R" && timeout -k 30 "$t" uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$h:8000/v1" --model $MODEL \
      --tokenizer "$TOK" --metrics-url "http://$h:8000/metrics" --save-result "$RES/$u/$n.json" --format json --runs 3 \
      --prompt-mode task --no-force-length --enable-prefix-caching --pp 2048 --tg 512 --depth 0 "$@" \
      > "$RES/$u/$n.log" 2>&1 < /dev/null 9>&- ); rc=$?; mark "$u" "$n" "$s" $rc; log "$u $n exit=$rc"; }
coding() { # unit host mode conc passes
  local u=$1 h=$2 s=$EPOCHREALTIME rc n=coding-$3-c$4; mkdir -p "$RES/$1"
  timeout -k 30 5400 python3 "$CODING" "http://$h:8000" "$3" "$4" "$5" "$RES/$u/$n.json" > "$RES/$u/$n.log" 2>&1 < /dev/null 9>&-
  rc=$?; mark "$u" "$n" "$s" $rc; log "$u $n exit=$rc"; }
idle() { local s=$EPOCHREALTIME; sleep "$2"; mark "$1" idle "$s" 0; }
std_cells() { # unit host passes [nodefault]
  idle "$1" 120
  bench "$1" "$2" tgdef-c1 900 --concurrency 1 $SAMP
  bench "$1" "$2" tgdef-c8 1500 --concurrency 8 $SAMP
  bench "$1" "$2" tgt0-c1 900 --concurrency 1 --temperature 0 --chat-template-kwargs "$NOTHINK"
  bench "$1" "$2" tgt0-c8 1500 --concurrency 8 --temperature 0 --chat-template-kwargs "$NOTHINK"
  coding "$1" "$2" t0-nothink 1 "$3"; coding "$1" "$2" t0-nothink 8 "$3"
  [ "${4:-}" = nodefault ] && return 0
  coding "$1" "$2" default 1 "$3"; coding "$1" "$2" default 8 "$3"; }

serve_log() { # host out: the serve log of the sparkrun container on host (running or exited)
  x "$1" "c=\$(docker ps -a --format '{{.Names}}' | grep -E '_solo|sparkrun|node_0' | head -1); [ -n \"\$c\" ] && { docker exec \$c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs \$c 2>&1; }" > "$2" 2>&1; }
boot() { # unit recipe timeout host|2x  -> 0 when /health is 200 and the pong answers on the serving host
  local u=$1 rec=$2 to=$3 w=$4 h s=$(date +%s) a; mkdir -p "$RES/$u"; cp "$rec" "$RES/$u/recipe.yaml"
  # two boots may run at once (one per Spark): sparkrun calls take turns, the health waits overlap
  if [ "$w" = 2x ]; then h=$H1; ( cd "$(dirname "$rec")" && flock "$RES/.sparkrun.lock" timeout -k 60 1800 sparkrun run "$rec" --no-follow ) >> "$RES/$u/sparkrun.log" 2>&1 < /dev/null 9>&-
  else h=$w; ( cd "$(dirname "$rec")" && flock "$RES/.sparkrun.lock" timeout -k 60 900 sparkrun run "$rec" --hosts "$w" --solo --no-follow ) >> "$RES/$u/sparkrun.log" 2>&1 < /dev/null 9>&-; fi
  while [ "$(health $h)" != 200 ]; do
    a=$(( $(date +%s) - s ))
    if [ $a -gt "$to" ]; then log "$u: no /health after $a s"; serve_log $h "$RES/$u/serve-$(hn $h).log"; return 1; fi
    if [ $a -gt 240 ] && [ -z "$(x $h "docker ps -q --filter name=sparkrun")" ]; then
      log "$u: no sparkrun container on $(hn $h) after $a s"; serve_log $h "$RES/$u/serve-$(hn $h).log"; return 1; fi
    sleep 20; done
  log "$u: healthy after $(( $(date +%s) - s )) s"; echo "boot_s $(( $(date +%s) - s ))" > "$RES/$u/boot.txt"
  serve_log $h "$RES/$u/serve-$(hn $h).log"
  for g in $H1 $H2; do [ "$w" = 2x ] || [ $g = "$w" ] || continue
    x $g "docker image inspect $(sed -n 's/^container: *//p' "$rec" | head -1)" > "$RES/$u/image-$(hn $g).json" 2>/dev/null; done
  a=$(pong $h); echo "pong $a" >> "$RES/$u/boot.txt"; echo "$a" | grep -qi pong; }
pair() { # unit1 recipe1 unit2 recipe2 timeout: boots a 1x recipe on each Spark at once (unit1 on dgx-01)
  boot "$1" "$2" "$5" $H1 & local a=$!; boot "$3" "$4" "$5" $H2 & local b=$!; local r=0
  CH="$a $b"; wait $a || r=1; wait $b || r=1; CH=; return $r; }
