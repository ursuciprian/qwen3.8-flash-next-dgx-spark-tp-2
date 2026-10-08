#!/usr/bin/env bash
# r3data (2026-10-08, #97): new on-policy data for refit run3, on both Sparks, plus the p2 responses the p2 capture
# skipped. Copy of this file runs as ~/GEN-AI/backlog/r3data/run.sh on dgx-01.
#   why     p2 captured only responses of <= 4096 tokens (tail 6144 - 2048 context): 1536 of 1766 docs, 0.96M of the
#           3.52M generated tokens. The 230 longer ones (2.6M tokens, 164 of them code) never reached training.
#   gen     the p2 generation server (refit-p2/gen.yaml: v3d f4..03 on the mtpcap image, capture off, max_num_seqs 8)
#           on each Spark (solo), gen.py run at c8, server sampling, effort medium, max 16384 (as p2), on the r3 prompt
#           mix (mix-r3.yaml: 6000 new prompts, none sharing content with the p2 set or the benchmarks), even/odd split
#           by Spark, BUDGET/2 response tokens each; a server that dies mid-run is rebooted and the run resumed (x2)
#   capture refit-p2/cap.yaml (capture on, decode numerics, no prefix caching) in three passes per Spark, each with
#           a tail just long enough for its responses plus 2048 context rows (p2 stored 6144 rows for every doc):
#             t3  tail 3072, responses <= 1024        t6  tail 6144, 1025..4096        t18  tail 18432, 4097..16384
#           dgx-01's t18 pass also takes the p2 responses > 4096 tokens (train -> training, held-out -> heldout-long)
#   assemble per pass on the Spark that captured it, judged by assemble's stats (no incomplete, no length mismatch,
#           every manifest doc seen), then that pass's shards are deleted (disk); dgx-02's data dirs copied to dgx-01
#   end     lib.sh restores the shipped 2x with a pong
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; comment.md + RESULT for the poster.
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=r3data; K=$HOME/GEN-AI/backlog/r3data
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/refit-r3data-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
W=$G/refit-r3; DAY=20261008
D3=$HOME/.cache/huggingface/mtp-refit/r3-$DAY; CD3=/cache/huggingface/mtp-refit/r3-$DAY
D2=$HOME/.cache/huggingface/mtp-refit/p2-20261006
IMG=spark-vllm-b12x:mtpcap-21e0b201-$(cut -c1-8 "$G/refit-p2/COMMIT")
GENREC=$G/refit-p2/gen.yaml; CAPREC=$G/refit-p2/cap.yaml
BUDGET=${BUDGET:-7200000}; ROWB=20600   # bytes per captured row (p2: 92 GiB shards / 4.76M rows)
PASSES="t3:3072:0 t6:6144:1024 t18:18432:4096"   # name:tail:skip responses <= this
CN=refit-r3-py

ip() { [ $1 = 1 ] && echo $H1 || echo $H2; }
x() { if [ $1 = 1 ]; then bash -c "$2"; else ssh -n -o ConnectTimeout=10 $H2 "export PATH=\$HOME/.local/bin:\$PATH; $2"; fi; }
ctr() { x $1 "docker ps -a --format '{{.Names}}' | grep -E '_solo|sparkrun' | head -1"; }
stop_h() { timeout -k 30 300 sparkrun stop --all --hosts $(ip $1) >> "$RES/$J.log" 2>&1 9>&-; sleep 5
  x $1 "docker ps -aq --filter name=sparkrun | xargs -r docker rm -f; docker ps -a --format '{{.Names}}' | grep _solo | xargs -r docker rm -f" >/dev/null 2>&1; }
keep_logs() { local c; c=$(ctr $1); [ -n "$c" ] && x $1 "docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1" > "$2/serve.log" 2>&1; }
boot() { # host recipe outdir -> 0 healthy with pong
  local h=$1 rec=$2 d=$3 s c p; mkdir -p "$d"; stop_h $h; cp "$rec" "$d/recipe.yaml"
  if [ $h = 1 ]; then ( cd "$W" && timeout -k 30 900 sparkrun run "$rec" --hosts $H1 --solo --no-follow ) > "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  else scp -q "$rec" "$H2:$W/$(basename "$rec")" && x 2 "cd $W && timeout -k 30 900 sparkrun run $(basename "$rec") --hosts $H2 --solo --no-follow" > "$d/sparkrun.log" 2>&1 9>&-; fi
  s=$(date +%s)
  until [ "$(health $(ip $h))" = 200 ]; do
    c=$(ctr $h)
    if [ $(( $(date +%s) - s )) -gt 180 ] && { [ -z "$c" ] || x $h "docker exec $c grep -qE 'Worker failed with error|EngineCore failed to start|Engine core initialization failed' /tmp/sparkrun_serve.log"; }; then
      keep_logs $h "$d"; log "s0$h: boot $(basename "$rec") FAILED"; return 1; fi
    [ $(( $(date +%s) - s )) -gt 3600 ] && { keep_logs $h "$d"; log "s0$h: boot $(basename "$rec") health timeout"; return 1; }
    sleep 15; done
  keep_logs $h "$d"; p=$(pong $(ip $h))
  log "s0$h: boot $(basename "$rec") up in $(( $(date +%s) - s ))s, pong=$p, b12x measured lines: $(grep -hoE 'b12x ready [a-z_.]+: [0-9]+/[0-9]+ ready, [0-9]+ measured' "$d/serve.log" | grep -cv ' 0 measured')"
  echo "$p" | grep -qi pong; }
boot2() { boot "$@" || { log "s0$1: retrying boot once"; boot "$@"; }; }
toks() { python3 - "$@" <<'EOF'
import glob, json, os, sys
n = t = 0
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        for l in open(f):
            r = json.loads(l); n += 1; t += len(r["output_token_ids"])
print(n, t)
EOF
}
rows_needed() { # skip gen dirs... -> rows the three passes store for responses > skip tokens (min(length, tail) per doc)
  python3 - "$@" <<'EOF'
import glob, json, os, sys
rows, skip = 0, int(sys.argv[1])
for d in sys.argv[2:]:
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        for l in open(f):
            r = json.loads(l); o = len(r["output_token_ids"]); n = o + len(r["prompt_token_ids"])
            tail = 3072 if o <= 1024 else 6144 if o <= 4096 else 18432 if o <= 16384 else 0
            rows += min(n, tail) if o > skip else 0
print(rows)
EOF
}
pyrun_h() { # host args... -> python3 in IMG on that host (CPU), tools = $W/src, data under /cache/huggingface
  x $1 "docker rm -f $CN >/dev/null 2>&1; docker run --rm --name $CN --ipc=host -v \$HOME/.cache/huggingface:/cache/huggingface \
    -v $W:/work -w /work/src -e PYTHONPATH=/work/src -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 $IMG ${*:2} < /dev/null"; }

spark() { # host -> gen, capture passes, assemble; writes $RES/s0<h>.done with "ok" or the failing step
  local h=$1 g=$D3/gen-0$1 b=$(( BUDGET / 2 )) try n t pass tl lo rec ok=1 out
  local srv=http://$(ip $h):8000
  for try in 1 2 3; do
    read -r n t <<< "$(toks "$g")"; [ "${t:-0}" -ge $(( b * 95 / 100 )) ] && break
    boot2 $h "$GENREC" "$RES/s0$h-gen-boot$try" || { echo "gen boot" > "$RES/s0$h.done"; return 1; }
    log "s0$h: gen try $try, $t of $b tokens so far"
    ( cd "$W/src" && python3 -m tools.mtp_refit.gen run --server $srv --prompts "$D3/prompts" --ids "$D3/ids-0$h.txt" --out "$g" \
        --concurrency 8 --budget-tokens $(( b - t )) --timeout 3600 ) >> "$RES/s0$h-gen.log" 2>&1
    [ "$(health $(ip $h))" = 200 ] && break   # finished with the server up: budget reached or prompts exhausted
    log "s0$h: gen server not healthy after the run"; keep_logs $h "$RES/s0$h-gen-boot$try"; done
  read -r n t <<< "$(toks "$g")"; log "s0$h: gen done, $n records, $t response tokens"; stop_h $h
  [ "${t:-0}" -gt 0 ] || { echo "gen produced nothing" > "$RES/s0$h.done"; return 1; }
  local need free; need=$(rows_needed 0 "$g"); [ $h = 1 ] && need=$(( need + $(rows_needed 4096 "$D2/gen") ))
  free=$(x $h "df -BG --output=avail \$HOME/.cache/huggingface | tail -1 | tr -d ' G'")
  log "s0$h: capture stores ~$need rows, ~$(( need * ROWB / 1000000000 )) GB per copy (shards, then data), $free GB free"
  [ $(( need * ROWB * 2 / 1000000000 + 50 )) -le "$free" ] || { echo "disk: $free GB free < 2 x $(( need * ROWB / 1000000000 )) + 50" > "$RES/s0$h.done"; return 1; }
  for pass in $PASSES; do
    IFS=: read -r pn tl lo <<< "$pass"; rec=$W/cap-s0$h-$pn.yaml
    x $h "mkdir -p $D3/capture/shards-s0$h-$pn"
    boot2 $h "$rec" "$RES/s0$h-cap-$pn" || { echo "cap boot $pn" > "$RES/s0$h.done"; return 1; }
    out=$D3/capture/man-s0$h-$pn; rm -f "$out.all.jsonl"
    ( cd "$W/src" && python3 -m tools.mtp_refit.capture_client --server $srv --gen "$g" --out "$out-gen" --concurrency 4 \
        --tail $tl --min-response $lo ) > "$RES/s0$h-capture-$pn.log" 2>&1
    if [ $h = 1 ] && [ $pn = t18 ]; then   # the p2 responses p2's capture skipped (> 4096 tokens), both splits
      ( cd "$W/src" && python3 -m tools.mtp_refit.capture_client --server $srv --gen "$D2/gen" --out "$out-p2" --concurrency 4 \
          --tail $tl --min-response $lo ) > "$RES/s01-capture-t18-p2.log" 2>&1; fi
    sleep 40   # the hook flushes its last shard after 30 s idle
    keep_logs $h "$RES/s0$h-cap-$pn"; stop_h $h
    cat "$out"-*/manifest.jsonl > "$out.all.jsonl" 2>/dev/null
    log "s0$h: capture $pn: $(tail -qn1 "$RES"/s0$h-capture-$pn*.log | tr '\n' ' ')"
    [ -s "$out.all.jsonl" ] || { log "s0$h: capture $pn: empty manifest"; continue; }
    [ $h = 2 ] && scp -q "$out.all.jsonl" "$H2:$out.all.jsonl"
    x $h "rm -rf $D3/data-s0$h-$pn"
    pyrun_h $h -m tools.mtp_refit.assemble --capture $CD3/capture/shards-s0$h-$pn --manifest $CD3/capture/man-s0$h-$pn.all.jsonl \
      --out $CD3/data-s0$h-$pn > "$RES/s0$h-assemble-$pn.txt" 2>&1
    if python3 - "$RES/s0$h-assemble-$pn.txt" "$out.all.jsonl" <<'EOF'
import json, sys
s = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
n = sum(1 for _ in open(sys.argv[2]))
sys.exit(0 if s["docs"] == n and s["incomplete"] == 0 and s["length_mismatch"] == 0 and s["manifest_unseen"] == 0 else 1)
EOF
    then x $h "rm -rf $D3/capture/shards-s0$h-$pn"; log "s0$h: assemble $pn ok, shards deleted: $(tail -1 "$RES/s0$h-assemble-$pn.txt")"
    else ok=0; log "s0$h: assemble $pn NOT clean, shards kept: $(tail -1 "$RES/s0$h-assemble-$pn.txt")"; fi
  done
  [ $ok = 1 ] && echo ok > "$RES/s0$h.done" || echo "assemble not clean" > "$RES/s0$h.done"; }

anchors() { # data dir -> "anchors docs" (refit-p3/job.sh anchors(): loss_mask ones minus one per doc)
  python3 - "$1" <<'EOF'
import glob, json, os, struct, sys
n = docs = 0
for p in glob.glob(os.path.join(sys.argv[1], "part-*.safetensors")):
    with open(p, "rb") as f:
        h = json.loads(f.read(struct.unpack("<Q", f.read(8))[0])); base = f.tell()
        a, b = h["loss_mask"]["data_offsets"]; f.seek(base + a); m = f.read(b - a)
        n += len(m) - m.count(0); docs += len(json.loads(h["__metadata__"]["docs"]))
print(n - docs, docs)
EOF
}

preflight() {
  local ok=0 h
  for h in 1 2; do
    x $h "docker image inspect $IMG >/dev/null 2>&1 && test -s $HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/f400000000000000000000000000000000000003/model-00035-of-00036.safetensors && command -v sparkrun >/dev/null" \
      || { echo "dgx-0$h: $IMG, f4..03 or sparkrun missing"; ok=1; }; done
  [ -s "$GENREC" ] && [ -s "$CAPREC" ] && grep -q '^  max_num_seqs: 8$' "$GENREC" || { echo "refit-p2 gen/cap recipes missing or not max_num_seqs 8"; ok=1; }
  for p in $PASSES; do IFS=: read -r pn tl lo <<< "$p"; for h in 1 2; do
    [ -s "$W/cap-s0$h-$pn.yaml" ] && grep -qx "  VLLM_MTP_CAPTURE_TAIL: \"$tl\"" "$W/cap-s0$h-$pn.yaml" \
      && grep -qx "  VLLM_MTP_CAPTURE_DIR: \"$CD3/capture/shards-s0$h-$pn\"" "$W/cap-s0$h-$pn.yaml" \
      && [ "$(diff "$CAPREC" "$W/cap-s0$h-$pn.yaml" | grep -c '^>')" = 2 ] || { echo "$W/cap-s0$h-$pn.yaml wrong"; ok=1; }; done; done
  ( cd "$W/src" && python3 -m py_compile tools/mtp_refit/*.py && python3 -m tools.mtp_refit.capture_client --help | grep -q -- --min-response ) \
    || { echo "$W/src tools do not parse or lack --min-response"; ok=1; }
  x 2 "test -s $W/src/tools/mtp_refit/assemble.py" || { echo "dgx-02: $W/src missing"; ok=1; }
  python3 -c "import json,sys; m=json.load(open(sys.argv[1])); assert sum(c['count'] for c in m['categories'].values()) >= 5000" "$D3/prompts/MANIFEST.json" 2>/dev/null \
    && [ -s "$D3/ids-01.txt" ] && [ -s "$D3/ids-02.txt" ] || { echo "prompts/ids in $D3 missing"; ok=1; }
  for h in 1 2; do [ -e "$D3/gen-0$h" ] && echo "note: $D3/gen-0$h exists (resumed: gen skips finished ids)"; done
  ls "$D2"/gen/*.jsonl >/dev/null 2>&1 || { echo "p2 gen missing"; ok=1; }
  echo "free GB: dgx-01 $(df -BG --output=avail "$HOME/.cache/huggingface" | tail -1 | tr -d ' G'), dgx-02 $(x 2 "df -BG --output=avail \$HOME/.cache/huggingface | tail -1 | tr -d ' G'")"
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  preflight; rc=$?
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "p2 long responses to recapture: $(python3 -c "
import glob, json; r=[json.loads(l) for f in glob.glob('$D2/gen/*.jsonl') for l in open(f)]; l=[x for x in r if len(x['output_token_ids'])>4096]
print(len(l), 'docs', sum(len(x['output_token_ids']) for x in l), 'tokens')")"
  echo "estimate: gen $BUDGET tokens at ~2 x 137 tok/s (p2 c8 rate) ~$(( BUDGET / 274 / 3600 )) h, 6 cap boots + capture at ~776 rows/s per Spark ~2-3 h, assemble ~1 h, restore ~10 min"
  echo "dry-run exit=$rc"; exit $rc; fi
need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
eval "lib_$(declare -f restore)"
restore() { [ -n "${S1:-}" ] && kill $S1 2>/dev/null; [ -n "${S2:-}" ] && kill $S2 2>/dev/null; x 1 "docker rm -f $CN" >/dev/null 2>&1; x 2 "docker rm -f $CN" >/dev/null 2>&1; lib_restore; }
job_begin
trap 'log "signal: stopping spark pipelines ${S1:-} ${S2:-}"; for p in ${S1:-} ${S2:-}; do kill -TERM $(ps -o pid= --ppid $p) $p 2>/dev/null; done; FINAL="FAILED: stopped by signal"; exit 143' TERM INT HUP
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
REP=$RES/r3data.txt
echo "r3data $(TZ=Europe/Bucharest date '+%F %T %Z'): #97 run3 data, tools $(cat "$W/src/COMMIT" 2>/dev/null), image $IMG, budget $BUDGET, data $D3" > "$REP"
st "running: preflight"
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
stop_all
st "running: gen + capture on both Sparks"
spark 1 & S1=$!; spark 2 & S2=$!
wait $S1; wait $S2; S1=; S2=
for h in 1 2; do log "s0$h: $(cat "$RES/s0$h.done" 2>/dev/null || echo 'no result')"; done
st "running: copy dgx-02 data"
for d in $(x 2 "ls -d $D3/data-s02-* 2>/dev/null"); do rsync -a "$H2:$d/" "$d/" >> "$RES/$J.log" 2>&1 || log "rsync $d failed"; done

{ echo; echo "== generation (response tokens; budget $BUDGET over both Sparks)"
  for h in 1 2; do echo "dgx-0$h: $(toks "$D3/gen-0$h" | awk '{print $1" records, "$2" tokens"}')"; done
  echo; echo "== capture + assemble per pass (docs, rows; anchors = generated tokens with context)"
  for d in "$D3"/data-s0*; do for s in train heldout; do [ -d "$d/$s" ] || continue
    echo "$(basename "$d")/$s: $(anchors "$d/$s" | awk '{print $1" anchors, "$2" docs"}')"; done; done
  for f in "$RES"/s0*-assemble-*.txt; do echo "$(basename "$f" .txt): $(tail -1 "$f")"; done
  echo; for h in 1 2; do echo "dgx-0$h pipeline: $(cat "$RES/s0$h.done" 2>/dev/null || echo 'no result')"; done; } >> "$REP"
NEW=$(for d in "$D3"/data-s0*/train; do anchors "$d"; done | awk '{s+=$1} END {print s+0}')
P2T=$(anchors "$D2/data/train" | cut -d' ' -f1)
echo "train anchors: p2 captured $P2T, new $NEW (r3 gen + p2 long), total $(( P2T + NEW ))" >> "$REP"
log "train anchors: p2 $P2T + new $NEW"
[ "$NEW" -gt 0 ] || { FINAL="FAILED: no new training data ($REP)"; exit 1; }
S=$(cat "$RES"/s0?.done 2>/dev/null | sort -u | tr '\n' ' ')
FINAL="DONE: r3data $NEW new train anchors (p2 had $P2T), pipelines: $S($REP)"
{ echo "Run3 data is captured: $NEW new training anchors next to the $P2T that run1 and run2 trained on."
  echo; echo "The p2 capture kept only responses of up to 4096 tokens (tail 6144 minus 2048 context rows), so 230 of the 1766 p2 responses (2.6M of the 3.52M generated tokens, mostly code) never reached training. This job captured those with a longer tail, plus new generations from a new prompt mix (no content shared with the p2 set or the benchmarks; C++/Rust/Go code and long documents added) on both Sparks."
  echo; echo '```'; sed -n '2,$p' "$REP" | head -c 30000; echo '```'
  echo; echo "Raw data on dgx-01: \`$RES\`."; } > "$K/comment.md"
