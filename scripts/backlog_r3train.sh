#!/usr/bin/env bash
# r3train (2026-10-08, #97): refit run3 on the p2 data plus r3data's (backlog/r3data), two inits side by side, one per
# Spark, same data, settings and seed. Copy of this file runs as ~/GEN-AI/backlog/r3train/run.sh on dgx-01.
#   run3a   dgx-01, from refit run1 (snapshot f4..61 = f4..03 + the run1 dense tensors)
#   run3b   dgx-02, from the original drafter (f4..03)
#   train   run1/run2 settings (--trainable dense --depth 6 --depth-weights equal --loss kl --topk 20 --window 2048
#           --lr 2e-5 --warmup 20 --tokens-per-step 8192 --kv-fp8 on) plus a linear lr decay to 0.1 x at the last step
#           (--min-lr-frac 0.1; run1/run2 kept the lr constant), seed 113. Steps = one epoch of the combined train
#           anchors, capped at TRAIN_S of dgx-01's warm-up rate. Checkpoint every quarter.
#   eval    refit-p3's eval_offline on the p2 held-out set (the run1/run2 eval set: EVARGS, fp8 drafter K/V, HC MXFP8,
#           T=0 and T=1, every category), each run on its own Spark; report-only: the p2 held-out responses > 4096
#           tokens (r3data captured them; never in any eval before) for run1, run3a, run3b
#   pick    run3 = the run with the larger mean T=0 per-position gain over run1 (positions 1-6, category all) ->
#           refit-r3/runs/run3 + refit-r3/eval-refit-run3.json, which backlog/k56c screens live on >= +1.00 pt/pos
#   pre     the parity gate under the user's coverage acceptance (refit-p3 rescore, as refit-p3/job.sh)
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; comment.md + RESULT for the poster; 2x restored on exit.
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=r3train; K=$HOME/GEN-AI/backlog/r3train
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/refit-run3-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
W=$G/refit-r3; P3=$G/refit-p3; DAY=20261008
D3=$HOME/.cache/huggingface/mtp-refit/r3-$DAY; CD3=/cache/huggingface/mtp-refit/r3-$DAY
D2=$HOME/.cache/huggingface/mtp-refit/p2-20261006; CD2=/cache/huggingface/mtp-refit/p2-20261006
IMG=spark-vllm-b12x:mtpcap-21e0b201-$(cut -c1-8 "$G/refit-p2/COMMIT")
SNR=hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots
SN03=/cache/huggingface/$SNR/f400000000000000000000000000000000000003; SN61=/cache/huggingface/$SNR/f400000000000000000000000000000000000061
VOCAB=/opt/mtp-vocab/ids-v2-K131072.txt.gz; TEXTFILE=/var/lib/node_exporter/textfile_collector
TRAIN_S=${TRAIN_S:-43200}; WARM_STEPS=8; TPS=8192; ACCEPT=coverage-2026-10-07; CN=refit-r3-py; TN=refit-r3-train
COMMON=(--draft-vocab $VOCAB --data $CD3/train-all --trainable dense --depth 6 --depth-weights equal --loss kl --topk 20
        --window 2048 --lr 2e-5 --warmup 20 --tokens-per-step $TPS --eval-windows 0 --kv-fp8 on --seed 113 --min-lr-frac 0.1)
EVARGS=(--snapshot $SN03 --draft-vocab $VOCAB --depth 6 --kv-fp8 on --hc-mxfp8 on)

x() { if [ $1 = 1 ]; then bash -c "$2"; else ssh -n -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=20 $H2 "$2"; fi; }
dock() { # host name gpu(0|1) args... -> docker run in IMG on that host; tools = $W/src, refit-p3 at /p3, textfile at /metrics
  local g=; [ $3 = 1 ] && g="--gpus all"
  x $1 "docker rm -f $2 >/dev/null 2>&1; docker run --rm --sig-proxy=false --name $2 $g --ipc=host -v \$HOME/.cache/huggingface:/cache/huggingface \
    -v $W:/work -v $P3:/p3 -v $TEXTFILE:/metrics -w /work/src -e PYTHONPATH=/work/src -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint python3 $IMG ${*:4} < /dev/null"; }
anchors() { # data dir -> "anchors docs"
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
gain() { # base.json new.json -> "mean d1 .. d6" (T=0, category all, positions 1-6; refit-p3/job.sh gain())
  python3 - "$1" "$2" <<'EOF'
import json, sys
b, r = (json.load(open(f))["all"]["t0"]["per_position"] for f in sys.argv[1:3])
d = [y - x for x, y in zip(b, r) if x is not None and y is not None]
assert len(d) == 6, (b, r)
print(f"{sum(d) / len(d):.4f}", *(f"{x:+.4f}" for x in d))
EOF
}
report() { # base.json new.json "label" -> per-category per-position lines (backlog/run2 report())
  python3 - "$@" <<'EOF'
import json, sys
b, r = (json.load(open(f)) for f in sys.argv[1:3])
print(f"\n== offline acceptance, {sys.argv[3]} (per position = vLLM's cumulative rate)")
for cat in sorted(r):
    for m in ("t0", "t1"):
        sb, sr = b.get(cat, {}).get(m), r[cat][m]
        if not sb: continue
        pp = " ".join(f"{x:.3f}->{y:.3f}" for x, y in zip(sb["per_position"], sr["per_position"]) if x is not None and y is not None)
        print(f"{cat:8s} {m} per position {pp} | tokens/step d4 {sb['tokens_per_step_d4']}->{sr['tokens_per_step_d4']}"
              f" d6 {sb['tokens_per_step']}->{sr['tokens_per_step']}")
EOF
}
datadirs() { echo "$D2/data/train"; ls -d "$D3"/data-s0*/train 2>/dev/null; }
preflight() {
  local ok=0 h d
  for d in $(datadirs); do ls "$d"/part-*.safetensors >/dev/null 2>&1 || { echo "no parts in $d"; ok=1; }; done
  [ "$(datadirs | wc -l)" -ge 2 ] || { echo "no r3data train dirs in $D3"; ok=1; }
  ls "$D2"/data/heldout/part-*.safetensors >/dev/null 2>&1 || { echo "p2 held-out missing"; ok=1; }
  head -1 "$BL/r3data/STATE" 2>/dev/null | grep -q '^DONE' || { echo "r3data not DONE"; ok=1; }
  for h in 1 2; do x $h "docker image inspect $IMG >/dev/null 2>&1 && test -s \$HOME/.cache/huggingface/$SNR/f400000000000000000000000000000000000061/SPLICE.json \
      && test -d \$HOME/.cache/huggingface/$SNR/f400000000000000000000000000000000000003" || { echo "dgx-0$h: $IMG, f4..61 or f4..03 missing"; ok=1; }; done
  for f in "$P3/eval-shipped.json" "$P3/eval-refit-run1.json" "$P3/runs/run1/mtp_refit.safetensors"; do [ -r "$f" ] || { echo "missing $f"; ok=1; }; done
  [ "$(gain "$P3/eval-shipped.json" "$P3/eval-refit-run1.json" | cut -d' ' -f1)" = 0.0688 ] || { echo "gain() does not reproduce run1's +0.0688"; ok=1; }
  x 2 "test -s $W/src/tools/mtp_refit/train.py" || { echo "dgx-02: $W/src missing"; ok=1; }
  dock 1 $CN 0 -m tools.mtp_refit.train --help 2>&1 | grep -q -- --min-lr-frac || { echo "train CLI lacks --min-lr-frac"; ok=1; }
  GV=$(dock 1 $CN 0 -m tools.mtp_refit.parity gate --drafts /p3/rescore-20261007/parity-drafts.json \
       --live /p3/rescore-20261007/live-t0.json /p3/rescore-20261007/live-t1.json --accepted $ACCEPT 2>&1 | tail -1)
  echo "parity gate (accepted $ACCEPT): $GV"; echo "$GV" | grep -q '^PASS' || ok=1
  [ -e "$W/runs/run3a" ] || [ -e "$W/runs/run3" ] && { echo "$W/runs/run3a or run3 exists (move it away)"; ok=1; }
  [ -w "$TEXTFILE" ] || { echo "$TEXTFILE not writable"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  preflight; rc=$?
  for d in $(datadirs); do echo "$d: $(anchors "$d")"; done 2>/dev/null
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "estimate: links + copy to dgx-02 ~30 min, warm-up ~8 min, train <= one epoch capped at $TRAIN_S s, evals ~20 min, restore ~10 min"
  echo "dry-run exit=$rc"; exit $rc; fi
need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
eval "lib_$(declare -f restore)"
restore() { for h in 1 2; do x $h "docker rm -f $CN $TN" >/dev/null 2>&1; done; lib_restore; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
REP=$RES/refit-run3.txt
echo "refit run3 $(TZ=Europe/Bucharest date '+%F %T %Z'): #97, tools $(cat "$W/src/COMMIT" 2>/dev/null), image $IMG, data p2-20261006 + r3-$DAY" > "$REP"
st "running: preflight"
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
grep '^parity gate' "$RES/preflight.txt" >> "$REP"
stop_all

st "running: train-all links + copy to dgx-02"
rm -rf "$D3/train-all" "$D3/heldout-long"; mkdir -p "$D3/train-all" "$D3/heldout-long"; i=0
for d in $(datadirs); do for p in "$d"/part-*.safetensors; do ln "$p" "$D3/train-all/$(printf 'part-%05d' $i).safetensors"; i=$((i + 1)); done; done
for p in "$D3"/data-s01-t18/heldout/part-*.safetensors; do [ -e "$p" ] && ln "$p" "$D3/heldout-long/"; done
read -r NA ND <<< "$(anchors "$D3/train-all")"; read -r HA HD <<< "$(anchors "$D2/data/heldout")"; read -r LA LD <<< "$(anchors "$D3/heldout-long")"
EPOCH=$(( NA / TPS ))
{ echo "data: train $ND docs / $NA anchors (~$EPOCH steps of $TPS per epoch; p2 alone: $(anchors "$D2/data/train" | cut -d' ' -f1)), eval p2 held-out $HD docs / $HA anchors, report-only held-out-long $LD docs / $LA anchors"; } >> "$REP"
s=$(date +%s)
x 2 "mkdir -p $D3/train-all $D3/heldout-long $D2/data/heldout $W/runs"
child rsync -a --delete "$D3/train-all/" "$H2:$D3/train-all/" && child rsync -a "$D3/heldout-long/" "$H2:$D3/heldout-long/" \
  && child rsync -a "$D2/data/heldout/" "$H2:$D2/data/heldout/" || { FINAL="FAILED: copy to dgx-02"; exit 1; }
[ "$(x 2 "ls $D3/train-all | wc -l")" = "$(ls "$D3/train-all" | wc -l)" ] || { FINAL="FAILED: dgx-02 train-all part count"; exit 1; }
log "copied to dgx-02 in $(( $(date +%s) - s )) s"

st "running: warm-up"
rm -rf "$W/runs/warmup"
child dock 1 $TN 1 -m tools.mtp_refit.train --snapshot $SN61 "${COMMON[@]}" --epochs 1 --max-steps $WARM_STEPS --out /work/runs/warmup \
  > "$RES/warmup.txt" 2>&1 || { FINAL="FAILED: warm-up ($RES/warmup.txt)"; exit 1; }
x 1 "docker run --rm --network none -v $W/runs:/r --entrypoint chmod $IMG -R a+rX /r"
RATE=$(python3 -c '
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
assert len(r) >= 3, len(r)
print(int(sum(x["anchors"][0] for x in r[1:]) / (r[-1]["s"] - r[0]["s"])))' "$W/runs/warmup/train.jsonl") || { FINAL="FAILED: warm-up rate"; exit 1; }
STEPS=$(( TRAIN_S * RATE / TPS )); [ $STEPS -gt $EPOCH ] && STEPS=$EPOCH; [ $STEPS -ge 100 ] || { FINAL="FAILED: only $STEPS steps planned"; exit 1; }
SAVE=$(( STEPS / 4 ))
echo "warm-up: $RATE anchors/s; plan: $STEPS steps x $TPS = $(( STEPS * TPS )) anchors ($(awk -v s=$STEPS -v e=$EPOCH 'BEGIN{printf "%.2f", s/e}') epoch), lr 2e-5 decaying linearly to 2e-6, checkpoint every $SAVE, ~$(( STEPS * TPS / RATE / 60 )) min" >> "$REP"
log "warm-up $RATE anchors/s -> $STEPS steps (~$(( STEPS * TPS / RATE / 60 )) min)"

st "running: train run3a (dgx-01, from run1) + run3b (dgx-02, from the original drafter), $STEPS steps"
TA=(--epochs 1 --max-steps $STEPS --save-every $SAVE)
dock 1 $TN 1 -m tools.mtp_refit.train --snapshot $SN61 "${COMMON[@]}" "${TA[@]}" --out /work/runs/run3a \
  --metrics-file /metrics/mtp_refit_train.prom > "$RES/train-run3a.txt" 2>&1 & A=$!
dock 2 $TN 1 -m tools.mtp_refit.train --snapshot $SN03 "${COMMON[@]}" "${TA[@]}" --out /work/runs/run3b > "$RES/train-run3b.txt" 2>&1 & B=$!
CH=$A; wait $A; RA=$?; CH=$B; wait $B; RB=$?; CH=
until ! x 2 "docker ps -q --filter name=^$TN\$" | grep -q .; do log "run3b still running on dgx-02 after its ssh client ended"; sleep 60; done
log "train exit: run3a $RA, run3b $RB"
x 1 "docker run --rm --network none -v $W/runs:/r --entrypoint chmod $IMG -R a+rX /r"
x 2 "docker run --rm --network none -v $W/runs:/r --entrypoint chmod $IMG -R a+rX /r"
mkdir -p "$W/runs/run3b"; rsync -a "$H2:$W/runs/run3b/" "$W/runs/run3b/" >> "$RES/$J.log" 2>&1
for r in run3a run3b; do [ -s "$W/runs/$r/mtp_refit.safetensors" ] && cp "$W/runs/$r/train.jsonl" "$RES/train-$r.jsonl" \
  && echo "train $r: last step $(tail -1 "$W/runs/$r/train.jsonl")" >> "$REP"; done
[ -s "$W/runs/run3a/mtp_refit.safetensors" ] || [ -s "$W/runs/run3b/mtp_refit.safetensors" ] || { FINAL="FAILED: no run3 weights (run3a exit $RA, run3b exit $RB)"; exit 1; }

st "running: evals"
ev() { # host run data out
  dock $1 $CN 1 -m tools.mtp_refit.eval_offline "${EVARGS[@]}" --data $3 ${2:+--refit /work/runs/$2/mtp_refit.safetensors} --out /work/$4 \
    > "$RES/${4%.json}.txt" 2>&1; }
evals() { # host run
  [ -s "$W/runs/$2/mtp_refit.safetensors" ] || return 0
  ev $1 $2 $CD2/data/heldout eval-refit-$2.json; [ "$LD" -gt 0 ] && ev $1 $2 $CD3/heldout-long eval-long-$2.json; return 0; }
( evals 1 run3a; [ "$LD" -gt 0 ] && dock 1 $CN 1 -m tools.mtp_refit.eval_offline "${EVARGS[@]}" --data $CD3/heldout-long \
    --refit /p3/runs/run1/mtp_refit.safetensors --out /work/eval-long-run1.json > "$RES/eval-long-run1.txt" 2>&1 ) & E1=$!
( evals 2 run3b ) & E2=$!
wait $E1; wait $E2
x 1 "docker run --rm --network none -v $W:/r --entrypoint sh $IMG -c 'chmod a+r /r/eval-*.json'" 2>/dev/null
x 2 "docker run --rm --network none -v $W:/r --entrypoint sh $IMG -c 'chmod a+r /r/eval-*.json'" 2>/dev/null
scp -q "$H2:$W/eval-*-run3b.json" "$W/" 2>/dev/null
cp "$P3/eval-shipped.json" "$P3/eval-refit-run1.json" "$RES/"; cp "$W"/eval-*.json "$RES/" 2>/dev/null

best=; bg=-1
for r in run3a run3b; do [ -s "$W/eval-refit-$r.json" ] || continue
  GL=$(gain "$P3/eval-refit-run1.json" "$W/eval-refit-$r.json"); GS=$(gain "$P3/eval-shipped.json" "$W/eval-refit-$r.json")
  echo "$r: T=0 gain vs run1 $(awk -v g=${GL%% *} 'BEGIN{printf "%+.2f", g*100}') pts/pos [${GL#* }], vs shipped $(awk -v g=${GS%% *} 'BEGIN{printf "%+.2f", g*100}') pts/pos" >> "$REP"
  awk -v g=${GL%% *} -v b=$bg 'BEGIN{exit !(g > b)}' && { best=$r; bg=${GL%% *}; }; done
[ -n "$best" ] || { FINAL="FAILED: no run3 eval ($RES)"; exit 1; }
{ for r in run3a run3b; do [ -s "$W/eval-refit-$r.json" ] && report "$P3/eval-refit-run1.json" "$W/eval-refit-$r.json" "p2 held-out, refit run1 -> $r"; done
  for r in run3a run3b; do [ -s "$W/eval-long-$r.json" ] && [ -s "$W/eval-long-run1.json" ] && report "$W/eval-long-run1.json" "$W/eval-long-$r.json" "report only: p2 held-out responses > 4096 tokens, refit run1 -> $r"; done; } >> "$REP"
rm -rf "$W/runs/run3"; cp -r "$W/runs/$best" "$W/runs/run3"; cp "$W/eval-refit-$best.json" "$W/eval-refit-run3.json"; echo "$best" > "$W/WINNER"
GPTS=$(awk -v g=$bg 'BEGIN{printf "%+.2f", g*100}')
echo "run3 = $best ($GPTS pts/pos T=0 over run1); k56c screens it live if >= +1.00" >> "$REP"
log "run3 = $best, $GPTS pts/pos over run1"
FINAL="DONE: run3 = $best, $STEPS steps, T=0 gain vs run1 $GPTS/pos ($REP)"
{ echo "Refit run 3 is trained: one epoch-sized run over the p2 data plus the new r3 data, two inits side by side (run3a from run1 on dgx-01, run3b from the original drafter on dgx-02), linear lr decay to 0.1x."
  echo; echo "Offline acceptance on the same p2 held-out set as run1 and run2, T=0, mean gain per draft position over run 1: $GPTS ($best). The k56c live screen (run 3 vs run 1) runs next only at +1.00 pt or more."
  echo; echo '```'; sed -n '2,$p' "$REP" | head -c 50000; echo '```'
  echo; echo "Raw data on dgx-01: \`$RES\`."; } > "$K/comment.md"
