#!/usr/bin/env bash
# refit-p3 (2026-10-06, #97 phase 3): first training run of the MTP drafter refit on dgx-01 (TP=1, GPU), offline eval
# against the shipped drafter, stop rule, and (only on a gain) a staged, not started, Thunderdome arm k56 vs shipped v3d.
# The shipped 2x is stopped for the run and restored at the end (also on failure). Design: drafter-refit-design.md.
#   0. preflight  refit-p2 ended DONE with parity drafts/live t0/live t1 all PASS, or REFIT_USER_ACCEPTED=coverage-<date>
#                 (user decision, 2026-10-07 18:55) and `parity gate --accepted` PASSes on the chain-K/V-fixed re-score
#                 in $P/rescore-20261007 (waives the decisive fraction only); else STATE "skipped: parity ...", nothing
#                 touched. Training image (refit-p2's mtpcap image), tools = this branch (refit-p3/src)
#   1. assemble   only if refit-p2 left no data/{train,heldout} parts (same command as p2; 5% held-out by prompt hash
#                 comes from the prompt split field)
#   2. warm-up    train.py --max-steps $WARM_STEPS, no metrics: anchors/s from train.jsonl (steps 2..N)
#   3. train run1 design settings: --trainable dense (experts + K131072 head frozen at the served NVFP4 values),
#                 --depth 6 --depth-weights equal --loss kl --topk 20 --window 2048 --lr 2e-5 --warmup 20
#                 --tokens-per-step 8192. Steps = min(2 epochs, $TRAIN_S x anchors/s / 8192); epochs = what those
#                 steps need (1 or 2). Checkpoint every ~1/6 of the run (ckpt-step*.safetensors). Metrics:
#                 /var/lib/node_exporter/textfile_collector/mtp_refit_train.prom (dashboard "MTP drafter refit").
#   4. eval       eval_offline.py on held-out, refit run1 (depth 6, window 2048, stride 1024: the p2 baseline's args),
#                 T=0 and T=1 per position + tokens/step; shipped = re-evaluated before training (step 1b) with the same
#                 tool, data and args (refit-p2's eval-baseline.json predates the chain-K/V fix and used BF16 K/V).
#                 Metrics: mtp_refit_eval_{refit,shipped}.prom.
#   5. stop rule  mean over positions 1-6 of (refit - shipped) per-position acceptance at T=0, category "all":
#                 < +0.02 -> "DONE: gain below threshold"; >= +0.02 -> splice f4..61 on both Sparks, b12x seed
#                 re-keyed to the f4..61 path (k48/remap_seed.py), image k56 on both Sparks, arm dirs k56/refit-01
#                 (dgx-01) + k56/refit-02 (dgx-02), shipped fp8 drafter KV (recipe = v3d + model/container/cache root),
#                 thunderdome --dry-run, k56/READY, then queues k56/start.sh (TD_ACC_RISE_OK=1, gate on PROMOTE) on the
#                 gpu-lock right after this job (user approval, 2026-10-07).
#   6. restore    the shipped 2x, pong
# Never takes the gpu-lock: the caller holds it (GPU_LOCK_HELD=1; after_p2.sh does that once refit-p2 is over).
# Usage:  flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash ~/GEN-AI/refit-p3/job.sh   (after_p2.sh does this)
#         bash ~/GEN-AI/refit-p3/job.sh --dry-run    (CPU only, safe next to a running job: files, image, tools,
#                                                     CPU tests incl. a tiny train step, k56 inputs)
# Output: $RES (results/refit-p3-<DAY>), $RES/refit-p3.txt, ~/GEN-AI/refit-p3/STATE. Runs in ~/GEN-AI/refit-p3/runs.
# Stop: kill -TERM <job pid in refit-p3.log>, never pkill -f.
set -u
export PATH="$HOME/.local/bin:$PATH"
P=$HOME/GEN-AI/refit-p3; P2=$HOME/GEN-AI/refit-p2; G=$HOME/GEN-AI; R=$G/qwen3.8-flash-next-dgx-spark-tp-2
K=$G/k56; TD=$R/scripts/thunderdome.sh
H1=192.168.100.62; H2=192.168.100.53
DAY=${DAY:-20261006}   # refit-p2 data day
D=$HOME/.cache/huggingface/mtp-refit/p2-$DAY; CD=/cache/huggingface/mtp-refit/p2-$DAY
RES2=$R/results/refit-p2-$DAY; RES=${RES:-$R/results/refit-p3-$DAY}
IMG=spark-vllm-b12x:mtpcap-21e0b201-$(cut -c1-8 "$P2/COMMIT")
V3D=ghcr.io/ursuciprian/spark-vllm-b12x:tp1-v3d-20261005-21e0b201-5dad364d-warm
SEED=8bb5a5b317a3975df599d900e295b12b6ee7225bd8eff6f0de3adc67fd55202e.json   # v3d's b12x seed in $V3D (f4..03 key)
F403=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/f400000000000000000000000000000000000003
F461=${F403%03}61
SN=/cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/f400000000000000000000000000000000000003
VOCAB=/opt/mtp-vocab/ids-v2-K131072.txt.gz
TEXTFILE=/var/lib/node_exporter/textfile_collector
TRAIN_S=${TRAIN_S:-5400}; WARM_STEPS=${WARM_STEPS:-5}; TPS=8192; MAX_EPOCHS=2; GAIN_MIN=0.02
PARITY_RE='^DONE: parity drafts=PASS live: t0=PASS t1=PASS'
COMMON=(--snapshot "$SN" --draft-vocab $VOCAB --data "$CD/data/train" --trainable dense --depth 6 --depth-weights equal
        --loss kl --topk 20 --window 2048 --lr 2e-5 --warmup 20 --tokens-per-step $TPS --eval-windows 0 --kv-fp8 on)
EVARGS=(--snapshot "$SN" --draft-vocab $VOCAB --data "$CD/data/heldout" --depth 6 --kv-fp8 on --hc-mxfp8 on)
# 2026-10-07: train and eval the SHIPPED v3d serving format: fp8 e4m3 drafter K/V at scale 1.0 (--kv-fp8 on), draft
# steps attend the prefill keys only (MtpRef chain_kv=False, the default since a73c0bd), HC MXFP8 on for eval. The BF16
# drafter KV (k65) did not ship; k56 keeps v3d's fp8 drafter KV. The earlier fp8 replay gap was the chain-key bug.
ACCEPT=${REFIT_USER_ACCEPTED:-}; ACCEPT_WHY="user accepted in chat 2026-10-07 18:55"
RESCORE=$P/rescore-20261007   # parity-drafts.json, live-t0.json, live-t1.json of the chain-K/V-fixed MtpRef (#97)

pyrun() { # python3 args... in IMG with the GPU (or CPU with PYDEV=cpu), tools = $P/src, textfile dir at /metrics
  docker rm -f refit-p3-py >/dev/null 2>&1
  local gpu=(--gpus all); [ "${PYDEV:-}" = cpu ] && gpu=()
  docker run --rm --name refit-p3-py "${gpu[@]}" --ipc=host -v "$HOME/.cache/huggingface:/cache/huggingface" -v "$P:/work" \
    -v "$TEXTFILE:/metrics" -w /work/src -e PYTHONPATH=/work/src -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 "$IMG" "$@" < /dev/null; }
anchors() { # data dir -> sum of loss_mask over its parts (stdlib safetensors read; = depth-0 anchors + 1 per doc)
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
gain() { # shipped.json refit.json -> "mean_gain per-position-diffs..." (T=0, category all, positions 1-6)
  python3 - "$1" "$2" <<'EOF'
import json, sys
b, r = (json.load(open(f))["all"]["t0"]["per_position"] for f in sys.argv[1:3])
d = [y - x for x, y in zip(b, r) if x is not None and y is not None]
assert len(d) == 6, (b, r)
print(f"{sum(d) / len(d):.4f}", *(f"{x:+.4f}" for x in d))
EOF
}

if [ "${1:-}" = --dry-run ]; then
  rc=0
  for f in "$P2/COMMIT" "$P/src/tools/mtp_refit/train.py" "$P/src/tools/mtp_refit/splice.py" "$P/after_p2.sh" "$K/start.sh" \
           "$G/k48/remap_seed.py" "$G/k53/v3d.yaml" "$G/k53/hook.sh" "$TD"; do [ -f "$f" ] || { echo "missing $f"; rc=1; }; done
  docker image inspect "$IMG" >/dev/null 2>&1 || { echo "training image $IMG missing"; rc=1; }
  chk="docker image inspect $V3D >/dev/null 2>&1 && test -d $F403 && ! test -e $F461"
  bash -c "$chk" || { echo "dgx-01: v3d image or f4..03 missing, or f4..61 exists"; rc=1; }
  ssh -n -o ConnectTimeout=10 $H2 "$chk" || { echo "dgx-02: v3d image or f4..03 missing, or f4..61 exists"; rc=1; }
  docker run --rm --entrypoint test "$V3D" -f /opt/b12x-seed/preparation/$SEED || { echo "seed $SEED not in $V3D"; rc=1; }
  [ -w "$TEXTFILE" ] || { echo "$TEXTFILE not writable"; rc=1; }
  grep -qx "model: $F403" "$G/k53/v3d.yaml" && grep -qx "container: $V3D" "$G/k53/v3d.yaml" || { echo "k53/v3d.yaml is not v3d"; rc=1; }
  python3 -m py_compile "$P"/src/tools/mtp_refit/*.py && python3 "$P/src/tools/mtp_refit/splice.py" 2>&1 | grep -q Overlay \
    || { echo "tools do not parse on the host"; rc=1; }
  # CPU in the training image: the whole CPU suite (toy model: tiny train steps with --save-every and --metrics-file,
  # eval, splice round trip) and the real CLIs' argument parsing. nice'd, no GPU, a few hundred MB.
  # pytest is not in the image: installed once into $P/testdeps (test runs only; training never sees it)
  [ -d "$P/testdeps/pytest" ] || docker run --rm -v "$P:/work" --entrypoint python3 "$IMG" -m pip install -q --no-cache-dir \
    --target /work/testdeps pytest > "$P/testdeps.log" 2>&1
  out=$(docker run --rm --cpus 4 -v "$P:/work" -w /work/src -e PYTHONPATH=/work/src:/work/testdeps -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint python3 "$IMG" -m pytest -q -p no:cacheprovider tools/mtp_refit/tests 2>&1 | tail -3); echo "CPU tests: $out"
  echo "$out" | grep -q passed && ! echo "$out" | grep -qE "failed|error" || { echo "CPU tests failed"; rc=1; }
  PYDEV=cpu pyrun -m tools.mtp_refit.train --help | grep -q -- --save-every && PYDEV=cpu pyrun -m tools.mtp_refit.eval_offline --help \
    | grep -q -- --metrics-file || { echo "train/eval CLI in the image lacks --save-every/--metrics-file"; rc=1; }
  echo "refit-p2 STATE: $(head -1 "$P2/STATE")"
  if [ -n "$ACCEPT" ]; then echo "parity gate (REFIT_USER_ACCEPTED=$ACCEPT): $(PYDEV=cpu pyrun -m tools.mtp_refit.parity gate \
    --drafts /work/rescore-20261007/parity-drafts.json --live /work/rescore-20261007/live-t0.json /work/rescore-20261007/live-t1.json \
    --accepted "$ACCEPT" 2>&1 | tail -1)"; fi
  grep -q -- '--kv-fp8 on' <<< "${COMMON[*]}" && grep -q -- '--kv-fp8 on --hc-mxfp8 on' <<< "${EVARGS[*]}" || { echo "train/eval not fp8 drafter KV + HC MXFP8"; rc=1; }
  ls "$D"/data/train/part-*.safetensors >/dev/null 2>&1 && echo "data: train $(anchors "$D/data/train") / heldout $(anchors "$D/data/heldout") (anchors docs)" \
    || echo "data: not assembled yet (refit-p2 checks step does it)"
  echo "estimate: stop 2x ~2 min, warm-up ~10 min, train ${TRAIN_S}s, eval ~= p2 baseline eval, stage k56 ~20 min if gain, restore ~10 min"
  echo "dry-run exit=$rc"; exit $rc; fi
[ "${GPU_LOCK_HELD:-}" = 1 ] || { echo "refused: hold ~/GEN-AI/gpu-lock and set GPU_LOCK_HELD=1" >&2; exit 2; }

mkdir -p "$RES" "$P/runs"
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$RES/refit-p3.log"; }
NOTE=
st() { { echo "$*"; [ -n "$NOTE" ] && echo "$NOTE"; } > "$P/STATE"; cp "$P/STATE" "$RES/STATE"; log "STATE: $*"; }
REP=$RES/refit-p3.txt
# refit-p2 gate, again under the lock (after_p2.sh checked it before waiting)
s2=$(head -1 "$P2/STATE" 2>/dev/null)
if ! echo "$s2" | grep -qE "$PARITY_RE"; then
  [ -n "$ACCEPT" ] || { st "skipped: parity not all PASS (refit-p2: $s2); nothing run"; exit 0; }
  GV=$(PYDEV=cpu pyrun -m tools.mtp_refit.parity gate --drafts /work/rescore-20261007/parity-drafts.json \
       --live /work/rescore-20261007/live-t0.json /work/rescore-20261007/live-t1.json --accepted "$ACCEPT" 2>&1 | tail -1)
  log "parity gate with REFIT_USER_ACCEPTED=$ACCEPT ($ACCEPT_WHY), re-score $RESCORE: $GV"
  echo "$GV" | grep -q '^PASS' || { st "skipped: parity gate FAIL under REFIT_USER_ACCEPTED=$ACCEPT: $GV; nothing run"; exit 0; }
  NOTE="override: REFIT_USER_ACCEPTED=$ACCEPT skips only the drafts decisive-fraction criterion ($ACCEPT_WHY); re-score $RESCORE: $GV"
  st "running: p3 gate (override)"; fi

health() { curl -s -m 5 -o /dev/null -w '%{http_code}' "$1:8000/health"; }
pong() { curl -s -m 120 "$1:8000/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Reply with exactly one word: pong"}],"max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>&1; }
n0() { docker ps --format '{{.Names}}' | grep -E 'node_0|_solo|sparkrun' | head -1; }
E=$(docker ps --format '{{.Names}} {{.Image}}' | awk '/node_0/ && /spark-vllm-b12x/ {print $2; exit}' | sed 's|.*:||')
E=${E:-b1.4-20261001-b7fbaf96-a7e649d8-warm}
is_shipped() { [ "$(health localhost)" = 200 ] && docker ps --format '{{.Image}}' | grep -q ":$E\$" \
  && ssh -n -o ConnectTimeout=10 $H2 "docker ps --format '{{.Image}}'" | grep -q ":$E\$" \
  && docker exec "$(n0)" sh -c "ps aux | grep '[v]llm serve'" | grep -q -- '--tensor-parallel-size 2'; }
stop_all() { ( cd "$R" && timeout -k 30 600 sparkrun stop --all ) >> "$RES/refit-p3.log" 2>&1 9>&-; sleep 10
  for h in $H1 $H2; do timeout -k 30 300 sparkrun stop --all --hosts $h >> "$RES/refit-p3.log" 2>&1 9>&-; done; sleep 5
  docker ps -q --filter name=sparkrun | xargs -r docker rm -f >/dev/null 2>&1
  ssh -n -o ConnectTimeout=10 $H2 "docker ps -q --filter name=sparkrun | xargs -r docker rm -f" >/dev/null 2>&1; }
restore() {
  docker rm -f refit-p3-py >/dev/null 2>&1
  if is_shipped && pong localhost | grep -qi pong; then log "restore: shipped 2x already serving"; return 0; fi
  stop_all; ( cd "$R" && timeout -k 60 3600 sparkrun run qwen3.8-flash-next-2x-dgx-spark --no-follow ) >> "$RES/refit-p3.log" 2>&1 < /dev/null 9>&-
  local s=$(date +%s); until [ "$(health localhost)" = 200 ] || [ $(( $(date +%s) - s )) -gt 1800 ]; do sleep 15; done
  local p; p=$(pong localhost)
  is_shipped && echo "$p" | grep -qi pong && { log "restore: shipped 2x ($E) verified, pong=$p"; return 0; }
  log "FAILED: restore -- MANUAL INTERVENTION (pong=$p)"; return 1; }
child() { "$@" & CH=$!; wait $CH; local rc=$?; CH=; return $rc; }

GUARD='while :; do m=$(awk "/^MemAvailable/{print int(\$2/1024)}" /proc/meminfo); echo "$(date +%T) $m"; if [ "$m" -lt 2048 ]; then docker rm -f refit-p3-py >/dev/null 2>&1; echo "ABORT MemAvailable $m MiB < 2048"; exit 3; fi; sleep 1; done'
bash -c "$GUARD" > "$RES/guard-dgx01.log" 2>&1 < /dev/null 9>&- & GU=$!
CH=; FINAL="FAILED: job"
trap 'kill $GU 2>/dev/null; restore || { sleep 60; restore; } || FINAL="FAILED: restore -- $FINAL"; st "$FINAL"' EXIT
trap 'log "signal: stopping child $CH"; [ -n "$CH" ] && kill -TERM $CH 2>/dev/null && wait $CH; docker rm -f refit-p3-py >/dev/null 2>&1; FINAL="FAILED: stopped by signal"; exit 143' TERM INT HUP
log "refit-p3 pid $$, RES $RES, data $D, image $IMG, tools $(cat "$P/src/COMMIT" 2>/dev/null), 2x tag for restore: $E"
if [ "${STAGE_ONLY:-}" = 1 ]; then { echo; echo "== restage $(TZ=Europe/Bucharest date '+%F %T %Z'): STAGE_ONLY=1, k56 from run1 and the evals above, tools $(cat "$P/src/COMMIT" 2>/dev/null)"; } >> "$REP"
else echo "refit-p3 $(TZ=Europe/Bucharest date '+%F %T %Z'): #97 phase 3 run1, tools $(cat "$P/src/COMMIT" 2>/dev/null), image $IMG; refit-p2: $s2" > "$REP"; fi
[ -n "$NOTE" ] && echo "$NOTE" >> "$REP"
echo "format: fp8 drafter K/V (--kv-fp8 on), chain_kv=False, HC MXFP8 on for eval (shipped v3d)" >> "$REP"

# STAGE_ONLY=1: skip steps 0-4 (no stop of the 2x, no training or eval); stop rule and staging from $RES's evals and
# runs/run1 (used after the 2026-10-07 staging failure: run1 files were root 0600)
if [ "${STAGE_ONLY:-}" != 1 ]; then
# 0. preflight
st "running: p3 preflight"   # not "running: preflight": the dashboard maps that to stage 1
docker image inspect "$IMG" >/dev/null 2>&1 || { FINAL="FAILED: preflight: image $IMG missing"; exit 1; }
# dashboard sidecar for phase 3 states (second p2_metrics instance, own file; the p2 file keeps the parity results)
( cd "$P/src" && setsid nohup nice -n 19 python3 -m tools.mtp_refit.p2_metrics --state "$P/STATE" --res "$RES2" --data "$D" \
    --run "refit-p3-$DAY" --out "$TEXTFILE/mtp_refit_p3.prom" >> "$P/p3_metrics.log" 2>&1 < /dev/null 9>&- & echo $! > "$P/p3_metrics.pid" )
log "dashboard sidecar pid $(cat "$P/p3_metrics.pid")"
stop_all   # the 2x refit-p2 restored; training wants the whole GPU and memory

# 1. assemble (refit-p2's checks step normally did it)
if ! ls "$D"/data/train/part-*.safetensors >/dev/null 2>&1 || ! ls "$D"/data/heldout/part-*.safetensors >/dev/null 2>&1; then
  st "running: assemble"
  rm -rf "$D/data"
  child pyrun -m tools.mtp_refit.assemble --capture "$CD/capture/shards" --manifest "$CD/capture/main/manifest.jsonl" --out "$CD/data" \
    > "$RES/assemble.txt" 2>&1 || { FINAL="FAILED: assemble ($RES/assemble.txt)"; exit 1; }
  echo "assemble: $(tail -1 "$RES/assemble.txt")" >> "$REP"; fi
read -r NA ND <<< "$(anchors "$D/data/train")"; read -r HA HD <<< "$(anchors "$D/data/heldout")"
[ "${NA:-0}" -gt $TPS ] && [ "${HA:-0}" -gt 0 ] || { FINAL="FAILED: data: train $NA anchors, heldout $HA"; exit 1; }
SPE=$(( NA / TPS ))
echo "data: train $ND docs / $NA anchors (~$SPE steps of $TPS per epoch), heldout $HD docs / $HA anchors" >> "$REP"
log "data: train $ND docs $NA anchors, heldout $HD docs $HA anchors"

# 1b. baseline: the shipped drafter on held-out with the fixed tool, before training (T=0 and T=1)
st "running: eval shipped"
child pyrun -m tools.mtp_refit.eval_offline "${EVARGS[@]}" --out /work/eval-shipped.json --metrics-file /metrics/mtp_refit_eval_shipped.prom \
  > "$RES/eval-shipped.txt" 2>&1 || { FINAL="FAILED: eval shipped ($RES/eval-shipped.txt)"; exit 1; }
cp "$P/eval-shipped.json" "$RES/eval-shipped.json"
python3 - "$RES/eval-shipped.json" >> "$REP" <<'EOF2'
import json, sys
r = json.load(open(sys.argv[1]))["all"]
print("baseline shipped (held-out, fixed tool, fp8 drafter KV):",
      "; ".join(f"{m} per position {r[m]['per_position']} tok/step d4 {r[m]['tokens_per_step_d4']}" for m in ("t0", "t1")))
EOF2
log "$(tail -1 "$REP")"

# 2. warm-up: measured anchors/s at the real settings
st "running: warm-up"
rm -rf "$P/runs/warmup"
child pyrun -m tools.mtp_refit.train "${COMMON[@]}" --epochs 1 --max-steps $WARM_STEPS --out /work/runs/warmup \
  > "$RES/warmup.txt" 2>&1 || { FINAL="FAILED: warm-up ($RES/warmup.txt)"; exit 1; }
RATE=$(python3 - "$P/runs/warmup/train.jsonl" <<'EOF'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
assert len(r) >= 3, len(r)
print(int(sum(x["anchors"][0] for x in r[1:]) / (r[-1]["s"] - r[0]["s"])))
EOF
) || { FINAL="FAILED: warm-up rate ($P/runs/warmup/train.jsonl)"; exit 1; }
STEPS=$(( TRAIN_S * RATE / TPS )); MAXS=$(( MAX_EPOCHS * SPE )); [ $STEPS -gt $MAXS ] && STEPS=$MAXS; [ $STEPS -ge 1 ] || STEPS=1
EPOCHS=$(( (STEPS + SPE - 1) / SPE )); [ $EPOCHS -ge 1 ] || EPOCHS=1
SAVE=$(( STEPS / 6 )); [ $SAVE -ge 1 ] || SAVE=1
ETA_S=$(( STEPS * TPS / RATE ))
{ echo "warm-up: $RATE anchors/s ($(tail -1 "$P/runs/warmup/train.jsonl"))"
  echo "run1 plan: $STEPS steps x $TPS anchors = $(( STEPS * TPS )) anchors ($(awk -v s=$STEPS -v e=$SPE 'BEGIN{printf "%.2f", s/e}') epochs, --epochs $EPOCHS), checkpoint every $SAVE steps, ~$(( ETA_S / 60 )) min"
  [ $STEPS -lt 100 ] && echo "WARN: under 100 optimizer steps (lr warm-up is 20)"; } >> "$REP"
log "warm-up: $RATE anchors/s -> run1 $STEPS steps, $EPOCHS epochs, ~$(( ETA_S / 60 )) min"

# 3. training run 1
st "running: train run1"
rm -rf "$P/runs/run1"
child pyrun -m tools.mtp_refit.train "${COMMON[@]}" --epochs $EPOCHS --max-steps $STEPS --save-every $SAVE --out /work/runs/run1 \
  --metrics-file /metrics/mtp_refit_train.prom > "$RES/train-run1.txt" 2>&1 \
  || { FINAL="FAILED: train run1 ($RES/train-run1.txt; checkpoints in $P/runs/run1)"; exit 1; }
[ -f "$P/runs/run1/mtp_refit.safetensors" ] || { FINAL="FAILED: train run1 wrote no mtp_refit.safetensors"; exit 1; }
# the container writes root 0600 files; splice.py and scp on the host read them
docker run --rm --network none -v "$P/runs:/r" --entrypoint chmod "$IMG" -R a+rX /r || { FINAL="FAILED: chmod runs"; exit 1; }
cp "$P/runs/run1/train.jsonl" "$RES/train-run1.jsonl"
echo "train run1: $(tail -1 "$RES/train-run1.txt"); last step $(tail -1 "$P/runs/run1/train.jsonl")" >> "$REP"

# 4. offline eval on held-out, T=0 and T=1
st "running: eval"
child pyrun -m tools.mtp_refit.eval_offline "${EVARGS[@]}" --refit /work/runs/run1/mtp_refit.safetensors --out /work/eval-refit-run1.json \
  --metrics-file /metrics/mtp_refit_eval_refit.prom > "$RES/eval-refit-run1.txt" 2>&1 \
  || { FINAL="FAILED: eval refit ($RES/eval-refit-run1.txt)"; exit 1; }
cp "$P/eval-refit-run1.json" "$RES/"
fi
[ -r "$P/runs/run1/mtp_refit.safetensors" ] && [ -s "$RES/eval-shipped.json" ] && [ -s "$RES/eval-refit-run1.json" ] \
  || { FINAL="FAILED: run1 weights or evals missing/unreadable"; exit 1; }
GL=$(gain "$RES/eval-shipped.json" "$RES/eval-refit-run1.json") || { FINAL="FAILED: gain computation"; exit 1; }
G0=${GL%% *}; GPTS=$(awk -v g=$G0 'BEGIN{printf "%+.2f", g*100}')
python3 - "$RES/eval-shipped.json" "$RES/eval-refit-run1.json" >> "$REP" <<'EOF'
import json, sys
b, r = (json.load(open(f)) for f in sys.argv[1:3])
print("\n== offline acceptance on held-out, shipped -> refit run1 (per position = vLLM's cumulative rate)")
for cat in sorted(r):
    for m in ("t0", "t1"):
        sb, sr = b.get(cat, {}).get(m), r[cat][m]
        if not sb: continue
        pp = " ".join(f"{x:.3f}->{y:.3f}" for x, y in zip(sb["per_position"], sr["per_position"]) if x is not None and y is not None)
        print(f"{cat:8s} {m} per position {pp} | tokens/step d4 {sb['tokens_per_step_d4']}->{sr['tokens_per_step_d4']}"
              f" d6 {sb['tokens_per_step']}->{sr['tokens_per_step']}")
EOF
echo "stop rule: mean T=0 per-position gain (all, positions 1-6) = $GPTS pts [${GL#* }], threshold +2.00" >> "$REP"
log "gain T=0: $GPTS pts per position (${GL#* })"

# 5. stop rule
if awk -v g=$G0 -v m=$GAIN_MIN 'BEGIN{exit !(g < m)}'; then
  FINAL="DONE: gain below threshold ($GPTS pts/pos T=0 < +2; $REP)"; exit 0; fi
st "running: stage k56"
RF=$P/runs/run1/mtp_refit.safetensors
stage() { # run as a plain "( stage )" (never under || or &&, which would switch set -e off)
  set -e
  mkdir -p "$K/seed" "$K/img"; rm -f "$K/READY" "$K"/seed/*.json
  python3 "$P/src/tools/mtp_refit/splice.py" "$F403" "$RF" "$F461" > "$RES/splice-dgx01.json"
  ssh -n $H2 "mkdir -p $P/runs/run1 $P/src/tools/mtp_refit"
  scp -q "$RF" "$H2:$RF"; scp -q "$P/src/tools/mtp_refit/splice.py" "$H2:$P/src/tools/mtp_refit/splice.py"
  ssh -n $H2 "python3 $P/src/tools/mtp_refit/splice.py $F403 $RF $F461" > "$RES/splice-dgx02.json"
  cmp <(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["shards"])' "$F461/SPLICE.json") \
      <(ssh -n $H2 "python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[\"shards\"])' $F461/SPLICE.json")
  docker run --rm --entrypoint cat "$V3D" /opt/b12x-seed/preparation/$SEED > "$K/seed-src.json"
  docker run --rm --entrypoint python3 -v "$K:/k" -v "$G/k48:/k48:ro" "$V3D" /k48/remap_seed.py /k/seed-src.json "$F461" /k/seed \
    > "$RES/remap-seed.txt"
  [ "$(ls "$K"/seed/*.json | wc -l)" = 1 ]
  local tag; tag=k56-refit-run1-$(sha256sum "$RF" | cut -c1-8); KIMG=spark-vllm-b12x:$tag
  { echo "# k56 (#97): shipped v3d warm image + the b12x seed re-keyed to the refit snapshot path f4..61 (k48/remap_seed.py"
    echo "# from the image's v3d seed $SEED). Nothing else changes: the refit MTP tensors live in the snapshot."
    echo "FROM $V3D"; echo "COPY seed/*.json /opt/b12x-seed/preparation/"; } > "$K/img/Dockerfile"
  rm -rf "$K/img/seed"; cp -r "$K/seed" "$K/img/seed"
  docker build -q -t "$KIMG" "$K/img" > "$RES/k56-build-dgx01.txt" 2>&1
  ssh -n $H2 "mkdir -p $K"; rsync -a --delete "$K/img/" "$H2:$K/img/"
  ssh -n $H2 "docker build -q -t $KIMG $K/img" > "$RES/k56-build-dgx02.txt" 2>&1
  cp "$G/k53/v3d.yaml" "$K/v3d.yaml"
  sed -e "1i # k56 arm (#97 refit run1, $(TZ=Europe/Bucharest date +%F)): v3d with the refit MTP dense tensors (snapshot f4..61 = f4..03 + spliced shard), image $KIMG (v3d + re-keyed seed)" \
      -e "s|^name: qwen3.8-flash-next-1x-dgx-spark-v3d\$|name: qwen3.8-flash-next-1x-dgx-spark-k56-refit|" \
      -e "s|^container: .*|container: $KIMG|" -e "s|^model: $F403\$|model: $F461|" \
      -e "s|^env:\$|env:\n  VLLM_CACHE_ROOT: \"/cache/runtime/vllm-k56-refit\"|" "$K/v3d.yaml" > "$K/k56-refit.yaml"
  [ "$(diff "$K/v3d.yaml" "$K/k56-refit.yaml" | grep -c '^>')" = 5 ] && ! grep -q 'kv_cache_dtype"' "$K/k56-refit.yaml" && grep -qx "model: $F461" "$K/k56-refit.yaml" \
    && grep -qx "container: $KIMG" "$K/k56-refit.yaml" || { echo "k56 recipe edits did not apply"; exit 1; }
  for a in refit-01 refit-02; do mkdir -p "$K/$a"; cp "$K/k56-refit.yaml" "$K/$a/k56-refit.yaml"
    { echo "# k56 arm (#97 refit run1) on $([ $a = refit-01 ] && echo dgx-01 || echo dgx-02) vs shipped v3d: refit MTP dense tensors"
      echo "# ($GPTS pts/pos offline T=0). Own VLLM_CACHE_ROOT + bake (new model path); seed re-keyed in the image."
      echo "name: $a"; echo "recipe: k56-refit.yaml"; echo "base: $K/v3d.yaml"; echo "bake: yes"; echo "hook: $G/k53/hook.sh"; } > "$K/$a/thunderdome.arm"; done
  env RES="$RES/k56-dry" GATE=0 CHAIN_NO_RESTORE=1 TD_ACC_RISE_OK=1 bash "$TD" "$K/refit-01" "$K/refit-02" --dry-run > "$RES/k56-dry-run.txt" 2>&1
  if grep -q REFUSED "$RES/k56-dry-run.txt"; then echo "thunderdome dry-run dropped an arm"; exit 1; fi
  echo "$KIMG" > "$K/IMAGE"; date > "$K/READY"; }
( stage ) > "$RES/k56-stage.txt" 2>&1; [ $? = 0 ] || { FINAL="FAILED: k56 staging ($RES/k56-stage.txt); refit gain $GPTS pts/pos T=0"; exit 1; }
# queue k56 right after this job (user approval 2026-10-07): start.sh blocks on the gpu-lock, which the caller holds
# until this job (and its restore) has exited
( cd "$K" && setsid nohup bash "$K/start.sh" > "$K/start.nohup" 2>&1 < /dev/null 9>&- & echo $! > "$K/start.pid" )
{ echo; echo "== k56 staged: $(cat "$K/IMAGE"), snapshot $F461 on both Sparks, fp8 drafter KV, seed $(cat "$RES/remap-seed.txt")"
  echo "queued: k56/start.sh pid $(cat "$K/start.pid") on the gpu-lock (TD_ACC_RISE_OK=1, gate on PROMOTE)"; } >> "$REP"
FINAL="DONE: gain $GPTS pts/pos T=0, k56 staged and queued (start.sh pid $(cat "$K/start.pid")) ($REP)"
