#!/usr/bin/env bash
# k89-reddit-credibility (2026-10-11, tp-2 #155): two more boots of the reference stack of k84b (3 boots in all, sd
# between boots) and IFEval on my shipped 2x against the reference stack. Measure only; nothing built or shipped.
#   reference  ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-spark-20261007-d071fb3dde0ba979 (digest pinned),
#              PRESET=qwen38-dgx-spark-x2 unchanged (MTP 3 probabilistic drafts), 7c4f1bc1 pinned and offline, the same
#              docker/RoCE settings, API name and JIT cache dirs (~/GEN-AI/backlog/k84b/cache-rN) as k84b's ref-2x-mtp3.
#              Boots ref-2x-mtp3-b2 and -b3, k84b's cells (cred.sh std_cells: idle 120 s, benchy tg512 c1/c8 at
#              T=1.0 top-p 0.95 top-k 20 thinking on and at T=0 thinking off, coding-36 T=0 c1/c8 one pass, GPU power).
#              The report pools k84b's ref-2x-mtp3 boot with these two (pooled/: units, marks and power of both jobs).
#   ifeval     all 541 prompts, lm-eval 0.4.13 local-chat-completions, c16 (both servers allow 16 sequences), seed
#              1234, task ifeval_think_med (k89/tasks): T=1.0 top-p 0.95 top-k 20, thinking on, reasoning_effort
#              medium sent in every request (the reference preset has no server default; my recipe's default is
#              medium), max_gen_toks 32768, --use_cache in the job dir.
#              (a) my 2x v2.0.0 as shipped, run first on the shipped 2x (booted fresh by lib.sh restore when it is not
#              serving); (b) the reference on boot b2, after that boot's cells. evalcmp.py: strict/loose prompt and
#              instruction accuracy, paired discordant counts, exact McNemar p.
#   report     comment.md: reference 3-boot mean and sd next to k86's 3-boot means of my 2x v2.0.0 and 1x v2.2.0,
#              ratio mine/reference per cell, and the IFEval table.
# ETA ~3 h 20 min: IFEval (a) ~40 min, boot b2 ~6 min + cells 25 + IFEval (b) ~70, boot b3 ~6 + cells 25, restore 6.
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; EXIT trap removes the reference containers and restores the shipped 2x.
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k89; K=$HOME/GEN-AI/backlog/k89
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k89-reddit-credibility-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"
IMG=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-spark-20261007-d071fb3dde0ba979
DIG=sha256:af14f1a90e141ad7a1b4951f0fd37d695101ebd5887fcddb499f2f427eff5319
PRESET=qwen38-dgx-spark-x2; REV=7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd; PASSES=1
K84B=$(cat "$BL/k84b/RESULT" 2>/dev/null); K86=$(cat "$BL/k86/RESULT" 2>/dev/null); EV=$G/evals; CK=$BL/k84b
HCA=rocep1s0f1,roceP2p1s0f1; IF=enp1s0f1np1
dargs() { # rank host-ip (identical to k84b, container JIT cache shared with k84b)
  echo "--init --privileged --gpus all --network host --ipc host --shm-size 32g --ulimit memlock=-1 \
 -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $CK/cache-r$1:/cache \
 -e PRESET=$PRESET -e NODE_RANK=$1 -e NNODES=2 -e MASTER_ADDR=$H1 -e PORT=8000 -e MODEL_REVISION=$REV -e HF_HUB_OFFLINE=1 \
 -e NCCL_IB_HCA=$HCA -e NCCL_SOCKET_IFNAME=$IF -e GLOO_SOCKET_IFNAME=$IF -e VLLM_HOST_IP=$2 -e NCCL_CROSS_NIC=1"; }
OPTS0="--served-model-name qwen3.8-flash-next"
printcfg() { timeout -k 30 300 docker run --rm $(dargs 0 $H1) "$IMG" --print-config $OPTS0 "$@" 2>&1; }
rmc() { docker rm -f sparkrun-k89-r0 > /dev/null 2>&1; ssh -n -o ConnectTimeout=10 $H2 "docker rm -f sparkrun-k89-r1" > /dev/null 2>&1; }

checks() {
  local ok=0 h t
  for h in $H1 $H2; do
    t=$(x $h "docker image inspect --format '{{.Architecture}} {{index .RepoDigests 0}}' $IMG 2>/dev/null")
    echo "$t" | grep -q "arm64 .*$DIG" && echo "$(hn $h): reference image present ($DIG)" || { echo "$(hn $h): reference image missing ($t)"; ok=1; }
    x $h "test -d $CK/cache-r$([ $h = $H1 ] && echo 0 || echo 1)" && echo "$(hn $h): k84b JIT cache dir present" || { echo "$(hn $h): k84b JIT cache dir missing"; ok=1; }; done
  printcfg > "$K/print-config.txt"
  t=$(python3 -c "
import json; a = json.load(open('$K/print-config.txt'))['argv']; v = lambda k: a[a.index(k) + 1] if k in a else '-'
s = json.loads(v('--speculative-config')); print(v('--revision'), v('--tensor-parallel-size'), s['num_speculative_tokens'], s['draft_sample_method'])" 2>&1)
  [ "$t" = "$REV 2 3 probabilistic" ] && echo "reference print-config: 7c4f1bc1, TP2, 3 probabilistic drafts" || { echo "reference print-config not as k84b: $t"; ok=1; }
  cmp -s <(python3 -c "import json; print(json.load(open('$K/print-config.txt'))['argv'])") <(python3 -c "import json; print(json.load(open('$K84B/ref-2x-mtp3/print-config.txt'))['argv'])") \
    && echo "argv identical to k84b ref-2x-mtp3" || { echo "argv differs from k84b ref-2x-mtp3"; ok=1; }
  [ -s "$K84B/units.tsv" ] && grep -q '^ref-2x-mtp3' "$K84B/units.tsv" && echo "k84b ref-2x-mtp3 boot to pool: $K84B" || { echo "k84b results missing"; ok=1; }
  [ -s "$K86/boots.csv" ] && echo "k86 results: $K86" || { echo "k86 results missing"; ok=1; }
  [ -s "$K/tasks/ifeval_think_med/ifeval_think_med.yaml" ] && grep -q 'reasoning_effort: medium' "$K/tasks/ifeval_think_med/ifeval_think_med.yaml" && echo "task ifeval_think_med (effort medium per request)" || { echo "task file missing"; ok=1; }
  t=$(cd "$EV" && HF_DATASETS_TRUST_REMOTE_CODE=1 timeout 600 venv/bin/python -c "
import datasets
from lm_eval.tasks import TaskManager
tm = TaskManager(include_path='$K/tasks'); assert 'ifeval_think_med' in tm.all_tasks
print('ifeval', len(datasets.load_dataset('google/IFEval', split='train')))" 2>/dev/null | tail -1)
  [ "$t" = "ifeval 541" ] && echo "lm-eval sees ifeval_think_med; IFEval 541 prompts cached" || { echo "lm-eval/IFEval not ready: '$t'"; ok=1; }
  python3 "$K/evalcmp.py" --selftest > /dev/null && python3 "$CRED/cred.py" --selftest > /dev/null && python3 "$CODING" --selftest > /dev/null || { echo "selftests failed"; ok=1; }
  [ "$(img=$(sed -n 's/^container: *//p' "$BL/k86/recipe-2x-v2.0.0.yaml"); echo ${img##*:})" = "$SHIPPED_TAG_DEFAULT" ] && echo "shipped 2x = $SHIPPED_TAG_DEFAULT (lib.sh restore target)" || { echo "lib.sh shipped tag differs from k86's 2x v2.0.0"; ok=1; }
  mkdir -p "$(dirname "$RES")" && [ -w "$(dirname "$RES")" ] && echo "results dir writable" || { echo "results dir not writable"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  checks; rc=$?
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "plan: IFEval on shipped 2x v2.0.0; ref-2x-mtp3 boot b2 (cells + IFEval), boot b3 (cells); pool with k84b's boot; report vs k86"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
trap 'pw_stop; rmc; kill $G1 $G2 2>/dev/null; restore || { sleep 60; restore; } || FINAL="FAILED: restore -- $FINAL"; st "$FINAL"' EXIT
echo "$RES" > "$K/RESULT"; : > "$K/comment.md"
st "running: preflight"
checks > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
cp -r "$K/run.sh" "$K/evalcmp.py" "$K/tasks" "$K/print-config.txt" "$RES/"; mkdir -p "$RES/evals"
attempt() { echo "$(TZ=Europe/Bucharest date '+%F %T') $*" >> "$RES/attempts.txt"; }
lmeval() { # arm
  local o=$RES/evals/$1 rc; mkdir -p "$o" "$K/cache/$1"
  ( cd "$EV" && HF_DATASETS_TRUST_REMOTE_CODE=1 timeout -k 120 14400 venv/bin/lm_eval run --model local-chat-completions \
      --model_args "model=$MODEL,base_url=http://$H1:8000/v1/chat/completions,num_concurrent=16,max_retries=5,timeout=7200,tokenized_requests=False" \
      --tasks ifeval_think_med --include_path "$K/tasks" --apply_chat_template --output_path "$o" --log_samples \
      --seed 1234 --use_cache "$K/cache/$1/lm" > "$o/lm_eval.log" 2>&1 < /dev/null 9>&- ); rc=$?
  log "ifeval $1 exit=$rc"; echo "rc=$rc" > "$o/status.txt"; attempt "IFEval $1: exit $rc"; }
rboot() { # unit -> 0 when rank 0 answers /health and the pong
  local u=$1 s=$(date +%s) a; mkdir -p "$RES/$u"; stop_all; rmc
  printcfg > "$RES/$u/print-config.txt"
  ssh -n $H2 "docker run -d --name sparkrun-k89-r1 $(dargs 1 $H2) $IMG $OPTS0" > "$RES/$u/run-r1.txt" 2>&1
  docker run -d --name sparkrun-k89-r0 $(dargs 0 $H1) "$IMG" $OPTS0 > "$RES/$u/run-r0.txt" 2>&1
  while [ "$(health $H1)" != 200 ]; do a=$(( $(date +%s) - s ))
    if [ $a -gt 2700 ] || { [ $a -gt 120 ] && { [ -z "$(docker ps -q --filter name=sparkrun-k89-r0)" ] || [ -z "$(x $H2 "docker ps -q --filter name=sparkrun-k89-r1")" ]; }; }; then
      docker logs sparkrun-k89-r0 > "$RES/$u/serve-r0.log" 2>&1; x $H2 "docker logs sparkrun-k89-r1" > "$RES/$u/serve-r1.log" 2>&1
      log "$u: not up after $a s"; return 1; fi
    sleep 20; done
  echo "boot_s $(( $(date +%s) - s ))" > "$RES/$u/boot.txt"
  docker logs sparkrun-k89-r0 > "$RES/$u/serve-r0.log" 2>&1; x $H2 "docker logs sparkrun-k89-r1" > "$RES/$u/serve-r1.log" 2>&1
  a=$(pong $H1); echo "pong $a" >> "$RES/$u/boot.txt"; echo "$a" | grep -qi pong; }
pw_start

st "running: IFEval (a) on the shipped 2x v2.0.0 (~40 min)"
restore || { FINAL="FAILED: shipped 2x not serving before IFEval (a)"; exit 1; }
docker ps --format '{{.Names}} {{.Image}}' > "$RES/evals/shipped-containers.txt"
lmeval shipped-2x-v2.0.0

nb=0
for b in 2 3; do
  u=ref-2x-mtp3-b$b; st "running: reference boot $b"
  if rboot $u; then nb=$((nb + 1)); attempt "$u: booted ($(tr '\n' ' ' < "$RES/$u/boot.txt"))"
    unit $u ref-2x-mtp3 $b dgx01+dgx02 "$IMG $PRESET"; st "running: $u cells (~25 min)"; std_cells $u $H1 $PASSES nodefault
    [ -s "$RES/evals/reference/status.txt" ] || { st "running: IFEval (b) on the reference, boot $b (~70 min)"; lmeval reference; }
  else attempt "$u: did not boot; $(grep -hE -m3 'Error|error:|out of memory|Traceback' "$RES/$u"/serve-r*.log 2>/dev/null | cut -c1-200 | tr '\n' ' ')"; fi
  rmc; done
stop_all; pw_stop

st "running: report"
P=$RES/pooled; mkdir -p "$P"
grep -P '^ref-2x-mtp3\t' "$K84B/units.tsv" > "$P/units.tsv"; cat "$RES/units.tsv" >> "$P/units.tsv" 2>/dev/null
grep -P '^ref-2x-mtp3\t' "$K84B/marks.tsv" > "$P/marks.tsv"; cat "$RES/marks.tsv" >> "$P/marks.tsv" 2>/dev/null
for h in dgx01 dgx02; do cat "$K84B/power-$h.csv" "$RES/power-$h.csv" > "$P/power-$h.csv" 2>/dev/null; done
ln -sfn "$K84B/ref-2x-mtp3" "$P/ref-2x-mtp3"; for u in $(cut -f1 "$RES/units.tsv" 2>/dev/null); do ln -sfn "$RES/$u" "$P/$u"; done
python3 "$CRED/cred.py" report "$P" --vs "$K86" > "$RES/report.log" 2>&1 || log "report: no rows"
python3 - "$P/boots.csv" "$K86/boots.csv" > "$RES/ratio.md" 2>&1 <<'EOF'
import csv, sys
ref = {(r["cell"], r["metric"]): r for r in csv.DictReader(open(sys.argv[1])) if r["setup"] == "ref-2x-mtp3"}
k86 = {(r["setup"], r["hosts"], r["cell"], r["metric"]): r for r in csv.DictReader(open(sys.argv[2]))}
f = lambda v, d=1: "-" if v in ("", None) else f"{float(v):.{d}f}"
print("| Cell | Reference 2x, mean (sd), boots | My 2x v2.0.0, mean (sd), 3 boots | 2x / reference | My 1x v2.2.0, mean (sd), 6 boots | 1x / reference |")
print("|---|---|---|---|---|---|")
name = {"tgdef-c1": "tg512 c1, T=1.0 thinking on", "tgdef-c8": "tg512 c8, T=1.0 thinking on", "tgt0-c1": "tg512 c1, T=0 thinking off",
        "tgt0-c8": "tg512 c8, T=0 thinking off", "coding-t0-nothink-c1": "coding-36 c1, T=0", "coding-t0-nothink-c8": "coding-36 c8, T=0"}
for c in name:
    m = "tg_tok_s" if c.startswith("tg") else "agg_tok_s"
    r = ref.get((c, m)); a = k86.get(("2x-v2.0.0", "dgx01+dgx02", c, m)); b = k86.get(("1x-v2.2.0", "pooled", c, m))
    if not r:
        continue
    rv = float(r["mean"]); cell = lambda x: f"{f(x['mean'])} ({f(x['sd_between_boots'], 2)})" if x else "-"
    print(f"| {name[c]} | {f(r['mean'])} ({f(r['sd_between_boots'], 2)}), {r['units']} | {cell(a)} | "
          f"{f(float(a['mean']) / rv, 2) if a else '-'} | {cell(b)} | {f(float(b['mean']) / rv, 2) if b else '-'} |")
for key, lab in (("idle", "idle GPU W"),):
    r = ref.get(("idle", "idle")); a = k86.get(("2x-v2.0.0", "dgx01+dgx02", "idle", "idle"))
    if r and a:
        print(f"| GPU power idle, W (both Sparks) | {f(r['watts'])} | {f(a['watts'])} | | | |")
EOF
python3 "$K/evalcmp.py" "$RES/evals/shipped-2x-v2.0.0" mine-2x "$RES/evals/reference" reference "$RES/evals.json" > "$RES/evals.txt" 2>&1 || log "evalcmp: no paired samples"
{ echo "Reference stack against my builds, same Sparks, same harnesses (k89-reddit-credibility, with k84b and k86)."
  echo; echo "Reference: the checkpoint publisher's vLLM image for DGX Spark, \`$IMG\` (\`$DIG\`), with its two-Spark preset \`$PRESET\` unchanged (TP2 over RoCE, MTP 3 probabilistic drafts, 16 sequences, utilization 0.75), on \`local-inference-lab/Qwen3.8-Flash-Next-NVFP4\` @ \`7c4f1bc1\`. I set only the checkpoint revision, the RoCE interface settings, the API name and the container names. Three boots: one from k84b and two from this job ($nb of 2 came up)."
  echo; echo "Mine: 2x v2.0.0 and 1x v2.2.0 from k86 (3 boots of each on each Spark set). Cells: llama-benchy pp2048/tg512 task mode, 3 runs per boot; the 36-prompt coding probe, aggregate output tok/s (one pass per reference boot, 3 passes per boot in k86). Values are tok/s, mean of the per-boot means, with the standard deviation between boots in parentheses."
  echo; cat "$RES/ratio.md"
  echo; echo "IFEval, all 541 prompts, lm-eval 0.4.13, 16 requests at a time, T=1.0 top-p 0.95 top-k 20, thinking on, reasoning effort medium sent in every request (the reference preset has no server default). Paired per prompt; McNemar p is exact and two-sided; instruction-level accuracy is pooled over all instructions, so it has no paired test."
  echo; echo '```'; cat "$RES/evals.txt"; echo '```'
  echo; echo "Full cell tables with GPU power are in \`pooled/cred.txt\`."
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
cp "$K/comment.md" "$RES/comment.md"
FINAL="DONE: k89 reference $nb/2 extra boots, IFEval rows $(grep -c '%' "$RES/evals.txt") ($RES/ratio.md, $RES/evals.txt)"
exit 0
