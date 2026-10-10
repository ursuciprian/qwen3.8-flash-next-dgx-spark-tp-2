#!/usr/bin/env bash
# k85-quality-kld-evals (2026-10-10, tp-2 #155; trimmed the same evening, user "address all"): quality of the GDN-MSE
# requant against the unmodified 7c4f1bc1 GDN weights, on one engine. Measure only; nothing built or shipped.
# Why one engine: GDN-MSE is 7c4f1bc1 with the 108 GDN projection tensors in NVFP4 instead of MXFP8 (plus the retrained
# drafter); every other tensor is byte-identical. The shipped 1x recipe v2.2.0 (GDN-MSE @ 03f4a057, image tp1-d3-hf,
# vLLM 5dad364d) loads 7c4f1bc1's MXFP8 GDN tensors (shard 35) next to the NVFP4 ones, and
# VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS picks per call: 1 = 7c4f1bc1's MXFP8 tensors on every call (the stock 7c4f1bc1 GDN
# path), 0 = the NVFP4 GDN-MSE tensors on every call, 41 = the shipped dispatch.
# Reference: 7c4f1bc1, not BF16. The BF16 base (336 GB, dgx-02 ~/models) does not fit in the 2 x 128 GB of the pair.
#   evals    IFEval only (541 prompts): dgx-01 = v2.2.0 with MIN 1, dgx-02 = v2.2.0 unchanged (shipped), both at once,
#            MTP on. lm-eval 0.4.13 local-chat-completions, c8, seed 1234, ~/GEN-AI/evals/tasks/ifeval_think (T=1.0 top-p
#            0.95 top-k 20, thinking on, server default effort medium, max_gen_toks 32768), --use_cache (job dir).
#            evalcmp.py: strict/loose prompt accuracy per arm, paired discordant counts, exact McNemar p.
#   kld      one round after the evals, capped (boot <= 20 min, else skipped): MTP off, --max-logprobs 64; MIN 1 on
#            dgx-01 scored twice (noise floor: same server twice), MIN 0 on dgx-02 scored once. Corpus corpus.jsonl:
#            44 held-out sequences of the model's own outputs (agentic 14, chat 5, code 10, math 5, tools 10; 32,351
#            scored tokens). kld.py: top-64 KLD, top-1 agreement, PPL. Prompt scoring is prefill, where the shipped
#            dispatch runs MXFP8, so MIN 0 is what shows the NVFP4 GDN error.
#   no MMLU-Pro / GSM8K: trimmed. The 2026-09-28 MMLU-Pro run was stopped by me at 770/2000 after 3 h 10 min (~4000
#            thinking tokens per question at effort xhigh) and left no score (no --use_cache).
# ETA ~2 h: boot 10 min, IFEval ~75 min (one Spark per arm, at once), kld <=30 min, restore 6 min.
# Output: $RES/evals/<arm>/ifeval_think/..., evals.json, evals.txt; kld/*.json + logs, kld.txt (raw top-64 files in
# ~/GEN-AI/backlog/k85/kld-raw/<results dir>, not published); per boot folder recipe.yaml + serve log; comment.md + RESULT.
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; EXIT trap restores the shipped 2x (lib.sh).
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k85; K=$HOME/GEN-AI/backlog/k85
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k85-quality-kld-evals-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"
EV=$G/evals; BASE=$K/recipe-1x-v2.2.0.yaml; CORPUS=$K/corpus.jsonl; PYK="python3 $K/kld.py"
MSE=$HOME/.cache/huggingface/hub/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE/snapshots/03f4a0570496bbe741189f4a2a7d611679e85406

variant() { # name min-tokens spec(keep|drop) -> recipe path (MIN_TOKENS edited; drop = no MTP and --max-logprobs 64)
  local o=$K/rec/$1.yaml; mkdir -p "$K/rec"
  python3 - "$BASE" "$o" "$1" "$2" "$3" <<'EOF'
import sys
src, out, name, mt, spec = sys.argv[1:]
t = open(src).read()
for a, b in ((f"name: qwen3.8-flash-next-1x-dgx-spark\n", f"name: qwen3.8-flash-next-1x-dgx-spark-{name}\n"),
             ('VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS: "41"', f'VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS: "{mt}"')):
    assert t.count(a) == 1, a; t = t.replace(a, b)
if spec == "drop":
    lines = t.split("\n"); i = [n for n, l in enumerate(lines) if "--speculative-config" in l]
    assert len(i) == 1; lines[i[0]] = "    --max-logprobs 64 \\"; t = "\n".join(lines)
open(out, "w").write(t)
EOF
  echo "$o"; }

checks() {
  local ok=0 t h f
  python3 "$K/kld.py" --selftest > /dev/null && python3 "$K/evalcmp.py" --selftest > /dev/null || { echo "kld.py/evalcmp.py selftest failed"; ok=1; }
  [ "$(sed -n 1p "$BASE")" = "# Release: v2.2.0" ] && grep -q '^container: ghcr.io/ursuciprian/spark-vllm-b12x:tp1-d3-hf-20261010-21e0b201-5dad364d-warm' "$BASE" \
    && echo "base recipe: 1x v2.2.0 ($BASE)" || { echo "base recipe is not 1x v2.2.0"; ok=1; }
  for v in "kld-mxfp8 1 drop" "kld-nvfp4 0 drop" "eval-7c4 1 keep"; do set -- $v
    f=$(variant $1 $2 $3) && grep -q "MIN_TOKENS: \"$2\"" "$f" && echo "variant $1: MIN_TOKENS=$2, spec lines $(grep -c speculative-config "$f"), max-logprobs $(grep -c max-logprobs "$f")" \
      || { echo "variant $1 failed"; ok=1; }; done
  [ "$(grep -c . "$CORPUS" 2>/dev/null)" = 44 ] && echo "corpus: 44 sequences, sha256 $(sha256sum "$CORPUS" | cut -c1-16)" || { echo "corpus $CORPUS missing or not 44 lines"; ok=1; }
  for h in $H1 $H2; do
    t=$(x $h "ls $MSE/model-000*-of-00036.safetensors 2>/dev/null | wc -l; ls $TOK/model-00035-of-00036.safetensors 2>/dev/null | wc -l; docker image inspect $(sed -n 's/^container: *//p' "$BASE") > /dev/null 2>&1 && echo img")
    [ "$(echo $t)" = "36 1 img" ] && echo "$(hn $h): GDN-MSE 03f4a057 36 shards, 7c4f1bc1 shard 35, image present" || { echo "$(hn $h): missing pieces ($t)"; ok=1; }; done
  [ -x "$EV/venv/bin/lm_eval" ] && [ -s "$EV/tasks/ifeval_think/ifeval_think.yaml" ] && echo "lm-eval $($EV/venv/bin/python -c 'import lm_eval;print(lm_eval.__version__)'), task ifeval_think" \
    || { echo "lm-eval venv or task missing"; ok=1; }
  t=$(cd "$EV" && HF_DATASETS_TRUST_REMOTE_CODE=1 timeout 600 venv/bin/python -c "
import datasets
from lm_eval.tasks.ifeval import instructions_util
print('datasets', len(datasets.load_dataset('google/IFEval', split='train')))" 2>/dev/null | tail -1)
  [ "$t" = "datasets 541" ] && echo "IFEval 541 prompts cached, nltk data loaded" || { echo "IFEval not loadable: '$t'"; ok=1; }
  mkdir -p "$(dirname "$RES")" && [ -w "$(dirname "$RES")" ] && echo "results dir writable: $(dirname "$RES")" || { echo "results dir not writable"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  checks; rc=$?
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "plan: evals eval-7c4 (MIN 1) @dgx01 + v2.2.0 @dgx02: ifeval_think 541, c8; then kld mxfp8 @dgx01 (2 passes) + nvfp4 @dgx02 (1 pass), boot cap 20 min"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; : > "$K/comment.md"
st "running: preflight"
checks > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
cp "$K/run.sh" "$K/kld.py" "$K/evalcmp.py" "$BASE" "$K"/rec/*.yaml "$RES/"; sha256sum "$CORPUS" > "$RES/corpus.sha256"; cp "$CORPUS" "$RES/"
KR=$K/kld-raw/$(basename "$RES"); mkdir -p "$RES/kld" "$RES/evals" "$KR"   # raw top-64 files stay on dgx-01 (not published)
KM=$(variant kld-mxfp8 1 drop); KN=$(variant kld-nvfp4 0 drop); E7=$(variant eval-7c4 1 keep)

st "running: IFEval (7c4f1bc1 GDN on dgx-01, GDN-MSE shipped on dgx-02, ~75 min)"; stop_all
lmeval() { # host arm
  local h=$1 a=$2 t=ifeval_think o rc; o=$RES/evals/$a/$t; mkdir -p "$o" "$K/cache/$a-$t"
  ( cd "$EV" && HF_DATASETS_TRUST_REMOTE_CODE=1 timeout -k 120 14400 venv/bin/lm_eval run --model local-chat-completions \
      --model_args "model=$MODEL,base_url=http://$h:8000/v1/chat/completions,num_concurrent=8,max_retries=5,timeout=7200,tokenized_requests=False" \
      --tasks $t --include_path "$EV/tasks" --apply_chat_template --output_path "$o" --log_samples \
      --seed 1234 --use_cache "$K/cache/$a-$t/lm" > "$o/lm_eval.log" 2>&1 < /dev/null 9>&- ); rc=$?
  log "evals $a $t exit=$rc"; echo "$t rc=$rc" >> "$RES/evals/$a/status.txt"; }
if pair eval-7c4-dgx01 "$E7" eval-mse-dgx02 "$BASE" 2400; then
  lmeval $H1 7c4f1bc1 & a=$!; lmeval $H2 gdn-mse & b=$!; CH="$a $b"; wait $a; wait $b; CH=
  python3 "$K/evalcmp.py" "$RES/evals/7c4f1bc1" 7c4f1bc1 "$RES/evals/gdn-mse" GDN-MSE "$RES/evals.json" > "$RES/evals.txt" 2>&1 \
    || log "evalcmp: no paired samples ($RES/evals.txt)"
else log "evals: boot failed (serve logs in eval-*)"; echo "IFEval not run: boot failed" > "$RES/evals.txt"; fi

st "running: kld round (boot cap 20 min)"; stop_all
scorep() { # host tag passes...
  local h=$1 t=$2 p; shift 2
  for p in "$@"; do timeout -k 30 900 $PYK score "http://$h:8000" "$CORPUS" "$KR/$t-$p.jsonl.gz" > "$RES/kld/$t-$p.log" 2>&1 < /dev/null 9>&-
    log "kld score $t-$p exit=$?"; done; }
if pair kld-mxfp8-dgx01 "$KM" kld-nvfp4-dgx02 "$KN" 1200; then
  scorep $H1 mxfp8-dgx01 a b & a=$!; scorep $H2 nvfp4-dgx02 a & b=$!; CH="$a $b"; wait $a; wait $b; CH=
else log "kld: boot over 20 min or failed, skipped (serve logs in kld-*)"; fi
{ echo "Reference first. mxfp8 = 7c4f1bc1's GDN tensors (MIN 1), nvfp4 = GDN-MSE's GDN tensors (MIN 0); same image, same checkpoint folder, MTP off, top-64 prompt logprobs on 44 sequences."
  for c in "mxfp8-dgx01-a nvfp4-dgx02-a GDN-MSE vs 7c4f1bc1" "mxfp8-dgx01-a mxfp8-dgx01-b floor: 7c4f1bc1, same server twice"; do
    set -- $c; r=$KR/$1.jsonl.gz; t=$KR/$2.jsonl.gz; shift 2
    if [ -s "$r" ] && [ -s "$t" ]; then python3 -c "
import json, sys; sys.path.insert(0, '$K'); import kld
r = kld.compare('$r', '$t', '$RES/kld/' + '$(basename "$r" .jsonl.gz)__$(basename "$t" .jsonl.gz).json')
print(kld.line('$*', r))
print('    by category: ' + ', '.join(f\"{c} KLD {v['kld_mean']:.5f} top-1 {100 * v['top1_agreement']:.2f}%\" for c, v in r['by_category'].items()))" 2>&1
    else echo "$*: not measured (missing $(basename "$r") or $(basename "$t"))"; fi; done; } > "$RES/kld.txt"
stop_all

st "running: report"
nk=$(grep -c 'KLD mean' "$RES/kld.txt"); ne=$(grep -c '%' "$RES/evals.txt")
{ echo "Quality of the GDN-MSE requant (k85-quality-kld-evals), one engine for both arms: the shipped 1x recipe v2.2.0 (vLLM 5dad364d, b12x 21e0b201) on \`ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE\` @ \`03f4a057\`, where \`VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS\` selects which copy of the GDN projection weights runs: \`1\` = the MXFP8 tensors of \`7c4f1bc1\` (shard 35, byte-identical to the unmodified checkpoint) on every call, \`0\` = the NVFP4 GDN-MSE tensors on every call, \`41\` = the shipped dispatch. Every other tensor is byte-identical between the two checkpoints, so the arms differ only in the GDN weights."
  echo; echo "Reference: \`7c4f1bc1\`, not BF16. The BF16 base (336 GB) is on disk but does not fit in the 2 x 128 GB of the pair, so it cannot be served as a reference."
  echo; echo "IFEval, all 541 prompts, at the served sampling (T=1.0, top-p 0.95, top-k 20, thinking on, server default effort medium), lm-eval 0.4.13, c8, one Spark per arm at the same time: \`7c4f1bc1\` = v2.2.0 with \`MIN_TOKENS=1\`, \`GDN-MSE\` = v2.2.0 unchanged. Paired per prompt; McNemar p is exact and two-sided."
  echo; echo '```'; cat "$RES/evals.txt"; echo '```'
  echo; echo "KLD and top-1 agreement, one round: 44 held-out sequences of the model's own outputs (agentic 14, chat 5, code 10, math 5, tools 10; 32,351 scored tokens), teacher-forced at the output positions, top-64 prompt logprobs, MTP off. KLD is over the reference's top 64 plus a rest bucket. Prompt scoring is prefill, where the shipped dispatch runs the MXFP8 copy, so the \`0\` arm shows the NVFP4 GDN error at every position; in the shipped recipe it only reaches decode steps (below 41 rows). Noise floor: the reference server scored twice."
  echo; echo '```'; cat "$RES/kld.txt"; echo '```'
  echo; echo "I trimmed this round to IFEval and one KLD round. I did not run MMLU-Pro: my 2000-question run on 2026-09-28 was stopped at 770/2000 after 3 h 10 min (about 4000 thinking tokens per question at effort xhigh) and left no score, because lm-eval writes samples only at the end and that run had no \`--use_cache\`."
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
cp "$K/comment.md" "$RES/comment.md"
FINAL="DONE: k85 IFEval $ne rows, $nk KLD comparisons ($RES/evals.txt, $RES/kld.txt)"
exit 0
