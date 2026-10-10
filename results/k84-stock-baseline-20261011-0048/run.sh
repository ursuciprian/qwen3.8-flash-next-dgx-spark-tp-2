#!/usr/bin/env bash
# k84-stock-baseline (2026-10-10, tp-2 #155): stock vLLM on the unmodified checkpoint, same Sparks, same
# harnesses as k76. Measure only; nothing built or shipped.
#   image    official vllm/vllm-openai:v0.31.0 (2026-10-04, first release line with Qwen3.8-Flash-Next support, which
#            upstream added on 2026-08-31); fallback the 2026-10-10 nightly (commit 7d0b4e57) when the release does
#            not boot. Both pulled on both Sparks under the lock (~10 GB each); arch check without a GPU first.
#   model    local-inference-lab/Qwen3.8-Flash-Next-NVFP4 @ 7c4f1bc1 (already in the HF cache on both Sparks)
#   recipe   stock.yaml.in: upstream recipe flags + GB10 memory settings of the 2x recipe (see its header)
#   setup    trimmed 2026-10-10 evening (user "address all"): stock-2x-mtp3 only (TP=2 on both Sparks, MTP 3 drafts,
#            the upstream setting); MTP off and the single-Spark attempt dropped. If no image boots, the attempts
#            and their errors are the result.
#   cells    cred.sh std_cells, the k86 cell set: 120 s idle; llama-benchy pp2048/tg512 task mode, 3 runs, c1 and c8, at
#            T=1.0 top-p 0.95 top-k 20 thinking on and at T=0 thinking off; coding-36 T=0 thinking off at c1 and c8,
#            ONE pass (k86 used 3; same metric: aggregate output tok/s per pass). GPU power (nvidia-smi, 1 Hz) on both
#            Sparks throughout. cred.py --vs the k86 results puts stock and shipped side by side, cell for cell.
# ETA ~1 h 15 min: checks 5 min (images already on both Sparks), boot <=45 min (cold compile, upstream image), cells ~25 min, restore 6 min.
# Output: $RES/<unit>/ benchy + coding json/log, recipe.yaml, serve log, image json; units.tsv, marks.tsv,
# power-dgx0N.csv, images.txt, attempts.txt, cells.csv, boots.csv, cred.txt; comment.md here + RESULT.
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; EXIT trap stops the power loggers and restores the shipped 2x (lib.sh).
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k84; K=$HOME/GEN-AI/backlog/k84
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k84-stock-baseline-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$K/cred.sh"
IMGS="vllm/vllm-openai:v0.31.0 vllm/vllm-openai:nightly-7d0b4e57aac4c4323225b0eee9ae960ce75ffffe"
ARCHS=$(python3 -c "import json; print(' '.join(json.load(open('$TOK/config.json'))['architectures']))" 2>/dev/null); PASSES=1; K86=$(cat "$BL/k86/RESULT" 2>/dev/null)
REGS=$HOME/.cache/sparkrun/registries; N2=qwen3.8-flash-next-2x-dgx-spark

mkrec() { # name image 2x|1x on|off -> prints the recipe path
  local o=$K/rec/$1.yaml; mkdir -p "$K/rec"
  NAME=$1 IMAGE=$2 TOPO=$3 SPEC=$4 python3 - "$K/stock.yaml.in" "$o" <<'EOF'
import os, sys
t = open(sys.argv[1]).read(); e = os.environ; two = e["TOPO"] == "2x"
v = {"NAME": e["NAME"], "IMAGE": e["IMAGE"], "TP": "2" if two else "1", "UTIL": "0.80" if two else "0.88",
     "LEN": "262144" if two else "65536", "SEQS": "16" if two else "8",
     "TOPO": "cluster_only: true\nmin_nodes: 2\nmax_nodes: 2" if two else "solo_only: true\nmin_nodes: 1\nmax_nodes: 1",
     "SPEC": "\\\n    --speculative-config '{{\"method\":\"mtp\",\"num_speculative_tokens\":3}}'" if e["SPEC"] == "on" else ""}
for k, s in v.items():
    t = t.replace(f"@{k}@", s)
assert not any(f"@{k}@" in t for k in v)
open(sys.argv[2], "w").write(t)
EOF
  echo "$o"; }
# 2026-10-10 fix: the first k84 run tested only the alias Qwen3_8FlashNextForConditionalGeneration and failed although
# both images register Qwen4ExpForConditionalGeneration, the first entry of the checkpoint's architectures (vLLM takes
# the first registered one). Now any listed architecture counts, read from the image files without importing vllm.
archcheck() { # image -> "ARCHCHECK <vllm version> <first registered arch|NONE> <modelopt_mixed mentions>"
  timeout -k 30 300 docker run --rm --entrypoint sh "$1" -c 'p=$(python3 -c "import importlib.util as u; print(u.find_spec(\"vllm\").submodule_search_locations[0])"); v=$(python3 -c "import importlib.metadata as m; print(m.version(\"vllm\"))"); q=$(grep -c modelopt_mixed $p/model_executor/layers/quantization/__init__.py); for a in '"$ARCHS"'; do grep -q "\"$a\"" $p/model_executor/models/registry.py && { echo "ARCHCHECK $v $a $q"; exit 0; }; done; echo "ARCHCHECK $v NONE $q"' 2>/dev/null | grep ARCHCHECK; }
archok() { set -- $1; [ "${3:-NONE}" != NONE ] && [ "${4:-0}" -gt 0 ]; }
slug() { local t=${1##*:}; echo "${t:0:16}"; }
recf() { grep -rlx --include='*.yaml' "name: $2" "$(readlink -f "$REGS/$1")/recipes" 2>/dev/null | head -1; }

checks() {
  local ok=0 t h i f
  python3 "$K/cred.py" --selftest > /dev/null || { echo "cred.py selftest failed"; ok=1; }
  python3 "$CODING" --selftest || { echo "coding.py selftest failed"; ok=1; }
  for t in sparkrun uvx python3 curl docker setsid flock; do command -v $t > /dev/null || { echo "missing $t"; ok=1; }; done
  [ -s "$TOK/tokenizer.json" ] && echo "tokenizer $TOK" || { echo "tokenizer missing"; ok=1; }
  for h in $H1 $H2; do
    t=$(x $h "ls $TOK/model-000*-of-00036.safetensors 2>/dev/null | wc -l; df -BG --output=avail / | tail -1 | tr -dc 0-9")
    set -- $t; [ "${1:-0}" = 36 ] && echo "$(hn $h): 7c4f1bc1 36/36 shards" || { echo "$(hn $h): 7c4f1bc1 shards ${1:-0}/36"; ok=1; }
    [ "${2:-0}" -ge 40 ] && echo "$(hn $h): ${2} GB free" || { echo "$(hn $h): only ${2:-?} GB free (need 40)"; ok=1; }
    x $h "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits" | grep -qE '^[0-9.]+' && echo "$(hn $h): power.draw readable" \
      || { echo "$(hn $h): nvidia-smi power.draw unreadable"; ok=1; }; done
  for i in $IMGS; do
    t=$(curl -s -m 30 "https://hub.docker.com/v2/repositories/${i%%:*}/tags/${i##*:}" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(" ".join("arm64:"+x["digest"] for x in d.get("images",[]) if x.get("architecture")=="arm64"))' 2>/dev/null)
    [ -n "$t" ] && echo "image $i on Docker Hub: $t" || { echo "image $i: no arm64 manifest on Docker Hub"; ok=1; }; done
  [ -n "$ARCHS" ] && echo "checkpoint architectures: $ARCHS" || { echo "cannot read architectures from $TOK/config.json"; ok=1; }
  for i in $IMGS; do docker image inspect "$i" > /dev/null 2>&1 || { echo "image $i not pulled yet (the job pulls it)"; continue; }
    t=$(archcheck "$i"); archok "$t" && echo "image $i: $t (registered, modelopt_mixed present)" || echo "image $i: $t (not usable)"; done
  for s in "2x on"; do set -- $s
    f=$(mkrec k84-dry-$1-$2 "${IMGS%% *}" $1 $2) && grep -q 'served-model-name qwen3.8-flash-next' "$f" \
      && echo "recipe $1 mtp $2: $f ($(grep -c speculative-config "$f") spec lines)" || { echo "recipe $1 $2 not generated"; ok=1; }; done
  f=$(recf qwen38-flashnext $N2)
  [ -s "$f" ] && [ "$(sed -n 's/^container: *//p' "$f" | sed 's|.*:||')" = "$SHIPPED_TAG_DEFAULT" ] && echo "registry 2x = $SHIPPED_TAG_DEFAULT (restore target)" \
    || { echo "registry 2x tag differs from lib.sh SHIPPED_TAG_DEFAULT $SHIPPED_TAG_DEFAULT"; ok=1; }
  t=$(cd "$R" && timeout 300 uvx --from "$BENCHY_SRC" llama-benchy --version 2>/dev/null | tail -1)
  [ -n "$t" ] && echo "benchy: $t ($BENCHY_SRC @ $(git -C "$BENCHY_SRC" rev-parse --short HEAD))" || { echo "llama-benchy fork not runnable"; ok=1; }
  mkdir -p "$(dirname "$RES")" && [ -w "$(dirname "$RES")" ] && echo "results dir writable: $(dirname "$RES")" || { echo "results dir not writable"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  checks; rc=$?
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "plan: pull $IMGS on both Sparks if missing; arch check ($ARCHS) without GPU; setup stock-2x-mtp3 only"
  echo "cells: idle 120 s; benchy tgdef-c1/c8 ($SAMP), tgt0-c1/c8 (T=0, $NOTHINK); coding t0-nothink c1/c8 x $PASSES pass"
  [ -s "$K86/boots.csv" ] && echo "compare with k86: $K86" || { echo "k86 results missing ($K86)"; rc=1; }
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
trap 'pw_stop; kill $G1 $G2 2>/dev/null; restore || { sleep 60; restore; } || FINAL="FAILED: restore -- $FINAL"; st "$FINAL"' EXIT
echo "$RES" > "$K/RESULT"; : > "$K/comment.md"
st "running: preflight"
checks > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
cp "$K/run.sh" "$K/cred.sh" "$K/cred.py" "$K/stock.yaml.in" "$CODING" "$RES/"

st "running: pull stock images on both Sparks"
OK_IMGS=
for i in $IMGS; do
  ok=1; for h in $H1 $H2; do x $h "docker image inspect $i > /dev/null 2>&1 || timeout -k 30 2400 docker pull -q $i" >> "$RES/pull.log" 2>&1 || { ok=0; log "pull $i on $(hn $h) failed"; }; done
  [ $ok = 1 ] || continue
  for h in $H1 $H2; do echo "$(hn $h) $i $(x $h "docker image inspect --format '{{index .RepoDigests 0}} {{.Id}}' $i")" >> "$RES/images.txt"; done
  t=$(archcheck "$i"); echo "$i: $t" >> "$RES/images.txt"; log "$i: $t"
  archok "$t" && OK_IMGS="$OK_IMGS $i"; done
[ -n "$OK_IMGS" ] || { FINAL="FAILED: no stock image registers any of $ARCHS with modelopt_mixed ($RES/images.txt)"; exit 1; }

pw_start; stop_all
attempt() { echo "$(TZ=Europe/Bucharest date '+%F %T') $*" >> "$RES/attempts.txt"; }
why() { grep -hE -m6 'Error|error:|OutOfMemory|CUDA out of memory|not supported|unrecognized|No module' "$RES/$1"/serve-*.log 2>/dev/null | cut -c1-300; }
BOOTED=
for spec in on; do
  s=stock-2x-mtp$([ $spec = on ] && echo 3 || echo off); ok=0
  for i in $BOOTED $OK_IMGS; do
    [ $ok = 1 ] && break; u=$s-$(slug $i); [ -d "$RES/$u" ] && continue
    f=$(mkrec $u "$i" 2x $spec); st "running: boot $u"
    if boot $u "$f" 2700 2x; then ok=1; BOOTED=$i; attempt "$u: booted ($(cat "$RES/$u/boot.txt" | tr '\n' ' '))"
    else attempt "$u: did not boot; $(why $u | head -3 | tr '\n' ' ')"; stop_all; fi; done
  [ $ok = 1 ] || continue
  unit $u $s 1 dgx01+dgx02 "$f"; st "running: $u cells (~1 h)"; std_cells $u $H1 $PASSES nodefault; stop_all; done

stop_all; pw_stop

st "running: report"
python3 "$K/cred.py" report "$RES" ${K86:+--vs "$K86"} > "$RES/report.log" 2>&1 || log "report: no measured cells ($RES/report.log)"
n=$(grep -c . "$RES/units.tsv" 2>/dev/null || echo 0)
{ echo "Correction to the earlier k84 failure on this issue: it came from my image check, which looked up only the alias \`Qwen3_8FlashNextForConditionalGeneration\`. Both stock images register \`Qwen4ExpForConditionalGeneration\`, the first architecture in the checkpoint's config, and vLLM uses that one. This rerun checks every listed architecture."
  echo; echo "Stock baseline (k84-stock-baseline): upstream vLLM on the unmodified checkpoint \`local-inference-lab/Qwen3.8-Flash-Next-NVFP4\` @ \`7c4f1bc1\`, on the same two Sparks, with the harnesses of the capability matrix (k76)."
  echo; echo "Images (digest and vLLM version per Spark):"; echo '```'; cat "$RES/images.txt"; echo '```'
  echo; echo "Boot attempts:"; echo '```'; cat "$RES/attempts.txt" 2>/dev/null; echo '```'
  echo; echo "Recipe: upstream recipe flags (prefix caching, fp8 KV and indexer KV, no flashinfer autotune, qwen3 reasoning parser) with the memory settings of my 2x recipe; MTP on means 3 draft tokens, the upstream setting. The only addition is the server default reasoning effort \"medium\", which my recipes also set. The generated recipes are in each setup folder (\`recipe.yaml\`)."
  echo; echo "Setup: TP=2 on both Sparks with MTP on (3 draft tokens). I dropped MTP off and the single-Spark attempt to keep the job short."
  echo; echo "Cells, the same as k86 so the numbers compare cell for cell: llama-benchy pp2048/tg512 task mode, 3 runs, c1 and c8, at T=1.0 top-p 0.95 top-k 20 with thinking on (\`tgdef\`) and at T=0 with thinking off (\`tgt0\`); the 36-prompt coding probe at T=0 with thinking off, c1 and c8, one pass (aggregate output tok/s of the pass). GPU power is nvidia-smi power.draw at 1 Hz (GPU only, not wall power); 2x adds both Sparks. The last table lists the shipped builds from k86 (3 boots each) next to stock."
  echo; echo '```'; cat "$RES/cred.txt" 2>/dev/null || echo "no measured cells"; echo '```'
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
cp "$K/comment.md" "$RES/comment.md"
[ "$n" -gt 0 ] && FINAL="DONE: k84 stock 2x MTP 3 measured ($RES/cred.txt)" || FINAL="DONE: k84 no stock setup booted ($RES/attempts.txt)"
exit 0
