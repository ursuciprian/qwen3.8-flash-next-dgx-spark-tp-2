#!/usr/bin/env bash
# refit-p2 (2026-10-06, #97 phase 2): GPU parity of the MTP refit training model against vLLM, offline-vs-live
# acceptance on 50 held-out prompts, then on-policy generation and capture for training. dgx-01 only (TP=1); the
# shipped 2x is stopped for the run and restored at the end (also on failure).
#   0. preflight  disk; base image files = vLLM f11fbbbf (BASE_MD5); build IMG (base + exp/mtp-refit-capture, 3 files);
#                 imports (vLLM hook, tools.mtp_refit with transformers qwen4_exp + b12x oracle; transformers 5.18
#                 into $P/pydeps if the image lacks qwen4_exp); CPU tests in the image; prompt mix (evals venv,
#                 `datasets`); 50 held-out ids for the live check
#   1. gen boot   v3d recipe on IMG, capture off. b12x seed check: every "b12x ready" line must say 0 measured
#                 (k48 lesson: same model path f4..03 as the image seed; a miss means retuning, reported as WARN)
#   2. live       the 50 prompts at c1, T=0 and T=1 (top_p 1, top_k -1), max 2048 tokens, /metrics before/after
#   3. gen        all prompts (train + held-out) at c8 until $BUDGET response tokens (default 2M, ~3 h)
#   4. cap boot   v3d + VLLM_MTP_CAPTURE_DIR/TOPK 20/TAIL 6144 + --no-enable-prefix-caching, own VLLM_CACHE_ROOT,
#                 and the decode numerics regime for prefill (review): VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS=0 (GDN NVFP4
#                 at every M, as verify passes of <= 40 rows) + VLLM_B12X_MXFP8_ACTIVATION_MODE=a16. Both are compile
#                 factors: cold first boot, b12x plans measured there (expected; only the gen boot must say 0).
#                 Drops VLLM_GDN_DEFERRED_CHECKPOINTS and VLLM_GDN_COMPACT_RECORDS: prefix caching off forces
#                 mamba_cache_mode none, and both refuse anything but align (16:02 cap boot failure). Decode-only
#                 state bookkeeping, bit-identical replay; capture rows are prefill rows, so no numerics change.
#                 capture_client for live-t0, live-t1 and the main set (records the drafter's greedy chains)
#   5. checks     (GPU container, no server) assemble; parity drafts (argmax per depth >= 0.995 on <= 500 docs of
#                 <= 2048 rows); eval_offline on the live sets (window 16384) + parity live (|offline - live| <=
#                 0.01 per position, T=0 and T=1); baseline eval_offline of the shipped drafter on held-out (depth 6)
#   6. restore    the shipped 2x, pong
# Never takes the gpu-lock: the caller holds it (GPU_LOCK_HELD=1; after_k53.sh does that once k53 is over).
# Usage:  flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash ~/GEN-AI/refit-p2/job.sh   (after_k53.sh does this)
#         bash ~/GEN-AI/refit-p2/job.sh --dry-run      (static checks only: files, recipes, base image present)
#         ... job.sh --resume-cap   (DAY pinned: start at 4. cap boot; prompts, gen and live must already be on disk,
#                                    they are checked and never regenerated)
# Output: $RES (results/refit-p2-<date>), $RES/refit-p2.txt, ~/GEN-AI/refit-p2/STATE; data in $D (~250 GB).
# Stop: kill -TERM <job pid in refit-p2.log>, never pkill -f.
set -u
export PATH="$HOME/.local/bin:$PATH"
P=$HOME/GEN-AI/refit-p2; G=$HOME/GEN-AI; R=$G/qwen3.8-flash-next-dgx-spark-tp-2
H1=192.168.100.62; H2=192.168.100.53
BASE=ghcr.io/ursuciprian/spark-vllm-b12x:tp1-v3d-20261005-21e0b201-5dad364d-warm
IMG=spark-vllm-b12x:mtpcap-21e0b201-$(cut -c1-8 "$P/COMMIT")
SNAPREL=hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/f400000000000000000000000000000000000003
VOCAB=/opt/mtp-vocab/ids-v2-K131072.txt.gz
DAY=${DAY:-$(TZ=Europe/Bucharest date +%Y%m%d)}   # data dir day; the queue pins it
D=$HOME/.cache/huggingface/mtp-refit/p2-$DAY; CD=/cache/huggingface/mtp-refit/p2-$DAY   # host / container
RES=${RES:-$R/results/refit-p2-$DAY}
BUDGET=${BUDGET:-2000000}; DISKMIN_GB=400
EVALPY=$G/evals/venv/bin/python
SRC=$G/k53/v3d.yaml

mkrecipes() { # gen.yaml / cap.yaml from the v3d control recipe, every edit checked
  sed -e "1i # refit-p2 gen (#97): v3d on $IMG (vLLM exp/mtp-refit-capture $(cut -c1-8 "$P/COMMIT")), capture off" \
      -e "s|^name: qwen3.8-flash-next-1x-dgx-spark-v3d\$|name: qwen3.8-flash-next-1x-dgx-spark-refit-gen|" \
      -e "s|^container: .*|container: $IMG|" "$SRC" > "$P/gen.yaml"
  sed -e "1s|.*|# refit-p2 cap (#97): v3d on $IMG with the MTP capture hook on, prefix caching off|" \
      -e "s|^name: .*|name: qwen3.8-flash-next-1x-dgx-spark-refit-cap|" \
      -e "s|^env:\$|env:\n  VLLM_MTP_CAPTURE_DIR: \"$CD/capture/shards\"\n  VLLM_MTP_CAPTURE_TOPK: \"20\"\n  VLLM_MTP_CAPTURE_TAIL: \"6144\"\n  VLLM_CACHE_ROOT: \"/cache/runtime/vllm-refit-cap\"|" \
      -e "s|^  VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS: \"41\"\$|  VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS: \"0\"\n  VLLM_B12X_MXFP8_ACTIVATION_MODE: \"a16\"|" \
      -e "/^  VLLM_GDN_DEFERRED_CHECKPOINTS: /d" -e "/^  VLLM_GDN_COMPACT_RECORDS: /d" \
      -e "s|    --enable-prefix-caching \\\\|    --no-enable-prefix-caching \\\\|" "$P/gen.yaml" > "$P/cap.yaml"
  grep -qx "container: $IMG" "$P/gen.yaml" && grep -q "^name: .*refit-gen\$" "$P/gen.yaml" \
    && [ "$(diff "$SRC" "$P/gen.yaml" | grep -c '^>')" = 3 ] \
    && grep -qx "  VLLM_MTP_CAPTURE_DIR: \"$CD/capture/shards\"" "$P/cap.yaml" \
    && grep -q -- "--no-enable-prefix-caching" "$P/cap.yaml" && ! grep -q -- "    --enable-prefix-caching" "$P/cap.yaml" \
    && grep -qx '  VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS: "0"' "$P/cap.yaml" \
    && grep -qx '  VLLM_B12X_MXFP8_ACTIVATION_MODE: "a16"' "$P/cap.yaml" \
    && [ "$(diff "$P/gen.yaml" "$P/cap.yaml" | grep -c '^>')" = 9 ] \
    && grep -q '^  VLLM_GDN_DEFERRED_CHECKPOINTS: "1"$' "$P/gen.yaml" && grep -q '^  VLLM_GDN_COMPACT_RECORDS: "1"$' "$P/gen.yaml" \
    && ! grep -qE '^  VLLM_GDN_(DEFERRED_CHECKPOINTS|COMPACT_RECORDS):' "$P/cap.yaml" \
    || { echo "mkrecipes: edits did not apply"; return 1; }; }

if [ "${1:-}" = --dry-run ]; then
  rc=0
  for f in COMMIT img/Dockerfile img/BASE_MD5 img/vllm/envs.py img/vllm/v1/worker/gpu/model_runner.py \
           img/vllm/v1/worker/gpu/mtp_capture.py src/tools/mtp_refit/train.py src/tools/mtp_refit/mix.yaml; do
    [ -f "$P/$f" ] || { echo "missing $P/$f"; rc=1; }; done
  docker image inspect "$BASE" >/dev/null 2>&1 || { echo "base image $BASE missing"; rc=1; }
  [ -x "$EVALPY" ] && "$EVALPY" -c "import datasets, yaml" 2>/dev/null || { echo "$EVALPY lacks datasets/yaml"; rc=1; }
  ( cd "$P/src" && "$EVALPY" -m tools.mtp_refit.gen validate --out "$D/prompts" ) && echo "prompts: valid, step skipped" \
    || echo "prompts: not built yet (preflight builds them, ~15 min)"
  ( cd "$P/src" && python3 -m py_compile tools/mtp_refit/*.py && python3 -m tools.mtp_refit.gen run --help >/dev/null \
    && python3 -m tools.mtp_refit.capture_client --help >/dev/null ) || { echo "tools do not parse on the host"; rc=1; }
  T=$(mktemp -d); P0=$P; P=$T; cp "$P0/COMMIT" "$T/"; mkrecipes && echo "recipes: ok (gen +3 lines vs v3d, cap +9 vs gen)" \
    || rc=1; diff "$T/gen.yaml" "$T/cap.yaml"; P=$P0; rm -rf "$T"
  echo "free: $(df -BG --output=avail "$HOME/.cache/huggingface" | tail -1 | tr -d ' G') GB (need $DISKMIN_GB)"
  echo "k53 STATE: $(head -1 $G/k53/STATE 2>/dev/null)"
  echo "estimate: preflight ~20 min (build, tests, prompts) + 2 boots + live ~1 h + gen ~3 h at 2M tokens + capture ~1 h"
  echo "          + checks ~30 min + restore ~10 min: ~6-7 h"
  echo "dry-run exit=$rc"; exit $rc; fi
[ "${GPU_LOCK_HELD:-}" = 1 ] || { echo "refused: hold ~/GEN-AI/gpu-lock and set GPU_LOCK_HELD=1" >&2; exit 2; }
RESUME=; [ "${1:-}" = --resume-cap ] && RESUME=cap
[ -n "$RESUME" ] && [ ! -d "$D/gen" ] && { echo "refused: --resume-cap but no $D/gen (pin DAY)" >&2; exit 2; }

mkdir -p "$RES" "$D"
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" | tee -a "$RES/refit-p2.log"; }
st() { echo "$*" > "$P/STATE"; echo "$*" > "$RES/STATE"; log "STATE: $*"; }
health() { curl -s -m 5 -o /dev/null -w '%{http_code}' "$1:8000/health"; }
pong() { curl -s -m 120 "$1:8000/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Reply with exactly one word: pong"}],"max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>&1; }
n0() { docker ps --format '{{.Names}}' | grep -E 'node_0|_solo|sparkrun' | head -1; }
E=$(docker ps --format '{{.Names}} {{.Image}}' | awk '/node_0/ && /spark-vllm-b12x/ {print $2; exit}' | sed 's|.*:||')
E=${E:-b1.4-20261001-b7fbaf96-a7e649d8-warm}
is_shipped() { [ "$(health localhost)" = 200 ] && docker ps --format '{{.Image}}' | grep -q ":$E\$" \
  && ssh -n -o ConnectTimeout=10 $H2 "docker ps --format '{{.Image}}'" | grep -q ":$E\$" \
  && docker exec "$(n0)" sh -c "ps aux | grep '[v]llm serve'" | grep -q -- '--tensor-parallel-size 2'; }
stop_all() { ( cd "$R" && timeout -k 30 600 sparkrun stop --all ) >> "$RES/refit-p2.log" 2>&1 9>&-; sleep 10
  for h in $H1 $H2; do timeout -k 30 300 sparkrun stop --all --hosts $h >> "$RES/refit-p2.log" 2>&1 9>&-; done; sleep 5
  docker ps -q --filter name=sparkrun | xargs -r docker rm -f >/dev/null 2>&1
  ssh -n -o ConnectTimeout=10 $H2 "docker ps -q --filter name=sparkrun | xargs -r docker rm -f" >/dev/null 2>&1; }
restore() {
  docker rm -f refit-p2-py >/dev/null 2>&1
  if is_shipped && pong localhost | grep -qi pong; then log "restore: shipped 2x already serving"; return 0; fi
  stop_all; ( cd "$R" && timeout -k 60 3600 sparkrun run qwen3.8-flash-next-2x-dgx-spark --no-follow ) >> "$RES/refit-p2.log" 2>&1 < /dev/null 9>&-
  local s=$(date +%s); until [ "$(health localhost)" = 200 ] || [ $(( $(date +%s) - s )) -gt 1800 ]; do sleep 15; done
  local p; p=$(pong localhost)
  is_shipped && echo "$p" | grep -qi pong && { log "restore: shipped 2x ($E) verified, pong=$p"; return 0; }
  log "FAILED: restore -- MANUAL INTERVENTION (pong=$p)"; return 1; }
boot() { # recipe outdir -> 0 up
  local rec=$1 d=$2 s c; mkdir -p "$d"; stop_all; cp "$rec" "$d/recipe.yaml"
  ( cd "$P" && timeout -k 30 900 sparkrun run "$rec" --hosts $H1 --solo --no-follow ) > "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  s=$(date +%s)
  until [ "$(health localhost)" = 200 ]; do
    c=$(n0)
    if [ $(( $(date +%s) - s )) -gt 120 ] && { [ -z "$c" ] || docker exec "$c" grep -qE 'Worker failed with error|EngineCore failed to start|Engine core initialization failed' /tmp/sparkrun_serve.log 2>/dev/null; }; then
      keep_logs "$d"; log "boot $(basename "$rec") FAILED"; return 1; fi
    [ $(( $(date +%s) - s )) -gt 3600 ] && { keep_logs "$d"; log "boot $(basename "$rec") health timeout"; return 1; }
    sleep 15; done
  keep_logs "$d"; local p; p=$(pong localhost)
  log "boot $(basename "$rec") up in $(( $(date +%s) - s ))s, pong=$p, $(grep -m1 -oE 'Directly load AOT compilation|Dynamo bytecode transform time' "$d/serve.log" || echo 'no AOT line')"
  echo "$p" | grep -qi pong; }
keep_logs() { local c; c=$(docker ps -a --format '{{.Names}}' | grep -E 'sparkrun|_solo' | head -1); [ -n "$c" ] || return 0
  docker exec "$c" cat /tmp/sparkrun_serve.log > "$1/serve.log" 2>/dev/null || docker logs "$c" > "$1/serve.log" 2>&1; }
seedcheck() { # serve.log -> the b12x ready lines; WARN unless every one says 0 measured
  local l; l=$(grep -hoE "b12x ready [a-z_.]+: [0-9]+/[0-9]+ ready, [0-9]+ measured, [0-9]+ cached" "$1" | sort -u)
  echo "$l"
  if [ -z "$l" ]; then echo "WARN: no b12x ready lines"
  elif echo "$l" | grep -qvE ", 0 measured,"; then echo "WARN: b12x measured plans (seed miss: retuned at boot)"
  else echo "seed: 0 measured"; fi; }
pyrun() { # python3 args... in IMG with the GPU, tools.mtp_refit on the path
  docker rm -f refit-p2-py >/dev/null 2>&1
  docker run --rm --name refit-p2-py --gpus all --ipc=host -v "$HOME/.cache/huggingface:/cache/huggingface" -v "$P:/work" \
    -w /work/src -e PYTHONPATH=/work/src:/work/pydeps --entrypoint python3 "$IMG" "$@" < /dev/null; }
child() { "$@" & CH=$!; wait $CH; local rc=$?; CH=; return $rc; }

GUARD='while :; do m=$(awk "/^MemAvailable/{print int(\$2/1024)}" /proc/meminfo); echo "$(date +%T) $m"; if [ "$m" -lt 2048 ]; then docker ps -q --filter name=sparkrun | xargs -r docker rm -f >/dev/null 2>&1; docker rm -f refit-p2-py >/dev/null 2>&1; echo "ABORT MemAvailable $m MiB < 2048"; exit 3; fi; sleep 1; done'
bash -c "$GUARD" > "$RES/guard-dgx01.log" 2>&1 < /dev/null 9>&- & GU=$!
CH=; FINAL="FAILED: job"
trap 'kill $GU 2>/dev/null; restore || { sleep 60; restore; } || FINAL="FAILED: restore -- $FINAL"; st "$FINAL"' EXIT
trap 'log "signal: stopping child $CH"; [ -n "$CH" ] && kill -TERM $CH 2>/dev/null && wait $CH; docker rm -f refit-p2-py >/dev/null 2>&1; FINAL="FAILED: stopped by signal"; exit 143' TERM INT HUP
log "refit-p2 pid $$, RES $RES, data $D, image $IMG, budget $BUDGET, 2x tag for restore: $E"
REP=$RES/refit-p2.txt
echo "refit-p2 $(TZ=Europe/Bucharest date '+%F %T %Z'): #97 phase 2, vLLM exp/mtp-refit-capture $(cut -c1-8 "$P/COMMIT"), image $IMG" > "$REP.new"
if [ -n "$RESUME" ]; then { echo; sed 's/^/resumed at cap boot: /' "$REP.new"; } >> "$REP"; rm -f "$REP.new"; else mv "$REP.new" "$REP"; fi

# 0. preflight (no GPU)
st "running: preflight"
free=$(df -BG --output=avail "$HOME/.cache/huggingface" | tail -1 | tr -d ' G')
[ "$free" -ge $DISKMIN_GB ] || { FINAL="FAILED: preflight: $free GB free < $DISKMIN_GB"; exit 1; }
if [ -n "$RESUME" ]; then # prompts, gen and live are done: check them, never regenerate
  docker image inspect "$IMG" >/dev/null 2>&1 || { FINAL="FAILED: resume: image $IMG missing"; exit 1; }
  mkrecipes > "$RES/recipes.txt" 2>&1 || { FINAL="FAILED: resume: recipes ($RES/recipes.txt)"; exit 1; }
  { echo; echo "== resume: gen data on disk"; } >> "$REP"
  python3 - "$D" >> "$REP" 2>&1 <<'EOF' || { FINAL="FAILED: resume: gen data check ($REP)"; exit 1; }
import collections, glob, json, os, sys
for sub in ("gen", "gen-live-t0", "gen-live-t1"):
    n, toks, ids = collections.Counter(), 0, set()
    for f in sorted(glob.glob(os.path.join(sys.argv[1], sub, "*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            assert r["id"] not in ids and r["prompt_token_ids"] and r["output_token_ids"], (f, r["id"])
            ids.add(r["id"]); n[r["category"]] += 1; toks += len(r["output_token_ids"])
    assert n, f"{sub}: no records"
    print(f"{sub}: {sum(n.values())} records {dict(sorted(n.items()))}, {toks} response tokens")
EOF
  [ -d "$RES/cap-boot" ] && mv "$RES/cap-boot" "$RES/cap-boot-failed-$(date +%Y%m%d-%H%M%S)"
  log "resume at cap boot: image $IMG, $(grep -m1 '^gen: ' "$REP"), $free GB free"
else
docker run --rm --entrypoint md5sum "$BASE" /usr/local/lib/python3.12/dist-packages/vllm/envs.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py > "$RES/base-md5.txt" 2>&1
while read -r m f; do grep -q "^$m  .*/$f\$" "$RES/base-md5.txt" || { FINAL="FAILED: preflight: base image $f differs from f11fbbbf ($RES/base-md5.txt)"; exit 1; }
  done < "$P/img/BASE_MD5"
docker image inspect "$IMG" >/dev/null 2>&1 || child docker build --build-arg BASE="$BASE" -t "$IMG" "$P/img" > "$RES/build.log" 2>&1 \
  || { FINAL="FAILED: preflight: image build ($RES/build.log)"; exit 1; }
IMPORTS='import vllm.v1.worker.gpu.mtp_capture, vllm.envs as e; assert "VLLM_MTP_CAPTURE_DIR" in e.environment_variables; import tools.mtp_refit.mtp_ref, tools.mtp_refit.parity; print("imports ok")'
if ! pyrun -c "$IMPORTS" > "$RES/imports.txt" 2>&1; then
  log "imports failed in the image, installing transformers 5.18.0 into $P/pydeps"
  pyrun -m pip install -q --no-deps --target /work/pydeps "transformers==5.18.0" >> "$RES/imports.txt" 2>&1
  pyrun -c "$IMPORTS" >> "$RES/imports.txt" 2>&1 || { FINAL="FAILED: preflight: imports ($RES/imports.txt)"; exit 1; }; fi
pyrun -c "import pytest" >/dev/null 2>&1 && { pyrun -m pytest -q -p no:cacheprovider tools/mtp_refit/tests > "$RES/tests-tools.txt" 2>&1
  docker run --rm --entrypoint python3 -w /usr/local/lib/python3.12/dist-packages -v "$P/vllm-tests:/t" "$IMG" \
    -m pytest -q -p no:cacheprovider --noconftest /t/test_mtp_capture_cpu.py > "$RES/tests-vllm.txt" 2>&1
  log "CPU tests in image: tools $(tail -1 "$RES/tests-tools.txt"); vllm $(tail -1 "$RES/tests-vllm.txt")"
  grep -qE "failed|error" "$RES/tests-tools.txt" "$RES/tests-vllm.txt" && { FINAL="FAILED: preflight: CPU tests in the image"; exit 1; }; }
mkrecipes > "$RES/recipes.txt" 2>&1 || { FINAL="FAILED: preflight: recipes ($RES/recipes.txt)"; exit 1; }
# judged by the validated files (MANIFEST.json, counts, JSON), not the exit code: datasets/pyarrow can abort at
# interpreter shutdown after a complete run. The step resumes per category, so a retry only redoes what is missing.
pvalid() { ( cd "$P/src" && "$EVALPY" -m tools.mtp_refit.gen validate --out "$D/prompts" ) >> "$RES/prompts.log" 2>&1; }
for try in 1 2 3; do pvalid && break
  log "prompt mix: attempt $try"
  ( cd "$P/src" && child nice -n 19 ionice -c3 "$EVALPY" -m tools.mtp_refit.gen prompts --out "$D/prompts" ) >> "$RES/prompts.log" 2>&1
done
pvalid || { FINAL="FAILED: preflight: prompt mix not valid after 3 attempts ($RES/prompts.log)"; exit 1; }
python3 - "$D/prompts" "$D/live-ids.txt" > "$RES/live-ids.log" 2>&1 <<'EOF' || { FINAL="FAILED: preflight: live ids"; exit 1; }
import glob, json, os, sys
quota = {"agentic": 20, "tools": 10, "code": 10, "chat": 5, "math": 5}   # the mix, 50 prompts
ids = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.jsonl"))):
    cat = os.path.basename(f)[:-6]
    ps = [json.loads(l) for l in open(f)]
    ps = [p for p in ps if p["split"] == "heldout" and len(json.dumps(p["messages"]) + json.dumps(p.get("tools", []))) <= 20000]
    ids += [p["id"] for p in ps[: quota.get(cat, 0)]]
    print(cat, min(len(ps), quota.get(cat, 0)))
assert len(ids) >= 40, len(ids)
open(sys.argv[2], "w").write("\n".join(ids) + "\n")
EOF
cat "$RES/prompts.log" "$RES/live-ids.log" 2>/dev/null | tail -8 >> "$REP"
log "preflight ok: image $IMG, $(wc -l < "$D/live-ids.txt") live ids, $free GB free"

# 1-3. generation server (shipped v3d config on IMG)
st "running: gen boot"
child boot "$P/gen.yaml" "$RES/gen-boot" || { FINAL="FAILED: gen boot ($RES/gen-boot)"; exit 1; }
{ echo; echo "== gen boot (v3d on $IMG, capture off)"; seedcheck "$RES/gen-boot/serve.log"; } >> "$REP"
GEN() { ( cd "$P/src" && python3 -m tools.mtp_refit.gen run --server http://localhost:8000 --prompts "$D/prompts" "$@" ); }
st "running: live (50 prompts, c1, T=0 and T=1)"
child GEN --ids "$D/live-ids.txt" --out "$D/gen-live-t0" --concurrency 1 --max-tokens 2048 --temperature 0 --metrics \
  > "$RES/gen-live-t0.log" 2>&1 || { FINAL="FAILED: live T=0 ($RES/gen-live-t0.log)"; exit 1; }
child GEN --ids "$D/live-ids.txt" --out "$D/gen-live-t1" --concurrency 1 --max-tokens 2048 --temperature 1 --top-p 1 \
  --top-k -1 --metrics > "$RES/gen-live-t1.log" 2>&1 || { FINAL="FAILED: live T=1 ($RES/gen-live-t1.log)"; exit 1; }
st "running: gen ($BUDGET response tokens)"
child GEN --out "$D/gen" --concurrency 8 --budget-tokens "$BUDGET" > "$RES/gen.log" 2>&1 \
  || { FINAL="FAILED: generation ($RES/gen.log)"; exit 1; }
{ echo; echo "== generation"; tail -1 "$RES/gen-live-t0.log"; tail -1 "$RES/gen-live-t1.log"; tail -1 "$RES/gen.log"; } >> "$REP"
fi

# 4. capture server
st "running: cap boot"
mkdir -p "$D/capture/shards"
child boot "$P/cap.yaml" "$RES/cap-boot" || { FINAL="FAILED: cap boot ($RES/cap-boot)"; exit 1; }
{ echo; echo "== cap boot (capture on, decode numerics, no prefix caching, own VLLM_CACHE_ROOT; cold, measured plans expected)"
  seedcheck "$RES/cap-boot/serve.log"; } >> "$REP"
st "running: capture"
for s in live-t0 live-t1 main; do g=$D/gen; [ $s = main ] || g=$D/gen-$s
  ( cd "$P/src" && child python3 -m tools.mtp_refit.capture_client --server http://localhost:8000 --gen "$g" \
      --out "$D/capture/$s" --concurrency 4 ) > "$RES/capture-$s.log" 2>&1 || { FINAL="FAILED: capture $s ($RES/capture-$s.log)"; exit 1; }
  echo "capture $s: $(tail -1 "$RES/capture-$s.log")" >> "$REP"; done
sleep 40   # the hook flushes its last shard after 30 s idle
keep_logs "$RES/cap-boot"; stop_all
echo "capture shards: $(ls "$D/capture/shards" | wc -l), $(du -sh "$D/capture/shards" | cut -f1)" >> "$REP"

# 5. checks (GPU container, nothing serving)
st "running: checks"
SN=/cache/huggingface/$SNAPREL
for s in main live-t0 live-t1; do o=$CD/data-$s; [ $s = main ] && o=$CD/data
  child pyrun -m tools.mtp_refit.assemble --capture "$CD/capture/shards" --manifest "$CD/capture/$s/manifest.jsonl" --out "$o" \
    > "$RES/assemble-$s.txt" 2>&1 || { FINAL="FAILED: assemble $s ($RES/assemble-$s.txt)"; exit 1; }
  echo "assemble $s: $(tail -1 "$RES/assemble-$s.txt")" >> "$REP"; done
child pyrun -m tools.mtp_refit.parity drafts --snapshot "$SN" --draft-vocab $VOCAB --data "$CD/data/heldout" "$CD/data/train" \
  --max-docs 500 --out /work/parity-drafts.json > "$RES/parity-drafts.txt" 2>&1; PD=$?
cp "$P/parity-drafts.json" "$RES/" 2>/dev/null
LV=
for t in t0 t1; do
  child pyrun -m tools.mtp_refit.eval_offline --snapshot "$SN" --draft-vocab $VOCAB --data "$CD/data-live-$t/heldout" \
    --depth 4 --window 16384 --out "/work/eval-live-$t.json" > "$RES/eval-live-$t.txt" 2>&1 \
    && child pyrun -m tools.mtp_refit.parity live --offline "/work/eval-live-$t.json" --key $t \
         --metrics "$CD/gen-live-$t" --out "/work/live-$t.json" > "$RES/parity-live-$t.txt" 2>&1
  LV="$LV $t=$([ $? = 0 ] && echo PASS || echo FAIL)"; cp "$P/eval-live-$t.json" "$P/live-$t.json" "$RES/" 2>/dev/null; done
child pyrun -m tools.mtp_refit.eval_offline --snapshot "$SN" --draft-vocab $VOCAB --data "$CD/data/heldout" --depth 6 \
  --out /work/eval-baseline.json > "$RES/eval-baseline.txt" 2>&1; cp "$P/eval-baseline.json" "$RES/" 2>/dev/null
{ echo; echo "== parity drafts (vLLM greedy chain vs MtpRef argmax per depth, pass >= 0.995): $([ $PD = 0 ] && echo PASS || echo FAIL)"
  tail -1 "$RES/parity-drafts.txt"
  echo "== offline vs live per position (pass |diff| <= 0.01):$LV"
  for t in t0 t1; do tail -1 "$RES/parity-live-$t.txt"; done
  echo "== baseline offline acceptance of the shipped drafter on held-out (all categories)"
  tail -40 "$RES/eval-baseline.txt"; } >> "$REP"
FINAL="DONE: parity drafts=$([ $PD = 0 ] && echo PASS || echo FAIL) live:$LV ($REP)"
log "$FINAL"
