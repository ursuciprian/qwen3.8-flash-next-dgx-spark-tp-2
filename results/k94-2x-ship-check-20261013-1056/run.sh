#!/usr/bin/env bash
# k94-2x-ship-check (2026-10-13, tp-2 #137): ship check of 2x v2.1.0 = v2.0.0 with drafter D3 (checkpoint @ 03f4a057,
# k92 PROMOTE + gate PASS) + --enable-prompt-tokens-details, on the pair.
#   image   TAG = the v2.0.0 image + plan 225bf665 (682 records, the k92 re-keyed plan, identical on both Sparks) + the
#           torch AOT cache of both ranks for the 03f4a057 path (keys f509586e, 295b0fe4, k92 arm boots); local base built
#           by hand, CI build-b0-warm (run before this job; also tagged 2x-v2.1.0).
#   check   plain `sparkrun run` of the branch recipe ($REC) with the real runtime caches, from which the plan 225bf665 and
#           both AOT keys are moved out first on both Sparks (moved back at exit unless the boot recreated them): pass =
#           AOT loaded on both ranks (no Dynamo compile), 0 b12x plans measured on rank 0 and 682 records on both ranks,
#           pong, revision 03f4a057 + the flag on rank 0's command line, cached_tokens > 0 on a repeated ~12K prompt,
#           T=0 fresh-c8 acceptance per position within 0.03 of the k92 D3 arm, llama-benchy pp2048 + tg512 c1/c4/c8/c16
#           (3 runs) not below the k92 arm means by more than k92's noise band per cell.
#   end     lib.sh restores the shipped 2x (v2.0.0 from main); the promotion happens outside this job after a pass.
# ETA ~30 min + restore ~6 min. Contract: both locks; exit 75 if busy; STATE DONE/FAILED; comment.md + RESULT.
NEEDS="dgx01 dgx02"
set -u
J=k94; K=$HOME/GEN-AI/backlog/k94
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k94-2x-ship-check-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"
TAG=ghcr.io/ursuciprian/spark-vllm-b12x:k94-20261013-21e0b201-d21d7ade-warm
REV=03f4a0570496bbe741189f4a2a7d611679e85406; REC=$K/recipe.yaml
PLAN=225bf66521da31897db0719256f6a56cc38d43f87dd4f85f1f52549540986fe2.json; NREC=682
KEYS="f509586e7e4d7ee777946c1a37707d0f8702a0822867b3bf8a7e80d6f41ca1d5 295b0fe486ccbab14bbb3347df2365302e2ec9768c32913819908295ab17a18e"
RC=$HOME/.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923
CORPUS=$R/results/corpus-code.txt
GEN="List the numbers from 1 to 300 separated by commas. Output only the numbers, nothing else, no commentary."
on2() { ssh -n -o ConnectTimeout=10 $H2 "$@"; }
recok() { grep -qx "container: $TAG" "$REC" && grep -qx "model_revision: $REV" "$REC" && grep -q -- "--revision $REV" "$REC" \
  && grep -qx -- "    --enable-prompt-tokens-details \\\\" "$REC" && sed -n 1p "$REC" | grep -q '^# Release: v2.1.0'; }
if [ "${1:-}" = --dry-run ]; then rc=0
  recok && echo "ok recipe v2.1.0" || { echo "recipe check failed"; rc=1; }
  docker image inspect $TAG >/dev/null 2>&1 && echo "ok image on dgx-01" || { echo "image missing on dgx-01"; rc=1; }
  on2 "docker image inspect $TAG >/dev/null 2>&1" && echo "ok image on dgx-02" || echo "image not on dgx-02 yet (sparkrun copies it)"
  for f in check_seed.py depth_decode_probe.py; do [ -s "$R/scripts/$f" ] || { echo "missing $f"; rc=1; }; done
  echo "dry-run exit=$rc"; exit $rc; fi
need_lock
others_busy && { echo "another GPU job is active" >&2; exit 75; }
mkdir -p "$RES"; echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"; job_begin
MOVED=
unmove() { [ -n "$MOVED" ] || return 0; local h
  for h in localhost $H2; do x2 $h "cd $RC; for k in $KEYS; do [ -d vllm/torch_compile_cache/torch_aot_compile/\$k ] || { [ -d $K/moved/\$k ] && mv $K/moved/\$k vllm/torch_compile_cache/torch_aot_compile/; }; done
    [ -e b12x/compile/preparation/$PLAN ] || { [ -e $K/moved/$PLAN ] && mv $K/moved/$PLAN b12x/compile/preparation/; }; true"; done; MOVED=; }
x2() { if [ "$1" = localhost ]; then bash -c "$2"; else on2 "$2"; fi; }
trap 'kill $G1 $G2 2>/dev/null; unmove; restore || { sleep 60; restore; } || FINAL="FAILED: restore -- $FINAL"; st "$FINAL"' EXIT
recok || { FINAL="FAILED: k94 recipe check"; exit 1; }
cp "$K/run.sh" "$REC" "$RES/"
st "running: stop 2x"; stop_all
st "running: clear the 03f4a057 TP=2 seed entries from the runtime caches"
for h in localhost $H2; do x2 $h "mkdir -p $K/moved; cd $RC; for k in $KEYS; do [ -d vllm/torch_compile_cache/torch_aot_compile/\$k ] && mv vllm/torch_compile_cache/torch_aot_compile/\$k $K/moved/ && echo moved \$k; done
  [ -e b12x/compile/preparation/$PLAN ] && mv b12x/compile/preparation/$PLAN $K/moved/ && echo moved plan; ls vllm/torch_compile_cache/torch_aot_compile | grep -c . ; true"; done > "$RES/cleared.txt" 2>&1; MOVED=1
st "running: check boot"
D=$RES/check; mkdir -p "$D"; s=$(date +%s)
( cd "$K" && timeout -k 60 1800 sparkrun run "$REC" --no-follow ) > "$D/sparkrun.log" 2>&1 < /dev/null 9>&-
up=
while [ $(( $(date +%s) - s )) -lt 1800 ]; do [ "$(health localhost)" = 200 ] && { up=$(( $(date +%s) - s )); break; }
  [ $(( $(date +%s) - s )) -gt 240 ] && [ -z "$(n0)" ] && break; sleep 5; done
c=$(n0); [ -n "$c" ] || c=$(docker ps -a --format '{{.Names}}' | grep node_0 | head -1)
docker exec "$c" cat /tmp/sparkrun_serve.log > "$D/serve.log" 2>&1 || docker logs "$c" > "$D/serve.log" 2>&1
on2 'c=$(docker ps -a --format "{{.Names}}" | grep node_1 | head -1); [ -n "$c" ] && { docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1; }' > "$D/serve-node1.log" 2>&1
[ -n "$up" ] || { FINAL="FAILED: k94 check boot: no /health in 1800 s ($D)"; exit 1; }
docker exec "$c" sh -c "ps aux | grep '[v]llm serve'" > "$D/cmdline.txt" 2>&1
{ docker inspect --format '{{.Config.Image}}' "$c"; on2 "docker ps --format '{{.Image}}' --filter name=sparkrun"; } > "$D/image.txt" 2>&1
P=$(pong localhost); CT=$(python3 - "http://localhost:8000" "$CORPUS" <<'PY'
import json, sys, urllib.request
base, corpus = sys.argv[1:3]
text = open(corpus, errors="replace").read()[:48000]
body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": text + "\n\nIn one word: which language is most of the code above?"}],
        "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
for _ in range(2):
    r = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    u = json.load(urllib.request.urlopen(r, timeout=600))["usage"]
print(f"prompt_tokens={u['prompt_tokens']} cached_tokens={(u.get('prompt_tokens_details') or {}).get('cached_tokens')}")
PY
)
nr() { x2 $1 "python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))[\"records\"]))' $RC/b12x/compile/preparation/$PLAN" 2>/dev/null; }
N1=$(nr localhost); N2=$(nr $H2)
{ echo "image: $(tr '\n' ' ' < "$D/image.txt")"; echo "boot: health after $up s; pong: $P"
  for f in serve.log serve-node1.log; do echo "$f: Directly load AOT $(grep -c 'Directly load AOT compilation' "$D/$f"), saved AOT $(grep -c 'saved AOT compiled function' "$D/$f"), Dynamo bytecode transform $(grep -c 'Dynamo bytecode transform' "$D/$f"), MXFP8 copies $(grep -c 'MXFP8 copy serves rows >= 41 ' "$D/$f"), Traceback $(grep -c Traceback "$D/$f")"; done
  python3 "$R/scripts/check_seed.py" "$D/serve.log" 2>&1 | tail -2
  echo "plan $PLAN: rank 0 ${N1:-missing} records, rank 1 ${N2:-missing} records (seed $NREC)"
  echo "revision: $(grep -oE -- "--revision [0-9a-f]+" "$D/cmdline.txt" | head -1); prompt tokens details flag: $(grep -c -- '--enable-prompt-tokens-details' "$D/cmdline.txt")"
  echo "usage on a repeated prompt: $CT"; cat "$RES/cleared.txt" | sed 's/^/cleared before boot: /'; } > "$D/facts.txt"
WARM=WARM; for f in serve.log serve-node1.log; do grep -q 'Directly load AOT' "$D/$f" && ! grep -qE 'saved AOT compiled function|Dynamo bytecode transform' "$D/$f" || WARM=COLD; done
grep -q 'seed check PASS' "$D/facts.txt" && [ "$N1" = $NREC ] && [ "$N2" = $NREC ] || WARM="$WARM/plans-measured"
st "running: fresh-c8 acceptance probe + llama-benchy"
pos() { curl -s -m 10 "localhost:8000/metrics" | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens(_per_pos)?_total'; }
( cd "$R" && timeout -k 30 300 python3 scripts/depth_decode_probe.py --base "http://$H1:8000" --label warm --corpus "$CORPUS" --concurrency 1 --repeats 1 --temperature 0 --out "$D/probe-warm.json" --depth 4096 --new 2048 --offset 900000 --max-tokens 512 > "$D/probe-warm.log" 2>&1 )
pos > "$D/pos.before"
( cd "$R" && timeout -k 30 500 python3 scripts/depth_decode_probe.py --base "http://$H1:8000" --label fresh-c8 --corpus "$CORPUS" --concurrency 8 --repeats 2 --temperature 0 --out "$D/probe-fresh-c8.json" --depth 256 --new 2048 --offset 418916 --max-tokens 512 > "$D/probe-fresh-c8.log" 2>&1 )
pos > "$D/pos.after"
ACC=$(python3 - "$D/pos.before" "$D/pos.after" <<'PY'
import re, sys
def rd(p):
    c = {"accepted": {}, "draft": {}}
    for l in open(p):
        m = re.match(r'vllm:spec_decode_num_(accepted|draft)_tokens_per_pos_total\{.*position="(\d+)"\} ([0-9.e+]+)', l)
        if m: c[m[1]][int(m[2])] = c[m[1]].get(int(m[2]), 0) + float(m[3])
    return c
a, b = rd(sys.argv[1]), rd(sys.argv[2])
print(" ".join(f"{(b['accepted'].get(k,0)-a['accepted'].get(k,0))/max(b['draft'][k]-a['draft'].get(k,0),1):.3f}" for k in sorted(b["draft"])))
PY
)
( cd "$R" && child timeout -k 30 1800 uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$H1:8000/v1" --model $MODEL \
    --tokenizer "$TOK" --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 --concurrency 1 4 8 16 --runs 3 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$H1:8000/metrics" \
    --save-result "$D/task.csv" > "$D/task.log" 2>&1 )
BV=$(python3 - "$D/task.csv" "$ACC" <<'PY'
import sys
ref = {("pp", 1): (2775.8, 2.73), ("tg", 1): (83.8, 11.94), ("tg", 4): (161.9, 8.16), ("tg", 8): (190.9, 5.15), ("tg", 16): (240.9, 7.48)}
acc_ref = [0.853, 0.702, 0.576, 0.469]
cells, bad = [], []
for line in open(sys.argv[1]):
    f = [x.strip() for x in line.split("|")]
    if len(f) < 4 or not (f[2].startswith("pp") or f[2].startswith("tg")): continue
    test = f[2]; kind = test[:2]
    try: c = int(test.split("(c")[1].rstrip(")")); v = float(f[3].split()[0])
    except Exception: continue
    if (kind, c) not in ref: continue
    m, band = ref[(kind, c)]; d = (v / m - 1) * 100
    cells.append(f"{test} {v:.1f} ({d:+.1f}% vs k92 D3 arm {m}, band {band}%)")
    if d < -band: bad.append(test)
acc = [float(x) for x in sys.argv[2].split()] if sys.argv[2].strip() else []
if len(acc) < 4 or any(abs(a - b) > 0.03 for a, b in zip(acc, acc_ref)): bad.append(f"acceptance {acc} vs {acc_ref}")
if len(cells) < 5: bad.append(f"benchy cells {len(cells)}/5")
print(("OK" if not bad else "BAD (" + "; ".join(bad) + ")") + " | " + "; ".join(cells))
PY
)
keep() { docker exec "$c" cat /tmp/sparkrun_serve.log > "$D/serve-end.log" 2>&1; }; keep; stop_all
V=PASS; why=
[ "$WARM" = WARM ] || { V=FAIL; why="$why boot $WARM;"; }
echo "$P" | grep -qi pong || { V=FAIL; why="$why no pong;"; }
grep -q -- "--revision $REV" "$D/cmdline.txt" && grep -q -- '--enable-prompt-tokens-details' "$D/cmdline.txt" || { V=FAIL; why="$why cmdline;"; }
CTN=$(echo "$CT" | sed -n 's/.*cached_tokens=\([0-9]*\).*/\1/p'); [ "${CTN:-0}" -gt 0 ] 2>/dev/null || { V=FAIL; why="$why cached_tokens ${CTN:-missing};"; }
[ "${BV%% *}" = OK ] || { V=FAIL; why="$why ${BV%% |*};"; }
{ echo "k94 $(TZ=Europe/Bucharest date '+%F %T %Z'): 2x v2.1.0 ship check, image $TAG: $V"; cat "$D/facts.txt"
  echo "fresh-c8 acceptance per position (T=0): $ACC (k92 D3 arm 0.853 0.702 0.576 0.469)"; echo "benchy: $BV"; } > "$RES/k94.txt"
FINAL="DONE: k94 $V boot=$WARM ${up}s cached_tokens=${CTN:-?} acc=$ACC ($RES/k94.txt)${why:+ --$why}"
{ echo "k94-2x-ship-check result: I checked the 2x v2.1.0 candidate (v2.0.0 with drafter D3 at checkpoint revision \`${REV:0:8}\`, plus \`--enable-prompt-tokens-details\`) on the pair with a plain \`sparkrun run\` of the recipe. Before the boot I moved the D3 plan file and both AOT keys out of the runtime caches on both Sparks, so the image had to supply them, as on a new install."
  echo; echo "- Image \`$TAG\` (also tagged \`2x-v2.1.0\`): up in $up s, $WARM on both ranks; $(sed -n '/check PASS\|check FAIL/p' "$D/facts.txt"); plan records $N1 / $N2."
  echo "- Rank 0 serves \`--revision $REV\` with the new flag; on the second of two identical prompts the usage field reads $CT."
  echo "- T=0 fresh-c8 acceptance per draft position: $ACC (k92 D3 arm: 0.853 0.702 0.576 0.469)."
  echo "- llama-benchy (T=1, 3 runs) against the k92 D3 arm means: ${BV#*| }."
  echo; echo "Verdict: $V.${why:+ Reasons:$why} The quality gate is the k92 gate (same image, weights and plan selections; the new flag only adds a usage field)."
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"; cp "$K/comment.md" "$RES/comment.md"
exit 0
