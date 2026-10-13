#!/usr/bin/env bash
# k94b-2x-ship-recheck (2026-10-13, tp-2 #137): rule in RULE.md. Run by hand under all_locks (no runner alive).
set -u
J=k94b; K=$HOME/GEN-AI/backlog/k94b; source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"
RES=$R/results/k94b-2x-ship-recheck-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M); mkdir -p "$RES"; echo "$RES" > "$K/RESULT"
REC=$BL/k94/recipe.yaml; K94=$(cat "$BL/k94/RESULT")
cp "$K/RULE.md" "$K/run.sh" "$REC" "$RES/"; st "running: recheck boot"
stop_all; D=$RES/check; mkdir -p "$D"; s=$(date +%s)
( cd "$BL/k94" && timeout -k 60 1800 sparkrun run "$REC" --no-follow ) > "$D/sparkrun.log" 2>&1 < /dev/null 9>&-
up=; while [ $(( $(date +%s) - s )) -lt 1800 ]; do [ "$(health localhost)" = 200 ] && { up=$(( $(date +%s) - s )); break; }; sleep 5; done
c=$(n0); docker exec "$c" cat /tmp/sparkrun_serve.log > "$D/serve.log" 2>&1
docker exec "$c" sh -c "ps aux | grep '[v]llm serve'" > "$D/cmdline.txt" 2>&1
P=$(pong localhost); echo "boot ${up:-none} s; pong $P; AOT $(grep -c 'Directly load AOT' "$D/serve.log"), Dynamo $(grep -c 'Dynamo bytecode transform' "$D/serve.log"); $(python3 "$R/scripts/check_seed.py" "$D/serve.log" | tail -1)" > "$D/boot.txt"
[ -n "$up" ] || { st "FAILED: k94b recheck boot"; exit 1; }
st "running: llama-benchy"
( cd "$R" && timeout -k 30 1800 uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$H1:8000/v1" --model $MODEL \
    --tokenizer "$TOK" --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 --concurrency 1 4 8 16 --runs 3 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$H1:8000/metrics" \
    --save-result "$D/task.csv" > "$D/task.log" 2>&1 )
V=$(python3 - "$K94/check/task.csv" "$D/task.csv" <<'PY'
import sys
ref = {"pp2048 (c1)": (2775.8, 2.73, 2659.2), "tg512 (c1)": (83.8, 11.94, 74.5), "tg512 (c4)": (161.9, 8.16, 146.8),
       "tg512 (c8)": (190.9, 5.15, 180.9), "tg512 (c16)": (240.9, 7.48, 225.0)}
def rd(p):
    o = {}
    for line in open(p):
        f = [x.strip() for x in line.split("|")]
        if len(f) > 3 and f[2] in ref:
            o[f[2]] = float(f[3].split()[0])
    return o
a, b = rd(sys.argv[1]), rd(sys.argv[2]); bad, out = [], []
for k, (m, band, lo) in ref.items():
    if k not in a or k not in b: bad.append(f"{k} missing"); continue
    mean = (a[k] + b[k]) / 2; d = (mean / m - 1) * 100
    ok = d >= -band and min(a[k], b[k]) >= lo
    out.append(f"{k}: k94 {a[k]:.1f}, recheck {b[k]:.1f}, mean {mean:.1f} ({d:+.1f}% vs {m}, band {band}%), lowest k92 boot {lo} {'ok' if ok else 'FAIL'}")
    if not ok: bad.append(k)
print(("PASS" if not bad else "FAIL (" + ", ".join(bad) + ")") + "\n" + "\n".join(out))
PY
)
echo "$V" > "$RES/k94b.txt"; cat "$D/boot.txt" >> "$RES/k94b.txt"
st "DONE: k94b ${V%%$'\n'*} (server left up: $( [ "${V%% *}" = PASS ] && echo v2.1.0 branch recipe || echo stopping))"
[ "${V%% *}" = PASS ] || { stop_all; restore; }
