#!/usr/bin/env python3
"""Compare one candidate benchy table against the day's SGLang control envelope.

The four SGLang control boots of 2026-09-06 (same recipe, fresh boot each) give a
per-cell min..max. A candidate cell outside that envelope is a real difference;
inside it is noise. Prefill envelopes are tight (1.9-3.2% median drift), decode
envelopes are wide (10% median, 25% worst) - read them accordingly.

    python3 scripts/compare_engines.py results/arms/fn-vllm-tony-speed.csv
"""
import pathlib, re, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
CONTROLS = [
    "results/arms/01-fn-bigkv-control.csv",
    "results/arms/03-fn-bigkv-control.csv",
    "results/arms/sep03-01-control.csv",
    "results/arms/sep03-03-control.csv",
]
CELL = re.compile(r"^(?P<m>\w+)(?: @ d(?P<d>\d+))?(?: \(c(?P<c>\d+)\))?$")

def parse(p):
    o = {}
    for l in pathlib.Path(p).read_text().splitlines():
        l = l.strip()
        if not l.startswith("|") or l.startswith("|:"): continue
        c = [x.strip() for x in l.strip("|").split("|")]
        if len(c) < 4 or c[1] == "test": continue
        m = CELL.match(c[1])
        if not m: continue
        try: v = float(c[2].split("±")[0])
        except ValueError: continue
        if v > 5000 and m["m"] in ("tg128", "ctx_tg"): continue  # benchy artifact
        o[(m["m"], int(m["d"] or 0), int(m["c"] or 1))] = v
    return o

cand = parse(sys.argv[1])
ctrls = [parse(REPO / c) for c in CONTROLS if (REPO / c).exists()]
print(f"candidate: {sys.argv[1]}   vs {len(ctrls)} SGLang control boots\n")
for metric in ("tg128", "pp2048"):
    print(f"=== {metric} ===")
    print(f"{'depth':>6}{'c':>3}{'sglang env':>18}{'sglang mid':>11}{'vllm':>9}{'vs mid':>9}  verdict")
    above = below = inside = 0
    for k in sorted(k for k in cand if k[0] == metric):
        env = [c[k] for c in ctrls if k in c]
        if not env: continue
        lo, hi, mid = min(env), max(env), sum(env) / len(env)
        v = cand[k]; d = (v - mid) / mid * 100
        if v > hi: verdict, above = f"ABOVE +{(v-hi)/hi*100:.0f}%", above + 1
        elif v < lo: verdict, below = f"below -{(lo-v)/lo*100:.0f}%", below + 1
        else: verdict, inside = "inside", inside + 1
        print(f"{k[1]:>6}{k[2]:>3}{lo:>9.1f}-{hi:<8.1f}{mid:>11.1f}{v:>9.1f}{d:>+8.1f}%  {verdict}")
    print(f"  -> above {above}, inside {inside}, below {below}\n")
