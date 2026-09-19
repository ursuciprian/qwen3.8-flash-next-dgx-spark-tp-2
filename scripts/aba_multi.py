#!/usr/bin/env python3
"""Drift-aware A/B/A for a ladder of the form  ctrl, armA, ctrl, armB, ctrl, ...

Each test arm is judged against the two controls that bracket it. A cell is
significant only if |delta| >= 5% AND |delta| > 1.5x the bracket's own drift.
Also prints the pooled control drift across all controls in the ladder.

    python3 scripts/aba_multi.py results/arms 01-fn-bigkv-control 02-fn-bigkv-nospec 03-fn-bigkv-control 04-fn-bigkv-mtp1 05-fn-bigkv-control
"""
import re, sys, pathlib, statistics
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
        if v > 5000 and m["m"] in ("tg128", "ctx_tg"): continue
        o[(m["m"], int(m["d"] or 0), int(m["c"] or 1))] = v
    return o

d = pathlib.Path(sys.argv[1]); names = sys.argv[2:]
arms = [(n, parse(d / f"{n}.csv")) for n in names if (d / f"{n}.csv").exists()]
ctrl_idx = [i for i, (n, _) in enumerate(arms) if "control" in n]
for metric in ("tg128", "pp2048"):
    print(f"\n===== {metric} =====")
    for i, (n, a) in enumerate(arms):
        if "control" in n: continue
        lo = max([j for j in ctrl_idx if j < i], default=None); hi = min([j for j in ctrl_idx if j > i], default=None)
        if lo is None or hi is None: print(f"{n}: no bracketing controls"); continue
        A, B = arms[lo][1], arms[hi][1]
        print(f"\n-- {n}  (vs {arms[lo][0]} / {arms[hi][0]})")
        print(f"{'depth':>6}{'c':>3}{'ctrlA':>8}{'ctrlB':>8}{'arm':>8}{'drift':>7}{'delta':>8}  sig")
        sig = 0; n_cells = 0; deltas = []
        for k in sorted(k for k in a if k[0] == metric and k in A and k in B):
            mean = (A[k] + B[k]) / 2; drift = abs(B[k] - A[k]) / mean * 100; delta = (a[k] - mean) / mean * 100
            s = abs(delta) >= 5 and abs(delta) > drift * 1.5; sig += s; n_cells += 1; deltas.append(delta)
            print(f"{k[1]:>6}{k[2]:>3}{A[k]:>8.1f}{B[k]:>8.1f}{a[k]:>8.1f}{drift:>6.1f}%{delta:>+7.1f}%  {'YES' if s else '.'}")
        if deltas: print(f"   significant {sig}/{n_cells}   median delta {statistics.median(deltas):+.1f}%")
    # pooled control drift
    cs = [arms[j][1] for j in ctrl_idx]
    if len(cs) >= 2:
        drifts = []
        for k in set.intersection(*(set(c) for c in cs)):
            if k[0] != metric: continue
            v = [c[k] for c in cs]; drifts.append((max(v) - min(v)) / statistics.mean(v) * 100)
        if drifts: print(f"\n   pooled control spread over {len(cs)} boots: median {statistics.median(drifts):.1f}%  worst {max(drifts):.1f}%")
