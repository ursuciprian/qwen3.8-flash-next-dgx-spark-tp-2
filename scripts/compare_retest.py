#!/usr/bin/env python3
"""Compare a llama-benchy result table against the Aug-30 baseline for the same recipe.

benchy writes a pipe-delimited markdown table into a file named .csv, so this
parses the table rather than using the csv module.

    python3 scripts/compare_retest.py <new.csv> <baseline.csv> [out.csv]
"""
import re
import sys
from pathlib import Path

CELL = re.compile(r"^(?P<metric>\w+)(?: @ d(?P<depth>\d+))?(?: \(c(?P<conc>\d+)\))?$")


def parse(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|:"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 4 or cols[1] == "test":
            continue
        m = CELL.match(cols[1])
        if not m:
            continue
        try:
            value = float(cols[2].split("±")[0].strip())
        except ValueError:
            continue
        out[(m["metric"], int(m["depth"] or 0), int(m["conc"] or 1))] = value
    return out


new_path, base_path = sys.argv[1], sys.argv[2]
new, base = parse(new_path), parse(base_path)

rows = []
for key in sorted(set(new) | set(base), key=lambda k: (k[0], k[1], k[2])):
    metric, depth, conc = key
    if metric not in ("pp2048", "tg128"):
        continue
    b, n = base.get(key), new.get(key)
    delta = f"{(n - b) / b * 100:+.1f}%" if b and n else ""
    rows.append((metric, depth, conc, b, n, delta))

hdr = f"{'metric':<8} {'depth':>6} {'c':>2} {'baseline':>10} {'retest':>10} {'delta':>8}"
print(hdr)
print("-" * len(hdr))
for metric, depth, conc, b, n, delta in rows:
    print(f"{metric:<8} {depth:>6} {conc:>2} "
          f"{b if b is not None else float('nan'):>10.2f} "
          f"{n if n is not None else float('nan'):>10.2f} {delta:>8}")

if len(sys.argv) > 3:
    with open(sys.argv[3], "w") as fh:
        fh.write("metric,depth,concurrency,baseline_tok_s,retest_tok_s,delta_pct\n")
        for metric, depth, conc, b, n, delta in rows:
            fh.write(f"{metric},{depth},{conc},"
                     f"{'' if b is None else f'{b:.2f}'},"
                     f"{'' if n is None else f'{n:.2f}'},"
                     f"{delta.rstrip('%')}\n")
    print(f"\nwrote {sys.argv[3]}")
