#!/usr/bin/env python3
"""Summarise a tp1_kv16k_screen.sh results dir: tg tok/s per cell, arm vs control, boot memory, preemptions.

Usage: tp1_kv16k_report.py <results dir>     (self-check: tp1_kv16k_report.py --test)
A cell regresses when arm mean < control mean by more than the noise: the larger of the two control
runs' spread and the mean of the reported per-run sds (all in % of the control mean).
"""
import glob, os, re, statistics, sys

ROW = re.compile(r"^\|\s*[^|]+\|\s*(tg512(?: @ d16384)? \(c\d+\))\s*\|\s*([\d.]+)\s*±\s*([\d.]+)")


def parse(text):
    return {m.group(1): (float(m.group(2)), float(m.group(3)))
            for m in map(ROW.match, text.splitlines()) if m}


def counter(path, name):
    try:
        return sum(float(l.split()[-1]) for l in open(path) if l.startswith(name))
    except OSError:
        return 0.0


def main(d):
    runs = {}  # arm -> list of {cell: (mean, sd)}
    for cell in sorted(glob.glob(os.path.join(d, "*-p*-dgx0*"))):
        name = os.path.basename(cell); arm = name.split("-p")[0]
        state = open(os.path.join(cell, "STATE")).read().strip() if os.path.exists(os.path.join(cell, "STATE")) else "missing"
        boot = open(os.path.join(cell, "boot.txt")).read().strip() if os.path.exists(os.path.join(cell, "boot.txt")) else "-"
        a, b = (os.path.join(cell, f"counters.{x}") for x in ("before", "after"))
        pre = counter(b, "vllm:num_preemptions") - counter(a, "vllm:num_preemptions")
        q = counter(b, "vllm:prefix_cache_queries") - counter(a, "vllm:prefix_cache_queries")
        h = counter(b, "vllm:prefix_cache_hits") - counter(a, "vllm:prefix_cache_hits")
        print(f"{name:18} {state:22} preempt {pre:5.0f}  prefix-hit {h / q if q else 0:.3f}  {boot}")
        csv = os.path.join(cell, "task.csv")
        if os.path.exists(csv) and state.startswith("DONE"):
            runs.setdefault(arm, []).append(parse(open(csv).read()))
    ctrl = runs.get("ctrl", [])
    if not ctrl:
        print("no control runs"); return
    cells = sorted({c for r in ctrl for c in r}, key=lambda c: ("@" in c, int(re.search(r"c(\d+)", c).group(1))))
    arms = [a for a in runs if a != "ctrl"]
    print(f"\n{'cell':22} {'ctrl':>8} {'noise%':>7}" + "".join(f" {a:>8} {'d%':>6}" for a in arms))
    for c in cells:
        cv = [r[c] for r in ctrl if c in r]; cm = statistics.mean(v for v, _ in cv)
        noise = max((max(v for v, _ in cv) - min(v for v, _ in cv)) / cm, statistics.mean(s for _, s in cv) / cm) * 100
        line = f"{c:22} {cm:8.1f} {noise:7.1f}"
        for a in arms:
            av = [r[c][0] for r in runs[a] if c in r]
            if not av:
                line += f" {'-':>8} {'':>6}"; continue
            am = statistics.mean(av); dp = (am / cm - 1) * 100
            line += f" {am:8.1f} {dp:+6.1f}" + ("!" if dp < -noise else " ")
        print(line)
    print("\n! = below control by more than noise")


def _test():
    t = ("| qwen3.8-flash-next |           tg512 (c8) |    120.02 ± 0.24 |    19.77 ± 3.01 |\n"
         "| qwen3.8-flash-next |  tg512 @ d16384 (c8) |     21.87 ± 1.72 |     7.60 ± 8.11 |\n"
         "| qwen3.8-flash-next |          pp2048 (c8) |  1972.70 ± 43.98 |  310.92 ± 67.05 |\n")
    r = parse(t)
    assert r == {"tg512 (c8)": (120.02, 0.24), "tg512 @ d16384 (c8)": (21.87, 1.72)}, r
    print("ok")


if __name__ == "__main__":
    _test() if sys.argv[1:] == ["--test"] else main(sys.argv[1])
