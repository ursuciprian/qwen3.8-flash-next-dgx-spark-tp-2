#!/usr/bin/env python3
"""prof_compare.py <dir_a> <dir_b> [steps]: per-kernel us/decode-step, B vs A.

Each dir holds rank*.json[.gz] chrome traces from mods/vllm-decode-profiler
(one window of `steps` decode steps). Sums CUDA kernel/memcpy/memset time per
kernel name per rank, divides by steps, averages ranks, and prints the rows
with the largest absolute per-step change first, then the total. Kernels are
keyed by their first 60 name chars (prof_summary.py's display width), so
template variants of one kernel family sum into one row.
"""
import glob
import gzip
import json
import sys
from collections import defaultdict

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def per_step(directory, steps):
    ranks = sorted(glob.glob(f"{directory}/rank*.json*"))
    if not ranks:
        sys.exit(f"no traces in {directory}")
    total = defaultdict(float)
    for path in ranks:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            events = json.load(f)["traceEvents"]
        for e in events:
            if e.get("cat") in GPU_CATS and "dur" in e:
                total[e["name"][:60]] += e["dur"] / steps / len(ranks)
    return total


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    a, b = per_step(a_dir, steps), per_step(b_dir, steps)
    rows = sorted(set(a) | set(b), key=lambda k: -abs(b.get(k, 0) - a.get(k, 0)))
    print(f"{'kernel':60} {'A us/step':>10} {'B us/step':>10} {'B-A':>9}")
    for k in rows[:40]:
        print(f"{k:60} {a.get(k, 0):10.1f} {b.get(k, 0):10.1f} {b.get(k, 0) - a.get(k, 0):+9.1f}")
    ta, tb = sum(a.values()), sum(b.values())
    print(f"{'TOTAL GPU time per step':60} {ta:10.1f} {tb:10.1f} {tb - ta:+9.1f} ({(tb / ta - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
