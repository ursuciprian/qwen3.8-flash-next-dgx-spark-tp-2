#!/usr/bin/env python3
"""r8 Phase 1 table: TP=1 (warm dgx-01, cold dgx-02) per-category step profile vs TP=2.

    r8_profile_table.py <r8 profile RESULTS dir> [--tp2 <TP=2 prof dir>] > profile_table.md

Reads <RESULTS>/prof/<node>/prof-<window>/breakdown-rank0.json (scripts/step_breakdown.py
--json), the host counters around each window (counters-<window>.before/.after: NVMe reads,
MemAvailable, Cached, EngineCore major faults) and the PLE reader stats lines. The TP=2
reference defaults to the b1.2-image wm2off profile (results/r5-p2d-wm2prof), rank 0.
"""

import argparse
import json
import os
import re

TP2_DEFAULT = ("/home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/"
               "r5-p2d-wm2prof/prof-r5-wm2offprof")
NODES = (("dgx01", "TP1 warm"), ("dgx02", "TP1 cold"))


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def gap(j, key):
    return "n/a" if "gaps_us" not in j else f"{j['gaps_us'].get(key, 0) / 1000:.2f}"


def counters(node_dir, window):
    a, b = (load(f"{node_dir}/counters-{window}.{x}") for x in ("before", "after"))
    if not a or not b:
        return None
    dt = max(b["t"] - a["t"], 1e-9)
    fl = (b["enginecore_majflt"] - a["enginecore_majflt"]) if None not in (
        a.get("enginecore_majflt"), b.get("enginecore_majflt")) else None
    return dict(seconds=dt, nvme_iops=(b["nvme_reads"] - a["nvme_reads"]) / dt,
                majflt_per_s=None if fl is None else fl / dt,
                mem_available=b.get("MemAvailable", 0) / 2**30, cached=b.get("Cached", 0) / 2**30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--tp2", default=TP2_DEFAULT)
    a = ap.parse_args()
    prof = os.path.join(a.results, "prof")
    windows = sorted({d[5:] for n, _ in NODES if os.path.isdir(os.path.join(prof, n))
                      for d in os.listdir(os.path.join(prof, n)) if d.startswith("prof-")})
    print(f"# r8 Phase 1 profile: {a.results}\n\nms per decode step (60-step windows, rank 0); "
          f"TP=2 = {a.tp2}\n")
    for w in windows:
        cols = [("TP2", load(f"{a.tp2}/prof-{w}/breakdown-rank0.json"))]
        cols += [(lab, load(f"{prof}/{n}/prof-{w}/breakdown-rank0.json")) for n, lab in NODES]
        cats = sorted({c for _, j in cols if j for c in j["summary_us"]},
                      key=lambda c: -max((j or {}).get("summary_us", {}).get(c, 0) for _, j in cols))
        print(f"## {w}\n\n| category | " + " | ".join(lab for lab, _ in cols) + " |")
        print("|---|" + "---:|" * len(cols))
        cell = lambda j, f: "n/a" if not j else f(j)  # noqa: E731
        print("| **step wall** | " + " | ".join(cell(j, lambda j: f"**{j['step_ms']:.2f}**") for _, j in cols) + " |")
        for c in cats:
            print(f"| {c} | " + " | ".join(cell(j, lambda j: f"{j['summary_us'].get(c, 0) / 1000:.2f}") for _, j in cols) + " |")
        print("| gap eager->target (front of target graph) | " + " | ".join(
            cell(j, lambda j: gap(j, "eager->target")) for _, j in cols) + " |")
        print("| gap eager->eager | " + " | ".join(
            cell(j, lambda j: gap(j, "eager->eager")) for _, j in cols) + " |\n")
    print("## host counters per window (whole probe, both repeats)\n")
    print("| window | node | s | NVMe reads/s | EngineCore majflt/s | MemAvailable GiB | Cached GiB |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for w in windows:
        for n, lab in NODES:
            c = counters(f"{prof}/{n}", w)
            if c:
                mf = "n/a" if c["majflt_per_s"] is None else f"{c['majflt_per_s']:.0f}"
                print(f"| {w} | {lab} | {c['seconds']:.0f} | {c['nvme_iops']:.0f} | {mf} | "
                      f"{c['mem_available']:.1f} | {c['cached']:.1f} |")
    print("\n## PLE reader (VLLM_PLE_MMAP_STATS, last 6 lines per node)\n")
    for n, lab in NODES:
        path = f"{prof}/{n}/ple_reader.txt"
        lines = [x.strip() for x in open(path)] if os.path.exists(path) else []
        print(f"{lab} ({n}):\n```")
        for line in [x for x in lines if "PLE reader:" in x][-6:]:
            print(re.sub(r"^.*PLE reader: ", "", line))
        print("```")


if __name__ == "__main__":
    main()
