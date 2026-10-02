#!/usr/bin/env python3
"""k18 A16 boot-test verdict: per-position acceptance of the A16 arm vs the A16-off control.

    boot_check.py <control pass dir> <arm pass dir>   (exit 0 = pass)

Pass = the arm booted (no FAILED marker), every probe has a pos-*.after file, and in every
temperature-0 cell the acceptance at each draft position is >= 0.9x the control's (the screening
canary drops an arm at < 0.7x on position 0 only). Paired probes use the same prompts and offsets
on both nodes, so the cells compare one to one.
"""
import glob, os, re, sys

def acc(d, tag):
    def rd(f):
        v = {}
        for line in open(f):
            m = re.match(r'vllm:spec_decode_num_(accepted|draft)_tokens_per_pos_total\{.*position="(\d+)"\} ([0-9.e+]+)', line)
            if m:
                v[(m.group(1), int(m.group(2)))] = float(m.group(3))
        return v
    a, b = rd(f"{d}/pos-{tag}.before"), rd(f"{d}/pos-{tag}.after")
    dr = b.get(("draft", 0), 0) - a.get(("draft", 0), 0)
    if not dr:
        return None
    return [(b.get(("accepted", p), 0) - a.get(("accepted", p), 0)) / dr for p in range(8) if ("accepted", p) in b]

ctl, arm = sys.argv[1], sys.argv[2]
if os.path.exists(f"{arm}/FAILED") or not os.path.isdir(arm):
    print(f"FAIL: {arm} did not boot"); sys.exit(1)
bad = 0
tags = sorted(os.path.basename(f)[4:-6] for f in glob.glob(f"{ctl}/pos-*.after"))
for tag in tags:
    if tag == "warm":
        continue
    o, x = acc(ctl, tag), acc(arm, tag)
    if x is None or o is None:
        print(f"{tag}: missing ({'arm' if x is None else 'control'})"); bad += 1; continue
    worse = [p for p, (u, v) in enumerate(zip(o, x)) if v < 0.9 * u]
    bad += bool(worse)
    print(f"{tag}: control {'/'.join(f'{v:.2f}' for v in o)}  arm {'/'.join(f'{v:.2f}' for v in x)}"
          + (f"  BELOW 0.9x at {worse}" if worse else ""))
print("PASS" if not bad and tags else "FAIL", f"({bad} bad cells of {len(tags)})")
sys.exit(1 if bad or not tags else 0)
