"""Per-cell MTP acceptance per draft position from Thunderdome pos-<cell>.before/.after /metrics snapshots.
usage: python3 acc_by_cell.py <boot dir>...   (pools the given boots per cell)"""
import collections, re, sys, os
CELLS = ["fresh-c1", "fresh-c4", "fresh-c8", "d16k-c4", "d16k-c8", "count-c8"]
pat = re.compile(r'^vllm:spec_decode_num_(accepted|draft)_tokens_per_pos_total\{.*position="(\d)"\} ([0-9.e+]+)')
def snap(f):
    out = {}
    for line in open(f):
        m = pat.match(line)
        if m: out[m.group(1), int(m.group(2))] = float(m.group(3))
    return out
tot = collections.defaultdict(float); nb = collections.Counter()
for b in sys.argv[1:]:
    for c in CELLS:
        a, z = os.path.join(b, f"pos-{c}.before"), os.path.join(b, f"pos-{c}.after")
        if not (os.path.exists(a) and os.path.exists(z)): continue
        s0, s1 = snap(a), snap(z)
        if not s1: continue
        nb[c] += 1
        for k in s1: tot[c, k] += s1[k] - s0.get(k, 0.0)
print("cell       boots drafts/pos   acc p0  p1    p2    p3  | cond p1|p0 p2|p1 p3|p2")
for c in CELLS:
    if not nb[c]: continue
    d = tot[c, ("draft", 0)]
    acc = [tot[c, ("accepted", k)] / tot[c, ("draft", k)] for k in range(4)]
    cond = [acc[k] / acc[k - 1] for k in (1, 2, 3)]
    print(f"{c:10s} {nb[c]:5d} {d:10.0f}   " + " ".join(f"{x:.3f}" for x in acc) + " | " + "     ".join(f"{x:.3f}" for x in cond))
