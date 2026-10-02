#!/usr/bin/env python3
"""One-line numerics summary of an r9_dense_check results dir (opus-kernel-17).

  r9_dense_summary.py <results dir>
Prints weight rel err (from the snapshot's r9-report.json if given via SNAPDIR env), acceptance per
position base -> dense for each probe cell, logits drift vs self-noise (mean |dlogprob| on the common
greedy prefix, prefix agreement) and fidelity lines.
"""
import glob, json, os, re, statistics as st, sys

d = sys.argv[1]


def acc(sub, tag):
    def rd(f):
        v = {}
        for l in open(f):
            m = re.match(r'vllm:spec_decode_num_(accepted|draft)_tokens_per_pos_total\{.*position="(\d)"\} ([0-9.e+]+)', l)
            if m:
                v[(m.group(1), m.group(2))] = float(m.group(3))
        return v
    a, b = rd(f"{d}/{sub}/pos-{tag}.before"), rd(f"{d}/{sub}/pos-{tag}.after")
    g = lambda k: b.get(k, 0) - a.get(k, 0)
    dr = g(("draft", "0")) or 1
    return [g(("accepted", str(i))) / dr for i in range(4)]


def drift(p, q):
    a, b = (json.load(open(f"{d}/{x}"))["records"] for x in (p, q))
    n = tot = 0; dl = []
    for x, y in zip(a, b):
        for i, (tx, ty) in enumerate(zip(x["tokens"], y["tokens"])):
            if tx != ty:
                break
            n += 1; dl.append(abs(x["token_logprobs"][i] - y["token_logprobs"][i]))
        tot += len(x["tokens"])
    return n / tot, st.mean(dl)


for t in ("fresh-c1", "d16k-c1", "count-c1", "fresh-c4"):
    try:
        b, x = acc("base", t), acc("dense", t)
        print(f"{t:9s} acc base {'/'.join('%.3f' % v for v in b)} -> dense {'/'.join('%.3f' % v for v in x)}"
              f"  max|d| {max(abs(u - v) for u, v in zip(b, x)):.3f}")
    except OSError:
        print(t, "missing")
for name, (p, q) in {"self base": ("base/logits-a.json", "base/logits-b.json"),
                     "self dense": ("dense/logits-a.json", "dense/logits-b.json"),
                     "cross a": ("base/logits-a.json", "dense/logits-a.json"),
                     "cross b": ("base/logits-b.json", "dense/logits-b.json")}.items():
    try:
        f, m = drift(p, q); print(f"logits {name:10s} prefix agree {f:.3f} mean|dlogprob| {m:.4f}")
    except OSError:
        pass
for f in sorted(glob.glob(f"{d}/dense/fidelity*.txt")):
    for l in open(f):
        if l.startswith("depth"):
            print(os.path.basename(f), " ".join(l.split()[:8]))
print("STATE", open(f"{d}/STATE").read().strip())
