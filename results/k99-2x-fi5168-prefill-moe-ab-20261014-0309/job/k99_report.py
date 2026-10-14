#!/usr/bin/env python3
"""k99 screen report (stdlib): FlashInfer-prefill arm vs the shipped 2x, counted boots <dir>/{ctl,arm}-pN/.

Per boot: cold.json (k97/cold.py, cold 16K/64K/128K x reps), task.csv (llama-benchy pp2048 + tg512 c1/c4/c8),
pos-c8.before/.after (spec-decode counters around the fresh-c8 T=0 probe), serve.log (rank 0) + serve-node1.log (rank 1).
  TTFT     mean per boot; side mean over boots; noise = the larger boot-to-boot spread of either side (floor 0.5%)
  decode   benchy cells, arm vs ctl side means; band = max(k92 noise band of the cell, either side's spread)
  accept   MTP acceptance per draft position, pooled per side: |arm - ctl| <= 0.03 (numerics canary)
  path     arm boots: 'K99FI calls=1000' on both ranks (serve-time routing, not only boot); control boots: no K99 line
  KV       GPU KV cache tokens and 'Available KV cache memory' per boot: the arm's memory cost (reported, not judged)
RULE: EMPTY (a counted boot measured nothing: missing cells, < 2 boots per side) | FAIL (path wrong, a TTFT cell or a
decode cell worse beyond noise, acceptance off) | PASS (64K or 128K TTFT better beyond noise) | FLAT.
Usage: k99_report.py <screen dir> [reps] | selftest
"""
import glob
import json
import os
import re
import statistics
import sys

LENS = ("16384", "65536", "131072")
BANDS = {"pp2048 (c1)": 2.73, "tg512 (c1)": 11.94, "tg512 (c4)": 8.16, "tg512 (c8)": 5.15}   # k92 noise bands, %
FLOOR = 0.5
MIN_CALLS = 1000   # 'K99FI calls=1000' is printed once the 1000th call happens


def rd(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


def boot(d, reps):
    b = {"name": os.path.basename(d), "ttft": {}, "bench": {}, "acc": None, "fi": [], "kv": None, "kvgib": None, "bad": []}
    if os.path.exists(os.path.join(d, "FAILED")) or os.path.exists(os.path.join(d, "COLD")):
        b["bad"].append("boot failed or cold")
    try:
        s = json.load(open(os.path.join(d, "cold.json")))["summary"]
    except (OSError, ValueError, KeyError):
        s = {}
    for n in LENS:
        v = s.get(n, {})
        if v.get("n") == reps:
            b["ttft"][n] = v["ttft_mean_s"]
        else:
            b["bad"].append(f"cold {int(n) // 1024}K n={v.get('n', 0)}/{reps}")
    for line in rd(os.path.join(d, "task.csv")).splitlines():
        f = [x.strip() for x in line.split("|")]
        if len(f) >= 4 and f[2] in BANDS:
            try:
                b["bench"][f[2]] = float(f[3].split()[0])
            except ValueError:
                pass
    if len(b["bench"]) < len(BANDS):
        b["bad"].append(f"benchy cells {len(b['bench'])}/{len(BANDS)}")
    b["acc"] = acc(rd(os.path.join(d, "pos-c8.before")), rd(os.path.join(d, "pos-c8.after")))
    if not b["acc"]:
        b["bad"].append("no acceptance counters")
    for f in ("serve.log", "serve-node1.log"):
        t = rd(os.path.join(d, f))
        c = [int(x) for x in re.findall(r"K99FI calls=(\d+)", t)]
        b["fi"].append((max(c) if c else 0, "K99" in t))
    t = rd(os.path.join(d, "serve.log"))
    m = re.search(r"GPU KV cache size: ([0-9,]+) tokens", t)
    b["kv"] = int(m.group(1).replace(",", "")) if m else None
    m = re.search(r"Available KV cache memory: ([0-9.]+) GiB", t)
    b["kvgib"] = float(m.group(1)) if m else None
    return b


def acc(before, after):
    """per-position (accepted, drafted) deltas from two scrapes of the spec-decode counters"""
    def parse(t):
        c = {"accepted": {}, "draft": {}}
        for m in re.finditer(r'vllm:spec_decode_num_(accepted|draft)_tokens_per_pos_total\{.*position="(\d+)"\} ([0-9.e+]+)', t):
            c[m[1]][int(m[2])] = c[m[1]].get(int(m[2]), 0) + float(m[3])
        return c
    a, b = parse(before), parse(after)
    out = {k: (b["accepted"].get(k, 0) - a["accepted"].get(k, 0), b["draft"][k] - a["draft"].get(k, 0)) for k in b["draft"]}
    return out if out and all(d > 0 for _, d in out.values()) else None


def spread(v):
    return 100 * (max(v) - min(v)) / statistics.mean(v) if len(v) > 1 else 0.0


def report(d, reps=3):
    boots = {w: [boot(p, reps) for p in sorted(glob.glob(os.path.join(d, f"{w}-p[0-9]")))] for w in ("ctl", "arm")}
    L, empty, fail, better = [], [], [], []
    for w, bs in boots.items():
        if len(bs) < 2:
            empty.append(f"{len(bs)} {w} boots")
        for b in bs:
            empty += [f"{b['name']}: {x}" for x in b["bad"]]
    L.append(f"== cold TTFT (s, mean of {reps} per boot); noise = larger boot-to-boot spread of either side (floor {FLOOR}%)")
    for n in LENS:
        c = [b["ttft"][n] for b in boots["ctl"] if n in b["ttft"]]
        a = [b["ttft"][n] for b in boots["arm"] if n in b["ttft"]]
        if len(c) < 2 or len(a) < 2:
            L.append(f"  {int(n) // 1024}K: too few boots ({len(c)}/{len(a)})"); continue
        dd, nz = 100 * (statistics.mean(a) / statistics.mean(c) - 1), max(spread(c), spread(a), FLOOR)
        v = "better" if dd < -nz else "WORSE" if dd > nz else "flat"
        L.append(f"  {int(n) // 1024}K: ctl {' '.join(f'{x:.2f}' for x in c)} | arm {' '.join(f'{x:.2f}' for x in a)} -> "
                 f"{dd:+.1f}% ({statistics.mean(a) - statistics.mean(c):+.2f} s; noise {nz:.1f}%) {v}")
        if v == "WORSE":
            fail.append(f"TTFT {int(n) // 1024}K worse")
        if v == "better" and n != "16384":
            better.append(f"{int(n) // 1024}K")
    L.append("== llama-benchy pp2048 + tg512 (tok/s, side means; band = max(k92 band, side spread))")
    for cell, band in BANDS.items():
        c = [b["bench"][cell] for b in boots["ctl"] if cell in b["bench"]]
        a = [b["bench"][cell] for b in boots["arm"] if cell in b["bench"]]
        if not c or not a:
            L.append(f"  {cell}: missing"); continue
        dd, bd = 100 * (statistics.mean(a) / statistics.mean(c) - 1), max(band, spread(c), spread(a))
        worse = dd < -bd
        L.append(f"  {cell}: ctl {statistics.mean(c):.1f} arm {statistics.mean(a):.1f} {dd:+.1f}% (band {bd:.1f}%){' WORSE' if worse else ''}")
        if worse:
            fail.append(f"{cell} worse")
    L.append("== MTP acceptance per draft position (fresh c8, T=0), pooled per side")
    pooled = {}
    for w, bs in boots.items():
        t = {}
        for b in bs:
            for k, (x, y) in (b["acc"] or {}).items():
                t[k] = (t.get(k, (0, 0))[0] + x, t.get(k, (0, 0))[1] + y)
        pooled[w] = [t[k][0] / t[k][1] for k in sorted(t)]
        L.append(f"  {w}: {' '.join(f'{x:.3f}' for x in pooled[w]) or 'none'}")
    if pooled["ctl"] and pooled["arm"] and (len(pooled["ctl"]) != len(pooled["arm"])
                                            or any(abs(x - y) > 0.03 for x, y in zip(pooled["ctl"], pooled["arm"]))):
        fail.append("acceptance moved > 0.03")
    L.append("== path: K99FI calls (rank 0 / rank 1) and KV pool per boot")
    for w, bs in boots.items():
        for b in bs:
            L.append(f"  {b['name']}: FlashInfer calls {b['fi'][0][0]} / {b['fi'][1][0]}; KV {b['kv']} tokens, "
                     f"{b['kvgib']} GiB available for KV")
            # boot alone (profile/warm-up dummy runs at 8192 rows) logs ~100 calls; served cold prompts log thousands
            if w == "arm" and not all(n >= MIN_CALLS for n, _ in b["fi"]):
                fail.append(f"{b['name']}: fewer than {MIN_CALLS} FlashInfer calls on a rank (boot-time calls only?)")
            if w == "ctl" and any(k for _, k in b["fi"]):
                fail.append(f"{b['name']}: the control logged K99 lines")
    kc = [b["kv"] for b in boots["ctl"] if b["kv"]]
    ka = [b["kv"] for b in boots["arm"] if b["kv"]]
    gc = [b["kvgib"] for b in boots["ctl"] if b["kvgib"]]
    ga = [b["kvgib"] for b in boots["arm"] if b["kvgib"]]
    if kc and ka:
        L.append(f"  KV pool arm vs ctl: {100 * (statistics.mean(ka) / statistics.mean(kc) - 1):+.1f}% tokens"
                 + (f", {statistics.mean(ga) - statistics.mean(gc):+.2f} GiB (the FlashInfer path's memory cost per rank)" if gc and ga else ""))
    if empty:
        rule = "EMPTY"
    elif fail:
        rule = "FAIL"
    else:
        rule = "PASS" if better else "FLAT"
    for x in empty:
        L.append(f"empty: {x}")
    for x in fail:
        L.append(f"fail: {x}")
    L.append(f"better: {', '.join(better) or 'none'}")
    L.append(f"RULE={rule}")
    return "\n".join(L)


def selftest():
    import tempfile
    t = tempfile.mkdtemp()
    pos = lambda a: "".join(f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="{k}"}} {a[k] * 1000}\n'
                            f'vllm:spec_decode_num_draft_tokens_per_pos_total{{position="{k}"}} 1000\n' for k in range(4))
    def mk(name, ttft, fi, acc0=0.85, tg8=190.0):
        d = os.path.join(t, name); os.makedirs(d)
        json.dump({"summary": {n: {"n": 3, "ttft_mean_s": v} for n, v in zip(LENS, ttft)}}, open(os.path.join(d, "cold.json"), "w"))
        open(os.path.join(d, "task.csv"), "w").write("".join(f"| m | {c} | {v} ± 1 |\n" for c, v in
                                                             (("pp2048 (c1)", 2800), ("tg512 (c1)", 84), ("tg512 (c4)", 160), ("tg512 (c8)", tg8))))
        open(os.path.join(d, "pos-c8.before"), "w").write(pos([0, 0, 0, 0]).replace(" 1000\n", " 0\n"))
        open(os.path.join(d, "pos-c8.after"), "w").write(pos([acc0, 0.70, 0.57, 0.47]))
        log = "GPU KV cache size: 3,526,557 tokens\nAvailable KV cache memory: 29.31 GiB\n" if not fi else \
              "K99MOD active\nK99FI calls=1000 M=5728\nGPU KV cache size: 3,300,000 tokens\nAvailable KV cache memory: 27.40 GiB\n"
        for f in ("serve.log", "serve-node1.log"):
            open(os.path.join(d, f), "w").write(log)
    mk("ctl-p1", (5.60, 24.6, 55.0), 0); mk("ctl-p2", (5.62, 24.7, 55.2), 0)
    mk("arm-p1", (5.50, 24.0, 53.8), 1); mk("arm-p2", (5.49, 24.05, 53.9), 1)
    out = report(t)
    assert "RULE=PASS" in out and "64K" in out.split("better:")[1], out
    assert "-1.91 GiB" in out, out
    os.rename(os.path.join(t, "arm-p2", "cold.json"), os.path.join(t, "arm-p2", "cold.json.bak"))
    assert "RULE=EMPTY" in report(t)
    os.rename(os.path.join(t, "arm-p2", "cold.json.bak"), os.path.join(t, "arm-p2", "cold.json"))
    open(os.path.join(t, "arm-p2", "serve-node1.log"), "w").write("K99MOD active\n")
    assert "RULE=FAIL" in report(t) and "fewer than 1000 FlashInfer calls" in report(t)
    open(os.path.join(t, "arm-p2", "serve-node1.log"), "w").write("K99MOD active\nK99FI calls=10 M=8192\nK99FI calls=100 M=8192\n")
    assert "RULE=FAIL" in report(t), "boot-time calls alone must not pass the path check"
    print("k99_report selftest ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        selftest()
    elif len(sys.argv) >= 2:
        print(report(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3))
    else:
        sys.exit(__doc__)
