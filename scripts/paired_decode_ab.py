#!/usr/bin/env python3
"""Paired decode A/B judge over two depth_decode_probe.py result files.

Run the probe with identical arguments on each arm (same --corpus, --depth,
--new, --repeats, --concurrency, --offset, temperature 0), so request i sees
the same prompt on both arms. Then:

  python3 paired_decode_ab.py shipped-d16k.json cand-d16k.json [--phase inf]

Per prompt it pairs:
  step_ms   median inter-chunk gap (one chunk per engine step, so this is the
            decode step time and does not depend on MTP acceptance)
  tok/step  1 + num_spec * acceptance
  tok/s     decode rate excluding stalls (> --stall-ms gaps)
and reports the mean paired delta with a 95% bootstrap CI and the win count.
step_ms is the kernel/host speed judge; tok/step says whether acceptance
moved; tok/s is their product and the noisiest of the three. Requests that
were profiled or stalled are excluded from step_ms only if --drop-stalled.

  python3 paired_decode_ab.py --selftest
"""
import argparse
import json
import random
import statistics


def load(path, phase):
    rows = json.load(open(path))["rows"]
    return {r["index"]: r[phase] for r in rows}


def boot_ci(deltas, n=4000, seed=0):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def compare(a, b, num_spec=4, drop_stalled=False):
    keys = sorted(set(a) & set(b))
    metrics = {
        "step_ms": lambda r: r["gap_ms_p50"],
        "tok_per_step": lambda r: 1 + num_spec * r["accept_rate"] if r.get("accept_rate") is not None else None,
        "tok_s": lambda r: r["decode_tps_no_stalls"],
    }
    out = {}
    for name, get in metrics.items():
        deltas = []
        for k in keys:
            if drop_stalled and name == "step_ms" and (a[k]["stalls"] or b[k]["stalls"]):
                continue
            x, y = get(a[k]), get(b[k])
            if x and y:
                deltas.append(100.0 * (y / x - 1))
        if len(deltas) < 2:
            out[name] = None
            continue
        lo, hi = boot_ci(deltas)
        out[name] = {"n": len(deltas), "mean_pct": statistics.mean(deltas), "ci95": (lo, hi),
                     "b_better": sum((d < 0) if name == "step_ms" else (d > 0) for d in deltas)}
    return out


def verdict(res, tol=1.0, min_pairs=5):
    s = res.get("step_ms")
    if not s or s["n"] < min_pairs:
        return f"insufficient pairs (need >= {min_pairs} unstalled)"
    lo, hi = s["ci95"]
    if hi < 0:
        return f"B steps faster ({s['mean_pct']:+.1f}%, CI95 excludes 0)"
    if lo > 0:
        return f"B steps slower ({s['mean_pct']:+.1f}%, CI95 excludes 0)"
    if -tol <= lo and hi <= tol:
        return f"step time equal within +-{tol}%"
    return "inconclusive: add repeats"


def selftest():
    rng = random.Random(1)
    a, b = {}, {}
    for i in range(12):
        acc = rng.uniform(0.35, 0.6)
        a[i] = {"gap_ms_p50": 46.0 + rng.gauss(0, 0.2), "accept_rate": acc, "decode_tps_no_stalls": 55, "stalls": 0}
        b[i] = {"gap_ms_p50": 45.3 + rng.gauss(0, 0.2), "accept_rate": acc, "decode_tps_no_stalls": 56, "stalls": 0}
    res = compare(a, b)
    assert res["step_ms"]["b_better"] >= 11 and res["step_ms"]["ci95"][1] < 0, res
    assert abs(res["tok_per_step"]["mean_pct"]) < 1e-9, res
    assert verdict(res).startswith("B steps faster"), verdict(res)
    assert verdict(compare(a, a)) == "step time equal within +-1.0%"
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--phase", default="inf", choices=("ctx", "inf"))
    ap.add_argument("--num-spec", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1.0, help="equal-speed band, percent")
    ap.add_argument("--drop-stalled", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    res = compare(load(args.a, args.phase), load(args.b, args.phase), args.num_spec, args.drop_stalled)
    for name, r in res.items():
        if r:
            print(f"{name:13} n={r['n']:2d}  B vs A {r['mean_pct']:+6.2f}%  "
                  f"CI95 [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  B better {r['b_better']}/{r['n']}")
    print("verdict:", verdict(res, args.tol))


if __name__ == "__main__":
    main()
