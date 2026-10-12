#!/usr/bin/env python3
"""k56d: pool k56c + k56d boots of one Spark and judge the pooled cells.

Usage: pool.py <k56c node dir> <k56d node dir> <pooled dir>   |   pool.py --selftest
  pooled dir gets ctl-p1/p2, arm-p1/p2 -> k56c and ctl-p3..p6, arm-p3..p6 -> k56d p1..p4 (symlinks), then
  thunderdome_report.py runs on it (env TD_ACC_RISE_OK passes through). Adds a boot-level Welch 95% CI for the benchy
  cells (reported, the rule uses the report's noise bands) and post checks per boot.
Last line RULE=PASS (no cell worse than its band, >= 1 better, post checks ok) | FAIL (a worse cell, KILL or a post
check) | FLAT (nothing worse, nothing better).
"""
import glob, math, os, re, statistics, subprocess, sys

REPORT = os.path.expanduser("~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/scripts/thunderdome_report.py")
ROW = re.compile(r"^\|\s*[^|]+\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*±")
T975 = [12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26, 2.23, 2.20, 2.18, 2.16, 2.14, 2.13, 2.12, 2.11, 2.10, 2.09, 2.09]


def link(c56c, c56d, out):
    os.makedirs(out, exist_ok=True)
    for who in ("ctl", "arm"):
        for src, k in [(os.path.join(c56c, f"{who}-p{k}"), k) for k in (1, 2)] + \
                      [(os.path.join(c56d, f"{who}-p{k}"), k + 2) for k in (1, 2, 3, 4)]:
            dst = os.path.join(out, f"{who}-p{k}")
            if os.path.isdir(src) and not os.path.lexists(dst):
                os.symlink(src, dst)


def welch(c, a):
    """(delta %, lo %, hi %) of mean(a) vs mean(c), 95% Welch CI, in % of mean(c)."""
    mc, ma = statistics.mean(c), statistics.mean(a)
    vc, va = statistics.variance(c) / len(c), statistics.variance(a) / len(a)
    se = math.sqrt(vc + va)
    df = (vc + va) ** 2 / ((vc ** 2 / (len(c) - 1)) + (va ** 2 / (len(a) - 1))) if se else 1
    t = T975[min(len(T975), max(1, int(df))) - 1]
    d = ma - mc
    return d / mc * 100, (d - t * se) / mc * 100, (d + t * se) / mc * 100


def benchy(p):
    try:
        return {m.group(1): float(m.group(2)) for m in map(ROW.match, open(os.path.join(p, "task.csv"))) if m}
    except OSError:
        return {}


def post(p):
    """problems of one boot dir (empty list = ok), plus its KV size line."""
    bad, log = [], ""
    try:
        log = open(os.path.join(p, "serve.log"), errors="replace").read()
    except OSError:
        bad.append("no serve.log")
    if re.search(r"\b[1-9]\d* measured", log):
        bad.append("b12x plans measured")
    if "Traceback" in log:
        bad.append("Traceback in serve.log")
    try:
        if "Directly load AOT" not in open(os.path.join(p, "aot.txt")).read():
            bad.append("AOT not loaded")
    except OSError:
        bad.append("no aot.txt")
    try:
        m = min(float(x.split()[1]) for x in open(os.path.join(p, "mem.log")) if len(x.split()) == 2)
        if m < 6:
            bad.append(f"MemAvailable {m:.2f} GiB < 6")
    except (OSError, ValueError):
        bad.append("no mem.log")
    kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
    return bad, (kv[0] if kv else None)


def rule(report, problems):
    cells = [l for l in report.splitlines() if re.match(r"\s+(probe|wall|benchy)\s", l)]
    worse = [l.strip() for l in cells if re.search(r"\s(worse|WORSE>2x)\s", l)]
    better = [l.strip() for l in cells if re.search(r"\sbetter\s", l)]
    kill = "VERDICT=KILL" in report
    if worse or kill or problems:
        return "FAIL", worse, better
    return ("PASS" if better else "FLAT"), worse, better


def main(c56c, c56d, out):
    link(c56c, c56d, out)
    r = subprocess.run([sys.executable, REPORT, out], capture_output=True, text=True)
    print(r.stdout.rstrip())
    print("== boot-level Welch 95% CI, benchy cells (reported; the rule uses the noise bands above)")
    boots = {w: sorted(glob.glob(os.path.join(out, f"{w}-p[0-9]"))) for w in ("ctl", "arm")}
    for cell in ("pp2048 (c1)", "tg512 (c1)", "tg512 (c8)"):
        c = [v for v in (benchy(p).get(cell) for p in boots["ctl"]) if v]
        a = [v for v in (benchy(p).get(cell) for p in boots["arm"]) if v]
        if len(c) < 2 or len(a) < 2:
            print(f"  {cell:12s} too few boots ({len(c)}/{len(a)})")
            continue
        d, lo, hi = welch(c, a)
        print(f"  {cell:12s} {d:+6.2f}%  CI [{lo:+.2f},{hi:+.2f}]  control {' '.join(f'{x:.1f}' for x in c)} | arm {' '.join(f'{x:.1f}' for x in a)}")
    print("== post checks per boot")
    problems, kvs = [], {}
    for w in ("ctl", "arm"):
        for p in boots[w]:
            bad, kv = post(p)
            kvs[os.path.basename(p)] = kv
            print(f"  {os.path.basename(p)} -> {os.path.realpath(p).split('/results/')[-1]}: kv {kv} {'; '.join(bad) or 'ok'}")
            problems += [f"{os.path.basename(p)}: {b}" for b in bad]
    if len(set(kvs.values())) != 1:
        problems.append(f"KV sizes differ: {sorted(set(map(str, kvs.values())))}")
    v, worse, better = rule(r.stdout, problems)
    for x in problems:
        print("post:", x)
    print("worse:", "; ".join(worse) or "none")
    print("better:", "; ".join(better) or "none")
    print(f"RULE={v}")


def selftest():
    d, lo, hi = welch([100.0, 101.0, 99.0, 100.0], [95.0, 96.0, 94.0, 95.0])
    assert -6 < d < -4 and hi < 0 and lo < d < hi, (d, lo, hi)
    d, lo, hi = welch([100.0, 110.0], [104.0, 106.0])
    assert lo < 0 < hi, (d, lo, hi)
    rep = ("  benchy tg512 (c8)    -3.79%  noise  3.05%  worse     control\n"
           "  probe  fresh-c8      +4.04%  noise  1.60%  better    n=32\nVERDICT=INCONCLUSIVE\n")
    assert rule(rep, [])[0] == "FAIL"
    assert rule(rep.replace("worse ", "      "), [])[0] == "PASS"
    assert rule(rep.replace("worse ", "      ").replace("better", "      "), [])[0] == "FLAT"
    assert rule(rep.replace("worse ", "      "), ["x"])[0] == "FAIL"
    assert rule("  probe  fresh-c8  +4%  noise 1%  better  n\nVERDICT=KILL\n", [])[0] == "FAIL"
    print("pool selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    elif len(sys.argv) == 4:
        main(*sys.argv[1:])
    else:
        sys.exit(__doc__)
