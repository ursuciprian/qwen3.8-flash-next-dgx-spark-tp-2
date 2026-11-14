#!/usr/bin/env python3
"""2x screen report for k120 (2026-11-14): copy of k107fix/report2x.py; one change: the tg512 (c16) benchy cell is back
(k92 band 7.48%), so the rule covers c1..c16 (RULE.md).
Earlier: copy of k100b/report2x.py with the arm's own path check, a paired fresh-c1 probe cell, the boot-level CI guard
of k103/pool.py and a per-job set of MAIN cells.

Per boot: cold.json (k101/cold.py, cold 16K/64K/128K x reps), task.csv (llama-benchy pp2048 + tg512 c1/c4/c8/c16),
pos-c8.before/.after (spec-decode counters around the fresh-c8 T=0 probe), probe-fresh-c1.json (depth_decode_probe.py,
T=0), serve.log + serve-node1.log, env.txt + env-node1.txt (VLLM/B12X/XDG/NCCL of each rank), image.txt.
  TTFT     mean per boot; side mean over boots; noise = the larger boot-to-boot spread of either side (floor 0.5%)
  decode   benchy cells, arm vs ctl side means; band = max(k92 noise band of the cell, either side's spread)
  fresh-c1 thunderdome_report.probe_cell: pass K of the arm paired with pass K of the control (same prompts), mean
           paired decode tok/s change; noise = max(1%, control boot-to-boot change, CI half-width)
  CI guard boot-level Welch 95% interval (k103/pool.py): a benchy cell whose whole interval is below -1%, or a TTFT
           length whose whole interval is above +1%, is worse even inside its band
  accept   MTP acceptance per draft position, pooled per side: |arm - ctl| <= 0.03 (numerics canary)
  path     every arm boot carries each REPORT_ENV pair on both ranks and runs REPORT_ARM_IMG; no control boot carries
           an arm pair and every control boot runs REPORT_CTL_IMG
  KV       GPU KV cache tokens and 'Available KV cache memory' per boot (reported, not judged)
RULE2: EMPTY (a counted boot measured nothing: missing cells, < 2 boots per side, arm path not engaged) | LOSS (a
control on the arm path, a cell worse beyond noise or by the CI guard, acceptance off) | WIN (a REPORT_MAIN cell better
beyond noise) | FLAT. REPORT_MAIN: cells separated by '|', e.g. 'tg512 (c1)|fresh-c1' or 'pp2048 (c1)|16K'; empty =
no cell can make WIN (a correctness arm: FLAT is the good outcome).
k120 review additions: the image check needs one image line per rank (a missing rank-1 line fails); marker (BF16 path
proof from the prebake serve logs of both ranks), testgate (GPU test regressions incl. a crash or a missing test on the
arm) and verdict (RULE.md 7) are subcommands so run.sh and the selftest share one implementation.
Usage: report2x.py <screen dir> [reps] | marker <rank0 log> <rank1 log> | testgate <res> <suite>... |
       verdict T2 PATH PONG IDENT IDENTC GATE | selftest
"""
import glob
import json
import math
import os
import re
import statistics
import sys

LENS = ("16384", "65536", "131072")
BANDS = {"pp2048 (c1)": 2.73, "tg512 (c1)": 11.94, "tg512 (c4)": 8.16, "tg512 (c8)": 5.15, "tg512 (c16)": 7.48}   # k92 noise bands, %
FLOOR = 0.5
CI_FLOOR = 1.0   # %: the CI guard
T975 = (12.707, 4.303, 3.183, 2.777, 2.571, 2.447, 2.365, 2.307, 2.263, 2.229,
        2.201, 2.179, 2.161, 2.145, 2.132, 2.120, 2.110, 2.101, 2.094, 2.086)
TD = os.environ.get("TD_SCRIPTS", "/home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/scripts")


def rd(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


def pairs(s):
    return [tuple(x.split("=", 1)) for x in (s or "").split()]


def boot(d, reps):
    b = {"name": os.path.basename(d), "dir": d, "ttft": {}, "bench": {}, "acc": None, "kv": None, "kvgib": None, "bad": [],
         "env": [rd(os.path.join(d, "env.txt")), rd(os.path.join(d, "env-node1.txt"))], "image": rd(os.path.join(d, "image.txt"))}
    if any(os.path.exists(os.path.join(d, m)) for m in ("FAILED", "COLD", "DIED")):
        b["bad"].append("boot failed, cold or died")
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
    if not os.path.exists(os.path.join(d, "probe-fresh-c1.json")):
        b["bad"].append("no fresh-c1 probe")
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


def welch(c, a):
    """(delta %, lo %, hi %) of mean(a) vs mean(c), 95% Welch CI, in % of mean(c) (k103/pool.py)."""
    mc, ma = statistics.mean(c), statistics.mean(a)
    vc, va = statistics.variance(c) / len(c), statistics.variance(a) / len(a)
    se = math.sqrt(vc + va)
    df = (vc + va) ** 2 / ((vc ** 2 / (len(c) - 1)) + (va ** 2 / (len(a) - 1))) if se else 1
    t = T975[min(len(T975), max(1, int(df))) - 1]
    d = ma - mc
    return d / mc * 100, (d - t * se) / mc * 100, (d + t * se) / mc * 100


def fresh_c1(boots):
    """thunderdome_report.probe_cell on {K: dir} of both sides -> (delta %, noise %, info) | str | None"""
    sys.path.insert(0, TD)
    import thunderdome_report as tdr
    side = {w: {int(b["name"][-1]): b["dir"] for b in bs if not b["bad"] or b["bad"] == ["no fresh-c1 probe"]}
            for w, bs in boots.items()}
    return tdr.probe_cell(side["ctl"], side["arm"], "fresh-c1")


def report(d, reps=3, main=None, envs=None, arm_img=None, ctl_img=None):
    main = [x for x in (os.environ.get("REPORT_MAIN", "") if main is None else main).split("|") if x]
    envs = pairs(os.environ.get("REPORT_ENV", "") if envs is None else envs)
    arm_img = os.environ.get("REPORT_ARM_IMG", "") if arm_img is None else arm_img
    ctl_img = os.environ.get("REPORT_CTL_IMG", "") if ctl_img is None else ctl_img
    boots = {w: [boot(p, reps) for p in sorted(glob.glob(os.path.join(d, f"{w}-p[0-9]")))] for w in ("ctl", "arm")}
    L, empty, fail, better = [], [], [], []
    for w, bs in boots.items():
        if len(bs) < 2:
            empty.append(f"{len(bs)} {w} boots")
        for b in bs:
            empty += [f"{b['name']}: {x}" for x in b["bad"]]
    L.append(f"main cells (can make WIN): {', '.join(main) or 'none (correctness arm)'}")
    L.append(f"== cold TTFT (s, mean of {reps} per boot); noise = larger boot-to-boot spread of either side (floor {FLOOR}%); "
             f"CI guard: whole boot-level 95% CI above +{CI_FLOOR:.0f}% is worse")
    for n in LENS:
        c = [b["ttft"][n] for b in boots["ctl"] if n in b["ttft"]]
        a = [b["ttft"][n] for b in boots["arm"] if n in b["ttft"]]
        lab = f"{int(n) // 1024}K"
        if len(c) < 2 or len(a) < 2:
            L.append(f"  {lab}: too few boots ({len(c)}/{len(a)})"); continue
        dd, nz = 100 * (statistics.mean(a) / statistics.mean(c) - 1), max(spread(c), spread(a), FLOOR)
        _, lo, hi = welch(c, a)
        v = "better" if dd < -nz else "WORSE" if dd > nz else "flat"
        if v != "WORSE" and lo > CI_FLOOR:
            v = "WORSE (CI guard)"
        L.append(f"  {lab}: ctl {' '.join(f'{x:.2f}' for x in c)} | arm {' '.join(f'{x:.2f}' for x in a)} -> "
                 f"{dd:+.1f}% ({statistics.mean(a) - statistics.mean(c):+.2f} s; noise {nz:.1f}%; CI [{lo:+.1f},{hi:+.1f}]) {v}")
        if v.startswith("WORSE"):
            fail.append(f"TTFT {lab} worse")
        if v == "better":
            better.append(lab)
    L.append("== llama-benchy pp2048 + tg512 (tok/s, side means; band = max(k92 band, side spread); "
             f"CI guard: whole boot-level 95% CI below -{CI_FLOOR:.0f}% is worse)")
    for cell, band in BANDS.items():
        c = [b["bench"][cell] for b in boots["ctl"] if cell in b["bench"]]
        a = [b["bench"][cell] for b in boots["arm"] if cell in b["bench"]]
        if len(c) < 2 or len(a) < 2:
            L.append(f"  {cell}: too few boots ({len(c)}/{len(a)})"); continue
        dd, bd = 100 * (statistics.mean(a) / statistics.mean(c) - 1), max(band, spread(c), spread(a))
        _, lo, hi = welch(c, a)
        v = "WORSE" if dd < -bd else "better" if dd > bd else ""
        if v != "WORSE" and hi < -CI_FLOOR:
            v = "WORSE (CI guard)"
        L.append(f"  {cell}: ctl {statistics.mean(c):.1f} arm {statistics.mean(a):.1f} {dd:+.1f}% (band {bd:.1f}%; "
                 f"CI [{lo:+.2f},{hi:+.2f}]) {v}".rstrip())
        if v.startswith("WORSE"):
            fail.append(f"{cell} worse")
        if v == "better":
            better.append(cell)
    L.append("== fresh-c1 T=0 probe, pass K of the arm paired with pass K of the control (decode tok/s, stalls excluded)")
    try:
        r = fresh_c1(boots)
    except Exception as e:  # noqa: BLE001 - a broken pairing is a missing cell, never a pass
        r = f"pairing failed: {type(e).__name__}: {e}"
    if isinstance(r, tuple):
        dd, nz, info = r
        v = "better" if dd > nz else "WORSE" if dd < -nz else "flat"
        L.append(f"  fresh-c1: {dd:+.2f}% (noise {nz:.2f}%; {info}) {v}")
        if v == "WORSE":
            fail.append("fresh-c1 worse")
        if v == "better":
            better.append("fresh-c1")
    else:
        L.append(f"  fresh-c1: missing ({r or 'no paired data'})")
        empty.append(f"fresh-c1: {r or 'no paired data'}")
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
    L.append(f"== path: arm pairs {' '.join(f'{k}={v}' for k, v in envs) or 'none'} on both ranks, images; KV pool per boot")
    for w, bs in boots.items():
        for b in bs:
            have = [all(re.search(rf"^{re.escape(k)}={re.escape(v)}$", e, re.M) for e in b["env"]) for k, v in envs]
            anyp = [any(re.search(rf"^{re.escape(k)}={re.escape(v)}$", e, re.M) for e in b["env"]) for k, v in envs]
            imgs = b["image"].split()
            # one "<Config.Image> <Id>" line per rank; a missing rank-1 line fails
            img_ok = len(imgs[0::2]) == 2 and all(x == (arm_img if w == "arm" else ctl_img) for x in imgs[0::2]) if (arm_img or ctl_img) else True
            L.append(f"  {b['name']}: pairs {sum(have)}/{len(envs)} on both ranks; image {'ok' if img_ok else 'WRONG ' + ' '.join(imgs[0::2])}; "
                     f"KV {b['kv']} tokens, {b['kvgib']} GiB available for KV")
            if w == "arm" and (not all(have) or not img_ok):
                empty.append(f"{b['name']}: arm path not engaged (pairs {sum(have)}/{len(envs)}, image {'ok' if img_ok else 'wrong'})")
            if w == "ctl" and (any(anyp) or not img_ok):
                fail.append(f"{b['name']}: the control ran with an arm variable or the wrong image")
    kc = [b["kv"] for b in boots["ctl"] if b["kv"]]
    ka = [b["kv"] for b in boots["arm"] if b["kv"]]
    gc = [b["kvgib"] for b in boots["ctl"] if b["kvgib"]]
    ga = [b["kvgib"] for b in boots["arm"] if b["kvgib"]]
    if kc and ka:
        L.append(f"  KV pool arm vs ctl: {100 * (statistics.mean(ka) / statistics.mean(kc) - 1):+.1f}% tokens"
                 + (f", {statistics.mean(ga) - statistics.mean(gc):+.2f} GiB per rank" if gc and ga else ""))
    main_better = [x for x in better if x in main]
    if empty:
        rule = "EMPTY"
    elif fail:
        rule = "LOSS"
    else:
        rule = "WIN" if main_better else "FLAT"
    for x in empty:
        L.append(f"empty: {x}")
    for x in fail:
        L.append(f"fail: {x}")
    L.append(f"better: {', '.join(better) or 'none'}; in the main cells: {', '.join(main_better) or 'none'}")
    L.append(f"RULE2={rule}")
    return "\n".join(L)


MARK_RUN = re.compile(r"k120 marker: BF16 path ran n=(\d+) k=(\d+) rows=(\d+)")
MARK_PLAN = re.compile(r"k120 marker: plan .*")


def marker(logs):
    """serve logs of rank 0 and rank 1 of the prebake boot (mods/b12x-a16-marker) -> ("ENGAGED"|"NOT-ENGAGED", lines).
    ENGAGED = every rank logged a BF16-path call on a narrow-N (<= 512), long-K (>= 8192) weight with >= 1264 rows."""
    lines, ok = [], True
    for r, p in enumerate(logs):
        t = rd(p)
        runs = [(int(n), int(k), int(m)) for n, k, m in MARK_RUN.findall(t)]
        hit = [x for x in runs if x[0] <= 512 and x[1] >= 8192 and x[2] >= 1264]
        plans = sorted(set(MARK_PLAN.findall(t)))
        lines.append(f"  rank {r}: {len(plans)} plan decisions logged ({sum('engaged=True' in x for x in plans)} engaged), "
                     f"BF16 path calls {' '.join(f'n{n}/k{k}/rows{m}' for n, k, m in runs) or 'none'}"
                     + ("" if t else " (no serve log)"))
        lines += [f"    {x[len('k120 marker: '):]}" for x in plans[:8]]
        ok = ok and bool(hit)
    if len(logs) != 2:
        ok = False
    return ("ENGAGED" if ok else "NOT-ENGAGED"), lines


def outcomes(p):
    """'<test id> <STATUS>' lines -> {id: status}"""
    d = {}
    for line in rd(p).splitlines():
        f = line.split()
        if len(f) == 2:
            d[f[0]] = f[1]
    return d


def testgate(res, names):
    """regressions of the GPU test gate: a test that passes on the control and fails, errors or does not run on the
    arm, or a suite whose control run exits 0 while the arm run exits non-zero (crash, 124 = timeout)."""
    reg = []
    for b in names:
        c, a = outcomes(os.path.join(res, f"tests-ctl-{b}.outcomes")), outcomes(os.path.join(res, f"tests-arm-{b}.outcomes"))
        rc = {s: rd(os.path.join(res, f"tests-{s}-{b}.rc")).strip() or "missing" for s in ("ctl", "arm")}
        for tid, st in sorted(c.items()):
            if st != "PASSED":
                continue
            if a.get(tid) in ("FAILED", "ERROR"):
                reg.append(f"{tid} {a[tid]} on the arm")
            elif tid not in a:
                reg.append(f"{tid} not run on the arm")
        if rc["ctl"] == "0" and rc["arm"] != "0":
            reg.append(f"{b}: arm exit {rc['arm']}, control exit 0")
    return reg


def verdict(t2, path, pong, ident, identc, gate):
    """RULE.md 6 -> (VERDICT, reason). Quality failure or LOSS = REJECT; path not engaged or anything missing = EMPTY."""
    if pong == "FAIL":
        return "REJECT", "prebake pong failed"
    if path != "ENGAGED":
        return "EMPTY", "path not engaged" if path == "NOT-ENGAGED" else "path check missing"
    bad = [x for x, b in (("IDENTC WORSE", identc == "WORSE"), ("IDENT FAIL (short prompts)", ident == "FAIL"),
                          ("gate " + gate, gate.startswith("FAIL"))) if b]
    if t2 == "LOSS" or bad:
        return "REJECT", "; ".join((["RULE2 LOSS"] if t2 == "LOSS" else []) + bad)
    miss = [x for x, b in ((f"RULE2 {t2}", t2 not in ("WIN", "FLAT")), (f"pong {pong}", pong != "OK"),
                           (f"IDENT {ident}", ident not in ("PASS", "UNSTABLE")), (f"IDENTC {identc}", identc != "OK"),
                           (f"gate {gate}", gate != "PASS")) if b]
    if miss:
        return "EMPTY", "; ".join(miss)
    return ("SHIP-CANDIDATE" if t2 == "WIN" else "FLAT"), ""


def selftest_gates():
    import tempfile
    t = tempfile.mkdtemp()
    # path marker: both ranks engaged; rank 1 only plan lines -> NOT-ENGAGED; one log -> NOT-ENGAGED
    run = "k120 marker: plan recipe=mxfp8 k=10240 n=336 expected_m=None sm=48 engaged=True\n" \
          "k120 marker: BF16 path ran n=336 k=10240 rows=8192\n"
    for f, s in (("r0.log", run), ("r1.log", run), ("r1b.log", run.split("\n")[0] + "\n"), ("r1c.log", "")):
        open(os.path.join(t, f), "w").write(s)
    p = lambda *f: [os.path.join(t, x) for x in f]
    assert marker(p("r0.log", "r1.log"))[0] == "ENGAGED"
    assert marker(p("r0.log", "r1b.log"))[0] == "NOT-ENGAGED"
    assert marker(p("r0.log", "r1c.log"))[0] == "NOT-ENGAGED"
    assert marker(p("r0.log"))[0] == "NOT-ENGAGED"
    open(os.path.join(t, "r1d.log"), "w").write("k120 marker: BF16 path ran n=336 k=10240 rows=1000\n")
    assert marker(p("r0.log", "r1d.log"))[0] == "NOT-ENGAGED", "rows below MIN_ROWS do not prove the path"
    # test gate
    def put(b, side, lines, rc):
        open(os.path.join(t, f"tests-{side}-{b}.outcomes"), "w").write("".join(f"{x}\n" for x in lines))
        open(os.path.join(t, f"tests-{side}-{b}.rc"), "w").write(f"{rc}\n")
    put("a", "ctl", ["t::x PASSED", "t::y PASSED"], 0); put("a", "arm", ["t::x PASSED", "t::y PASSED"], 0)
    assert testgate(t, ["a"]) == []
    put("a", "arm", ["t::x PASSED", "t::y FAILED"], 1)
    assert testgate(t, ["a"]) == ["t::y FAILED on the arm", "a: arm exit 1, control exit 0"]
    # arm crash after the control passed: the crashed tests never report, rc 139
    put("a", "arm", ["t::x PASSED"], 139)
    r = testgate(t, ["a"])
    assert r == ["t::y not run on the arm", "a: arm exit 139, control exit 0"], r
    # control fails too (new module): not a regression
    put("n", "ctl", ["t::z ERROR"], 2); put("n", "arm", ["t::z PASSED"], 0)
    assert testgate(t, ["n"]) == []
    # verdict mapping (RULE.md 6)
    assert verdict("WIN", "ENGAGED", "OK", "PASS", "OK", "PASS")[0] == "SHIP-CANDIDATE"
    assert verdict("FLAT", "ENGAGED", "OK", "UNSTABLE", "OK", "PASS")[0] == "FLAT"
    assert verdict("FLAT", "NOT-ENGAGED", "OK", "PASS", "OK", "PASS") == ("EMPTY", "path not engaged")
    assert verdict("EMPTY", "NOT-ENGAGED", "OK", "none", "none", "none") == ("EMPTY", "path not engaged")
    assert verdict("FAILED", "none", "FAIL", "none", "none", "none")[0] == "REJECT"
    assert verdict("LOSS", "ENGAGED", "OK", "EMPTY", "EMPTY", "none")[0] == "REJECT", "LOSS with missing identity"
    assert verdict("WIN", "ENGAGED", "OK", "PASS", "OK", "FAIL (hardmode [80])")[0] == "REJECT"
    assert verdict("WIN", "ENGAGED", "OK", "FAIL", "OK", "PASS")[0] == "REJECT"
    assert verdict("WIN", "ENGAGED", "OK", "PASS", "WORSE", "none")[0] == "REJECT"
    assert verdict("WIN", "ENGAGED", "OK", "PASS", "OK", "none")[0] == "EMPTY"
    assert verdict("EMPTY", "ENGAGED", "OK", "PASS", "OK", "none")[0] == "EMPTY"


def selftest():
    import tempfile
    selftest_gates()
    t = tempfile.mkdtemp()
    pos = lambda a: "".join(f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="{k}"}} {a[k] * 1000}\n'
                            f'vllm:spec_decode_num_draft_tokens_per_pos_total{{position="{k}"}} 1000\n' for k in range(4))
    ENV = "NCCL_BUFFSIZE=1048576 NCCL_MAX_CTAS=8"
    ARM, CTL = "img:arm", "img:ctl"

    def mk(name, ttft, tg1=84.0, tps=80.0, acc0=0.85, arm=False):
        d = os.path.join(t, name); os.makedirs(d)
        json.dump({"summary": {n: {"n": 3, "ttft_mean_s": v} for n, v in zip(LENS, ttft)}}, open(os.path.join(d, "cold.json"), "w"))
        open(os.path.join(d, "task.csv"), "w").write("".join(f"| m | {c} | {v} ± 1 |\n" for c, v in
                                                             (("pp2048 (c1)", 2800), ("tg512 (c1)", tg1), ("tg512 (c4)", 160), ("tg512 (c8)", 190), ("tg512 (c16)", 235))))
        open(os.path.join(d, "pos-c8.before"), "w").write(pos([0, 0, 0, 0]).replace(" 1000\n", " 0\n"))
        open(os.path.join(d, "pos-c8.after"), "w").write(pos([acc0, 0.70, 0.57, 0.47]))
        rows = [{"index": i, "inf": {"gap_ms_p50": 40.0 * 80 / tps, "accept_rate": 0.65, "decode_tps_no_stalls": tps + 0.1 * i,
                                     "stalls": 0}} for i in range(4)]
        json.dump({"rows": rows}, open(os.path.join(d, "probe-fresh-c1.json"), "w"))
        env = "".join(f"{p}\n" for p in ENV.split()) if arm else "VLLM_GDN_PREFILL_VIEWS=1\n"
        for f in ("env.txt", "env-node1.txt"):
            open(os.path.join(d, f), "w").write(env)
        open(os.path.join(d, "image.txt"), "w").write(f"{ARM if arm else CTL} sha256:x\n{ARM if arm else CTL} sha256:x\n")
        open(os.path.join(d, "serve.log"), "w").write("GPU KV cache size: 3,526,557 tokens\nAvailable KV cache memory: 29.31 GiB\n")
    for k in (1, 2, 3, 4):
        mk(f"ctl-p{k}", (5.60 + 0.01 * k, 24.6, 55.0), tps=80.0 + 0.2 * k)
        mk(f"arm-p{k}", (5.20 + 0.01 * k, 24.0, 53.8), tps=80.1 + 0.2 * k, arm=True)
    run = lambda main: report(t, 3, main, ENV, ARM, CTL)
    out = run("pp2048 (c1)|16K")
    assert "RULE2=WIN" in out and "in the main cells: 16K" in out, out
    assert "fresh-c1: " in out and "missing" not in out.split("== fresh-c1")[1].split("==")[0], out
    assert "RULE2=FLAT" in run("tg512 (c1)|fresh-c1"), "16K better is not a decode main cell"
    assert "RULE2=FLAT" in run(""), "no main cells: a clean correctness arm is FLAT"
    # CI guard: tg512 c1 inside its 11.94% band but the whole boot CI below -1%
    for k, v in zip((1, 2, 3, 4), (79.0, 78.6, 79.4, 79.1)):
        open(os.path.join(t, f"arm-p{k}", "task.csv"), "w").write("".join(f"| m | {c} | {x} ± 1 |\n" for c, x in
            (("pp2048 (c1)", 2800), ("tg512 (c1)", v), ("tg512 (c4)", 160), ("tg512 (c8)", 190), ("tg512 (c16)", 235))))
    o = run("pp2048 (c1)|16K")
    assert "RULE2=LOSS" in o and "tg512 (c1) worse" in o and "CI guard" in o.split("tg512 (c1):")[1].split("\n")[0], o
    for k in (1, 2, 3, 4):
        open(os.path.join(t, f"arm-p{k}", "task.csv"), "w").write("".join(f"| m | {c} | {x} ± 1 |\n" for c, x in
            (("pp2048 (c1)", 2800), ("tg512 (c1)", 84.0), ("tg512 (c4)", 160), ("tg512 (c8)", 190), ("tg512 (c16)", 235))))
    assert "RULE2=WIN" in run("16K")
    # k120: c16 is judged like the other decode cells; worse beyond its band -> LOSS
    for c16 in (200, 235):   # every arm boot 200 vs control 235 (-14.9%, band 7.48%), then back
        for k in (1, 2, 3, 4):
            open(os.path.join(t, f"arm-p{k}", "task.csv"), "w").write("".join(f"| m | {c} | {x} ± 1 |\n" for c, x in
                (("pp2048 (c1)", 2800), ("tg512 (c1)", 84.0), ("tg512 (c4)", 160), ("tg512 (c8)", 190), ("tg512 (c16)", c16))))
        if c16 == 200:
            o = run("16K")
            assert "RULE2=LOSS" in o and "tg512 (c16) worse" in o, o
    # path: an arm boot without the pair on rank 1 -> EMPTY; a control with the pair -> LOSS
    open(os.path.join(t, "arm-p2", "env-node1.txt"), "w").write("NCCL_BUFFSIZE=1048576\n")
    assert "RULE2=EMPTY" in run("16K") and "arm path not engaged" in run("16K")
    open(os.path.join(t, "arm-p2", "env-node1.txt"), "w").write("NCCL_BUFFSIZE=1048576\nNCCL_MAX_CTAS=8\n")
    open(os.path.join(t, "ctl-p3", "env.txt"), "w").write("NCCL_MAX_CTAS=8\n")
    assert "RULE2=LOSS" in run("16K")
    open(os.path.join(t, "ctl-p3", "env.txt"), "w").write("VLLM_GDN_PREFILL_VIEWS=1\n")
    # wrong image on an arm boot -> EMPTY
    open(os.path.join(t, "arm-p1", "image.txt"), "w").write(f"{CTL} sha256:x\n{CTL} sha256:x\n")
    assert "RULE2=EMPTY" in run("16K")
    open(os.path.join(t, "arm-p1", "image.txt"), "w").write(f"{ARM} sha256:x\n{ARM} sha256:x\n")
    # rank-1 image line missing on an arm boot -> EMPTY; on a control boot -> LOSS
    open(os.path.join(t, "arm-p1", "image.txt"), "w").write(f"{ARM} sha256:x\n")
    o = run("16K")
    assert "RULE2=EMPTY" in o and "arm-p1: arm path not engaged" in o, o
    open(os.path.join(t, "arm-p1", "image.txt"), "w").write(f"{ARM} sha256:x\n{ARM} sha256:x\n")
    open(os.path.join(t, "ctl-p2", "image.txt"), "w").write(f"{CTL} sha256:x\n")
    assert "RULE2=LOSS" in run("16K")
    open(os.path.join(t, "ctl-p2", "image.txt"), "w").write(f"{CTL} sha256:x\n{CTL} sha256:x\n")
    assert "RULE2=WIN" in run("16K")
    # acceptance canary
    open(os.path.join(t, "arm-p1", "pos-c8.after"), "w").write(pos([0.70, 0.70, 0.57, 0.47]))
    assert "RULE2=LOSS" in run("16K") and "acceptance moved" in run("16K")
    open(os.path.join(t, "arm-p1", "pos-c8.after"), "w").write(pos([0.85, 0.70, 0.57, 0.47]))
    # missing fresh-c1 probe on one boot -> EMPTY
    os.rename(os.path.join(t, "arm-p4", "probe-fresh-c1.json"), os.path.join(t, "arm-p4", "x.json"))
    assert "RULE2=EMPTY" in run("16K")
    print("report2x selftest ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        selftest()
    elif sys.argv[1:2] == ["marker"]:      # marker <rank0 serve.log> <rank1 serve.log>
        v, lines = marker(sys.argv[2:])
        print("\n".join(lines)); print(f"PATH={v}")
    elif sys.argv[1:2] == ["testgate"]:    # testgate <res> <suite>...: one regression per line, nothing = clean
        print("\n".join(testgate(sys.argv[2], sys.argv[3:])))
    elif sys.argv[1:2] == ["verdict"]:     # verdict T2 PATH PONG IDENT IDENTC GATE -> "VERDICT<TAB>reason"
        print("\t".join(verdict(*sys.argv[2:8])))
    elif len(sys.argv) >= 2:
        print(report(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3))
    else:
        sys.exit(__doc__)
