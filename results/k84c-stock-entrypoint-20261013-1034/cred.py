#!/usr/bin/env python3
"""k84/k85/k86 report helper (2026-10-10, tp-2 #155). Stdlib only.

  cred.py report <RES> [--vs <other RES>]   cells.csv, boots.csv and cred.txt in <RES>
  cred.py --selftest

Reads <RES>/units.tsv (unit, setup, boot, hosts, recipe), <RES>/marks.tsv (unit, cell, start, end, rc),
<RES>/power-dgx0N.csv ("epoch,W,MHz,C" at 1 Hz) and per unit the llama-benchy json (tgdef-c1, tgdef-c8, tgt0-c1,
tgt0-c8) and coding.py json (coding-<mode>-c<N>). Power of a cell = mean W of the samples inside its window (first 5 s
dropped) per Spark; a 2x unit adds both Sparks. tok/s/W = the cell's aggregate output tok/s / that mean.
--vs adds a table of the other run's setups next to this one's (k86 against k84: stock vs shipped, same harness).
"""
import csv, glob, json, os, statistics, sys

BENCH = ("tgdef-c1", "tgdef-c8", "tgt0-c1", "tgt0-c8")
LABEL = {"tgdef": "tg512, T=1.0 top-p 0.95 top-k 20, thinking on", "tgt0": "tg512, T=0, thinking off",
         "coding-t0-nothink": "coding-36, T=0, thinking off", "coding-default": "coding-36, server sampling, thinking on"}


def rd_tsv(p):
    try:
        return [l.rstrip("\n").split("\t") for l in open(p) if l.strip()]
    except OSError:
        return []


def power(res):
    out = {}
    for h in ("dgx01", "dgx02"):
        rows = []
        try:
            for l in open(f"{res}/power-{h}.csv"):
                f = [x.strip() for x in l.split(",")]
                try:
                    rows.append((float(f[0]), float(f[1])))
                except (ValueError, IndexError):
                    pass
        except OSError:
            pass
        out[h] = rows
    return out


def watts(pw, hosts, a, b):
    tot, per = 0.0, {}
    for h in hosts.split("+"):
        v = [w for t, w in pw.get(h, []) if a + 5 <= t <= b]
        if not v:
            return None, per
        per[h] = statistics.mean(v); tot += per[h]
    return tot, per


def sd(v):
    return statistics.stdev(v) if len(v) > 1 else 0.0


def cell_metrics(udir, cell):
    """-> {metric: [values per run/pass]}"""
    p = f"{udir}/{cell}.json"
    try:
        d = json.load(open(p))
    except (OSError, ValueError):
        return {}
    m = {}
    if cell in BENCH:
        for b in d.get("benchmarks", []):
            if b.get("is_context_prefill_phase"):
                continue
            m["tg_tok_s"] = b["tg_throughput"]["values"]
            m["tg_req_tok_s"] = b["tg_req_throughput"]["values"]
            m["pp_tok_s"] = b["pp_throughput"]["values"]
            m["ttft_ms"] = b["e2e_ttft"]["values"]
            if b.get("accept_per_draft") is not None:
                m["accept_per_draft"] = [b["accept_per_draft"]]
    else:
        runs = d.get("runs", [])
        m["agg_tok_s"] = [r["agg_tps"] for r in runs]
        m["median_req_tok_s"] = [statistics.median(r["tps"]) for r in runs if r["tps"]]
        m["hit_max_tokens"] = [r["hit_max"] for r in runs]
        m["ok_requests"] = [r["ok"] for r in runs]
    return m


def collect(res):
    units = {u[0]: u for u in rd_tsv(f"{res}/units.tsv")}
    marks = rd_tsv(f"{res}/marks.tsv")
    pw = power(res)
    rows = []
    for u, c, a, b, rc in marks:
        if u not in units:
            continue
        _, setup, boot, hosts, _rec = units[u][:5]
        w, per = watts(pw, hosts, float(a), float(b))
        base = {"setup": setup, "boot": boot, "unit": u, "hosts": hosts, "cell": c, "rc": rc,
                "window_s": round(float(b) - float(a), 1), "watts": None if w is None else round(w, 2),
                "watts_per_spark": ";".join(f"{h}={v:.2f}" for h, v in per.items())}
        if c == "idle":
            rows.append({**base, "metric": "idle", "value": None, "sd": None, "n": 0, "tok_s_per_w": None})
            continue
        m = cell_metrics(f"{res}/{u}", c)
        agg = "tg_tok_s" if c in BENCH else "agg_tok_s"
        for k, v in m.items():
            if not v:
                continue
            val = statistics.mean(v)
            tpw = round(val / w, 4) if (k == agg and w) else None
            rows.append({**base, "metric": k, "value": round(val, 3), "sd": round(sd(v), 3), "n": len(v),
                         "tok_s_per_w": tpw})
    return rows


def boots(rows):
    """per (setup, hosts, cell, metric): per-unit means -> mean, sd between units, pooled within sd; plus 1x pooled"""
    g = {}
    for r in rows:
        if r["value"] is None and r["metric"] != "idle":
            continue
        keys = [(r["setup"], r["hosts"], r["cell"], r["metric"])]
        if "+" not in r["hosts"]:
            keys.append((r["setup"], "pooled", r["cell"], r["metric"]))
        for k in keys:
            g.setdefault(k, []).append(r)
    out = []
    for (s, h, c, m), rs in sorted(g.items()):
        if h == "pooled" and len({r["hosts"] for r in rs}) < 2:
            continue
        vals = [r["value"] for r in rs if r["value"] is not None]
        ws = [r["watts"] for r in rs if r["watts"] is not None]
        tpw = [r["tok_s_per_w"] for r in rs if r["tok_s_per_w"] is not None]
        within = [r["sd"] for r in rs if r["sd"] is not None and r["n"] > 1]
        mean = statistics.mean(vals) if vals else None
        out.append({"setup": s, "hosts": h, "cell": c, "metric": m, "units": len(rs),
                    "mean": None if mean is None else round(mean, 3),
                    "sd_between_boots": round(sd(vals), 3) if vals else None,
                    "cv_between_pct": round(100 * sd(vals) / mean, 2) if vals and mean else None,
                    "sd_within_boot": round(statistics.mean(within), 3) if within else None,
                    "watts": round(statistics.mean(ws), 2) if ws else None,
                    "tok_s_per_w": round(statistics.mean(tpw), 4) if tpw else None})
    return out


def fmt(v, d=1):
    return "-" if v is None else f"{v:.{d}f}"


def text(res, rows, bt, vs=None):
    L = [f"Results: {res}", ""]
    setups = sorted({b["setup"] for b in bt})
    for s in setups:
        L.append(f"== {s}")
        L.append(f"{'cell':<26}{'hosts':<13}{'units':>5}{'tok/s mean':>12}{'sd boots':>10}{'cv %':>7}{'sd within':>10}"
                 f"{'GPU W':>8}{'tok/s/W':>9}")
        for b in bt:
            if b["setup"] != s or b["metric"] not in ("tg_tok_s", "agg_tok_s", "idle"):
                continue
            L.append(f"{b['cell']:<26}{b['hosts']:<13}{b['units']:>5}{fmt(b['mean']):>12}{fmt(b['sd_between_boots'], 2):>10}"
                     f"{fmt(b['cv_between_pct'], 1):>7}{fmt(b['sd_within_boot'], 2):>10}{fmt(b['watts']):>8}"
                     f"{fmt(b['tok_s_per_w'], 3):>9}")
        L.append("")
    if vs:
        L.append("== same cells, this run against " + vs[0])
        other = {(b["setup"], b["hosts"], b["cell"], b["metric"]): b for b in vs[1]}
        mine = {(b["setup"], b["hosts"], b["cell"], b["metric"]): b for b in bt}
        cells = sorted({(k[2], k[3]) for k in list(other) + list(mine) if k[3] in ("tg_tok_s", "agg_tok_s")})
        for c, m in cells:
            parts = [f"{s}/{h}: {fmt(b['mean'])}" + (f" (sd {fmt(b['sd_between_boots'], 2)}, {b['units']} boots)" if b["units"] > 1 else "")
                     for src in (other, mine) for (s, h, cc, mm), b in sorted(src.items()) if cc == c and mm == m]
            L.append(f"{c:<26}" + " | ".join(parts))
        L.append("")
    L.append("tok/s = llama-benchy tg_throughput (aggregate over the concurrent requests, reasoning tokens counted) or "
             "coding.py aggregate output tok/s per pass; mean of the runs/passes of a boot, then of the boots. GPU W = "
             "nvidia-smi power.draw mean over the cell window (both Sparks added for 2x); idle = 120 s with the server up.")
    return "\n".join(L)


def report(res, vs=None):
    rows = collect(res)
    if not rows:
        print("no rows (units.tsv/marks.tsv empty)", file=sys.stderr)
        return 1
    with open(f"{res}/cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    bt = boots(rows)
    with open(f"{res}/boots.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bt[0].keys())); w.writeheader(); w.writerows(bt)
    other = (vs, boots(collect(vs))) if vs and os.path.isdir(vs) else None
    t = text(res, rows, bt, other)
    open(f"{res}/cred.txt", "w").write(t + "\n"); print(t)
    return 0


def selftest():
    import tempfile
    d = tempfile.mkdtemp()
    with open(f"{d}/units.tsv", "w") as f:
        f.write("u1\t2x-test\t1\tdgx01+dgx02\tr.yaml\nu2\t1x-test\t1\tdgx01\tr.yaml\nu3\t1x-test\t1\tdgx02\tr.yaml\n")
    with open(f"{d}/marks.tsv", "w") as f:
        f.write("u1\tidle\t100\t200\t0\nu1\ttgdef-c1\t200\t300\t0\nu2\tcoding-t0-nothink-c8\t100\t200\t0\n"
                "u3\tcoding-t0-nothink-c8\t100\t200\t0\n")
    for h, w in (("dgx01", 10.0), ("dgx02", 20.0)):
        with open(f"{d}/power-{h}.csv", "w") as f:
            for t in range(90, 310):
                f.write(f"{t}.5,{w if t < 200 else w * 3}, 2000, 50\n")
    for u in ("u1", "u2", "u3"):
        os.makedirs(f"{d}/{u}")
    json.dump({"benchmarks": [{"is_context_prefill_phase": False, "tg_throughput": {"values": [90, 90, 90]},
                               "tg_req_throughput": {"values": [90, 90, 90]}, "pp_throughput": {"values": [1, 1, 1]},
                               "e2e_ttft": {"values": [5, 5, 5]}, "accept_per_draft": 0.6}]}, open(f"{d}/u1/tgdef-c1.json", "w"))
    for u, a in (("u2", 100.0), ("u3", 110.0)):
        json.dump({"runs": [{"agg_tps": a, "tps": [10, 20, 30], "hit_max": 0, "ok": 36}] * 2},
                  open(f"{d}/{u}/coding-t0-nothink-c8.json", "w"))
    rows = collect(d)
    tg = [r for r in rows if r["cell"] == "tgdef-c1" and r["metric"] == "tg_tok_s"][0]
    assert tg["value"] == 90 and abs(tg["watts"] - 90.0) < 1e-6 and abs(tg["tok_s_per_w"] - 1.0) < 1e-6, tg
    idle = [r for r in rows if r["cell"] == "idle"][0]
    assert abs(idle["watts"] - 30.0) < 1e-6, idle
    bt = boots(rows)
    pooled = [b for b in bt if b["hosts"] == "pooled" and b["metric"] == "agg_tok_s"][0]
    assert pooled["units"] == 2 and pooled["mean"] == 105.0 and abs(pooled["sd_between_boots"] - 7.071) < 1e-3, pooled
    assert report(d) == 0 and os.path.getsize(f"{d}/cred.txt") > 0
    print("cred selftest OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        selftest()
    elif a[:1] == ["report"] and len(a) >= 2:
        vs = a[a.index("--vs") + 1] if "--vs" in a else None
        sys.exit(report(a[1], vs))
    else:
        print(__doc__); sys.exit(2)
