#!/usr/bin/env python3
"""window.py (2026-11-14, k122): steady-state evidence and profiler-overhead reference for one profiled decode window.

  window.py sample --base URL --out FILE.jsonl   poll /metrics every 0.25 s until killed: one line per poll with the
                                                 time, requests running/waiting and prompt/generation token counters
  window.py report OB TAG C                      text block for k122.txt from OB/winmon-TAG.jsonl, OB/trig-TAG.t
                                                 (trigger epoch), OB/prof-TAG/t_end (epoch the rank-0 trace appeared),
                                                 OB/probe-TAG.json, OB/prof-TAG/breakdown-rank0.json, OB/prof-TAG/MEASURED
  window.py selftest

Window = [last poll at or before the trigger, first poll at or after the trace appeared] (an upper bound: it also holds
the export). STEADY = every poll in it shows C requests running and none waiting, and the prompt-token counter did not
move (no prefill of another request inside the window); else MIXED with the reasons. The profiler-overhead reference is
the inf-phase decode gap p50 of repeat 0 (unprofiled) against repeat 1 (the repeat that holds the window, so the
ratio understates the window's own overhead), next to the trace's wall time per step.
"""
import json
import os
import re
import statistics
import sys
import time
import urllib.request

PAT = {"run": "vllm:num_requests_running", "wait": "vllm:num_requests_waiting",
       "ptok": "vllm:prompt_tokens_total", "gtok": "vllm:generation_tokens_total"}


def parse(text):
    out = {}
    for k, name in PAT.items():
        v = [float(m) for m in re.findall(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$", text, re.M)]
        out[k] = sum(v) if v else None
    return out


def sample(base, out):
    with open(out, "a") as f:
        while True:
            t = time.time()
            try:
                with urllib.request.urlopen(base + "/metrics", timeout=2) as r:
                    row = parse(r.read().decode(errors="replace"))
            except Exception:
                row = {}
            f.write(json.dumps({"t": t, **row}) + "\n"); f.flush()
            time.sleep(max(0.0, 0.25 - (time.time() - t)))


def rd(p):
    try:
        return open(p).read().strip()
    except OSError:
        return ""


def steady(rows, t0, t1, c):
    rows = [r for r in rows if r.get("run") is not None]
    before = [r for r in rows if r["t"] <= t0]
    after = [r for r in rows if r["t"] >= t1]
    win = ([before[-1]] if before else []) + [r for r in rows if t0 < r["t"] < t1] + ([after[0]] if after else [])
    if not before or not after or len(win) < 3:
        return "n/a", f"{len(win)} polls covering the window (need the poll before the trigger and after the trace)"
    runs = [r["run"] for r in win]; waits = [r.get("wait") or 0 for r in win]
    dp = (win[-1].get("ptok") or 0) - (win[0].get("ptok") or 0)
    dg = (win[-1].get("gtok") or 0) - (win[0].get("gtok") or 0)
    why = []
    if min(runs) != c or max(runs) != c:
        why.append(f"running {min(runs):g}..{max(runs):g} (want {c})")
    if max(waits) > 0:
        why.append(f"waiting up to {max(waits):g}")
    if dp > 0:
        why.append(f"prompt tokens +{dp:g} (a prefill inside the window)")
    info = f"{len(win)} polls over {t1 - t0:.1f} s, running {min(runs):g}..{max(runs):g}, waiting max {max(waits):g}, " \
           f"prompt tokens +{dp:g}, generation tokens +{dg:g}"
    return ("STEADY" if not why else "MIXED (" + "; ".join(why) + ")"), info


def gaps(probe, c):
    try:
        rows = json.load(open(probe))["rows"]
    except (OSError, ValueError, KeyError):
        return None
    g = {r: [x["inf"]["gap_ms_p50"] for x in rows if x.get("index", -1) // c == r and (x.get("inf") or {}).get("gap_ms_p50")]
         for r in (0, 1)}
    return {r: statistics.mean(v) for r, v in g.items() if v}


def report(ob, tag, c):
    c = int(c)
    L = [f"  window {tag} (c{c}):"]
    try:
        rows = [json.loads(x) for x in open(os.path.join(ob, f"winmon-{tag}.jsonl")) if x.strip()]
    except (OSError, ValueError):
        rows = []
    t0, t1 = rd(os.path.join(ob, f"trig-{tag}.t")), rd(os.path.join(ob, f"prof-{tag}", "t_end"))
    if t0 and t1 and rows:
        v, info = steady(rows, float(t0), float(t1), c)
    else:
        v, info = "n/a", f"trigger time {'ok' if t0 else 'missing'}, trace time {'ok' if t1 else 'missing'}, {len(rows)} polls"
    L.append(f"    steady state: {v}; {info}")
    g = gaps(os.path.join(ob, f"probe-{tag}.json"), c)
    try:
        sm = json.load(open(os.path.join(ob, f"prof-{tag}", "breakdown-rank0.json")))["step_ms"]
    except (OSError, ValueError, KeyError):
        sm = None
    gtxt = "n/a" if not g else (f"repeat 0 (unprofiled) {g.get(0, float('nan')):.2f}, repeat 1 (holds the window) "
                                f"{g.get(1, float('nan')):.2f}" + (f", ratio {g[1] / g[0]:.3f}" if 0 in g and 1 in g else ""))
    L.append(f"    decode gap p50 ms per stream chunk: {gtxt}; trace wall per step rank 0: "
             f"{'n/a' if sm is None else f'{sm:.2f} ms'} (profiled)")
    m = rd(os.path.join(ob, f"prof-{tag}", "MEASURED"))
    L.append(f"    b12x plan measurement during the window: {m or 'none'}")
    return "\n".join(L), v


def selftest():
    import tempfile
    d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "prof-fresh-c4"))
    txt = 'vllm:num_requests_running{model_name="q"} 4.0\nvllm:num_requests_waiting{model_name="q"} 0.0\n' \
          'vllm:prompt_tokens_total{model_name="q"} 1000.0\nvllm:generation_tokens_total{model_name="q"} 50.0\n'
    p = parse(txt)
    assert p == {"run": 4.0, "wait": 0.0, "ptok": 1000.0, "gtok": 50.0}, p
    rows = [{"t": 100 + 0.25 * i, "run": 4, "wait": 0, "ptok": 1000, "gtok": 50 + 10 * i} for i in range(40)]
    assert steady(rows, 101.0, 105.0, 4)[0] == "STEADY"
    rows2 = [dict(r, ptok=1000 + (2048 if r["t"] > 103 else 0), run=(3 if r["t"] > 104 else 4)) for r in rows]
    v, _ = steady(rows2, 101.0, 105.0, 4)
    assert v.startswith("MIXED") and "prompt tokens +2048" in v and "running 3..4" in v, v
    assert steady(rows, 99.0, 105.0, 4)[0] == "n/a", "no poll before the trigger"
    open(os.path.join(d, "winmon-fresh-c4.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in rows))
    open(os.path.join(d, "trig-fresh-c4.t"), "w").write("101.0\n")
    open(os.path.join(d, "prof-fresh-c4", "t_end"), "w").write("105.0\n")
    pr = [{"index": i, "inf": {"gap_ms_p50": 20.0 if i < 4 else 22.0}} for i in range(8)]
    json.dump({"rows": pr}, open(os.path.join(d, "probe-fresh-c4.json"), "w"))
    json.dump({"step_ms": 61.5}, open(os.path.join(d, "prof-fresh-c4", "breakdown-rank0.json"), "w"))
    open(os.path.join(d, "prof-fresh-c4", "MEASURED"), "w").write("ranks 0+1 measured +3 plans\n")
    txt, v = report(d, "fresh-c4", 4)
    assert v == "STEADY" and "ratio 1.100" in txt and "61.50 ms" in txt and "measured +3" in txt, txt
    print("window selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["sample"]:
        kw = dict(zip(a[1::2], a[2::2])); sample(kw["--base"], kw["--out"])
    elif a[:1] == ["report"] and len(a) == 4:
        print(report(a[1], a[2], a[3])[0])
    else:
        sys.exit(__doc__)
