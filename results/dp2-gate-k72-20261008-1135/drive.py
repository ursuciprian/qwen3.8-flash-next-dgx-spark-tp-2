#!/usr/bin/env python3
"""k60 driver (2026-10-07, promised on #100): agent-style multi-turn replay, the same workload against the 2x TP2 and
against DP=2 (one shipped 1x v3d per Spark behind pa_router.py).

A session = a synthetic agent transcript (fidelity_probe's tool-call blocks, its own seed, ~start tokens), then turns:
each turn sends the whole conversation (tools on, thinking off, temperature 0.6), appends the model's reply and a
~add-token tool result, and goes on. Sessions start together and run in parallel. Workloads (same seeds on both setups):
  agent8    8 sessions x 6 turns, start ~32K tokens, +2K per turn, max_tokens 512
  agent16  16 sessions x 4 turns, start ~32K, +2K per turn, max_tokens 512
  long-N    N sessions x 2 turns, start ~128K, +1K, max_tokens 256, N = 4, 8, 12, 16; stops after the first N with
            preemptions or request errors (max concurrent long contexts = the largest clean N)
Per workload: aggregate output tok/s (completion tokens / wall), TTFT of the first turn and of follow-up turns,
per-request decode tok/s, prefix-cache hit rate (hit / queried tokens, summed over replicas), preemptions, max KV
usage and running requests per replica, min MemAvailable per Spark, router sessions per replica (DP).

  drive.py --setup tp2|dp2 --base URL --metrics URL[,URL] --res DIR
  drive.py --report DIR     (DIR holds tp2/ and dp2/) -> DIR/k60.txt, DIR/comment.md
  drive.py --selftest
"""
import argparse, json, os, random, sys, threading, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "k59")]
import vm  # noqa: E402
import fidelity_probe as fp  # noqa: E402

MODEL, CPT = "qwen3.8-flash-next", 2.4  # chars per token of the transcript text (fidelity's 4.6 gives ~1.9x the depth)
WORKLOADS = [("agent8", 8, 6, 32000, 2000, 512, 100), ("agent16", 16, 4, 32000, 2000, 512, 200)] + \
            [(f"long-{n}", n, 2, 128000, 1000, 256, 300 + 20 * n) for n in (4, 8, 12, 16)]
NEXT = "Continue the investigation: run the next shell command you need with the bash tool."


def tool_output(rng, ntok):
    lines, chars = [], 0
    while chars < ntok * CPT:
        s = (f"2026-10-0{rng.randint(1, 7)}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z "
             f"{rng.choice(fp.SERVICES)} {rng.choice(['INFO', 'WARN', 'ERROR'])} invoice INV-{rng.randint(1000, 9999)}-"
             f"{fp.hexid(rng, 4).upper()} {rng.choice(['processed', 'retried', 'queued', 'rejected'])} in {rng.randint(2, 900)}ms "
             f"path={fp.rand_path(rng)}")
        lines.append(s); chars += len(s) + 1
    return "\n".join(lines)


def session(base, sid, seed, turns, start, add, max_tokens, out, lock):
    rng = random.Random(seed)
    msgs, _, _ = fp.build_transcript(start, 1, seed, CPT)
    msgs = msgs + [{"role": "user", "content": NEXT}]
    for t in range(turns):
        rec = {"session": sid, "seed": seed, "turn": t, "t_start": time.time()}
        try:
            m, usage, ttft, dt = fp.ask(base, MODEL, msgs, 0.6, max_tokens, False, 3600)
        except Exception as ex:
            rec.update(error=f"{type(ex).__name__}: {str(ex)[:200]}")
            with lock:
                out.append(rec)
            return
        ct = usage.get("completion_tokens")
        rec.update(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=ct, ttft=ttft, dt=dt,
                   decode_tps=(ct / (dt - ttft)) if ct and dt > ttft else None,
                   tool_call=bool(m.get("tool_calls")))
        with lock:
            out.append(rec)
        if m.get("tool_calls"):
            tc = m["tool_calls"][0]
            msgs = msgs + [{"role": "assistant", "content": m.get("content"), "tool_calls": [tc]},
                           {"role": "tool", "tool_call_id": tc["id"], "content": tool_output(rng, add)}]
        else:
            msgs = msgs + [{"role": "assistant", "content": m.get("content") or ""},
                           {"role": "user", "content": "Command output:\n" + tool_output(rng, add) + "\n\n" + NEXT}]


def workload(base, murls, res, name, n, turns, start, add, max_tokens, seed0, router=None):
    d = os.path.join(res, name); os.makedirs(d, exist_ok=True)
    rstat0 = get_json(router + "/router/stats") if router else None
    m0 = [vm.scrape(u) for u in murls]
    poll, out, lock, t0 = vm.Poller(murls), [], threading.Lock(), time.time(); poll.start()
    th = [threading.Thread(target=session, args=(base, i, seed0 + i, turns, start, add, max_tokens, out, lock)) for i in range(n)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    t1, gmax = time.time(), poll.stop()
    m1 = [vm.scrape(u) for u in murls]
    dm = {k: 0.0 for k in vm.COUNTERS}
    for a, b in zip(m0, m1):
        for k, v in vm.delta(a, b).items():
            dm[k] = None if v is None or dm[k] is None else dm[k] + v
    ok = [r for r in out if "error" not in r]
    q = dm["vllm:prefix_cache_queries_total"]
    r = {"workload": name, "sessions": n, "turns": turns, "start_tokens": start, "add_tokens": add, "max_tokens": max_tokens,
         "expected": n * turns, "completed": len(ok), "errors": len(out) - len(ok),
         "prompt_tokens_max": max([x["prompt_tokens"] or 0 for x in ok] or [0]),
         "wall_s": t1 - t0, "agg_output_tps": sum(x["completion_tokens"] or 0 for x in ok) / (t1 - t0),
         "ttft_first_mean": vm.mean([x["ttft"] for x in ok if x["turn"] == 0]),
         "ttft_first_max": max([x["ttft"] for x in ok if x["turn"] == 0] or [0]) or None,
         "ttft_follow_mean": vm.mean([x["ttft"] for x in ok if x["turn"] > 0]),
         "ttft_follow_p90": vm.pct([x["ttft"] for x in ok if x["turn"] > 0], 0.9),
         "decode_tps_mean": vm.mean([x["decode_tps"] for x in ok]),
         "prefix_hit_rate": vm.div(dm["vllm:prefix_cache_hits_total"], q),
         "preemptions": dm["vllm:num_preemptions_total"],
         "kv_usage_max": [vm.kv_max(gmax.get(u, {})) for u in murls],
         "running_max": [gmax.get(u, {}).get("vllm:num_requests_running") for u in murls],
         "waiting_max": [gmax.get(u, {}).get("vllm:num_requests_waiting") for u in murls],
         "min_mem_dgx01": vm.minmem(os.path.join(os.path.dirname(res), "guard-dgx01.log"), int(t0), int(t1) + 1),
         "min_mem_dgx02": vm.minmem(os.path.join(os.path.dirname(res), "guard-dgx02.log"), int(t0), int(t1) + 1),
         "metrics_delta": dm, "t0": t0, "t1": t1}
    if router:
        s1 = get_json(router + "/router/stats")
        if rstat0 and s1:
            r["router_sessions"] = [b - a for a, b in zip(rstat0["sessions"], s1["sessions"])]
            r["router_requests"] = [b - a for a, b in zip(rstat0["requests"], s1["requests"])]
    r["clean"] = r["errors"] == 0 and r["completed"] == r["expected"] and r["preemptions"] == 0
    json.dump({"summary": r, "turns": sorted(out, key=lambda x: (x["session"], x["turn"]))},
              open(os.path.join(d, "run.json"), "w"), indent=1)
    return r


def get_json(url):
    try:
        return json.load(urllib.request.urlopen(url, timeout=10))
    except Exception:
        return None


def f(x, nd=1, suf=""):
    return "-" if x is None else f"{x:.{nd}f}{suf}"


def pc(x):
    return "-" if x is None else f"{x * 100:.0f}%"


def lst(xs, fn):
    return " / ".join(fn(x) for x in xs)


def main_run(a):
    os.makedirs(a.res, exist_ok=True)
    murls = a.metrics.split(",")
    logf = open(os.path.join(a.res, "drive.log"), "a")
    out = []
    for name, n, turns, start, add, mt, seed0 in WORKLOADS:
        if a.only and name not in a.only.split(","):
            continue
        if name.startswith("long-") and any(r["workload"].startswith("long-") and not r["clean"] for r in out):
            continue
        r = workload(a.base, murls, a.res, name, n, turns, start, add, mt, seed0, a.router)
        out.append(r)
        s = (f"[{time.strftime('%F %T')}] {a.setup} {name}: {r['completed']}/{r['expected']} turns, errors {r['errors']}, "
             f"output {f(r['agg_output_tps'])} tok/s, TTFT first {f(r['ttft_first_mean'])} s follow {f(r['ttft_follow_mean'])} s, "
             f"decode {f(r['decode_tps_mean'])} tok/s/request, hit {pc(r['prefix_hit_rate'])}, preempt {f(r['preemptions'], 0)}, "
             f"KV max {lst(r['kv_usage_max'], pc)}, sessions/replica {r.get('router_sessions', '-')}")
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
        json.dump({"setup": a.setup, "base": a.base, "metrics": murls, "workloads": out},
                  open(os.path.join(a.res, "k60.json"), "w"), indent=1)


def report(res):
    S = {}
    for s in ("tp2", "dp2"):
        try:
            S[s] = {w["workload"]: w for w in json.load(open(os.path.join(res, s, "k60.json")))["workloads"]}
        except Exception:
            S[s] = {}
    names = [w[0] for w in WORKLOADS if any(w[0] in S[s] for s in S)]
    rows = ["| workload | setup | turns done | output tok/s total | TTFT first turn mean / max (s) | TTFT follow-up mean / p90 (s) | "
            "decode tok/s per request | prefix hit | preemptions | KV max per replica | sessions per replica | min MemAvailable dgx-01 / dgx-02 (GiB) |",
            "|---|---|---:|---:|---|---|---:|---:|---:|---|---|---|"]
    for nm in names:
        for s, label in (("tp2", "2x TP=2"), ("dp2", "DP=2 (2 x 1x v3d)")):
            r = S[s].get(nm)
            if not r:
                rows.append(f"| {nm} | {label} | not run | | | | | | | | | |"); continue
            rows.append(f"| {nm} | {label} | {r['completed']}/{r['expected']}{' (' + str(r['errors']) + ' errors)' if r['errors'] else ''} | "
                        f"{f(r['agg_output_tps'])} | {f(r['ttft_first_mean'])} / {f(r['ttft_first_max'])} | "
                        f"{f(r['ttft_follow_mean'])} / {f(r['ttft_follow_p90'])} | {f(r['decode_tps_mean'])} | {pc(r['prefix_hit_rate'])} | "
                        f"{f(r['preemptions'], 0)} | {lst(r['kv_usage_max'], pc)} | {'/'.join(map(str, r.get('router_sessions', ['-'])))} | "
                        f"{f(r['min_mem_dgx01'], 2)} / {f(r['min_mem_dgx02'], 2)} |")
    def maxlong(s):
        c = [int(k.split("-")[1]) for k, r in S[s].items() if k.startswith("long-") and r["clean"]]
        return max(c) if c else 0
    def firstbad(s):
        b = [int(k.split("-")[1]) for k, r in S[s].items() if k.startswith("long-") and not r["clean"]]
        return min(b) if b else None
    plong = max([r["prompt_tokens_max"] for s in S for k, r in S[s].items() if k.startswith("long-")] or [0])
    ml = []
    for s, label in (("tp2", "2x TP=2"), ("dp2", "DP=2")):
        b = firstbad(s)
        if not any(k.startswith("long-") for k in S[s]):
            ml.append(f"- {label}: not run."); continue
        ml.append(f"- {label}: {maxlong(s)} sessions of ~{plong:,} tokens ran with no preemption or error"
                  + (f"; at {b} sessions: {f(S[s]['long-' + str(b)]['preemptions'], 0)} preemptions, "
                     f"{S[s]['long-' + str(b)]['errors']} errors (ladder stopped there)." if b else
                     " (the largest step of the ladder)."))
    kv = open(os.path.join(res, "kvpool.txt")).read().strip() if os.path.exists(os.path.join(res, "kvpool.txt")) else ""
    lines = [
        "DP=2 against TP=2 on the same agent-style replay, measured " + time.strftime("%Y-%m-%d") + ".",
        "",
        "- 2x TP=2: the shipped `qwen3.8-flash-next-2x-dgx-spark` (b1.4), `max_num_seqs` 16.",
        "- DP=2: the shipped `qwen3.8-flash-next-1x-dgx-spark` (v3d) on each Spark, `max_num_seqs` 8 each, behind a minimal "
        "prefix-affinity router (`scripts/pa_router.py`: sha1 of the first two non-system messages picks the replica, so all turns "
        "of a session go to the replica that holds its prefix cache).",
        "",
        "Workload (`drive.py` in the results dir, same seeds on both setups): each session is a synthetic agent transcript of tool calls; "
        "every turn sends the whole conversation with tools on, thinking off, temperature 0.6, then appends the reply and a "
        "tool result. Sessions start together.",
        "",
        "- `agent8`: 8 sessions x 6 turns, ~32K-token start, +2K tokens per turn, up to 512 output tokens.",
        "- `agent16`: 16 sessions x 4 turns, same sizes.",
        "- `long-N`: N sessions x 2 turns, ~128K-token start, +1K per turn, up to 256 output tokens, N = 4, 8, 12, 16. "
        "The ladder stops after the first N with a preemption or a request error.",
        "",
        "\n".join(rows),
        "",
        "Max concurrent long contexts:",
        "\n".join(ml),
        "",
        ("KV pools from the boot logs:\n```\n" + kv + "\n```\n" if kv else ""),
        "Columns: prefix hit = cache hit tokens / queried tokens, summed over replicas; preemptions = `vllm:num_preemptions_total` "
        "delta; KV max = highest `vllm:kv_cache_usage_perc` sample (every 5 s) per replica; sessions per replica = distinct "
        "conversations the router sent to each replica; decode tok/s per request = completion tokens / time after the first token.",
        "",
        "Raw data: {RESULTS_URL} (per-turn JSON per workload, router stats, boot log KV lines, guard logs).",
    ]
    open(os.path.join(res, "comment.md"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(res, "k60.txt"), "w").write("\n".join(rows) + "\n\n" + "\n".join(ml) + "\n")
    # complete = both agent workloads and at least one ladder step on both setups
    return all({"agent8", "agent16"} <= set(S[s]) and any(k.startswith("long-") for k in S[s]) for s in S)


def selftest():
    vm.selftest()
    rng = random.Random(1)
    t = tool_output(rng, 1000)
    assert 2000 < len(t) < 3000, len(t)
    msgs, _, _ = fp.build_transcript(32000, 1, 100, CPT)
    n = sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls", ""))) for m in msgs)
    assert 32000 * CPT <= n < 32000 * CPT * 1.05, n
    import tempfile
    d = tempfile.mkdtemp()
    for s in ("tp2", "dp2"):
        os.makedirs(os.path.join(d, s))
        w = [{"workload": "agent8", "completed": 48, "expected": 48, "errors": 0, "agg_output_tps": 300.0, "ttft_first_mean": 20.0,
              "ttft_first_max": 40.0, "ttft_follow_mean": 1.0, "ttft_follow_p90": 2.0, "decode_tps_mean": 40.0, "prefix_hit_rate": 0.9,
              "preemptions": 0.0, "kv_usage_max": [0.3] if s == "tp2" else [0.5, 0.4], "min_mem_dgx01": 10.0, "min_mem_dgx02": 11.0,
              "prompt_tokens_max": 40000, "clean": True},
             {"workload": "long-4", "completed": 8, "expected": 8, "errors": 0, "agg_output_tps": 30.0, "ttft_first_mean": 200.0,
              "ttft_first_max": 300.0, "ttft_follow_mean": 3.0, "ttft_follow_p90": 4.0, "decode_tps_mean": 30.0, "prefix_hit_rate": 0.5,
              "preemptions": 0.0 if s == "tp2" else 3.0, "kv_usage_max": [0.3], "min_mem_dgx01": 10.0, "min_mem_dgx02": 11.0,
              "prompt_tokens_max": 130000, "clean": s == "tp2"}]
        if s == "dp2":
            w[0]["router_sessions"] = [5, 3]
        json.dump({"workloads": w}, open(os.path.join(d, s, "k60.json"), "w"))
    assert report(d) is False  # no agent16 in the fixture
    c = open(os.path.join(d, "comment.md")).read()
    assert "| agent8 | DP=2 (2 x 1x v3d) | 48/48 | 300.0 |" in c and "5/3" in c, c
    assert "- DP=2: 0 sessions" in c and "at 4 sessions: 3 preemptions" in c and "- 2x TP=2: 4 sessions" in c, c
    print("k60 selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=["tp2", "dp2"]); ap.add_argument("--base"); ap.add_argument("--metrics")
    ap.add_argument("--router"); ap.add_argument("--res"); ap.add_argument("--only", default="")
    ap.add_argument("--report"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.report:
        sys.exit(0 if report(a.report) else 3)
    else:
        main_run(a)
