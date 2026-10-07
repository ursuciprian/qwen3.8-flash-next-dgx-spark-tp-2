#!/usr/bin/env python3
"""k59 driver (2026-10-07, promised on #100): long-context concurrency on the shipped 2x b1.4.

fidelity_probe.py (this dir's copy: + completion_tokens/turns per result) at depth 128000 (~245k actual tokens) and at
DMAX, the depth whose prompt is as close to 262,144 - 4,096 (max_tokens) - 1,024 (follow-up turn) as the server's
/tokenize says. Per depth:
  shared c1/c2/c4   one transcript (seed 7), 5 targets x 4 trials, --concurrency 1/2/4 (prefix cache warm after c1)
  sessions s2/s4    2 or 4 probe processes at once, each its own transcript (distinct seeds, cold), 5 targets x 2 trials:
                    2 or 4 different 245k-262k contexts resident at the same time
Per run: exact/near/wrong/explore/no_call, TTFT (mean/p50/max), per-request decode tok/s (completion tokens over
time after the first token, single-turn results), aggregate output tok/s, preemptions, prefix-cache hit rate, max KV
usage, max running/waiting, min MemAvailable on both Sparks (guard logs).

  drive.py --res DIR [--base http://localhost:8000] [--only d128k-c1,...]
  drive.py --selftest
"""
import argparse, json, os, subprocess, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import vm  # noqa: E402
import fidelity_probe as fp  # noqa: E402

MODEL, K, MAXTOK = "qwen3.8-flash-next", 5, 4096
TARGET = 262144 - MAXTOK - 1024
ASK = ("The log for invoice_id {tid} needs a check. Call the bash tool with exactly one command: "
       "`tail -n 50` on that client's {fname} (the exact path that appeared earlier), nothing else.")


def runs(dmax):
    out = []
    for tag, d, s2, s4 in (("d128k", 128000, [21, 22], [31, 32, 33, 34]), ("dmax", dmax, [41, 42], [51, 52, 53, 54])):
        out += [(f"{tag}-c1", d, 1, [7], 4), (f"{tag}-c2", d, 2, [7], 4), (f"{tag}-c4", d, 4, [7], 4),
                (f"{tag}-s2", d, 1, s2, 2), (f"{tag}-s4", d, 1, s4, 2)]
    return out


def ntok(base, depth, seed):
    msgs, targets, _ = fp.build_transcript(depth, K, seed, 4.6)
    t = targets[-1]
    msgs = msgs + [{"role": "user", "content": ASK.format(tid=t["id"], fname=t["path"].rsplit("/", 1)[1])}]
    body = {"model": MODEL, "messages": msgs, "tools": fp.TOOLS, "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(base + "/tokenize", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["count"]


def calibrate(base, actual128, seeds, log):
    d = int(128000 * TARGET / actual128)
    try:
        for _ in range(4):
            n = max(ntok(base, d, s) for s in seeds)
            log(f"calibrate: depth {d} -> max {n} tokens over seeds {seeds} (target {TARGET})")
            if TARGET - 1500 <= n <= TARGET:
                return d, n
            d = int(d * (TARGET - 500) / n)
        n = max(ntok(base, d, s) for s in seeds)
        if n <= TARGET:
            return d, n
    except Exception as ex:
        log(f"calibrate: /tokenize failed ({type(ex).__name__}: {str(ex)[:120]}); ratio estimate x 0.98")
    return int(128000 * TARGET / actual128 * 0.98), None


def one_run(base, res, name, depth, conc, seeds, trials, log):
    d = os.path.join(res, name); os.makedirs(d, exist_ok=True)
    murl = base + "/metrics"
    m0, poll, t0 = vm.scrape(murl), vm.Poller([murl]), time.time(); poll.start()
    procs = [subprocess.Popen([sys.executable, os.path.join(HERE, "fidelity_probe.py"), "--base", base, "--model", MODEL,
                               "--depths", str(depth), "--k", str(K), "--trials", str(trials), "--concurrency", str(conc),
                               "--seed", str(s), "--max-tokens", str(MAXTOK), "--timeout", "3600",
                               "--out", os.path.join(d, f"seed{s}.json")],
                              stdout=open(os.path.join(d, f"seed{s}.out"), "w"), stderr=subprocess.STDOUT) for s in seeds]
    rcs = [p.wait() for p in procs]
    t1, gmax, m1 = time.time(), poll.stop(), vm.scrape(murl)
    res_ = []
    for s in seeds:
        try:
            res_ += json.load(open(os.path.join(d, f"seed{s}.json")))["depths"][0]["results"]
        except Exception:
            pass
    dm = vm.delta(m0, m1)
    counts = {c: sum(r["cat"] == c for r in res_) for c in ("exact", "near", "wrong", "explore", "no_call")}
    errors = sum(str(r.get("cmd", "")).startswith(("ERROR", "FOLLOWUP ERROR")) for r in res_)
    dec = [r["completion_tokens"] / (r["dt"] - r["ttft"]) for r in res_
           if r.get("turns") == 1 and r.get("completion_tokens") and r.get("dt") and r.get("ttft") is not None and r["dt"] > r["ttft"]]
    ttft = [r["ttft"] for r in res_ if r.get("ttft") is not None]
    q = dm["vllm:prefix_cache_queries_total"]
    g = gmax.get(murl, {})
    out = {"run": name, "depth": depth, "concurrency": conc, "sessions": len(seeds), "seeds": seeds, "trials": trials,
           "expected": K * trials * len(seeds), "returned": len(res_), "probe_exit": rcs, "errors": errors, "counts": counts,
           "prompt_tokens_max": max([r["prompt_tokens"] or 0 for r in res_] or [0]),
           "ttft_mean": vm.mean(ttft), "ttft_p50": vm.pct(ttft, 0.5), "ttft_max": max(ttft) if ttft else None,
           "decode_tps_mean": vm.mean(dec), "decode_tps_p50": vm.pct(dec, 0.5), "decode_n": len(dec),
           "agg_output_tps": vm.div(dm["vllm:generation_tokens_total"], t1 - t0), "wall_s": t1 - t0,
           "preemptions": dm["vllm:num_preemptions_total"], "prefix_hit_rate": vm.div(dm["vllm:prefix_cache_hits_total"], q),
           "kv_usage_max": vm.kv_max(g), "running_max": g.get("vllm:num_requests_running"),
           "waiting_max": g.get("vllm:num_requests_waiting"),
           "min_mem_dgx01": vm.minmem(os.path.join(res, "guard-dgx01.log"), int(t0), int(t1) + 1),
           "min_mem_dgx02": vm.minmem(os.path.join(res, "guard-dgx02.log"), int(t0), int(t1) + 1),
           "metrics_delta": dm, "t0": t0, "t1": t1}
    out["ok"] = out["returned"] == out["expected"] and errors == 0 and dm["vllm:num_preemptions_total"] is not None
    json.dump(out, open(os.path.join(d, "run.json"), "w"), indent=1)
    log(f"{name}: {line(out)}")
    return out


def f(x, nd=1, suf=""):
    return "-" if x is None else f"{x:.{nd}f}{suf}"


def line(r):
    c = r["counts"]
    return (f"exact {c['exact']} near {c['near']} wrong {c['wrong']} explore {c['explore']} no_call {c['no_call']} "
            f"(of {r['expected']}, errors {r['errors']}) | prompt {r['prompt_tokens_max']} | TTFT mean {f(r['ttft_mean'])} s "
            f"max {f(r['ttft_max'])} s | decode {f(r['decode_tps_mean'])} tok/s/request | output {f(r['agg_output_tps'])} tok/s | "
            f"preempt {f(r['preemptions'], 0)} | hit {f(r['prefix_hit_rate'] and r['prefix_hit_rate'] * 100, 0, '%')} | "
            f"KV max {f(r['kv_usage_max'] and r['kv_usage_max'] * 100, 0, '%')} | MemAvail min {f(r['min_mem_dgx01'], 2)}/{f(r['min_mem_dgx02'], 2)} GiB")


def table(rs):
    h = ("| run | sessions x concurrency | prompt tokens | exact | near | wrong | explore | no_call | TTFT mean / max (s) | "
         "decode tok/s per request | output tok/s total | preemptions | prefix hit | KV max | min MemAvailable dgx-01 / dgx-02 (GiB) |\n"
         "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|")
    rows = []
    for r in rs:
        c = r["counts"]
        rows.append(f"| {r['run']} | {r['sessions']} x {r['concurrency']} | {r['prompt_tokens_max']:,} | {c['exact']} | {c['near']} | "
                    f"{c['wrong']} | {c['explore']} | {c['no_call']} | {f(r['ttft_mean'])} / {f(r['ttft_max'])} | "
                    f"{f(r['decode_tps_mean'])} | {f(r['agg_output_tps'])} | {f(r['preemptions'], 0)} | "
                    f"{f(r['prefix_hit_rate'] and r['prefix_hit_rate'] * 100, 0, '%')} | {f(r['kv_usage_max'] and r['kv_usage_max'] * 100, 0, '%')} | "
                    f"{f(r['min_mem_dgx01'], 2)} / {f(r['min_mem_dgx02'], 2)} |")
    return h + "\n" + "\n".join(rows)


def report(res, image):
    """k59.txt + comment.md (issue body; {RESULTS_URL} is filled by the poster) from k59.json."""
    k = json.load(open(os.path.join(res, "k59.json")))
    rs = k["runs"]
    bad = [r["run"] for r in rs if not r["ok"]]
    p128 = max([r["prompt_tokens_max"] for r in rs if r["run"].startswith("d128k")] or [0])
    pmax = max([r["prompt_tokens_max"] for r in rs if r["run"].startswith("dmax")] or [0])
    tot = {c: sum(r["counts"][c] for r in rs) for c in ("exact", "near", "wrong", "explore", "no_call")}
    n = sum(r["expected"] for r in rs)
    pre = sum(r["preemptions"] or 0 for r in rs)
    mins = [x for r in rs for x in (r["min_mem_dgx01"], r["min_mem_dgx02"]) if x is not None]
    lines = [
        f"Long-context concurrency on the shipped 2x recipe (image {image}), measured {time.strftime('%Y-%m-%d', time.localtime(rs[0]['t0']))}.",
        "",
        "Setup: `scripts/fidelity_probe.py` with its defaults (thinking on, temperature 0.6, max_tokens 4096, 5 planted paths per "
        f"transcript). Two depths: `--depths 128000` ({p128:,} prompt tokens) and `--depths {k['dmax']}` ({pmax:,} prompt tokens, "
        "as close to the 262,144-token limit as a 4,096-token answer plus a follow-up turn allows).",
        "",
        "- `cN`: one transcript, `--concurrency N` (N requests at once; the prefix cache is warm after `c1`, so this tests decode under load).",
        "- `sN`: N probe processes at once, each with its own transcript and seed, 5 paths x 2 trials each. N different long contexts "
        "are resident together and all start cold.",
        "",
        table(rs),
        "",
        f"Totals over {len(rs)} runs: {tot['exact']} exact, {tot['near']} near, {tot['wrong']} wrong, {tot['explore']} explore, "
        f"{tot['no_call']} no_call out of {n} requests; {pre:.0f} preemptions; lowest MemAvailable {min(mins) if mins else '-'} GiB.",
        ("Every run returned all its requests without errors." if not bad else
         f"Runs with missing requests or request errors: {', '.join(bad)} (see run.json)."),
        "",
        "Columns: preemptions = `vllm:num_preemptions_total` delta over the run; prefix hit = cache hit tokens / queried tokens over the run; "
        "KV max = highest `vllm:kv_cache_usage_perc` sample (every 5 s); decode tok/s per request = completion tokens / time after the "
        "first token (single-turn requests only); output tok/s total = generated tokens / run wall time; MemAvailable from a 1 s "
        "sampler on both nodes.",
        "",
        "Raw data: {RESULTS_URL} (one JSON per probe process, `run.json` per run, guard logs).",
    ]
    open(os.path.join(res, "comment.md"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(res, "k59.txt"), "w").write("\n".join(f"{r['run']}: {line(r)}" for r in rs) + "\n")
    return len(rs), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res"); ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--only", default=""); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report", metavar="IMAGE")
    a = ap.parse_args()
    if a.report:
        nr, bad = report(a.res, a.report); print(f"report: {nr} runs, incomplete: {bad}"); return
    if a.selftest:
        vm.selftest(); fp.selftest()
        assert len(runs(130000)) == 10 and runs(130000)[5][1] == 130000
        r = {"run": "x", "depth": 1, "concurrency": 2, "sessions": 1, "expected": 20, "errors": 0, "prompt_tokens_max": 245000,
             "counts": {"exact": 20, "near": 0, "wrong": 0, "explore": 0, "no_call": 0}, "ttft_mean": 1.0, "ttft_max": 2.0,
             "decode_tps_mean": 50.0, "agg_output_tps": 90.0, "preemptions": None, "prefix_hit_rate": 0.98,
             "kv_usage_max": 0.07, "min_mem_dgx01": 12.0, "min_mem_dgx02": None}
        assert "| x | 1 x 2 | 245,000 | 20 |" in table([r]) and "98%" in line(r) and "preempt - |" in line(r)
        print("k59 selftest OK"); return
    os.makedirs(a.res, exist_ok=True)
    logf = open(os.path.join(a.res, "drive.log"), "a")
    def log(s):
        s = f"[{time.strftime('%F %T')}] {s}"; print(s, flush=True); logf.write(s + "\n"); logf.flush()
    only = set(filter(None, a.only.split(",")))
    out, dmax, ntoks = [], None, None
    for name, d, conc, seeds, trials in runs(0):
        if name.startswith("dmax"):
            if dmax is None:
                c1 = next((r for r in out if r["run"] == "d128k-c1" and r["prompt_tokens_max"]), None)
                if not c1:
                    log("no d128k-c1 prompt size: dmax runs skipped"); break
                dmax, ntoks = calibrate(a.base, c1["prompt_tokens_max"], [7, 41, 42, 51, 52, 53, 54], log)
                log(f"dmax depth {dmax} ({ntoks or '?'} tokens by /tokenize)")
            d = dmax
        if only and name not in only:
            continue
        log(f"run {name}: depth {d}, concurrency {conc}, seeds {seeds}, trials {trials}")
        out.append(one_run(a.base, a.res, name, d, conc, seeds, trials, log))
        json.dump({"dmax": dmax, "dmax_tokens": ntoks, "target": TARGET, "runs": out},
                  open(os.path.join(a.res, "k59.json"), "w"), indent=1)
    with open(os.path.join(a.res, "table.md"), "w") as fh:
        fh.write(table(out) + "\n")
    log(f"done: {sum(r['ok'] for r in out)}/{len(out)} runs complete")


if __name__ == "__main__":
    main()
