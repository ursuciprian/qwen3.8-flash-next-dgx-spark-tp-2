#!/usr/bin/env python3
"""Decode at depth, per stream chunk: separates kernel slowdown from stalls.

llama-benchy reports one mean tok/s per cell, which cannot tell a uniformly
slower decode step from a few multi-second stalls or lower MTP acceptance.
This probe replays its depth pattern (context-load request, then the same
context + new prompt) and records, per request:

  ttft, decode tok/s = (completion-1)/(t_last-t_first), every inter-chunk gap,
  gaps over --stall-ms, tok/s with those gaps removed, and /metrics deltas
  (spec-decode accepted/draft, prefix-cache hit tokens).

Token counts are exact: contexts are trimmed with the server's /tokenize.
Each repeat uses a distinct corpus slice, so repeats never hit each other's
prefix cache. --trigger-cmd runs once, mid-decode of the chosen request
(used to arm mods/vllm-decode-profiler on both ranks).

  python3 depth_decode_probe.py --depth 16384 --new 2048 --repeats 6 \
      --concurrency 1 --out res.json
"""
import argparse
import json
import statistics
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

COUNTERS = {
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    "drafts": "vllm:spec_decode_num_draft_tokens_total",
    "prefix_hits": "vllm:prefix_cache_hits_total",
    "prefix_queries": "vllm:prefix_cache_queries_total",
}


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def metrics(base):
    text = urllib.request.urlopen(f"{base}/metrics", timeout=10).read().decode()
    out = dict.fromkeys(COUNTERS, 0.0)
    for line in text.splitlines():
        for key, name in COUNTERS.items():
            if line.startswith(name):
                out[key] += float(line.rsplit(" ", 1)[1])
    return out


def ntokens(base, model, text):
    with post(f"{base}/tokenize", {"model": model, "prompt": text}, 60) as r:
        return json.load(r)["count"]


def trim(base, model, text, target):
    """Longest prefix of text (whole chars) with at most target tokens."""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ntokens(base, model, text[:mid]) <= target:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def stream(base, model, content, args, on_chunk=None):
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": args.max_tokens, "min_tokens": args.max_tokens,
            "ignore_eos": True, "temperature": args.temperature, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    times, usage = [], {}
    with post(f"{base}/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            obj = json.loads(line[6:])
            usage = obj.get("usage") or usage
            ch = obj.get("choices") or []
            d = (ch[0].get("delta") or {}) if ch else {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                times.append(time.perf_counter())
                if on_chunk:
                    on_chunk(len(times))
    comp = usage.get("completion_tokens", 0)
    gaps = [b - a for a, b in zip(times, times[1:])]
    stalls = [g for g in gaps if g * 1000 > args.stall_ms]
    span = (times[-1] - times[0]) if len(times) > 1 else 0.0
    return {
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": comp,
        "ttft_s": (times[0] - t0) if times else None,
        "decode_tps": (comp - 1) / span if span > 0 else None,
        "decode_tps_no_stalls": (comp - 1) / (span - sum(stalls)) if span - sum(stalls) > 0 else None,
        "gap_ms_p50": 1000 * statistics.median(gaps) if gaps else None,
        "gap_ms_p95": 1000 * sorted(gaps)[int(0.95 * (len(gaps) - 1))] if gaps else None,
        "gap_ms_max": 1000 * max(gaps) if gaps else None,
        "stalls": len(stalls), "stall_s": sum(stalls),
    }


def one_pair(base, model, corpus, args, index, trigger):
    stride = len(corpus) // 3 // max(1, args.repeats * args.concurrency + 1)
    start = args.offset + index * stride
    ctx = trim(base, model, corpus[start:start + 8 * args.depth], args.depth)
    new = trim(base, model, corpus[start + len(ctx):start + len(ctx) + 8 * args.new], args.new)
    out = {"index": index}
    for phase, content in (("ctx", ctx), ("inf", ctx + "\n\n" + new)):
        hook = None
        if trigger and phase == "inf":
            fired = threading.Event()

            def hook(n, fired=fired):
                if n >= args.trigger_after_chunks and not fired.is_set():
                    fired.set()
                    subprocess.Popen(trigger, shell=True)
        m0 = metrics(base)
        res = stream(base, model, content, args, hook)
        m1 = metrics(base)
        delta = {k: m1[k] - m0[k] for k in COUNTERS}
        res["accept_rate"] = delta["accepted"] / delta["drafts"] if delta["drafts"] else None
        res["prefix_hit_tokens"] = delta["prefix_hits"]
        out[phase] = res
    return out


def summarize(rows, phase):
    vals = lambda k: [r[phase][k] for r in rows if r[phase].get(k) is not None]
    s = {}
    for k in ("decode_tps", "decode_tps_no_stalls", "ttft_s", "gap_ms_p50", "gap_ms_p95",
              "gap_ms_max", "accept_rate", "prefix_hit_tokens", "prompt_tokens"):
        v = vals(k)
        if v:
            s[k] = {"mean": statistics.mean(v), "min": min(v), "max": max(v)}
    s["stalls_total"] = sum(vals("stalls"))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--corpus", default="results/corpus-prose.txt")
    ap.add_argument("--depth", type=int, default=16384)
    ap.add_argument("--new", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--stall-ms", type=float, default=250.0)
    ap.add_argument("--trigger-cmd", default="")
    ap.add_argument("--trigger-repeat", type=int, default=2)
    ap.add_argument("--trigger-after-chunks", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0,
                    help="corpus char offset; give each probe in one boot its own so they share no prefix")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # Tripled so late slices never run off the end; distinct start offsets
    # already guarantee distinct prefixes, overlap in content is harmless.
    corpus = open(args.corpus, errors="ignore").read() * 3

    rows = []
    for rep in range(args.repeats):
        trigger = args.trigger_cmd if (args.trigger_cmd and rep == args.trigger_repeat) else ""
        with ThreadPoolExecutor(args.concurrency) as pool:
            futs = [pool.submit(one_pair, args.base, args.model, corpus, args,
                                rep * args.concurrency + i, trigger if i == 0 else "")
                    for i in range(args.concurrency)]
            batch = [f.result() for f in futs]
        rows.extend(batch)
        for r in batch:
            print(json.dumps({"rep": rep, **{p: {k: r[p][k] for k in (
                "decode_tps", "decode_tps_no_stalls", "ttft_s", "stalls",
                "gap_ms_max", "accept_rate", "prefix_hit_tokens")} for p in ("ctx", "inf")}}),
                flush=True)
    report = {"label": args.label, "args": vars(args), "rows": rows,
              "summary": {p: summarize(rows, p) for p in ("ctx", "inf")}}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
