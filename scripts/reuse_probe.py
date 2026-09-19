#!/usr/bin/env python3
"""First-pass prefix-cache reuse probe: one ~20k-token prompt, sent three times.

  reuse_probe.py BASE MODEL CORPUS_TXT [OUT_JSON]

Request 1 is fresh, request 2 identical, request 3 identical except the last
line. A working cache makes request 2 fast with a large hit delta; a cache that
only saves state on the second pass makes request 2 slow and request 3 fast
(observed 2026-09-07: 44.6s / 7.4s / 1.6s, hits 0 / 0 / 16000). Run with no
other client on the server. Uses server-side vllm:prefix_cache_hits_total
because usage.cached_tokens is 0 on this build.
"""
import json, sys, time, urllib.request
B, M, CORPUS = sys.argv[1:4]
OUT = sys.argv[4] if len(sys.argv) > 4 else None
txt = open(CORPUS, encoding="utf-8", errors="ignore").read()[100000:180000]
def counters():
    m = urllib.request.urlopen(B + "/metrics", timeout=10).read().decode()
    d = {}
    for l in m.splitlines():
        for k in ("prefix_cache_hits_total", "prefix_cache_queries_total"):
            if l.startswith("vllm:" + k):
                d[k] = d.get(k, 0) + float(l.split()[-1])
    return d
def req(tail):
    body = json.dumps({"model": M, "messages": [{"role": "user", "content": txt + tail}], "max_tokens": 4,
                       "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600)
    r.readline(); ttft = round(time.time() - t, 2); r.read()
    return ttft
rows = []
for i, tail in enumerate(["\nSummarise.", "\nSummarise.", "\nList three names."]):
    c0 = counters(); ttft = req(tail); c1 = counters()
    row = {"req": i + 1, "ttft_s": ttft, "hits": int(c1["prefix_cache_hits_total"] - c0["prefix_cache_hits_total"]),
           "queries": int(c1["prefix_cache_queries_total"] - c0["prefix_cache_queries_total"])}
    rows.append(row); print(f"req{row['req']} ttft={ttft}s hits+{row['hits']} queries+{row['queries']}", flush=True)
verdict = "FIRST_PASS_REUSE" if rows[1]["hits"] > rows[1]["queries"] // 2 else ("SECOND_PASS_ONLY" if rows[2]["hits"] > 0 else "NO_REUSE")
print("REUSE", verdict, flush=True)
if OUT: json.dump({"rows": rows, "verdict": verdict}, open(OUT, "w"), indent=1)
