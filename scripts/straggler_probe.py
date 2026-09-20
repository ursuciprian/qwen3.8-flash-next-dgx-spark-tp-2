"""Straggler probe: per-request wall time + /metrics deltas at given concurrencies.

Recreated after /tmp/straggler.py was lost in the 2026-09-19 power cycle.
Matches the log format gate_arm.sh already parses (results/arms/*/straggler.log):

  c=N round wall Ts | per request (submit order): #0:Xs/320t #1:...
        preemptions +K | queue time total Qs over N req | prefill total Ps | decode total Ds | accept A.AA/draft

"counting prompt" workload (matches results/arms/*/straggler.log baselines) so
runs are comparable across boots. 320 output tokens, one round per concurrency
level (repeat externally if a less noisy number is needed).

Usage: python3 straggler_probe.py [c ...]   (default: 5 6 7 8 12 16)
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://localhost:8000"
MODEL = "qwen3.8-flash-next"
MAX_TOK = 320
PROMPT = "Count from 1 to 400, one number per line, nothing else."


def metrics():
    with urllib.request.urlopen(f"{BASE}/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or "{" not in line:
            continue
        name = line.split("{", 1)[0]
        try:
            out[name] = float(line.rsplit(" ", 1)[1])
        except ValueError:
            pass
    return out


def one_request(_):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOK,
        "temperature": 0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                  headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    t_first = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: "):])
            usage = chunk.get("usage")
            if usage:
                tokens = usage.get("completion_tokens", tokens)
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content") and t_first is None:
                t_first = time.monotonic()
    t_last = time.monotonic()
    return t0, t_first or t_last, t_last, tokens


def run_level(c):
    before = metrics()
    round_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=c) as pool:
        results = list(pool.map(one_request, range(c)))
    round_wall = time.monotonic() - round_start
    after = metrics()

    per_req = " ".join(
        f"#{i}:{t_last - t0:.2f}s/{tok}t" for i, (t0, _, t_last, tok) in enumerate(results)
    )
    prefill_total = sum(t_first - t0 for t0, t_first, _, _ in results)
    decode_total = sum(t_last - t_first for _, t_first, t_last, _ in results)

    def delta(key):
        return after.get(key, 0.0) - before.get(key, 0.0)

    preempt = delta("vllm:num_preemptions_total")
    accepted = delta("vllm:spec_decode_num_accepted_tokens_total")
    drafts = delta("vllm:spec_decode_num_drafts_total")
    accept_per_draft = accepted / drafts if drafts else 0.0
    queue_time = delta("vllm:request_queue_time_seconds_sum")

    print(f"c={c} round wall {round_wall:.1f}s | per request (submit order): {per_req}")
    print(f"      preemptions +{preempt:.0f} | queue time total {queue_time:.2f}s over {c} req "
          f"| prefill total {prefill_total:.2f}s | decode total {decode_total:.1f}s "
          f"| accept {accept_per_draft:.2f}/draft")


def main():
    levels = [int(x) for x in sys.argv[1:]] or [5, 6, 7, 8, 12, 16]
    for c in levels:
        run_level(c)


if __name__ == "__main__":
    main()
