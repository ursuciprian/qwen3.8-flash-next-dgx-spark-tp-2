#!/usr/bin/env python3
"""k64 prefill client (2026-10-07, #101): c1 chat requests of ~2K prompt tokens, max_tokens 1, temperature 0, thinking off;
reports TTFT per request (time to the first streamed chunk with content or the final chunk) and prompt_tokens.
Prompts are code from results/corpus-code.txt at fixed, distinct offsets (no prefix-cache hits between requests).

  pp.py --base URL --corpus FILE --offset N --n K --out OUT.json
  pp.py --selftest
"""
import argparse, json, statistics, sys, time, urllib.request

CHARS = 6800  # ~2K tokens of code


def one(base, text):
    body = json.dumps({"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "Summarize this code:\n" + text}],
                       "max_tokens": 1, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.time(); ttft = None; ptok = None
    r = urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600)
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ttft = ttft or time.time() - t0
        u = json.loads(line[6:]).get("usage")
        if u:
            ptok = u.get("prompt_tokens")
    return {"ttft_s": ttft, "prompt_tokens": ptok}


def summary(rows):
    t = [r["ttft_s"] for r in rows if r["ttft_s"]]
    p = [r["prompt_tokens"] for r in rows if r["prompt_tokens"]]
    return {"n": len(t), "ttft_mean_s": statistics.mean(t) if t else None, "ttft_min_s": min(t) if t else None,
            "prompt_tokens_mean": statistics.mean(p) if p else None,
            "prefill_tok_s": (statistics.mean(p) / statistics.mean(t)) if t and p else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--corpus"); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--n", type=int, default=5); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        assert summary([{"ttft_s": 1.0, "prompt_tokens": 2000}, {"ttft_s": 1.0, "prompt_tokens": 2000}])["prefill_tok_s"] == 2000
        assert summary([{"ttft_s": None, "prompt_tokens": None}])["n"] == 0
        print("pp selftest ok"); return
    c = open(a.corpus, errors="replace").read()
    rows = [one(a.base, c[a.offset + i * CHARS * 2: a.offset + i * CHARS * 2 + CHARS]) for i in range(a.n)]
    json.dump({"rows": rows, "summary": summary(rows)}, open(a.out, "w"), indent=1)
    print(json.dumps(summary(rows)))


if __name__ == "__main__":
    main()
