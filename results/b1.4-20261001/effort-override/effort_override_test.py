#!/usr/bin/env python3
"""Which request fields override the server default reasoning_effort (b1.4 promotion, opus-kernel-14).

prompt_tokens is the exact signal: at medium the template adds no system sentence, at xhigh/low it adds one.
Thinking tokens = /tokenize of reasoning_content. Server default sampling, 2 runs per prompt x condition.
"""
import json, statistics, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
PROMPTS = [
    "Is 1001 a prime number? Answer yes or no with one line of justification.",
    "Write a bash one-liner that prints the 5 largest files under the current directory.",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
]
CONDS = {
    "a_none": {},
    "b_top_xhigh": {"reasoning_effort": "xhigh"},
    "c_ctk_xhigh": {"chat_template_kwargs": {"reasoning_effort": "xhigh"}},
    "d_top_medium": {"reasoning_effort": "medium"},
    "e_top_low+ctk_xhigh": {"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "xhigh"}},
    "f_top_none": {"reasoning_effort": "none"},
    "g_top_high": {"reasoning_effort": "high"},
}
RUNS = 2


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def one(job):
    cond, pi, run = job
    body = {"model": "qwen3.8-flash-next", "max_tokens": 6144,
            "messages": [{"role": "user", "content": PROMPTS[pi]}], **CONDS[cond]}
    try:
        r = post("/v1/chat/completions", body)
    except Exception as e:  # a 400 from the template is a result too
        return {"cond": cond, "prompt": pi, "run": run, "error": str(e)[:200]}
    m = r["choices"][0]["message"]
    think = m.get("reasoning_content") or m.get("reasoning") or ""
    tt = len(post("/tokenize", {"model": "qwen3.8-flash-next", "prompt": think, "add_special_tokens": False})["tokens"]) if think else 0
    return {"cond": cond, "prompt": pi, "run": run, "prompt_tokens": r["usage"]["prompt_tokens"],
            "completion_tokens": r["usage"]["completion_tokens"], "thinking_tokens": tt,
            "finish": r["choices"][0]["finish_reason"], "answer": (m.get("content") or "")[:120]}


jobs = [(c, p, k) for c in CONDS for p in range(len(PROMPTS)) for k in range(RUNS)]
with ThreadPoolExecutor(5) as ex:
    rows = list(ex.map(one, jobs))
for row in rows:
    print(json.dumps(row))
print("\ncond                  prompt_tokens(p0/p1/p2)   thinking tokens median (per prompt)   errors")
for c in CONDS:
    rs = [r for r in rows if r["cond"] == c]
    pt = "/".join(str(sorted({r["prompt_tokens"] for r in rs if r["prompt"] == p and "error" not in r})) for p in range(len(PROMPTS)))
    th = "/".join(str(int(statistics.median([r["thinking_tokens"] for r in rs if r["prompt"] == p and "error" not in r] or [0]))) for p in range(len(PROMPTS)))
    print(f"{c:22s}{pt:26s}{th:38s}{sum('error' in r for r in rs)}")
