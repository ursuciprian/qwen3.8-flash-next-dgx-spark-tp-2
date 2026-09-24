"""Decode cost of a tool-call grammar, paired on one boot.

Same tools request twice: plain (auto, non-strict: no grammar) and with
strict=true on the tools (auto + strict: the xgrammar triggered tag that
VLLM_TOOL_GRAMMAR_ALL=1 attaches to every tools request). The prompt asks for
prose without tools, so both arms decode the same free text and the delta is
the structured-output overhead (bitmask fill, deferred sampling / lost
async-scheduling overlap, draft validation). Needs the candidate vLLM
(auto + strict builds the tag there). Modes alternate per round.

usage: tool_grammar_cost.py [base_url] [rounds]
"""

import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.100.62:8000"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
LEVELS = (1, 4, 8)
MAX_TOK = 512
PROMPT = (
    "Do not call any tool. Write a detailed essay about the history of "
    "weather forecasting."
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": [next(iter(props))],
                "additionalProperties": False,
            },
        },
    }
    for name, desc, props in [
        (
            "get_weather",
            "Get current weather for a location",
            {
                "location": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
        ),
        ("web_search", "Search the web", {"query": {"type": "string"}}),
        ("calculator", "Evaluate arithmetic", {"expression": {"type": "string"}}),
    ]
]


def tools(strict):
    if not strict:
        return TOOLS
    return [{**t, "function": {**t["function"], "strict": True}} for t in TOOLS]


def one(strict, thinking):
    body = json.dumps(
        {
            "model": "qwen3.8-flash-next",
            "messages": [{"role": "user", "content": PROMPT}],
            "tools": tools(strict),
            "tool_choice": "auto",
            "max_tokens": MAX_TOK,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            obj = json.loads(line[6:])
            if obj.get("usage"):
                tokens = obj["usage"]["completion_tokens"]
            ch = obj.get("choices") or []
            d = (ch[0].get("delta") or {}) if ch else {}
            if first is None and (d.get("content") or d.get("reasoning")):
                first = time.perf_counter()
            if d.get("tool_calls"):
                print("WARN tool call emitted; prompt did not hold", flush=True)
    return tokens, (first or t0) - t0


def metric(name):
    text = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith(name) and not line.startswith("#")
    )


def level(c, strict, thinking):
    acc0 = metric("vllm:spec_decode_num_accepted_tokens_total")
    dr0 = metric("vllm:spec_decode_num_drafts_total")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(c) as ex:
        out = list(ex.map(lambda _: one(strict, thinking), range(c)))
    wall = time.perf_counter() - t0
    drafts = metric("vllm:spec_decode_num_drafts_total") - dr0
    acc = metric("vllm:spec_decode_num_accepted_tokens_total") - acc0
    return (
        sum(t for t, _ in out) / wall,
        statistics.median(f for _, f in out),
        acc / drafts if drafts else 0.0,
    )


one(False, False)
one(True, False)
print(f"{BASE} rounds={ROUNDS} max_tokens={MAX_TOK} temp=0")
print(f"{'think':<6}{'c':>3} {'arm':<7}{'agg tok/s':>11}{'ttft s':>8}{'acc/draft':>10}")
for thinking in (False, True):
    for c in LEVELS:
        res = {False: [], True: []}
        for r in range(ROUNDS):
            for strict in (r % 2 == 0, r % 2 != 0):
                res[strict].append(level(c, strict, thinking))
        for strict in (False, True):
            tps, ttft, acc = (statistics.median(x) for x in zip(*res[strict]))
            arm = "grammar" if strict else "plain"
            print(
                f"{str(thinking):<6}{c:>3} {arm:<7}{tps:>11.1f}{ttft:>8.2f}{acc:>10.2f}",
                flush=True,
            )
