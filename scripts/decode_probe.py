"""Decode-rate probe matching tonyd2wild's published method.

Streaming chat-completions, temp 0, warmed, measured on the head node against
the CX-7 address. decode = (completion_tokens - 1) / (t_last - t_first), which
excludes prefill and time-to-first-token.

Repeats each prompt class, because run-to-run variance on these boxes is large
and a single sample is noise.
"""
import json, statistics, sys, time, urllib.request

URL = "http://192.168.100.62:8000/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8-flash-next"
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAX_TOK = 512

PROMPTS = {
    "code": "Write a complete Python implementation of a red-black tree with insert, delete, and search. Include docstrings.",
    "structured": "Emit a JSON array of 40 objects, each with keys id (int), name (string), score (float), tags (array of 2 strings). No prose.",
    "counting": "Count from 1 to 200, one number per line, nothing else.",
    "prose": "Write a reflective essay about the experience of walking through a city at night.",
}


def run(prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOK,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                headers={"Content-Type": "application/json"})
    t_first = t_last = None
    completion = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            usage = obj.get("usage")
            if usage:
                completion = usage.get("completion_tokens", completion)
            ch = obj.get("choices") or []
            d = (ch[0].get("delta") or {}) if ch else {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
    if not t_first or not t_last or t_last <= t_first or completion < 2:
        return None
    return (completion - 1) / (t_last - t_first)


print("warming up", flush=True)
run(PROMPTS["code"])

print(f"{MODEL}  temp=0  max_tokens={MAX_TOK}  repeats={REPEATS}\n")
print(f"{'workload':<12} {'runs':<28} {'mean':>7} {'peak':>7}")
results = {}
for name, prompt in PROMPTS.items():
    vals = [v for v in (run(prompt) for _ in range(REPEATS)) if v]
    if not vals:
        print(f"{name:<12} FAILED", flush=True)
        continue
    results[name] = vals
    runs = " ".join(f"{v:.1f}" for v in vals)
    print(f"{name:<12} {runs:<28} {statistics.mean(vals):>7.1f} {max(vals):>7.1f}", flush=True)

if results:
    print(f"\npeak decode across workloads: {max(max(v) for v in results.values()):.1f} tok/s")
