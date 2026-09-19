#!/usr/bin/env python3
"""Prefix-caching correctness gate for vLLM on Qwen3.8-Flash-Next (vLLM #54173).

The reported failure: successive requests that share a prefix but GROW in length
(a normal multi-turn chat) kill the engine within 5-10 requests on GB10 with
--enable-prefix-caching. Output stays coherent until the crash. This is a gate,
not a timing run: any non-200, empty reply, token-0 burst or failed /health is a
FAIL and the arm must not be promoted.

  prefix_gate.py BASE MODEL CORPUS_TXT OUT_JSON

Phases: growing chat (20 turns), shrinking history (5), tool-result insertion (3),
5 concurrent growing chats (8 turns each). Records cached_tokens from usage so
"caching was actually active" is evidence, not assumption.
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

BASE, MODEL, CORPUS, OUT = sys.argv[1:5]
TEXT = open(CORPUS, encoding="utf-8", errors="ignore").read()
SLICE = 6000  # chars, ~1500 tokens
pos = [0]
results = {"requests": [], "fail": []}


def cache_hits():
    """Server-side prefix_cache_hits_total; usage.cached_tokens is 0 on this build."""
    try:
        m = urllib.request.urlopen(f"{BASE}/metrics", timeout=10).read().decode()
        return sum(float(l.split()[-1]) for l in m.splitlines()
                   if l.startswith("vllm:prefix_cache_hits_total") or l.startswith("sglang:cache_hit"))
    except Exception:
        return -1


hits0 = cache_hits()


def slice_(n=SLICE):
    s = TEXT[pos[0]:pos[0] + n]
    pos[0] = (pos[0] + n) % (len(TEXT) - n)
    return s


def health():
    try:
        return urllib.request.urlopen(f"{BASE}/health", timeout=10).status == 200
    except Exception:
        return False


def chat(messages, tag, max_tokens=48):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    rec = {"tag": tag, "turns": len(messages)}
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        content = (r["choices"][0]["message"].get("content") or "")
        u = r.get("usage", {})
        rec.update(status=200, s=round(time.time() - t0, 2), prompt=u.get("prompt_tokens"),
                   cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                   out=len(content), head=content[:60])
        if not content.strip() or not content.strip("!").strip():
            rec["fail"] = "empty-or-token0"
    except Exception as e:
        rec.update(status="ERR", s=round(time.time() - t0, 2), fail=str(e)[:200])
        content = ""
    if not health():
        rec["fail"] = (rec.get("fail") or "") + " health-down"
    results["requests"].append(rec)
    if rec.get("fail"):
        results["fail"].append(rec)
        print("FAIL", rec, flush=True)
    else:
        print(f"ok {tag} turns={rec['turns']} prompt={rec['prompt']} cached={rec['cached']} {rec['s']}s", flush=True)
    return content


def growing(tag, turns, msgs=None):
    msgs = msgs or []
    for i in range(turns):
        msgs.append({"role": "user", "content": slice_() + "\n\nContinue the story in two sentences."})
        msgs.append({"role": "assistant", "content": chat(msgs, f"{tag}-grow{i}") or "..."})
        if results["fail"] and "health-down" in results["fail"][-1].get("fail", ""):
            return msgs
    return msgs


msgs = growing("A", 20)
# shrinking: drop the last two exchanges, add a different slice - the cache now
# holds a longer prefix than the request.
for i in range(5):
    msgs = msgs[:-4]
    msgs.append({"role": "user", "content": slice_(3000) + "\n\nSummarise in one sentence."})
    msgs.append({"role": "assistant", "content": chat(msgs, f"A-shrink{i}") or "..."})
# tool-result insertion into the shared history
for i in range(3):
    msgs.append({"role": "user", "content": "Look up the weather in Baker Street."})
    msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"call{i}", "type": "function",
                 "function": {"name": "weather", "arguments": json.dumps({"city": "London", "n": i})}}]})
    msgs.append({"role": "tool", "tool_call_id": f"call{i}", "content": json.dumps({"temp_c": 11 + i, "sky": "fog" * (i + 1)})})
    msgs.append({"role": "assistant", "content": chat(msgs, f"A-tool{i}") or "..."})
# concurrency: five independent growing chats at once
if not any("health-down" in f.get("fail", "") for f in results["fail"]):
    with cf.ThreadPoolExecutor(5) as ex:
        list(ex.map(lambda k: growing(f"C{k}", 8), range(5)))

cached = [r["cached"] for r in results["requests"] if r.get("cached")]
results["summary"] = {"requests": len(results["requests"]), "failures": len(results["fail"]),
                      "cached_hits": len(cached), "max_cached_tokens": max(cached) if cached else 0,
                      "server_prefix_hit_tokens": cache_hits() - hits0,
                      "health": health(), "verdict": "PASS" if not results["fail"] and health() else "FAIL"}
json.dump(results, open(OUT, "w"), indent=1)
print("GATE", json.dumps(results["summary"]), flush=True)
sys.exit(0 if results["summary"]["verdict"] == "PASS" else 1)
