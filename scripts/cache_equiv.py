#!/usr/bin/env python3
"""Cached-vs-fresh output equivalence probe.

  cache_equiv.py BASE MODEL CORPUS_TXT OUT_JSON [N_PROMPTS]

For N distinct prompts (corpus slices of 6k-20k tokens, each ending in the same
short instruction) send request A fresh, then request B identical, greedy,
64 tokens, no other client. Report exact-match rate and first divergence
position. Then, for the same prompts, send request C: identical prefix but a
different final line, and compare its first 8 tokens against a fresh run of the
same request on a cold prefix (D). B measures "cached == fresh"; C/D measures
"cached prefix + new suffix == fresh". Greedy decoding on this stack is not
bit-reproducible above ~2k prompt tokens (vLLM #54173 comment), so a few
divergences are noise; run the same probe on two configs and compare rates.
"""
import json, sys, time, urllib.request
B, M, CORPUS, OUT = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 10
TEXT = open(CORPUS, encoding="utf-8", errors="ignore").read()
def chat(content, max_tokens=64):
    body = json.dumps({"model": M, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
                       "temperature": 0, "seed": 1, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", body,
                                                                 {"Content-Type": "application/json"}), timeout=600))
    return (r["choices"][0]["message"].get("content") or ""), round(time.time() - t, 2), r.get("usage", {}).get("prompt_tokens")
def diverge(a, b):
    for i, (x, y) in enumerate(zip(a.split(), b.split())):
        if x != y: return i
    return None if a.split() == b.split() else min(len(a.split()), len(b.split()))
rows = []
for i in range(N):
    size = 24000 + (i % 4) * 16000  # 6k-18k tokens
    start = (50000 + i * 90000) % max(1, len(TEXT) - size)  # wrap instead of running off the corpus
    prefix = TEXT[start:start + size]
    a, ta, pt = chat(prefix + "\n\nContinue the story in three sentences.")
    b, tb, _ = chat(prefix + "\n\nContinue the story in three sentences.")
    c, tc, _ = chat(prefix + "\n\nName the characters mentioned above.")
    rows.append({"i": i, "prompt_tokens": pt, "ttft_fresh": ta, "ttft_cached": tb, "AB_match": a == b,
                 "AB_first_divergence_word": diverge(a, b), "A": a[:200], "B": b[:200], "C": c[:200], "ttft_C": tc})
    print(f"{i} prompt={pt} fresh={ta}s cached={tb}s match={a == b} div@{diverge(a, b)}", flush=True)
summary = {"n": N, "exact_match": sum(r["AB_match"] for r in rows),
           "divergences": [r["AB_first_divergence_word"] for r in rows if not r["AB_match"]]}
json.dump({"rows": rows, "summary": summary}, open(OUT, "w"), indent=1)
print("EQUIV", json.dumps(summary), flush=True)
