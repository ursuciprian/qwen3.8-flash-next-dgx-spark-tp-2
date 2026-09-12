#!/usr/bin/env python3
"""Reproduce sgl-project/sglang#37111 on this deployment: QSA + NEXTN decode CUDA graphs
returning HTTP 200 while silently corrupting output on GB10 TP2.

Two tests from the report, plus a repetition detector, because the failure mode is a 200 with
semantically invalid content and no error anywhere.

  A. 1024-token essay. Report: degenerates into repeated punctuation ('!').
  B. ~25k-token prompt carrying four ordered markers. Report: retrieval fails and the answer
     degenerates the same way.

    gate_37111.py BASE MODEL [--thinking]

Exit 0 = both pass. Exit 1 = corruption reproduced here.
"""
import json, re, sys, urllib.request

BASE, MODEL = sys.argv[1].rstrip("/").removesuffix("/v1"), sys.argv[2]  # accept base with or without /v1
THINK = "--thinking" in sys.argv


def ask(prompt, max_tokens):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": THINK}}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=900))
    c = r["choices"][0]
    return c["message"].get("content") or "", c.get("finish_reason"), r["usage"]["completion_tokens"]


def degenerate(text):
    """True when the text looks like the reported collapse, not like prose."""
    s = text.strip()
    if len(s) < 40:
        return True, "shorter than 40 chars"
    punct = sum(c in "!?.,;:-*#" for c in s) / len(s)
    if punct > 0.30:
        return True, f"punctuation {punct:.0%} of characters"
    for run in re.findall(r"(.{1,20}?)\1{5,}", s):          # a short string repeated 6+ times
        return True, f"repeated fragment {run!r}"
    words = re.findall(r"[A-Za-z']+", s)
    if len(words) >= 40 and len(set(w.lower() for w in words)) / len(words) < 0.15:
        return True, f"only {len(set(w.lower() for w in words))} distinct words in {len(words)}"
    return False, ""


fails = []

# A. long non-looping generation
print("A. 1024-token essay", flush=True)
text, finish, n = ask(
    "Write a continuous 1000-word essay on how coastal lighthouses were built and maintained "
    "in the nineteenth century. Use ordinary prose paragraphs. Do not use lists or headings.",
    1024)
bad, why = degenerate(text)
print(f"   {n} tokens, finish={finish}, {'CORRUPT: ' + why if bad else 'clean'}")
print(f"   tail: {text[-160:]!r}")
if bad:
    fails.append(f"A: {why}")

# B. four ordered markers in a long prompt
print("\nB. ~25k-token prompt, four ordered markers", flush=True)
MARKERS = ["ALPHA-4417", "BRAVO-9082", "CHARLIE-3351", "DELTA-7765"]
filler = ("The keeper trimmed the wick and logged the tide, the wind, and every ship that passed "
          "the headland before the fog closed in for the night. ")
block = filler * 130                                        # ~4k tokens per block
prompt = ("Read the log below. Four codes are embedded in it.\n\n"
          + block + f"\nFIRST CODE: {MARKERS[0]}\n" + block
          + f"\nSECOND CODE: {MARKERS[1]}\n" + block
          + f"\nTHIRD CODE: {MARKERS[2]}\n" + block
          + f"\nFOURTH CODE: {MARKERS[3]}\n" + block
          + "\nList the four codes in the order they appeared, one per line, nothing else.")
text, finish, n = ask(prompt, 200)
found = [m for m in MARKERS if m in text]
order_ok = found == MARKERS
bad, why = degenerate(text)
print(f"   {n} tokens, finish={finish}, found {len(found)}/4, order {'ok' if order_ok else 'WRONG'}")
print(f"   answer: {text.strip()[:200]!r}")
if not order_ok:
    fails.append(f"B: retrieved {found} expected {MARKERS}")
if bad:
    fails.append(f"B: {why}")

print()
if fails:
    print("GATE FAILED - #37111 reproduced here:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("GATE PASSED - no silent corruption on this deployment")
