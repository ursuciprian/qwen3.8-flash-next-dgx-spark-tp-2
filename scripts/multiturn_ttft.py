#!/usr/bin/env python3
"""Continuation-shaped TTFT probe: does turn 2 of a real conversation reuse turn 1?

llama-benchy's depth mode measures the wrong shape for `--prefix-match-unit`:
its "context load" request ends with a short probe message, so the fine tail
state it publishes sits PAST the point where the measured request diverges and
can never be matched (see results/kernel-pass/long-prefix-ttft.md, F2). Real
agent traffic is a continuation -- turn N+1 contains turn N's prompt AND its
answer verbatim -- so turn 1's fine tail IS inside turn 2's prefix.

Per depth d:
  turn 1  = [user: d-token context + question]                 -> answer (tg 128)
  turn 2  = [user: same, assistant: answer, user: ~200 new tokens]
and TTFT is measured on both. Turn 2 is the number that matters: with a working
fine tail it should be ~(200 new tokens + tokenization), not ~(one 2864-block
re-prefill + 200).

Exact token depths come from the server's own /tokenize + /detokenize, so no
tokenizer dependency and no drift against the chat template.

  multiturn_ttft.py --base http://localhost:8000 --model qwen3.8-flash-next \
                    --depths 16384 65536 --trials 3
"""
import argparse
import json
import os
import random
import statistics
import time
import urllib.request

NEW_USER_TOKENS = 200
QUESTION = "\n\nSummarise the passage above in one sentence."


def post(base, path, payload, timeout=1800, stream=False):
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp if stream else json.loads(resp.read())


def tokenize(base, model, text):
    return post(base, "/tokenize", {"model": model, "prompt": text})["tokens"]


def detokenize(base, model, tokens):
    return post(base, "/detokenize", {"model": model, "tokens": list(tokens)})["prompt"]


def exact_tokens(base, model, pool_tokens, start, n):
    """Text whose tokenization is n tokens, cut from the corpus token pool."""
    return detokenize(base, model, pool_tokens[start : start + n])


def chat_ttft(base, model, messages, max_tokens):
    """Stream a chat completion; return (ttft_seconds, full_text)."""
    t0 = time.perf_counter()
    resp = post(
        base,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        },
        stream=True,
    )
    ttft = None
    out = []
    thought = []
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        chunk = json.loads(body)
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            piece = delta.get("content") or ""
            think = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if (piece or think) and ttft is None:
                ttft = time.perf_counter() - t0  # first token of either kind
            if piece:
                out.append(piece)
            elif think:
                thought.append(think)
    resp.close()
    if ttft is None:
        raise RuntimeError("no content chunk in stream")
    return ttft, ("".join(out) or "".join(thought))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--depths", type=int, nargs="+", default=[16384, 65536])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument(
        "--corpus",
        default=os.path.join(here, "..", "results", "corpus-prose.txt"),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="also write the raw per-trial numbers here")
    args = ap.parse_args()

    with open(args.corpus, encoding="utf-8", errors="ignore") as fh:
        pool = tokenize(args.base, args.model, fh.read())
    print(f"corpus: {len(pool)} tokens from {os.path.relpath(args.corpus, here)}")

    rng = random.Random(args.seed)
    q_len = len(tokenize(args.base, args.model, QUESTION))
    rows = []

    print(f"\n{'depth':>7} {'trial':>6} {'turn1 TTFT':>11} {'turn2 TTFT':>11}")
    for depth in args.depths:
        ctx_len = depth - q_len
        if ctx_len <= 0 or ctx_len + NEW_USER_TOKENS >= len(pool):
            raise SystemExit(f"corpus too small for depth {depth}")
        t1s, t2s = [], []
        for trial in range(args.trials):
            # fresh slice per trial so turn 1 is always cold
            start = rng.randrange(0, len(pool) - ctx_len - NEW_USER_TOKENS)
            ctx = exact_tokens(args.base, args.model, pool, start, ctx_len)
            turn1_user = ctx + QUESTION
            t1, answer = chat_ttft(
                args.base, args.model, [{"role": "user", "content": turn1_user}], args.tg
            )
            new_user = exact_tokens(
                args.base, args.model, pool, start + ctx_len, NEW_USER_TOKENS
            )
            t2, _ = chat_ttft(
                args.base,
                args.model,
                [
                    {"role": "user", "content": turn1_user},
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": new_user},
                ],
                args.tg,
            )
            t1s.append(t1)
            t2s.append(t2)
            rows.append(
                {"depth": depth, "trial": trial, "turn1_s": t1, "turn2_s": t2}
            )
            print(f"{depth:>7} {trial:>6} {t1:>10.3f}s {t2:>10.3f}s")
        print(
            f"{depth:>7} {'median':>6} {statistics.median(t1s):>10.3f}s "
            f"{statistics.median(t2s):>10.3f}s"
        )

    print(f"\n{'depth':>7} {'turn1 median':>13} {'turn2 median':>13} {'turn2/turn1':>12}")
    for depth in args.depths:
        t1 = statistics.median(r["turn1_s"] for r in rows if r["depth"] == depth)
        t2 = statistics.median(r["turn2_s"] for r in rows if r["depth"] == depth)
        print(f"{depth:>7} {t1:>12.3f}s {t2:>12.3f}s {t2 / t1:>11.2f}x")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
