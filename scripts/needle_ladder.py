#!/usr/bin/env python3
"""Needle-in-a-haystack ladder against an OpenAI-compatible endpoint.

For each depth, builds a deterministic prose haystack of roughly that many tokens, plants three
unrelated facts at 10%, 50% and 90% of it, then asks for each fact in a separate request with
thinking off and temperature 0. Scores exact containment of the answer token. Prints one line per
depth with actual prompt tokens (from usage), needles found, and first-token latency, and writes
a JSON summary. Exit code 1 if any depth scores below 3/3.

  needle_ladder.py --base http://host:8000 --model m --depths 8000,32000,64000,100000 --out x.json
"""
import argparse, json, random, sys, time, urllib.request

WORDS = ("harbor lantern granite meadow copper orchard willow quarry ember thistle saddle canvas "
         "ledger anvil compass furrow tallow gable mortar plinth cistern spindle rafter tether "
         "kiln bramble fathom marrow pewter ballast").split()
NEEDLES = [
    ("The access code for the Larkspur vault is 7391-KESTREL.", "What is the access code for the Larkspur vault?", "7391-KESTREL"),
    ("Dr. Imre Solvay's houseboat is moored at berth forty-two in Tromso.", "At which berth is Dr. Imre Solvay's houseboat moored, and in which city?", "forty-two"),
    ("The annual rainfall recorded at Station Quillon in 1987 was 611 millimetres.", "How much rainfall was recorded at Station Quillon in 1987?", "611"),
]


def haystack(chars, seed=7):
    rng = random.Random(seed); out = []; n = 0
    while n < chars:
        k = rng.randint(9, 17)
        s = " ".join(rng.choice(WORDS) for _ in range(k)).capitalize() + "."
        out.append(s); n += len(s) + 1
        if rng.random() < 0.08: out.append("\n"); n += 1
    return " ".join(out)


def plant(text, needles):
    parts = []
    marks = [0.10, 0.50, 0.90]
    last = 0
    for (fact, _, _), frac in zip(needles, marks):
        cut = text.find(". ", int(len(text) * frac)) + 2
        parts.append(text[last:cut]); parts.append(fact + " "); last = cut
    parts.append(text[last:])
    return "".join(parts)


def ask(base, model, prompt, question, timeout):
    body = {"model": model, "temperature": 0, "max_tokens": 48, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": "Answer from the document only, in one short sentence."},
                         {"role": "user", "content": f"<document>\n{prompt}\n</document>\n\n{question}"}]}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r["choices"][0]["message"]["content"], r["usage"]["prompt_tokens"], time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--depths", default="8000,32000,64000,100000")
    ap.add_argument("--chars-per-token", type=float, default=4.6)
    ap.add_argument("--timeout", type=int, default=1800); ap.add_argument("--out", default="needle.json")
    a = ap.parse_args()
    summary = {}; fail = False
    for depth in [int(x) for x in a.depths.split(",")]:
        doc = plant(haystack(int(depth * a.chars_per_token)), NEEDLES)
        hits = 0; toks = 0; lat = []; answers = []
        for fact, q, key in NEEDLES:
            try:
                ans, toks, dt = ask(a.base, a.model, doc, q, a.timeout)
            except Exception as ex:
                ans, dt = f"ERROR {type(ex).__name__}: {str(ex)[:120]}", 0.0
            ok = key.lower() in ans.lower(); hits += ok; lat.append(dt); answers.append((q, ans.strip()[:120], ok))
        fail |= hits < len(NEEDLES)
        print(f"depth {depth:>7} | prompt_tokens {toks:>7} | needles {hits}/{len(NEEDLES)} | "
              f"first request {lat[0]:6.1f}s, cached {min(lat[1:]) if len(lat)>1 else 0:5.1f}s", flush=True)
        for q, ans, ok in answers:
            if not ok: print(f"    MISS: {q} -> {ans!r}", flush=True)
        summary[depth] = {"prompt_tokens": toks, "hits": hits, "of": len(NEEDLES), "latency_s": lat, "answers": answers}
    json.dump(summary, open(a.out, "w"), indent=1)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
