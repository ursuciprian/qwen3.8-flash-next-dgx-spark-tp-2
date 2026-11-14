#!/usr/bin/env python3
"""k107fix identity under load (stdlib): 16 greedy completions sent at once, 3 rounds per boot, compared within the boot.

  identc.py run --base URL --corpus FILE --out OUT.json    16 prompts (corpus windows of 1K..12K chars plus a question),
                                                           temperature 0, 160 new tokens, all 16 in flight together
  identc.py report <screen dir>                            reads {ctl,arm}-p*/identc.json
  identc.py selftest

Why: the GDN decode CTA race (FlashInfer c9b29c91) can only fire when a CTA runs more than one persistent work item,
and the grouped multi-token path (MTP verify with more than one sequence) had no barrier at all in our fork. One
request at a time (k101/ident.py) never reaches it. At 16 requests in flight, a prompt whose completion changes from
round to round inside one boot is unstable. Batch composition also changes between rounds, so the control is not
expected to be fully stable; the judge compares the two sides.
Per boot: unstable = prompts whose 3 completions are not all equal. Per side: mean unstable count over its boots.
Last line IDENTC=OK (arm mean <= control mean + 1) | WORSE (arm less stable than the control by more than one prompt)
| EMPTY (missing rounds or fewer than 2 boots per side).
"""
import concurrent.futures as cf
import glob
import json
import os
import statistics
import sys
import urllib.request

MODEL = "qwen3.8-flash-next"
WINDOWS = tuple((7_000 + 27_000 * i, 1000 + 700 * (i % 16)) for i in range(16))
ASK = "\n\nIn two sentences, what does the code above do?"
ROUNDS = 3


def one(base, prompt):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 160, "temperature": 0, "seed": 0}
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["choices"][0]["text"]
    except Exception as e:  # a failed request is a missing cell, never a match
        print(f"request failed: {e}", file=sys.stderr)
        return None


def run(base, corpus, out):
    text = open(corpus, errors="replace").read()
    prompts = [f"[{i}] " + text[off:off + n] + ASK for i, (off, n) in enumerate(WINDOWS)]
    rounds = []
    with cf.ThreadPoolExecutor(len(prompts)) as ex:
        for _ in range(ROUNDS):
            rounds.append(list(ex.map(lambda p: one(base, p), prompts)))
    json.dump({"rounds": rounds}, open(out, "w"))
    print(f"identc: {sum(t is not None for r in rounds for t in r)}/{ROUNDS * len(prompts)} completions")


def unstable(rounds):
    """count of prompts whose completions differ across rounds, or None when a completion is missing"""
    if len(rounds) != ROUNDS or any(t is None for r in rounds for t in r):
        return None
    return sum(len({r[i] for r in rounds}) > 1 for i in range(len(rounds[0])))


def report(node):
    side, lines = {}, []
    for who in ("ctl", "arm"):
        for p in sorted(glob.glob(os.path.join(node, f"{who}-p*", "identc.json"))):
            try:
                u = unstable(json.load(open(p))["rounds"])
            except (OSError, ValueError, KeyError):
                u = None
            lines.append(f"  {os.path.basename(os.path.dirname(p))}: {'missing rounds' if u is None else f'{u}/{len(WINDOWS)} unstable'}")
            if u is not None:
                side.setdefault(who, []).append(u)
    c, a = side.get("ctl", []), side.get("arm", [])
    if len(c) < 2 or len(a) < 2:
        return lines + [f"identc: {len(c)} control / {len(a)} arm boots with every round"], "EMPTY"
    mc, ma = statistics.mean(c), statistics.mean(a)
    lines.append(f"  mean unstable prompts per boot: control {mc:.2f}, arm {ma:.2f} (of {len(WINDOWS)})")
    return lines, ("OK" if ma <= mc + 1 else "WORSE")


def selftest():
    import tempfile
    t = tempfile.mkdtemp()
    def mk(name, flips):
        d = os.path.join(t, name); os.makedirs(d)
        base = [f"text {i}" for i in range(len(WINDOWS))]
        rounds = [list(base) for _ in range(ROUNDS)]
        for i in range(flips):
            rounds[2][i] = "other"
        json.dump({"rounds": rounds}, open(os.path.join(d, "identc.json"), "w"))
    mk("ctl-p1", 3); mk("ctl-p2", 2); mk("arm-p1", 0); mk("arm-p2", 1)
    lines, v = report(t)
    assert v == "OK", (lines, v)
    mk("arm-p3", 12)
    assert report(t)[1] == "WORSE"
    assert unstable([["a"], ["a"], [None]]) is None and unstable([["a"], ["b"], ["a"]]) == 1
    print("identc selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k: a[a.index(k) + 1] if k in a else None
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["run"]:
        run(opt("--base"), opt("--corpus"), opt("--out"))
    elif a[:1] == ["report"] and len(a) == 2:
        lines, v = report(a[1])
        print("\n".join(lines))
        print(f"IDENTC={v}")
    else:
        sys.exit(__doc__)
