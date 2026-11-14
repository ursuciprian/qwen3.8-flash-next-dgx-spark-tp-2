#!/usr/bin/env python3
"""k120 identity (stdlib): copy of k101/ident.py with one rule change for a non-bit-exact arm.

  ident.py run --base URL --corpus FILE --out OUT.json     8 prompts (corpus windows of 1K..24K chars plus a question),
                                                           temperature 0, 96 new tokens, one at a time; records each
                                                           prompt's prompt_tokens (usage)
  ident.py report <node dir>                               reads <node>/{ctl,arm}-p*/hook/ident.json
  ident.py selftest

The k120 arm's BF16 HC down-projection path runs for prefills of SHORT_MAX (1264) rows or more and matches the A16
kernel only up to the order of the FP32 additions, so a long prompt's greedy completion may legitimately differ from the
control. Only prompts under SHORT_MAX prompt tokens (largest count over all boots) are judged: there the arm runs the
same kernels as the control, so a prompt every control boot reproduces must come out the same on every arm boot. Long
prompts are reported (identical / different), not judged. Prompts where the control boots disagree among themselves
are counted but not judged. A prompt without a recorded prompt_tokens count is treated as long (not judged).
Last line IDENT=PASS|FAIL|UNSTABLE (no control-stable short prompt)|EMPTY.
"""
import glob
import json
import os
import sys
import urllib.request

MODEL = "qwen3.8-flash-next"
SHORT_MAX = 1264   # b12x _a16_dequant.MIN_ROWS at 35573bc4
WINDOWS = ((0, 1000), (5_000, 3000), (40_000, 6000), (90_000, 12000), (150_000, 24000),
           (260_000, 2000), (300_000, 9000), (420_000, 16000))
ASK = "\n\nIn two sentences, what does the code above do?"


def run(base, corpus, out):
    text = open(corpus, errors="replace").read()
    res, ptoks = [], []
    for i, (off, n) in enumerate(WINDOWS):
        body = {"model": MODEL, "prompt": f"[{i}] " + text[off:off + n] + ASK, "max_tokens": 96,
                "temperature": 0, "seed": 0}
        req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                j = json.load(r)
            res.append(j["choices"][0]["text"])
            ptoks.append((j.get("usage") or {}).get("prompt_tokens"))
        except Exception as e:  # a failed request is a missing cell, never a match
            res.append(None); ptoks.append(None)
            print(f"prompt {i}: {e}", file=sys.stderr)
    json.dump({"texts": res, "prompt_tokens": ptoks}, open(out, "w"))
    print(f"ident: {sum(t is not None for t in res)}/{len(res)} completions, prompt tokens {ptoks}")


def report(node):
    boots = {}
    for who in ("ctl", "arm"):
        for p in sorted(glob.glob(os.path.join(node, f"{who}-p*", "hook", "ident.json"))):
            boots.setdefault(who, []).append(json.load(open(p)))
    ctl, arm = boots.get("ctl", []), boots.get("arm", [])
    if len(ctl) < 2 or not arm:
        return [f"ident: {len(ctl)} control / {len(arm)} arm boots"], "EMPTY"
    stable = same = unstable = long_same = long_diff = 0
    lines = []
    for i in range(len(WINDOWS)):
        c = [b["texts"][i] for b in ctl]
        a = [b["texts"][i] for b in arm]
        if None in c or None in a:
            lines.append(f"  prompt {i}: missing completion"); return lines, "EMPTY"
        pt = [b.get("prompt_tokens", [None] * len(WINDOWS))[i] for b in ctl + arm]
        short = None not in pt and max(pt) < SHORT_MAX
        tag = f"{max(p for p in pt if p is not None) if any(p is not None for p in pt) else '?'} tok"
        if len(set(c)) > 1:
            unstable += 1
            lines.append(f"  prompt {i} ({tag}): control boots disagree ({len(set(c))} variants), arm matches one: "
                         f"{all(x in c for x in a)}")
            continue
        ok = all(x == c[0] for x in a)
        if not short:
            long_same += ok; long_diff += not ok
            lines.append(f"  prompt {i} ({tag}, BF16 path range: reported, not judged): control stable, arm "
                         f"{'identical' if ok else 'different'}")
            continue
        stable += 1
        same += ok
        lines.append(f"  prompt {i} ({tag}): control stable, arm {'identical' if ok else 'DIFFERENT'}")
    lines.insert(0, f"ident: {stable} short prompts (< {SHORT_MAX} tokens) stable across {len(ctl)} control boots, arm "
                    f"identical on {same}; long prompts not judged: {long_same} identical, {long_diff} different; "
                    f"{unstable} control-unstable (not judged)")
    return lines, ("UNSTABLE" if not stable else "PASS" if same == stable else "FAIL")


def selftest():
    import tempfile
    t = tempfile.mkdtemp()
    pt = [300, 800, 1600, 3100, 6200, 550, 2300, 4100]   # prompts 0, 1, 5 short

    def put(b, texts, p=pt):
        os.makedirs(os.path.join(t, b, "hook"), exist_ok=True)
        json.dump({"texts": texts, "prompt_tokens": p}, open(os.path.join(t, b, "hook", "ident.json"), "w"))
    base = [f"t{i}" for i in range(len(WINDOWS))]
    put("ctl-p1", base); put("ctl-p2", base[:7] + ["x"]); put("arm-p1", base); put("arm-p2", base)
    lines, v = report(t)
    assert v == "PASS" and "3 short prompts" in lines[0], lines
    # long-prompt divergence (prompt 4, 6200 tokens) is reported, not judged
    put("arm-p2", base[:4] + ["long differs"] + base[5:])
    lines, v = report(t)
    assert v == "PASS" and "1 different" in lines[0] and "reported, not judged): control stable, arm different" in lines[5], lines
    # short-prompt divergence (prompt 1, 800 tokens) fails
    put("arm-p2", base[:1] + ["short differs"] + base[2:])
    assert report(t)[1] == "FAIL"
    # a prompt over the threshold on any boot counts as long
    put("arm-p2", base[:1] + ["short differs"] + base[2:], pt[:1] + [1300] + pt[2:])
    assert report(t)[1] == "PASS"
    # missing prompt_tokens (old json) -> long, not judged; every prompt long -> UNSTABLE
    for b in ("ctl-p1", "ctl-p2", "arm-p1", "arm-p2"):
        d = json.load(open(os.path.join(t, b, "hook", "ident.json"))); d.pop("prompt_tokens")
        json.dump(d, open(os.path.join(t, b, "hook", "ident.json"), "w"))
    assert report(t)[1] == "UNSTABLE"
    put("arm-p2", base[:3] + [None] + base[4:])
    assert report(t)[1] == "EMPTY"
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["report"] and len(a) == 2:
        lines, v = report(a[1]); print("\n".join(lines)); print(f"IDENT={v}")
    elif a[:1] == ["run"]:
        kw = dict(zip(a[1::2], a[2::2]))
        run(kw["--base"], kw["--corpus"], kw["--out"])
    else:
        sys.exit(__doc__)
