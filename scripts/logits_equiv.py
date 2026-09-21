#!/usr/bin/env python3
"""Top-5 logprob equivalence check between an unfused and a fused server.

Only one vLLM server can own the DGX pair at a time, so this runs in two
phases:

    # boot the baseline (no mod), then
    scripts/logits_equiv.py capture --out /tmp/logits_base.json

    # boot with mods: [vllm-qwen38-bf16-gemv], then
    scripts/logits_equiv.py capture --out /tmp/logits_fused.json
    scripts/logits_equiv.py diff /tmp/logits_base.json /tmp/logits_fused.json

`diff` exits non-zero when any sampled token differs or any top-5 logprob
moves by more than --tol (default 1e-3).

Uses /v1/completions, not /v1/chat/completions: no chat template, no
reasoning parser, nothing between the prompt and the logits.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):\n    if len(arr) <= 1:",
    "In 1969, humans first",
    "The derivative of x^3 with respect to x is",
    "Q: What is 17 * 23?\nA:",
    "SELECT name, COUNT(*) FROM orders GROUP BY",
    "The three laws of thermodynamics state that",
    "import torch\nimport torch.nn as nn\n\nclass Attention(nn.Module):",
    "Once upon a time in a small village near the",
    "The main difference between TCP and UDP is",
    "curl -sS https://api.example.com/v1/users -H 'Authorization:",
    "Translate to French: The weather is nice today.\n",
    "A linked list reversal in C looks like:\n",
    "The Pythagorean theorem states",
    "terraform {\n  required_providers {",
    "Explain the CAP theorem in one sentence:",
    "The mitochondria is the",
    "apiVersion: apps/v1\nkind: Deployment\nmetadata:",
    "Given f(x) = 2x + 5, solve f(x) = 17.",
    "The largest planet in the solar system is",
]

MAX_TOKENS = 16
TOP_LOGPROBS = 5


def _post(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def capture(args: argparse.Namespace) -> int:
    url = args.base_url.rstrip("/") + "/v1/completions"
    records = []
    for index, prompt in enumerate(PROMPTS):
        body = _post(
            url,
            {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 1234,
                "logprobs": TOP_LOGPROBS,
                "stream": False,
            },
            args.timeout,
        )
        lp = body["choices"][0]["logprobs"]
        records.append(
            {
                "index": index,
                "prompt": prompt,
                "tokens": lp["tokens"],
                "token_logprobs": lp["token_logprobs"],
                "top_logprobs": lp["top_logprobs"],
            }
        )
        print(f"  [{index + 1:2d}/{len(PROMPTS)}] {len(lp['tokens'])} tokens", flush=True)
    with open(args.out, "w") as handle:
        json.dump({"model": args.model, "records": records}, handle)
    print(f"wrote {args.out}")
    return 0


def compare(base: dict, fused: dict, tol: float) -> list[str]:
    """Every mismatch between two capture payloads, worst first."""
    problems: list[str] = []
    by_index = {r["index"]: r for r in fused["records"]}
    for ref in base["records"]:
        got = by_index.get(ref["index"])
        if got is None:
            problems.append(f"prompt {ref['index']}: missing from the fused capture")
            continue
        if ref["tokens"] != got["tokens"]:
            for pos, (a, b) in enumerate(zip(ref["tokens"], got["tokens"])):
                if a != b:
                    problems.append(
                        f"prompt {ref['index']} pos {pos}: token {a!r} -> {b!r}"
                    )
                    break
            else:
                problems.append(
                    f"prompt {ref['index']}: length {len(ref['tokens'])} -> "
                    f"{len(got['tokens'])}"
                )
            continue
        for pos, (a_top, b_top) in enumerate(
            zip(ref["top_logprobs"], got["top_logprobs"])
        ):
            a_top = a_top or {}
            b_top = b_top or {}
            if set(a_top) != set(b_top):
                problems.append(
                    f"prompt {ref['index']} pos {pos}: top-5 set changed "
                    f"{sorted(a_top)} -> {sorted(b_top)}"
                )
                continue
            for token, value in a_top.items():
                delta = abs(value - b_top[token])
                if delta > tol:
                    problems.append(
                        f"prompt {ref['index']} pos {pos} token {token!r}: "
                        f"|dlogprob| = {delta:.3e} > {tol:.1e}"
                    )
    return problems


def diff(args: argparse.Namespace) -> int:
    with open(args.base) as handle:
        base = json.load(handle)
    with open(args.fused) as handle:
        fused = json.load(handle)
    problems = compare(base, fused, args.tol)
    if not problems:
        print(
            f"PASS: {len(base['records'])} prompts, identical tokens, "
            f"all top-{TOP_LOGPROBS} logprobs within {args.tol:.1e}"
        )
        return 0
    print(f"FAIL: {len(problems)} mismatches (tol {args.tol:.1e})")
    for line in problems[:40]:
        print("  " + line)
    if len(problems) > 40:
        print(f"  ... and {len(problems) - 40} more")
    return 1


def selftest(_: argparse.Namespace) -> int:
    def rec(tokens, tops):
        return {"records": [{"index": 0, "prompt": "", "tokens": tokens,
                             "token_logprobs": [0.0] * len(tokens),
                             "top_logprobs": tops}]}

    a = rec([" Paris"], [{" Paris": -0.10, " Lyon": -3.0}])
    assert compare(a, a, 1e-3) == []
    within = rec([" Paris"], [{" Paris": -0.1005, " Lyon": -3.0}])
    assert compare(a, within, 1e-3) == [], compare(a, within, 1e-3)
    beyond = rec([" Paris"], [{" Paris": -0.11, " Lyon": -3.0}])
    assert len(compare(a, beyond, 1e-3)) == 1
    other_token = rec([" Lyon"], [{" Paris": -0.10, " Lyon": -3.0}])
    assert len(compare(a, other_token, 1e-3)) == 1
    other_set = rec([" Paris"], [{" Paris": -0.10, " Nice": -3.0}])
    assert len(compare(a, other_set, 1e-3)) == 1
    assert compare(a, {"records": []}, 1e-3) == [
        "prompt 0: missing from the fused capture"
    ]
    print("selftest ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="record top-5 logprobs from one server")
    cap.add_argument("--base-url", default="http://127.0.0.1:8000")
    cap.add_argument("--model", default="qwen3.8-flash-next")
    cap.add_argument("--timeout", type=float, default=180.0)
    cap.add_argument("--out", required=True)
    cap.set_defaults(func=capture)

    cmp_ = sub.add_parser("diff", help="compare two captures")
    cmp_.add_argument("base")
    cmp_.add_argument("fused")
    cmp_.add_argument("--tol", type=float, default=1e-3)
    cmp_.set_defaults(func=diff)

    st = sub.add_parser("selftest", help="check the comparator, no server needed")
    st.set_defaults(func=selftest)

    args = parser.parse_args()
    try:
        return args.func(args)
    except urllib.error.URLError as exc:
        print(f"server unreachable: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
