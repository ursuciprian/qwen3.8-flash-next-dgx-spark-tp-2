#!/usr/bin/env python3
"""Teacher-forced next-token top-1 of the target LM head on a held-out text set (r10).

  head_top1.py capture --base http://host:8000 --out top1.json
  head_top1.py diff a.json b.json

capture: fixed held-out documents (results/corpus-prose.txt and the tail of
results/corpus-code.txt, past every depth_decode_probe offset) are tokenized by the
server; for each document, every STRIDE-th prefix is sent as token ids with max_tokens=1,
temperature 0 and top-5 logprobs. The first output token comes from the prefill logits,
i.e. the main lm_head only (no MTP draft involved), and every build sees the same
contexts, unlike free-running greedy text that diverges after the first different token.
diff: top-1 agreement, top-5 set overlap, mean |dlogprob| of A's top-1 token where B lists
it, and each side's accuracy against the true next token.
"""
import argparse, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# 12 + 12 documents x range(32, 768, 8) = 24 x 92 = 2,208 positions. Code docs start past every
# depth_decode_probe offset (< 1.6M chars); prose (1.28M chars) is not read by the screening probes.
DOCS = [("prose", REPO / "results/corpus-prose.txt", 100_000 + i * 95_000) for i in range(12)] + \
       [("code", REPO / "results/corpus-code.txt", 2_200_000 + i * 145_000) for i in range(12)]
CHARS, TOKENS, START, STRIDE = 6000, 768, 32, 8
MIN_POSITIONS = 2000


def post(base, path, body, timeout=300, tries=3):
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for i in range(tries):  # one transient HTTP error must not lose a whole capture
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except (OSError, ValueError):
            if i == tries - 1:
                raise
            time.sleep(5)


def capture(a):
    rows = []
    for kind, path, off in DOCS:
        text = path.read_text(errors="replace")[off:off + CHARS]
        ids = post(a.base, "/tokenize", {"model": a.model, "prompt": text, "add_special_tokens": False})["tokens"][:TOKENS]
        rows += [(kind, off, p, ids[:p], ids[p]) for p in range(START, len(ids), STRIDE)]
    if len(rows) < MIN_POSITIONS:
        sys.exit(f"only {len(rows)} positions (< {MIN_POSITIONS}): a document tokenized short")

    def one(r):
        kind, off, p, ctx, true = r
        lp = post(a.base, "/v1/completions", {"model": a.model, "prompt": ctx, "max_tokens": 1, "temperature": 0.0,
                                              "logprobs": 5, "return_tokens_as_token_ids": True})["choices"][0]["logprobs"]
        top = {int(k.split(":")[1]): v for k, v in (lp["top_logprobs"][0] or {}).items()}
        return {"kind": kind, "doc": off, "pos": p, "true": true, "top5": top}

    with ThreadPoolExecutor(a.concurrency) as ex:
        recs = list(ex.map(one, rows))
    json.dump({"base": a.base, "records": recs}, open(a.out, "w"))
    print(f"wrote {a.out}: {len(recs)} positions from {len(DOCS)} documents")


def compare(x, y):
    # JSON object keys come back as strings; token ids are ints everywhere below.
    for rec in x["records"] + y["records"]:
        rec["top5"] = {int(k): v for k, v in rec["top5"].items()}
    out = {}
    for kind in ("all", "prose", "code"):
        pairs = [(p, q) for p, q in zip(x["records"], y["records"]) if kind in ("all", p["kind"])]
        if not pairs:
            continue
        assert all((p["doc"], p["pos"]) == (q["doc"], q["pos"]) for p, q in pairs), "captures differ in layout"
        top1 = lambda r: max(r["top5"], key=r["top5"].get)
        agree = sum(top1(p) == top1(q) for p, q in pairs)
        overlap = sum(len(set(p["top5"]) & set(q["top5"])) for p, q in pairs) / (5 * len(pairs))
        dl = [abs(p["top5"][top1(p)] - q["top5"][top1(p)]) for p, q in pairs if top1(p) in q["top5"]]
        out[kind] = {"n": len(pairs), "top1_agree": agree / len(pairs), "top5_overlap": overlap,
                     "mean_abs_dlogprob_top1": sum(dl) / len(dl) if dl else None,
                     "acc_true_a": sum(top1(p) == p["true"] for p, _ in pairs) / len(pairs),
                     "acc_true_b": sum(top1(q) == q["true"] for _, q in pairs) / len(pairs)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--base", default="http://127.0.0.1:8000")
    c.add_argument("--model", default="qwen3.8-flash-next")
    c.add_argument("--concurrency", type=int, default=4)
    c.add_argument("--out", required=True)
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "capture":
        capture(a)
    else:
        print(json.dumps(compare(json.load(open(a.a)), json.load(open(a.b))), indent=1))


if __name__ == "__main__":
    sys.exit(main())
