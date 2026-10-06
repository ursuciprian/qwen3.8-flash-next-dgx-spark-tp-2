"""Replay generated token lists through the capture server (vLLM with VLLM_MTP_CAPTURE_DIR set).

    python -m tools.mtp_refit.capture_client --server http://localhost:8000 --gen gen --out capture

Each record of gen/<category>.jsonl is sent as /v1/completions with prompt = prompt + output token
ids, temperature 0: one prefill whose rows the hook stores (the few decode steps after it are not
stored; max_tokens > 1 only makes the prefill step draft, for the parity check). capture/manifest.jsonl
gets {id, category, split, sha1, response_start, length}; sha1 is the digest the hook writes per
request, so assemble.py can join them. Stdlib only. Reruns skip records already in the manifest.

--cut-max-len N (parity re-capture, #97): send each record cut at a seeded random point inside its
response, keeping only cuts of <= N tokens (--limit records at most). The drafter's recorded chain then
starts from a generated token mid-response, a state serving drafts from (the full documents end after
the end of turn, where depth-4 drafts are near-ties). Manifest ids get "#cut<c>", split "heldout".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor

from .gen import _post, token_sha1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=262143, help="skip sequences longer than this")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--tail", type=int, default=6144, help="the capture server's VLLM_MTP_CAPTURE_TAIL")
    ap.add_argument("--min-context", type=int, default=2048, help="prompt rows kept before the response")
    ap.add_argument("--max-tokens", type=int, default=8,
                    help=">1 so the prefill step schedules drafts; the hook records that greedy chain")
    ap.add_argument("--cut-max-len", type=int, help="parity re-capture: cut inside the response, <= this many tokens")
    ap.add_argument("--cut-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="at most this many records (after cutting)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man_path = os.path.join(a.out, "manifest.jsonl")
    done = {json.loads(l)["sha1"] for l in open(man_path)} if os.path.exists(man_path) else set()
    recs = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.gen, "*.jsonl"))) for l in open(f)]
    if a.cut_max_len:
        recs = cut_records(recs, a.cut_max_len, a.cut_seed)
    if a.limit:
        recs = recs[: a.limit]
    lock, man, n = threading.Lock(), open(man_path, "a"), [0, 0, 0]

    def one(r):
        toks = r["prompt_token_ids"] + r["output_token_ids"]
        sha = token_sha1(toks)
        if sha in done or len(toks) > a.max_len or not r["output_token_ids"]:
            return
        if len(r["output_token_ids"]) > a.tail - a.min_context:  # the hook's tail would cut its context
            with lock:
                n[2] += 1
            return
        try:
            _post(f"{a.server}/v1/completions",
                  {"model": a.model, "prompt": toks, "max_tokens": a.max_tokens, "temperature": 0}, a.timeout)
        except Exception as e:  # noqa: BLE001
            print(f"{r['id']}: {e}")
            with lock:
                n[1] += 1
            return
        with lock:
            man.write(json.dumps({"id": r["id"], "category": r["category"], "split": r["split"], "sha1": sha,
                                  "response_start": len(r["prompt_token_ids"]), "length": len(toks)}) + "\n")
            man.flush()
            n[0] += 1

    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, recs))
    man.close()
    print(f"captured {n[0]}, failed {n[1]}, skipped as too long for the tail {n[2]}, already done {len(done)}")


def cut_records(recs, max_len, seed):
    """Records cut at a seeded point c in [1, len(output) - 1] (response keeps >= 1 generated token before the
    anchor and drops >= 1), prompt + c <= max_len; shuffled with the same seed so --limit takes a mix."""
    rng, out = random.Random(seed), []
    for r in recs:
        n, p = len(r["output_token_ids"]), len(r["prompt_token_ids"])
        hi = min(n - 1, max_len - p)
        if hi < 1:
            continue
        c = rng.randint(1, hi)
        out.append({**r, "id": f"{r['id']}#cut{c}", "split": "heldout", "output_token_ids": r["output_token_ids"][:c]})
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    main()
