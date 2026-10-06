"""Replay generated token lists through the capture server (vLLM with VLLM_MTP_CAPTURE_DIR set).

    python -m tools.mtp_refit.capture_client --server http://localhost:8000 --gen gen --out capture

Each record of gen/<category>.jsonl is sent as /v1/completions with prompt = prompt + output token
ids, max_tokens 1, temperature 0: one prefill whose rows the hook stores. capture/manifest.jsonl
gets {id, category, split, sha1, response_start, length}; sha1 is the digest the hook writes per
request, so assemble.py can join them. Stdlib only. Reruns skip records already in the manifest.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
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
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man_path = os.path.join(a.out, "manifest.jsonl")
    done = {json.loads(l)["sha1"] for l in open(man_path)} if os.path.exists(man_path) else set()
    recs = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.gen, "*.jsonl"))) for l in open(f)]
    lock, man, n = threading.Lock(), open(man_path, "a"), [0, 0]

    def one(r):
        toks = r["prompt_token_ids"] + r["output_token_ids"]
        sha = token_sha1(toks)
        if sha in done or len(toks) > a.max_len or not r["output_token_ids"]:
            return
        try:
            _post(f"{a.server}/v1/completions",
                  {"model": a.model, "prompt": toks, "max_tokens": 1, "temperature": 0}, a.timeout)
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
    print(f"captured {n[0]}, failed {n[1]}, already done {len(done)}")


if __name__ == "__main__":
    main()
