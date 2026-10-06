"""Join vLLM capture shards with the capture manifest into training parts, and read them back.

    python -m tools.mtp_refit.assemble --capture capture/shards --manifest capture/manifest.jsonl --out data

Capture shards (vLLM VLLM_MTP_CAPTURE_DIR, mtp-capture-v1) hold rows of many requests,
split across steps and shards; request metadata carries the sha1 of the full token list.
The manifest (capture_client.py) maps that sha1 to the document id, category, split and
response start. A document is complete when its rows cover [length - stored, length)
without gaps. Output: data/{train,heldout}/part-%05d.safetensors with

    hidden [N, S, H] bf16, tokens [N] int32, positions [N] int32, topk_ids [N, K] int32,
    topk_logprobs [N, K] fp16, loss_mask [N] uint8 (1 = generated token), doc_offsets [D+1] int64

and metadata "docs" = JSON list of {id, category, sha1, length, drafts}; drafts is the vLLM drafter's
greedy chain [sampled token, d0, d1, ...] from the last row, when the hook recorded it (parity.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROW_KEYS = ("hidden", "tokens", "positions", "topk_ids", "topk_logprobs")
PART_ROWS = 65536


class PartWriter:
    def __init__(self, out_dir: str, hc: int):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.hc, self.n = out_dir, hc, len(glob.glob(os.path.join(out_dir, "part-*.safetensors")))
        self.docs, self.meta, self.rows = [], [], 0

    def add(self, doc: dict[str, torch.Tensor], meta: dict) -> None:
        self.docs.append(doc)
        self.meta.append(meta)
        self.rows += doc["tokens"].shape[0]
        if self.rows >= PART_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self.docs:
            return
        t = {k: torch.cat([d[k] for d in self.docs]) for k in (*ROW_KEYS, "loss_mask")}
        n, w = t["hidden"].shape
        t["hidden"] = t["hidden"].view(n, self.hc, w // self.hc)
        lens = torch.tensor([0] + [d["tokens"].shape[0] for d in self.docs])
        t["doc_offsets"] = lens.cumsum(0)
        path = os.path.join(self.out_dir, f"part-{self.n:05d}.safetensors")
        save_file({k: v.contiguous() for k, v in t.items()}, path + ".tmp",
                  metadata={"format": "mtp-refit-train-v1", "docs": json.dumps(self.meta)})
        os.replace(path + ".tmp", path)
        self.n, self.docs, self.meta, self.rows = self.n + 1, [], [], 0


def assemble(capture_dirs: list[str], manifest: str, out: str, hc: int = 4) -> dict:
    man = {}
    for line in open(manifest):
        m = json.loads(line)
        man[m["sha1"]] = m
    shards = sorted(p for d in capture_dirs for p in glob.glob(os.path.join(d, "shard-*.safetensors")))
    # pass 1: metadata only -> request id -> (sha1, prefill_len)
    reqs: dict[tuple[str, str], dict] = {}
    for p in shards:
        with safe_open(p, "pt") as f:
            for r in json.loads(f.metadata()["requests"]):
                key = (os.path.dirname(p), r["id"])  # request ids are unique per server run (= dir)
                info = reqs.setdefault(key, {"sha1": None, "prefill_len": r["prefill_len"], "drafts": None})
                info["sha1"] = r["sha1"] or info["sha1"]
                info["drafts"] = r.get("drafts") or info["drafts"]
    writers = {s: PartWriter(os.path.join(out, s), hc) for s in ("train", "heldout")}
    pending: dict[tuple[str, str], list[dict]] = {}
    stats = {"docs": 0, "rows": 0, "no_manifest": 0, "incomplete": 0, "length_mismatch": 0}
    for p in shards:
        with safe_open(p, "pt") as f:
            t = {k: f.get_tensor(k) for k in (*ROW_KEYS, "req")}
            req_list = json.loads(f.metadata()["requests"])
        for k, r in enumerate(req_list):
            rows = (t["req"] == k).nonzero().flatten()
            pending.setdefault((os.path.dirname(p), r["id"]), []).append({x: t[x][rows] for x in ROW_KEYS})
        for key in [k for k in pending if _complete(pending[k], reqs[k]["prefill_len"])]:
            pieces, info = pending.pop(key), reqs[key]
            m = man.get(info["sha1"])
            if m is None:
                stats["no_manifest"] += 1
                continue
            if m["length"] != info["prefill_len"]:
                stats["length_mismatch"] += 1
                continue
            doc = _stitch(pieces)
            doc["loss_mask"] = (doc["positions"] >= m["response_start"]).to(torch.uint8)
            writers[m.get("split", "train")].add(doc, {"id": m["id"], "category": m["category"], "sha1": info["sha1"],
                                                       "length": m["length"], "drafts": info["drafts"]})
            stats["docs"] += 1
            stats["rows"] += doc["tokens"].shape[0]
    stats["incomplete"] = len(pending)
    for w in writers.values():
        w.flush()
    return stats


def _stitch(pieces):
    doc = {k: torch.cat([p[k] for p in pieces]) for k in ROW_KEYS}
    order = doc["positions"].argsort()
    return {k: v[order] for k, v in doc.items()}


def _complete(pieces, length) -> bool:
    pos = torch.cat([p["positions"] for p in pieces]).sort().values
    return int(pos[-1]) == length - 1 and bool((pos.diff() == 1).all())


# ---------------------------------------------------------------- reader
def iter_windows(data_dir: str, window: int, stride: int | None = None, seed: int | None = None,
                 device="cpu"):
    """Yield training windows: dicts of the row tensors over <= window rows of one document,
    plus "category" and "id". Overlapping windows (stride < window) zero loss_mask on the
    rows already covered by the previous window of the document."""
    stride = stride or window
    parts = sorted(glob.glob(os.path.join(data_dir, "part-*.safetensors")))
    rng = random.Random(seed)
    if seed is not None:
        rng.shuffle(parts)
    for p in parts:
        with safe_open(p, "pt") as f:
            t = {k: f.get_tensor(k) for k in f.keys()}
            docs = json.loads(f.metadata()["docs"])
        off = t.pop("doc_offsets").tolist()
        spans = []
        for d, meta in enumerate(docs):
            a, b = off[d], off[d + 1]
            starts = list(range(a, max(a + 1, b - window + stride), stride))
            for i, s in enumerate(starts):
                spans.append((s, min(s + window, b), 0 if i == 0 else window - stride, meta))
        if seed is not None:
            rng.shuffle(spans)
        for s, e, cut, meta in spans:
            w = {k: v[s:e].to(device) for k, v in t.items()}
            w["hidden"] = w["hidden"].flatten(1)
            w["positions"] = w["positions"].long()
            if cut:
                w["loss_mask"] = w["loss_mask"].clone()
                w["loss_mask"][:cut] = 0
            w["category"], w["id"], w["meta"] = meta["category"], meta["id"], meta
            yield w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", nargs="+", required=True, help="capture shard dir(s), one per server run")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hc", type=int, default=4)
    a = ap.parse_args()
    print(json.dumps(assemble(a.capture, a.manifest, a.out, a.hc)))


if __name__ == "__main__":
    main()
