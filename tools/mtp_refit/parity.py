"""Phase 2 checks of the training model against vLLM (#97, design 5.1 and 5.2).

    python -m tools.mtp_refit.parity drafts --snapshot S --draft-vocab V --data data/heldout data/train \
        --max-docs 500 --out parity-drafts.json
    python -m tools.mtp_refit.parity live --offline eval-t0.json --key t0 --metrics gen-live-t0 --out live-t0.json

drafts: for captured documents short enough to be stored whole and attended densely (<= --max-len
rows, starting at position 0), MtpRef runs the vLLM drafter's own greedy chain recorded by the
capture hook (input tokens: the sampled token, then the drafter's earlier drafts) from the last row
and its argmax must equal vLLM's draft at every depth. Pass: agreement >= --min-agree at each depth.
This compares argmaxes, not logits: vLLM does not expose draft logits outside its CUDA graphs.

live: offline per-position acceptance (eval_offline.py JSON, "all") against the live per-position
counters of the generation run that produced the same documents (gen.py run --metrics). Pass:
|offline - live| <= --tol at every served position.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch

from .assemble import iter_windows
from .mtp_ref import load_mtp_ref


@torch.no_grad()
def drafts(a):
    model = load_mtp_ref(a.snapshot, a.draft_vocab, a.device, experts_impl=a.experts_impl).eval()
    hit, n, docs = None, None, 0
    for d in a.data:
        for w in iter_windows(d, a.max_len, device=a.device):
            m = w["meta"]
            chain = m.get("drafts")
            L = w["tokens"].shape[0]
            if not chain or len(chain) < 2 or L != m["length"] or int(w["positions"][0]) != 0:
                continue
            D = len(chain) - 1
            hit = hit or [0] * D
            n = n or [0] * D
            ext = torch.tensor(chain[:D], dtype=w["tokens"].dtype, device=a.device)
            tokens = torch.cat([w["tokens"], ext])
            hidden = torch.cat([w["hidden"], w["hidden"].new_zeros(D, w["hidden"].shape[1])])
            pos = torch.arange(L + D, device=a.device)
            with torch.autocast(torch.device(a.device).type, dtype=torch.bfloat16):
                samples = model.unroll(tokens, hidden, pos, D)
            for k in range(min(D, len(hit))):
                got = int(model.draft_ids[model.logits(samples[k][L - 1 : L]).argmax(-1)])
                hit[k] += got == chain[k + 1]
                n[k] += 1
            docs += 1
            if docs >= a.max_docs:
                break
        if docs >= a.max_docs:
            break
    agree = [round(h / c, 4) for h, c in zip(hit or [], n or [])]
    ok = bool(agree) and min(agree) >= a.min_agree
    res = {"docs": docs, "per_depth_agree": agree, "min_agree": a.min_agree, "pass": ok}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))
    return 0 if ok else 1


def live_rates(metrics_dir):
    def read(p):
        out = {}
        for ln in open(p):
            m = re.match(r"^(vllm:spec_decode_\w+_total)(\{[^}]*\})?\s+([0-9.eE+-]+)$", ln.strip())
            if m:
                pos = re.search(r'position="(\d+)"', m.group(2) or "")
                key = (m.group(1), int(pos.group(1)) if pos else None)
                out[key] = out.get(key, 0.0) + float(m.group(3))
        return out

    b, e = (read(os.path.join(metrics_dir, f"metrics-{x}.txt")) for x in ("before", "after"))
    d = {k: v - b.get(k, 0.0) for k, v in e.items()}
    pos = sorted(k[1] for k in d if k[0] == "vllm:spec_decode_num_draft_tokens_per_pos_total")
    return [d.get(("vllm:spec_decode_num_accepted_tokens_per_pos_total", p), 0.0)
            / d[("vllm:spec_decode_num_draft_tokens_per_pos_total", p)] for p in pos
            if d[("vllm:spec_decode_num_draft_tokens_per_pos_total", p)]]


def live(a):
    off = json.load(open(a.offline))["all"][a.key]["per_position"]
    lv = live_rates(a.metrics)
    k = min(len(lv), len(off))
    diff = [round(off[i] - lv[i], 4) for i in range(k)]
    ok = k > 0 and max(abs(x) for x in diff) <= a.tol
    res = {"key": a.key, "offline": off[:k], "live": [round(x, 4) for x in lv[:k]], "diff": diff, "tol": a.tol,
           "pass": ok}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("drafts")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--draft-vocab")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--max-len", type=int, default=2048, help="QSA budget: dense attention is exact up to here")
    p.add_argument("--max-docs", type=int, default=500)
    p.add_argument("--min-agree", type=float, default=0.995)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--experts-impl")
    p.add_argument("--out", required=True)
    q = sub.add_parser("live")
    q.add_argument("--offline", required=True, help="eval_offline.py JSON of the same documents")
    q.add_argument("--key", choices=["t0", "t1"], required=True)
    q.add_argument("--metrics", required=True, help="gen.py run --metrics output dir")
    q.add_argument("--tol", type=float, default=0.01)
    q.add_argument("--out", required=True)
    a = ap.parse_args()
    raise SystemExit({"drafts": drafts, "live": live}[a.cmd](a))


if __name__ == "__main__":
    main()
