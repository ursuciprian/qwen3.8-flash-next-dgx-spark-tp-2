"""Offline acceptance of the MTP drafter on assembled data, per draft position and category.

    python -m tools.mtp_refit.eval_offline --snapshot <f4..03> --draft-vocab ids-v2-K131072.txt.gz \
        --data data/heldout [--refit runs/x/mtp_refit.safetensors] --depth 6 --out eval.json

Per depth k, over anchors valid at k (mtp_ref.depth_targets), teacher-forced on the data's tokens:
- t1: mean of sum_x min(p, q) over the target's stored top-k (expected acceptance at T=1; the
  tail of p outside the top-k is dropped, so a lower bound)
- t0: P(draft argmax == target argmax | depths < k matched and the data token was the target
  argmax), i.e. exact greedy chains that coincide with the data
Both report the per-draft (conditional) rate a_k, vLLM's per-position rate prod_{j<=k} a_j
(spec_decode_num_accepted_tokens_per_pos / num_draft_tokens_per_pos) and tokens per step
1 + sum_k prod_{j<=k} a_j at depth 4 and at the full depth.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import torch

from . import prom
from .assemble import iter_windows
from .mtp_ref import acceptance, depth_targets, full_to_draft, load_mtp_ref


def load_refit(model, path: str) -> int:
    from safetensors.torch import load_file

    params = dict(model.named_parameters())
    t = load_file(path)
    with torch.no_grad():
        for name, v in t.items():
            params[name.removeprefix("mtp.")].copy_(v)
    return len(t)


@torch.no_grad()
def evaluate(model, data_dir, depth=6, window=2048, stride=None, max_windows=None, device="cpu", autocast=True):
    model.eval()
    lookup = full_to_draft(model.draft_ids, model.cfg.vocab_size)
    acc = defaultdict(lambda: [[0.0, 0, 0, 0] for _ in range(depth)])  # cat -> per k [t1 sum, n, t0 hit, t0 n]
    for i, w in enumerate(iter_windows(data_dir, window, stride, seed=0, device=device)):  # fixed shuffle
        if max_windows and i >= max_windows:
            break
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=autocast):
            samples = model.unroll(w["tokens"], w["hidden"], w["positions"], depth)
        n = w["tokens"].shape[0]
        ar = torch.arange(n, device=w["tokens"].device)
        # greedy serving drafts from anchor t only if x[t+1] was the target's argmax at row t
        chain = w["tokens"][(ar + 1).clamp(max=n - 1)].long() == w["topk_ids"][:, 0].long()
        for k, (rows, valid) in enumerate(depth_targets(n, depth, w["loss_mask"])):
            ids, lp = w["topk_ids"][rows], w["topk_logprobs"][rows]
            t1, t0 = acceptance(model.logits(samples[k]), ids, lp, lookup, model.draft_ids)
            lab = w["tokens"][(ar + 2 + k).clamp(max=n - 1)]
            for cat in (w["category"], "all"):
                a = acc[cat][k]
                a[0] += float(t1[valid].sum())
                a[1] += int(valid.sum())
                a[2] += int((t0 & valid & chain).sum())
                a[3] += int((valid & chain).sum())
            chain &= t0 & (lab.long() == ids[:, 0].long())
    return {cat: _summary(v) for cat, v in acc.items()}


def _summary(per_k):
    out = {"anchors": [a[1] for a in per_k]}
    for key, num, den in (("t1", 0, 1), ("t0", 2, 3)):
        cond = [round(a[num] / a[den], 4) if a[den] else None for a in per_k]
        cum, c = [], 1.0
        for x in cond:
            c = c * x if x is not None and c is not None else None
            cum.append(round(c, 4) if c is not None else None)
        out[key] = {"per_draft": cond, "per_position": cum,
                    "tokens_per_step_d4": round(1 + sum(x for x in cum[:4] if x), 3),
                    "tokens_per_step": round(1 + sum(x for x in cum if x), 3)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--draft-vocab")
    ap.add_argument("--data", required=True)
    ap.add_argument("--refit", help="mtp_refit.safetensors to overlay (default: the shipped drafter)")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--window-stride", type=int, help="default window // 2: every anchor sees >= window/2 keys")
    ap.add_argument("--max-windows", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--experts-impl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics-file", help="Prometheus textfile for the result (off by default); run label = --out stem")
    a = ap.parse_args()
    model = load_mtp_ref(a.snapshot, a.draft_vocab, a.device, experts_impl=a.experts_impl)
    if a.refit:
        print(f"overlay: {load_refit(model, a.refit)} tensors from {a.refit}")
    res = evaluate(model, a.data, a.depth, a.window, a.window_stride or a.window // 2, a.max_windows, a.device)
    json.dump(res, open(a.out, "w"), indent=1)
    if a.metrics_file:
        run = os.path.splitext(os.path.basename(a.out))[0]
        prom.write(a.metrics_file, prom.acceptance(res, {"run": run, "phase": "eval",
                                                         "drafter": "refit" if a.refit else "shipped"}))
    print(json.dumps(res.get("all"), indent=1))


if __name__ == "__main__":
    main()
