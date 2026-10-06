"""Offline acceptance of the MTP drafter on assembled data, per draft position and category.

    python -m tools.mtp_refit.eval_offline --snapshot <f4..03> --draft-vocab ids-v2-K131072.txt.gz \
        --data data/heldout [--refit runs/x/mtp_refit.safetensors] --depth 6 --out eval.json

Per depth k, over anchors valid at k (mtp_ref.depth_targets), teacher-forced on the data's tokens:
- t1: mean of sum_x min(p, q) over the target's stored top-k (expected acceptance at T=1; the
  tail of p outside the top-k is dropped, so a lower bound)
- t0: P(draft argmax == target argmax | depths < k matched and the data token was the target
  argmax), i.e. exact greedy chains that coincide with the data
Both report the per-draft (conditional) rate a_k, the per-position rate prod_{j<=k} a_j and tokens per
step 1 + sum_k prod_{j<=k} a_j at depth 4 and at the full depth. These average over every anchor.

"replay" (whole documents only: one window from position 0) is what vLLM's per-position counters
(spec_decode_num_accepted_tokens_per_pos / num_draft_tokens_per_pos) measure: steps, not anchors. A step
drafts from anchor t and the next one drafts from t + 1 + accepted, so anchors right after a rejection are
over-represented. Per anchor and depth: t0 = draft argmax == the data token (the emitted token, exact for
data generated at T=0), t1 = sum_x min(p, q) over the stored top-k as above (t1_hi adds min(p, q) mass the
top-k cannot see, an upper bound); replay() takes the expectation over the step process. The live check
(parity.py live) compares this, not per_position, with the live counters.
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
    rep = defaultdict(lambda: {m: [0.0, [0.0] * depth] for m in ("t0", "t1", "t1_hi")})  # cat -> mode -> steps, hits
    for i, w in enumerate(iter_windows(data_dir, window, stride, seed=0, device=device)):  # fixed shuffle
        if max_windows and i >= max_windows:
            break
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=autocast):
            samples = model.unroll(w["tokens"], w["hidden"], w["positions"], depth)
        n = w["tokens"].shape[0]
        ar = torch.arange(n, device=w["tokens"].device)
        # greedy serving drafts from anchor t only if x[t+1] was the target's argmax at row t
        chain = w["tokens"][(ar + 1).clamp(max=n - 1)].long() == w["topk_ids"][:, 0].long()
        whole = int(w["positions"][0]) == 0 and n == w["meta"].get("length") and bool(w["loss_mask"].any())
        per = {m: [] for m in ("t0", "t1", "t1_hi")}
        for k, (rows, valid) in enumerate(depth_targets(n, depth, w["loss_mask"])):
            ids, lp = w["topk_ids"][rows], w["topk_logprobs"][rows]
            logits = model.logits(samples[k])
            t1, t0 = acceptance(logits, ids, lp, lookup, model.draft_ids)
            lab = w["tokens"][(ar + 2 + k).clamp(max=n - 1)]
            if whole:
                inb = (ar + 2 + k) < n
                per["t0"].append(((model.draft_ids[logits.argmax(-1)] == lab.long()) & inb).double())
                per["t1"].append(t1.double() * inb)
                per["t1_hi"].append(acceptance_hi(logits, ids, lp, lookup, t1).double() * inb)
            for cat in (w["category"], "all"):
                a = acc[cat][k]
                a[0] += float(t1[valid].sum())
                a[1] += int(valid.sum())
                a[2] += int((t0 & valid & chain).sum())
                a[3] += int((valid & chain).sum())
            chain &= t0 & (lab.long() == ids[:, 0].long())
        if whole:
            start = int(w["loss_mask"].nonzero()[0]) - 1  # drafts x[r+1..] from the first generated token x[r]
            for m, cols in per.items():
                steps, hits = replay(torch.stack(cols, 1)[: max(n - 2, 0)].cpu(), max(start, 0))
                for cat in (w["category"], "all"):
                    r = rep[cat][m]
                    r[0] += steps
                    r[1] = [x + y for x, y in zip(r[1], hits)]
    out = {cat: _summary(v) for cat, v in acc.items()}
    for cat, r in rep.items():
        out[cat]["replay"] = {m: {"steps": round(s, 1), "per_position": [round(h / s, 4) for h in hits],
                                  "tokens_per_step": round(1 + sum(hits) / s, 3)}
                              for m, (s, hits) in r.items() if s}
    return out


def acceptance_hi(logits, topk_ids, topk_lp, lookup, t1):
    """t1 plus the most sum_x min(p, q) can add outside the stored top-k: min(p tail, q mass off the top-k)."""
    j = lookup[topk_ids.long()]
    q_on = (logits.softmax(-1).gather(1, j.clamp(min=0)) * (j >= 0)).sum(-1)
    p_tail = 1 - topk_lp.float().exp().sum(-1)
    return t1 + torch.minimum(p_tail.clamp(min=0), (1 - q_on).clamp(min=0))


def replay(acc: torch.Tensor, start: int) -> tuple[float, list[float]]:
    """vLLM's step process over one document. acc [anchors, D]: probability that a step drafting from anchor t
    accepts depth k given depths < k accepted (0/1 at T=0: an exact replay). The first step drafts from
    `start`; a step accepting j drafts is followed by one drafting from t + 1 + j; anchors past the last row
    end the document. Returns expected (steps, steps accepting >= k + 1 drafts for each k)."""
    n, D = acc.shape
    cum = acc.double().cumprod(1).tolist()
    pi = [0.0] * (n + D + 1)
    if start < n:
        pi[start] = 1.0
    steps, hits = 0.0, [0.0] * D
    for t in range(start, n):
        w = pi[t]
        if not w:
            continue
        steps += w
        prev = 1.0
        for j in range(D + 1):
            c = cum[t][j] if j < D else 0.0
            if j < D:
                hits[j] += w * c
            pi[t + j + 1] += w * (prev - c)
            prev = c
    return steps, hits


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
    ap.add_argument("--kv-fp8", choices=["on", "off"], default="on", help="fp8 e4m3 round trip of K/V (served cache)")
    ap.add_argument("--qsa-select", action="store_true", help="QSA block top-k over depth-0 keys (long documents)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics-file", help="Prometheus textfile for the result (off by default); run label = --out stem")
    a = ap.parse_args()
    model = load_mtp_ref(a.snapshot, a.draft_vocab, a.device, experts_impl=a.experts_impl)
    model.kv_fp8, model.qsa_select = a.kv_fp8 == "on", a.qsa_select
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
