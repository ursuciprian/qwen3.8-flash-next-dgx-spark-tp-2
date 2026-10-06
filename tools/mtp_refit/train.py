"""Train the dense part of the MTP layer against the served target (#97, design 4.6).

    python -m tools.mtp_refit.train \
      --snapshot <f4..03> --draft-vocab /opt/mtp-vocab/ids-v2-K131072.txt.gz \
      --data data/train --heldout data/heldout \
      --trainable dense --depth 6 --depth-weights equal --loss kl --topk 20 \
      --window 2048 --lr 2e-5 --warmup 20 --epochs 2 --tokens-per-step 8192 --out runs/<name>

Writes runs/<name>/mtp_refit.safetensors (the trainable tensors, BF16, checkpoint names), train.jsonl
(per optimizer step: soft CE per depth, anchors per depth, lr) and eval.json (offline acceptance
before and after, eval_offline.py). --epochs 0 exports the loaded tensors unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .assemble import iter_windows
from .eval_offline import evaluate
from .mtp_ref import export, full_to_draft, load_mtp_ref, trainable_names, window_loss


def depth_weights(spec: str, depth: int) -> list[float]:
    if spec == "equal":
        return [1.0] * depth
    if spec.startswith("decay"):
        r = float(spec.partition(":")[2] or 0.6)
        return [r**k for k in range(depth)]
    raise ValueError(f"--depth-weights {spec}: equal or decay[:r]")


def train(a) -> dict:
    torch.manual_seed(a.seed)
    model = load_mtp_ref(a.snapshot, a.draft_vocab, a.device, a.trainable, a.experts_impl)
    names = trainable_names(model, a.trainable)
    params = [p for n, p in model.named_parameters() if n in set(names)]
    print(f"trainable: {len(names)} tensors, {sum(p.numel() for p in params) / 1e6:.1f}M params")
    os.makedirs(a.out, exist_ok=True)
    meta = {k: v for k, v in vars(a).items() if k != "func"}
    res = {"args": meta}
    stride = a.window_stride or a.window // 2
    ev = dict(depth=a.depth, window=a.window, stride=stride, max_windows=a.eval_windows, device=a.device)
    if a.heldout and a.eval_windows:
        res["before"] = evaluate(model, a.heldout, **ev)
    if a.epochs:
        weights = depth_weights(a.depth_weights, a.depth)
        lookup = full_to_draft(model.draft_ids, model.cfg.vocab_size)
        opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
        log = open(os.path.join(a.out, "train.jsonl"), "a")
        step, seen, ce_acc, cnt_acc, t0 = 0, 0, [0.0] * a.depth, [0] * a.depth, time.time()
        model.train()
        for epoch in range(a.epochs):
            for w in iter_windows(a.data, a.window, stride, seed=a.seed + epoch, device=a.device):
                if not w["loss_mask"][1:].any():  # prompt-only window: no anchor counts
                    continue
                if w["topk_ids"].shape[1] != a.topk:
                    raise SystemExit(f"data has top-{w['topk_ids'].shape[1]}, --topk {a.topk}")
                with torch.autocast(torch.device(a.device).type, dtype=torch.bfloat16):
                    loss, counts, ce = window_loss(model, w, a.depth, weights, lookup)
                if not counts[0]:
                    continue
                (loss / a.tokens_per_step).backward()
                seen += counts[0]
                for k in range(a.depth):
                    ce_acc[k] += ce[k]
                    cnt_acc[k] += counts[k]
                if seen < a.tokens_per_step:
                    continue
                step += 1
                lr = a.lr * min(1.0, step / max(1, a.warmup))
                for g in opt.param_groups:
                    g["lr"] = lr
                gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip))
                opt.step()
                opt.zero_grad(set_to_none=True)
                rec = {"step": step, "epoch": epoch, "lr": lr, "grad_norm": round(gn, 4), "anchors": cnt_acc,
                       "ce": [round(c / n, 5) if n else None for c, n in zip(ce_acc, cnt_acc)],
                       "s": round(time.time() - t0, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                seen, ce_acc, cnt_acc = 0, [0.0] * a.depth, [0] * a.depth
                if a.max_steps and step >= a.max_steps:
                    break
            if a.max_steps and step >= a.max_steps:
                break
        opt.zero_grad(set_to_none=True)  # a partial accumulation at the end is dropped
        log.close()
        res["steps"] = step
    export(model, names, os.path.join(a.out, "mtp_refit.safetensors"), meta)
    if a.heldout and a.eval_windows:
        res["after"] = evaluate(model, a.heldout, **ev)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--draft-vocab", help="draft vocab id list (default: full vocab)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--heldout")
    ap.add_argument("--trainable", default="dense", choices=["dense", "dense-norouter"])
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--depth-weights", default="equal", help="equal | decay[:r] (r default 0.6)")
    ap.add_argument("--loss", default="kl", choices=["kl"], help="soft CE to the target top-k (= KL + const)")
    ap.add_argument("--topk", type=int, default=20, help="must match the capture's VLLM_MTP_CAPTURE_TOPK")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--window-stride", type=int,
                    help="default window // 2: overlapping windows, each anchor trained once with >= window/2 keys")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--tokens-per-step", type=int, default=8192, help="valid depth-0 anchors per optimizer step")
    ap.add_argument("--eval-windows", type=int, default=200, help="held-out windows per eval (0 = skip)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--experts-impl", help="transformers experts implementation (default eager loop)")
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = train(a)
    print(json.dumps({k: res[k] for k in ("steps",) if k in res}))


if __name__ == "__main__":
    main()
