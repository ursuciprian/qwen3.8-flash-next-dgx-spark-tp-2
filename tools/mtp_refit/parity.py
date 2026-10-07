"""Phase 2 checks of the training model against vLLM (#97, design 5.1 and 5.2).

    python -m tools.mtp_refit.parity drafts --snapshot S --draft-vocab V --data data/heldout data/train \
        --max-docs 500 --out parity-drafts.json
    python -m tools.mtp_refit.parity live --offline eval-t0.json --key t0 --metrics gen-live-t0 --out live-t0.json
    python -m tools.mtp_refit.parity gate --drafts parity-drafts.json --live live-t0.json live-t1.json [--accepted A]

drafts: for captured documents short enough to be stored whole and attended densely (<= --max-len
rows, starting at position 0), MtpRef runs the vLLM drafter's own greedy chain recorded by the
capture hook (input tokens: the sampled token, then the drafter's earlier drafts) from the last row.
With the served per-step top-20 logits (hook draft_topk, vLLM cbee9971), the design 5.1 criterion,
fixed before the served logits were looked at (2026-10-06):
  - logits: mean over documents of ||ref - served|| / ||served|| over the served top-20 ids < 1e-2
    at every depth;
  - argmax: ref argmax == served top-1 on >= 0.995 of the decisive anchors at every depth, where an
    anchor is decisive when the served top-1 - top-2 margin exceeds delta, the numerical noise floor
    of the reference itself (`parity noise`: delta = 2 x the 99.5th percentile of |logit difference|
    of the same token between two MtpRef runs on different devices; a margin below 2x the per-logit
    noise can flip between two exact implementations);
  - decisive anchors >= 0.9 of all at every depth (else the argmax check vouches for too little).
Without draft_topk (older captures) the plain argmax agreement >= --min-agree is the criterion.
--dump writes the reference logit of vLLM's draft per document and depth (input to `parity noise`).

noise: delta from two --dump files of the same documents (e.g. GPU and CPU).

live: offline per-position acceptance against the live per-position counters of the generation run
that produced the same documents (gen.py run --metrics). The offline side is eval_offline's "replay"
(the serving step process: counters count steps, and a step starts right after a rejection), not its
anchor-averaged per_position, which over-states position 1 by ~0.035 at T=0 on the p2 live set. T=1
uses the top-k lower bound and reports the upper bound next to it. Pass: |offline - live| <= --tol at
every served position.

gate: the phase 2 verdict from the drafts and live JSONs. --accepted coverage-<YYYY-MM-DD> records a user
decision to train despite the decisive fraction (< 0.9 at delta 3.5 on the served margins, 2026-10-07); it
waives that criterion only, logit error, decisive argmax and both live checks still have to pass.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch

from .assemble import iter_windows
from .mtp_ref import full_to_draft, load_mtp_ref, serve_hc_mxfp8

LOGIT_REL_ERR_MAX, ARGMAX_MIN, DECISIVE_MIN = 1e-2, 0.995, 0.9


@torch.no_grad()
def drafts(a):
    model = load_mtp_ref(a.snapshot, a.draft_vocab, a.device, experts_impl=a.experts_impl).eval()
    model.kv_fp8 = getattr(a, "kv_fp8", "on") == "on"
    if getattr(a, "hc_mxfp8", "off") == "on":
        serve_hc_mxfp8(model)
    delta = json.load(open(a.noise))["delta"] if getattr(a, "noise", None) else None
    lookup = full_to_draft(model.draft_ids, model.cfg.vocab_size)
    rel, dec, dec_hit, dump, with_topk = None, None, None, {}, 0
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
            rel, dec, dec_hit = rel or [[] for _ in range(D)], dec or [0] * D, dec_hit or [0] * D
            topk = m.get("draft_topk")
            with_topk += topk is not None
            ext = torch.tensor(chain[:D], dtype=w["tokens"].dtype, device=a.device)
            tokens = torch.cat([w["tokens"], ext])
            hidden = torch.cat([w["hidden"], w["hidden"].new_zeros(D, w["hidden"].shape[1])])
            pos = torch.arange(L + D, device=a.device)
            with torch.autocast(torch.device(a.device).type, dtype=torch.bfloat16):
                samples = model.unroll(tokens, hidden, pos, D)
            dump[m["id"]] = []
            for k in range(min(D, len(hit))):
                lg = model.logits(samples[k][L - 1 : L])[0]
                got = int(model.draft_ids[lg.argmax(-1)])
                hit[k] += got == chain[k + 1]
                n[k] += 1
                j = int(lookup[chain[k + 1]])
                dump[m["id"]].append(float(lg[j]) if j >= 0 else None)
                if topk is not None and delta is not None:
                    ids = torch.tensor(topk[0][k], device=lg.device)
                    vals = torch.tensor(topk[1][k], device=lg.device)
                    keep = ids < lookup.numel()
                    keep[keep.clone()] = lookup[ids[keep]] >= 0
                    ref, srv = lg[lookup[ids[keep]]], vals[keep]
                    rel[k].append(float((ref - srv).norm() / srv.norm()))
                    if float(vals[0] - vals[1]) > delta:
                        dec[k] += 1
                        dec_hit[k] += got == int(ids[0])
            docs += 1
            if docs >= a.max_docs:
                break
        if docs >= a.max_docs:
            break
    agree = [round(h / c, 4) for h, c in zip(hit or [], n or [])]
    ok = bool(agree) and min(agree) >= a.min_agree
    res = {"docs": docs, "per_depth_agree": agree, "min_agree": a.min_agree, "pass": ok}
    if delta is not None and docs and with_topk == docs:
        err = [round(sum(r) / len(r), 5) for r in rel]
        frac = [round(d / c, 4) for d, c in zip(dec, n)]
        dagree = [round(h / d, 4) if d else None for h, d in zip(dec_hit, dec)]
        ok = (all(e < LOGIT_REL_ERR_MAX for e in err) and all(f >= DECISIVE_MIN for f in frac)
              and all(x is not None and x >= ARGMAX_MIN for x in dagree))
        res = {"docs": docs, "logit_rel_err": err, "logit_rel_err_max": LOGIT_REL_ERR_MAX, "delta": delta,
               "decisive_frac": frac, "decisive_agree": dagree, "argmax_min": ARGMAX_MIN,
               "decisive_min": DECISIVE_MIN, "per_depth_agree_all": agree, "pass": ok}
    elif delta is not None:
        res["note"] = f"draft_topk on {with_topk} of {docs} documents: logit criterion not applicable"
        ok = res["pass"] = False
    if getattr(a, "dump", None):
        json.dump(dump, open(a.dump, "w"))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))
    return 0 if ok else 1


def noise(a):
    """delta = 2 x q99.5 of |logit_a - logit_b| of the same token over documents and depths."""
    x, y = json.load(open(a.a)), json.load(open(a.b))
    d = sorted(abs(p - q) for k in x.keys() & y.keys() for p, q in zip(x[k], y[k]) if p is not None and q is not None)
    if not d:
        raise SystemExit("no common documents")
    q = d[min(len(d) - 1, int(0.995 * len(d)))]
    res = {"delta": round(2 * q, 4), "q995_abs_logit_diff": q, "samples": len(d), "median": d[len(d) // 2],
           "max": d[-1], "a": a.a, "b": a.b,
           "derivation": "a top-1/top-2 margin can flip between two exact implementations only if it is below the "
                         "sum of their per-logit errors; 2 x the 99.5th percentile of the per-logit difference"}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))
    return 0


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
    rep = json.load(open(a.offline))["all"]["replay"]
    off = rep[a.key]["per_position"]
    lv = live_rates(a.metrics)
    k = min(len(lv), len(off))
    diff = [round(off[i] - lv[i], 4) for i in range(k)]
    ok = k > 0 and max(abs(x) for x in diff) <= a.tol
    res = {"key": a.key, "offline": off[:k], "live": [round(x, 4) for x in lv[:k]], "diff": diff, "tol": a.tol,
           "pass": ok}
    if a.key == "t1" and "t1_hi" in rep:
        res["offline_hi"] = rep["t1_hi"]["per_position"][:k]
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))
    return 0 if ok else 1


ACCEPT_RE = re.compile(r"^coverage-\d{4}-\d{2}-\d{2}$")


def gate_verdict(drafts_res, live_res, accepted=""):
    """(pass, reasons) of the phase 2 gate from `drafts` and `live` JSONs. accepted="coverage-<date>" (a user
    decision, REFIT_USER_ACCEPTED) waives only the decisive-fraction criterion; every other criterion stays."""
    if accepted and not ACCEPT_RE.match(accepted):
        raise ValueError(f"unknown acceptance {accepted!r}: only coverage-<YYYY-MM-DD> waives a criterion")
    why, d = [], drafts_res
    if "logit_rel_err" not in d:
        why.append("drafts: no logit criterion (draft_topk missing)")
    else:
        if not all(e < d["logit_rel_err_max"] for e in d["logit_rel_err"]):
            why.append(f"drafts: logit rel err {d['logit_rel_err']} >= {d['logit_rel_err_max']}")
        if not all(x is not None and x >= d["argmax_min"] for x in d["decisive_agree"]):
            why.append(f"drafts: decisive argmax {d['decisive_agree']} < {d['argmax_min']}")
        if not all(f >= d["decisive_min"] for f in d["decisive_frac"]) and not accepted:
            why.append(f"drafts: decisive fraction {d['decisive_frac']} < {d['decisive_min']}")
    for r in live_res:
        if not r["pass"]:
            why.append(f"live {r['key']}: diff {r['diff']} beyond {r['tol']}")
    return not why, why


def gate(a):
    ok, why = gate_verdict(json.load(open(a.drafts)), [json.load(open(f)) for f in a.live], a.accepted or "")
    note = f" (user accepted {a.accepted}: decisive fraction waived)" if a.accepted else ""
    print(("PASS" + note) if ok else "FAIL: " + "; ".join(why))
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
    p.add_argument("--kv-fp8", choices=["on", "off"], default="on", help="as eval_offline / train")
    p.add_argument("--hc-mxfp8", choices=["on", "off"], default="off", help="as eval_offline")
    p.add_argument("--noise", help="parity noise JSON (delta): enables the logit criterion")
    p.add_argument("--dump", help="write {doc id: [reference logit of vLLM's draft per depth]}")
    p.add_argument("--out", required=True)
    q = sub.add_parser("live")
    q.add_argument("--offline", required=True, help="eval_offline.py JSON of the same documents")
    q.add_argument("--key", choices=["t0", "t1"], required=True)
    q.add_argument("--metrics", required=True, help="gen.py run --metrics output dir")
    q.add_argument("--tol", type=float, default=0.01)
    q.add_argument("--out", required=True)
    z = sub.add_parser("noise")
    z.add_argument("--a", required=True)
    z.add_argument("--b", required=True)
    z.add_argument("--out", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--drafts", required=True, help="parity drafts JSON")
    g.add_argument("--live", nargs="+", required=True, help="parity live JSONs (t0, t1)")
    g.add_argument("--accepted", help="REFIT_USER_ACCEPTED: coverage-<YYYY-MM-DD> waives the decisive fraction only")
    a = ap.parse_args()
    raise SystemExit({"drafts": drafts, "live": live, "noise": noise, "gate": gate}[a.cmd](a))


if __name__ == "__main__":
    main()
