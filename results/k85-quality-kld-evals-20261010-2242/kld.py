#!/usr/bin/env python3
"""k85 teacher-forced KLD and top-1 agreement between two servers (2026-10-10, tp-2 #155). Stdlib only.

  kld.py corpus <out.jsonl> <gen.jsonl>...        fixed corpus: id, category, ids (prompt + output, at most 8192), start
  kld.py score <base-url> <corpus> <out.jsonl.gz> [K]
        one /v1/completions request per sequence (max_tokens 1, prompt_logprobs K, default 64); keeps, for every output
        position (start..end), the logprob of the actual token and the top-K (token, logprob) list
  kld.py compare <ref.jsonl.gz> <test.jsonl.gz> [out.json]
  kld.py --selftest

KLD per position = KL(P_ref || Q_test) over the reference's top-K tokens plus one "rest" bucket (a coarse-grained KL,
which is a lower bound of the full-vocabulary KL when q is exact). A reference token missing from the test top-K gets
the test's K-th logprob (an upper bound of its true value). Top-1 agreement = same argmax. PPL from the actual tokens'
logprobs (exact, the actual token is always returned).
"""
import gzip, json, math, statistics, sys, time, urllib.request

MAXLEN = 8192


def corpus(out, files):
    n = 0
    with open(out, "w") as f:
        for p in files:
            for l in open(p):
                r = json.loads(l)
                ids = (r["prompt_token_ids"] + r["output_token_ids"])[:MAXLEN]
                start = len(r["prompt_token_ids"])
                if start >= len(ids) - 16:
                    continue
                f.write(json.dumps({"id": r["id"], "category": r["category"], "start": start, "ids": ids}) + "\n"); n += 1
    print(f"{n} sequences -> {out}")


def post(base, body, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(base.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.load(r)
        except Exception as ex:   # transient: retry, then raise
            if i == tries - 1:
                raise
            print(f"retry after {ex!r}"[:300], flush=True); time.sleep(10)


def score(base, cpath, out, k=64):
    seqs = [json.loads(l) for l in open(cpath)]
    t0 = time.time()
    with gzip.open(out, "wt") as f:
        for s in seqs:
            d = post(base, {"model": "qwen3.8-flash-next", "prompt": s["ids"], "max_tokens": 1, "temperature": 0,
                            "prompt_logprobs": k})
            pl = d["choices"][0]["prompt_logprobs"]
            assert len(pl) == len(s["ids"]), (len(pl), len(s["ids"]))
            act, top = [], []
            for i in range(s["start"], len(s["ids"])):
                e = {int(t): v for t, v in pl[i].items()}
                act.append(round(e[s["ids"][i]]["logprob"], 5))
                top.append([[t, round(v["logprob"], 5)] for t, v in sorted(e.items(), key=lambda x: x[1]["rank"]) if v["rank"] <= k])
            f.write(json.dumps({"id": s["id"], "category": s["category"], "start": s["start"], "act": act, "top": top}) + "\n")
            print(f"{s['id']}: {len(act)} positions, {time.time() - t0:.0f} s", flush=True)


def pos_kl(p, q):
    """p, q: [[tok, logprob], ...] top lists -> (kl, same_top1)"""
    P = {t: lp for t, lp in p}
    Q = {t: lp for t, lp in q}
    qmin = min(Q.values())
    kl, pc, qc = 0.0, 0.0, 0.0
    for t, lp in P.items():
        lq = Q.get(t, qmin); pt = math.exp(lp)
        kl += pt * (lp - lq); pc += pt; qc += math.exp(lq)
    pr, qr = max(1.0 - pc, 1e-12), max(1.0 - qc, 1e-12)
    kl += pr * math.log(pr / qr)
    return max(kl, 0.0), max(P, key=P.get) == max(Q, key=Q.get), pc


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def compare(ref, test, out=None):
    R = {r["id"]: r for r in map(json.loads, gzip.open(ref, "rt"))}
    T = {r["id"]: r for r in map(json.loads, gzip.open(test, "rt"))}
    ids = [i for i in R if i in T]
    kls, agree, cov, cat, nll_r, nll_t = [], [], [], {}, [], []
    for i in ids:
        r, t = R[i], T[i]
        assert len(r["top"]) == len(t["top"]), i
        for a, b, x, y in zip(r["top"], t["top"], r["act"], t["act"]):
            kl, same, pc = pos_kl(a, b)
            kls.append(kl); agree.append(same); cov.append(pc); nll_r.append(-x); nll_t.append(-y)
            c = cat.setdefault(r["category"], [[], []]); c[0].append(kl); c[1].append(same)
    res = {"ref": ref, "test": test, "sequences": len(ids), "positions": len(kls),
           "missing_sequences": sorted(set(R) ^ set(T)),
           "kld_mean": statistics.mean(kls), "kld_median": statistics.median(kls), "kld_p90": pct(kls, 0.90),
           "kld_p99": pct(kls, 0.99), "kld_max": max(kls), "top1_agreement": sum(agree) / len(agree),
           "ref_ppl": math.exp(statistics.mean(nll_r)), "test_ppl": math.exp(statistics.mean(nll_t)),
           "ref_topk_mass_mean": statistics.mean(cov),
           "by_category": {c: {"positions": len(v[0]), "kld_mean": statistics.mean(v[0]),
                               "top1_agreement": sum(v[1]) / len(v[1])} for c, v in sorted(cat.items())}}
    if out:
        json.dump(res, open(out, "w"), indent=1)
    return res


def line(name, r):
    return (f"{name:<44} KLD mean {r['kld_mean']:.5f} median {r['kld_median']:.5f} p99 {r['kld_p99']:.4f} | top-1 "
            f"{100 * r['top1_agreement']:.2f}% | PPL ref {r['ref_ppl']:.4f} test {r['test_ppl']:.4f} | "
            f"{r['positions']} positions, {r['sequences']} seqs")


def selftest():
    import os, tempfile
    lp = lambda *p: [math.log(x) for x in p]
    a = [[1, v] for v in lp(0.7)] + [[2, math.log(0.2)], [3, math.log(0.1)]]
    kl, same, pc = pos_kl(a, a)
    assert abs(kl) < 1e-12 and same and abs(pc - 1) < 1e-9
    b = [[1, math.log(0.5)], [2, math.log(0.3)], [3, math.log(0.2)]]
    exact = 0.7 * math.log(0.7 / 0.5) + 0.2 * math.log(0.2 / 0.3) + 0.1 * math.log(0.1 / 0.2)
    kl, same, _ = pos_kl(a, b)
    assert abs(kl - exact) < 1e-9 and same, (kl, exact)
    kl, same, _ = pos_kl(a, [[2, math.log(0.6)], [1, math.log(0.3)], [4, math.log(0.1)]])
    assert not same and kl > 0
    d = tempfile.mkdtemp()
    for n, top in (("r", a), ("t", b)):
        with gzip.open(f"{d}/{n}.gz", "wt") as f:
            f.write(json.dumps({"id": "x", "category": "c", "start": 0, "act": [math.log(0.7)], "top": [top]}) + "\n")
    r = compare(f"{d}/r.gz", f"{d}/t.gz")
    assert abs(r["kld_mean"] - exact) < 1e-9 and r["top1_agreement"] == 1.0 and r["positions"] == 1
    src = f"{d}/gen.jsonl"
    with open(src, "w") as f:
        f.write(json.dumps({"id": "g", "category": "chat", "prompt_token_ids": [1] * 10, "output_token_ids": [2] * 9000}) + "\n")
    corpus(f"{d}/c.jsonl", [src])
    c = json.loads(open(f"{d}/c.jsonl").readline())
    assert len(c["ids"]) == MAXLEN and c["start"] == 10
    print("kld selftest OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        selftest()
    elif a[:1] == ["corpus"]:
        corpus(a[1], a[2:])
    elif a[:1] == ["score"]:
        score(a[1], a[2], a[3], int(a[4]) if len(a) > 4 else 64)
    elif a[:1] == ["compare"]:
        r = compare(a[1], a[2], a[3] if len(a) > 3 else None); print(line("compare", r))
    else:
        print(__doc__); sys.exit(2)
