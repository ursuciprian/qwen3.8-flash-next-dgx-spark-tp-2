#!/usr/bin/env python3
"""k85 paired comparison of two lm-eval runs (2026-10-10, tp-2 #155). Stdlib only.

  evalcmp.py <dirA> <labelA> <dirB> <labelB> [out.json]
  evalcmp.py --selftest

Reads samples_<task>_*.jsonl under each dir (lm-eval --log_samples). Per task and metric: accuracy of A and B on the
questions both answered, the discordant pairs (A right/B wrong, A wrong/B right) and the exact two-sided McNemar p.
gsm8k_think: exact_match per filter (strict-match, flexible-extract); ifeval_think: prompt_level_strict_acc and
prompt_level_loose_acc per prompt, inst_level_strict_acc pooled over instructions (unpaired).
"""
import glob, json, math, os, sys

METRICS = {"gsm8k_think": ["exact_match"], "ifeval_think": ["prompt_level_strict_acc", "prompt_level_loose_acc"]}


def load(d, task):
    out, inst = {}, []
    for p in glob.glob(f"{d}/**/samples_{task}_*.jsonl", recursive=True):
        for l in open(p):
            r = json.loads(l)
            filt = r.get("filter", "none")
            for m in METRICS[task]:
                if m in r:
                    out[(f"{m}/{filt}" if task == "gsm8k_think" else m, str(r["doc_id"]))] = float(r[m])
            if "inst_level_strict_acc" in r:
                inst += [float(x) for x in r["inst_level_strict_acc"]]
    return out, inst


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def cmp(da, la, db, lb):
    res = {}
    for task in METRICS:
        A, ia = load(da, task); B, ib = load(db, task)
        for m in sorted({k[0] for k in A} | {k[0] for k in B}):
            docs = sorted({k[1] for k in A if k[0] == m} & {k[1] for k in B if k[0] == m})
            if not docs:
                continue
            a = [A[(m, d)] >= 0.5 for d in docs]; b = [B[(m, d)] >= 0.5 for d in docs]
            ab = sum(x and not y for x, y in zip(a, b)); ba = sum(y and not x for x, y in zip(a, b))
            res[f"{task}:{m}"] = {"n": len(docs), la: sum(a) / len(a), lb: sum(b) / len(b),
                                  f"{la}_right_{lb}_wrong": ab, f"{la}_wrong_{lb}_right": ba, "mcnemar_p": mcnemar(ab, ba)}
        if ia and ib:
            res[f"{task}:inst_level_strict_acc"] = {"n": f"{len(ia)}/{len(ib)} instructions", la: sum(ia) / len(ia),
                                                    lb: sum(ib) / len(ib), "mcnemar_p": None}
    return res


def text(res, la, lb):
    L = [f"{'eval:metric':<46}{'n':>6}{la:>14}{lb:>14}   discordant ({la} only / {lb} only)   McNemar p"]
    for k, r in res.items():
        d = "" if r["mcnemar_p"] is None else f"{r[f'{la}_right_{lb}_wrong']:>6} / {r[f'{la}_wrong_{lb}_right']:<6}"
        p = "-" if r["mcnemar_p"] is None else f"{r['mcnemar_p']:.3f}"
        L.append(f"{k:<46}{str(r['n']):>6}{100 * r[la]:>13.2f}%{100 * r[lb]:>13.2f}%   {d:<33}{p}")
    return "\n".join(L)


def selftest():
    import tempfile
    assert mcnemar(0, 0) == 1.0 and abs(mcnemar(0, 6) - 2 / 64) < 1e-12 and mcnemar(3, 3) == 1.0
    d = tempfile.mkdtemp()
    for name, right in (("a", [1, 1, 1, 0]), ("b", [1, 0, 0, 0])):
        os.makedirs(f"{d}/{name}/m")
        with open(f"{d}/{name}/m/samples_gsm8k_think_x.jsonl", "w") as f:
            for i, v in enumerate(right):
                f.write(json.dumps({"doc_id": i, "filter": "strict-match", "exact_match": v}) + "\n")
        with open(f"{d}/{name}/m/samples_ifeval_think_x.jsonl", "w") as f:
            f.write(json.dumps({"doc_id": 0, "prompt_level_strict_acc": right[1] == 1, "prompt_level_loose_acc": True,
                                "inst_level_strict_acc": [True, right[1] == 1]}) + "\n")
    r = cmp(f"{d}/a", "A", f"{d}/b", "B")
    g = r["gsm8k_think:exact_match/strict-match"]
    assert g["n"] == 4 and g["A"] == 0.75 and g["B"] == 0.25 and g["A_right_B_wrong"] == 2 and g["A_wrong_B_right"] == 0
    assert r["ifeval_think:inst_level_strict_acc"]["B"] == 0.5
    print(text(r, "A", "B")); print("evalcmp selftest OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        selftest()
    elif len(a) >= 4:
        r = cmp(a[0], a[1], a[2], a[3])
        if len(a) > 4:
            json.dump(r, open(a[4], "w"), indent=1)
        print(text(r, a[1], a[3])); sys.exit(0 if r else 1)
    else:
        print(__doc__); sys.exit(2)
