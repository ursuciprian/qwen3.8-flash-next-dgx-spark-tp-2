#!/usr/bin/env python3
"""Draft-vocab id lists for VLLM_MTP_DRAFT_VOCAB, ranked on our own traffic (r7 dvocab v2).

r6 used a 65,536-id list built on another project's English/code corpus. Its MTP
acceptance at 16k context fell 3-5 points (0.73/0.52/0.37/0.29 -> 0.70/0.48/0.33/0.24):
identifiers and rare tokens copied from long contexts fall outside the list, the draft
proposes q=0 for them and the target rejects. This script ranks ids on our corpora,
benchmark prompts and model outputs, and always keeps:
  - every added/special token (chat template, tool-call and reasoning markers),
  - every byte-level alphabet token (the 256 single-byte symbols),
  - every token that decodes to one character (incl. partial UTF-8 byte fragments),
so a copied rare string can always be drafted byte- or char-wise.

Score per id = sum over groups of weight * (count in group / tokens in group), so a large
group cannot drown a small one. Ids never seen fill the tail in the order of --prior
(the r6 list, from a larger corpus), then ascending id (BPE merge order ~ frequency).

Output format = the r6 list: gzip text, one id per line, ascending, no header.
The vLLM compile key hashes the path, not the file: name each list by version and K.

  build_draft_vocab_v2.py --tokenizer tokenizer.json --k 65536,98304,131072 \\
    --src outputs:0.45:'responses/**/*.json' --src code:0.35:corpus-code.txt ... \\
    --eval devops-heldout:'responses/*/1[234]-*.json' --prior ids-K65536.txt.gz \\
    --out-dir out --tag v2
Selftest: build_draft_vocab_v2.py --selftest --tokenizer tokenizer.json
"""
import argparse
import fnmatch
import glob
import gzip
import json
import os
from collections import Counter

from tokenizers import Tokenizer


def texts(path):
    """Every string in a .json/.jsonl file (recursive), or the whole text file."""
    def walk(o):
        if isinstance(o, str):
            yield o
        elif isinstance(o, dict):
            for v in o.values():
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)
    raw = open(path, errors="replace").read()
    if path.endswith(".jsonl"):
        for line in raw.splitlines():
            if line.strip():
                yield from walk(json.loads(line))
    elif path.endswith(".json"):
        yield from walk(json.loads(raw))
    else:
        yield raw


def split_spec(spec):
    """'a.txt[0:0.8]' -> (a.txt, 0, 0.8): a char-fraction slice of a text file."""
    if spec.endswith("]") and "[" in spec:
        p, rng = spec[:-1].split("[", 1)
        lo, hi = (float(x) for x in rng.split(":"))
        return p, lo, hi
    return spec, 0.0, 1.0


def count(tok, specs, excludes=()):
    c = Counter()
    for spec in specs:
        pat, lo, hi = split_spec(spec)
        for path in sorted(glob.glob(pat, recursive=True)):
            if os.path.isdir(path) or any(fnmatch.fnmatch(path, e) for e in excludes):
                continue
            for t in texts(path):
                t = t[int(lo * len(t)):int(hi * len(t))]
                # 64 KiB chunks on line breaks: tokenizers is slow on multi-MB strings
                chunks, i = [], 0
                while i < len(t):
                    j = t.find("\n", i + 65536)
                    j = len(t) if j < 0 else j + 1
                    chunks.append(t[i:j])
                    i = j
                for e in tok.encode_batch(chunks, add_special_tokens=False):
                    c.update(e.ids)
    return c


def forced_ids(tok, tj):
    vocab = tok.get_vocab()
    ids = {a["id"] for a in tj["added_tokens"]}
    ids |= {i for s, i in vocab.items() if len(s) == 1}  # byte-level alphabet
    ids |= {i for i in vocab.values() if len(tok.decode([i], skip_special_tokens=False)) == 1}
    return ids


def rank(groups, weights, forced, prior, vocab_ids):
    score = Counter()
    for g, c in groups.items():
        tot = sum(c.values()) or 1
        for i, n in c.items():
            score[i] += weights[g] * n / tot
    order = sorted(forced)
    seen = set(order)
    for i in [i for i, _ in score.most_common()] + list(prior) + sorted(vocab_ids):
        if i not in seen:
            seen.add(i)
            order.append(i)
    return order


def coverage(ids, c):
    tot = sum(c.values()) or 1
    return sum(n for i, n in c.items() if i in ids) / tot


def read_ids(path):
    with gzip.open(path, "rt") as fh:
        return [int(l.split("#", 1)[0]) for l in fh if l.split("#", 1)[0].strip()]


def selftest(tok, tj):
    f = forced_ids(tok, tj)
    assert all(a["id"] in f for a in tj["added_tokens"])
    groups = {"a": Counter({5000: 10, 6000: 1}), "b": Counter({7000: 1})}
    order = rank(groups, {"a": 1.0, "b": 1.0}, {1, 2}, [9000, 5000], range(10))
    assert order[:2] == [1, 2] and order[2] == 7000 and order[3] == 5000, order[:6]
    assert order[5] == 9000 and order[6] == 0 and len(order) == len(set(order))
    enc = lambda t: Counter(tok.encode(t, add_special_tokens=False).ids)
    for ch in ("\u00e9", "\u4e2d", "\U0001f600", "\u2e3b"):  # single chars and byte fragments
        assert coverage(f, enc(ch)) == 1.0, ch
    assert coverage(f, enc(" fetch_invoice_identifier")) < 1.0
    print("selftest ok, forced ids:", len(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--src", action="append", default=[],
                    help="group:weight:glob[,glob...]; a text glob may end in [lo:hi] (char fractions)")
    ap.add_argument("--eval", action="append", default=[], help="name:glob[,glob...]")
    ap.add_argument("--exclude", action="append", default=[], help="fnmatch pattern dropped from --src")
    ap.add_argument("--k", default="65536,98304,131072")
    ap.add_argument("--prior", help="existing id list (.txt.gz) used to order unseen ids")
    ap.add_argument("--baseline", action="append", default=[], help="name:ids.txt.gz, coverage only")
    ap.add_argument("--out-dir")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    tj = json.load(open(a.tokenizer))
    if a.selftest:
        return selftest(tok, tj)
    vocab_ids = set(tok.get_vocab().values())
    forced = forced_ids(tok, tj)
    groups, weights = {}, {}
    for s in a.src:
        g, w, globs = s.split(":", 2)
        groups[g] = groups.get(g, Counter()) + count(tok, globs.split(","), a.exclude)
        weights[g] = float(w)
        print(f"src {g}: w={w} {sum(groups[g].values()):,} tokens, {len(groups[g]):,} distinct")
    prior = read_ids(a.prior) if a.prior else []
    order = rank(groups, weights, forced, prior, vocab_ids)
    seen_all = set().union(*groups.values()) if groups else set()
    print(f"forced {len(forced):,}; distinct seen {len(seen_all):,}; vocab {len(vocab_ids):,}")
    lists = {f"{a.tag}-K{k}": set(order[:k]) for k in map(int, a.k.split(","))}
    for b in a.baseline:
        n, p = b.split(":", 1)
        lists[n] = set(read_ids(p))
    evals = {n: count(tok, g.split(",")) for n, g in (e.split(":", 1) for e in a.eval)}
    if evals:
        print("coverage (fraction of eval token occurrences inside the list)")
        print(f"{'list':<22}" + "".join(f"{n:>22}" for n in evals))
        for ln, ids in lists.items():
            print(f"{ln:<22}" + "".join(f"{coverage(ids, c):>22.4f}" for c in evals.values()))
        for n, c in evals.items():
            print(f"eval {n}: {sum(c.values()):,} tokens, {len(c):,} distinct")
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        for k in map(int, a.k.split(",")):
            p = os.path.join(a.out_dir, f"ids-{a.tag}-K{k}.txt.gz")
            with gzip.GzipFile(p, "wb", mtime=0) as fh:  # mtime 0: reproducible sha
                fh.write("".join(f"{i}\n" for i in sorted(order[:k])).encode())
            print("wrote", p)


if __name__ == "__main__":
    main()
