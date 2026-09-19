#!/usr/bin/env python3
"""Build the reduced draft vocabulary for vllm-0001-mtp-draft-vocab.patch
(VLLM_QWEN_MTP_DRAFT_VOCAB_PATH).

Our own script, not vendored from MiaAI-Lab's AGPL-3.0 patch_mtp_draft_vocab.py
sibling (files/build_draft_vocab.py in MiaAI-Lab/Qwen3.8-Flash-Next-*-DGX-Spark)
-- we reproduce their documented method instead of shipping their code:

    "Counts token frequencies over a corpus and writes the most frequent ids
    ... The corpus that matters is the *model's own output distribution*,
    because that is what the drafter has to predict -- not a general text
    corpus and not the prompts."

So the corpus for OUR checkpoint should be OUR own model's generation
transcripts, not a copy of anyone else's ranked-id list -- swapping model
family changes byte-level BPE merge order, so blindly reusing a third
party's `draft_vocab.txt` silently mis-targets the reduced head.

Recommended corpus for this deployment (not run in this prep session --
build worker/dgx-01 job): concatenate assistant-turn text from
  dgx-01:~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/{spectrace,
  localimg,quality-arms,thirdparty-A0}/**/*.{jsonl,log}
into one corpus.jsonl (one {"text": ...} per generated turn) before running
this script. Prompts/inputs do not belong in the corpus (see docstring
above) -- only what the target model itself produced.

Usage:
  python3 scripts/build_draft_vocab.py corpus.jsonl \
      --model local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \
      --out draft_vocab.pt --size 65149

Output: an int64 torch tensor of kept vocab ids, sorted ascending, saved with
torch.save -- this is exactly what vllm-0001-mtp-draft-vocab.patch loads via
VLLM_QWEN_MTP_DRAFT_VOCAB_PATH (torch.load(..., weights_only=True)).
"""
import argparse
import json
import sys
from collections import Counter


def iter_texts(path: str):
    if path.endswith(".jsonl"):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get("text", "")
                if text:
                    yield text
    else:
        with open(path, errors="replace") as handle:
            while True:
                block = handle.read(1 << 20)
                if not block:
                    break
                yield block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+", help="corpus .jsonl (text field) or .txt files")
    ap.add_argument("--model", default="local-inference-lab/Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--out", default="draft_vocab.pt")
    ap.add_argument("--size", type=int, default=65149,
                     help="MiaAI-Lab measured +8-21% at this size on this model family; "
                          "re-tune from the printed coverage-vs-size table, don't copy blind")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = len(tok)

    counts: Counter[int] = Counter()
    docs = 0
    for path in args.corpus:
        for text in iter_texts(path):
            counts.update(tok(text, add_special_tokens=False)["input_ids"])
            docs += 1
            if docs % 200 == 0:
                print(f"  ... {docs} chunks, {sum(counts.values()):,} tokens", file=sys.stderr)
    total = sum(counts.values())
    if not total:
        sys.exit("ERROR: corpus produced no tokens -- check --model matches the corpus, "
                  "and that the corpus holds MODEL OUTPUT, not prompts")

    special = {i for i in (tok.all_special_ids or []) if 0 <= i < vocab_size}
    added = getattr(tok, "added_tokens_encoder", {}) or {}
    special |= {int(i) for i in added.values() if 0 <= int(i) < vocab_size}

    ranked = [tid for tid, _ in counts.most_common()]
    keep = sorted(special)
    seen = set(keep)
    for tid in ranked:
        if len(keep) >= args.size:
            break
        if tid not in seen:
            keep.append(tid)
            seen.add(tid)
    keep = sorted(seen)

    covered = sum(counts[t] for t in seen if t in counts)
    print(f"corpus:     {docs} chunks, {total:,} token occurrences, {len(counts):,} distinct ids")
    print(f"vocabulary: {vocab_size:,} -> {len(keep):,} ({100.0 * len(keep) / vocab_size:.1f}%), "
          f"{len(special)} special/added kept unconditionally")
    print(f"coverage:   {100.0 * covered / total:.4f}% of corpus occurrences at size {len(keep):,}")
    for cut in (8192, 16384, 32768, 65149, 131072):
        sub = set(special) | set(ranked[:max(0, cut - len(special))])
        cov = sum(counts[t] for t in sub if t in counts)
        print(f"  size {cut:>7,}: coverage {100.0 * cov / total:7.4f}%")

    if args.report_only:
        return
    torch.save(torch.tensor(keep, dtype=torch.int64), args.out)
    print(f"wrote {len(keep):,} ids -> {args.out}")


if __name__ == "__main__":
    main()
