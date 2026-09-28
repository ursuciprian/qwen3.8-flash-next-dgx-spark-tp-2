import json, random, collections, sys
from datasets import load_dataset
SEED = 20260928
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
cats = collections.Counter(ds["category"])
tot = sum(cats.values())
for N in (1000, 2000):
    # largest-remainder proportional allocation, then a seeded sample per subject
    raw = {c: N * n / tot for c, n in cats.items()}
    k = {c: int(v) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: raw[c] - k[c], reverse=True)[: N - sum(k.values())]:
        k[c] += 1
    rng = random.Random(SEED)
    samples = {}
    for c in sorted(cats):
        name = "mmlupt_" + c.replace(" ", "_")
        samples[name] = sorted(rng.sample(range(cats[c]), k[c]))
    json.dump(samples, open(f"mmlu_pro_subset_{N}_seed{SEED}.json", "w"))
    print(N, sum(len(v) for v in samples.values()), {c: k[c] for c in sorted(k)})
