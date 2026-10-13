#!/usr/bin/env python3
"""k76 coding cells: k55's 36-prompt corpus at concurrency N (stdlib wrapper around ~/GEN-AI/k55/coding_probe.py).

  coding.py <base-url> <t0-nothink|default> <conc> <runs> <out.json>
  coding.py --selftest            imports the corpus, checks 12 Python + 8 C++ + 8 Rust + 8 Go and max_tokens 768

The prompts, request body, max_tokens (768), both settings and the per-request decode tok/s
((completion_tokens - 1) / (t_last - t_first)) are k55's own run(); this file only sends them N at a time (a pool of N
workers over the 36 prompts) for <runs> passes after k55's warm-up request. Per pass: per-request decode tok/s, how many
hit max_tokens, and the aggregate output tok/s (sum of completion tokens / wall time of the pass). k55 reads /metrics
around every request for tokens/step; here that is off (the deltas overlap at N > 1 and the DP=2 router has no /metrics).
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

K55 = os.path.expanduser("~/GEN-AI/k55")
argv, sys.argv = sys.argv, ["coding_probe.py", "localhost", "k76", "/nonexistent", "768"]
sys.path.insert(0, K55)
import coding_probe as cp  # noqa: E402
sys.argv = argv
cp.metrics = lambda: (0.0, 0.0)
PROMPTS = [p for lang in ("python", "cpp", "rust", "go") for p in cp.LANGS[lang][1]]


def one(prompt, extra):
    try:
        return cp.run(prompt, extra)
    except Exception as ex:   # a failed request is recorded, the pass goes on
        return {"error": str(ex)[:300], "decode_tps": None, "completion_tokens": 0, "finish": None}


def main(base, mode, conc, runs, out):
    cp.BASE = base.rstrip("/")
    extra = cp.SETTINGS[mode]
    one("Write a Python hello world.", cp.SETTINGS["t0-nothink"])   # k55's warm-up, not counted
    res = {"mode": mode, "conc": conc, "max_tokens": cp.MAX_TOK, "base": base, "runs": []}
    for i in range(runs):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(conc) as ex:
            rows = list(ex.map(lambda p: one(p, extra), PROMPTS))
        wall = time.perf_counter() - t0
        tps = [r["decode_tps"] for r in rows if r.get("decode_tps")]
        res["runs"].append({"n": len(rows), "ok": len(tps), "tps": tps, "wall_s": wall,
                            "hit_max": sum(r.get("finish") == "length" for r in rows),
                            "agg_tps": sum(r.get("completion_tokens") or 0 for r in rows) / wall, "rows": rows})
        print(f"{mode} c{conc} run {i + 1}: {len(tps)}/{len(rows)} ok, wall {wall:.0f} s, "
              f"agg {res['runs'][-1]['agg_tps']:.1f} tok/s, hit max_tokens {res['runs'][-1]['hit_max']}", flush=True)
        json.dump(res, open(out, "w"), indent=1)
    return 0 if all(r["ok"] == r["n"] for r in res["runs"]) else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        assert [len(cp.LANGS[k][1]) for k in ("python", "cpp", "rust", "go")] == [12, 8, 8, 8] and len(PROMPTS) == 36
        assert cp.MAX_TOK == 768 and set(cp.SETTINGS) == {"t0-nothink", "default"}
        print("coding selftest OK (36 prompts, max_tokens 768)")
    else:
        b, m, c, n, o = sys.argv[1:6]
        sys.exit(main(b, m, int(c), int(n), o))
