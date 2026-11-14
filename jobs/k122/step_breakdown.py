#!/usr/bin/env python3
"""step_breakdown.py <steps> <rank_trace.json[.gz]> [--json out.json]

Per-decode-step breakdown of a mods/vllm-decode-profiler chrome trace into the
c1-c4 round categories (2026-09-26, opus-executor-3). Kernels are grouped by
CUDA graph first: the largest graph is the target forward, graphs that run
mtp_feedback / MTP kernels are the draft passes, graph 0 (no graph) is eager
work (main lm_head, sampling/rejection, metadata). Then by kernel name.
GPU idle = window wall - union of kernel intervals (all streams).

ponytail: name-keyword table, extend CATS when a new kernel family lands in "other".

r8 (opus-kernel-11): PLE category (b12x hash + lookup kernels, eager at TP=1 where the
table is a disk table) and the GPU idle split by phase transition (prev -> next kernel
group); "eager->target" is the gap in front of the target graph, where TP=1 waits for
the host PLE gather.
"""
import collections, gzip, json, sys

CATS = [  # (category, keywords) checked in order, lower-case substrings
    ("PLE", ["_hash_ids_kernel", "_request_ids_kernel", "_lookup_kernel", "ple_"]),
    ("all-reduce (RoCE)", ["roce", "nccl", "allreduce", "all_reduce", "allgather"]),
    ("QSA select", ["qsa_score", "qsa_select", "stable_select", "selectkernel", "qsa_topk", "qsa_index",
                    "representativescore", "qsa_compress", "qsa_"]),
    ("attention", ["attention", "attn", "paged_selected", "flash"]),
    ("GDN", ["gdn", "causal_conv1d", "recurrent", "mamba", "ssm"]),
    ("NVFP4 MoE", ["b12xmoe", "moe", "route", "_sorted_softmax_topk", "expert"]),
    ("norm/hyperconn", ["norm", "hyperconnection", "_gate_mean", "rmsnorm"]),
    ("dense GEMM (NVFP4/FP8)", ["libdense_gemm", "blockscaled", "_quantize", "fp8", "nvfp4", "mxfp8"]),
    ("BF16 GEMM", ["nvjet", "cutlass_80_wmma", "cublas", "gemvx", "gemv", "splitkreduce", "gemm", "cutlass"]),
    ("sampling/rejection", ["sampl", "reject", "argmax", "multinomial", "uniform", "top_k", "topk", "softmax",
                            "exponential", "probs", "logit"]),
    ("memcpy/fill/meta", ["memcpy", "memset", "fill", "copy", "index", "elementwise", "slot_mapping", "metadata",
                          "arange", "cumsum", "multi_tensor_apply", "update_draft"]),
]


def cat_of(name):
    n = name.lower()
    for c, keys in CATS:
        if any(k in n for k in keys):
            return c
    return "other"


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    d = json.load(op(path, "rt"))
    return d["traceEvents"] if isinstance(d, dict) else d


def union(iv):
    iv.sort(); tot = 0.0; s = e = None
    for a, b in iv:
        if s is None: s, e = a, b
        elif a <= e: e = max(e, b)
        else: tot += e - s; s, e = a, b
    return tot + ((e - s) if s is not None else 0.0)


def main():
    steps = int(sys.argv[1]); path = sys.argv[2]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    ev = [e for e in load(path) if e.get("ph") == "X"]
    K = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    t0 = min(e["ts"] for e in ev); t1 = max(e["ts"] + e.get("dur", 0) for e in ev)
    graphs = collections.defaultdict(list)
    for e in K:
        graphs[e.get("args", {}).get("graph id", 0) or 0].append(e)
    tot = {g: sum(e["dur"] for e in ks) for g, ks in graphs.items()}
    target = max((g for g in graphs if g != 0), key=lambda g: tot[g])
    role = {}
    for g, ks in graphs.items():
        names = " ".join(set(e["name"].lower() for e in ks))
        role[g] = ("eager" if g == 0 else "target" if g == target
                   else "draft" if ("mtp" in names or "draft" in names or len(ks) < len(graphs[target]) / 3) else "other-graph")
    rows = collections.defaultdict(float)  # (role, cat) -> us
    per_kernel = collections.defaultdict(lambda: [0, 0.0])
    for g, ks in graphs.items():
        cnt = collections.Counter(e["name"] for e in ks)
        for e in ks:
            c = cat_of(e["name"])
            # lm_head: big dense GEMM launched <= 5x per step (verify head in eager/target, MTP head in draft graphs)
            if c in ("dense GEMM (NVFP4/FP8)", "BF16 GEMM") and cnt[e["name"]] <= 5 * steps and e["dur"] >= 250:
                c = "MTP lm_head" if role[g] == "draft" else "main lm_head"
            rows[(role[g], c)] += e["dur"]
            pk = per_kernel[(role[g], e["name"][:100])]; pk[0] += 1; pk[1] += e["dur"]
    wall = t1 - t0; busy = union([(e["ts"], e["ts"] + e["dur"]) for e in K])
    gaps = collections.defaultdict(float); end = prev = None  # idle by prev -> next group
    for e in sorted(K, key=lambda e: e["ts"]):
        r = role[e.get("args", {}).get("graph id", 0) or 0]
        if end is not None and e["ts"] > end:
            gaps[f"{prev}->{r}"] += e["ts"] - end
        if end is None or e["ts"] + e["dur"] > end:
            end, prev = e["ts"] + e["dur"], r
    step_ms = wall / steps / 1000
    print(f"trace {path}\nwall/step {step_ms:.2f} ms  busy(union)/step {busy/steps/1000:.2f} ms  "
          f"GPU idle/launch gaps {100*(1-busy/wall):.1f}%  kernel-sum/step {sum(tot.values())/steps/1000:.2f} ms")
    print(f"graphs: " + ", ".join(f"{role[g]}={tot[g]/steps/1000:.2f}ms" for g in sorted(graphs, key=lambda g: -tot[g])))
    ksum = sum(tot.values())
    print(f"{'group':8s} {'category':24s} {'us/step':>9s} {'% kernel-sum':>12s}")
    bycat = collections.defaultdict(float)
    for (r, c), us in sorted(rows.items(), key=lambda x: -x[1]):
        bycat[c if r != "draft" or c == "MTP lm_head" else "MTP draft passes (ex lm_head)"] += us
        print(f"{r:8s} {c:24s} {us/steps:9.1f} {100*us/ksum:11.1f}%")
    print("\nsummary categories (% of step wall; overlap between streams makes the sum exceed busy):")
    summ = {c: us / steps for c, us in bycat.items()}
    summ["GPU idle/launch gaps"] = (wall - busy) / steps
    for c, us in sorted(summ.items(), key=lambda x: -x[1]):
        print(f"  {c:32s} {us/1000:7.2f} ms  {100*us/(wall/steps):5.1f}%")
    print("\nGPU idle by transition (prev -> next kernel group):")
    for k, us in sorted(gaps.items(), key=lambda x: -x[1]):
        print(f"  {k:24s} {us/steps/1000:7.3f} ms/step")
    print("\ntop kernels:")
    for (r, n), (c, us) in sorted(per_kernel.items(), key=lambda x: -x[1][1])[:25]:
        print(f"  {r:6s} {us/steps:8.1f} us/step x{c/steps:5.1f}  [{cat_of(n)}] {n}")
    if out_json:
        json.dump({"step_ms": step_ms, "idle_pct": 100 * (1 - busy / wall), "summary_us": summ,
                   "gaps_us": {k: us / steps for k, us in gaps.items()},
                   "rows": {f"{r}|{c}": us / steps for (r, c), us in rows.items()}}, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
