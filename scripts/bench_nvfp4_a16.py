#!/usr/bin/env python3
"""r9 requant gate 1 (opus-kernel-17, 2026-10-02): b12x weight-only NVFP4 A16 vs MXFP8 A16.

For the dense shapes in the r9 requant scope (docs/r9-dense-nvfp4.md), at TP=1 and TP=2 shard
shapes and live rows M, race every legal A16 tile/split config for each recipe (the autotuner's
race), with enough weight copies per CUDA graph to exceed L2 (weights read from DRAM, as in
back-to-back serving layers). Prints the best per recipe and ratio = NVFP4 best / MXFP8 best.
Gate: ratio <= 0.65 at M=5. Outputs are checked against an FP32 dequant reference first.
Harness follows k15's benchmarks/benchmark_mxfp8_gemv.py; b12x is the image's (b1.4, b7fbaf96).

  python3 bench_nvfp4_a16.py --out /out/nvfp4.jsonl
"""
import argparse, itertools, json, statistics

import torch

from b12x.gemm import blockscaled
from b12x.gemm.blockscaled import _a16, _preparation, _tuning
from b12x.gemm.blockscaled._tuning import BlockscaledConfig
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation.device import detect_device
from b12x._lib.intrinsics import swizzle_block_scale
from b12x._lib.runtime_control import kernel_resolution_guard

HC = [("hc.mix_down", 336, 10240), ("hc.mix_up", 10240, 320), ("hc.final_down", 320, 10240)]
SHAPES = {1: [("gdn.in_proj_qkvz", 16384, 2560), ("attn.qkv", 13312, 2560), ("gdn.out/attn.o", 2560, 6144), *HC],
          2: [("gdn.in_proj_qkvz", 8192, 2560), ("attn.qkv", 6656, 2560), ("gdn.out/attn.o", 2560, 3072), *HC]}
A16 = tuple(itertools.product((64, 128), (64, 128), (1, 2, 4, 8)))
L2_TARGET = 96 << 20
LUT = [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]


def weight(recipe, n, k):
    if recipe == "nvfp4":
        codes = torch.randint(0, 16, (n, k), device="cuda")
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
        scales = (torch.rand(n, k // 16, device="cuda") * 2 + 0.0625).to(torch.float8_e4m3fn)
        gs = torch.tensor([0.125], dtype=torch.float32, device="cuda")
        w = blockscaled.pack_weight(packed, swizzle_block_scale(scales), recipe="nvfp4", global_scale=gs)
        ref = torch.tensor(LUT, device="cuda")[codes] * scales.float().repeat_interleave(16, 1)
        return w, ref.to(torch.bfloat16).float() * gs, n * k // 2 + n * (k // 16)
    values = (torch.randn(n, k, device="cuda") * 2).to(torch.float8_e4m3fn)
    exponent = torch.randint(118, 124, (n, k // 32), device="cuda", dtype=torch.uint8)
    w = blockscaled.pack_weight(values, exponent)
    ref = values.float() * torch.exp2(exponent.float() - 127).repeat_interleave(32, 1)
    return w, ref, n * k + n * (k // 32)


def race(recipe, name, n, k, m, args, device):
    torch.manual_seed(n * 7 + k + m)
    packed, ref_w, nbytes = weight(recipe, n, k)
    copies = max(1, min(args.max_copies, -(-L2_TARGET // nbytes)))
    weights = [packed] + [weight(recipe, n, k)[0] for _ in range(copies - 1)]
    source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    reference = source.float() @ ref_w.T
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    query = blockscaled.query_from_call(source, packed, activation_mode="a16", out=out, expected_m=m)
    cands, seen = [], set()
    for tn, tk, s in A16:
        c = BlockscaledConfig(mode="a16", tile_n=tn, tile_k=tk, split_k=s)
        try:
            _tuning._validate_config(query, c, device)
        except ValueError:
            continue
        key = repr(_tuning._equivalence(query, device, c))
        if key not in seen:
            seen.add(key); cands.append((f"a16_{tn}_{tk}_{s}", c))
    need = max(_preparation._workspace_bytes(query, c) for _, c in cands)
    scratch = torch.empty(max(need, 16), device="cuda", dtype=torch.uint8)
    query = blockscaled.query_from_call(source, packed, activation_mode="a16", out=out, workspace=scratch, expected_m=m)
    plans, requests = {}, []
    for label, c in cands:
        for i, w in enumerate(weights):
            plan = blockscaled.plan(query, override=c)
            v, sc, gs, _ = _a16._weight_parts(w)
            requests.append(plan.request(name=f"{recipe}.{name}.{label}.{i}", prepare_call=lambda st, v=v, sc=sc, gs=gs: PreparedCall(
                run=lambda: st.run(source, v, sc, gs, out=out, workspace=scratch))))
            plans[(label, i)] = plan
    with PreparationSession(device=source.device, autotune=False, compile_workers=args.workers) as session:
        session.prepare(tuple(requests))
        graphs = {}
        with kernel_resolution_guard("nvfp4 a16 bench"):
            for label, _ in cands:
                out.fill_(float("nan"))
                blockscaled.mm(source, packed, out=out, workspace=scratch, plan=plans[(label, 0)])
                rel = float(torch.linalg.vector_norm(out.float() - reference) / torch.linalg.vector_norm(reference))
                if not rel < 5e-3:
                    raise RuntimeError(f"{recipe} {name} M={m} {label}: relative error {rel}")
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(args.reps):
                        for i, w in enumerate(weights):
                            blockscaled.mm(source, w, out=out, workspace=scratch, plan=plans[(label, i)])
                graphs[label] = g
            calls = args.reps * len(weights)
            for g in graphs.values():
                g.replay()
            torch.cuda.synchronize()
            samples = {l: [] for l in graphs}
            for t in range(args.iters):
                for l in (list(graphs) if t % 2 == 0 else list(graphs)[::-1]):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); graphs[l].replay(); b.record(); b.synchronize()
                    samples[l].append(a.elapsed_time(b) * 1000 / calls)
    med = {l: statistics.median(v) for l, v in samples.items()}
    best = min(med, key=med.get)
    return dict(best=best, us=med[best], gbs=nbytes / med[best] / 1e3, mb=nbytes / 1e6, all_us=med)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, nargs="+", default=[1, 2])
    p.add_argument("--rows", type=int, nargs="+", default=[1, 5, 8])
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--max-copies", type=int, default=24)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = detect_device(torch.device("cuda")).identity
    with open(a.out, "a") as fh:
        for tp in a.tp:
            for name, n, k in SHAPES[tp]:
                for m in a.rows:
                    r8, r4 = race("mxfp8", name, n, k, m, a, device), race("nvfp4", name, n, k, m, a, device)
                    row = dict(tp=tp, name=name, n=n, k=k, m=m, mxfp8=r8, nvfp4=r4, ratio=r4["us"] / r8["us"])
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"TP{tp} {name:18s} N={n:6d} K={k:5d} M={m:2d} mxfp8 {r8['best']:14s} {r8['us']:7.2f}us "
                          f"{r8['gbs']:6.1f}GB/s | nvfp4 {r4['best']:14s} {r4['us']:7.2f}us {r4['gbs']:6.1f}GB/s "
                          f"ratio {row['ratio']:.3f}", flush=True)
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
