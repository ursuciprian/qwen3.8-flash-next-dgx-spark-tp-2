#!/usr/bin/env python3
"""k18: vLLM Marlin W4A16 MoE (INT4 g128, BF16 activations) at Qwen3.8 TP=1 shapes, reference ceiling.

Runs inside the dime-online v16b image (their vLLM), with the same routing harness as
bench_moe_pad.py / bench_moe_a16.py: 512-expert bank, each timed call routes M tokens to D distinct
experts from a window that advances by D per call, >= 512 MB rotation, CUDA graph of all calls,
median of replays. Weight values are random (timing only): uint4b8 packed in Marlin layout,
BF16 group scales (K/128 groups). fused_marlin_moe includes its moe_align_block_size and the topk
sum, as in their serving graphs.

    python3 bench_moe_marlin.py --intermediate 640 --shapes 5:33,10:60,20:96,40:170
"""

from __future__ import annotations

import argparse

import torch

from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.scalar_type import scalar_types

E, K, TOPK, GROUP = 512, 2560, 10, 128
MIN_ROTATION_BYTES = 512 << 20


def expert_bytes(i):
    return 3 * i * K // 2 + (K // GROUP * 2 * i + i // GROUP * K) * 2


def route_sets(device, m, d):
    calls = -(-E // d) + 1
    routes = torch.arange(m * TOPK).reshape(m, TOPK) % d
    return [((routes + c * d) % E).to(torch.int32).to(device).contiguous() for c in range(calls)]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--intermediate", type=int, default=640)
    p.add_argument("--shapes", default="5:33,10:60,20:96,40:170")
    p.add_argument("--replays", type=int, default=30)
    args = p.parse_args()
    i = args.intermediate
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(1)
    # Marlin 4-bit layout: w1 [E, K/16, 2I*2] int32, w2 [E, I/16, K*2] int32 (fused_marlin_moe asserts these)
    w1 = torch.randint(-2**31, 2**31 - 1, (E, K // 16, 2 * i * 2), dtype=torch.int32, device=dev, generator=g)
    w2 = torch.randint(-2**31, 2**31 - 1, (E, i // 16, K * 2), dtype=torch.int32, device=dev, generator=g)
    s1 = (torch.rand(E, K // GROUP, 2 * i, device=dev, generator=g) * 0.01 + 0.001).to(torch.bfloat16)
    s2 = (torch.rand(E, i // GROUP, K, device=dev, generator=g) * 0.01 + 0.001).to(torch.bfloat16)
    ws = marlin_make_workspace_new(dev, 4)
    qid = scalar_types.uint4b8.id
    eb = expert_bytes(i)
    print(f"bank {E} experts x {eb} B = {E * eb / 1e6:.1f} MB (I={i}, INT4 g{GROUP} + BF16 scales)")
    print(f"{'backend':12} {'M':>3} {'D':>4} {'us/call':>9} {'min':>9} {'GB/s':>7} {'rot MB':>7} finite")
    for shape in args.shapes.split(","):
        m, d = (int(v) for v in shape.split(":"))
        sets = route_sets(dev, m, d)
        assert all(s.unique().numel() == d for s in sets)
        rot = len(sets) * d * eb
        assert rot >= MIN_ROTATION_BYTES, rot
        x = (torch.randn(m, K, device=dev, generator=g) * 0.35).to(torch.bfloat16)
        tw = torch.full((m, TOPK), 1.0 / TOPK, device=dev)

        def call(ids):
            return fused_marlin_moe(x, w1, w2, None, None, s1, s2, tw, ids, qid,
                                    global_num_experts=E, workspace=ws)

        out = call(sets[0])
        finite = bool(torch.isfinite(out).all())
        for s in sets:
            call(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for s in sets:
                call(s)
        graph.replay()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(args.replays):
            a.record()
            graph.replay()
            b.record()
            b.synchronize()
            samples.append(a.elapsed_time(b) * 1e3 / len(sets))
        graph.reset()
        samples.sort()
        t = samples[len(samples) // 2]
        print(f"{'marlin-w4a16':12} {m:>3} {d:>4} {t:>9.1f} {samples[0]:>9.1f} {d * eb / (t * 1e3):>7.1f} "
              f"{rot >> 20:>7} {finite}", flush=True)


if __name__ == "__main__":
    main()
