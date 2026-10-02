#!/usr/bin/env python3
"""k18: b12x NVFP4 MoE at the TP=1 width (I=640), W4A4 vs the W4A16 (A16) path on the same weights.

Same harness as bench_moe_pad.py (one 512-expert bank, every timed call routes M tokens to D
distinct experts from a window that advances by D per call, >= 512 MB rotation so weights stream
from DRAM, CUDA graph of all calls, median of replays). Weights are planned the way vLLM plans
them (ModelOpt NVFP4, w31 source layout, mode A4) and prepared with autotuning on, so each backend
runs the plan a serving boot would select.

Backends:
  dynamic        stock A4 dynamic kernel, fixed config, no autotune (bench_moe_pad baseline)
  a4             serving A4 plan, autotuned
  a16            same weights with ActivationSpec(a16_max_tokens=64): the W4A16 backend
                 (BF16 activations, NVFP4 weights) that VLLM_B12X_A16_MAX_TOKENS selects, autotuned
  a16-direct     a16 with w4a16_route_mode forced to direct (no autotune)
  a16-packed     a16 with w4a16_route_mode forced to packed (route-pack, no autotune)

Each row also prints the cosine of the backend's output against the a4 output on the same call,
to catch a backend returning garbage (A4 and A16 differ by activation rounding, so ~0.99, not 1).

    python3 bench_moe_a16.py --intermediate 640 --shapes 5:33,10:60,20:96,40:170
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_moe_pad as pad  # noqa: E402

from b12x._lib.intrinsics import swizzle_block_scale  # noqa: E402
from b12x.moe import fused_moe  # noqa: E402
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall  # noqa: E402

A16_CUTOFF = 64


def make_experts(device, *, a16_cutoff):
    E, K, I = pad.E, pad.K, pad.I
    gen = torch.Generator(device=device).manual_seed(1)
    w1 = torch.randint(0, 256, (E, 2 * I, K // 2), dtype=torch.uint8, device=device, generator=gen)
    w2 = torch.randint(0, 256, (E, K, I // 2), dtype=torch.uint8, device=device, generator=gen)

    def scales(shape):
        exp = torch.randint(-9, -4, shape, device=device, generator=gen).float()
        return swizzle_block_scale(torch.exp2(exp).to(torch.float8_e4m3fn)).contiguous()

    kwargs = {"a16_max_tokens": a16_cutoff} if a16_cutoff else {}
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w31"),
        activation=fused_moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16, **kwargs),
        geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=K, intermediate_size=I),
    )
    return fused_moe.prepare_weights(plan=plan, weights=fused_moe.PackedWeights(
        w13=w1, w2=w2, w13_block_scales=scales((E, 2 * I, K // 16)),
        w2_block_scales=scales((E, K, I // 16)),
        w13_global_scales=torch.full((E,), 0.6, device=device),
        w2_global_scales=torch.full((E,), 0.7, device=device),
        input_scale=torch.full((E,), 48.0, device=device),
        intermediate_scale=torch.full((E,), 384.0, device=device),
    ))


def make_plan(experts, m, backend):
    override = None
    if backend == "dynamic":
        override = fused_moe.MoeDecodeConfig(
            backend="dynamic", route_planner="internal", max_active_clusters=None,
            dynamic_tile_m=16, dynamic_route_mode="grouped", nvfp4_share_input=True,
        )
    elif backend in ("a16-direct", "a16-packed"):
        override = fused_moe.MoeDecodeConfig(
            backend="w4a16", route_planner="internal", max_active_clusters=None,
            w4a16_route_mode=backend.split("-")[1],
        )
    return fused_moe.plan_execution(
        experts=experts, capacity=fused_moe.ExecutionCapacity(max_tokens=m, top_k=pad.TOPK),
        invocation=FrozenMapping({"tuning_route_pattern": "cyclic_disjoint_topk"}), override=override,
    )


def bench(device, experts, backend, m, d, *, replays, reference=None):
    plan = make_plan(experts, m, backend)
    autotune = backend in ("a4", "a16")
    a = (torch.randn(m, pad.K, device=device, generator=torch.Generator(device=device).manual_seed(m)) * 0.35
         ).to(torch.bfloat16)
    weights = torch.full((m, pad.TOPK), 1.0 / pad.TOPK, device=device)
    sets = pad._route_sets(device, m, d)
    rotation = len(sets) * d * pad.EXPERT_BYTES
    if rotation < pad.MIN_ROTATION_BYTES:
        raise RuntimeError(f"weight rotation {rotation >> 20} MB < 512 MB: L2 would fake bandwidth")
    scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=device) for s in plan.scratch_specs())
    out = torch.empty(m, pad.K, dtype=torch.bfloat16, device=device)

    def factory(state):
        sc = tuple(torch.empty(s.shape, dtype=s.dtype, device=device) for s in state.scratch.scratch_specs())
        o = torch.empty_like(a)
        b = state.bind(a=a, topk_ids=sets[0], topk_weights=weights, scratch=sc, output=o,
                       input_scales_static=True)
        return PreparedCall(run=b.run, output=o, owners=(b, sc))

    with PreparationSession(device=device, autotune=autotune, compile_workers=0) as session:
        session.prepare((plan.request(name=f"bench-{backend}-{m}", prepare_call=factory),))
        session.freeze()
        bindings = [fused_moe.bind(plan, a=a, topk_ids=s, topk_weights=weights, scratch=scratch,
                                   output=out, input_scales_static=True) for s in sets]
        cfg = getattr(getattr(bindings[0], "state", None), "config", None)
        fused_moe.run(binding=bindings[0])
        first = out.float().clone()
        for b in bindings:
            fused_moe.run(binding=b)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for b in bindings:
                fused_moe.run(binding=b)
        graph.replay()
        torch.cuda.synchronize(device)
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(replays):
            start.record()
            graph.replay()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop) * 1e3 / len(bindings))
        graph.reset()
    samples.sort()
    t = samples[len(samples) // 2]
    finite = bool(torch.isfinite(first).all())
    cos = (torch.nn.functional.cosine_similarity(first.flatten(), reference.flatten(), dim=0).item()
           if reference is not None else 1.0)
    return dict(backend=backend, m=m, d=d, us=t, us_min=samples[0], gbps=d * pad.EXPERT_BYTES / (t * 1e3),
                rotation_mb=rotation >> 20, first=first, finite=finite, cos=cos,
                config=(f"{cfg.backend}/{cfg.w4a16_route_mode or cfg.dynamic_route_mode}/tm{cfg.dynamic_tile_m}"
                        if cfg is not None else "?"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backends", default="a4,dynamic,a16,a16-direct,a16-packed")
    parser.add_argument("--intermediate", type=int, default=640)
    parser.add_argument("--shapes", default="5:33,10:60,20:96,40:170")
    parser.add_argument("--replays", type=int, default=30)
    args = parser.parse_args()
    pad.set_intermediate(args.intermediate)
    device = torch.device("cuda")
    backends = args.backends.split(",")
    a4 = make_experts(device, a16_cutoff=0)
    a16 = make_experts(device, a16_cutoff=A16_CUTOFF) if any(b.startswith("a16") for b in backends) else None
    print(f"bank {pad.E} experts x {pad.EXPERT_BYTES} B = {pad.E * pad.EXPERT_BYTES / 1e6:.1f} MB "
          f"(I={pad.I}, NVFP4 payload + E4M3 block scales)")
    print(f"{'backend':12} {'M':>3} {'D':>4} {'us/call':>9} {'min':>9} {'GB/s':>7} {'rot MB':>7} "
          f"{'cos(a4)':>8} finite config")
    for shape in args.shapes.split(","):
        m, d = (int(v) for v in shape.split(":"))
        reference = None
        for backend in backends:
            try:
                r = bench(device, a16 if backend.startswith("a16") else a4, backend, m, d,
                          replays=args.replays, reference=reference)
            except Exception as exc:  # report and keep going: one failing backend must not hide the rest
                print(f"{backend:12} {m:>3} {d:>4} FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                continue
            if backend == "a4":
                reference = r["first"]
            print(f"{r['backend']:12} {r['m']:>3} {r['d']:>4} {r['us']:>9.1f} {r['us_min']:>9.1f} "
                  f"{r['gbps']:>7.1f} {r['rotation_mb']:>7} {r['cos']:>8.4f} {r['finite']} {r['config']}",
                  flush=True)


if __name__ == "__main__":
    main()
