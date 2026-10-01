#!/usr/bin/env python3
"""r8: BF16 projections of the Qwen3.8 MTP draft layer, cuBLAS vs b12x bf16_gemv (opus-kernel-16).

The draft layer's attention, shared expert and router weights are unquantized BF16 in the
checkpoint. At c1 they run as cuBLAS gemvx (M=1) / cutlass_80_wmma (M=5) at ~160 GB/s
(r8-prof fresh-c1 trace: q|k|v 13312x2560 417 us, o_proj 2560x6144 193 us per M=1 pass).
This times the same shapes through torch.nn.functional.linear (= cuBLAS, what serving runs)
and through every b12x bf16_gemv backend (simt rows_per_tile 1/2/4/8, mma), inside a CUDA
graph that rotates enough weight copies to keep L2 from serving them, and checks each
b12x result against cuBLAS (BF16 inputs, FP32 accumulate: only the summation order differs).

    python3 bench_draft_gemv.py [--rows 1,5] [--replays 30]
"""

import argparse

import torch
import torch.nn.functional as F

SHAPES = {  # name: (N, K) per TP=1 rank
    "qkv": (13312, 2560),
    "o_proj": (2560, 6144),
    "shared_gate_up": (1280, 2560),
    "shared_down": (2560, 640),
    "router": (512, 2560),
}
MIN_ROTATION = 256 << 20  # GB10 L2 is 24 MB; rotate far past it


def timed(fn, copies, replays):
    for c in copies:
        fn(c)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in copies:
            fn(c)
    g.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(replays):
        s.record(); g.replay(); e.record(); e.synchronize()
        out.append(s.elapsed_time(e) * 1e3 / len(copies))
    g.reset()
    return sorted(out)[len(out) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", default="1,5")
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    a = ap.parse_args()
    from b12x.gemm.bf16_gemv._kernel import compile_projection

    dev = torch.device("cuda")
    print(f"{'shape':16} {'M':>2} {'variant':12} {'us':>8} {'GB/s':>7} {'rel err':>8}")
    for name in a.shapes.split(","):
        n, k = SHAPES[name]
        nbytes = n * k * 2
        ncopy = max(2, -(-MIN_ROTATION // nbytes))
        ws = [(torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16) for _ in range(ncopy)]
        for m in (int(v) for v in a.rows.split(",")):
            x = (torch.randn(m, k, device=dev)).to(torch.bfloat16)
            ref = F.linear(x, ws[0]).float()
            outs = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            variants = [("cublas", None)] + [(f"simt{r}", ("simt", r)) for r in (1, 2, 4, 8)]
            if n >= 256:
                variants.append(("mma", ("mma", 8)))
            for label, cfg in variants:
                if cfg is None:
                    fn = lambda w: torch.mm(x, w.t(), out=outs)  # noqa: E731  (F.linear's cuBLAS call)
                else:
                    try:
                        launch = compile_projection(dev.index or 0, cfg[0], cfg[1], n, k,
                                                    "bfloat16", "bfloat16", "bfloat16", None)
                    except Exception as exc:  # noqa: BLE001  report and keep going
                        print(f"{name:16} {m:>2} {label:12} compile failed: {type(exc).__name__}: {exc}"[:160])
                        continue
                    fn = lambda w, launch=launch: launch(x, w, outs, None)  # noqa: E731
                fn(ws[0]); torch.cuda.synchronize()
                err = ((outs.float() - ref).norm() / ref.norm()).item()
                us = timed(fn, ws, a.replays)
                print(f"{name:16} {m:>2} {label:12} {us:8.1f} {nbytes / us / 1e3:7.1f} {err:8.1e}", flush=True)
        del ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
