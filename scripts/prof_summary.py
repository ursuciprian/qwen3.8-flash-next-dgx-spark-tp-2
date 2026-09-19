#!/usr/bin/env python3
"""prof_summary.py <label> [steps] <rank0_trace.json> [rank1_trace.json ...]

Reads chrome-trace JSON from torch.profiler.export_chrome_trace, groups CUDA
kernel events by name, and prints per-label/per-rank: count, total ms, mean
us, % of total CUDA time (top 25), plus decode-step wall and GPU idle
fraction. `steps` is the profiler's step count for that window (from
VLLM_LOCAL_PROF_STEPS -- the trace has no explicit per-step CPU marker to
count, so this is passed in rather than inferred); defaults to 100.
Classifies each kernel name into a bucket by pattern.

ponytail: pattern matching is a fixed keyword table, not a learned classifier
-- good enough for one profiling pass; extend BUCKET_PATTERNS if a new kernel
family shows up unclassified in "other".
"""
import json
import sys
import gzip
from collections import defaultdict

BUCKET_PATTERNS = [
    # all-reduce/roce checked before gemm: b12x's ROCE one-shot/allgather
    # collective kernels are named "kernel_cutlass_kernel_b12xcommroce_...",
    # so "cutlass" would otherwise swallow them into the gemm bucket.
    ("all-reduce/nccl", ["nccl", "allreduce", "all_reduce", "reduce_scatter", "all_gather", "roce"]),
    ("attention", ["attn", "attention", "flash", "paged_attention", "sdpa", "qsa"]),
    ("gdn/ssm", ["gdn", "mamba", "ssm", "conv1d", "selective_scan", "chunk_scan"]),
    ("gemm", ["gemm", "cutlass", "cublas", "matmul", "linear", "lm_head", "nvfp4", "mm_", "fp8_gemm"]),
    ("sampler/argmax", ["argmax", "sampl", "topk", "top_k", "top_p", "softmax", "multinomial"]),
    ("mtp/draft", ["mtp", "draft", "speculat", "eagle", "rejection"]),
    ("memcpy/other", ["memcpy", "memset", "copy_"]),
]


def classify(name):
    n = name.lower()
    for bucket, keys in BUCKET_PATTERNS:
        if any(k in n for k in keys):
            return bucket
    return "other"


def load_events(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data.get("traceEvents", data) if isinstance(data, dict) else data


def merged_busy_us(intervals):
    """Union of (start, end) intervals -> total covered length. Needed because
    concurrent CUDA streams overlap: summing raw kernel durations can exceed
    wall time, which would make idle fraction negative and meaningless."""
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def summarize(path, steps):
    events = load_events(path)
    cuda_events = defaultdict(lambda: [0, 0.0])  # name -> [count, total_us]
    gpu_intervals = []
    wall_min, wall_max = None, None
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        name = e.get("name", "")
        dur = e.get("dur", 0) or 0
        ts = e.get("ts", 0) or 0
        if wall_min is None or ts < wall_min:
            wall_min = ts
        if wall_max is None or ts + dur > wall_max:
            wall_max = ts + dur
        # Only actual GPU-side execution (cat=="kernel"/"gpu_memcpy"/"gpu_memset").
        # "cuda_runtime"/"cuda_driver" are CPU-side launch-API call durations
        # (cudaGraphLaunch, cudaEventSynchronize, ...) -- real host overhead,
        # but not GPU work, and counting them here both double-counts time
        # against overlapping async kernels and wrongly inflates "other".
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            cuda_events[name][0] += 1
            cuda_events[name][1] += dur
            gpu_intervals.append((ts, ts + dur))

    total_cuda_us = sum(v[1] for v in cuda_events.values())
    busy_union_us = merged_busy_us(gpu_intervals)
    wall_us = (wall_max - wall_min) if (wall_min is not None and wall_max is not None) else 0

    rows = sorted(cuda_events.items(), key=lambda kv: -kv[1][1])
    return {
        "rows": rows,
        "total_cuda_us": total_cuda_us,
        "busy_union_us": busy_union_us,
        "wall_us": wall_us,
        "num_steps": steps,
        "step_wall_us": (wall_us / steps) if steps else None,
    }


def print_report(label, rank, summary):
    total = summary["total_cuda_us"]
    busy = summary["busy_union_us"]
    wall = summary["wall_us"]
    idle_frac = (1 - busy / wall) if wall else None
    print(f"\n=== label={label} rank={rank} ===")
    print(f"decode steps in window (by construction, VLLM_LOCAL_PROF_STEPS): {summary['num_steps']}")
    if summary["step_wall_us"] is not None:
        print(f"wall per decode step: {summary['step_wall_us']/1000:.3f} ms")
    if wall:
        print(f"window wall: {wall/1000:.2f} ms, CUDA busy (sum, streams overlap): {total/1000:.2f} ms, "
              f"CUDA busy (union, non-overlapping): {busy/1000:.2f} ms, idle fraction: {idle_frac:.3f}")
    else:
        print("window wall: n/a")
    print(f"{'kernel':60s} {'count':>8s} {'total_ms':>10s} {'mean_us':>10s} {'% cuda':>8s} bucket")
    bucket_totals = defaultdict(float)
    for name, (count, total_us) in summary["rows"]:
        bucket_totals[classify(name)] += total_us
    for name, (count, total_us) in summary["rows"][:25]:
        pct = 100 * total_us / total if total else 0
        mean_us = total_us / count if count else 0
        print(f"{name[:60]:60s} {count:8d} {total_us/1000:10.3f} {mean_us:10.2f} {pct:7.2f}% {classify(name)}")
    print("\nbucket shares (of total CUDA time):")
    for bucket, us in sorted(bucket_totals.items(), key=lambda kv: -kv[1]):
        pct = 100 * us / total if total else 0
        print(f"  {bucket:20s} {us/1000:10.3f} ms  {pct:6.2f}%")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    label = sys.argv[1]
    steps = int(sys.argv[2]) if sys.argv[2].isdigit() else 100
    paths = sys.argv[3:] if sys.argv[2].isdigit() else sys.argv[2:]
    for i, path in enumerate(paths):
        summary = summarize(path, steps)
        print_report(label, i, summary)


if __name__ == "__main__":
    main()
