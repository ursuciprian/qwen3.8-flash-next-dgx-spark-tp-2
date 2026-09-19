# Phase C: decode-step kernel profiling

Per-kernel decode-step timing for `recipes/eugr/eugr-agents-serve-local16-la.yaml`
at c1 and c8, captured 2026-09-19 with the `vllm-decode-profiler` mod (a
rank-local `torch.profiler` wrapper, no cross-rank RPC). Both nodes: dgx-01
(rank0, TP0) and worker 192.168.100.53 (rank1, TP1).

## Prior blocker (kept for context, now bypassed)

The built-in `--profiler-config` `start_profile`/`stop_profile` endpoint
deadlocks under this 2-node TP=2 boot (`collective_rpc should not be called
on follower node`, then `/health` itself stops responding) -- see git
history of this file for the full trace. `nsys` is also absent from the
image. Both are still true; this pass avoids the endpoint entirely with an
in-process, rank-local profiler instead.

## Mechanism

`mods/vllm-decode-profiler/` appends a monkeypatch to the installed
`vllm/v1/worker/gpu/model_runner.py` (the runner selected by
`VLLM_USE_V2_MODEL_RUNNER=1`, which this recipe already sets): it wraps
`GPUModelRunner.execute_model` so each worker process, independently, can run
a `torch.profiler.profile(activities=[CPU, CUDA])` window over N decode
steps and export a chrome trace. No collective call is added anywhere, so it
cannot hit the follower-node RPC bug.

Two arming mechanisms are implemented:

- `VLLM_LOCAL_PROF_TRIGGER_DIR`: drop a uniquely-named file into that
  directory and profiling starts on the next decode step (file content =
  label). **Tried first, dropped as unreliable**: this worker process is
  long-lived, and a file that reappears at a path the process has
  repeatedly `stat()`'d as missing is not always observed on the next
  `os.listdir()`/`exists()` either -- confirmed by a live diagnostic
  (`heartbeat.log`/`debug.log` instrumentation, since removed from the
  shipped snippet) showing hundreds of ticks with the file demonstrably
  present on disk (a fresh `docker exec python3` saw it every time) but
  invisible to the long-running process. A brand-new filename each round
  worked once, then failed again on the next round at that same repeated
  path -- consistent with some form of long-lived-process directory-cache
  staleness on this bind-mounted cache dir, not chased further.
- `VLLM_LOCAL_PROF_START` (fixed step index) + `VLLM_LOCAL_PROF_LABEL`
  (fixed label): no filesystem polling after boot, so nothing to go stale.
  **This is what actually produced the results below** -- one boot per
  label, `VLLM_LOCAL_PROF_START` set past CUDA-graph warmup, `bench_sweep.py`
  run normally, decode steps 220-320 (c1) / 260-360 (c8) captured.

Output: `/cache/runtime/prof/<label>/rank<tp_rank>.json` (chrome trace),
via `scripts/arm_trigger.sh` for the trigger-dir path (kept in the repo for
a future retry) and via the recipe's env for the fixed-step path actually
used. Every profiler action is try/except-guarded; a failure only disables
profiling for that process, serving is never affected (confirmed: the boots
below all reached the gated `--levels 1` >=95 tok/s bar on la restore).

Recipe: `recipes/eugr/eugr-agents-serve-local16-la-lprof.yaml` (copy of
`la` + `mods: [vllm-decode-profiler]` + the profiler env vars). Analysis:
`scripts/prof_summary.py <label> <steps> <rank0.json> [rank1.json ...]`
groups CUDA-kernel-only trace events (`cat in {kernel, gpu_memcpy,
gpu_memset}` -- `cuda_runtime`/`cuda_driver` categories are CPU-side launch
API durations, not GPU work, and are excluded) by name, buckets them by
pattern, and reports GPU idle fraction from the *union* of kernel intervals
(the naive sum double-counts time across the two concurrent CUDA streams
b12x/ROCE communication uses, which made idle fraction go negative before
the fix).

## c1 (single request)

`bench_sweep.py --levels 1`: **94.4 tok/s** during profiling (baseline
without the mod, immediately after: **98.8 tok/s** -- the ~4% delta is
profiler overhead from the CPU+CUDA event recorder, not a regression).

Wall per decode step: rank0 **55.03 ms**, rank1 **56.47 ms**. GPU idle
fraction (union-of-kernels basis): rank0 **0.054**, rank1 **0.076** -- i.e.
the GPU is busy ~93-95% of wall time even at concurrency 1; the bottleneck
is GPU compute (small per-step GEMMs), not host-side gaps.

Top 25 kernels, rank0 (rank1 is materially identical, all deltas <1pp;
totals not repeated):

| kernel | count | total ms | mean us | % CUDA | bucket |
|---|---:|---:|---:|---:|---|
| moe_sharedkernelssiluMoEDynamicKer... | 4704 | 1472.450 | 313.02 | 26.72% | gemm |
| cutlass::Kernel2\<wmma_tensorop_bf16... | 24576 | 1278.668 | 52.03 | 23.20% | gemm |
| b12x_libdense_gemmDenseGemmKernel (a) | 8636 | 860.007 | 99.58 | 15.61% | gemm |
| b12x_libdense_gemmDenseGemmKernel (b) | 4608 | 205.051 | 44.50 | 3.72% | gemm |
| b12xcommroce_oneshot_cute_RoceOneshotL... | 10679 | 159.354 | 14.92 | 2.89% | all-reduce/nccl |
| b12x_libdense_gemmDenseGemmKernel (c) | 5868 | 100.758 | 17.17 | 1.83% | gemm |
| b12xsequencegdn_decode_cute_kernels_Pa... | 3528 | 100.373 | 28.45 | 1.82% | gdn/ssm |
| gemvx::kernel (gemv) | 588 | 90.981 | 154.73 | 1.65% | gemm |
| b12xnormhyperconnection_cute_PackedCom... | 10094 | 90.912 | 9.01 | 1.65% | gemm |
| _gate_mean_kernel | 10682 | 69.680 | 6.52 | 1.26% | other |
| b12xsequencemtp_feedback_cute_prefillM... (a) | 392 | 59.791 | 152.53 | 1.09% | gemm |
| ncclDevKernel_AllReduce_bf16_RING | 101 | 59.746 | 591.55 | 1.08% | all-reduce/nccl |
| b12xattentionpaged_selected_forward_Se... | 1568 | 58.945 | 37.59 | 1.07% | attention |
| b12xmoe_sharedkernelsw4a16kernelW4A16F... | 392 | 54.354 | 138.66 | 0.99% | gemm |
| cutlass::Kernel2\<wmma_tensorop_bf16...\> (b) | 978 | 51.080 | 52.23 | 0.93% | gemm |
| cudaDeviceSynchronize (host sync, not GPU exec) | 1 | 49.587 | 49587.18 | 0.90%* | other |
| b12x_libdense_gemmDenseGemmKernel (d) | 4656 | 43.749 | 9.40 | 0.79% | gemm |
| b12xgemmblockscaled_reduceWeightOnlySp... | 5072 | 42.188 | 8.32 | 0.77% | gemm |
| cutlass::Kernel2\<wmma_tensorop_bf16...\> (c) | 882 | 34.124 | 38.69 | 0.62% | gemm |
| b12xsequencemtp_feedback_cute_prefillM... (b) | 392 | 33.450 | 85.33 | 0.61% | gemm |
| cublasLt::splitKreduce_kernel | 15378 | 32.639 | 2.12 | 0.59% | gemm |
| at::native::vectorized_elementwise_kernel | 28410 | 32.253 | 1.14 | 0.59% | other |
| nvjet_sm121_tst_mma_128x128x64... | 100 | 27.853 | 278.53 | 0.51% | other |
| _quantize | 6012 | 26.306 | 4.38 | 0.48% | other |
| gemvx::kernel (gemv, b) | 931 | 25.543 | 27.44 | 0.46% | gemm |

\* `cudaDeviceSynchronize` was left in `kernel`-adjacent bucketing by an
earlier pass of the script; it is a CPU synchronization call, not GPU
kernel time -- ignore its row, it doesn't affect the bucket totals below
(those come only from `cat in {kernel, gpu_memcpy, gpu_memset}`).

Bucket shares, rank0 (of total CUDA time, corrected for the ROCE
misclassification noted below):

| bucket | ms | % |
|---|---:|---:|
| gemm | 4605.3 | 83.6% |
| other | 375.7 | 6.8% |
| all-reduce/nccl (ROCE + NCCL) | 219.1 | 4.0% |
| gdn/ssm | 127.2 | 2.3% |
| attention | 87.9 | 1.6% |
| sampler/argmax | 54.2 | 1.0% |
| memcpy/other | 39.1 | 0.7% |
| mtp/draft | 2.0 | 0.04% |

rank1: gemm 83.3%, all-reduce/nccl 4.3%, other 6.8%, gdn/ssm 2.3%,
attention 1.6%, sampler 1.0%, memcpy 0.7%, mtp 0.04% -- same shape.

(Caveat: `prof_summary.py`'s first pass classified the ROCE one-shot/allgather
collective kernels as `gemm` because their names contain "cutlass" and the
all-reduce pattern check ran after the gemm check; fixed in the script
by checking all-reduce patterns first. The c1 numbers above are hand-corrected
from that first pass's kernel table since the raw c1 traces were deleted
before the fix landed; c8 below was re-run against the fixed script directly.)

## c8 (8 concurrent)

`bench_sweep.py --levels 8`: **412.0 tok/s** aggregate, 52.7 tok/s
per-stream.

Wall per decode step: rank0 **87.11 ms**, rank1 **87.10 ms** (~1.6x the c1
per-step wall, for 8x the batch -- the step does much more work per call,
consistent with the GEMMs getting M-dimension-bound rather than
launch-bound). GPU idle fraction: rank0 **0.056**, rank1 **0.055** --
almost identical to c1; this workload is compute-bound at both
concurrencies, not host-launch-bound.

Top 25 kernels, rank0:

| kernel | count | total ms | mean us | % CUDA | bucket |
|---|---:|---:|---:|---:|---|
| moe_sharedkernelssiluMoEDynamicKer... | 4656 | 2914.070 | 625.87 | 33.88% | gemm |
| b12xsequencegdn_decode_cute_kernels_Pa... | 3492 | 1125.513 | 322.31 | 13.08% | gdn/ssm |
| b12x_libdense_gemmDenseGemmKernel (a) | 13780 | 980.009 | 71.12 | 11.39% | gemm |
| nvjet_sm121_tst_mma_64x32x64_8_... splitK | 9095 | 450.155 | 49.49 | 5.23% | other |
| b12xcommroce_oneshot_cute_RoceOneshotL... | 10670 | 435.265 | 40.79 | 5.06% | all-reduce/nccl |
| cutlass::Kernel2\<wmma_tensorop_bf16...\> (a) | 9189 | 430.932 | 46.90 | 5.01% | gemm |
| cutlass::Kernel2\<tensorop_bf16_s16816...\> | 396 | 315.809 | 797.50 | 3.67% | gemm |
| b12x_libdense_gemmDenseGemmKernel (b) | 4320 | 208.794 | 48.33 | 2.43% | gemm |
| b12x_libdense_gemmDenseGemmKernel (c) | 4464 | 164.243 | 36.79 | 1.91% | gemm |
| b12xattentionpaged_selected_forward_Se... | 1552 | 141.556 | 91.21 | 1.65% | attention |
| cutlass::Kernel2\<wmma_tensorop_bf16...\> (b) | 3119 | 136.012 | 43.61 | 1.58% | gemm |
| b12xmoe_sharedkernelsw4a16kernelW4A16F... | 388 | 118.047 | 304.24 | 1.37% | gemm |
| nvjet_sm121_tst_mma_32x32x64_12_... | 4463 | 89.722 | 20.10 | 1.04% | other |
| b12xcommroce_allgather_cute_RoceAllGat... | 485 | 81.161 | 167.34 | 0.94% | all-reduce/nccl |
| b12xnormhyperconnection_cute_PackedCom... | 9991 | 79.167 | 7.92 | 0.92% | gemm |
| _emit_stable_topk_kernel | 1261 | 66.471 | 52.71 | 0.77% | sampler/argmax |
| b12xattentionqsa_score_cute_Representa... | 1261 | 63.199 | 50.12 | 0.73% | attention |
| cutlass::Kernel2\<wmma_tensorop_bf16...\> (c) | 343 | 56.241 | 163.97 | 0.65% | gemm |
| _gate_mean_kernel | 10573 | 42.198 | 3.99 | 0.49% | other |
| b12xsequencemtp_feedback_cute_prefillM... (a) | 388 | 41.245 | 106.30 | 0.48% | gemm |
| b12xsequencemtp_feedback_cute_prefillM... (b) | 388 | 34.514 | 88.95 | 0.40% | gemm |
| gemvx::kernel | 4804 | 33.529 | 6.98 | 0.39% | gemm |
| _sorted_softmax_topk_kernel | 4897 | 30.665 | 6.26 | 0.36% | sampler/argmax |
| at::native::vectorized_elementwise_kernel | 28449 | 28.715 | 1.01 | 0.33% | other |
| b12x_libdense_gemmDenseGemmKernel (d) | 444 | 27.559 | 62.07 | 0.32% | gemm |

Bucket shares, rank0:

| bucket | ms | % |
|---|---:|---:|
| gemm | 5646.6 | 65.6% |
| gdn/ssm | 1163.5 | 13.5% |
| other | 873.3 | 10.2% |
| all-reduce/nccl | 517.8 | 6.0% |
| attention | 235.2 | 2.7% |
| sampler/argmax | 134.5 | 1.6% |
| memcpy/other | 29.4 | 0.3% |
| mtp/draft | 1.9 | 0.02% |

rank1: gemm 65.7%, gdn/ssm 13.6%, other 10.2%, all-reduce/nccl 5.9%,
attention 2.8%, sampler 1.6%, memcpy 0.3%, mtp 0.02% -- same shape.

## c8 spec-decode metrics (`/metrics` deltas over the sweep)

- Accepted/draft tokens (cumulative counters, boot-lifetime, dominated by
  the c8 sweep): `spec_decode_num_accepted_tokens_total` 8265 /
  `spec_decode_num_draft_tokens_total` 8360 = **98.9%** overall acceptance.
- Accepted tokens per draft position (num_speculative_tokens=4): pos0
  2087/2090 (99.86%), pos1 2071/2090 (99.09%), pos2 2060/2090 (98.56%),
  pos3 2047/2090 (97.94%) -- gentle, expected falloff with draft depth,
  still very high at position 3.
- `vllm:num_preemptions_total`: **0** over the sweep.
- `vllm:request_time_per_output_token_seconds`: sum 2.2045 s / count 40
  requests = **55.1 ms mean per-request ITT** (this average spans the
  warm-up's lower-concurrency requests too, so it reads higher than the
  52.7 tok/s ≈ 19 ms/token measured for the steady c8 rounds alone).

## Decode step: wall vs CUDA busy vs idle (summary)

| label | wall/step | CUDA busy (union) | idle fraction |
|---|---:|---:|---:|
| c1 (rank0) | 55.03 ms | 51.97 ms | 5.4% |
| c1 (rank1) | 56.47 ms | 52.17 ms | 7.6% |
| c8 (rank0) | 87.11 ms | 82.22 ms | 5.6% |
| c8 (rank1) | 87.10 ms | 82.34 ms | 5.5% |

## Top 5 optimization targets, ranked

1. **MoE `siluMoEDynamicKer` GEMM -- 27-34% of step.** Largest single kernel
   at both concurrencies and the one whose share *grows* with concurrency
   (26.7% at c1 -> 33.9% at c8), meaning it's the primary scaling
   bottleneck, not a fixed per-step tax. Lever: retile for larger M (batched
   experts) at c8, or fuse the SiLU activation into the GEMM epilogue if
   not already (name suggests it may already fuse SiLU into the dynamic
   kernel -- worth confirming against b12x's kernel source before assuming
   headroom).
2. **GDN/SSM decode kernel -- 13.5% of step at c8, only 1.8% at c1.** Its
   *absolute* time is similar at both concurrencies (100 ms @c1, 1126 ms
   c8 total across ~35x more steps... actually per-step it's roughly flat
   count-wise), but as a *share* it's much larger at c8 because the MoE
   GEMM hasn't scaled as fast as the SSM path per batch -- worth checking
   whether the b12x GDN decode kernel batches multiple sequences into one
   launch at c8 or is doing 8 separate per-sequence launches (3492-3528
   launches in both windows suggests the latter: a batching opportunity).
3. **ROCE one-shot/allgather all-reduce -- 4-6% of step, ~10,000-10,700
   kernel launches per 100-step window (~100+ launches/step).** That launch
   count is high for a communication op; if the two-node all-reduce is
   being invoked multiple times per layer instead of fused across layers
   (`fuse_gemm_comms`/`fuse_allreduce_rms` are both `false` in this recipe's
   compilation_config -- see the vllm engine config log), enabling
   `fuse_allreduce_rms` is a direct, already-implemented lever worth an A/B
   arm.
4. **`cudaGraphLaunch`/`cudaEventSynchronize` host overhead (measured
   separately from GPU busy time, ~7.2s / ~5.5s wall in the *uncorrected*
   c1 pass before excluding CPU-side categories) -- not GPU work, but real
   host-side latency per step.** With CUDA graphs already enabled
   (`cudagraph_mode: FULL_AND_PIECEWISE`), the number of distinct graph
   launches per step (344 in the c1 window, i.e. ~3.4 launches/step) is
   the thing to reduce -- fewer, larger captured graphs per decode step
   would cut host dispatch overhead, which shows up as the ~5-8% idle
   fraction measured above.
5. **Sampler/argmax (`_emit_stable_topk_kernel` + `_sorted_softmax_topk_kernel`)
   -- 1.5-1.6% of step, but 1261-4897 launches per window.** Small in
   aggregate GPU time but a lot of tiny launches; if these run once per
   draft position per request rather than batched across the whole
   scheduled batch, batching them would reduce launch count without
   changing total FLOPs -- a cheap win if not already batched.

## Files

- `mods/vllm-decode-profiler/run.sh`, `mods/vllm-decode-profiler/profiler_snippet.py`
- `recipes/eugr/eugr-agents-serve-local16-la-lprof.yaml`
- `scripts/arm_trigger.sh` (trigger-dir path, kept for a future retry; not
  what produced the numbers above)
- `scripts/prof_summary.py`
- `results/sweep_c1_lprof.json`, `results/sweep_c8_lprof.json`,
  `results/arms/la-final/sweep.json` (post-profiling recovery check)

Raw chrome traces (~500 MB per rank per label) were analyzed in place on
each host (`~/.cache/sparkrun/runtime-cache/vllm/<model>/prof/<label>/rank<N>.json`)
and are not checked into the repo.

## Final state

- Serving recipe: `recipes/eugr/eugr-agents-serve-local16-la.yaml`, health
  200, c1 = 98.8 tok/s (>=95 gate passed) after the profiling passes above.
