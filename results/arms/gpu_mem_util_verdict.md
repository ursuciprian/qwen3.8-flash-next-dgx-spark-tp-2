## gpu_memory_utilization > 0.80 not viable on Spark (unified memory)

Two boots this session pushed `gpu_memory_utilization` above la's default
0.80, both on top of `max_num_seqs=24`:

- **0.88**: crashed 9s into post-KV b12x preparation (`sequence.gdn_prefill`
  retune) with `RuntimeError: b12x preparation failed on rank 0:
  SystemExit:` -- no CUDA OOM traceback, consistent with a host-RAM OOM
  kill of a candidate-measurement subprocess. See
  `results/arms/seqs24_mem88_CRASHED.md` for the full analysis.
- **0.84**: aborted live on operator instruction when dgx-01 MemAvailable
  hit 2 GiB during the boot -- earlyoom's own `-m 2` threshold (~2.4 GiB)
  was about to fire. Stopped before a kill or crash happened.

**Root cause is structural, not a per-arm tuning mistake**: the DGX Spark's
121 GiB is a single unified pool shared between host and GPU (see the
dgx-spark-serving skill's opening line). `--gpu-memory-utilization` only
governs the *device-side* allocator's share of that pool for weights/KV/
activations; it does not reserve anything for the host-side vLLM/b12x
processes, RoCE proxy buffers, and (worst of all) b12x's own preparation
subprocess, which forks parallel candidate-measurement/compile workers
that are host-RAM-heavy (each compiles a fresh CUDA kernel via
nvcc/cutlass). Those host-side needs are roughly 15-20 GiB on this
workload. At 0.80 there is enough slack for both; above 0.80 the device
share grows at the host's expense, and a b12x retune (any retune -- new
`max_num_seqs`, new `max_num_batched_tokens`, anything that misses the
tuned-shape cache) pushes the host into or past the danger band before the
device-side gain is ever realized.

**Verdict: gpu_memory_utilization > 0.80 is not viable on this box for this
model/recipe family, regardless of max_num_seqs.** Any future concurrency
lever must stay at 0.80 and get its gain from scheduling
(`max_num_seqs`) alone, or from KV-dtype/quantization changes that shrink
the KV footprint without touching the utilization knob.

Last concurrency arm to try per instruction: `max_num_seqs=24` at the
default `gpu_memory_utilization=0.80` (KV cache stays ~3.6M tokens, same
budget as la; this only tests whether scheduling more concurrent
sequences into the same KV budget helps c16+ via better
preemption/packing, not whether more KV headroom helps).
