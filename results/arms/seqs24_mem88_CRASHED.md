## max_num_seqs=24, gpu_memory_utilization=0.88: CRASHED (2026-09-20)

Boot reached KV cache sizing successfully -- `Available KV cache memory:
38.63 GiB`, `GPU KV cache size: 4,648,156 tokens, Maximum concurrency for
262,144 tokens per request: 17.73x` (up from 3.60M tokens / 13.74x at the
la baseline's 0.80/16, as expected from more GPU headroom) -- then crashed
9 seconds into the post-KV b12x "state" preparation stage, retuning
`sequence.gdn_prefill` for the new max_num_seqs=24 batch shapes (this
shape was never cached for seqs=24, only for the seqs=16 la baseline, so a
retune was forced, same mechanism as every other cache-miss retune seen
this session):

```
b12x compiling sequence.gdn_prefill: 1/411 ready, candidates 0/48 prepared, rank 0 batch 1, 48 measured, 0 cached, 54 compilations, 0:09
b12x failed sequence.gdn_prefill: 1/411 ready, candidates 0/48 prepared, rank 0 batch 1, 48 measured, 0 cached, 54 compilations, 0:09
(EngineCore pid=225) ERROR 09-20 09:46:07 [core.py:1484] EngineCore failed to start.
...
RuntimeError: b12x preparation failed on rank 0: SystemExit:
```

**No CUDA OOM traceback is present anywhere in the 60 lines before the
failure** -- no `CUDA out of memory`, no `cudaErrorMemoryAllocation`. The
failure is a bare `SystemExit` surfacing through b12x's own preparation
subprocess after exactly 54 compilations in 9 seconds (i.e. mid-candidate-
measurement, not at the start). Combined with the coordinator's observation
that host MemAvailable was down to 8 GiB by the time this was
investigated (45 min after the crash, container still up but engine dead)
and this session's already-established pattern (the 16384 arm forced a
retune that visibly ate host RAM, 14->11 GiB, in the same
compile-many-candidates-in-parallel phase), **the most likely cause is a
host-side OOM kill of a b12x candidate-measurement subprocess** (each
candidate compiles a fresh CUDA kernel via nvcc/cutlass -- a host-RAM-heavy
step, and b12x measures multiple candidates in parallel per the "batch N"
labeling seen throughout this session) that surfaces to the caller as a bare
`SystemExit` rather than a normal Python exception, precisely because the
kernel killed the subprocess rather than the subprocess raising cleanly.
gpu_memory_utilization=0.88 (up from 0.80) leaves ~4 GiB less device-side
headroom but does not by itself explain a host-RAM OOM -- the real
squeeze is host RAM, worsened at 0.88 only insofar as a larger designed
concurrency (max_num_seqs=24) implies a larger KV cache allocation and
more retune candidates in flight, not because gpu_memory_utilization
itself consumes host RAM.

Secondary candidate not ruled out: an OOM specifically inside the
RoCE/collectives prep path co-running at this point (`b12x_roce_all_reduce`
lines appear ~30s earlier in the same boot, i.e. RoCE proxy state is
already live when the gdn_prefill retune starts) -- not distinguishable
from the host-RAM explanation without dmesg (unreadable, no sudo) or a
retry with a memory-usage sampler attached.

**Action per instruction: stop, retry once at gpu_memory_utilization=0.84,
then if that also fails the same way, at the default 0.80 with
max_num_seqs=24 alone (KV stays ~3.6M) to isolate whether more scheduled
seqs alone helps c16/c24. seqs=32 skipped entirely for this session.**
