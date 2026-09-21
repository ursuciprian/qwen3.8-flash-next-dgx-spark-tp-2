## Lever C: concurrency ceiling -- ABANDONED, all three arms failed

- **max_num_seqs=24, gpu_memory_utilization=0.88**: crashed 9s into b12x
  state-preparation retune (`RuntimeError: ... SystemExit`), no CUDA OOM
  traceback -- consistent with a host-RAM OOM kill of a candidate-measurement
  subprocess. `results/arms/seqs24_mem88_CRASHED.md`.
- **max_num_seqs=24, gpu_memory_utilization=0.84**: aborted live when
  dgx-01 MemAvailable hit 2 GiB (earlyoom's own `-m 2` threshold), before
  a kill or crash occurred. `results/arms/gpu_mem_util_verdict.md` records
  the structural conclusion: **gpu_memory_utilization > 0.80 is not viable
  on the Spark's unified-memory architecture** -- the device-side share
  grows at the host's expense, and host RAM (15-20 GiB needed for
  vLLM/RoCE/b12x-prep processes) runs out before any KV-budget gain is
  realized, regardless of max_num_seqs.
- **max_num_seqs=24, gpu_memory_utilization=0.80 (default)**: booted, KV
  budget unchanged (~3.6M tokens, matching la), but max_num_seqs=24 is
  itself an uncached b12x shape -- forced a full retune of
  `sequence.gdn_prefill` that hit the exact same deadlock as lever A's
  exhaustive-autotune arm: ticker kept printing its elapsed-time counter
  but every other field (`1475 measured, 0 cached, 1886 compilations,
  candidates 4/128, rank 0 batch 3`) froze solid from 5:37 through 8:08+
  while both GPUs sat at 0% util. Evidence:
  `results/kernel-pass/seqs24-hang/evidence.txt`,
  `serve-seqs24-mem80-hang.log`.

**Conclusion for the record**: any configuration change that alters
b12x's candidate shapes -- exhaustive autotune (lever A), a new
`max_num_batched_tokens` value forcing an uncached shape at scale (16384),
or a new `max_num_seqs` (24) -- forces a multi-batch retune that
deterministically deadlocks at TP=2 on this b12x build
(`spark-vllm-b12x:local-20260918-a8333658`) once the retune needs more
than one "batch" round of candidate racing. Only single-batch retunes
(roughly <=222-349 candidates, matching every la/lprof/mnbt4096/mnbt8192
boot this session, all of which showed `272`/`16`/`24`/`36` cached lines
and no "batch" ticker) complete. This is the same root cause
`results/kernel-pass/arms.md` already established for the b12x-HEAD and
forward-ported-commit arms earlier this week -- confirmed here to also
apply to purely config-driven retunes on the unmodified, currently-serving
image, with no source changes at all.

**Verdict: no promotion. Lever C yields nothing usable this session.**
Any future concurrency work on this box needs either (a) a fix to b12x's
multi-batch candidate-racing handshake upstream, or (b) a way to pre-seed
the tuned-shape cache for a new `max_num_seqs`/`max_num_batched_tokens`
value out-of-band (e.g. a single-node, single-batch-sized retune session)
before ever booting it at TP=2 -- not attempted this session, out of
scope for a same-day config sweep.
