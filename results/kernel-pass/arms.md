# Arms — 2026-09-19

> **Workload.** Unless a line says otherwise, every tok/s, c1/c8/c16 and
> ms/step figure in this file is the `tools/tony-bench/bench_sweep.py`
> counting diagnostic: "List the numbers from 1 to 300 separated by commas…",
> temperature 0, thinking off, non-streaming, 320 max tokens, fresh context,
> aggregate tok/s (c1 = per-stream). MTP accepts ~4 of 4 drafts on it, so it is
> a speculative-decoding ceiling, not coding or chat speed. Agent-coding and
> prose numbers: top-level `README.md`.

Baseline: `spark-vllm-b12x:local-20260918-a8333658`, `eugr-agents-serve-local16-la.yaml`,
c1 98.8 tok/s, c8 412.0 tok/s agg. Gate bar: c1 >= 101.8 (+3%) or c8 >= 432.6 (+5%),
nothing else worse than -2%.

| label | change | c1 tok/s | c1 delta | c8 tok/s (agg) | c8 delta | verdict |
|---|---|---:|---:|---:|---:|---|
| fusear | `compilation-config.pass_config.fuse_allreduce_rms: true` | 86.7 | -12.2% | 428.2 | +3.9% | **reject** — c1 regression blows the -2% floor despite the c8 gain |
| spec3 | `num_speculative_tokens: 3` (was 4) | 84.1 | -14.9% | 375.2 | -8.9% | **reject** — both concurrencies regress |
| b12x0f3a8cb | rebuild at b12x HEAD (0f3a8cb) | -- | -- | -- | -- | **HANG #1** -- see below |
| b12x0f3a8cb-noat | same image, `B12X_AUTOTUNE: "0"` (diagnostic only, not a promotion candidate) | 81.1 | -17.9% | 414.8 | +0.7% | boots and serves; confirms the image/model itself works, regression is expected (untuned MoE kernel selection) |
| b12x0f3a8cb-rev06809d5 | same image + mod reverting b12x commit 06809d5's Python changes, `B12X_AUTOTUNE: "1"`, empty plan cache | -- | -- | -- | -- | **HANG #2** -- see below (progressed further, then froze at a second barrier) |
| fwd57f3572 | OLD known-good image (a8333658) + mod forward-porting only b12x commit 57f3572 (MoE-only, no coordination-path touch), `B12X_AUTOTUNE: "1"`, empty plan cache | -- | -- | -- | -- | **HANG #3** -- see below (same disease reproduces on the old image too) |
| **la (final, restored)** | baseline recipe, old image, original 168-169 MB plan cache restored from backup after the three failed attempts | **94.9** | -3.9% (within normal 15-25% boot variance, effectively at parity) | not re-run (not required after restore) | | **RESTORED TO SERVING** -- health 200, plan cache confirmed warm (17 "cached" lines in boot log), this is the final state |

Recipe files added (not committed): `recipes/eugr/eugr-agents-serve-local16-la-fusear.yaml`,
`recipes/eugr/eugr-agents-serve-local16-la-spec3.yaml`.

## Correction to survey.md

The survey's tentative "GDN decode kernel batching opportunity" lead (item 2 in
"Top 5 optimization targets") is **refuted** by source inspection, not confirmed.
Traced the call path: `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
(`_bind_b12x_gdn_decode`) builds one `Binding` per model decode step covering the
*whole* batch (`state_indices`, `query_start_loc`, `num_seqs`, `num_tokens` are
batch-wide tensors), and `b12x/sequence/gdn_decode/_impl.py:run()` issues exactly
one `torch.ops.b12x.gdn_decode` custom-op call for that whole batch
(`b12x/sequence/gdn_decode/_kernels.py:_gdn_decode_op`). GDN decode is already
batched into one call per step, not one launch per request. This also explains
the profiling data directly: the ~3492-3528 launch counts were measured over two
*equal-size 100-step windows* (c1: steps 220-320, c8: steps 260-360), so a flat
launch count across c1/c8 is exactly what one-call-per-step predicts regardless
of concurrency -- it is not evidence of a missed per-request-batching win.
GDN's growing time-*share* at c8 (1.8% -> 13.5%) is because its per-call cost
scales with total routed/token volume while the MoE GEMM scales more
sub-linearly-efficient at higher M, not an unbatched-launch problem. This closes
off that Part-3 lead; no GDN kernel edit is warranted from this finding.

## b12x0f3a8cb hang — root cause and fix attempt

The b12x-HEAD rebuild boots but the MoE retune (forced by 57f3572's candidate-
contract bump, 222 -> 837 candidates) hangs deterministically: `nvidia-smi`
0% util / ~9.6W on both GPUs, rank0 `Worker_TP0` on-CPU spinning (wchan=0,
real CPU cycles confirmed via /proc utime/stime sampling) while rank1
`Worker_TP1` legitimately blocks in the kernel (wchan=`wait_woken`). Full
evidence in `results/kernel-pass/b12x0f3a8cb-hang/SUMMARY.md`. Reproduced
identically from a completely empty plan cache (moved the old 169 MB cache
aside on both nodes, backed up as `b12x.old-a8333658` and
`b12x.hang-emptycache-retry2` under
`~/.cache/sparkrun/runtime-cache/vllm/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-ca6f25af/`
on both nodes) -- rules out a stale/corrupt cache.

**Root cause (source-diff confirmed, not just circumstantial):**
`git show --stat 06809d5` (b12x, "Reconcile cached tuning selections before
distributed preparation") touches only Python files
(`b12x/preparation/{__init__.py,_cache.py,session.py,types.py}`, plus
docs/tests -- no `.cu`/`.so`). The diff adds a brand-new
`TuningCacheRequirement`/`ready_cache` handshake: `PreparationJob._run()`
now yields this requirement *before any tuning begins, whenever
`len(session._tuning_ranks) > 1`* (i.e. unconditionally for our TP=2 boot),
and requires the **caller's** coordinator (vLLM's startup code, not b12x)
to recognize `state.ready_cache` and answer with a `cache=` kwarg carrying
every rank's selection snapshot. Our pinned vllm fork tip (8e1f1e58,
dev/jovian-judgement, confirmed 0 commits ahead of origin all session) predates
this b12x feature and has no code path that answers `ready_cache` --
so `advance()` spins forever re-polling an unresolved state: rank0's spin
matches its on-CPU/no-GPU-work signature exactly, rank1's block matches a
rank correctly waiting on an exchange the other side never completes. This
is a **b12x/vllm-fork version-skew bug** introduced by 06809d5, not an
environment, memory, or cache issue.

`B12X_AUTOTUNE=0` (`session.py:249`) sets the session's stop event at
construction, short-circuiting before the new yield ever fires -- confirmed
this boots and serves (see `b12x0f3a8cb-noat` row above), isolating the
hang to the autotune/tuning-cache-reconciliation path specifically.

**Fix attempted:** verified byte-for-byte (`sha256sum`) that the four
touched files inside the built image
(`/usr/local/lib/python3.12/dist-packages/b12x/preparation/*.py`) match our
local `~/GEN-AI/build/b12x` checkout exactly, then reverse-applied
`git diff 06809d5^ 06809d5 -- <those 4 files>` as a sparkrun mod
(`mods/b12x-revert-06809d5/`) and re-booted with `B12X_AUTOTUNE: "1"` and
an empty plan cache, label `b12x0f3a8cb-rev06809d5`.

**Outcome: hang #2, at a different barrier.** The revert genuinely fixed
the first barrier -- preparation advanced with real, climbing counters
(3299 measured, 194 compilations, vs. permanently-zero before) instead of
freezing immediately. It then froze again at a second, structurally
similar barrier: `b12x compiling sequence.mtp_feedback: 47/349 ready,
candidates 0/8 prepared, rank 0 batch 1, ...`. `nvidia-smi` 0% util on both
GPUs again; this time both ranks' `Worker_TP*` processes showed the same
wchan (`poll_schedule_timeout.constprop.0`, symmetric mutual block) rather
than hang #1's asymmetric spin/wait. Evidence in
`results/kernel-pass/b12x0f3a8cb-hang2/`. One commit revert was not
sufficient -- unbounded scope to keep reverting individual
preparation-protocol commits, so the strategy pivoted (see below).

## fwd57f3572 hang — same disease reproduces on the OLD image

Rather than keep patching b12x HEAD's protocol, forward-ported only the
narrowly-scoped MoE fix (57f3572) onto the KNOWN-GOOD old image
(`spark-vllm-b12x:local-20260918-a8333658`, whose preparation protocol
already works fine with our pinned vllm fork -- confirmed serving `la`,
`spec3`, `fusear` all session). `57f3572^` is exactly `a8333658` (clean
linear parent, verified with `git rev-parse`), its diff touches only
`b12x/moe/_shared/kernels/w4a16/kernel.py` and
`b12x/moe/fused_moe/{_impl.py,_preparation.py,_tuning.py}` (a *different*
`_preparation.py` module from the one that hung -- MoE candidate-selection
logic, not the distributed rank-coordination protocol), and every touched
file's pre-image hash was verified byte-identical to `git cat-file -p
57f3572^:<file>` before patching (mod: `mods/b12x-fwd-57f3572/`, tested
standalone against the image before wiring into a recipe).

**Outcome: hang #3, at the same `rank N batch M` barrier**, this time on
the old image: `b12x priming gemm.bf16_vocab_projection: 45/349 ready, ...
rank 0 batch 1, ..., 1:00` then silence, `nvidia-smi` 0% on both GPUs
(confirmed independently twice), both ranks' wchan =
`poll_schedule_timeout.constprop.0`. Evidence in
`results/kernel-pass/fwd57f3572-hang3/`.

## Revised root-cause conclusion (final)

Since the deadlock reproduces on the OLD, otherwise-working a8333658 image
too, once its own existing `rank N batch M` multi-batch candidate-racing
code path (in `b12x/preparation/session.py`) is exercised, this is **not**
specifically a "b12x HEAD vs. stale vllm fork" version-skew bug. It is a
**scale-dependent deadlock in b12x's batched candidate-racing handshake**
that only activates once a retune has enough candidates to need more than
one batch round. a8333658's normal boots produce ~222 candidates and never
hit this path (no "batch" label ever appears in a normal boot log). Every
commit we tried that meaningfully grows the moe.decode candidate contract
(57f3572's 4->5 bump alone, or the full b12x-HEAD stack's chain of bumps
to 6) pushes the candidate count past whatever threshold triggers the
multi-batch path, and it deadlocks every time, regardless of which
specific commit or image carries the bump.

**Verdict: none of the MoE-retune-forcing commits (57f3572 and later) are
usable for a TP=2 boot against `local-inference-lab/vllm@dev/jovian-judgement`
@ 8e1f1e58 as currently pinned.** This needs either (a) that vLLM fork
branch to ship coordinator-side support for b12x's multi-batch tuning-cache
protocol, or (b) a b12x-side fix to the `rank N batch M` handshake itself
(the actual bug is almost certainly there, not in anything reachable from
a fork-side mod). Re-check when `local-inference-lab/vllm@dev/jovian-judgement`
moves past 8e1f1e58, or when b12x ships a fix to this handshake -- neither
is available to act on today without opening an upstream issue, which is
out of scope. No further kernel-pass MoE-retune work is possible this
session; reverted to the known-good baseline.

## Final state

Plan cache restored on both nodes (moved the fwd57f3572 attempt's cache
aside as `b12x.from-fwd57f3572`, restored `b12x.old-a8333658` -> `b12x`,
confirmed 168-169 MB matching the original untouched size). Booted
`recipes/eugr/eugr-agents-serve-local16-la.yaml` (the original baseline,
image `spark-vllm-b12x:local-20260918-a8333658`) -- health 200, plan cache
confirmed warm (17 "cached" lines in the boot log, no retune), **c1 = 94.9
tok/s** (`results/kernel-pass/sweep_la_final_restore.json`) -- within
normal 15-25% boot-to-boot variance of the original 98.8 tok/s baseline
measured at session start, effectively at parity. This is what is serving
at the end of this session. Part 3 (kernel edits) was never reached: no
gated arm cleared the promotion bar, and the MoE-retune path that would
have supplied a concrete kernel-level target is blocked by the hang above.

## MoE occupancy knob (Part 4, if reached)

`B12X_MICRO_MAX_ACTIVE_CLUSTERS` / `B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS` /
`B12X_MICRO_DYNAMIC_CUTOVER_PAIRS` are plain env-var overrides read directly by
the b12x MoE decode kernel dispatch -- no need to run
`scripts/sweep_moe_decode_max_active_clusters.py` (that script is a standalone
synthetic microbenchmark hardcoded to `TP_SIZE=4`, i.e. built for the DSV4 TP4
case, not our TP2 Qwen serving path). The lower-risk path is to set these two
env vars directly in a copy of the new image's recipe and screen c1/c8 live.
