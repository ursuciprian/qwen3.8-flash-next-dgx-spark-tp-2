# r8-eager: the TP=2 eager tail (design, 2026-10-01, opus-kernel-11)

PRD r8 arm `r8-eager`: move the eager tail (main lm_head ~1.38 ms, memcpy/meta ~1.19 ms per
c1 step) into CUDA graphs, expected -1.5 to -2.5 ms. Verdict: **design only, no patch this
round.** The expected gain on b1.4 is 0.3-0.6 ms per c1 step (<= 1.5%), below what one
screening pass resolves, and the parts worth doing change graph-captured addresses.

## What the eager tail is

Source: `results/r5-p2d-wm2prof/prof-r5-wm2offprof/prof-fresh-c1/rank0.json.gz` (TP=2 c1,
43.9 ms/step, 60 steps), re-read with `scripts/step_breakdown.py` (r8 adds the idle split by
phase transition). That boot ran without `VLLM_GDN_UNIFORM_DECODE_META_SKIP` (b1.3's change,
env.txt), so it still has the ~350 per-step GDN mixed-metadata launches the skip removes.

One step, in GPU order: target graph 33.0 ms (2,158 kernels) -> eager: hidden-state gather,
**main lm_head 1.377 ms** (b12x MXFP8 dense GEMM), weight-only reduce 18 us, **RoCE logits
all-gather 293 us**, rejection sampling (~10 Triton kernels, ~35 us) -> 3 draft graphs with
small eager input prep between them -> next step's input and attention metadata, eager.

| idle by transition (us/step) | |
|---|---:|
| eager -> eager | 1,636 (539 gaps, ~3 us each) |
| target -> target | 692 (inside the graph) |
| draft -> draft | 248 |
| eager -> draft, eager -> target, target -> eager, draft -> eager | 63 together |

Eager kernels: 546 per step, 0.63 ms of kernel time. Of these, ~400 sit in 36 identical
blocks, one per GDN KV-cache group: `copy_worklists_from` (3x `_foreach_copy_`),
`refresh_state_indices` (3x `zero_`, `index`, `copy_`), `_fill_uniform_spec_metadata`, 2
fills. b1.3's metadata skip removes the worklist copies, the zero/refill of the mixed state
indices and the live-count fills on uniform decode (~350 launches), which leaves ~36
`_fill_uniform_spec_metadata` + ~36 spec-state-index copies + ~120 runner/sampler/draft-prep
kernels.

## Why "lm_head into the graph" buys little

Graphs remove CPU launch latency, not kernel time. The CPU runs a full step ahead (async
scheduling), so the lm_head, the all-gather and the sampler are already queued when the
target graph ends: the target -> eager and eager -> draft transitions cost 3 us and 28 us.
The 1.38 ms lm_head and the 0.29 ms all-gather stay. Capturing `compute_logits` + the
rejection sampler in the uniform-decode FULL graph (logits_indices is arange(num_tokens) on
uniform decode, so the shapes are static) would remove ~15 launches x ~3 us = ~0.05 ms.

## Where the remaining eager time can go (b1.4)

1. **One launch for all GDN groups' uniform-spec metadata.** Every group writes the same
   `spec_query_start_loc`, `spec_token_indx`, `spec_sequence_masks`, `num_accepted_tokens`;
   only `spec_state_indices` differs (rows of `mamba_aligned_state_indices`, already computed
   for all groups by one `get_aligned_state_indices_multi_group_kernel`). A multi-group
   `_fill_uniform_spec_metadata` taking a device pointer table of the 36 builders' buffers
   replaces ~72 launches with 1. Saving ~0.2-0.3 ms/step. Risk: low-medium (new kernel,
   capture addresses unchanged because each group keeps its own buffers).
2. **Share the uniform buffers across groups** instead of filling 36 copies (builders alias
   group 0's tensors at capture). Same saving as 1 with no kernel, but every FULL graph then
   reads group 0's storage: a capture-order change, so it needs a full gate. Prefer 1.
3. **The 0.29 ms logits all-gather.** vLLM's `batch_sharder` splits sampling by request, which
   does nothing at c1. Removing the gather needs vocab-parallel rejection sampling (each rank
   reduces its vocab half, ranks exchange per-row max/sum/sample candidates). A project, not
   an arm, and it touches the sampler's numerics.
4. lm_head + sampler into the graph: ~0.05 ms, last.

Expected total for 1 + 4: 0.3-0.6 ms at c1 (0.7-1.4% of the b1.4 step).

## Next step

Re-profile b1.4 first (`prof:r8-prof` stage of `scripts/r8_driver.sh`, recipe from
`scripts/r8_make_recipes.py tp2`, ~30 min) and read the eager -> eager row. Build item 1
only if the b1.4 eager block is still >= 0.8 ms/step; then screen it with paired temp-0
probes (step CI ~0.5%), since grid tok/s cannot resolve 1%.
