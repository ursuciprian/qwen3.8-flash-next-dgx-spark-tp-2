# vllm-gdn-deferred

The vLLM half of deferred GDN checkpoints. Not bootable on the current image:
it needs a b12x wheel that does not exist yet (see **Requires** below).

GDN decode is the last DRAM-bandwidth lever on this model. `results/` and the
b12x-side review put the kernel at 79-86% of the GB10's 273 GB/s at c8-c16,
and head grouping — the obvious geometry fix — measured as a flat negative
result. What is left is bytes: at MTP 4 the kernel reads one recurrent-state
snapshot and writes **five**, one per verified token, because acceptance is
not known until after the sampler. Four of the five are thrown away.

Deferred checkpoints keep one base checkpoint in the running block plus a
compact per-token record (the normalized key, a per-row delta, and the decay)
in each speculative block, and replay the accepted prefix onto the base as
part of the state read the next decode step already performs. Six snapshots
become two: **2.66x less state traffic**, modelled at 13.5% -> 5.9-7.2% of the
c8 step. The records live inside the speculative state blocks the scheme stops
using, so nothing new is allocated.

## Requires

- **b12x** branch `feat/gdn-deferred-checkpoints` (fork `ursuciprian/b12x`),
  which adds `Caps(deferred_checkpoints=...)` and
  `commit_deferred_checkpoints`. **No published wheel carries it.** This mod
  applies cleanly without it and the feature then refuses to turn on, because
  the installed b12x rejects the caps keyword.
- `--mamba-cache-mode align` (already mandatory for Qwen3.8-Flash-Next).
- `num_speculative_tokens >= 1`.
- Request-boundary checkpoints **off**.

Until a matching wheel exists, adding this mod to a recipe is safe but
pointless: leave `VLLM_GDN_DEFERRED_CHECKPOINTS` unset and nothing changes.

## What it edits

Applied against the image's vLLM, built from `8e1f1e587f`. Equivalent to
`ursuciprian/vllm` `feat/gdn-deferred-checkpoints` (`29bf8477f`); the readable
diff is `patches/vllm-gdn-deferred.patch`.

- `vllm/v1/worker/gdn_deferred_commit.py` — **new file**, copied in whole.
  Owns the refusal rules, the per-step commit buffers, and the small Triton
  kernel that shapes a request's state window the way b12x's commit expects.
- `vllm/envs.py` — `VLLM_GDN_DEFERRED_CHECKPOINTS`, default off.
- `vllm/v1/worker/mamba_utils.py` — splits `_copy_mamba_state_block`'s single
  `token_bias` into `conv_bias` and `temporal_bias`; adds `DECISION_ONLY` and
  `DEFERRED_TEMPORAL` to the two boundary-copy kernels; runs decide -> commit
  -> copy in both drivers; asserts the state-window uniqueness the scheme now
  depends on.
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` — resolves the
  flag, declares it in the decode caps, lets the commit supply its own metadata
  window, and adds the per-layer commit entry point.

## Why the two biases

The shared copy helper does the conv *and* temporal state from one bias, and
with deferred checkpoints they stop agreeing. There is no speculative column to
select any more, so the temporal half must copy an already-committed block with
bias 0, while the conv half still shifts its window by the accepted-token bias.
Splitting the parameter is what makes that expressible.

## Why boundary checkpoints are refused, not handled

`checkpoint_mamba_states_kernel` is a third reader of the speculative columns.
One capture can ask for up to three different biases for the same request
(prompt, response, instruction), and a single accepted-prefix commit produces
one state. Rather than export a record block as if it were a state — which
would silently poison the boundary checkpoint and the prefix-cache entry built
from it — the mod refuses the combination at startup and says so.

## Applying

`run.sh` checks the pre-image sha256 of all three target files (hard fail on
drift), copies the new module and verifies its sha, then `apply.py` does
exact-block replacement: every edit must match its pre-image verbatim, nothing
is written until all of them do, and every touched file is `ast.parse`d first.
The post-image sha256 is verified afterwards, then an import self-check asserts
the default is off, that each unsupported dimension is named rather than
silently downgraded, that asking for an unsupported configuration raises, and
that the bias split is really there. Idempotent: a post-image sha match exits
0, and a marker present without the matching sha is treated as a half-applied
tree and refused.

`apply.py` is generated from the vLLM diff and verified to reproduce the exact
post images; do not hand-edit its blocks.

## Judging it

Not bit-identical to the shipped path by construction, so
`scripts/logits_equiv.py` is not the gate. It *is* bit-identical to the
checkpoint b12x would have written, which the b12x-side pytest proves on GPU
(`tests/sequence/test_gdn_deferred_checkpoints.py`). This arm is judged on
`scripts/gate_arm.sh`.
