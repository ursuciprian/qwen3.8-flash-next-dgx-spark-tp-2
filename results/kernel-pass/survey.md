# Kernel-pass survey — 2026-09-19

> **Workload.** Unless a line says otherwise, every tok/s, c1/c8/c16 and
> ms/step figure in this file is the `tools/tony-bench/bench_sweep.py`
> counting diagnostic: "List the numbers from 1 to 300 separated by commas…",
> temperature 0, thinking off, non-streaming, 320 max tokens, fresh context,
> aggregate tok/s (c1 = per-stream). MTP accepts ~4 of 4 drafts on it, so it is
> a speculative-decoding ceiling, not coding or chat speed. Agent-coding and
> prose numbers: top-level `README.md`.
>
> Third-party figures quoted from upstream commit notes keep their authors'
> workload.

Baseline: `spark-vllm-b12x:local-20260918-a8333658` (b12x@a8333658, vllm fork@8e1f1e58),
serving `recipes/eugr/eugr-agents-serve-local16-la.yaml`, c1 98.8 tok/s, health 200.
Profile verdict (results/profiling/README.md): c1 GEMM 83.6% (MoE `siluMoEDynamicKer`
26.7%->33.9% share growing with concurrency), c8 GDN/SSM share triples to 13.5%
(launch count flat ~3500 across c1/c8 -> one-launch-per-request suspected, not batched),
all-reduce/ROCE 4-6% at ~100+ launches/step (fuse_allreduce_rms/fuse_gemm_comms both off).

## 1. Upstream commit survey

**vllm fork (dev/jovian-judgement)**: 0 commits ahead of 8e1f1e58. Nothing to pull.

**b12x (master)**: 6 commits ahead of a8333658, newest first:
- `0f3a8cb` Preserve DSV4.1 sparse MLA head partitions for TP3 — MLA/TP3-only
  (attention/_shared/mla/prefill.py), not our model (GDN, TP2). Irrelevant.
- `d513903` (#394) perf(qsa): fuse exact stable winner construction — fuses
  threshold reduction/candidate-count/winner-emission into one CuTe kernel for
  QSA (sparse-attention token selection). Our profile shows attention at only
  1.6-2.7% of step, so low expected upside, but it's a free win if it lands
  with the other MoE-targeted commits.
- `8783519` Pass constexpr arguments to prepared W4A16 route pack launchers —
  bugfix for W4A16 MoE routing launchers (missing compile-time args). Not
  GPU-validated by its own author (OOM'd on GB10 during their test). Needed
  as a prerequisite for correctness once native NVFP4/W4A16 autotuning
  reactivates below.
- `06809d5` Reconcile cached tuning selections before distributed preparation
  — fixes a race where TP ranks could each pick a different cached kernel
  choice during autotune; makes a 2-node retune (which we need for the next
  commit) safe/deterministic instead of racy.
- `a2b5152` Tune compact-tail MoE with shared verifier workloads — sweeps CTA
  limits for **W4A8** compact/grouped MoE kernels at verifier sizes 4/6/8
  (matches our MTP verify width). Author's own DSV4.1 numbers: TP4 decode
  43.15->49.20 tok/s (128 tok) and 37.51->50.26 tok/s (4096 tok) — but their
  note flags KV capacity also changed 10->6 GiB/rank in that run, so the
  measurement doesn't isolate the tuning win. Bumps the moe.decode candidate
  contract 4->5->6 (each contract bump invalidates old cached plans).
- **`57f3572` fix(moe): restore native NVFP4 A16 autotuning — the direct hit.**
  "Admit supported source-native W4A16 direct routes to MoE precision tuning
  and select them by default **at capacities 1-8 on SM120/SM121**." That's
  exactly our decode shapes (c1 uses M~5 with 4 MTP drafts+1, c8 uses M~40 in
  8 parallel streams each still verifying small batches) on exactly our GPU
  (GB10/SM121). It also bumps the moe.decode candidate contract, which
  invalidates whatever's in our current 168 MB plan cache — meaning **our
  current image is running on stale/superseded MoE tuning by design** once
  this lands upstream.

No b12x issues/PRs beyond the fetched commits add signal (checked open
issues #367-#394 by title grep for moe/gemm/nvfp4/gdn/spark/sm121 — all are
either already-merged PRs reflected above or DSV4/TP4/TP3-specific).

**Verdict: build a new image at b12x HEAD (0f3a8cb).** The MoE-autotune chain
(57f3572 -> a2b5152 -> 06809d5 -> 8783519) is a coherent, self-referential
fix/tune/prereq sequence squarely aimed at our GEMM bottleneck and our exact
SM121 + decode-capacity range; taking HEAD gets the whole chain plus the free
QSA/MLA commits without cherry-pick risk.

## 2. Env knobs (b12x, non-test files)

- `B12X_AUTOTUNE`, `B12X_AUTOTUNE_EXHAUSTIVE` — gate whether autotune runs at
  all / exhaustively. Recipe already sets `B12X_AUTOTUNE=1` (default search,
  not exhaustive).
- `B12X_POLICY_MODE` — recipe uses `auto`. No other documented values found
  in non-test source in the time budget; `auto` already picks per-shape
  policy, so an explicit override arm has no clear alternative to try without
  reading `_lib/compiler.py`'s policy table (out of scope for this pass).
- `B12X_DENSE_SPLITK_TURBO` (default `"1"`) — split-K atomic path for dense
  GEMM; already on by default.
- `B12X_MICRO_MAX_ACTIVE_CLUSTERS` / `B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS` /
  `B12X_MICRO_DYNAMIC_CUTOVER_PAIRS` — occupancy knobs for the MoE
  micro/dynamic kernel paths, swept by `scripts/sweep_moe_decode_max_active_clusters.py`.
  This is the sanctioned way to touch MoE decode occupancy without a kernel
  edit — candidate for Part 3 if the new image still leaves headroom.
- MHC (hyperconnection) norm path has its own tile/stage knobs
  (`B12X_MHC_*`) — prefill-only tuning, not on our decode-bound path.
- `B12X_NVFP4_DYNAMIC_MATERIALIZED` (0=monolithic/1=split) — NVFP4 kernel
  variant switch mentioned in `gen_nvfp4_split_readme.py`; worth a follow-up
  arm if the rebuild alone doesn't hit the gate, not run this pass.
- Did not find a GDN-decode-specific env knob in non-test grep output in the
  time budget.
- **UPDATE (Part 3 prep, done): the GDN batching question is resolved, not
  deferred — the profiling README's suspicion was wrong.** Traced the call
  path: `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
  (`_bind_b12x_gdn_decode`) builds one `Binding` per decode step covering the
  *whole* batch (state_indices/query_start_loc/num_seqs/num_tokens are
  batch-wide tensors), and `b12x/sequence/gdn_decode/_impl.py:run()` issues
  exactly one `torch.ops.b12x.gdn_decode` call for that whole batch. GDN
  decode is already batched into one call per step, not one launch per
  request. The flat ~3492-3528 launch count across c1/c8 in the profile is
  explained by both windows being equal-size (100 steps each), not by
  unbatched per-request launches. See results/kernel-pass/arms.md
  ("Correction to survey.md") for the full note. No GDN kernel edit is
  warranted from this finding.

## 3. Plan cache

`~/.cache/sparkrun/runtime-cache/vllm/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-ca6f25af/b12x/`
is 168 MB: `compile/` has ~259 hash-named subdirectories (`00`-`ff`-style
prefixes), `loader/` has 2 hash-named entries. Cache keys are content
hashes, not shape-labeled, so confirming whether M=5/M=40 decode shapes are
tuned vs. falling to a default requires decoding the hash scheme or adding
a log line — not done in this pass (time-boxed). Practical proxy: the
57f3572 commit bumps the moe.decode candidate contract specifically to force
this cache to be treated as stale, so the planned Part 2(e) rebuild forces a
correct re-tune regardless of what's currently cached.

## 4. Fusion knobs (vllm fork)

- `fuse_allreduce_rms` (compilation-config pass_config): defined in
  `vllm/config/compilation.py`, defaulted per-model in `vllm/config/vllm.py`
  (`enable_allreduce_rms_fusion` computed flag, several call sites default it
  `False` for the paths that don't set the computed flag). The current `la`
  recipe's `--compilation-config` only sets `fuse_act_quant:true` and leaves
  `fuse_allreduce_rms`/`fuse_gemm_comms` unset -> both resolve to their
  model-family default, which the profile's own README already asserts is
  `false` here. Confirmed present, off, and directly implicated by the
  profile's 4-6%-of-step, ~100+-launches/step ROCE allreduce finding.
- `fuse_gemm_comms` — same pass_config, defaults `IS_DENSE`-gated (i.e. only
  for dense, non-MoE models per `vllm/config/vllm.py:321,344`); Qwen3.8-Flash-Next
  is MoE, so this one won't engage — not a viable arm here.
- `VLLM_USE_MEGA_AOT_ARTIFACT` — already `"1"` in the recipe; requires
  `VLLM_USE_STANDALONE_COMPILE=1` (asserted in `backends.py:97`) which isn't
  set explicitly in the recipe but must be resolving true already since the
  server boots (else the assert would fire at startup).
