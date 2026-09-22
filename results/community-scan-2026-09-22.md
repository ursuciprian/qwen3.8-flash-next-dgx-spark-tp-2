# Community scan, 2026-09-22 (since 2026-09-01)

Scope: speedups for Qwen3.8-Flash-Next NVFP4 (GDN+MoE+MTP) on DGX Spark GB10 SM121, vLLM+b12x, TP=2/RoCE. Baseline for "applies to our image": vllm `8e1f1e58` (jovian) / KK `57a80980`→now `622912b`/e9ce5477→now, b12x `a8333658`→now `b294e69`.

## Findings, ranked (gain × ease)

1. **Upstream vLLM #55260 — GDN decode-path opts (packed QKV, fused index prep, weight de-interleave + tiny-GEMM fold), Model Runner V2, 09-04.** Attacks exactly our GDN-decode-is-DRAM-bound-on-per-token-state-writes finding: fuses the per-step host-side checkpoint index prep into one Triton launch, folds `in_proj_ba` into the main GEMM (removes a split-K fp32 round-trip), de-interleaves QKVZ for zero-copy views. Not in our fork (predates jovian tip but Model Runner V2-gated; our fork/KK may not have it — needs a SHA check). **Applies to our stack conceptually, not byte-for-byte** (V2 runner path, flag-gated `VLLM_GDN_DEINTERLEAVE_QKVZ`/`VLLM_GDN_CONCAT_TINY_GEMMS`). Expected effect: cuts GDN decode launch/copy overhead, which is our second-largest c8 bucket (13.5%) — plausibly several % at c8-16. Adoption: cherry-pick as a mod (python/Triton only, no b12x kernel change) once V2 runner compat confirmed; medium effort, high value.

2. **KK fork (integration/karmic-kraken-beta) `a18246b` — "Fuse GDN metadata copies and state refresh across cache groups", 2026-09-22.** One Triton launch instead of several for group-owned worklist + recurrent-state/checkpoint index updates; their own measured mean verifier-step latency 41.19→40.16 ms (~2.5%) on TP2 Qwen serving with warmed prompts. Directly on our exact target arch (Qwen GDN, TP2). **Applies to our stack**: it's on the KK branch we already track (`ursuciprian/vllm@dgx-spark-kk`); pull forward as a cherry-pick, small diff, low risk (python/Triton, graph-replay tested). Adoption: cherry-pick, easy.

3. **Upstream vLLM #56577 — opt-in FP8 proposal (draft) head for Qwen4Exp MTP, 09-12.** Private rowwise-E4M3 copy of the MTP vocab-projection weight for the draft path only (target verification stays BF16); uses `scaled_fp8_quant`/`cutlass_scaled_mm`, TP1/TP2, eager, Model Runner V2. This is the same lever family as our shipped `VLLM_MXFP8_LM_HEAD=1` (promoted, +4.1% c1) but on the draft head specifically, and upstream on our exact model id (`qwen4_exp`/Qwen3.8-Flash-Next). Our MTP draft head is already NVFP4 per our journal (719 µs), so this PR may be redundant for us — worth a 30-min read to confirm it isn't targeting a BF16-draft-head config we don't have. If it targets a path we do have (verify-side lm_head is still BF16/cuBLAS per our lmhead-hc-design finding), it's a second, upstream-reviewed route to the same win we already captured differently. Adoption: read the diff, likely skip (already covered) or cherry-pick if it hits the verify head instead.

4. **Upstream vLLM issue #58020 — "Engine-resolved prefix-cache match unit is not propagated to workers", 09-21, still open.** `vllm/model_executor/layers/mamba/checkpoint.py:86` — workers fall back to `kv_cache_spec.block_size` instead of using the engine's resolved `prefix_match_unit`. This is the exact mechanism class behind our `la-pmu16` result (`--prefix-match-unit 16` helps at depth, has an unexplained depth-0 cost, and our own Fix-2 investigation into `scheduler.py`/tail-stop gating came up partly null). Confirms the propagation path is genuinely buggy upstream, independent of our stack — worth reading the full issue body/thread for the worker-side fix before spending more of our own time reverse-engineering the depth-0 regression. Adoption: read-only lead for now (upstream unfixed); if a patch lands, it's a direct cherry-pick candidate onto our `prefix_match_unit`/`long-prefix-ttft` work.

5. **`local-inference-lab/rtx6kpro` model guide for Qwen3.8-Flash-Next (KK beta 2026-09-19 snapshot).** Independent third-party numbers at **default temperature 1/top-p .95/top-k 20** (same regime as our mtpprob finding): C1 172.9→190.1 tok/s (+9.9%), C8 664.9→689.9 (+3.8%), and **acceptance 2.105→2.280 accepted/draft-token** between "Community R35" and their "wheel image" — i.e. someone else measured the same default-temperature MTP acceptance problem and a real gain from an upgraded image, corroborating that our `la-mtpprob` (+33% c1 default-temp, no c1/c8 cost) direction is the right one and that KK likely already bakes in some of the acceptance fix. Also documents their recipe: CPU-offloaded PLE tables (`VLLM_PLE_CPU_OFFLOAD=1`), FP8 attention KV, native `auto` prefix cache mode, graph cap 64. Adoption: no code to pull, but strengthens the case to (a) finish `la-mtpprob`'s hardmode/c16 gate, and (b) do the KK-vs-la gate specifically under default-temperature sampling, since that's the regime where KK's upstream deltas show up.

## Also seen, lower priority / not directly actionable now

- vLLM #55357 (open bug): Qwen3.8-Flash-Next + MTP episodic 0% draft acceptance + thinking-block repetition collapse, nightly build, cross-model (also hit Gemma 4) — an *engine-level* spec-decode state-divergence bug, distinct from our "acceptance collapses under temp 1.0" finding (that one is a modeled, reproducible sampling effect; this is an intermittent divergence). Worth a hardmode/long-run watch on our side but no fix to adopt yet.
- vLLM #55697 (RFC) "Application-Directed Prefix Checkpoints for Mamba/Hybrid Prefix Caching" — a different mechanism (client-declared `<|mamba_checkpoint|>` marker) from our GDN deferred-checkpoints branch (which defers full recurrent-state writes to the accepted-prefix commit); conceptually adjacent, not a drop-in.
- vLLM #56466 (WIP) "ReplaySSM prefix caching for GDN speculative decode" and #55868 (WIP) "GDN batch-invariant prefix caching" — both early/WIP, not mergeable, but confirm upstream is actively working the same mamba-prefix + spec-decode intersection our `la-pmu16`/tail-minblock work sits in.
- SGLang #39680 "Coalesce the KDA CuTe DSL decode state transpose: ~3x faster, bit-identical" (09-16) — KDA (Kimi Delta Attention) is architecturally close to GDN; the transpose-coalescing idea is portable inspiration for our own GDN decode kernel bandwidth problem, but it's a different kernel/DSL (CuTe vs b12x's own kernels) — would need a from-scratch b12x port, not a patch.
- eugr/spark-vllm-docker commits since 09-01: mostly GLM-5.3/DeepSeek-V4/instanttensor memory work; nothing Qwen3.8-Flash-Next/MTP/GDN-specific beyond what we already track (PLE offload support 09-10, b12x autotune re-enable 09-17 — both already in our lineage).
- b12x master since 09-01: heavy IQ2_XS/NVFP4-decode-layout/MoE-tuning churn, nothing GDN-decode-kernel-specific found in this pass; worth a follow-up targeted diff of `gdn_decode`/`_cute_kernels.py` between `a8333658` and current tip given the KK metadata-fusion commit above shows active GDN work upstream-adjacent.

## Not adopted / explicitly out of scope this pass

Qwen team GitHub guidance issues on MTP+sampling: none found dated since 09-01 beyond what our own journal (2026-09-21 10:29) already extracted from the model card (temp 0.6/0.95/20 recommended over greedy for thinking mode). No new Qwen-side guidance this window.

---

# Adoption verdicts (same day, read-only on GPUs)

## Item 2 — KK `a18246b` GDN metadata fusion: **ADOPTED**

Applies to `8e1f1e58` unchanged. `vllm/v1/attention/backends/b12x_gdn_metadata.py`
is byte-identical between our base and `a18246b06^`; `gdn_attn.py` diverges by
4 lines, none near the touched hunk; `git cherry-pick` is conflict-free. The
image `spark-vllm-b12x:local-20260918-a8333658` carries the same pre-image
bytes. Shipped as `patches/vllm-gdn-meta-fuse.patch` +
`mods/vllm-gdn-meta-fuse/` (CPU-tested in the container). Expected ~-1.9% at
c1 / ~-1.2% at c8, inside the busy 94% of the step. Test with
`mods: [b12x-startup-boundedwait, vllm-gdn-meta-fuse]`.

## Item 1 — upstream #55260 GDN decode-path opts: **NOT PORTED**, three reasons

The PR is a 4-commit stack on top of #54637 (the V2 all-mode prefix-caching
base). Per-piece against `8e1f1e58` with `--gdn-decode-kernel b12x`:

1. **Packed `mixed_qkv` for the fused gating kernel** (`51b571bf3`, 27+/2- in
   `qwen_gdn_linear_attn.py`, 74+/18- in `fused_sigmoid_gating.py`).
   **Dead path for us.** `fused_sigmoid_gating_delta_rule_update` is reached
   only from `_forward_core` (ours:3200-line `qwen_gdn_linear_attn.py`, lines
   2210/2237/2318). With the b12x decode kernel, `forward_cuda` short-circuits
   at `use_fused_gdn_decode` into
   `torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed` ->
   `_forward_core_fused_norm` -> `_forward_core_b12x_mixed` /
   `_forward_core_decode_b12x_fused_norm`; `_forward_core` is the fallback for
   shapes b12x refuses. Our fork also already keeps a packed `mixed_qkv`
   staging buffer (`_B12xGdnDecodeStaging.mixed_qkv`, `packed_qkv_width`).

2. **Fused block-index prep + in-kernel checkpoint stores + mask-select hoist**
   (`e34de8a4d`). **Real cost in our fork, but not a port.** We do have the
   host-side per-step prep the PR attacks: `B12xGdnMixedMetadata.stage()`
   (`b12x_gdn_metadata.py:156`) builds Python lists and issues ~14 `_copy_list`
   H2D copies on **every** `build()`, before the uniform-spec-decode fast path
   returns. But (a) our shape is `B12xGdnMixedMetadata` + the existing
   `_stage_b12x_gdn_metadata_kernel` Triton staging kernel, not upstream's new
   `gdn_index_prep.py` / `GDNBlockIdxPrepBuffers` /
   `MambaHybridModelState` -- the port is a rewrite of `stage()`, ~600 lines
   across 5 files we have in a different shape, far past the ~150-line bar;
   and (b) the win it buys is capped by our idle window: GPU idle fraction is
   **5.4% at c1 and 5.6% at c8** (`results/profiling/README.md`), so host-side
   prep is already almost entirely overlapped behind GPU work. Upstream's
   +3.0% -> +12.2% was measured on B200, where the GPU is fast enough relative
   to the host for that prep to be exposed. On GB10 we are compute-bound at
   both concurrencies. Revisit only if a future profile shows idle climbing.

3. **Weight de-interleave + tiny-GEMM fold at load** (`eb8742705`). **Already
   have the good half; the other half is measured dead.** De-interleave: our
   checkpoint is non-interleaved (`gqa_interleaved_layout=False`, `[q,k,v,z]`
   order) and the b12x path hands the raw `in_proj_qkvz` GEMM output straight
   to the packed op, splitting it into zero-copy views inside
   `_forward_core_fused_norm_packed` -- upstream's row-permute buys us nothing
   we do not already have. Tiny-GEMM fold (`in_proj_ba` into the main GEMM,
   N 12288 -> 12352): `results/kernel-pass/dense-gemm-fusion.md` §1 already
   measured `in_proj_ba` at 9.95 us on the **aux stream**, overlapped with the
   119.82 us `in_proj_qkvz` (`VLLM_QWEN3_8_FLASH_NEXT_OVERLAP`), i.e. ~0 wall,
   and noted the fold forces non-contiguous slices or two extra copies into the
   packed GDN op. That lead was already closed as dead on our stack.

(Optimization 4, the monolithic trtllm-gen NvFp4 experts, is out of scope here:
opt-in, DP>1 ag_rs only.)

## Item 4 — upstream #58020 prefix-match-unit propagation: **DOES NOT APPLY**

The buggy consumer does not exist in our fork. `8e1f1e58` has no
`vllm/model_executor/layers/mamba/checkpoint.py` at all, and no
`prefix_match_unit or block_size` fallback anywhere in `vllm/` -- a grep for
`prefix_match_unit` on the base hits only `config/cache.py`, `config/vllm.py`,
`engine/arg_utils.py`, `platforms/interface.py`, `v1/core/kv_cache_utils.py`
(the `resolve_kv_cache_block_sizes` GCD resolver), `v1/core/kv_cache_coordinator.py`
and `v1/core/sched/scheduler.py` -- **all engine/scheduler side, zero worker-side
consumers**. `vllm/v1/worker/gpu/boundary_checkpoint.py:86` is our own
prompt/instruction boundary-capture Triton kernel (stop-token loop), an
unrelated feature. So the `la-pmu16` depth-0 penalty (+0.19 s) is **not** this
mechanism and that lead is closed; no patch written.
