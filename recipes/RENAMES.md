# Recipe renames (2026-09-22)

The `recipes/eugr/` directory (60 files named after eugr's upstream base recipe,
with opaque internal labels like `agents`/`serve`/`local16`/`la`) was renamed to
`recipes/qwen3.8-flash-next/`, with each file renamed to say what it actually is:
`qwen3.8-flash-next-nvfp4-tp2[-<variant>].yaml`, where the variant names the one
thing that differs from the default. eugr is still credited in the README as the
origin of the base recipe this project bisected from.

| old path | new path | purpose | status |
|---|---|---|---|
| `recipes/eugr/eugr-b12x-cluster.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-cluster-baseline.yaml` | Plain (non-agent-tuned) cluster baseline from eugr's original recipe | historical |
| `recipes/eugr/eugr-agents.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-tuned-baseline.yaml` | Agent-tuned bisect baseline: 8 concurrent seqs, 8192-token prefill chunks, gpu util 0.85 | historical |
| `recipes/eugr/eugr-agents-serve.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-serving-baseline.yaml` | Historical serving default: agents baseline + decode-aware prefill scheduling, superseded by the local16-la default | historical |
| `recipes/eugr/eugr-agents-decode-aware.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware.yaml` | Agents baseline + decode-aware prefill scheduling (reserves a decode lane instead of letting long prefills starve decode) | historical |
| `recipes/eugr/eugr-gdn-triton.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-gdn-triton-kernel.yaml` | Bisect arm: --gdn-decode-kernel triton instead of b12x | experimental |
| `recipes/eugr/eugr-kv-auto.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-kv-bf16-bisect.yaml` | Bisect arm: --kv-cache-dtype auto (BF16) instead of fp8 | experimental |
| `recipes/eugr/eugr-linear-default.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-linear-backend-default.yaml` | Bisect arm: without --linear-backend b12x | experimental |
| `recipes/eugr/eugr-moe-default.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-moe-backend-default.yaml` | Bisect arm: without --moe-backend b12x | experimental |
| `recipes/eugr/eugr-no-roce-ar.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-roce-allreduce-off.yaml` | Bisect arm: VLLM_ENABLE_ROCE_ALLREDUCE off | experimental |
| `recipes/eugr/eugr-nomtp.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-no-mtp.yaml` | Bisect arm: without MTP speculative decoding (plain decode rate) | experimental |
| `recipes/eugr/eugr-agents-kvbf16.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-kv-bf16.yaml` | Quality arm A1: BF16 KV cache instead of fp8 | historical-rejected |
| `recipes/eugr/eugr-agents-nvidia-kvbf16.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-nvidia-ckpt-kv-bf16.yaml` | Quality arm N1 (fallback): nvidia/Qwen3.8-Flash-Next-NVFP4 checkpoint, BF16 KV | fallback |
| `recipes/eugr/eugr-agents-radixark.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-radixark-ckpt.yaml` | Quality arm R0: RadixArk/Qwen3.8-Flash-Next-NVFP4 checkpoint, fp8 KV | historical-rejected |
| `recipes/eugr/eugr-agents-radixark-kvbf16.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-radixark-ckpt-kv-bf16.yaml` | Quality arm R1: RadixArk checkpoint + BF16 KV cache | historical-rejected |
| `recipes/eugr/eugr-agents-oldckpt.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-old-checkpoint-rev.yaml` | Pinned older checkpoint revision (ada4da3) | historical-rejected |
| `recipes/eugr/eugr-agents-lpt4096.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-long-prefill-threshold-4096.yaml` | Caps each request's prefill at 4096 tokens (long_prefill_token_threshold) | experimental |
| `recipes/eugr/eugr-agents-mtp3.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp3.yaml` | Agents recipe with 3 MTP speculative tokens | experimental |
| `recipes/eugr/eugr-agents-mtp5.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp5.yaml` | Agents recipe with 5 MTP speculative tokens | experimental |
| `recipes/eugr/eugr-agents-mtp6.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp6.yaml` | Agents recipe with 6 MTP speculative tokens | experimental |
| `recipes/eugr/eugr-agents-roce1m.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-roce-allreduce-1mb.yaml` | RoCE all-reduce threshold at 1MB | experimental |
| `recipes/eugr/eugr-agents-roce4m.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-roce-allreduce-4mb.yaml` | RoCE all-reduce threshold at 4MB | experimental |
| `recipes/eugr/batched4096.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-batched-4096.yaml` | Decode-aware scheduling + max_num_batched_tokens 4096 | experimental |
| `recipes/eugr/gputil80.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-gpu-util-80.yaml` | Decode-aware scheduling + gpu_memory_utilization 0.80 | experimental |
| `recipes/eugr/mamba-bf16.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-mamba-bf16.yaml` | Decode-aware scheduling + MAMBA_SSM_CACHE_DTYPE bfloat16 | experimental |
| `recipes/eugr/prefill2.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-parallel-prefills-2.yaml` | Decode-aware scheduling + max-parallel-prefills 2 (was 4) | experimental |
| `recipes/eugr/seqs16.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-seqs-16.yaml` | Decode-aware scheduling + max_num_seqs 16 | experimental |
| `recipes/eugr/eugr-agents-serve-local.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-local-build.yaml` | Serving default, but container points at build.sh's own local image | experimental |
| `recipes/eugr/eugr-agents-serve-local16.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-local-build-seqs-16.yaml` | Local build arm + max_num_seqs 16 | experimental |
| `recipes/eugr/eugr-agents-serve-local16-autotune.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-local-build-seqs-16-autotune.yaml` | Local build arm, seqs 16, B12X_AUTOTUNE enabled, no runtime-cache mount | experimental |
| `recipes/eugr/eugr-agents-serve-local-dv.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-draft-vocab-reduced.yaml` | Lever 1: reduced-vocab MTP draft head (MiaAI-Lab 47k table) | experimental |
| `recipes/eugr/eugr-agents-serve-local-ep.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-expert-parallel.yaml` | Lever 2: --enable-expert-parallel; not expected to work, b12x MoE EP is W4A16-only | historical-rejected |
| `recipes/eugr/eugr-agents-serve-local-gdnbf16.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-gdn-state-bf16.yaml` | Lever 3: --mamba-ssm-cache-dtype bfloat16 (GDN recurrent state in BF16) | experimental |
| `recipes/eugr/eugr-agents-serve-capsizes.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-cudagraph-capture-sizes.yaml` | Straggler probe: expanded cudagraph_capture_sizes (3,5,6,7,12 added) | experimental |
| `recipes/eugr/eugr-agents-serve-mtp3.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp3-straggler-probe.yaml` | Straggler probe: num_speculative_tokens 3 (was 4) | experimental |
| `recipes/eugr/eugr-agents-serve-noshare16.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp-index-share-off.yaml` | Straggler probe: seqs 16 + index_share_for_mtp_iteration false | experimental |
| `recipes/eugr/eugr-agents-serve-nospec.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-no-speculative-decoding.yaml` | Probe: MTP speculative decoding removed entirely | experimental |
| `recipes/eugr/eugr-agents-serve-scratchfix16.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-scratch-isolation.yaml` | Straggler probe: seqs 16 + disjoint GDN/QSA projection scratch mod | experimental |
| `recipes/eugr/eugr-agents-serve-spectrace.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-spec-trace-debug.yaml` | Debug arm: vllm-spec-trace mod, logs drafts-proposed/accepted counts | experimental |
| `recipes/eugr/eugr-agents-serve-spectrace16.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-spec-trace-debug-seqs-16.yaml` | Debug arm: spec-trace + max_num_seqs 16 | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-argmax.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-argmax-drafts.yaml` | Previous default: one-hot MTP drafts + use_local_argmax_reduction, kept as fallback | fallback |
| `recipes/eugr/eugr-agents-serve-local16-la-b12x0f3a8cb.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-0f3a8cb.yaml` | Bisect arm: b12x build tag 0f3a8cb | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-b12x0f3a8cb-noat.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-0f3a8cb-no-autotune.yaml` | Bisect arm: b12x 0f3a8cb build with B12X_AUTOTUNE off | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-b12x0f3a8cb-rev06809d5.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-0f3a8cb-revert-06809d5.yaml` | Bisect arm: b12x 0f3a8cb build with commit 06809d5 reverted | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-fusear.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-fuse-allreduce-rms.yaml` | Compilation pass fuse_allreduce_rms enabled | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-fwd57f3572.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-fwd-57f3572.yaml` | Bisect arm: b12x-fwd-57f3572 mod applied | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-fwd57f3572-fix.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-fwd-57f3572-fix.yaml` | Bisect arm: b12x-fwd-57f3572 mod + startup boundedwait fix | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-fwd57f3572-trace.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-fwd-57f3572-trace.yaml` | Bisect arm: b12x-fwd-57f3572 mod + startup trace mod | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-gemv.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-bf16-gemv.yaml` | BF16 GEMV kernel path arm | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-ghcr.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-ghcr-image.yaml` | Same as default, but container pulled from ghcr.io instead of a local build | fallback |
| `recipes/eugr/eugr-agents-serve-local16-la-kk.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-kk-image.yaml` | Container built from the karmic-kraken-beta (KK) integration branch | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-lmq.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mxfp8-lmhead.yaml` | Online MXFP8 verify lm_head; measurements folded into the default recipe | historical |
| `recipes/eugr/eugr-agents-serve-local16-la-lprof.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-profiler-local-rank.yaml` | Profiling only: rank-local torch.profiler wrap, no collective RPC | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-occ_dynhigh.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-dyn-high.yaml` | B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS 64 | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-occ_dynlow.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-dyn-low.yaml` | B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS 32 | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-occ_microhigh.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-micro-high.yaml` | B12X_MICRO_MAX_ACTIVE_CLUSTERS 64 | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-occ_microlow.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-micro-low.yaml` | B12X_MICRO_MAX_ACTIVE_CLUSTERS 32 | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-prof.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-profiler-config.yaml` | Profiling only: built-in --profiler-config for Phase C decode-step profiling | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-spec3.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-argmax-mtp3.yaml` | Argmax-reduction arm with num_speculative_tokens 3 (was 4) | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la-tc.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-tc45-reasoning-fix.yaml` | tool_choice=required structural-tag fix so forced grammar admits the <think> prefix | experimental |
| `recipes/eugr/eugr-agents-serve-local16-la.yaml` | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml` | Default: local16 + probabilistic MTP draft sampling | default |
