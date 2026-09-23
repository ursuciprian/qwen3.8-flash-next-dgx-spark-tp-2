# Archive move (2026-09-23)

sparkrun lists every `*.yaml` under this registry's `recipes/` subpath
(recursive `rglob`, sparkrun 0.3.6 `core/registry.py`) and sparse-checks-out only
`recipes/`, `mods/` and `.sparkrun/`. To make `sparkrun recipe list` show only what
a user should run, everything except the default and one fallback moved to a
top-level `archive/`, which sparkrun neither clones nor scans. Moves were `git mv`,
so `git log --follow <new path>` shows each file's full history. Archived recipes
still resolve their mods through the relative `mods` symlinks, now pointing at
`archive/mods/`.

Still registry-visible: `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml`
(default) and `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-argmax-drafts.yaml`
(fallback, rebuilt on the default's warm ghcr image; its previous local-image
version is the `-argmax-drafts-local.yaml` row below).

| old path | moved to |
|---|---|
| mods/README.md | `archive/mods/README.md` |
| mods/b12x-fwd-57f3572/ | `archive/mods/b12x-fwd-57f3572/` |
| mods/b12x-revert-06809d5/ | `archive/mods/b12x-revert-06809d5/` |
| mods/b12x-startup-boundedwait/ | `archive/mods/b12x-startup-boundedwait/` |
| mods/b12x-startup-trace/ | `archive/mods/b12x-startup-trace/` |
| mods/sglang-gdn-b12x-decode/ | `archive/mods/sglang-gdn-b12x-decode/` |
| mods/sglang-radix-chunked-insert-fix/ | `archive/mods/sglang-radix-chunked-insert-fix/` |
| mods/sglang-sm121-qsa-fp8kv/ | `archive/mods/sglang-sm121-qsa-fp8kv/` |
| mods/vllm-decode-profiler/ | `archive/mods/vllm-decode-profiler/` |
| mods/vllm-dv-devicefix/ | `archive/mods/vllm-dv-devicefix/` |
| mods/vllm-flashnext-nightly-8a728663/ | `archive/mods/vllm-flashnext-nightly-8a728663/` |
| mods/vllm-gdn-deferred/ | `archive/mods/vllm-gdn-deferred/` |
| mods/vllm-gdn-meta-fuse/ | `archive/mods/vllm-gdn-meta-fuse/` |
| mods/vllm-instanttensor-memory/ | `archive/mods/vllm-instanttensor-memory/` |
| mods/vllm-qsa-fp8kv-pr55557/ | `archive/mods/vllm-qsa-fp8kv-pr55557/` |
| mods/vllm-qwen-scratch-isolation/ | `archive/mods/vllm-qwen-scratch-isolation/` |
| mods/vllm-qwen38-bf16-gemv/ | `archive/mods/vllm-qwen38-bf16-gemv/` |
| mods/vllm-qwen38-hc-mxfp8/ | `archive/mods/vllm-qwen38-hc-mxfp8/` |
| mods/vllm-qwen38-lmhead-b12x/ | `archive/mods/vllm-qwen38-lmhead-b12x/` |
| mods/vllm-spec-trace/ | `archive/mods/vllm-spec-trace/` |
| mods/vllm-tc45-cheap-kk/ | `archive/mods/vllm-tc45-cheap-kk/` |
| mods/vllm-tc45-reasoning-structag-fix/ | `archive/mods/vllm-tc45-reasoning-structag-fix/` |
| recipes/README.md | `archive/recipes/README.md` |
| recipes/arms/ | `archive/recipes/arms/` |
| recipes/dflash2/ | `archive/recipes/dflash2/` |
| recipes/flashnext-balanced.yaml | `archive/recipes/flashnext-balanced.yaml` |
| recipes/flashnext-bigkv-g8-c4096.yaml | `archive/recipes/flashnext-bigkv-g8-c4096.yaml` |
| recipes/flashnext-bigkv-g8.yaml | `archive/recipes/flashnext-bigkv-g8.yaml` |
| recipes/flashnext-bigkv-nospec.yaml | `archive/recipes/flashnext-bigkv-nospec.yaml` |
| recipes/flashnext-bigkv.yaml | `archive/recipes/flashnext-bigkv.yaml` |
| recipes/flashnext-c16-short-candidate.yaml | `archive/recipes/flashnext-c16-short-candidate.yaml` |
| recipes/flashnext-deep-concurrency.yaml | `archive/recipes/flashnext-deep-concurrency.yaml` |
| recipes/flashnext-dflash2-harvest.yaml | `archive/recipes/flashnext-dflash2-harvest.yaml` |
| recipes/flashnext-dflash2-serve.yaml | `archive/recipes/flashnext-dflash2-serve.yaml` |
| recipes/flashnext-fp8kv-1m8-b12x-debug.yaml | `archive/recipes/flashnext-fp8kv-1m8-b12x-debug.yaml` |
| recipes/flashnext-fp8kv-1m8-b12x.yaml | `archive/recipes/flashnext-fp8kv-1m8-b12x.yaml` |
| recipes/flashnext-fp8kv-1m8-cutedsl.yaml | `archive/recipes/flashnext-fp8kv-1m8-cutedsl.yaml` |
| recipes/flashnext-fp8kv-1m8-gdnfi.yaml | `archive/recipes/flashnext-fp8kv-1m8-gdnfi.yaml` |
| recipes/flashnext-fp8kv-1m8-nospec-b12x.yaml | `archive/recipes/flashnext-fp8kv-1m8-nospec-b12x.yaml` |
| recipes/flashnext-fp8kv-1m8-nospec-cutedsl.yaml | `archive/recipes/flashnext-fp8kv-1m8-nospec-cutedsl.yaml` |
| recipes/flashnext-fp8kv-1m8-nospec-triton.yaml | `archive/recipes/flashnext-fp8kv-1m8-nospec-triton.yaml` |
| recipes/flashnext-fp8kv-1m8.yaml | `archive/recipes/flashnext-fp8kv-1m8.yaml` |
| recipes/flashnext-vllm-cached.yaml | `archive/recipes/flashnext-vllm-cached.yaml` |
| recipes/flashnext-vllm.yaml | `archive/recipes/flashnext-vllm.yaml` |
| recipes/mods | `archive/recipes/mods` |
| recipes/qwen3.8-flash-next/mods | `archive/recipes/qwen3.8-flash-next/mods` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-argmax-drafts.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-argmax-drafts-local.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-argmax-mtp3.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-argmax-mtp3.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-0f3a8cb-no-autotune.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-0f3a8cb-no-autotune.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-0f3a8cb-revert-06809d5.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-0f3a8cb-revert-06809d5.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-0f3a8cb.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-0f3a8cb.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-fwd-57f3572-fix.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-fwd-57f3572-fix.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-fwd-57f3572-trace.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-fwd-57f3572-trace.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-fwd-57f3572.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-fwd-57f3572.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-bf16-gemv.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-bf16-gemv.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-cluster-baseline.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-cluster-baseline.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-cudagraph-capture-sizes.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-cudagraph-capture-sizes.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-batched-4096.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-batched-4096.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-gpu-util-80.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-gpu-util-80.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-mamba-bf16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-mamba-bf16.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-parallel-prefills-2.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-parallel-prefills-2.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-seqs-16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-seqs-16.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-draft-vocab-reduced.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-draft-vocab-reduced.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-expert-parallel.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-expert-parallel.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-fuse-allreduce-rms.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-fuse-allreduce-rms.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-gdn-deferred-off.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-gdn-deferred-off.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-gdn-deferred.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-gdn-deferred.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-gdn-state-bf16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-gdn-state-bf16.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-gdn-triton-kernel.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-gdn-triton-kernel.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-ghcr-image.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-ghcr-image.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-kk-image.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-kk-image.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-kv-bf16-bisect.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-kv-bf16-bisect.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-kv-bf16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-kv-bf16.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-linear-backend-default.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-linear-backend-default.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-local-build-seqs-16-autotune.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-local-build-seqs-16-autotune.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-local-build-seqs-16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-local-build-seqs-16.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-local-build.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-local-build.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-long-prefill-threshold-4096.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-long-prefill-threshold-4096.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-moe-backend-default.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-moe-backend-default.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp-index-share-off.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp-index-share-off.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp3-straggler-probe.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp3-straggler-probe.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp3.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp3.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp5.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp5.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp6.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp6.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mxfp8-lmhead.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mxfp8-lmhead.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-no-mtp.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-no-mtp.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-no-speculative-decoding.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-no-speculative-decoding.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-nvidia-ckpt-kv-bf16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-nvidia-ckpt-kv-bf16.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-dyn-high.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-dyn-high.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-dyn-low.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-dyn-low.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-micro-high.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-micro-high.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-micro-low.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-micro-low.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-old-checkpoint-rev.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-old-checkpoint-rev.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-profiler-config.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-profiler-config.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-profiler-local-rank.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-profiler-local-rank.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-radixark-ckpt-kv-bf16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-radixark-ckpt-kv-bf16.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-radixark-ckpt.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-radixark-ckpt.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-roce-allreduce-1mb.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-roce-allreduce-1mb.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-roce-allreduce-4mb.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-roce-allreduce-4mb.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-roce-allreduce-off.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-roce-allreduce-off.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-scratch-isolation.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-scratch-isolation.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-serving-baseline.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-serving-baseline.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-spec-trace-debug-seqs-16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-spec-trace-debug-seqs-16.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-spec-trace-debug.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-spec-trace-debug.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-tc45-reasoning-fix.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-tc45-reasoning-fix.yaml` |
| recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-tuned-baseline.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-tuned-baseline.yaml` |
| recipes/retired/ | `archive/recipes/retired/` |

# Recipe renames (2026-09-22)

The recipes/eugr/ directory (60 files named after eugr's upstream base recipe,
with opaque internal labels like `agents`/`serve`/`local16`/`la`) was renamed to
`recipes/qwen3.8-flash-next/`, with each file renamed to say what it actually is:
`qwen3.8-flash-next-nvfp4-tp2[-<variant>].yaml`, where the variant names the one
thing that differs from the default. eugr is still credited in the README as the
origin of the base recipe this project bisected from.

| old path | path today (after the 2026-09-23 archive move) | purpose | status |
|---|---|---|---|
| recipes/eugr/eugr-b12x-cluster.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-cluster-baseline.yaml` | Plain (non-agent-tuned) cluster baseline from eugr's original recipe | historical |
| recipes/eugr/eugr-agents.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-tuned-baseline.yaml` | Agent-tuned bisect baseline: 8 concurrent seqs, 8192-token prefill chunks, gpu util 0.85 | historical |
| recipes/eugr/eugr-agents-serve.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-serving-baseline.yaml` | Historical serving default: agents baseline + decode-aware prefill scheduling, superseded by the local16-la default | historical |
| recipes/eugr/eugr-agents-decode-aware.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware.yaml` | Agents baseline + decode-aware prefill scheduling (reserves a decode lane instead of letting long prefills starve decode) | historical |
| recipes/eugr/eugr-gdn-triton.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-gdn-triton-kernel.yaml` | Bisect arm: --gdn-decode-kernel triton instead of b12x | experimental |
| recipes/eugr/eugr-kv-auto.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-kv-bf16-bisect.yaml` | Bisect arm: --kv-cache-dtype auto (BF16) instead of fp8 | experimental |
| recipes/eugr/eugr-linear-default.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-linear-backend-default.yaml` | Bisect arm: without --linear-backend b12x | experimental |
| recipes/eugr/eugr-moe-default.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-moe-backend-default.yaml` | Bisect arm: without --moe-backend b12x | experimental |
| recipes/eugr/eugr-no-roce-ar.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-roce-allreduce-off.yaml` | Bisect arm: VLLM_ENABLE_ROCE_ALLREDUCE off | experimental |
| recipes/eugr/eugr-nomtp.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-no-mtp.yaml` | Bisect arm: without MTP speculative decoding (plain decode rate) | experimental |
| recipes/eugr/eugr-agents-kvbf16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-kv-bf16.yaml` | Quality arm A1: BF16 KV cache instead of fp8 | historical-rejected |
| recipes/eugr/eugr-agents-nvidia-kvbf16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-nvidia-ckpt-kv-bf16.yaml` | Quality arm N1 (fallback): nvidia/Qwen3.8-Flash-Next-NVFP4 checkpoint, BF16 KV | fallback |
| recipes/eugr/eugr-agents-radixark.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-radixark-ckpt.yaml` | Quality arm R0: RadixArk/Qwen3.8-Flash-Next-NVFP4 checkpoint, fp8 KV | historical-rejected |
| recipes/eugr/eugr-agents-radixark-kvbf16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-radixark-ckpt-kv-bf16.yaml` | Quality arm R1: RadixArk checkpoint + BF16 KV cache | historical-rejected |
| recipes/eugr/eugr-agents-oldckpt.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-old-checkpoint-rev.yaml` | Pinned older checkpoint revision (ada4da3) | historical-rejected |
| recipes/eugr/eugr-agents-lpt4096.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-long-prefill-threshold-4096.yaml` | Caps each request's prefill at 4096 tokens (long_prefill_token_threshold) | experimental |
| recipes/eugr/eugr-agents-mtp3.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp3.yaml` | Agents recipe with 3 MTP speculative tokens | experimental |
| recipes/eugr/eugr-agents-mtp5.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp5.yaml` | Agents recipe with 5 MTP speculative tokens | experimental |
| recipes/eugr/eugr-agents-mtp6.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp6.yaml` | Agents recipe with 6 MTP speculative tokens | experimental |
| recipes/eugr/eugr-agents-roce1m.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-roce-allreduce-1mb.yaml` | RoCE all-reduce threshold at 1MB | experimental |
| recipes/eugr/eugr-agents-roce4m.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-roce-allreduce-4mb.yaml` | RoCE all-reduce threshold at 4MB | experimental |
| recipes/eugr/batched4096.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-batched-4096.yaml` | Decode-aware scheduling + max_num_batched_tokens 4096 | experimental |
| recipes/eugr/gputil80.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-gpu-util-80.yaml` | Decode-aware scheduling + gpu_memory_utilization 0.80 | experimental |
| recipes/eugr/mamba-bf16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-mamba-bf16.yaml` | Decode-aware scheduling + MAMBA_SSM_CACHE_DTYPE bfloat16 | experimental |
| recipes/eugr/prefill2.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-parallel-prefills-2.yaml` | Decode-aware scheduling + max-parallel-prefills 2 (was 4) | experimental |
| recipes/eugr/seqs16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-decode-aware-seqs-16.yaml` | Decode-aware scheduling + max_num_seqs 16 | experimental |
| recipes/eugr/eugr-agents-serve-local.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-local-build.yaml` | Serving default, but container points at build.sh's own local image | experimental |
| recipes/eugr/eugr-agents-serve-local16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-local-build-seqs-16.yaml` | Local build arm + max_num_seqs 16; serving fallback before la | fallback |
| recipes/eugr/eugr-agents-serve-local16-autotune.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-local-build-seqs-16-autotune.yaml` | Local build arm, seqs 16, B12X_AUTOTUNE enabled, no runtime-cache mount | experimental |
| recipes/eugr/eugr-agents-serve-local-dv.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-draft-vocab-reduced.yaml` | Lever 1: reduced-vocab MTP draft head (MiaAI-Lab 47k table) | experimental |
| recipes/eugr/eugr-agents-serve-local-ep.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-expert-parallel.yaml` | Lever 2: --enable-expert-parallel; not expected to work, b12x MoE EP is W4A16-only | historical-rejected |
| recipes/eugr/eugr-agents-serve-local-gdnbf16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-gdn-state-bf16.yaml` | Lever 3: --mamba-ssm-cache-dtype bfloat16 (GDN recurrent state in BF16) | experimental |
| recipes/eugr/eugr-agents-serve-capsizes.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-cudagraph-capture-sizes.yaml` | Straggler probe: expanded cudagraph_capture_sizes (3,5,6,7,12 added) | experimental |
| recipes/eugr/eugr-agents-serve-mtp3.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp3-straggler-probe.yaml` | Straggler probe: num_speculative_tokens 3 (was 4) | experimental |
| recipes/eugr/eugr-agents-serve-noshare16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mtp-index-share-off.yaml` | Straggler probe: seqs 16 + index_share_for_mtp_iteration false | experimental |
| recipes/eugr/eugr-agents-serve-nospec.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-no-speculative-decoding.yaml` | Probe: MTP speculative decoding removed entirely | experimental |
| recipes/eugr/eugr-agents-serve-scratchfix16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-scratch-isolation.yaml` | Straggler probe: seqs 16 + disjoint GDN/QSA projection scratch mod | experimental |
| recipes/eugr/eugr-agents-serve-spectrace.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-spec-trace-debug.yaml` | Debug arm: vllm-spec-trace mod, logs drafts-proposed/accepted counts | experimental |
| recipes/eugr/eugr-agents-serve-spectrace16.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-spec-trace-debug-seqs-16.yaml` | Debug arm: spec-trace + max_num_seqs 16 | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-argmax.yaml | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-argmax-drafts.yaml` | Previous default: one-hot MTP drafts + use_local_argmax_reduction, kept as fallback | fallback |
| recipes/eugr/eugr-agents-serve-local16-la-b12x0f3a8cb.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-0f3a8cb.yaml` | Bisect arm: b12x build tag 0f3a8cb | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-b12x0f3a8cb-noat.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-0f3a8cb-no-autotune.yaml` | Bisect arm: b12x 0f3a8cb build with B12X_AUTOTUNE off | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-b12x0f3a8cb-rev06809d5.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-0f3a8cb-revert-06809d5.yaml` | Bisect arm: b12x 0f3a8cb build with commit 06809d5 reverted | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-fusear.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-fuse-allreduce-rms.yaml` | Compilation pass fuse_allreduce_rms enabled | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-fwd57f3572.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-fwd-57f3572.yaml` | Bisect arm: b12x-fwd-57f3572 mod applied | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-fwd57f3572-fix.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-fwd-57f3572-fix.yaml` | Bisect arm: b12x-fwd-57f3572 mod + startup boundedwait fix | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-fwd57f3572-trace.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b12x-fwd-57f3572-trace.yaml` | Bisect arm: b12x-fwd-57f3572 mod + startup trace mod | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-gemv.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-bf16-gemv.yaml` | BF16 GEMV kernel path arm | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-ghcr.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-ghcr-image.yaml` | Same as default, but container pulled from ghcr.io instead of a local build | fallback |
| recipes/eugr/eugr-agents-serve-local16-la-kk.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-kk-image.yaml` | Container built from the karmic-kraken-beta (KK) integration branch | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-lmq.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-mxfp8-lmhead.yaml` | Online MXFP8 verify lm_head; measurements folded into the default recipe | historical |
| recipes/eugr/eugr-agents-serve-local16-la-lprof.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-profiler-local-rank.yaml` | Profiling only: rank-local torch.profiler wrap, no collective RPC | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-occ_dynhigh.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-dyn-high.yaml` | B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS 64 | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-occ_dynlow.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-dyn-low.yaml` | B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS 32 | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-occ_microhigh.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-micro-high.yaml` | B12X_MICRO_MAX_ACTIVE_CLUSTERS 64 | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-occ_microlow.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-occupancy-micro-low.yaml` | B12X_MICRO_MAX_ACTIVE_CLUSTERS 32 | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-prof.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-profiler-config.yaml` | Profiling only: built-in --profiler-config for Phase C decode-step profiling | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-spec3.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-argmax-mtp3.yaml` | Argmax-reduction arm with num_speculative_tokens 3 (was 4) | experimental |
| recipes/eugr/eugr-agents-serve-local16-la-tc.yaml | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-tc45-reasoning-fix.yaml` | tool_choice=required structural-tag fix so forced grammar admits the <think> prefix | experimental |
| recipes/eugr/eugr-agents-serve-local16-la.yaml | `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml` | Default: local16 + probabilistic MTP draft sampling | default |
