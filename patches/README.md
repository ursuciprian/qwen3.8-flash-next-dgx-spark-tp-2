# Patches

Every file here is bind-mounted read-only over the matching file inside the
container. The recipes carry the mount lines; `scripts/run.sh` rewrites the
absolute source paths to wherever this repository is cloned, on both nodes.

## `qsa_upstream_fix.py` (SGLang, published recipe only)

Replaces `sglang/srt/layers/attention/qwen_sparse_attn_backend.py` on the
day-0 `lmsysorg/sglang:qwen38flashnext` image. On GB10 (SM121) the stock
backend selects a kernel path that does not exist for that architecture and
every decode crashes; this file is upstream's guard fix (sgl-project/sglang
#36556) applied to the image that predates it.

Do not mount it on the 2026-09-03 image (`lmsysorg/sglang:qwen38flashnext-sep03`)
used by the current recipes. That image ships a dedicated SM121 kernel
(sgl-project/sglang #36845); the guard fix would overwrite it with the older
path, which upstream found to corrupt contexts above roughly 95k tokens on SM121
(sgl-project/sglang #36806: token-0 output while HTTP still returns 200). The
current SGLang recipes therefore mount nothing.

## `vllm_ple_layer.py` (vLLM, published recipe only)

Replaces `vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py` on the
`vllm/vllm-openai:qwen38-flash-next` image. The RadixArk checkpoint is
ModelOpt NVFP4 but its PLE (n-gram embedding) shards are global-scale FP8 and
the checkpoint lists `*.ple.*` under `ignore`, so the stock gate never selects
the FP8 embedding path and weight loading fails on `ngram_embedding.weight_scale`.
Used together with `VLLM_PLE_FP8_CHECKPOINT=1`.

## `vllm-nightly-8a728663/` (vLLM, current recipe)

Six overlays for vLLM nightly `8a728663` serving the NVIDIA checkpoint
`nvidia/Qwen3.8-Flash-Next-NVFP4` at revision `fc694b54`:

| file | replaces | why |
|---|---|---|
| `ops_ple.py`, `ops_qsa.py`, `qsa.py` | `vllm/models/qwen4_exp/nvidia/...` | community SM121 fixes for the PLE and sparse-attention ops on this nightly |
| `platforms_interface.py` | `vllm/platforms/interface.py` | community SM121 platform fix on this nightly |
| `modelopt.py` | `vllm/model_executor/layers/quantization/modelopt.py` | MTP draft-layer weight loading for this checkpoint, plus a one-string alias: revision `fc694b54` renamed the MTP experts' quantization to `FP8_PB_WO`, the same 128x128 block layout as `FP8_BLOCK_SCALES`; without the alias loading dies at the last shard with `has no parameter 'w2_weight_scale_inv'` |
| `kv_cache_utils.py` | `vllm/v1/core/kv_cache_utils.py` | lets vLLM identify the MTP drafter's KV-cache group (the rule existed for DeepseekV4 only); without it every group is treated as a draft group and a warning says prefix-cache reuse is disabled. Upstream: vllm-project/vllm #55390. Diff: `kv_cache_utils.diff` |

The one flag that makes prefix caching reuse a prompt's first pass with the
drafter on, `"disable_eagle_block_drop": true`, is in the recipe, not a patch.
Mechanism, measurements and the stock-semantics alternative
(`--prefix-cache-retention-interval 3200`, which reuses one block less per
turn) are in the README.
