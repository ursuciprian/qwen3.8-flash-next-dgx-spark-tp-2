# Patches

The same files exist twice: here, for reading, and under `mods/`, where a
`run.sh` copies them into the container before serve. Every recipe in this
registry that needs a patch references its mod, so nothing depends on a clone
path.

## `vllm-flashnext-nightly-8a728663`

Referenced by `recipes/flashnext-vllm-cached.yaml`. Six overlays for vLLM nightly `8a728663` serving the NVIDIA checkpoint
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

## Licensing

Every overlay is a modified copy of a file from vLLM, which is Apache-2.0, and
each keeps its upstream copyright header. The modifications are described above
and in `kv_cache_utils.diff`. This repository is Apache-2.0 (see `LICENSE`).
