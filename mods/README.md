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

## `sglang-sm121-qsa-fp8kv`

Referenced by `recipes/flashnext-fp8kv-1m8.yaml`. One in-place edit to
`sglang/srt/layers/attention/qwen_sparse_attn_backend.py`: the two packed
gather scratch buffers are allocated in the query dtype instead of the KV
cache dtype. The only QSA decode kernel qualified for SM121 accepts BF16 only
and raises `unsupported SM121 QSA call: expected BF16 D=256, 12:1 GQA, ...`
when fp8 keys reach it, so with `--kv-cache-dtype fp8_e4m3` the scheduler died
on the first decode. With the scratch in the query dtype the extraction kernel
converts the selected keys on store, at most 2055 rows per sequence, and the
cache itself stays fp8. Idempotent and fail-closed: both call sites must match
exactly once or nothing is written. Correct while the KV scales are 1.0 (the
default without a calibration file). Upstream fixed the same class of problem
for the chunk-prefill kernel in sgl-project/sglang#38855; the paged decode
path was not covered.

## `vllm-qsa-fp8kv-pr55557`

Not referenced by a shipped recipe yet. It applies vllm-project/vllm#55557
(open) to a current vLLM nightly so `--kv-cache-dtype fp8_e4m3` works on the
Qwen4Exp QSA path without the six whole-file overlays above, which do not
survive on a newer engine (`split_with_sizes expects split_sizes to sum
exactly to 2560 ... got [512, 128]`). Measured 2026-09-11 on nightly
`e7edf17c`: 2,069,156 KV tokens against the pinned build's 2,048,795, decode
at parity, one tool-eval run at 90/100 with a single failure against the
pinned build's 100. It stays out of the shipped recipe until that repeats
clean. Dry-runs the diff and fails closed; skips once the gate is gone.

## Licensing

Every overlay is a modified copy of a file from vLLM, which is Apache-2.0, and
each keeps its upstream copyright header. The modifications are described above
and in `kv_cache_utils.diff`. This repository is Apache-2.0 (see `LICENSE`).
