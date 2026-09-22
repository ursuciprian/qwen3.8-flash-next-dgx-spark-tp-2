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

## `sglang-radix-chunked-insert-fix`

Referenced by all three SGLang recipes. `MambaRadixCache.cache_unfinished_req`
inserts each prefill chunk's KV pages into the radix tree mid-prefill; a retract
or abort then frees those pages while the tree still points at them, and the
next request sharing the prefix decodes token id 0 forever, the `!!!!` loop
that shows up after one or two hours of agent sessions at 33k-145k context
(sgl-project/sglang#38319, found by andreasknopke). The mod ports the closed
upstream PR #38355 onto the pinned nightly: the chunked insert is skipped and
deferred to request completion. `SGLANG_DISABLE_CHUNKED_RADIX_INSERT=0`
restores stock behaviour. Idempotent, fail-closed on the four anchors.

## Experimental (not in the default recipe)

Newer vLLM/b12x-era mods, tried against the `la` baseline 2026-09-20/21.
None of these are wired into `eugr-agents-serve-local16-la.yaml`; see each
mod's own patch and the linked verdict for detail.

- `vllm-qwen38-bf16-gemv` — awaiting GPU A/B. Boot failed
  (`results/arms/gemv/verdict.md`): b12x's candidate-racing preparation
  raises `ValueError: candidate races require an activation-producing
  context` for the new `gemm.bf16_gemv` target before `/health` ever came up.
- `vllm-qwen38-lmhead-b12x` — superseded by `VLLM_MXFP8_LM_HEAD` (see main
  README, Known issues / fixes) unless quality needs a BF16 head instead of
  MXFP8.
- `vllm-qwen38-hc-mxfp8` — quality-gated arm pending.
- `vllm-gdn-deferred` — **not bootable yet.** The vLLM half of deferred GDN
  checkpoints: the b12x decode kernel keeps one base checkpoint plus per-token
  records instead of one full state snapshot per verified token, so the
  block-boundary state copies have to go through an accepted-prefix commit
  first. 2.66x less GDN state traffic on a kernel measured at 86% of the GB10's
  DRAM bandwidth, modelled at 13.5% -> ~6-7% of the c8 step. Needs a b12x wheel
  built from `feat/gdn-deferred-checkpoints`, which does not exist; until then
  the mod applies cleanly and `VLLM_GDN_DEFERRED_CHECKPOINTS` must stay unset.
  See `mods/vllm-gdn-deferred/README.md`.

## Licensing

Every overlay is a modified copy of a file from vLLM, which is Apache-2.0, and
each keeps its upstream copyright header. The modifications are described above
and in `kv_cache_utils.diff`. This repository is Apache-2.0 (see `LICENSE`).
