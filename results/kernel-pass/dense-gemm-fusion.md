# Non-MoE decode GEMM: measured inventory, and why the fusion lead is dead

> **Workload.** Unless a line says otherwise, every tok/s, c1/c8/c16 and
> ms/step figure in this file is the `tools/tony-bench/bench_sweep.py`
> counting diagnostic: "List the numbers from 1 to 300 separated by commas…",
> temperature 0, thinking off, non-streaming, 320 max tokens, fresh context,
> aggregate tok/s (c1 = per-stream). MTP accepts ~4 of 4 drafts on it, so it is
> a speculative-decoding ceiling, not coding or chat speed. Agent-coding and
> prose numbers: top-level `README.md`.

Date: 2026-09-21. No GPU runs. Every number below comes from the checkpoint
safetensors headers, `hf_quant_config.json`, the vLLM fork at `8e1f1e58`
(serving), b12x `a8333658`, or a direct re-analysis of the saved rank0 decode
trace `results/profiling_c1test_rank0_diag/rank0.json` (98 c1 decode steps,
494 MB chrome trace, kernel events grouped by **CUDA grid dimensions**, which
identify each GEMM's `N` and split-K exactly).

**Verdict up front, in the order it matters:**

1. **The fusion lead in the brief does not exist.** It assumed the 36 GDN
   layers run five separate projections (`in_proj_qkv`, `in_proj_z`,
   `in_proj_a`, `in_proj_b`, `out_proj`) against the same activation. The fork
   already merges them: `in_proj_qkvz` is one `MergedColumnParallelLinear`
   over the checkpoint's `in_proj_qkv` + `in_proj_z`, and `in_proj_ba` is one
   more over `in_proj_b` + `in_proj_a`. Two input GEMMs per layer, not five.
   Concatenating those two would remove **36** launches/step, not 72.
2. **And those 36 launches cost ~0 wall time already.** `in_proj_ba` is
   9.95 us on the **aux stream**, overlapped with the 119.82 us `in_proj_qkvz`
   on the main stream (`VLLM_QWEN3_8_FLASH_NEXT_OVERLAP`, default 1,
   `qwen_gdn_input_projections`). Fusing them also forces either
   non-contiguous slices or two extra copies into the packed GDN core op.
   **No patch was written for it** -- see section 5.
3. **The 251 `cutlass_80_wmma` BF16 launches/step are not projections at all.**
   They are 194 hyper-connection mixer GEMMs + 48 MoE router gates + the
   lm_head + MTP. That bucket is **13.76 ms/step of a 55.03 ms step (25%)**,
   and it runs on a *one-warp-per-CTA SM80 WMMA* kernel that cuBLAS falls back
   to at M=5, at 53-64% of the LPDDR5x roofline.
4. **b12x already ships the kernel for this**, `gemm.bf16_gemv`, and the fork
   cannot reach it: the only route is `B12XFp6Config.get_quant_method`, which
   a modelopt `MIXED_PRECISION` checkpoint never instantiates. That is the
   patch in this pass: `patches/vllm-qwen38-bf16-gemv.patch`.

---

## 1. GEMM inventory per decode step (c1, M=5, TP=2, per node)

Weight dtypes are read off the safetensors headers; the quant group is from
`hf_quant_config.json` (`quant_algo: MIXED_PRECISION`, `ignore: []`, so
anything not in a group is plain BF16). `bytes/launch` is weight + weight-scale
traffic for the node's shard. `floor` is bytes / 273 GB/s (LPDDR5x spec; real
achievable is ~80-88% of that, so every efficiency column below is a lower
bound on how close the kernel already is).

### Linear-attention layers (36 of 48)

| projection | source tensors | N/node | K | dtype | kernel (grid) | bytes | launch/step | us | ms/step | floor us | eff |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `in_proj_qkvz` | `in_proj_qkv[10240,2560]` + `in_proj_z[6144,2560]` | 8192 | 2560 | MXFP8 g32 | b12x DenseGemm (1,1,128) | 21.63 MB | 36 | 119.82 | **4.18** | 79.2 | 66% |
| `in_proj_ba` | `in_proj_b[48,2560]` + `in_proj_a[48,2560]` | 48 | 2560 | MXFP8 g32 | b12x DenseGemm (1,8,1) | 0.127 MB | 36 | 9.95 | 0.35\* | 0.46 | 5% |
| `out_proj` | `out_proj[2560,6144]` | 2560 | 3072 | MXFP8 g32 | b12x DenseGemm (1,1,40) | 8.11 MB | 36 | ~45 | ~1.6 | 29.7 | ~66% |

\* on the aux stream, concurrent with `in_proj_qkvz`: **not** 0.35 ms of wall.

### Full-attention layers (12 of 48)

| projection | source tensors | N/node | K | dtype | kernel (grid) | bytes | launch/step | us | ms/step | floor us | eff |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `qkv_proj` | `q[12288,2560]`+`k[512,2560]`+`v[512,2560]` | 6656 | 2560 | MXFP8 g32 | b12x DenseGemm (1,4,104) | 17.57 MB | 12 | 101.99 | **1.19** | 64.4 | 63% |
| `o_proj` | `o_proj[2560,6144]` | 2560 | 3072 | MXFP8 g32 | b12x DenseGemm (1,1,40) | 8.11 MB | 12 | ~45 | ~0.54 | 29.7 | ~66% |
| `indexer.index_qk_proj` | `[640,2560]` | 640 | 2560 | MXFP8 g32 | b12x DenseGemm (1,1,10) | 1.69 MB | 12 | ~12.3 | ~0.15 | 6.2 | ~50% |

### All 48 layers

| projection | source tensors | N/node | K | dtype | kernel (grid) | bytes | launch/step | us | ms/step | floor us | eff |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `*_hyper_connection.input_mix_weight_up` | `[10240,320]` BF16 | 10240 | 320 | **BF16** | **cutlass_80_wmma 16x16 (8,80,1)** | 6.55 MB | 97.9 | 45.23 | **4.43** | 24.0 | **53%** |
| `*_hyper_connection.input_mix_weight_down_block_inject` | `input_mix_weight_down[320,10240]` + `block_inject_weight[4,10240]` (+12 pad rows) | 336 | 10240 | **BF16** | **cutlass_80_wmma 16x16 (8,3,9) + splitKreduce** | 6.88 MB | 96.9 | 39.09 | **3.79** | 25.2 | **64%** |
| `mlp.gate` (router) | `[512,2560]` BF16 | 512 | 2560 | **BF16** | **cutlass_80_wmma 16x16 (8,4,8)** | 2.62 MB | 47.5 | 17.82 | **0.85** | 9.6 | **54%** |
| `mlp.shared_expert.gate_up_proj` | `gate_proj[640,2560]`+`up_proj[640,2560]` | 640 | 2560 | MXFP8 g32 | b12x DenseGemm (1,1,10) | 1.69 MB | 48 | ~12.3 | ~0.59 | 6.2 | ~50% |
| `mlp.shared_expert.down_proj` | `[2560,640]` | 2560 | 320 | MXFP8 g32 | b12x DenseGemm (1,1,40) | 0.84 MB | 48 | ~7 | ~0.34 | 3.1 | ~44% |
| `mlp.experts` (routed) | 512 x {gate,up,down} | -- | -- | NVFP4 g16 | `siluMoEDynamicKer` | ~69 MB | 48 | 313.02 | **15.02** | 253 | 81% |

The two HC rows cover both `attn_hyper_connection` and `mlp_hyper_connection`
in all 48 layers **plus** the model-level `hyper_connection_mixer`
(48 x 2 + 1 = 97, matching the measured 96.9 / 97.9 per step).

### Head and MTP

| item | N/node | K | dtype | kernel (grid) | bytes | launch/step | us | ms/step |
|---|---:|---:|---|---|---:|---:|---:|---:|
| `lm_head` (verify, M=5) | 124160 | 2560 | **BF16** | **cutlass_80_wmma (8,970,1)** | 636 MB | 1 | 3711 | **3.60** |
| MTP draft head (M=1, x4) | 124160 | 2560 | b12x route (`VLLM_MTP_NVFP4_LM_HEAD=1`) | b12x DenseGemm (1,4,970) | -- | 4 | 719 | **2.88** |
| MTP layer q/k/v/o (BF16, unquantized in ckpt) | -- | -- | BF16 | `cublas gemvx` x5 groups | -- | ~18 | 4.6-217 | 1.21 |
| MTP HC down + up | 336 / 10240 | 10240 / 320 | BF16 | cutlass (8,3,5) / 32x32 (8,40,1) | 6.9 / 6.55 MB | 9 + 9 | 37.6 / 38.6 | 0.68 |
| `mtp.fc_embedding` + `fc_hidden` | 2560 | 2560 | BF16 | cutlass (8,20,1) | 13.1 MB | 2.9 | 61.2 | 0.18 |
| PLE `key_proj` + `value_proj` (layer 2 only) | 10240 / 2560 | 2560 | BF16 | cutlass | 52.4 / 13.1 MB | 2 | 297 / 82.6 | 0.38 |
| MTP MoE (`w4a16`) | -- | -- | NVFP4 W4A16 | `w4a16kernel` | -- | 4 | 138.7 | 0.55 |

### Roll-up

| bucket | launches/step | ms/step | % of 55.03 ms step |
|---|---:|---:|---:|
| MoE routed experts | 48 | 15.02 | 27.3% |
| **BF16 on `cutlass_80_wmma`** | **251** | **13.76** | **25.0%** |
| MXFP8 on b12x DenseGemm | 243 | 9.16 | 16.6% |
| b12x MTP draft head | 4 | 2.88 | 5.2% |
| BF16 on cuBLAS `gemvx` (MTP) | 18 | 1.21 | 2.2% |
| MoE W4A16 (MTP) | 4 | 0.55 | 1.0% |
| split-K reduces, nvjet, misc | ~210 | ~1.1 | 2.0% |
| **all GEMM** | **~780** | **43.7** | **79%** |

Launch-count cross-check against the brief: 251 cutlass + 243 b12x dense = 494,
and the brief's "~251 cutlass + b12x linear launches" matches exactly. What the
brief mis-attributed is *which* GEMMs those 251 are: **not** the linear-attn
projections (all MXFP8, all on b12x) but the hyper-connection mixers.

---

## 2. Launch overhead vs bandwidth

At M=5 nothing in this model is launch-*bound* in the host sense -- GPU idle is
5.4% of the step (2.97 ms spread over ~1500 launches, `profiling/README.md`).
The gap is per-kernel efficiency, not gaps between kernels:

| group | ms/step | floor @273 GB/s | gap | what the gap is |
|---|---:|---:|---:|---|
| HC up `[10240,320]` | 4.43 | 2.35 | **2.08** | K=320 is 2.5 k-tiles of 128; 640 single-warp CTAs, no split-K, short pipeline |
| GDN `in_proj_qkvz` | 4.18 | 2.77 | 1.41 | b12x MXFP8, already the best route in tree |
| HC down `[336,10240]` | 3.79 | 2.44 | 1.35 | 21 N-tiles x 9 split-K slices = 24 CTA columns; needs a splitKreduce pass |
| `lm_head` M=5 | 3.60 | 2.33 | 1.27 | see section 4 |
| MTP draft head | 2.88 | -- | -- | different (faster) route already |
| out_proj/o_proj/shared down | 2.43 | ~1.2 | ~1.2 | b12x MXFP8 |
| MTP BF16 gemv | 1.21 | ~0.4 | ~0.8 | cuBLAS gemvx on unquantized MTP weights |
| `qkv_proj` | 1.19 | 0.75 | 0.44 | b12x MXFP8 |
| router gate | 0.85 | 0.46 | 0.39 | 32 N-tiles x 8 split-K |
| shared gate_up + indexer | 0.72 | ~0.35 | 0.37 | b12x MXFP8 |
| MTP HC down + up | 0.68 | 0.35 | 0.33 | same kernel as the main HC |
| GDN `in_proj_ba` | 0.35 | 0.01 | 0.34 | **already hidden on the aux stream** |
| **non-MoE GEMM total** | **~28** | **~18** | **~10** | |

Ten ms of headroom exists, but **none of it is fusable**: every item above is a
separate kernel against a separate weight matrix with a *different* input, or
is already fused.

### "Same input, multiple weights" -- the complete list

| site | weights on the same activation | already fused? |
|---|---|---|
| GDN layers | `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` | **yes, into 2**: `MergedColumnParallelLinear` `in_proj_qkvz` (output_sizes `[2048,2048,6144,6144]`) and `in_proj_ba` (`[48,48]`) |
| full-attn | `q_proj`, `k_proj`, `v_proj` | **yes**: `QKVParallelLinear`, one `[13312,2560]` GEMM (grid (1,4,104) confirms N/node = 6656) |
| full-attn | `qkv_proj` + `indexer.index_qk_proj` | no -- 12 launches, ~12 us each, worth **0.15 ms/step (0.27%)** |
| shared expert | `gate_proj`, `up_proj` | **yes**: `MergedColumnParallelLinear` |
| HC mixer | `input_mix_weight_down` + `block_inject_weight` | **yes**: `MergedColumnParallelLinear` `input_mix_weight_down_block_inject` with `output_sizes [320, 4, 12]` (12 = alignment pad) -- and `_HC_WEIGHTS_MAPPER` in `model.py` already maps the two checkpoint tensors onto shards 0 and 1 |
| HC mixer | `input_mix_weight_up` | not fusable: its input is the post-SiLU bottleneck, a *different* activation |
| MoE block | `mlp.gate` + `mlp.shared_expert_gate` | `shared_expert_gate` is `[1,2560]`; no 48/step launch for it appears in the trace, so it is not a distinct GEMM on this path |

**Total remaining bit-identical N-concatenation value: `in_proj_ba` into
`in_proj_qkvz` (~0 wall, already overlapped) plus the indexer (0.15 ms).
Under 0.3% combined.** That is the answer to the fusion question.

---

## 3. The lever that is actually there: BF16 small-M routing

All three of the big BF16 GEMMs land on
`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8`,
`block=(32,1,1)`: cuBLAS's **one-warp, 16x16 tile, SM80 WMMA** fallback for
tiny M. On SM121 this is a legacy path, and it measures 53-64% of *spec*
bandwidth on shapes that are pure weight streaming.

b12x ships a purpose-built replacement, `b12x/gemm/bf16_gemv`
("Prepared CuTe small-N BF16 GEMV for exact decode-sized projections"), whose
own routing comment names the target: *"Unquantized bf16 linears with
N <= MAX_OUT and K >= MIN_IN are worth routing through this small-N GEMV:
catches narrow projections like the GDN `in_proj_ba` while excluding lm_head
and anything wide enough that cuBLAS tiles efficiently."*
(`SMALL_N_GEMV_MAX_OUT=1024`, `SMALL_N_GEMV_MIN_IN=1024`, `SMALL_M_MAX=8`.)

**The fork cannot reach it.** The only caller is
`B12XFp6Config.get_quant_method` in `b12x/integration/vllm/plugin.py`, which is
only constructed for the `b12x_fp6` quantization config. This model is served
`--quantization modelopt_mixed`, so every unquantized BF16 linear falls through
to `UnquantizedLinearMethod` -> `torch.nn.functional.linear` -> cuBLAS.
`grep -rn bf16_gemv` over the fork returns only `vllm/models/deepseek_v4_1/`,
which wires it up by hand for exactly this reason.

Two of our three big BF16 shapes are inside b12x's own thresholds:

| target | shape | N <= 1024 | K >= 1024 | ms/step | floor | eligible |
|---|---|---|---|---:|---:|---|
| `hc` HC down/block-inject | `[336, 10240]` | yes | yes | 3.79 | 2.44 | **yes** |
| `gate` MoE router | `[512, 2560]` | yes | yes | 0.85 | 0.46 | **yes** |
| `hcup` HC up | `[10240, 320]` | no | no | 4.43 | 2.35 | routing heuristic says no; `validate_query` accepts it -- opt-in arm |

### Expected gain -- state it honestly

`hc` + `gate` are 4.64 ms/step measured against a 2.90 ms roofline floor.

| assumption for bf16_gemv | ms/step | saved | tok/s (from 98.8) | delta |
|---|---:|---:|---:|---:|
| 100% of spec bandwidth (impossible) | 2.90 | 1.74 | 102.0 | +3.3% |
| 85% of spec | 3.41 | 1.23 | 101.0 | +2.3% |
| 75% of spec | 3.87 | 0.77 | 100.2 | +1.4% |
| no better than cuBLAS | 4.64 | 0.00 | 98.8 | 0% |

**`hc,gate` alone cannot clear a +5% bar -- the roofline caps it at +3.3%.**
Adding `hcup` (another 2.08 ms of gap, +3.9% at the roofline) is what would
take the pair past +5%, and `hcup` is unmeasured: b12x's own heuristic
excludes N=10240, so its `simt` backend at `rows_per_tile=8` may well lose to
cuBLAS's 640-CTA tiling there. It is a one-env-var A/B, not a guess to ship.

The one piece of in-tree evidence that b12x's BF16 route can be much faster
than this cuBLAS kernel: for the *same* `[*, 2560]` BF16 vocab projection, the
MTP draft head runs a b12x DenseGemm at **719 us** while the main verify head
runs `cutlass_80_wmma (8,970,1)` at **3711 us**. That comparison is confounded
(M=1 vs M=5, and `VLLM_MTP_NVFP4_LM_HEAD=1` likely means a quantized draft
head), so it is a reason to measure, not a prediction.

---

## 4. Two adjacent findings, not patched here

**(a) `lm_head` at M=5 costs 3.60 ms/step (6.5%).** `LogitsProcessor` already
has a `bf16_vocab_projection` route (`use_b12x_vocab_projection`, gated on
`linear_backend == "b12x"`, which this recipe sets), yet the verify-step head
measures as a cuBLAS `cutlass_80_wmma (8,970,1)` launch. Either
`get_b12x_bf16_vocab_projection()` returns None in this image, or
`is_supported()` is false, or `lm_head.quant_method` is not one of the two
accepted unquantized methods under `modelopt_mixed`. **One log line at boot
settles it** and it is a bigger single item than the router gate:

```python
# vllm/model_executor/layers/logits_processor.py, after use_b12x_vocab_projection
logger.info("b12x vocab projection: enabled=%s module=%s",
            self.use_b12x_vocab_projection, self._b12x_vocab_projection)
```

**(b) The hyper-connection weights are fully replicated across TP ranks.**
`input_mix_weight_up` is a `ReplicatedLinear` and
`input_mix_weight_down_block_inject` passes `disable_tp=True`, so each node
reads **1.29 GB** of HC weights per decode step -- more than the linear-attn
projections (1.08 GB/node) and the single largest dense weight stream in the
model. Column-sharding them would halve that, but the `gate_mean` consumer
needs the full `hc_count * hidden = 10240` width, so it costs an all-gather per
HC (194/step) on a 2-node ROCE fabric where a one-shot all-reduce already
measures 15-19 us. Not worth it as proposed; noted because the earlier census
in `moe-decode-analysis.md` put hyper-connections at "~315 MB", which counted
only `input_mix_weight_down` of one of the two HCs and then TP-halved a tensor
that is not sharded. The correct figure is **4x that, unsharded**.

---

## 5. Artifacts, and one that was deliberately not written

- `patches/vllm-qwen38-bf16-gemv.patch` -- git-format patch on `8e1f1e58`,
  two files, +239 lines. Binds `b12x.gemm.bf16_gemv` to the HC
  down/block-inject projection and the MoE router gate, with the HC
  up-projection available as an opt-in target.
- `mods/vllm-qwen38-bf16-gemv/` -- sparkrun mod for image
  `spark-vllm-b12x:local-20260918-a8333658`. SHA256 pre-image check on both
  files (`34f9ff5d...`, `7485ac00...`, verified byte-identical between
  `git show 8e1f1e58:` and the installed package in the image), `patch
  --dry-run` first, SHA256 post-image check, `ast.parse`, then an import smoke
  test that asserts the custom op registered. Fails closed on every step; a
  second application is a no-op.
- `scripts/logits_equiv.py` -- `capture` / `diff` / `selftest`. 20 fixed
  prompts, `/v1/completions`, `temperature=0`, `logprobs=5`, `max_tokens=16`;
  `diff` fails on any changed sampled token or any top-5 logprob moving more
  than `--tol` (default 1e-3). Two phases because only one server can own the
  pair at a time.
- **`patches/vllm-qwen38-gemm-fusion.patch` -- not written, deliberately.**
  Sections 1 and 2: the only remaining bit-identical N-concatenation is
  `in_proj_ba` into `in_proj_qkvz`, whose 36 launches/step already run
  concurrently on the aux stream and cost ~0 wall time; the concatenated
  output would then have to be re-split for
  `qwen_gdn_attention_core_fused_norm_packed`, either as non-contiguous views
  or via two extra copy kernels per layer. Measured ceiling under 0.6%,
  plausible outcome negative, and it would cost a full quality gate to
  qualify. Re-open it only if `VLLM_QWEN3_8_FLASH_NEXT_OVERLAP=0` ever becomes
  the default, or if a future GDN core op wants one packed tensor anyway.

### Design of the patch

`SmallNBF16LinearMethod(UnquantizedLinearMethod)` in `hyperconnection.py`,
ported from the shape of `vllm/models/deepseek_v4_1/b12x_layers.py`
(`B12xLinearMethod`) and `b12x/integration/vllm/plugin.py`
(`_VllmSmallNBF16Method`), with three deliberate differences:

- **No weight clone.** The plugin does `weight.data.detach().clone()`; our
  weights are already contiguous 16B-aligned BF16, and cloning 97 x 6.88 MB
  would cost ~670 MB of unified memory at `gpu_memory_utilization: 0.80`.
- **All non-static guards live inside the custom op.** `apply()` does no
  `data_ptr()` or `rows in plans` test, because the HC mixer is inside the
  compiled/CUDA-graph region and those would break Dynamo tracing. The op
  `torch.ops.vllm.qwen38_small_n_bf16_linear` runs eagerly, looks up the plan
  for the real M, and returns `torch.nn.functional.linear` when there is none.
  **Prefill and c8 decode (M=40 > `SMALL_M_MAX=8`) are therefore
  bit-identical to stock**; only c1-class decode changes.
- **Rebound at construction, before load.** `maybe_route_small_n_bf16` swaps
  `quant_method` in `GatedResidual.__init__` and
  `Qwen3_8FlashNextSparseMoeBlock.__init__`, so the replacement's
  `process_weights_after_loading` runs normally and registers the b12x
  preparation provider. **Weight loading is untouched**: the parameter layout,
  the `MergedColumnParallelLinear` shard loaders and `_HC_WEIGHTS_MAPPER` all
  see exactly what they saw before, because `create_weights` still comes from
  `UnquantizedLinearMethod`. Neither target is TP-sharded
  (`disable_tp=True` / `ReplicatedLinear`) and neither is quantized, so there
  are no per-block scales to concatenate anywhere in this patch.
- **CUDA-graph capture**: shapes per step are unchanged; the op allocates its
  output through the caching allocator inside the capture, the same as every
  other b12x call on this path. Plans are declared for every
  `workload.token_counts` value in `1..SMALL_M_MAX` during the weights
  preparation stage, i.e. before capture.
- **MTP head**: `mtp.py` builds its own `GatedResidual`s and its own
  `mlp.gate`, so the MTP layer picks the route up through the same code with
  no extra change. Its HC weights are the same `[336,10240]` / `[10240,320]`
  BF16 shapes and its M is 1, comfortably inside `SMALL_M_MAX`.

### Risks

| risk | mitigation |
|---|---|
| Not bit-identical -- different accumulation order in the GEMV | full quality gate + `logits_equiv.py`; the router gate feeds top-k expert selection, which is discrete, so a tiny logprob shift there can flip an expert |
| bf16_gemv slower than cuBLAS for these shapes | `VLLM_QWEN38_BF16_GEMV=off` reverts without a rebuild; per-target bisect with `hc` / `gate` |
| b12x plan preparation adds boot time / a plan-cache miss | 3 targets x <=8 M values; the plan cache already mounts via the recipe's `/tmp/.cache` volume, but **expect one slower first boot** while `B12X_AUTOTUNE=1` races them |
| Dynamo/AOT compile chokes on the new custom op | `direct_register_custom_op` with a fake impl, same pattern as `qwen_gdn_input_projections`; the import smoke test in the mod catches registration failures before serving |
| Pad rows (12 of the 336) are never loaded and may be non-finite | unchanged behaviour: cuBLAS computed them too and the caller slices `[:, :320]` and `[:, 320:324]`; the GEMV has no cross-N reduction |
| `hcup` regresses | not in the default target list |

---

## 6. Validation recipe for the GPU worker

Two boots. Keep `VLLM_QWEN3_8_FLASH_NEXT_OVERLAP` at its default.

**Boot A -- baseline, no mod** (`recipes/eugr/eugr-agents-serve-local16-la.yaml`):

```bash
cd ~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
sparkrun run recipes/eugr/eugr-agents-serve-local16-la.yaml --no-follow
# after health:
for i in 1 2 3; do ( cd tools/tony-bench && python3 bench_sweep.py \
  http://localhost:8000 qwen3.8-flash-next base_c1_$i --levels 1 ); done
python3 scripts/logits_equiv.py capture --out /tmp/logits_base.json
```

**Boot B -- fused**: copy the recipe to
`recipes/eugr/eugr-agents-serve-local16-la-gemv.yaml`, add
`vllm-qwen38-bf16-gemv` to `mods:`, then:

```bash
sparkrun run recipes/eugr/eugr-agents-serve-local16-la-gemv.yaml --no-follow
# confirm the mod ran: the boot log must contain
#   "mod vllm-qwen38-bf16-gemv: applied" and "import smoke ok; targets = ['gate', 'hc']"

for i in 1 2 3; do ( cd tools/tony-bench && python3 bench_sweep.py \
  http://localhost:8000 qwen3.8-flash-next gemv_c1_$i --levels 1 ); done
( cd tools/tony-bench && python3 bench_sweep.py \
  http://localhost:8000 qwen3.8-flash-next gemv_c8 --levels 8 )

python3 scripts/logits_equiv.py capture --out /tmp/logits_gemv.json
python3 scripts/logits_equiv.py diff /tmp/logits_base.json /tmp/logits_gemv.json --tol 1e-3

bash scripts/gate_arm.sh gemv-hc-gate
```

**Pass criteria** (all must hold; this is a model-code change, so the quality
gate is not optional):

| check | bar |
|---|---|
| `bench_sweep --levels 1`, median of 3 | **>= +2%** vs boot A's median. The roofline caps this arm at +3.3%, so the brief's +5% is not reachable with `hc,gate` alone -- treat +2% as ship, < +1% as revert. |
| `bench_sweep --levels 8` | no regression (the GEMV does not engage at M=40; any delta is noise or a real problem) |
| `logits_equiv.py diff` | identical sampled tokens; top-5 logprobs within 1e-3. A top-5 *set* change on any prompt is a fail. |
| `fidelity_probe` 8k/32k/64k/128k | **100 x 4** |
| tool-eval hardmode | **>= 88** |
| `straggler_probe` 5 6 7 8 12 16 | clean, no ~18 s c5-7 stall |
| MTP acceptance (`/metrics`) | within noise of the boot-A numbers (a router-gate flip shows up here first) |

**If it passes but under +2%:** bisect with `VLLM_QWEN38_BF16_GEMV=hc` and
`=gate` in the recipe env (no rebuild, no remod), then run the `hcup` arm:
`VLLM_QWEN38_BF16_GEMV=hc,gate,hcup`. `hcup` is the largest single gap in the
table (2.08 ms/step, +3.9% at the roofline) and is the only thing in this pass
that can take the total past +5%.

**If `logits_equiv` fails on the router gate only:** re-run with
`VLLM_QWEN38_BF16_GEMV=hc`. The HC path feeds a low-rank bottleneck through
SiLU and a mean-gate, which is far less sensitive than argmax expert routing.

---

## 7. What was already verified without a GPU

Run inside `spark-vllm-b12x:local-20260918-a8333658` (`docker run --rm`, no
`--gpus`, no serving touched):

```
$ docker run --rm -v .../mods:/modsro:ro --entrypoint bash <image> \
    -c 'cp -r /modsro/vllm-qwen38-bf16-gemv /tmp/m && bash /tmp/m/run.sh'
patching file vllm/models/qwen3_8_flash_next/hyperconnection.py
patching file vllm/models/qwen3_8_flash_next/model.py
ast ok
import smoke ok; targets = ['gate', 'hc']
mod vllm-qwen38-bf16-gemv: applied (HC down/block-inject + MoE router gate -> b12x bf16_gemv)
EXIT=0
```

- the installed `hyperconnection.py` / `model.py` in the image are
  **byte-identical** to `git show 8e1f1e58:` (same SHA256), so the patch is cut
  against the right pre-image;
- second run: `already applied, skipping`, exit 0 (idempotent);
- after corrupting one target file: `pre-image mismatch -- refusing to patch`,
  exit 1 (fail-closed);
- `torch.ops.vllm.qwen38_small_n_bf16_linear` registers and both modules
  import.

`scripts/logits_equiv.py selftest` passes (comparator checked against
identical, within-tolerance, beyond-tolerance, different-token,
different-top-5-set and missing-record cases).

**Still unverified, and only a GPU boot can settle it:** whether
`b12x.gemm.bf16_gemv` is actually faster than `cutlass_80_wmma` for
`[336,10240]` and `[512,2560]` at M=5, and whether the resulting logprobs stay
inside 1e-3.
