# The verify head and the hyper-connection weight stream

> **Workload.** Unless a line says otherwise, every tok/s, c1/c8/c16 and
> ms/step figure in this file is the `tools/tony-bench/bench_sweep.py`
> counting diagnostic: "List the numbers from 1 to 300 separated by commas…",
> temperature 0, thinking off, non-streaming, 320 max tokens, fresh context,
> aggregate tok/s (c1 = per-stream). MTP accepts ~4 of 4 drafts on it, so it is
> a speculative-decoding ceiling, not coding or chat speed. Agent-coding and
> prose numbers: top-level `README.md`.

Date: 2026-09-21. No GPU runs. Sources: the vLLM fork at `8e1f1e58` (serving),
b12x at `a8333658`, the saved rank0 decode trace
(`results/profiling_c1test_rank0_diag/rank0.json`, 98 c1 steps),
`results/profiling/README.md`, the boot log `results/serve-la-final.log`, the
b12x plan cache under
`~/.cache/sparkrun/runtime-cache/vllm/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-ca6f25af/b12x/compile/preparation/`,
and CPU-only `docker run --rm` evaluation of both b12x tuning contracts inside
`spark-vllm-b12x:local-20260918-a8333658`.

**Verdicts up front:**

| item | verdict | expected |
|---|---|---|
| **A** `lm_head` verify GEMM | **patched** -- `patches/vllm-qwen38-lmhead-b12x.patch` | **+0.9% to +2.3%**, not the +5% in the brief -- see §2 |
| **A2** MXFP8 verify head | **recipe env var, no code** | **+4.3%**, quality-gated, mutually exclusive with A |
| **B** HC TP=2 sharding | **not sound, no patch** -- §4 | best case +2.2%, realistic 0 to negative |
| **B2** HC/router online MXFP8 | **patched** -- `patches/vllm-qwen38-hc-mxfp8.patch`, §5 + §B2-implementation | **+7.8%** from bytes alone, quality-gated, mutually exclusive with `vllm-qwen38-bf16-gemv` |

---

## 1. A: why the verify head does not get a b12x kernel

The route is not blocked. Every gate in `LogitsProcessor` passes:

- `kernel_config.linear_backend == "b12x"` -- the recipe passes `--linear-backend b12x`.
- `get_b12x_bf16_vocab_projection()` returns the module and `is_supported()` is true.
- `Qwen3_8FlashNextForCausalLM.__init__` constructs `LogitsProcessor(config.vocab_size, lm_head=self.lm_head)`, and `_units_from_modules` (`vllm/model_executor/warmup/b12x_prepare.py:240`) re-binds the head at the `weights` stage after loading, when the weight is real, on CUDA, BF16, 2-D and contiguous.
- `ParallelLMHead(config.vocab_size, config.hidden_size, prefix=...)` is built with **no `quant_config` and no `lm_head_quantization`**, and `VLLM_MXFP8_LM_HEAD` defaults to `False`, so `quant_method` stays `UnquantizedEmbeddingMethod` -- which is exactly what both `prepare_b12x_vocab_projection` and `_apply_head` require.

Proof it is registered and primed, from the boot log:

```
results/serve-la-final.log:247
b12x priming gemm.bf16_vocab_projection: 237/349 ready, candidates 0/1 prepared, ...
```

`get_b12x_preparation_units` only emits a `VOCAB_PROJECTION` unit when a head
made it into `_b12x_vocab_heads`. So `_apply_head` **does** take the b12x
branch at the verify step.

**The branch is a dead end above M = 1.** `b12x/gemm/bf16_vocab_projection/_tuning.py`
admits its Triton backend only for `max_tokens == 1`, in all three places that
matter -- `_default_config`, `_validate_config` ("the Triton vocabulary GEMV
requires max_tokens=1") and `_tuning_parameters`, whose backend tuple collapses
to `("torch",)`. `_preparation.py:materialize` then sets
`runner = torch.nn.functional.linear` for `backend == "torch"`. The b12x call
is a wrapper around cuBLAS.

Evaluated on CPU against the real head geometry (N = 124160 per node,
K = 2560):

```
--- gemm.bf16_vocab_projection ---
 max_tokens=1   default=backend='triton', algorithm='row', block_k=4096, num_warps=8   candidates ('torch','triton')
 max_tokens=2   default=backend='torch'                                                candidates ('torch',)
 max_tokens=5   default=backend='torch'                                                candidates ('torch',)
 max_tokens=8   default=backend='torch'                                                candidates ('torch',)
--- gemm.bf16_gemv, same head ---
 max_rows=2..16  default=GemvConfig(backend='simt', rows_per_tile=8)
 max_rows=24+    default=GemvConfig(backend='mma',  rows_per_tile=8)
```

The plan cache agrees. Of 417 records, exactly **one** has the vocabulary
projection's config schema:

```
{'algorithm': 'row', 'backend': 'triton', 'block_k': 4096, 'num_warps': 4}
```

That is the M=1 plan -- `block_k = 4096 = next_pow2(2560)`. The torch backend
compiles nothing, so no other capacity leaves a record, which is also what
`candidates 0/1 prepared` in the boot line means. At an MTP verify step
`_b12x_vocab_plan_for(rows=5)` selects the smallest prepared capacity >= 5,
gets a torch plan, and the step runs
`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` at
grid (8, 970, 1) -- 970 = 124160/128 -- for 3.60 ms.

Widening the Triton kernel in place is not the fix: `_row_kernel` launches
grid `(N, rows)` with one program per *(vocab row, token row)* pair, so it
re-reads all 636 MB of weights **per token**. It is a true GEMV and is
correctly restricted to M=1.

### 1a. The 719 us MTP draft head is not evidence of a faster BF16 route

```
results/serve-la-final.log:222
INFO [vocab_parallel_embedding.py:417] Quantizing LM head shards to NVFP4 with BF16 activations.
```

`mtp.py` builds its own head with `lm_head_quantization="nvfp4"` because
`VLLM_MTP_NVFP4_LM_HEAD` defaults to 1, so `quant_method` becomes
`Nvfp4OnlineLinearMethod(use_a16=True)`. Its weights are 4-bit:

| head | weights | scales | total | floor @273 GB/s | measured | efficiency |
|---|---:|---:|---:|---:|---:|---:|
| MTP draft, NVFP4 g16 W4A16 | 158.9 MB | 19.9 MB (e4m3) | **178.8 MB** | 655 us | **719 us** | **91%** |
| main verify, BF16 | 635.7 MB | -- | **635.7 MB** | 2330 us | **3600 us** | **65%** |

The draft head is fast because it streams 3.6x fewer bytes, at 91% of spec.
**A BF16 verify head cannot go below 2.33 ms**, so the brief's "up to ~2.6 ms
saved" is unreachable by routing alone. The honest ceiling is 1.27 ms.

### 1b. Does `VLLM_MTP_NVFP4_LM_HEAD` touch the verify path?

No. The two heads are separate `ParallelLMHead` modules owned by separate
`LogitsProcessor` instances. The NVFP4 draft head fails the
`isinstance(quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod))`
test, so it never registers a vocabulary plan and never reaches `_apply_head`'s
b12x branch -- it goes through `Nvfp4OnlineLinearMethod.apply` to a b12x
blockscaled DenseGemm, grid (1, 4, 970). The only interaction runs the other
way: setting `VLLM_MTP_NVFP4_LM_HEAD=0` would make the draft head BF16, land it
on the same vocabulary route at M=1 (where the Triton GEMV *is* prepared), and
cost 4 x 636 MB per step. Leave it at 1.

---

## 2. A: the patch, and what it is worth

`patches/vllm-qwen38-lmhead-b12x.patch` (on `8e1f1e58`, two files,
`vllm/utils/b12x.py` + `vllm/model_executor/layers/logits_processor.py`).

`b12x.gemm.bf16_gemv`'s SIMT backend is the kernel for this shape: grid
`(N, ceil(rows/rows_per_tile))`, 128 threads per CTA, one CTA per output
column, `rows_per_tile = SMALL_M_MAX = 8` fp32 accumulators in registers, and
vectorised 16-byte global loads (`_dot_bf16x8`). **The weight row is read once
for all rows in the tile** -- at M = 5 that is a single row-tile, so the weight
matrix is streamed exactly once, 636 MB, the same traffic cuBLAS has but with a
kernel built for the shape.

What the patch does:

1. adds `b12x.gemm.bf16_gemv` to `_B12X_SUBMODULES` with a `get_b12x_bf16_gemv()` getter, matching the existing accessors;
2. in `_declare_b12x_vocab_plan`, declares a `GemvQuery` plan instead of a `Caps` plan when the capacity is in `1 < M <= SMALL_M_MAX` (and K % 8 == 0, weight contiguous and 16B-aligned);
3. in `get_b12x_preparation_units`, adds `1 + workload.speculative_tokens` to the declared capacities, so the exact verify width is prepared even if it is not a captured graph size;
4. in `_apply_head`, dispatches on `plan.contract.component_id`;
5. logs the routing decision twice (§6).

Deliberate boundaries:

- **M = 1 keeps the Triton vocabulary GEMV.** It is already a native kernel and is the right one at one row.
- **M > 8 keeps torch/cuBLAS.** Above `rows_per_tile = 8` the SIMT grid gains a second row-tile and re-reads the weight, while cuBLAS tiles better; `default_config` switches to the `mma` backend only at M >= 24, which is unmeasured territory. Prefill and c8 decode are therefore **bit-identical to stock**.
- **Fall back, never raise.** The GEMV's prepared state re-validates contiguity and 16-byte alignment and *raises* on violation, so `_apply_head` pre-checks and takes `lm_head.quant_method.apply` if the hidden-state buffer is not what preparation declared.
- **GEMV plans only ever enter the plan dict from the preparation path**, so a lazily declared runtime capacity can never hand an unprepared plan to the kernel.
- The change is in generic vLLM code, so it also applies to any other model with an unquantized BF16 vocabulary head on the b12x backend (GLM5next, DeepSeek-v4.1). Their `_apply_head` was running `F.linear` for exactly the same reason.

### Numerical equivalence

**Not bit-identical.** Both kernels form bf16 x bf16 products, which are exact
in fp32 (8-bit mantissas, product <= 16 bits), and accumulate in fp32. Only the
summation order differs: cuBLAS walks K in 16x16x16 WMMA tiles; the SIMT kernel
runs 128 thread-strided fp32 FMA chains over K and finishes with a warp/shared
tree reduction. The expected drift is ~`sqrt(2560) * 2^-24` ~ 3e-6 relative,
i.e. ~1e-4 absolute on logits of magnitude ~30 -- two orders inside the 1e-3
gate. `scripts/logits_equiv.py --tol 1e-3` is the gate, and a top-5 set change
on any prompt is a fail.

### Expected gain -- stated honestly

636 MB streamed once. Everything hinges on the achieved fraction of the
273 GB/s LPDDR5x roofline.

| assumption for bf16_gemv simt | ms/step | saved | % of 55.03 ms | tok/s (from 98.8) |
|---|---:|---:|---:|---:|
| 100% of spec (impossible) | 2.33 | 1.27 | 2.3% | 101.1 |
| 91% (what the NVFP4 head measures) | 2.56 | 1.04 | 1.9% | 100.7 |
| 85% | 2.74 | 0.86 | 1.6% | 100.4 |
| 75% | 3.11 | 0.49 | 0.9% | 99.7 |
| no better than cuBLAS (65%) | 3.60 | 0.00 | 0% | 98.8 |

**Ship at >= +1%, revert under +0.5%.** This arm cannot reach +5%; the roofline
caps it at +2.3%.

Known unknown: the SIMT kernel does only ~2.5 vector-load iterations per CTA
for K = 2560 (128 threads x 16 B = 2048 B per iteration against 5120 B of
weight row), then pays a block-wide tree reduction per row. With 124160 CTAs
the reduction tail could eat the advantage. `B12X_AUTOTUNE=1` measures
simt x {1,2,4,8} rows_per_tile against the `mma` backend at boot and picks the
winner, so the tuner will answer this on the first boot.

### A2: the bigger, cheaper lever -- MXFP8 verify head, no code at all

Add one line to the recipe env:

```yaml
env:
  VLLM_MXFP8_LM_HEAD: "1"
```

Every precondition in `_supports_default_lm_head_quantization` holds:
`model_config.dtype == bfloat16`, `params_dtype == bfloat16`,
`in_features 2560 % 128 == 0`, `out_features 124160 % 8 == 0`,
`linear_backend == "b12x"`, SM121, and the main head is unquantized with
`tie_word_embeddings: false`. `VLLM_LM_HEAD_A16` defaults to 1, so the head
becomes `Mxfp8OnlineLinearMethod(use_a16=True)` -- MXFP8 g32 weights, BF16
activations, quantized once at load.

| | weights | scales | total | floor | at 91% | saved | % step | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MXFP8 g32 W8A16 head | 317.8 MB | 9.9 MB (e8m0) | **327.8 MB** | 1.20 ms | 1.32 ms | **2.28 ms** | **4.1%** | **103.1** |

That *is* the ~2.6 ms the brief was reaching for, and it needs no patch.

It is mutually exclusive with A: an MXFP8 head is no longer
`UnquantizedEmbeddingMethod`, so `prepare_b12x_vocab_projection` returns early,
no vocabulary plans are declared, and the A patch simply never engages -- they
can be left stacked in the recipe without conflicting, but only one of them
ever runs.

The cost is quality, and unlike the NVFP4 *draft* head this changes the model's
actual output distribution. `logits_equiv --tol 1e-3` will fail by
construction (MXFP8 rounding is ~2^-8 relative per weight, so logits move by
~1e-2). Judge it on behaviour: `fidelity_probe` 100 x 4, tool-eval >= 88, and
read MTP acceptance with suspicion -- an MXFP8 verify head agrees with the
NVFP4 draft head more often than an exact one does, so *rising* acceptance here
is a quality signal, not a win.

---

## 3. B: the hyper-connection weight stream, measured

`hyperconnection.py` builds, per `GatedResidual`:

| tensor | shape | bytes | TP |
|---|---|---:|---|
| `input_mix_weight_down_block_inject` | `[336, 10240]` (= `[320 down, 4 inject, 12 pad]` x `hc_count*hidden = 4*2560`) | 6.88 MB | `MergedColumnParallelLinear(..., disable_tp=True)` -- **replicated** |
| `input_mix_weight_up` | `[10240, 320]` | 6.55 MB | `ReplicatedLinear` -- **replicated** |

97 instances per step (48 layers x {attn, mlp} + the model-level
`hyper_connection_mixer`), matching the trace's 96.9 / 97.9 launches, plus 9
more in the MTP layer. **1.303 GB per node per decode step** -- the largest
dense weight stream in the model, ahead of the linear-attention projections
(1.08 GB). Measured 4.43 + 3.79 = **8.22 ms/step**, i.e. 158 GB/s = 58% of
spec.

The dataflow is a low-rank bottleneck across the full width:

```
hidden [M,10240] --grouped_rmsnorm--> normalized [M,10240]
  --down_block_inject--> [M,336] -> down [M,320] + injection [M,4]
  --scaled_silu--> bottleneck [M,320]
  --up--> gate_logits [M,10240]
  --gate_mean(normalized, gate_logits)--> block_input [M,2560]
```

`grouped_rmsnorm` reduces over each stream's 2560 lanes; `gate_mean` reduces
across the 4 streams for each of the 2560 lanes; `down` reduces over all
10240. Both residual-stream tensors are replicated on every rank -- nothing on
this path is TP-sharded today.

---

## 4. B: verdict -- not sound

Splitting the 10240 axis (the only axis with anything to split) forces a
reduction at the bottleneck, because `down` contracts the whole width into 320
and `gate_mean` contracts the 4 streams into 1. Give rank *r* streams
{2r, 2r+1}: its `down` becomes a K-split needing an all-reduce of `[M,336]`
(3.4 KB) before the SiLU, and its `gate_mean` produces a partial `[M,2560]`
needing a second all-reduce (25.6 KB) -- **two collectives per hyper-connection,
194 per step**, on top of the 109 already there. The `up` GEMM is the only
piece that shards for free (N-split, bit-identical), and only because the
collective it needs is the one `gate_mean` already forces.

The arithmetic, using this pair's own measured one-shot ROCE all-reduce --
**14.92 us over 10,679 launches, 2.89% of step** (`results/profiling/README.md`):

| variant | bytes saved/step | GEMM time saved (at today's 58% eff) | collectives added/step | collective cost | **net** |
|---|---:|---:|---:|---:|---:|
| shard `up` only (N-split, bit-identical) | 318 MB | 2.22 ms | 97 | 1.45 ms | **+0.77 ms (1.4%)** |
| shard `down` only (K-split) | 334 MB | 1.90 ms | 97 | 1.45 ms | **+0.45 ms (0.8%)** |
| shard both | 651 MB | 4.11 ms | 194 | 2.89 ms | **+1.22 ms (2.2%)** |

Every one of those is a *best case* that assumes the halved GEMMs keep their
58% efficiency and that a collective costs nothing beyond its own kernel time.
Neither holds: all-reduce would go from 109 to 303 launches/step and from 4.0%
to ~9% of the step, and each one is a rendezvous between two nodes inserted
into the prologue of every block, where jitter compounds rather than averages.
A 19 us all-reduce instead of 14.9 us -- within the range this fabric has
already shown -- turns the "shard both" row negative.

Against that, `patches/vllm-qwen38-bf16-gemv.patch` already targets **the same
two GEMMs** for an estimated 1.2-1.7 ms with **zero** collectives, and it
attacks the 3.4 ms of kernel inefficiency that sharding cannot touch (sharding
only halves the 4.8 ms floor). The two are mutually exclusive in practice:
sharding changes both shapes and invalidates the GEMV plans and the `hcup` arm.
And the implementation is not small -- real TP on tensors currently built with
`disable_tp=True`, the merged down+inject loader with its 12 pad rows and
`_HC_WEIGHTS_MAPPER`, half-width `Caps(streams=2)` for b12x's
`grouped_rmsnorm`/`gate_mean` while `combine`/`combine_norm` still consume the
full-width residual, plus the model-level mixer and the MTP layer as separate
cases.

**A best case of +0.8% to +2.2%, plausibly zero or negative, for a
cross-cutting change to weight loading, the b12x hyper-connection contract and
the collective budget, on GEMMs a cheaper patch already claims. Not worth it.
No patch written.**

---

## 5. B2: what to do about the 1.3 GB instead

If the goal is to halve the HC weight stream, do it without touching the
fabric: quantize those two BF16 projections online, the same way the MTP head
already is. `Mxfp8OnlineLinearMethod(use_a16=True)` is in-tree, needs no
checkpoint change, and the rebinding hook already exists --
`maybe_route_small_n_bf16` in `patches/vllm-qwen38-bf16-gemv.patch` swaps
`quant_method` on exactly these two modules at construction time.

| | bytes/step/node | floor @273 GB/s | time | saved vs 8.22 ms |
|---|---:|---:|---:|---:|
| HC today, BF16 | 1303 MB | 4.77 ms | 8.22 ms measured (58% of spec) | -- |
| HC as MXFP8 g32 W8A16, same 58% | 672 MB | 2.46 ms | 4.24 ms | **3.98 ms (7.2%)** |
| ... if a b12x route also lifts it to 85% | 672 MB | 2.46 ms | 2.90 ms | **5.32 ms (9.7%)** |

The 7.2% row is the defensible one: it assumes only that the bytes halve. The
9.7% row stacks a kernel improvement on top and should be treated as an
upper bound.

Add the router gate (`[512,2560]` BF16 x 48) and it is more. **Written: see §B2-implementation** -- the real MXFP8 HC stream is 739 MB, not 672, because `blockscaled.pack_weight` pads the up-projection K from 320 to 384. That is the
largest single lever left in the dense path, it costs zero collectives and zero
launches, and its risk is entirely quality -- which is a gate we already run.
Worth a pass of its own.

---

## 6. Confirming the routing decision from the boot log

**On a stock boot, today**, three commands settle §1 with no code change:

```bash
# 1. exactly ONE "Quantizing LM head" line -- the NVFP4 *draft* head.
#    The verify head is unquantized BF16.
grep -n "Quantizing LM head shards" results/serve-la-final.log

# 2. the vocabulary plans are declared and primed, so the verify head IS
#    registered and _apply_head DOES take the b12x branch.
grep -n "b12x priming gemm.bf16_vocab_projection" results/serve-la-final.log

# 3. and yet only one vocabulary-projection kernel was ever compiled: the M=1
#    Triton one. Everything wider resolved to backend="torch".
python3 - <<'PY'
import glob, json
p = glob.glob("/home/nvidia/.cache/sparkrun/runtime-cache/vllm/"
              "local-inference-lab__Qwen3.8-Flash-Next-NVFP4-ca6f25af/"
              "b12x/compile/preparation/*.json")[0]
for key, rec in json.load(open(p))["records"].items():
    cfg = rec.get("config") or {}
    if {"backend", "block_k"} <= set(cfg):
        print(key[:12], cfg, rec.get("coverage"))
PY
# -> {'algorithm': 'row', 'backend': 'triton', 'block_k': 4096, 'num_warps': 4}
```

**With the mod applied**, the patch logs the decision directly:

```bash
grep -n "b12x vocab projection" serve.log
```

Expected at boot, one line per declared capacity:

```
b12x vocab projection plan: head=124160x2560 capacity=1  component=gemm.bf16_vocab_projection
b12x vocab projection plan: head=124160x2560 capacity=2  component=gemm.bf16_gemv
b12x vocab projection plan: head=124160x2560 capacity=5  component=gemm.bf16_gemv
b12x vocab projection plan: head=124160x2560 capacity=8  component=gemm.bf16_gemv
b12x vocab projection plan: head=124160x2560 capacity=16 component=gemm.bf16_vocab_projection
```

and once, at the first verify step:

```
b12x vocab projection: rows=5 component=gemm.bf16_gemv
```

`rows=5 component=gemm.bf16_vocab_projection` means the arm did **not** engage
-- check `VLLM_B12X_VOCAB_GEMV` and whether a capacity <= 8 was declared.

---

## 7. Validation recipe for the GPU worker

Two boots, one env-var arm. Keep `VLLM_QWEN3_8_FLASH_NEXT_OVERLAP` and
`VLLM_MTP_NVFP4_LM_HEAD` at their defaults.

**Boot A -- baseline** (`recipes/eugr/eugr-agents-serve-local16-la.yaml`):

```bash
cd ~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
sparkrun run recipes/eugr/eugr-agents-serve-local16-la.yaml --no-follow
# after health:
for i in 1 2 3; do ( cd tools/tony-bench && python3 bench_sweep.py \
  http://localhost:8000 qwen3.8-flash-next base_c1_$i --levels 1 ); done
python3 scripts/logits_equiv.py capture --out /tmp/logits_base.json
```

**Boot B -- lm_head on b12x.** Copy the recipe to
`recipes/eugr/eugr-agents-serve-local16-la-lmhead.yaml` and change one line:

```yaml
mods: [b12x-startup-boundedwait, vllm-qwen38-lmhead-b12x]
```

```bash
sparkrun run recipes/eugr/eugr-agents-serve-local16-la-lmhead.yaml --no-follow
# the boot log must contain:
#   "mod vllm-qwen38-lmhead-b12x: applied"
#   "import smoke ok; SMALL_M_MAX = 8 enabled = True"
#   "b12x vocab projection: rows=5 component=gemm.bf16_gemv"
# expect one slower first boot while B12X_AUTOTUNE=1 races simt x {1,2,4,8}
# against mma for 3-4 capacities at N=124160.

for i in 1 2 3; do ( cd tools/tony-bench && python3 bench_sweep.py \
  http://localhost:8000 qwen3.8-flash-next lmhead_c1_$i --levels 1 ); done
( cd tools/tony-bench && python3 bench_sweep.py \
  http://localhost:8000 qwen3.8-flash-next lmhead_c8 --levels 8 )

python3 scripts/logits_equiv.py capture --out /tmp/logits_lmhead.json
python3 scripts/logits_equiv.py diff /tmp/logits_base.json /tmp/logits_lmhead.json --tol 1e-3

bash scripts/gate_arm.sh lmhead-b12x
```

**Pass criteria** (all must hold):

| check | bar |
|---|---|
| `bench_sweep --levels 1`, median of 3 | **>= +1%** vs boot A. Roofline caps this arm at +2.3%; treat >= +1% as ship, < +0.5% as revert. |
| `bench_sweep --levels 8` | no regression. At c8 the verify head is M = 40 > 8, so the GEMV never engages and the path is bit-identical to stock; any delta is noise or a real problem. |
| `logits_equiv.py diff` | identical sampled tokens, top-5 logprobs within 1e-3, no top-5 set change. Expected drift is ~1e-4. |
| `fidelity_probe` 8k/32k/64k/128k | **100 x 4** |
| tool-eval hardmode | **>= 88** |
| `straggler_probe` 5 6 7 8 12 16 | clean, no ~18 s c5-7 stall |
| MTP acceptance (`/metrics`) | within noise of boot A |

**Bisect without a rebuild:** `VLLM_B12X_VOCAB_GEMV=0` in the recipe env
disables the route while leaving the mod applied, which isolates the patch from
the boot itself. Same shape as the `VLLM_QWEN38_BF16_GEMV` gate on the
`vllm-qwen38-bf16-gemv` mod, and the two mods patch disjoint files, so they
stack.

**Arm A2 (separate boot, quality first):** add `VLLM_MXFP8_LM_HEAD: "1"` to the
baseline recipe env, confirm a *second* `Quantizing LM head shards to MXFP8`
line at boot, then run the full `gate_arm.sh` battery **before** looking at
tok/s. `logits_equiv` is expected to fail here; that is not a blocker on its
own, it just means this arm is judged on `fidelity_probe` + tool-eval.

---

## 7a. Gotcha: a raced b12x unit must carry an activation producer

The first boot of `la + mods/vllm-qwen38-bf16-gemv` died in preparation:

```
b12x compiling gemm.bf16_gemv: candidates 2/3 prepared, 2 compilations
b12x failed gemm.bf16_gemv
RuntimeError: b12x preparation failed on rank 0:
    ValueError: candidate races require an activation-producing context
```

`b12x/preparation/_measurement.py:prepare_race_steps` refuses any
`PreparedCall` whose `produce` is None, because a candidate race re-produces
its activations before every timed candidate. The first cut of the bf16-gemv
patch built its benchmark source once with `torch.randn` and passed no
producer. Nothing catches this until `B12X_AUTOTUNE=1` actually races the
candidates -- with a single-candidate component (the stock vocabulary
projection at M > 1, whose backend tuple is `("torch",)`) there is no race and
no producer is needed, which is exactly why the stock path never hit it.

Both mods now follow the in-tree reference wiring in
`models/deepseek_v4_1/b12x_layers.py`: an empty source refilled by
`produce=...` before each candidate, and a caller-owned output so the
allocator stays out of the timed region. The lm_head units inherit a correct
producer from the stock `_b12x_vocab_call`, which has always set
`produce=produce`. Both mods' import smoke tests now assert the producer is
present, so this cannot regress silently.

**Any future b12x preparation unit in this tree must set `produce`.**

---

## 8. What was already verified without a GPU

Inside `spark-vllm-b12x:local-20260918-a8333658` (`docker run --rm`, no
`--gpus`, nothing serving touched):

- the installed `vllm/utils/b12x.py` (`79bc7e0b...`) and `vllm/model_executor/layers/logits_processor.py` (`9458a036...`) are **byte-identical** to `git show 8e1f1e58:`, so the patch is cut against the right pre-image;
- `bash run.sh` -> `patching file ... / ast ok / import smoke ok; SMALL_M_MAX = 8 enabled = True / applied`, exit 0;
- second run: `already applied, skipping`, exit 0 (idempotent);
- after appending one byte to `vllm/utils/b12x.py`: `pre-image mismatch -- refusing to patch`, exit 1 (fail-closed);
- `get_b12x_bf16_gemv()` resolves the module, `SMALL_M_MAX == 8`, and `GemvQuery`/`plan`/`mm` are reachable through the lazy API on a CPU-only container;
- both tuning contracts evaluated at the real head geometry (§1), which is the CPU-reproducible form of the root cause.
- the race gate itself, reproduced on CPU: `prepare_race_steps` with a
  `PreparedCall(run=...)` raises exactly
  `ValueError: candidate races require an activation-producing context`,
  while the same call with `produce=...` passes that check and fails only
  on the absent driver -- so the fix addresses the gate that failed;
- both mods apply on top of each other in either order (they patch
  disjoint files) and both import smoke tests assert their producer.

**Still unverified, and only a GPU boot can settle it:** whether
`gemm.bf16_gemv`'s SIMT backend beats `cutlass_80_wmma` for `[124160, 2560]` at
M = 5, and whether the resulting logprobs stay inside 1e-3.

---

## B2-implementation: online MXFP8 for the HC mixers and the router gate

Date: 2026-09-21. No GPU runs. Artifacts:
`patches/vllm-qwen38-hc-mxfp8.patch` (git-format on `8e1f1e58`, three files,
+208 lines) and `mods/vllm-qwen38-hc-mxfp8/{run.sh,vllm-qwen38-hc-mxfp8.patch}`
for image `spark-vllm-b12x:local-20260918-a8333658`.

### What the patch does

`maybe_route_hc_mxfp8(linear, target)` rebinds `quant_method` at construction
time -- the same hook shape and the same four call sites as
`maybe_route_small_n_bf16` in `patches/vllm-qwen38-bf16-gemv.patch`:

| site | module | shape | target |
|---|---|---|---|
| `GatedResidual.__init__` | `input_mix_weight_down_block_inject` | `[336, 10240]` | `hc` |
| `GatedResidual.__init__` | `input_mix_weight_down` (`use_combine=False`) | `[320, 10240]` | `hc` |
| `GatedResidual.__init__` | `input_mix_weight_up` | `[10240, 320]` | `hc` |
| `Qwen3_8FlashNextSparseMoeBlock.__init__` | `mlp.gate` | `[512, 2560]` | `gate` |

All 48 decoder layers x {attn, mlp}, the model-level `hyper_connection_mixer`
and the MTP layer are covered by those two `__init__`s, so the 97 main-path
instances, the 9 MTP instances and the 48 router gates are all routed by one
hook. `VLLM_QWEN38_HC_MXFP8` selects the targets: `hc,gate` (default), `hc`,
`gate`, `off`.

`HcMxfp8LinearMethod` subclasses `UnquantizedLinearMethod` and **keeps its
`create_weights`** -- asserted in the mod's smoke test. That is the whole point
of rebinding after construction rather than passing a `quant_config`: the
layer keeps its stock BF16 `ModelWeightParameter`, so the
`MergedColumnParallelLinear` shard loaders, the 12 alignment pad rows
(`output_sizes = [320, 4, 12]`) and `_HC_WEIGHTS_MAPPER`'s
`orig_to_new_stacked` mapping of `input_mix_weight_down` -> shard 0 and
`block_inject_weight` -> shard 1 all see exactly the parameter they saw before.
`_Fp8OnlineLinearBase.create_weights` would instead have put the weight on
`meta` and wrapped every loader through `initialize_online_processing`, which
is not needed here and is the one thing that could disturb the merged loader.

`process_weights_after_loading` then delegates to the in-tree
`Mxfp8OnlineLinearMethod`, which quantizes the loaded BF16 weight **once**
(`mxfp8_e4m3_quantize`, block-32 along K), calls
`B12xMxfp8LinearKernel.process_weights_after_loading` -- `blockscaled.pack_weight`,
`register_b12x_layer`, `set_b12x_preparation_provider` -- and frees the BF16
parameter. Nothing is quantized per step. `apply` is
`B12xMxfp8LinearKernel.apply_weights`, i.e. the
`vllm::b12x_blockscaled_linear` custom op that already carries this model's
243 MXFP8 linear-attention launches (9.16 ms/step), so the compiled region and
the launch count are unchanged.

Both HC tensors stay TP-replicated. `ReplicatedLinear` and
`MergedColumnParallelLinear(disable_tp=True)` both give `tp_size == 1`, MXFP8
group-32 scales run along K and never reduce across ranks, and
`process_weights_after_loading` in `model_loader/utils.py` re-runs
`update_param_tp_status()` after the parameter swap, so a later weight refit
still narrows correctly. Nothing in this patch adds a collective.

### Shapes: admitted, with one caveat

b12x's block-scaled contract
(`b12x/gemm/blockscaled/_tuning.py::_validate_query` / `_validate_config`,
b12x `a8333658`), re-evaluated on CPU inside the image for every shape the mod
routes, in both activation modes, against an SM121 device identity:

```
  admits hc   input_mix_weight_down_block_inject  N=336    K=10240  padK=10240  auto@M8=a16         6.88 MB ->   3.56 MB
  admits hc   input_mix_weight_up                 N=10240  K=320    padK=384    auto@M8=quantized   6.55 MB ->   4.06 MB
  admits hc   input_mix_weight_down (MTP mixer)   N=320    K=10240  padK=10240  auto@M8=a16         6.55 MB ->   3.40 MB
  admits gate mlp.gate                            N=512    K=2560   padK=2560   auto@M8=a16         2.62 MB ->   1.35 MB
```

The gates are `K % 32 == 0` (the MXFP8 group), `K % 8 == 0`, stored
`padded_K % 32 == 0` for A16 / `% 128 == 0` for the quantized path, and
`N % 8 == 0`. **All four pass**, for every M -- the A16 tile is 16 rows
(`_get_compiled_dense_gemm(..., (16, bn), ...)`), so `M <= 16` is a single
M-tile and M = 5 (c1 verify) and M = 40 (c8) are both served. No fallback to
BF16 is needed for any shape.

Two things are not free:

- **`input_mix_weight_up` pays a K pad.** `blockscaled.pack_weight` rounds K up
  to 128, so K = 320 is stored as 384. The weight is 4.06 MB rather than the
  3.38 MB an unpadded MXFP8 g32 would be -- still 38% below the 6.55 MB BF16,
  but 20% above the ideal. §5's 672 MB figure assumed no pad; the real MXFP8
  HC stream is **739 MB**.
- **`input_mix_weight_up` does not get W8A16 by default.** `_automatic_a16`
  requires `in_features == padded_in_features`, which 320 != 384 fails, so
  b12x picks the quantized-activation path for that GEMM at every M. The other
  three shapes get true W8A16 at every decode regime with M <= 8 and the
  quantized path at prefill capacity, chosen per exact-M regime by
  `plan_regimes`.

That is why the patch passes `Mxfp8OnlineLinearMethod()` with `use_a16` unset
rather than `use_a16=True`. Pinning `"a16"` the way `VLLM_MXFP8_LM_HEAD` does
would also pin the 16-row A16 tile for **prefill**, where the HC mixers see
the full `max_num_batched_tokens` and the LM head never does. The operator can
still force it with the existing `VLLM_B12X_MXFP8_ACTIVATION_MODE=a16` -- no
new knob, and it is a clean A/B arm. The bytes halve either way, which is the
only assumption §5's defensible row makes.

### Expected numbers

Per decode step per node, MXFP8 g32 = values + swizzled UE8M0 scales
(`ceil(N/128) * ceil(padK/32/4) * 512` bytes of F8_128x4 scale storage):

| bucket | count/step | BF16 | MXFP8 | saved |
|---|---:|---:|---:|---:|
| HC down/block-inject `[336,10240]` + up `[10240,320]` | 97 | 1303.2 MB | 739.0 MB | **564.2 MB** |
| router gate `[512,2560]` | 48 | 125.8 MB | 64.9 MB | **60.9 MB** |
| **hc + gate** | **145** | **1429.0 MB** | **803.9 MB** | **625.1 MB** |
| MTP layer HC (accounted separately in the trace) | 9 + 9 | 120.9 MB | 68.6 MB | 52.3 MB |

Time, against the measured baseline of 8.22 + 0.85 = **9.07 ms/step** for
hc + gate in a 55.03 ms step (98.8 tok/s):

| assumption for b12x blockscaled | ms/step | saved | % of 55.03 ms | tok/s | delta |
|---|---:|---:|---:|---:|---:|
| same 158 GB/s the BF16 kernel achieves today (defensible) | 5.10 | **3.97** | **7.2%** | **106.5** | **+7.8%** |
| 85% of the 273 GB/s roofline | 3.46 | 5.61 | 10.2% | 110.0 | +11.4% |
| 100% of spec (impossible) | 2.94 | 6.13 | 11.1% | 111.2 | +12.5% |
| blockscaled no faster per byte than `cutlass_80_wmma` | 9.07 | 0.00 | 0% | 98.8 | 0% |

The first row is the one to plan against: it assumes only that the bytes
halve. The MTP HC rows add another ~0.29 ms (+0.6%) on top, not counted above.
**Ship at >= +5%, revert under +2%** -- unlike the bf16-gemv arm this one is
not roofline-capped below the brief's bar.

### Quality risk

**This changes the model's numerics.** MXFP8 rounding is ~2^-8 relative per
weight, and it is applied to 145 GEMMs per step including the MoE router gate,
whose output feeds a discrete top-k expert selection. `scripts/logits_equiv.py
--tol 1e-3` **fails by construction** -- do not run it as a gate, and do not
read its failure as a defect. The gate is `scripts/gate_arm.sh`:

| check | bar |
|---|---|
| `fidelity_probe` 8k/32k/64k/128k | **100 x 4** |
| tool-eval hardmode | **>= 88** |
| `straggler_probe` 5 6 7 8 12 16 | clean, no ~18 s c5-7 stall |
| `bench_categories` c1 | no category collapse vs boot A |
| MTP acceptance (`/metrics`) | within noise of boot A -- **a rise is a warning sign, not a win** |

The acceptance caveat is the same one §2 records for the MXFP8 verify head and
it applies harder here: the MTP draft layer's own HC mixers and router gate are
quantized by this patch too, so the draft and verify paths are perturbed in a
correlated way. Rising acceptance means the two heads are agreeing because both
are wrong, not because the model got better. Read it together with
`fidelity_probe` and hardmode; if acceptance rises and either of those moves,
the arm is a quality regression wearing a throughput costume.

Two lower risks, both unchanged from stock: the merged tensor's 12 pad rows are
never loaded, so they quantize from `torch.empty` garbage to garbage (possibly
NaN) -- harmless because MXFP8 groups run along K with no cross-N reduction and
`_mix_normalized` slices `[:, :320]` and `[:, 320:324]`, exactly as cuBLAS did.
And `mxfp8_e4m3_quantize` lifts each weight to fp32 during load, a transient
4x spike on a 6.9 MB tensor, one layer at a time.

### The `hc`-only crash, 2026-09-22, and what it says about plan handles

The first GPU boot of `VLLM_QWEN38_HC_MXFP8=hc` (gate left BF16) died in
`Worker_TP0` right after the three expected `qwen38 HC MXFP8: target=hc` lines
(`/tmp/la-hcq-hconly_boot.log`, 13:55:55):

```
File ".../inductor_cache/7l/c7lwolq5id4zagoycemd2yvfhcjmf6vmnv4rhyfibjoinig6xnrk.py", line 165, in call
  torch.ops.b12x.hyperconnection_grouped_rmsnorm.default(arg0_1, arg2_1, arg3_1, 1e-06, 459, zero_centered=True)
File ".../b12x/norm/hyperconnection/_kernels.py", line 221, in _grouped_rmsnorm_op
  prepared = require_prepared(plan_from_handle(plan_handle), "norm.hyperconnection", state.device)
ValueError: plan belongs to gemm.blockscaled_precision, not norm.hyperconnection
```

`hc,gate` had booted and served fine the day before. The asymmetry is the tell.

**Root cause.** `459` is a b12x **plan handle**, and `plan_from_handle`
resolves it out of a process-local dict keyed by a monotonic counter
(`handle = next(_HANDLES)`, `b12x/preparation/types.py`). Handles are
therefore *positional*: their meaning depends on how many plans the boot
created before them, and torch bakes them into the compiled graph as integer
**constants**. vLLM's AOT compile cache
(`VLLM_CACHE_ROOT/torch_compile_cache/torch_aot_compile/<hash>`) relies on the
unstated invariant that an identical configuration replays an identical plan
sequence.

The first version of this patch read its gate with a bare
`os.getenv("VLLM_QWEN38_HC_MXFP8", "hc,gate")` in `hyperconnection.py`.
`envs.compile_factors()` builds the env half of that cache key by iterating
**`vllm.envs.environment_variables`** -- every *declared* vLLM env var, minus a
small ignore list. A variable read outside that dict is invisible to it. So:

- `hc,gate` and `hc` produced the **same** `torch_aot_compile` hash
  (`907927796a24…`, created 09-21 08:09 by the `hc,gate` boot);
- the on-load guard that re-checks traced source files passed, because both
  boots ran byte-identical `hyperconnection.py` / `model.py` -- only the env
  differed. That guard is why the `hc,gate` boot itself was safe: the mod had
  changed those files relative to stock, so it compiled fresh and was
  self-consistent. Its measured numbers stand;
- `hc` skips the 48 router-gate layers, so 48 blockscaled plans are never
  created and every later handle shifts. Handle `459`, an HC-norm plan in the
  `hc,gate` boot, is a `gemm.blockscaled_precision` plan in the `hc` boot.

It was never an HC/GEMM plan-lookup collision inside the patch; the two
providers are correctly separate. It was one cached graph shared by two
different plan populations.

**Fix.** Declare the gate in `vllm/envs.py` (`env_with_choices`, default
`"hc,gate"`, choices `hc,gate | gate,hc | hc | gate | off`, case-insensitive)
and read it through `envs.VLLM_QWEN38_HC_MXFP8`. That is the fork's own
convention -- `VLLM_MXFP8_LM_HEAD`, `VLLM_MTP_NVFP4_LM_HEAD` and
`VLLM_QWEN3_8_FLASH_NEXT_OVERLAP` are all declared there -- and it puts the
gate into `compile_factors()`, giving each target set its own cache entry. The
patch is now three files. A typo in the value now raises at boot instead of
silently disabling the arm, which is worth as much as the fix: a silent no-op
arm costs a whole boot to discover.

The mod's smoke test asserts the regression directly, so it cannot come back:

```
compile-cache key ok; 4 distinct keys for hc,gate / hc / gate / off
invalid gate value rejected at read time
```

**The same latent bug is in `mods/vllm-qwen38-bf16-gemv`.**
`VLLM_QWEN38_BF16_GEMV` is also a bare `os.getenv` in `hyperconnection.py` and
also changes plan population (`_SmallNBF16Provider` declares one GEMV plan per
enabled target per M). Bisecting that arm with `=hc` / `=gate` / `=hcup`
against a warm compile cache will hit the same class of crash. It has not been
fixed here -- the two mods are mutually exclusive and that arm is not in use --
but fix it the same way before running its bisect.

**General rule, worth carrying to every future mod:** any env var that changes
*which b12x plans a boot declares* must be declared in `vllm/envs.py`, not read
with `os.getenv`. Changing the number of plans without changing the compile
cache key is silent graph corruption, and it surfaces as an unrelated-looking
component-mismatch `ValueError` deep inside a b12x kernel.

### Validation recipe for the GPU worker

Mutually exclusive with `vllm-qwen38-bf16-gemv`: both rebind `quant_method` on
the same three modules and both patch the same two files. The mod refuses if
the other one's fingerprint is present, in either order.

**Boot A -- baseline** (`recipes/eugr/eugr-agents-serve-local16-la.yaml`,
`mods: [b12x-startup-boundedwait]`):

```bash
cd ~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
sparkrun run recipes/eugr/eugr-agents-serve-local16-la.yaml --no-follow
# after health -- SIX c1 sweeps, not three: c1 is bimodal (fast 94-99,
# slow 84-88, ~25% of runs), see results/arms/c1-decline/verdict.md.
for i in 1 2 3 4 5 6; do ( cd tools/tony-bench && python3 bench_sweep.py \
  http://localhost:8000 qwen3.8-flash-next base_c1_$i --levels 1 ); done
( cd tools/tony-bench && python3 bench_sweep.py \
  http://localhost:8000 qwen3.8-flash-next base_c8 --levels 8 )
bash scripts/gate_arm.sh hc-mxfp8-base
```

**Boot B -- HC + gate on MXFP8.** Copy the recipe to
`recipes/eugr/eugr-agents-serve-local16-la-hcmxfp8.yaml` and change one line:

```yaml
mods: [b12x-startup-boundedwait, vllm-qwen38-hc-mxfp8]
```

```bash
sparkrun run recipes/eugr/eugr-agents-serve-local16-la-hcmxfp8.yaml --no-follow
for i in 1 2 3 4 5 6; do ( cd tools/tony-bench && python3 bench_sweep.py \
  http://localhost:8000 qwen3.8-flash-next hcmxfp8_c1_$i --levels 1 ); done
( cd tools/tony-bench && python3 bench_sweep.py \
  http://localhost:8000 qwen3.8-flash-next hcmxfp8_c8 --levels 8 )
bash scripts/gate_arm.sh hc-mxfp8
```

**Proof the MXFP8 route engaged**, from the boot log:

```bash
grep -n "qwen38 HC MXFP8" serve.log
```

Expected, one line per distinct shape (four lines, logged once each):

```
qwen38 HC MXFP8: target=hc shape=[336,10240] -> mxfp8 g32, activation_mode=auto, 6.88 MB -> 3.56 MB per instance
qwen38 HC MXFP8: target=hc shape=[10240,320] -> mxfp8 g32, activation_mode=auto, 6.55 MB -> 4.06 MB per instance
qwen38 HC MXFP8: target=hc shape=[320,10240] -> mxfp8 g32, activation_mode=auto, 6.55 MB -> 3.40 MB per instance
qwen38 HC MXFP8: target=gate shape=[512,2560] -> mxfp8 g32, activation_mode=auto, 2.62 MB -> 1.35 MB per instance
```

Corroborating lines, both of which only appear if the b12x block-scaled path
took ownership of these layers:

```bash
grep -n "mod vllm-qwen38-hc-mxfp8: applied"      serve.log   # the mod ran
grep -n "import smoke ok; targets = "            serve.log   # ['gate', 'hc']
grep -c "b12x priming linear.mxfp8"              serve.log   # 145+ new units
```

A `qwen38 HC MXFP8: ... is outside the b12x block-scaled contract` warning
means a shape was refused and kept BF16 -- none is expected, all four are
admitted above. No `qwen38 HC MXFP8` line at all means the mod did not run or
`VLLM_QWEN38_HC_MXFP8=off` is set.

**Pass criteria** (all must hold; this is a numerics change, the quality gate
is not optional):

| check | bar |
|---|---|
| `bench_sweep --levels 1`, **median of 6 and the max** | median **>= +5%** vs boot A's median, and the fast-mode max must also improve. Compare medians AND fast-mode values per the c1-bimodality rule. < +2% is a revert. |
| `bench_sweep --levels 8` | **must also improve.** Unlike the bf16-gemv and lm_head arms, this one engages at every M, so c8 is a real second measurement, not a no-op control. A c8 regression with a c1 gain means the prefill/capacity regime picked a bad mode -- try `VLLM_B12X_MXFP8_ACTIVATION_MODE=quantized`. |
| `logits_equiv.py diff` | **expected to fail. Not a gate.** Capture it anyway for the record. |
| `fidelity_probe` 8k/32k/64k/128k | 100 x 4 |
| tool-eval hardmode | >= 88 |
| `straggler_probe` 5 6 7 8 12 16 | clean |
| MTP acceptance | within noise; a rise is a warning sign |

**Bisect without a rebuild**, all in the recipe env. Each value now has its
own `torch_aot_compile` entry (see the crash section above), so the first boot
on each one recompiles rather than reusing a neighbour's graph:
`VLLM_QWEN38_HC_MXFP8=hc` (HC only, 564 MB), `=gate` (router only, 61 MB),
`=off` (mod applied, route disabled -- isolates the patch from the boot),
`VLLM_B12X_MXFP8_ACTIVATION_MODE=a16` (pin W8A16 everywhere including prefill
and the up-projection), `=quantized` (pin the quantized-activation path).
Expect one slower first boot while `B12X_AUTOTUNE=1` races the A16 knobs
(`tile_n` x `tile_k` x `split_k`) for four new shapes.

### What was verified without a GPU

Inside `spark-vllm-b12x:local-20260918-a8333658` (`docker run --rm`, no
`--gpus`, nothing serving touched):

- the installed `envs.py` (`6b2122c8...`), `hyperconnection.py`
  (`34f9ff5d...`) and `model.py` (`7485ac00...`) are byte-identical to
  `git show 8e1f1e58:`, so the patch is cut against the right pre-image;
  post-images are `e053d175...`, `f6c33fa7...`, `1e1b0f84...`;
- `envs.compile_factors()` contains `VLLM_QWEN38_HC_MXFP8` and yields four
  distinct compile-cache keys for `hc,gate` / `hc` / `gate` / `off`, and
  `envs.VLLM_QWEN38_HC_MXFP8` raises on an invalid value;
- `bash run.sh` -> `patching ... / ast ok / admits x4 / import smoke ok;
  targets = ['gate', 'hc'] / applied`, exit 0;
- second run: `already applied, skipping`, exit 0 (idempotent);
- after appending one byte to `hyperconnection.py`: `pre-image mismatch --
  refusing to patch`, exit 1 (fail-closed);
- with `vllm-qwen38-bf16-gemv` applied first: `vllm-qwen38-bf16-gemv is
  applied -- refusing`, exit 1; in the other order the bf16-gemv mod refuses on
  its own pre-image check. Neither can stack on the other;
- `HcMxfp8LinearMethod.create_weights is UnquantizedLinearMethod.create_weights`
  (the merged loader guarantee), and `VLLM_QWEN38_HC_MXFP8` resolves to
  `[]` / `['hc']` / `['gate']` / `['gate', 'hc']` for `off` / `hc` / `gate` /
  `hc,gate`;
- b12x's MXFP8 query and config validators accept all four shapes in both
  `a16` and `quantized` modes at M = 16, and `_default_config` at M = 8 picks
  `a16` for three of them and `quantized` for `[10240, 320]` (the K-pad).

**Still unverified, and only a GPU boot can settle it:** whether
`gemm.blockscaled` at these shapes actually converts halved bytes into halved
time, whether `mlp.gate` in MXFP8 flips enough expert selections to move
hardmode, and where MTP acceptance lands.
