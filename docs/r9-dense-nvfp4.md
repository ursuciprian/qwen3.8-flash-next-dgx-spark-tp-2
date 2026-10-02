# r9: dense-only NVFP4 requant (plan, no GPU run yet)

Status: plan, 2026-10-02 (opus-kernel-17). Nothing has been quantized yet.

## Why

Dense MXFP8 work is the largest category of the decode step at both TP sizes, and it is
bandwidth-bound. The r8 profile (`results/r8-prof/profile_table.md`) puts it at
24.4-25.7 ms of the TP=1 step and 13.3-15.0 ms of the TP=2 step. opus-kernel-15 showed that
the kernels already run at roofline in isolation: qkvz reaches 244 GB/s. In serving they run
at 186 GB/s because of contention. So a faster kernel will not help; only reading fewer
bytes will.

## Scope: what changes and what stays

| Module (per layer) | Shape | Count | Today | r9 |
|---|---|---:|---|---|
| GDN `linear_attn.in_proj_qkv` | 10240 x 2560 | 36 | MXFP8, frozen in QAD | **NVFP4** |
| GDN `linear_attn.in_proj_z` | 6144 x 2560 | 36 | MXFP8, frozen | **NVFP4** |
| GDN `linear_attn.out_proj` | 2560 x 6144 | 36 | MXFP8, frozen | **NVFP4** |
| GDN `in_proj_a` / `in_proj_b` | 48 x 2560 | 36 | MXFP8 | unchanged (0.25 M params) |
| Attention `self_attn.q_proj` (with gate) | 12288 x 2560 | 12 | MXFP8, frozen | **NVFP4** |
| Attention `k_proj` / `v_proj` | 512 x 2560 | 12 | MXFP8, frozen | **NVFP4** |
| Attention `o_proj` | 2560 x 6144 | 12 | MXFP8, frozen | **NVFP4** |
| QSA `indexer.index_qk_proj` | 640 x 2560 | 12 | MXFP8 | unchanged (indexer recall is sensitive and the gain is small) |
| HC `{attn,mlp}_hyper_connection.input_mix_weight_{down,up}` | 320 x 10240, 10240 x 320 | 96 + 2 top | BF16 in the checkpoint (trained), online MXFP8 (`VLLM_QWEN38_HC_MXFP8=hc`) | **NVFP4** |
| Shared experts | 640 x 2560 x 3 | 48 | MXFP8, trained | unchanged (out of scope) |
| Routed experts | | 48 | NVFP4, trained | **untouched** (bytes copied as-is) |
| Routers, `block_inject_weight`, norms, PLE projections and table | | | BF16 / NVFP4 | untouched |
| MTP layer | | 1 | BF16 + NVFP4 experts | untouched (mtpq rejected in r4; drafter choice follows the r8 TP=1 arms) |
| lm_head | 248320 x 2560 | 1 | BF16, online MXFP8 (`VLLM_MXFP8_LM_HEAD=1`) | unchanged |
| Draft vocab | | | dvocab v2 K=131072 | unchanged |

**Bytes per target forward** (in-scope modules, all ranks together):
- GDN: 2,076 M params.
- Attention: 598 M params.
- HC: 636 M params.
- Total: 3.31 G params.
- MXFP8 at 1.031 B/param is 3.41 GB. NVFP4 at 0.5625 B/param (E2M1 + one E4M3 scale per 16) is 1.86 GB.
- That is **−1.55 GB per verify pass**, a 45% cut.

**Expected gain** if the category scales with bytes (measured dense ms × 0.455):

| | Dense today | Saved | Step now | Step after |
|---|---:|---:|---:|---:|
| TP=1 fresh c1 | ~24.6 ms | up to ~11 ms | ~64 ms (willneed) | ~53-56 ms (at 70-100% of the byte gain) |
| TP=2 fresh c1 | 13.3 ms | up to ~6 ms | 43 ms | ~37-39 ms |

This is the largest lever left on the table. The A16 kernels already exist:
`b12x.gemm.blockscaled` has `NVFP4LinearWeight` and the A16 path shares weight storage
with MXFP8. One thing must be measured before quantizing anything: the NVFP4 A16 throughput
at M = 1-8 on the five shapes. That is step 1 below.

## Source of each weight (the QAD rule)

The NVFP4 checkpoint is a QAD student. The model card says:
- attention projections, including GDN, were **frozen**. Their MXFP8 values are a PTQ of the base;
- residual-stream mixing (HC) and the shared experts were **trained**.

From that:
- **Attention and GDN:** quantize from the **BF16 base** (`Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d`).
  Requantizing MXFP8 to NVFP4 would compound two roundings.
  - Guard: before using the base, check that `MXFP8(base)` reproduces the checkpoint's MXFP8
    bytes for every in-scope tensor (bit-exact, or within one ULP of a scale). If it does not,
    the "frozen" claim is wrong for that tensor, and the source becomes the dequantized checkpoint.
- **HC:** quantize from the **checkpoint's own BF16** HC weights.
  Taking them from the base would undo the distillation.
- **Everything else** is copied byte-for-byte from `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`
  @ `7c4f1bc1`. Unchanged shards are hardlinked, not copied.

## Base checkpoint

| | |
|---|---|
| Repo | `Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` (named as `base_model` in the NVFP4 card) |
| Size | 144 files, 131 safetensors shards, 360.0 GB (335.3 GiB), BF16 |
| Gated | no |
| License | **Qwen Community License 1.0** (`license: other`, `qwen-community-1.0`): free use, modification and redistribution with the notice. Two conditions apply. (1) Above 100 M MAU or US$20 M monthly revenue, the model name must be shown in the UI. (2) A Model-as-a-Service or AI-coding/office-assistant business needs a separate license from Qwen for commercial use; internal use is exempt. The derived NVFP4 checkpoint inherits it. Our use (local serving, published recipes, no hosted service) is within the terms. A published requantized checkpoint must carry the LICENSE and notice |
| Local copy | `dgx-02:~/models/Qwen3.8-Flash-Next-BF16-de4b8e4d/` (started 2026-10-02 07:39, curl at 60 MB/s inside a 3 GiB memory scope so it does not evict the TP=1 PLE page cache; each file is sha256-checked against the HF LFS oid; `STATE` = DONE when all 144 are ok) |

Only ~5.3 GB of BF16 (attention and GDN) is actually read from the base.

## Method

Use weight-only NVFP4 (W4A16) on every in-scope module. Decode runs at M ≤ 8, where
b12x already runs the dense layers A16. Activation quantization would buy nothing there and
would add calibration risk.

1. **Kernel check (GPU, ~20 min, dgx-02, gpu-lock).** Microbench the b12x NVFP4 A16 path
   against MXFP8 A16 on qkvz [16384 x 2560], q [12288 x 2560], out/o [2560 x 6144],
   HC down/up [320 x 10240] / [10240 x 320], at M = 1, 2, 4, 5, 8, 10, 20, at TP=1 and
   TP=2 shard shapes. Use L2-cold weight rotation, the same harness as k15 (`~/GEN-AI/k15`).
   - **Bar:** NVFP4 ≤ 0.65x the MXFP8 time at M = 5.
   - If it misses, stop here. The kernel work comes before the checkpoint work.
2. **Frozen-equality guard (CPU).** Run the `MXFP8(base)` vs checkpoint comparison above, per tensor. Write a report.
3. **Quantize (CPU is enough: 3.3 G params, pure weight math).**
   - Prefer ModelOpt (`nvidia-modelopt`), so the export matches what vLLM's `modelopt_mixed`
     loader already parses: `weight` (packed E2M1 U8 [out, in/2]), `weight_scale` (E4M3 [out, in/16]),
     `weight_scale_2` (FP32 global).
   - Use a weight-only NVFP4 config restricted to the in-scope module patterns, with no
     calibration data. Check the exact config keys against the installed ModelOpt version
     before writing code.
   - The fallback is a standalone exporter with the same tensor layout. Also run the MSE
     scale search: per block, try amax/6 x {1.0, 0.95, 0.9, 0.85} and keep the lowest-error
     scale. This is cheap and usually worth 10-20% lower weight error than plain absmax.
   - Output: `~/models/Qwen3.8-Flash-Next-NVFP4-r9dense/` on both nodes. Changed shards are
     rewritten; `config.json` `quantization_config.config_groups` gets a new
     `group_w4a16_nvfp4_dense` (GDN qkv/z/out, attention q/k/v/o, HC down/up), and those
     targets come out of `group_mxfp8_attention`. Also update `hf_quant_config.json` and
     `export-manifest.json`.
4. **Loader wiring (vLLM fork, CPU tests):**
   - HC is not a quantized linear in the checkpoint today; it is BF16 plus online MXFP8. An
     NVFP4 HC needs the HC linear method to accept a serialized NVFP4 weight.
   - Add `VLLM_QWEN38_HC_MXFP8=off` handling when the checkpoint provides NVFP4 HC.
   - Declare any new knob in `vllm/envs.py`, so it keys the AOT compile cache.
5. **Offline numerics (GPU, one boot each, no serving A/B yet):**
   - logits KLD and top-1 agreement against b1.4 on the fidelity prompts (8k/32k/128k);
   - MTP acceptance per position on fresh/16k/count;
   - per-module ablations if KLD is above the self-noise floor: GDN-only, attention-only,
     HC-only, out_proj-only.
   - GDN projections feed the recurrent state, so expect long-context fidelity to be the
     first thing that breaks.
6. **Screen → A/B → gate → Jev → promote** as usual (TP=2 against b1.4 with the r8 TP=2 driver;
   TP=1 against the 1x recipe). The gate is non-negotiable: hardmode in band, TC-45 5/5,
   fidelity 20/20 to 128k plus seeds 11/13, no stragglers, acceptance unchanged.

## Risks

| Risk | Mitigation |
|---|---|
| NVFP4 PTQ error on frozen attention/GDN that the QAD student never saw (it adapted to MXFP8) | Per-module ablation. If needed, NVFP4 only for out_proj/o_proj and HC (about half the bytes, lower risk) |
| NVFP4 A16 kernel slower than MXFP8 A16 at small M (E2M1 decode cost) | Step 1 gates everything |
| HC NVFP4 changes the residual mix at every layer | HC is its own arm, as hcq was in r3 |
| The guard in step 2 fails (attention not truly frozen) | Quantize from the dequantized checkpoint instead, and accept the double rounding, or drop that tensor |
| Disk: +98 GiB per node for the new checkpoint (dgx-02 ~330 GB free after the base) | Hardlink the unchanged shards; delete the base after the guard + quantization |
