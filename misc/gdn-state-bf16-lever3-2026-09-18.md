# Lever 3: GDN recurrent state in BF16 — design note (2026-09-18)

**Verdict: no kernel patch needed. The 2026-09-17 `MAMBA_SSM_CACHE_DTYPE=bfloat16`
arm almost certainly failed for a trivial reason — it set an environment
variable that vLLM never reads. The real knob is a CLI flag,
`--mamba-ssm-cache-dtype bfloat16`, and every layer between it and the b12x
kernel already supports BF16 state end to end.** This overturns the brief's
premise that "the b12x GDN path ignores it" — no evidence of that was found;
what was found is that the flag never reached `CacheConfig` in the first
place.

## Trace, file:line (fork `local-inference-lab/vllm@8e1f1e587`, b12x `a833365`)

1. `--mamba-ssm-cache-dtype` is a real CLI flag, not an env var:
   `vllm/config/cache.py:204` — `mamba_ssm_cache_dtype: MambaDType = "auto"`
   `vllm/engine/arg_utils.py:796,1354,2178` — wired straight into
   `CacheConfig`. `MambaDType = Literal["auto", "float32", "float16",
   "bfloat16"]` (`vllm/config/cache.py:70`). There is **no**
   `MAMBA_SSM_CACHE_DTYPE` entry in `vllm/envs.py` — the old arm's env var
   was never consumed by anything; it was a silent no-op, not a kernel
   limitation.

2. Qwen's GDN layer reads the CLI-set value, not a hardcoded float32:
   `vllm/model_executor/layers/mamba/gdn/base.py:53-58`
   ```python
   def get_state_dtype(self) -> tuple[torch.dtype, ...]:
       return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
           self.model_config.dtype,
           self.cache_config.mamba_cache_dtype,
           self.cache_config.mamba_ssm_cache_dtype,
       )
   ```
   `vllm/model_executor/layers/mamba/mamba_utils.py:121-142`
   (`gated_delta_net_state_dtype` → `_mamba_state_dtype`):
   ```python
   if mamba_ssm_cache_dtype == "auto":
       temporal_state_dtype = conv_state_dtype
   else:
       temporal_state_dtype = STR_DTYPE_TO_TORCH_DTYPE[mamba_ssm_cache_dtype]
   return (conv_state_dtype, temporal_state_dtype)
   ```
   So `--mamba-ssm-cache-dtype bfloat16` flows straight to
   `temporal_state_dtype = torch.bfloat16` — this **is** the recurrent
   (SSM) state, as opposed to the short-conv state.

3. That value is what builds the b12x `Caps` object, at both the plain
   decode and the MTP/verify call sites:
   `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:845-857`
   (`_make_b12x_gdn_caps`): `state_dtype=self.get_state_dtype()[1]`.
   Line 955-965 (second `Caps(...)` construction, speculative/verify path):
   `state_dtype=recurrent_state.dtype` — consistent, driven by the same
   tensor whose dtype traces back to the same calculator. No separate
   float32 override for the draft/verify path was found.

4. b12x's kernel `Caps` dataclass already accepts BF16 state as a first-class
   option, not just float32:
   `b12x/sequence/gdn_decode/_impl.py:53` — `state_dtype: torch.dtype =
   torch.float32` (default only; not enforced), and line 91-94:
   ```python
   if self.state_dtype not in (torch.bfloat16, torch.float32):
       raise TypeError(...)
   ```
   Both allocation sites (`_impl.py:464,694`, tensor `recurrent_state`) use
   `dtype=caps.state_dtype` directly — no hardcoded float32 in the
   allocator.

5. The fused decode CUDA kernel's own eligibility gate explicitly accepts
   BF16 recurrent state:
   `qwen_gdn_linear_attn.py:194` — `FUSED_GDN_STATE_DTYPES = (torch.float32,
   torch.bfloat16)`, checked at line 1273
   (`_fused_gdn_decode_unsupported_reason`): BF16 state does not disable the
   fused path. (Requires `conv_state_dtype == torch.bfloat16`, which is the
   default via `mamba_cache_dtype="auto"` resolving to the model dtype —
   unaffected by this change.)

6. Accumulate-in-fp32 / store-in-bf16 is the kernel's existing pattern, not
   something this change has to add: the Triton decode kernel
   (`b12x/sequence/gdn_decode/_kernels.py`) repeatedly loads state/gate
   values and immediately `.to(tl.float32)`s them before the recurrent
   update math (lines 111-156), and the CuTe kernel's norm path has an
   explicit `norm_fp32` toggle (`_cute_kernels.py:95-189`) independent of
   `state_dtype`. Storage dtype and compute dtype are already decoupled in
   this kernel; halving `state_dtype` to BF16 only changes the memory
   footprint/bandwidth of the stored `recurrent_state` tensor
   (`(max_state_slots, h, 128, 128)`, `_preparation.py:127`), not the
   arithmetic precision of a single decode step.

## What this means for prefill / MTP / checkpoint paths

- **Prefill**: state allocation goes through the same `get_state_dtype()` /
  `Caps.state_dtype` path (no separate prefill-only dtype resolution was
  found in `qwen_gdn_linear_attn.py`), so prefill's initial recurrent state
  would also be BF16 once the flag is set — consistent end to end, not a
  decode-only change.
- **MTP / target-verify path**: the second `Caps(...)` construction at
  `qwen_gdn_linear_attn.py:955-965` derives `state_dtype` from
  `recurrent_state.dtype` directly (the already-allocated tensor), so it
  inherits whatever `get_state_dtype()` produced — no divergent dtype
  between the draft and verify passes.
- **`--mamba-cache-mode align`** (required for MTP, per
  `Qwen3_8FlashNextMTP.__init__`) is orthogonal to `state_dtype` — it
  governs block-grid alignment for prefix-cache reuse, not the tensor's
  numeric type. No interaction found.

## Expected gain

MiaAI-Lab measured **+8.5% at 8 concurrent streams** on stock vLLM (not b12x)
for this halved-bandwidth state. Our stack differs (b12x kernel, TP2/RoCE,
NVFP4 MoE, fp8 KV) so this number is a prior, not a promise — re-measure on
our own concurrency sweep (c1/c4/c5-c8, same shape as the 2026-09-17
baseline) rather than trusting it directly.

## Quality gate

Recurrent-state precision is exactly the kind of change that can degrade
long-range recall without moving short-context correctness at all, so the
gate is long-context-specific, per the brief:
- `scripts/fidelity_probe.py` 100% exact at 128k (not just the
  8k-123k band already gated for other levers).
- `scripts/needle_ladder.py` at 64k: recall must match the BF16-KV-rejected
  arm's own float32-state baseline from `project-qwen-verdicts-2026-09-17.md`
  (needle correct at every rung there) — any drop at 64k is the state
  precision change, not noise.
- Standard lanes: tool-eval hardmode within 86-93, MTP acceptance not below
  the 61% cumulative / 81-62-48-37% per-position baseline (a state precision
  regression would show up as an acceptance-rate drop before a visible
  output error, since MTP drafts are always re-verified).

## One-boot A/B

1. Copy `recipes/eugr/eugr-agents-serve-local.yaml` to
   `recipes/eugr/eugr-agents-serve-local-gdnbf16.yaml`, add
   `--mamba-ssm-cache-dtype bfloat16` to the `command:` block. (Already done
   — see that file.) No source patch, no rebuild: same container tag as the
   current local-image baseline.
2. Boot once, run: Tony's 40-prompt single-stream harness, concurrency sweep
   c1/c4/c5-c8 (KV headroom differs slightly since the recurrent-state
   tensor shrinks — watch `gpu_memory_utilization 0.80`'s effect on max
   concurrent seqs, don't assume identical to baseline), `fidelity_probe.py`
   at 128k, `needle_ladder.py` at 64k, one tool-eval hardmode run.
3. Compare against the `eugr-agents-serve-local.yaml` baseline
   (`project-qwen-verdicts-2026-09-17.md` / `project-qwen-straggler-batch5-7.md`
   numbers) from the same image, flag-only diff.
4. If it regresses recall at 64k/128k: revert to `mamba-ssm-cache-dtype
   auto` (float32 state) — this is a flag flip, not a code rollback, so the
   A/B is cheap in both directions.
