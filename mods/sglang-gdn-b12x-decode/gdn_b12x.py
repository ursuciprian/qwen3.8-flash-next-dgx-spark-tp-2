"""b12x CuTeDSL GDN decode kernel for SGLang (mod sglang-gdn-b12x-decode).

Wraps ``b12x.sequence.gdn_decode`` (local-inference-lab/b12x, Apache-2.0), the
kernel eugr's b12x vLLM route uses for Qwen3.8-Flash-Next on GB10, behind SGLang's
``LinearAttnKernelBase`` packed-decode contract.

b12x fuses the recurrent update with the gated RMSNorm.  SGLang applies that norm
in the model layer (``norm(core, z)`` with norm_before_gate).  Version 1 keeps the
model untouched: the kernel runs with unit norm weight and a constant gate input
whose activation is 1, so its output is ``rmsnorm(core)``; SGLang's own gated norm
then computes ``rmsnorm(rmsnorm(core)) * w * act(z) == rmsnorm(core) * w * act(z)``
(RMSNorm is idempotent up to eps).  Prefill/extend and target-verify stay on the
Triton kernels.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Dict, Optional, Tuple

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

logger = logging.getLogger(__name__)

_HEAD_DIM = 128
# act(z) == 1: sigmoid(20) = 1 - 2e-9; silu(1.278465) = 1.0
_GATE_CONST = {"sigmoid": 20.0, "silu": 1.278465, "swish": 1.278465}


class B12xGDNKernel(LinearAttnKernelBase):
    supports_packed_decode: bool = True

    def __init__(self):
        from b12x.sequence import gdn_decode as gdn  # noqa: F401  (import check)

        self._gdn = gdn
        self._triton = TritonGDNKernel()
        self._plan = None
        self._caps = None
        self._scratch: Optional[torch.Tensor] = None
        self._norm_weight: Optional[torch.Tensor] = None
        self._param_cache: Dict[int, torch.Tensor] = {}
        self._per_bs: Dict[int, dict] = {}
        self._vplan = None
        self._vcaps = None
        self._vscratch = None
        self._per_vbs: Dict[Tuple[int, int], dict] = {}
        self._steps = None
        self.verify_calls = 0
        self.gate_activation = os.environ.get("SGLANG_GDN_B12X_GATE", "sigmoid")
        self.max_bs = int(os.environ.get("SGLANG_GDN_B12X_MAX_BS", "0")) or None
        self.eps = float(os.environ.get("SGLANG_GDN_B12X_EPS", "1e-6"))
        self.calls = 0
        self.fallbacks = 0

    # ---- contract passthroughs (never the hot path for decode) ----
    def decode(self, *args, **kwargs):
        return self._triton.decode(*args, **kwargs)

    def extend(self, *args, **kwargs):
        return self._triton.extend(*args, **kwargs)

    # ---- NEXTN / MTP target verify: the hot path under speculative decoding ----
    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor = None,
        intermediate_state_indices: torch.Tensor = None,
        cache_steps: int = None,
        retrieve_parent_token: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Verify a linear chain of ``T = cache_steps`` draft tokens per request.

        Semantics mirrored from the Triton kernel with ``disable_state_update=True``:
        the committed state ``ssm_states[cache_indices[r]]`` is read, never written; the
        state after token ``t`` lands in ``intermediate_states_buffer[idx[r], t]``.
        b12x reads its initial state from ``state_indices[r, accepted-1]`` and writes the
        post-token-``t`` checkpoint to ``state_indices[r, t]``, so the intermediate buffer
        itself (viewed as a slot pool) is the recurrent state: the committed state is
        copied into slot ``(idx[r], 0)`` first, consumed as the initial state, then
        overwritten by the token-0 checkpoint. Output is ``rmsnorm(core)`` (see module doc).
        """
        if (
            retrieve_parent_token is not None
            or intermediate_states_buffer is None
            or intermediate_state_indices is None
            or not cache_steps
            or cache_steps > 8
            or ssm_states.dtype not in (torch.bfloat16, torch.float32)
            or kwargs.get("cache_ring")
        ):
            self.fallbacks += 1
            return self._triton.target_verify(
                A_log, dt_bias, q, k, v, a, b, ssm_states=ssm_states, cache_indices=cache_indices,
                query_start_loc=query_start_loc, intermediate_states_buffer=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices, cache_steps=cache_steps,
                retrieve_parent_token=retrieve_parent_token, **kwargs,
            )
        T = int(cache_steps)
        N = int(q.shape[1]); B = N // T
        HK, HV, V = int(k.shape[2]), int(v.shape[2]), int(v.shape[3])
        dev = q.device
        # packed [N, W] = [q | k | v]; one concat copy (the backend already split them)
        mixed = torch.cat((q.reshape(N, -1), k.reshape(N, -1), v.reshape(N, -1)), dim=-1)
        inter = intermediate_states_buffer  # [R_cap+1, steps_cap, HV, V, K]
        steps_cap = int(inter.shape[1])
        pool = inter.view(-1, HV, V, int(inter.shape[-1]))  # slots = (R_cap+1) * steps_cap
        self._ensure_verify_plan(mixed, pool, HK, HV, V, T, B)
        if B > self._vcaps.max_seqs:
            self.fallbacks += 1
            return self._triton.target_verify(
                A_log, dt_bias, q, k, v, a, b, ssm_states=ssm_states, cache_indices=cache_indices,
                query_start_loc=query_start_loc, intermediate_states_buffer=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices, cache_steps=cache_steps,
                retrieve_parent_token=retrieve_parent_token, **kwargs,
            )
        bufs = self._vbufs(B, T, HV, V, dev)
        idx = intermediate_state_indices[:B].to(torch.int64)
        base = idx * steps_cap
        bufs["state_idx"].copy_((base.unsqueeze(1) + self._steps[:T].unsqueeze(0)).to(torch.int32))
        # committed state -> slot (idx, 0); read as initial, then overwritten by the token-0 checkpoint
        pool.index_copy_(0, base, ssm_states.index_select(0, cache_indices[:B].to(torch.int64)))
        binding = self._gdn.bind(
            self._vplan,
            scratch=self._vscratch,
            mixed_qkv=mixed,
            a=a.reshape(N, HV).contiguous(),
            b=b.reshape(N, HV).contiguous(),
            z=bufs["z"],
            A_log=A_log.contiguous(),
            dt_bias=dt_bias.contiguous(),
            norm_weight=self._norm_weight,
            recurrent_state=pool,
            query_start_loc=bufs["qsl"],
            num_accepted_tokens=bufs["accepted"],
            state_indices=bufs["state_idx"],
            num_seqs=bufs["num_seqs"],
            num_tokens=bufs["num_tokens"],
            output=bufs["out"],
        )
        out = self._gdn.run(binding, eps=self.eps, scale=float(int(q.shape[3]) ** -0.5))
        self.verify_calls += 1
        return out.view(1, N, HV, V)

    def _ensure_verify_plan(self, mixed, pool, HK, HV, V, T, B):
        if self._vplan is not None:
            return
        gdn = self._gdn
        max_bs = self._resolve_max_bs(B)
        self._vcaps = gdn.Caps(
            device=mixed.device, max_tokens=max_bs * T, max_seqs=max_bs, max_state_slots=int(pool.shape[0]),
            key_heads=HK, value_heads=HV, key_head_dim=_HEAD_DIM, value_head_dim=V, state_index_columns=T,
            model_dtype=mixed.dtype, state_dtype=pool.dtype, gate_activation=self.gate_activation, qk_l2norm=True,
        )
        self._vplan = gdn.plan(self._vcaps)
        (spec,) = self._vplan.scratch_specs()
        self._vscratch = torch.empty(spec.shape, dtype=spec.dtype, device=mixed.device)
        if self._norm_weight is None:
            self._norm_weight = torch.ones(V, dtype=torch.bfloat16, device=mixed.device)
        self._steps = torch.arange(8, dtype=torch.int64, device=mixed.device)
        logger.info("b12x GDN verify: T=%d max_bs=%d slots=%d state=%s", T, max_bs, pool.shape[0], pool.dtype)
        if not torch.cuda.is_current_stream_capturing():
            try:
                from sglang.srt.runtime_context import get_exec
                sizes = set(int(x) for x in (get_exec().graph.cuda_graph_config.decode.bs or []))
            except Exception:
                sizes = set()
            sizes = sorted(x for x in (sizes | {B}) if x <= max_bs)
            dummy = torch.zeros_like(pool)
            for n in sizes:
                bufs = self._vbufs(n, T, HV, V, mixed.device)
                bufs["state_idx"].copy_((torch.arange(n, device=mixed.device).unsqueeze(1) * T + self._steps[:T].unsqueeze(0)).to(torch.int32) % pool.shape[0])
                binding = gdn.bind(
                    self._vplan, scratch=self._vscratch,
                    mixed_qkv=torch.zeros(n * T, mixed.shape[-1], dtype=mixed.dtype, device=mixed.device),
                    a=torch.zeros(n * T, HV, dtype=mixed.dtype, device=mixed.device), b=torch.zeros(n * T, HV, dtype=mixed.dtype, device=mixed.device),
                    z=bufs["z"], A_log=torch.zeros(HV, dtype=torch.float32, device=mixed.device), dt_bias=torch.zeros(HV, dtype=mixed.dtype, device=mixed.device),
                    norm_weight=self._norm_weight, recurrent_state=dummy, query_start_loc=bufs["qsl"], num_accepted_tokens=bufs["accepted"],
                    state_indices=bufs["state_idx"], num_seqs=bufs["num_seqs"], num_tokens=bufs["num_tokens"], output=bufs["out"],
                )
                gdn.run(binding, eps=self.eps, scale=_HEAD_DIM ** -0.5)
            torch.cuda.synchronize(); del dummy
            logger.info("b12x GDN verify: prewarmed batch sizes %s", sizes)

    def _vbufs(self, B, T, HV, V, dev):
        key = (B, T)
        b = self._per_vbs.get(key)
        if b is None:
            zc = _GATE_CONST.get(self.gate_activation, 20.0)
            b = {
                "qsl": (torch.arange(B + 1, device=dev) * T).to(torch.int32),
                "accepted": torch.ones(B, dtype=torch.int32, device=dev),
                "state_idx": torch.zeros((B, T), dtype=torch.int32, device=dev),
                "num_seqs": torch.tensor([B], dtype=torch.int32, device=dev),
                "num_tokens": torch.tensor([B * T], dtype=torch.int32, device=dev),
                "z": torch.full((B * T, HV, V), zc, dtype=torch.bfloat16, device=dev),
                "out": torch.empty((B * T, HV, V), dtype=torch.bfloat16, device=dev),
            }
            self._per_vbs[key] = b
        return b

    # ---- helpers ----
    def _resolve_max_bs(self, bs: int) -> int:
        if self.max_bs:
            return max(self.max_bs, bs)
        try:
            from sglang.srt.runtime_context import get_exec

            cfg = get_exec().graph.cuda_graph_config.decode.bs
            m = int(max(cfg)) if cfg else 0
        except Exception:
            m = 0
        return max(m, bs, 8)

    def _ensure_plan(self, mixed_qkv, ssm_states, num_v_heads, head_v_dim, bs):
        if self._plan is not None:
            return
        gdn = self._gdn
        width = mixed_qkv.shape[-1]
        key_heads = (width - num_v_heads * head_v_dim) // (2 * _HEAD_DIM)
        assert key_heads * 2 * _HEAD_DIM + num_v_heads * head_v_dim == width, (
            f"cannot split packed qkv width {width} into {num_v_heads}x{head_v_dim} values and 128-wide q/k"
        )
        max_bs = self._resolve_max_bs(bs)
        self._caps = gdn.Caps(
            device=mixed_qkv.device,
            max_tokens=max_bs,
            max_seqs=max_bs,
            max_state_slots=int(ssm_states.shape[0]),
            key_heads=int(key_heads),
            value_heads=int(num_v_heads),
            key_head_dim=_HEAD_DIM,
            value_head_dim=int(head_v_dim),
            state_index_columns=1,
            model_dtype=mixed_qkv.dtype,
            state_dtype=ssm_states.dtype,
            gate_activation=self.gate_activation,
            qk_l2norm=True,
        )
        self._plan = gdn.plan(self._caps)
        (spec,) = self._plan.scratch_specs()
        self._scratch = torch.empty(spec.shape, dtype=spec.dtype, device=mixed_qkv.device)
        self._norm_weight = torch.ones(head_v_dim, dtype=torch.bfloat16, device=mixed_qkv.device)
        logger.info(
            "b12x GDN decode: key_heads=%d value_heads=%d head_dim=%d max_bs=%d slots=%d state=%s gate=%s",
            key_heads, num_v_heads, head_v_dim, max_bs, ssm_states.shape[0], ssm_states.dtype, self.gate_activation,
        )
        self._prewarm(mixed_qkv, ssm_states, num_v_heads, head_v_dim, key_heads, bs)

    def _prewarm(self, mixed_qkv, ssm_states, num_v_heads, head_v_dim, key_heads, bs):
        """Compile every CUDA-graph batch size before any capture (b12x JIT-prepares on first use
        and refuses to do so under stream capture). Runs on private dummy tensors."""
        if torch.cuda.is_current_stream_capturing():
            return
        try:
            from sglang.srt.runtime_context import get_exec
            sizes = set(int(x) for x in (get_exec().graph.cuda_graph_config.decode.bs or []))
        except Exception:
            sizes = set()
        sizes = sorted(x for x in (sizes | {bs}) if x <= self._caps.max_seqs)
        dev = mixed_qkv.device
        dummy_state = torch.zeros_like(ssm_states)
        A_log = torch.zeros(num_v_heads, dtype=torch.float32, device=dev)
        dt_bias = torch.zeros(num_v_heads, dtype=mixed_qkv.dtype, device=dev)
        for n in sizes:
            bufs = self._bufs(n, num_v_heads, head_v_dim, dev)
            bufs["state_idx"].copy_(torch.arange(n, dtype=torch.int32, device=dev).view(n, 1) % ssm_states.shape[0])
            binding = self._gdn.bind(
                self._plan, scratch=self._scratch,
                mixed_qkv=torch.zeros(n, mixed_qkv.shape[-1], dtype=mixed_qkv.dtype, device=dev),
                a=torch.zeros(n, num_v_heads, dtype=mixed_qkv.dtype, device=dev),
                b=torch.zeros(n, num_v_heads, dtype=mixed_qkv.dtype, device=dev),
                z=bufs["z"], A_log=A_log, dt_bias=dt_bias, norm_weight=self._norm_weight,
                recurrent_state=dummy_state, query_start_loc=bufs["qsl"], num_accepted_tokens=bufs["accepted"],
                state_indices=bufs["state_idx"], num_seqs=bufs["num_seqs"], num_tokens=bufs["num_tokens"], output=bufs["out"],
            )
            self._gdn.run(binding, eps=self.eps, scale=_HEAD_DIM**-0.5)
        torch.cuda.synchronize()
        del dummy_state
        logger.info("b12x GDN decode: prewarmed batch sizes %s", sizes)

    def _param32(self, t: torch.Tensor) -> torch.Tensor:
        # A_log / dt_bias are constant after load; cache fp32 contiguous copies
        key = t.data_ptr()
        c = self._param_cache.get(key)
        if c is None or c.shape != t.shape:
            c = t.detach().to(torch.float32).contiguous()
            self._param_cache[key] = c
        return c

    def _bufs(self, bs: int, num_v_heads: int, head_v_dim: int, device) -> dict:
        b = self._per_bs.get(bs)
        if b is None:
            zc = _GATE_CONST.get(self.gate_activation, 20.0)
            b = {
                "qsl": torch.arange(bs + 1, dtype=torch.int32, device=device),
                "accepted": torch.ones(bs, dtype=torch.int32, device=device),
                "state_idx": torch.zeros((bs, 1), dtype=torch.int32, device=device),
                "num_seqs": torch.tensor([bs], dtype=torch.int32, device=device),
                "num_tokens": torch.tensor([bs], dtype=torch.int32, device=device),
                "z": torch.full((bs, num_v_heads, head_v_dim), zc, dtype=torch.bfloat16, device=device),
                "out": torch.empty((bs, num_v_heads, head_v_dim), dtype=torch.bfloat16, device=device),
            }
            self._per_bs[bs] = b
        return b

    # ---- hot path ----
    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs.get("replayssm_d") is not None or ssm_states.dtype not in (torch.bfloat16, torch.float32):
            self.fallbacks += 1
            return self._triton.packed_decode(
                mixed_qkv, a, b, A_log=A_log, dt_bias=dt_bias, scale=scale, ssm_states=ssm_states,
                cache_indices=cache_indices, num_v_heads=num_v_heads, head_v_dim=head_v_dim, **kwargs,
            )
        bs = int(mixed_qkv.shape[0])
        self._ensure_plan(mixed_qkv, ssm_states, num_v_heads, head_v_dim, bs)
        if bs > self._caps.max_seqs:
            self.fallbacks += 1
            return self._triton.packed_decode(
                mixed_qkv, a, b, A_log=A_log, dt_bias=dt_bias, scale=scale, ssm_states=ssm_states,
                cache_indices=cache_indices, num_v_heads=num_v_heads, head_v_dim=head_v_dim, **kwargs,
            )
        bufs = self._bufs(bs, num_v_heads, head_v_dim, mixed_qkv.device)
        bufs["state_idx"].copy_(cache_indices.view(bs, 1))
        binding = self._gdn.bind(
            self._plan,
            scratch=self._scratch,
            mixed_qkv=mixed_qkv.contiguous(),
            a=a.contiguous(),
            b=b.contiguous(),
            z=bufs["z"],
            A_log=A_log.contiguous(),
            dt_bias=dt_bias.contiguous(),
            norm_weight=self._norm_weight,
            recurrent_state=ssm_states,
            query_start_loc=bufs["qsl"],
            num_accepted_tokens=bufs["accepted"],
            state_indices=bufs["state_idx"],
            num_seqs=bufs["num_seqs"],
            num_tokens=bufs["num_tokens"],
            output=bufs["out"],
        )
        out = self._gdn.run(binding, eps=self.eps, scale=float(scale))
        self.calls += 1
        return out.view(1, bs, num_v_heads, head_v_dim)
