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

    def target_verify(self, *args, **kwargs):
        return self._triton.target_verify(*args, **kwargs)

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
