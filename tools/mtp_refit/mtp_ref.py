"""Training model of the Qwen3.8-Flash-Next MTP layer (#97).

MtpRef mirrors vLLM's ``Qwen3_8FlashNextMultiTokenPredictor`` (exp/v3d f11fbbbf):
feedback fusion from the b12x torch oracle (``b12x.sequence.mtp_feedback.reference``),
then decoder layer 48 and the MTP hyper-connection mixer built from transformers 5.18
``qwen4_exp`` blocks, with our own attention call so a D-draft unroll trains in one pass
the way the draft KV cache sees it (TTT style). Parameter names are the checkpoint names
without the ``mtp.`` prefix, so export is a rename.

Frozen at their served values: the routed experts (NVFP4 W4A16 dequantized), the shared
embedding, and the draft head (the K-row draft-vocab subset of lm_head after the NVFP4
round trip of vLLM's ``Nvfp4OnlineLinearMethod``; activations stay BF16 as with
VLLM_LM_HEAD_A16=1).

Known gaps, settled by the phase 2 GPU parity test, not here:
- served HC mixers of the drafter run online MXFP8 (VLLM_QWEN38_HC_MXFP8=hc); trained BF16
- QSA sparse selection is skipped: windows <= indexer_budget tokens make it dense and exact
- K/V go through the fp8 e4m3 round trip of the served KV cache (scale 1.0). Phase 2 (#97) did NOT confirm it:
  the T=0 step replay on the live set gives per position 0.854/0.692/0.550/0.427 with the round trip and
  0.856/0.724/0.612/0.518 without it (K is what matters, V is neutral), live 0.856/0.710/0.585/0.485 sits in
  between; the served K precision is unresolved
- the frozen head and experts are BF16 copies of fp4 x e4m3 values (~2^-9 extra rounding per weight)
- draft step k of anchor t runs at position t + k against depth-0 keys at their own positions; RoPE
  only sees relative positions, so a constant offset in vLLM's draft positions would not matter
"""
from __future__ import annotations

import gzip
import json
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from b12x.sequence.mtp_feedback.reference import feedback
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRMSNorm,
    Qwen4ExpTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

EMBED_NAMES = ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")
EXPERTS = "layers.0.mlp.experts."
FROZEN_PREFIXES = (EXPERTS, "layers.0.self_attn.indexer.")  # indexer: no gradient (dense attention)
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
FP4_MAX, FP8_MAX = 6.0, 448.0


# ---------------------------------------------------------------- NVFP4 helpers
def nvfp4_dequant(packed: torch.Tensor, scale: torch.Tensor, scale_2: torch.Tensor) -> torch.Tensor:
    """ModelOpt NVFP4 [out, in/2] u8 (low nibble first) x e4m3 [out, in/16] x f32 global -> f32 [out, in]."""
    out, half = packed.shape
    nib = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(out, 2 * half).long()
    vals = E2M1.to(packed.device)[nib].view(out, -1, 16)
    return (vals * scale.float()[..., None] * scale_2.float()).view(out, 2 * half)


def _e2m1_round(x: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 value, ties as vLLM's reference cast_to_fp4 (tests/kernels/quantization)."""
    a, q = x.abs(), torch.zeros_like(x)
    for lo, hi, v, closed in ((0.25, 0.75, 0.5, False), (0.75, 1.25, 1.0, True), (1.25, 1.75, 1.5, False),
                              (1.75, 2.5, 2.0, True), (2.5, 3.5, 3.0, False), (3.5, 5.0, 4.0, True)):
        m = ((a >= lo) & (a <= hi)) if closed else ((a > lo) & (a < hi))
        q = torch.where(m, v, q)
    return torch.where(a > 5.0, 6.0, q) * x.sign()


def nvfp4_round_trip(w: torch.Tensor) -> torch.Tensor:
    """Weight values vLLM serves after online NVFP4 (per-tensor global scale, group 16), f32."""
    rows, cols = w.shape
    w = w.float()
    gs = FP4_MAX * FP8_MAX / w.abs().amax().clamp_min(1e-8)
    blk = w.view(rows, cols // 16, 16)
    s = (gs * blk.abs().amax(-1, keepdim=True) / FP4_MAX).to(torch.float8_e4m3fn).float()
    inv = torch.where(s == 0, torch.zeros_like(s), gs / s)
    q = _e2m1_round((blk * inv).clamp(-FP4_MAX, FP4_MAX))
    return (q * s / gs).view(rows, cols)


def load_draft_vocab(path: str, vocab_size: int) -> torch.Tensor:
    """Same format as vLLM's load_draft_vocab_ids: one id per line, '#' comments, optional .gz."""
    with (gzip.open if path.endswith(".gz") else open)(path, "rt", encoding="utf-8") as fh:
        ids = sorted({int(s) for line in fh if (s := line.split("#", 1)[0].strip())})
    if not ids or ids[0] < 0 or ids[-1] >= vocab_size:
        raise ValueError(f"draft vocab {path}: empty or out of range")
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------- model
def text_config(snapshot: str, **overrides) -> Qwen4ExpTextConfig:
    cfg = json.load(open(os.path.join(snapshot, "config.json")))
    cfg = cfg.get("text_config", cfg)
    keep = {k: v for k, v in cfg.items() if hasattr(Qwen4ExpTextConfig, k)}
    keep.update(num_hidden_layers=1, layer_types=["full_attention"], ple_layer_ids=[], **overrides)
    return Qwen4ExpTextConfig(**keep)


class MtpRef(nn.Module):
    def __init__(self, cfg: Qwen4ExpTextConfig, draft_rows: int):
        super().__init__()
        self.cfg = cfg
        H, S = cfg.hidden_size, cfg.hc_count
        self.embed_tokens = nn.Embedding(cfg.vocab_size, H)
        self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(H, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(S * H, eps=cfg.rms_norm_eps)
        self.fc_embedding = nn.Linear(H, H, bias=False)
        self.fc_hidden = nn.Linear(H, H, bias=False)
        self.layers = nn.ModuleList([Qwen4ExpTextDecoderLayer(cfg, 0)])
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(cfg, use_combine=False)
        self.register_buffer("head", torch.empty(draft_rows, H), persistent=False)
        self.register_buffer("draft_ids", torch.empty(draft_rows, dtype=torch.long), persistent=False)
        self.rotary = None  # built after to_empty (holds computed buffers)
        self.compute_dtype = torch.bfloat16  # the b12x feedback oracle is BF16-only; tests use float64
        self.kv_fp8 = True  # served with --kv-cache-dtype fp8: K/V pass through e4m3 (scale 1.0, no MTP k/v scales)

    # -- one draft pass over rows, attention supplied by the caller
    def _qkv(self, x: torch.Tensor, pos: torch.Tensor):
        a, n, hd = self.layers[0].self_attn, x.shape[0], self.cfg.head_dim
        q, gate = a.q_proj(x).view(n, -1, 2 * hd).chunk(2, dim=-1)
        q = a.q_norm(q)
        k = a.k_norm(a.k_proj(x).view(n, -1, hd))
        v = a.v_proj(x).view(n, -1, hd)
        cos, sin = self.rotary(x, pos.view(1, 1, -1).expand(3, 1, -1))
        q, k = apply_rotary_pos_emb(q, k, cos[0], sin[0], unsqueeze_dim=1)
        if self.kv_fp8:
            k, v = _fp8_ste(k), _fp8_ste(v)
        return q, gate.reshape(n, -1), k, v

    def block(self, tokens, state, pos, attend):
        """tokens [n], state [n, S*H], pos [n] -> (multi_hidden [n, S*H], sample [n, H], k, v)."""
        L, n, S, H = self.layers[0], tokens.shape[0], self.cfg.hc_count, self.cfg.hidden_size
        bf = self.compute_dtype
        fused = feedback(
            self.embed_tokens(tokens).to(bf).contiguous(), state.to(bf).reshape(n, S, H).contiguous(),
            self.pre_fc_norm_embedding.weight.to(bf), self.pre_fc_norm_hidden.weight.to(bf),
            self.fc_embedding.weight.to(bf), self.fc_hidden.weight.to(bf), eps=self.cfg.rms_norm_eps,
        ).flatten(-2)
        x, hyper, inj = L.attn_hyper_connection(fused)
        q, gate, k, v = self._qkv(x, pos)
        o = attend(q, k, v).reshape(n, -1)
        h = hyper + (L.self_attn.o_proj(o * torch.sigmoid(gate))[:, None] * inj[..., None]).flatten(-2)
        x, hyper, inj = L.mlp_hyper_connection(h)
        h = hyper + (L.mlp(x[None])[0][:, None] * inj[..., None]).flatten(-2)
        return h, self.hyper_connection_mixer(h), k, v

    def logits(self, sample: torch.Tensor) -> torch.Tensor:
        return F.linear(sample.to(self.head.dtype), self.head).float()

    def unroll(self, tokens, hidden, pos, depth: int):
        """Parallel D-draft unroll of one window (one document, rows in position order).

        Depth k, anchor t: input token x[t+1+k], state hidden[t] (k=0) or the depth k-1
        multi_hidden of the same anchor, position pos[t]+k. Attention: depth-0 keys of rows
        0..t plus this anchor's own keys at depths 1..k. Rows past the end take token 0 and
        must be masked by the caller. Returns the per-depth sample hidden states [n, H].
        """
        n = tokens.shape[0]
        shifted = torch.cat([tokens[1:], tokens.new_zeros(depth)])
        causal = torch.ones(n, n, dtype=torch.bool, device=tokens.device).tril()
        ctx, chain, out, state = None, [], [], hidden
        for k in range(depth):
            def attend(q, kk, vv, k=k):
                if k == 0:
                    return _attention(q, kk, vv, causal, [])
                return _attention(q, ctx[0], ctx[1], causal, chain + [(kk, vv)])
            state, sample, kk, vv = self.block(shifted[k : k + n], state, pos + k, attend)
            if k == 0:
                ctx = (kk, vv)
            else:
                chain.append((kk, vv))
            out.append(sample)
        return out


def _fp8_ste(x: torch.Tensor) -> torch.Tensor:
    """x as the fp8 e4m3 KV cache returns it (saturating, scale 1.0), straight-through gradient."""
    y = x.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(x.dtype)
    return x + (y - x).detach()


def _attention(q, k0, v0, mask, diag):
    """q [n, heads, hd] over k0/v0 [m, kv_heads, hd] where mask [n, m], plus one key per (k, v) in
    diag that row i of q sees only at row i (the same anchor's earlier draft steps); GQA."""
    with torch.autocast(q.device.type, enabled=False):  # scores and softmax in fp32 as the served kernel
        return _attention_fp32(q, k0, v0, mask, diag)


def _attention_fp32(q, k0, v0, mask, diag):
    n, nh, hd = q.shape
    m, nkv = k0.shape[:2]
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    qg = q.view(n, nkv, nh // nkv, hd).permute(1, 2, 0, 3).to(acc)  # [kv, rep, n, hd]
    scale = hd ** -0.5
    s = qg @ k0.permute(1, 2, 0).to(acc)[:, None] * scale  # [kv, rep, n, n]
    s = s.masked_fill(~mask, float("-inf"))
    d = [(qg * kj.permute(1, 0, 2).to(acc)[:, None]).sum(-1, keepdim=True) * scale for kj, _ in diag]
    p = torch.cat([s, *d], dim=-1).softmax(-1)
    o = p[..., :m] @ v0.permute(1, 0, 2).to(acc)[:, None]
    for j, (_, vj) in enumerate(diag):
        o = o + p[..., m + j, None] * vj.permute(1, 0, 2).to(acc)[:, None]
    return o.permute(2, 0, 1, 3).reshape(n, nh, hd).to(q.dtype)


# ---------------------------------------------------------------- loading / export
class Snapshot:
    def __init__(self, path: str):
        from safetensors import safe_open

        self.path = path
        self.map = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self._open, self._files = safe_open, {}

    def get(self, name: str, device="cpu") -> torch.Tensor:
        f = self.map[name]
        if f not in self._files:
            self._files[f] = self._open(os.path.join(self.path, f), "pt", device=str(device))
        return self._files[f].get_tensor(name)


def trainable_names(model: MtpRef, trainable: str) -> list[str]:
    if trainable not in ("dense", "dense-norouter"):
        raise ValueError(f"--trainable {trainable}: dense or dense-norouter")
    skip = FROZEN_PREFIXES + (("layers.0.mlp.gate.",) if trainable == "dense-norouter" else ())
    return [n for n, _ in model.named_parameters() if n != "embed_tokens.weight" and not n.startswith(skip)]


def load_mtp_ref(snapshot: str, draft_vocab: str | None, device="cpu", trainable="dense",
                 experts_impl: str | None = None) -> MtpRef:
    """MtpRef with every tensor from `snapshot`: trainable ones fp32 masters, the rest frozen BF16."""
    cfg = text_config(snapshot)
    if experts_impl:
        cfg._experts_implementation = experts_impl
    snap = Snapshot(snapshot)
    ids = load_draft_vocab(draft_vocab, cfg.vocab_size) if draft_vocab else torch.arange(cfg.vocab_size)
    with torch.device("meta"):
        model = MtpRef(cfg, ids.numel())
    bf = torch.bfloat16
    exp = model.layers[0].mlp.experts
    for name in ("gate_up_proj", "down_proj"):
        setattr(exp, name, nn.Parameter(torch.empty_like(getattr(exp, name), dtype=bf), requires_grad=False))
    model.embed_tokens.weight = nn.Parameter(torch.empty_like(model.embed_tokens.weight, dtype=bf),
                                             requires_grad=False)
    model.to_empty(device=device)
    model.rotary = Qwen4ExpTextRotaryEmbedding(cfg).to(device)
    train = set(trainable_names(model, trainable))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.startswith(EXPERTS):
                continue
            src = next(snap.get(e, device) for e in EMBED_NAMES if e in snap.map) if n == "embed_tokens.weight" \
                else snap.get("mtp." + n, device)
            if tuple(src.shape) != tuple(p.shape):
                raise ValueError(f"mtp.{n}: checkpoint {tuple(src.shape)} vs model {tuple(p.shape)}")
            p.copy_(src)
            p.requires_grad_(n in train)
        I = cfg.moe_intermediate_size
        for e in range(cfg.num_experts):
            pre = f"mtp.{EXPERTS}{e}."
            w = {k: nvfp4_dequant(*(snap.get(f"{pre}{k}.{s}", device)
                                    for s in ("weight", "weight_scale", "weight_scale_2")))
                 for k in ("gate_proj", "up_proj", "down_proj")}
            exp.gate_up_proj[e, :I].copy_(w["gate_proj"])
            exp.gate_up_proj[e, I:].copy_(w["up_proj"])
            exp.down_proj[e].copy_(w["down_proj"])
        model.draft_ids.copy_(ids)
        lm = snap.get("lm_head.weight", device)
        model.head = nvfp4_round_trip(lm[ids.to(lm.device)]).to(bf)
    return model


def export(model: MtpRef, names: list[str], path: str, meta: dict | None = None) -> None:
    """Trained tensors under checkpoint names in checkpoint dtype (BF16), for splice.py."""
    from safetensors.torch import save_file

    params = dict(model.named_parameters())
    save_file({"mtp." + n: params[n].detach().to(torch.bfloat16).cpu().contiguous() for n in names}, path,
              metadata={k: str(v) for k, v in (meta or {}).items()})


# ---------------------------------------------------------------- loss / metrics
def depth_targets(n: int, depth: int, loss_mask: torch.Tensor):
    """Per depth k: (row of the target distribution, valid anchors). Anchor t predicts
    x[t+2+k] from the target's logits at row t+1+k. Counted when that token exists and both
    it and x[t+1] are generated tokens: serving drafts only from verified generated tokens,
    so the anchor whose label is the first generated token never drafts."""
    t = torch.arange(n, device=loss_mask.device)
    m = loss_mask.bool()
    out = []
    for k in range(depth):
        lab = t + 2 + k
        valid = (lab < n) & m[(t + 1).clamp(max=n - 1)] & m[lab.clamp(max=n - 1)]
        out.append(((t + 1 + k).clamp(max=n - 1), valid))
    return out


def full_to_draft(draft_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    m = torch.full((vocab_size,), -1, dtype=torch.long, device=draft_ids.device)
    m[draft_ids] = torch.arange(draft_ids.numel(), device=draft_ids.device)
    return m


def soft_ce(logits: torch.Tensor, topk_ids, topk_lp, lookup) -> torch.Tensor:
    """Sum over rows of -sum_x p(x) log q(x), p = target top-k renormalized, q over the draft
    vocab; target ids outside the draft vocab are unreachable and drop out (constant)."""
    logq = logits.log_softmax(-1)
    j = lookup[topk_ids.long()]
    p = topk_lp.float().softmax(-1) * (j >= 0)
    return -(p * logq.gather(1, j.clamp(min=0))).sum()


def head_loss(model, sample, topk_ids, topk_lp, lookup):
    return soft_ce(model.logits(sample), topk_ids, topk_lp, lookup)


def acceptance(logits, topk_ids, topk_lp, lookup, draft_ids):
    """Per row: T=1 acceptance sum_x min(p, q) over the stored top-k (a lower bound: tail mass
    of p is dropped), and T=0 match (draft argmax == target argmax)."""
    q = logits.softmax(-1)
    j = lookup[topk_ids.long()]
    p = topk_lp.float().exp()
    qx = q.gather(1, j.clamp(min=0)) * (j >= 0)
    t1 = torch.minimum(p, qx).sum(-1)
    t0 = draft_ids[logits.argmax(-1)] == topk_ids[:, 0].long()
    return t1, t0


def window_loss(model, w, depth, weights, lookup, use_checkpoint=True):
    """Sum over depths of weight_k * soft CE over valid anchors, plus counts per depth."""
    samples = model.unroll(w["tokens"], w["hidden"], w["positions"], depth)
    total, counts, ce = 0.0, [], []
    for k, (rows, valid) in enumerate(depth_targets(w["tokens"].shape[0], depth, w["loss_mask"])):
        idx = valid.nonzero().flatten()
        counts.append(int(idx.numel()))
        if not idx.numel():
            ce.append(0.0)
            continue
        args = (samples[k][idx], w["topk_ids"][rows[idx]], w["topk_logprobs"][rows[idx]], lookup)
        loss = checkpoint(head_loss, model, *args, use_reentrant=False) if use_checkpoint else head_loss(model, *args)
        ce.append(float(loss.detach()))
        total = total + weights[k] * loss
    return total, counts, ce
