#!/usr/bin/env python3
"""r10 offline NVFP4 requant of the LM heads from the BF16 base (CPU).

  r10_requant_heads.py --base <BF16 dir> [--shipped <snapshot dir>] [--ids ids-v2-K131072.txt.gz]
                       [--out <dir>]

Quantizes base lm_head.weight [248320, 2560] with the vLLM fork's codec
(nvfp4_quantize_mse, vLLM exp/r10-heads4; put that tree on PYTHONPATH), the code
VLLM_LM_HEAD_NVFP4 runs at load, once as the target head (full vocab) and once as the
MTP draft head (the draft-vocab rows, own global scale). Reports weight rel err for
NVFP4 absmax (rtn), NVFP4 MSE block scales (mse) and MXFP8 (ceil448 E8M0, the rule of
the shipped MXFP8 tensors), plus the chosen-factor histogram. --shipped checks that the
served head is the base head byte for byte (the QAD frozen rule), so requant from the
base equals requant from what serves today. --out writes the mse tensors
(lm_head{,_draft}.weight / .weight_scale / .weight_scale_2, ModelOpt layout) and the report.
"""
import argparse, gzip, hashlib, json, os, struct

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    NVFP4_MSE_SCALE_FACTORS, break_fp4_bytes, nvfp4_quantize_mse)


def head_file(d):  # a snapshot dir, or the .safetensors shard holding lm_head.weight
    if d.endswith(".safetensors"):
        return d
    return os.path.join(d, json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]["lm_head.weight"])


def tensor_sha(path, name="lm_head.weight"):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        a, b = json.loads(f.read(n))[name]["data_offsets"]
        f.seek(8 + n + a); h = hashlib.sha256(); left = b - a
        while left:
            c = f.read(min(left, 1 << 24)); h.update(c); left -= len(c)
    return h.hexdigest()


def rel_err(w, deq, rows=16384):
    num = den = 0.0
    for r in range(0, w.shape[0], rows):
        x = w[r:r + rows].float(); num += float(((deq(r, r + rows) - x) ** 2).sum()); den += float((x ** 2).sum())
    return (num / den) ** 0.5


def nvfp4_deq(q, s, gs):
    return lambda a, b: (break_fp4_bytes(q[a:b], torch.float32).view(-1, s.shape[1], 16)
                         * s[a:b].float().unsqueeze(-1) / gs).view(q[a:b].shape[0], -1)


def mxfp8_deq(w):
    def deq(a, b):
        blk = w[a:b].float(); blk = blk.view(blk.shape[0], -1, 32)
        e = torch.ceil(torch.log2(blk.abs().amax(-1).clamp(min=2.0 ** -126) / 448.0)).clamp(-127, 127)
        v = (blk / torch.exp2(e).unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        return (v * torch.exp2(e).unsqueeze(-1)).view(blk.shape[0], -1)
    return deq


def quant(w):
    gs = 2688.0 / w.abs().amax().float().clamp_min(1e-8)  # = Nvfp4OnlineLinearMethod
    q_r, s_r = nvfp4_quantize_mse(w, gs, factors=(1.0,))
    q_m, s_m = nvfp4_quantize_mse(w, gs)
    # factor actually chosen per block: ratio of the mse scale to the absmax scale
    ratio = (s_m.float() / s_r.float().clamp_min(1e-30)).flatten()
    hist = {f"{f:.2f}": float(((ratio - f).abs() < 0.025).float().mean()) for f in NVFP4_MSE_SCALE_FACTORS}
    rep = {"rows": w.shape[0], "rel_err_nvfp4_rtn": rel_err(w, nvfp4_deq(q_r, s_r, gs)),
           "rel_err_nvfp4_mse": rel_err(w, nvfp4_deq(q_m, s_m, gs)), "rel_err_mxfp8": rel_err(w, mxfp8_deq(w)),
           "mse_scale_ratio_hist": hist, "weight_scale_2": float(1 / gs),
           "bytes_nvfp4": q_m.numel() + s_m.numel(), "bytes_mxfp8": w.numel() + w.numel() // 32}
    return rep, (q_m, s_m, (1 / gs).reshape(()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--shipped")
    ap.add_argument("--ids", help="MTP draft vocab ids (one per line, .gz ok)")
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_num_threads(min(16, os.cpu_count() or 4))
    bf = head_file(a.base)
    rep = {"base_head_sha256": tensor_sha(bf)}
    if a.shipped:
        rep["shipped_head_sha256"] = tensor_sha(head_file(a.shipped))
        rep["frozen"] = rep["shipped_head_sha256"] == rep["base_head_sha256"]
        print("frozen (shipped head == base head):", rep["frozen"], flush=True)
    w = safe_open(bf, "pt", device="cpu").get_tensor("lm_head.weight")
    rep["main"], main_t = quant(w)
    print("main", json.dumps(rep["main"]), flush=True)
    out = {}
    if a.ids:
        op = gzip.open if a.ids.endswith(".gz") else open
        ids = sorted({int(l) for l in op(a.ids, "rt") if l.strip() and not l.startswith("#")})
        rep["draft"], draft_t = quant(w[torch.tensor(ids)].contiguous())
        rep["draft"]["ids"] = os.path.basename(a.ids)
        print("draft", json.dumps(rep["draft"]), flush=True)
        out.update(zip(("lm_head_draft.weight", "lm_head_draft.weight_scale", "lm_head_draft.weight_scale_2"), draft_t))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        out.update(zip(("lm_head.weight", "lm_head.weight_scale", "lm_head.weight_scale_2"), main_t))
        save_file({k: v.contiguous() for k, v in out.items()}, os.path.join(a.out, "lm_heads-nvfp4-mse.safetensors"))
        json.dump(rep, open(os.path.join(a.out, "r10-heads-report.json"), "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
