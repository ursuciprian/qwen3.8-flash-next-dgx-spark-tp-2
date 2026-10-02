#!/usr/bin/env python3
"""r9 dense-only NVFP4 requant (opus-kernel-17, 2026-10-02; plan docs/r9-dense-nvfp4.md).

CPU only. Builds a checkpoint snapshot = the shipped local-inference-lab/Qwen3.8-Flash-Next-NVFP4
@ 7c4f1bc1 with only the GDN in_proj_qkv / in_proj_z / out_proj and attention q/k/v/o_proj weights
replaced by weight-only NVFP4 (W4A16_NVFP4: E2M1 values, E4M3 scale per 16, FP32 global
weight_scale_2), quantized from the BF16 base Qwen/Qwen3.8-Flash-Next @ de4b8e4d. HC, shared
experts, a/b projections, indexer, MTP, experts and everything else are byte-identical.

  r9_requant_dense.py --shipped <snapshot dir> --base <BF16 dir> --out <new snapshot dir> [--limit-layers N]

Steps (each writes into <out>/r9-report.json):
  1. codec check: requantize the base BF16 vision linear_fc2 (shipped as W4A16_NVFP4, vision was
     not distilled) with this codec and compare bytes with the shipped tensors;
  2. frozen guard: MXFP8(base) == shipped MXFP8 for every in-scope tensor (values + E8M0 scales);
     refuses to write anything if a tensor is not bit-exact (the QAD rule: only frozen tensors
     may come from the base);
  3. quantize, one global scale per fused vLLM linear (qkvz = in_proj_qkv + in_proj_z, qkv_proj =
     q/k/v), so the W4A16 loader's shared-global-scale warning cannot fire;
  4. write the changed shard + config.json / hf_quant_config.json / index; every other file is a
     relative symlink to the same blob as the shipped snapshot (layout of an HF cache snapshot).
"""
import argparse, json, os, re, shutil, sys
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MID = (E2M1[1:] + E2M1[:-1]) / 2
SCOPE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|"
                   r"self_attn\.(q_proj|k_proj|v_proj|o_proj))$")


def fused_group(mod):
    """Modules that vLLM fuses into one linear share one global scale."""
    layer, rest = SCOPE.match(mod).group(1), mod.rsplit(".", 1)[1]
    key = {"in_proj_qkv": "qkvz", "in_proj_z": "qkvz", "q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv"}.get(rest, rest)
    return f"{layer}.{key}"


def nvfp4(w, gs2=None, right=False):
    """W [N,K] float -> (packed U8 [N,K/2], E4M3 scale [N,K/16], FP32 weight_scale_2)."""
    w = w.float()
    n, k = w.shape
    if gs2 is None:
        gs2 = w.abs().amax() / (6.0 * 448.0)
    gs2 = torch.as_tensor(gs2, dtype=torch.float32)
    blk = w.view(n, k // 16, 16)
    s = (blk.abs().amax(-1) / 6.0 / gs2).clamp(max=448.0).to(torch.float8_e4m3fn)
    denom = (s.float() * gs2).unsqueeze(-1)
    x = torch.where(denom > 0, blk / denom, torch.zeros_like(blk))
    mag = torch.bucketize(x.abs().clamp(max=6.0), MID, right=right)
    code = (mag | ((x < 0) & (mag > 0)).to(torch.long) << 3).view(n, k).to(torch.uint8)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed.contiguous(), s.contiguous(), gs2.reshape(())


def nvfp4_dequant(packed, s, gs2):
    lo, hi = packed & 0xF, packed >> 4
    code = torch.stack([lo, hi], -1).view(packed.shape[0], -1).long()
    val = E2M1[code & 7] * torch.where(code & 8 > 0, -1.0, 1.0)
    return (val.view(val.shape[0], -1, 16) * s.float().unsqueeze(-1)).view(val.shape[0], -1) * gs2


def mxfp8(w, rule):
    w = w.float()
    n, k = w.shape
    blk = w.view(n, k // 32, 32)
    amax = blk.abs().amax(-1).clamp(min=2.0 ** -126)
    if rule == "ceil448":
        e = torch.ceil(torch.log2(amax / 448.0))
    else:  # OCP MX spec: floor(log2(amax)) - emax(E4M3) = 8
        e = torch.floor(torch.log2(amax)) - 8
    e = e.clamp(-127, 127)
    vals = (blk / torch.exp2(e).unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).view(n, k)
    return vals, (e + 127).to(torch.uint8)


class Ckpt:
    def __init__(self, d):
        self.d = d
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.h = {}

    def get(self, name):
        f = self.map[name]
        if f not in self.h:
            self.h[f] = safe_open(os.path.join(self.d, f), "pt", device="cpu")
        return self.h[f].get_tensor(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-layers", type=int, default=None, help="dry run: only the first N in-scope layers, write nothing")
    a = ap.parse_args()
    torch.set_num_threads(min(16, os.cpu_count() or 4))
    shipped, base = Ckpt(a.shipped), Ckpt(a.base)
    rep = {"codec": {}, "guard": {}, "quant": {}}

    # 1. codec check on vision fc2 (W4A16_NVFP4 in the shipped checkpoint, vision unchanged by QAD)
    for i in (0, 13):
        m = f"model.visual.blocks.{i}.mlp.linear_fc2"
        wb = base.get(m + ".weight")
        pk, sc, g2 = shipped.get(m + ".weight"), shipped.get(m + ".weight_scale"), shipped.get(m + ".weight_scale_2")
        best = None
        for right in (False, True):
            p2, s2, gg = nvfp4(wb, right=right)
            r = {"right": right, "global_eq": bool(torch.isclose(gg, g2.float(), rtol=1e-6)),
                 "scale_eq": float((s2.view(torch.uint8) == sc.view(torch.uint8)).float().mean()),
                 "packed_eq": float((p2 == pk).float().mean())}
            best = r if best is None or r["packed_eq"] > best["packed_eq"] else best
        deq = nvfp4_dequant(pk, sc, g2.float())
        best["shipped_rel_err_vs_base"] = float((deq - wb.float()).norm() / wb.float().norm())
        rep["codec"][m] = best
        print("codec", m, best, flush=True)
    right = all(v["right"] for v in rep["codec"].values())

    mods = sorted({k.rsplit(".", 1)[0] for k in shipped.map if SCOPE.match(k.rsplit(".", 1)[0])},
                  key=lambda m: (int(SCOPE.match(m).group(1)), m))
    if a.limit_layers is not None:
        layers = sorted({int(SCOPE.match(m).group(1)) for m in mods})[: a.limit_layers]
        mods = [m for m in mods if int(SCOPE.match(m).group(1)) in layers]
    print(f"{len(mods)} in-scope modules", flush=True)

    # 2. frozen guard
    rule_ok = defaultdict(int)
    for m in mods:
        wb = base.get(m + ".weight")
        v, s = shipped.get(m + ".weight"), shipped.get(m + ".weight_scale")
        res = {}
        for rule in ("ceil448", "ocp"):
            v2, s2 = mxfp8(wb, rule)
            res[rule] = {"scale_eq": float((s2 == s).float().mean()),
                         "value_eq": float((v2.view(torch.uint8) == v.view(torch.uint8)).float().mean())}
            if res[rule]["scale_eq"] == 1.0 and res[rule]["value_eq"] == 1.0:
                rule_ok[rule] += 1
        rep["guard"][m] = res
    exact = max(rule_ok.values(), default=0)
    rep["guard_summary"] = {"modules": len(mods), "bit_exact_per_rule": dict(rule_ok)}
    print("guard", rep["guard_summary"], flush=True)
    if exact != len(mods):
        bad = [m for m in mods if not any(r["scale_eq"] == 1 and r["value_eq"] == 1 for r in rep["guard"][m].values())]
        rep["guard_failed"] = bad
        os.makedirs(a.out, exist_ok=True)
        json.dump(rep, open(os.path.join(a.out, "r9-report.json"), "w"), indent=1)
        sys.exit(f"frozen guard FAILED on {len(bad)} modules (first: {bad[:3]}); nothing written")

    # 3. quantize, shared global scale per fused linear
    amax = defaultdict(float)
    for m in mods:
        amax[fused_group(m)] = max(amax[fused_group(m)], float(base.get(m + ".weight").float().abs().amax()))
    new = {}
    for m in mods:
        wb = base.get(m + ".weight")
        gs2 = torch.tensor(amax[fused_group(m)] / (6.0 * 448.0), dtype=torch.float32)
        pk, sc, g2 = nvfp4(wb, gs2, right=right)
        deq = nvfp4_dequant(pk, sc, g2)
        mx = shipped.get(m + ".weight").float() * torch.exp2(shipped.get(m + ".weight_scale").float() - 127).repeat_interleave(32, 1)
        rep["quant"][m] = {"rel_err_nvfp4": float((deq - wb.float()).norm() / wb.float().norm()),
                           "rel_err_mxfp8_shipped": float((mx - wb.float()).norm() / wb.float().norm()),
                           "weight_scale_2": float(g2)}
        new[m + ".weight"], new[m + ".weight_scale"], new[m + ".weight_scale_2"] = pk, sc, g2
    errs = [v["rel_err_nvfp4"] for v in rep["quant"].values()]
    rep["quant_summary"] = {"rel_err_nvfp4_mean": sum(errs) / len(errs), "rel_err_nvfp4_max": max(errs),
                            "rel_err_mxfp8_mean": sum(v["rel_err_mxfp8_shipped"] for v in rep["quant"].values()) / len(errs)}
    print("quant", rep["quant_summary"], flush=True)
    if a.limit_layers is not None:
        os.makedirs(a.out, exist_ok=True)
        json.dump(rep, open(os.path.join(a.out, "r9-report.json"), "w"), indent=1)
        print("dry run: nothing written"); return

    # 4. write the snapshot
    os.makedirs(a.out, exist_ok=True)
    files = {shipped.map[m + ".weight"] for m in mods}
    wmap = dict(shipped.map)
    for f in files:
        names = [k for k, v in shipped.map.items() if v == f]
        out = {}
        for k in names:
            mod, leaf = k.rsplit(".", 1)
            if mod in mods and leaf in ("weight", "weight_scale"):
                continue
            out[k] = shipped.get(k)
        for k, t in new.items():
            if shipped.map[k.rsplit(".", 1)[0] + ".weight"] == f:
                out[k] = t
                wmap[k] = f
        save_file(out, os.path.join(a.out, f), metadata={"format": "pt"})
        print("wrote", f, len(out), "tensors", flush=True)
    for name in os.listdir(a.shipped):
        src = os.path.join(a.shipped, name)
        dst = os.path.join(a.out, name)
        if name in files or name in ("config.json", "hf_quant_config.json", "model.safetensors.index.json") or os.path.exists(dst):
            continue
        os.symlink(os.path.relpath(os.path.realpath(src), a.out), dst)
    idx = json.load(open(os.path.join(a.shipped, "model.safetensors.index.json")))
    idx["weight_map"] = wmap
    json.dump(idx, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    targets = sorted(mods)

    def requant(qc):
        groups = qc["config_groups"]
        groups["group_mxfp8_attention"]["targets"] = [t for t in groups["group_mxfp8_attention"]["targets"] if t not in mods]
        groups["group_w4a16_nvfp4_dense"] = {"weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                                             "targets": targets}
        if "quantized_layers" in qc:
            for t in targets:
                qc["quantized_layers"][t] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        return qc

    cfg = json.load(open(os.path.join(a.shipped, "config.json")))
    cfg["quantization_config"] = requant(cfg["quantization_config"])
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    hq = json.load(open(os.path.join(a.shipped, "hf_quant_config.json")))
    requant(hq.get("quantization", hq))
    json.dump(hq, open(os.path.join(a.out, "hf_quant_config.json"), "w"), indent=2)
    rep["base_revision"] = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
    rep["shipped_revision"] = "7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd"
    json.dump(rep, open(os.path.join(a.out, "r9-report.json"), "w"), indent=1)
    print("DONE", a.out)


if __name__ == "__main__":
    main()
