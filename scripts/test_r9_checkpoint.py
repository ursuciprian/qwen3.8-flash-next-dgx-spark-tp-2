#!/usr/bin/env python3
"""CPU loader check for the r9 dense NVFP4 snapshot (opus-kernel-17).

  python3 test_r9_checkpoint.py <new snapshot dir> <shipped snapshot dir>

Asserts: every shipped tensor name survives except the replaced weight/weight_scale pairs; the
replaced modules carry U8 [N,K/2] / F8_E4M3 [N,K/16] / F32 [] with one global scale per fused
linear; every index entry resolves to a file that holds it; vLLM's ModelOptMixedPrecisionConfig
resolves the fused vLLM linears (in_proj_qkvz, qkv_proj, out_proj, o_proj) to W4A16_NVFP4 and the
untouched ones (in_proj_ba, shared expert, indexer) to MXFP8.
"""
import json, os, re, sys
from safetensors import safe_open

new, old = sys.argv[1], sys.argv[2]
ni = json.load(open(f"{new}/model.safetensors.index.json"))["weight_map"]
oi = json.load(open(f"{old}/model.safetensors.index.json"))["weight_map"]
scope = re.compile(r"layers\.\d+\.(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj))\.")
dropped = set(oi) - set(ni)
assert all(scope.search(k) for k in dropped), sorted(dropped)[:5]
added = set(ni) - set(oi)
assert added and all(k.endswith(".weight_scale_2") and scope.search(k) for k in added), sorted(added)[:5]
handles = {}
def t(k):
    f = ni[k]
    handles.setdefault(f, safe_open(f"{new}/{f}", "pt", device="cpu"))
    return handles[f].get_tensor(k)
mods = sorted({k.rsplit(".", 1)[0] for k in added})
assert len(mods) in (156, 108, 72, 84), len(mods)  # all / gdn / qkvz / attn_out
gs = {}
for m in mods:
    w, s, g = t(m + ".weight"), t(m + ".weight_scale"), t(m + ".weight_scale_2")
    k2 = w.shape[1]
    assert str(w.dtype) == "torch.uint8" and str(s.dtype) == "torch.float8_e4m3fn" and g.numel() == 1, m
    assert s.shape == (w.shape[0], k2 * 2 // 16), (m, w.shape, s.shape)
    layer, leaf = re.search(r"layers\.(\d+)\.\w+\.(\w+)$", m).groups()
    key = (layer, {"in_proj_qkv": "qkvz", "in_proj_z": "qkvz", "q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv"}.get(leaf, leaf))
    gs.setdefault(key, set()).add(float(g))
assert all(len(v) == 1 for v in gs.values()), "fused shards must share weight_scale_2"
for k in list(ni)[:: max(1, len(ni) // 400)]:  # sample: every index entry lives in its file
    f = ni[k]
    handles.setdefault(f, safe_open(f"{new}/{f}", "pt", device="cpu"))
    assert k in handles[f].keys(), k
from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig
qc = json.load(open(f"{new}/config.json"))["quantization_config"]
cfg = ModelOptMixedPrecisionConfig.from_config(qc)
cfg.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
                              "in_proj_ba": ["in_proj_b", "in_proj_a"], "gate_up_proj": ["gate_proj", "up_proj"]}
P = "model.language_model.layers"
algo = lambda m: "W4A16_NVFP4" if m in mods else "MXFP8"
want = {f"{P}.0.linear_attn.in_proj_qkvz": algo(f"{P}.0.linear_attn.in_proj_qkv"), f"{P}.0.linear_attn.out_proj": algo(f"{P}.0.linear_attn.out_proj"),
        f"{P}.3.self_attn.qkv_proj": algo(f"{P}.3.self_attn.q_proj"), f"{P}.3.self_attn.o_proj": algo(f"{P}.3.self_attn.o_proj"),
        f"{P}.0.linear_attn.in_proj_ba": "MXFP8", f"{P}.3.self_attn.indexer.index_qk_proj": "MXFP8",
        f"{P}.0.mlp.shared_expert.gate_up_proj": "MXFP8"}
got = {p: cfg._resolve_quant_algo(p) for p in want}
assert got == want, got
print(f"OK: {len(mods)} NVFP4 modules, {len(gs)} fused global scales, {len(dropped)} tensors dropped, {len(added)} added; resolver {got}")
