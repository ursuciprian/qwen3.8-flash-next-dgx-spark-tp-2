"""Numeric check: B12xGDNKernel.packed_decode vs TritonGDNKernel.packed_decode (+ RMSNorm) on random inputs."""
import os, torch
os.environ.setdefault("SGLANG_GDN_B12X", "1"); os.environ.setdefault("SGLANG_GDN_B12X_MAX_BS", "16")
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.kernels.gdn_b12x import B12xGDNKernel
torch.manual_seed(0); dev = "cuda"
HK, HV, K, V, slots = 8, 24, 128, 128, 40
def rms(x, eps=1e-6):
    xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
tri, b12 = TritonGDNKernel(), B12xGDNKernel()
for bs in (1, 4, 13):
    mixed = torch.randn(bs, HK * K * 2 + HV * V, device=dev, dtype=torch.bfloat16)
    a = torch.randn(bs, HV, device=dev, dtype=torch.bfloat16); b = torch.randn(bs, HV, device=dev, dtype=torch.bfloat16)
    A_log = (torch.randn(HV, device=dev) * 0.1).float(); dt_bias = (torch.randn(HV, device=dev) * 0.1).to(torch.bfloat16)
    state = (torch.randn(slots, HV, V, K, device=dev) * 0.1).to(torch.bfloat16)
    idx = torch.randperm(slots, device=dev)[:bs].to(torch.int32)
    s1, s2 = state.clone(), state.clone()
    o1 = tri.packed_decode(mixed, a, b, A_log=A_log, dt_bias=dt_bias, scale=K**-0.5, ssm_states=s1, cache_indices=idx, num_v_heads=HV, head_v_dim=V)
    o2 = b12.packed_decode(mixed, a, b, A_log=A_log, dt_bias=dt_bias, scale=K**-0.5, ssm_states=s2, cache_indices=idx, num_v_heads=HV, head_v_dim=V)
    ref = rms(o1)  # b12x output is rmsnorm(core) with unit weight and act(z)=1
    d_out = (ref.float() - o2.float()).abs().max().item(); d_state = (s1.float() - s2.float()).abs().max().item()
    fin = torch.isfinite(o2).all().item()
    print(f"bs={bs:3d} shape_ok={tuple(o2.shape)==tuple(o1.shape)} finite={fin} max|rms(triton)-b12x|={d_out:.4f} max|state diff|={d_state:.5f} out_rms={o2.float().pow(2).mean().sqrt().item():.3f}")
    assert fin and d_out < 0.05 and d_state < 0.02, "MISMATCH"
print("b12x GDN decode: PASS")
