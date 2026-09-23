"""Numeric check: B12xGDNKernel.target_verify vs TritonGDNKernel.target_verify (+RMSNorm on output); intermediate buffer and committed states compared."""
import os, torch
os.environ.setdefault("SGLANG_GDN_B12X", "1"); os.environ.setdefault("SGLANG_GDN_B12X_MAX_BS", "16")
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.kernels.gdn_b12x import B12xGDNKernel
torch.manual_seed(1); dev = "cuda"
HK, HV, K, V, slots, T, Rcap = 8, 24, 128, 128, 40, 4, 16
def rms(x, eps=1e-6):
    xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
tri, b12 = TritonGDNKernel(), B12xGDNKernel()
for B in (1, 3, 8):
    N = B * T
    q = torch.randn(1, N, HK, K, device=dev, dtype=torch.bfloat16); k = torch.randn(1, N, HK, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, N, HV, V, device=dev, dtype=torch.bfloat16)
    a = torch.randn(N, HV, device=dev, dtype=torch.bfloat16); b = torch.randn(N, HV, device=dev, dtype=torch.bfloat16)
    A_log = (torch.randn(HV, device=dev) * 0.1).float(); dt_bias = (torch.randn(HV, device=dev) * 0.1).to(torch.bfloat16)
    ssm = (torch.randn(slots, HV, V, K, device=dev) * 0.1).to(torch.bfloat16)
    idx = torch.randperm(slots, device=dev)[:B].to(torch.int32)
    qsl = (torch.arange(B + 1, device=dev) * T).to(torch.int32)
    inter1 = torch.zeros(Rcap + 1, T, HV, V, K, device=dev, dtype=torch.bfloat16); inter2 = inter1.clone()
    iidx = torch.randperm(Rcap, device=dev)[:B].to(torch.int32)
    s1, s2 = ssm.clone(), ssm.clone()
    o1 = tri.target_verify(A_log, dt_bias, q, k, v, a, b, ssm_states=s1, cache_indices=idx, query_start_loc=qsl,
                           intermediate_states_buffer=inter1, intermediate_state_indices=iidx, cache_steps=T, retrieve_parent_token=None)
    o2 = b12.target_verify(A_log, dt_bias, q, k, v, a, b, ssm_states=s2, cache_indices=idx, query_start_loc=qsl,
                           intermediate_states_buffer=inter2, intermediate_state_indices=iidx, cache_steps=T, retrieve_parent_token=None)
    ref = rms(o1.reshape(N, HV, V)); got = o2.reshape(N, HV, V)
    d_out = (ref.float() - got.float()).abs().max().item()
    d_inter = (inter1[iidx.long()].float() - inter2[iidx.long()].float()).abs().max().item()
    d_commit = (s1.float() - s2.float()).abs().max().item(); untouched = (s2.float() - ssm.float()).abs().max().item()
    print(f"B={B} T={T} out_shape_ok={tuple(o2.shape)==tuple(o1.shape)} finite={torch.isfinite(o2).all().item()} max|rms(tri)-b12x|={d_out:.4f} max|inter diff|={d_inter:.5f} committed_diff={d_commit:.5f} committed_untouched={untouched:.5f} fallbacks={b12.fallbacks} verify_calls={b12.verify_calls}")
    assert d_out < 0.05 and d_inter < 0.02 and d_commit == 0.0 and untouched == 0.0 and b12.fallbacks == 0, "MISMATCH"
print("b12x GDN target_verify: PASS")
