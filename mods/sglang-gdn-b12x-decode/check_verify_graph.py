"""target_verify under CUDA graph capture + replay with changed inputs, vs Triton eager."""
import os, torch
os.environ.setdefault("SGLANG_GDN_B12X", "1"); os.environ.setdefault("SGLANG_GDN_B12X_MAX_BS", "16")
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.kernels.gdn_b12x import B12xGDNKernel
torch.manual_seed(2); dev = "cuda"; HK, HV, K, V, slots, T, Rcap, B = 8, 24, 128, 128, 40, 4, 16, 4; N = B * T
def rms(x, eps=1e-6):
    xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
tri, b12 = TritonGDNKernel(), B12xGDNKernel()
def inputs():
    return dict(q=torch.randn(1, N, HK, K, device=dev, dtype=torch.bfloat16), k=torch.randn(1, N, HK, K, device=dev, dtype=torch.bfloat16),
                v=torch.randn(1, N, HV, V, device=dev, dtype=torch.bfloat16), a=torch.randn(1, N, HV, device=dev, dtype=torch.bfloat16), b=torch.randn(1, N, HV, device=dev, dtype=torch.bfloat16))
A_log = (torch.randn(HV, device=dev) * 0.1).float(); dt_bias = (torch.randn(HV, device=dev) * 0.1).to(torch.bfloat16)
ssm = (torch.randn(slots, HV, V, K, device=dev) * 0.1).to(torch.bfloat16)
idx = torch.randperm(slots, device=dev)[:B].to(torch.int32); iidx = torch.randperm(Rcap, device=dev)[:B].to(torch.int32)
qsl = (torch.arange(B + 1, device=dev) * T).to(torch.int32)
st = inputs()  # static buffers for the graph
inter_b = torch.zeros(Rcap + 1, T, HV, V, K, device=dev, dtype=torch.bfloat16)
def run_b12(): return b12.target_verify(A_log, dt_bias, st["q"], st["k"], st["v"], st["a"], st["b"], ssm_states=ssm, cache_indices=idx, query_start_loc=qsl, intermediate_states_buffer=inter_b, intermediate_state_indices=iidx, cache_steps=T, retrieve_parent_token=None)
# warmup (JIT prepare) then capture
for _ in range(2): out_static = run_b12()
torch.cuda.synchronize(); g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    with torch.cuda.graph(g): out_static = run_b12()
torch.cuda.current_stream().wait_stream(s)
for trial in range(3):
    new = inputs()
    for kk in st: st[kk].copy_(new[kk])
    inter_t = torch.zeros_like(inter_b); s1 = ssm.clone()
    o1 = tri.target_verify(A_log, dt_bias, new["q"], new["k"], new["v"], new["a"].reshape(N, HV), new["b"].reshape(N, HV), ssm_states=s1, cache_indices=idx, query_start_loc=qsl, intermediate_states_buffer=inter_t, intermediate_state_indices=iidx, cache_steps=T, retrieve_parent_token=None)
    inter_b.zero_(); g.replay(); torch.cuda.synchronize()
    d_out = (rms(o1.reshape(N, HV, V)).float() - out_static.reshape(N, HV, V).float()).abs().max().item()
    d_int = (inter_t[iidx.long()].float() - inter_b[iidx.long()].float()).abs().max().item()
    print(f"replay {trial}: max|rms(tri)-b12x graph|={d_out:.4f} max|inter diff|={d_int:.5f} finite={torch.isfinite(out_static).all().item()} fallbacks={b12.fallbacks}")
print("graph verify:", "PASS" if d_out < 0.05 and d_int < 0.02 else "MISMATCH")
