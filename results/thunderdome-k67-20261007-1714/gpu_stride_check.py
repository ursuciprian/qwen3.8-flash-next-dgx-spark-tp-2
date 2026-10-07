"""k67 GPU microtest (#113): which keys the sparse-GQA program attends for a 2-row MTP draft-reuse step.

Runs inside an image (GPU, no network): python3 gpu_stride_check.py. The variant is read from the installed b12x:
  served (21e0b201): prepare writes rows at a 2055 stride, the reuse attention runs programs.sparse (width 2051)
  k67a:              prepare writes rows at a 2051 stride (no tail), programs.sparse (2051)
  k67b:              prepare writes rows at a 2055 stride, programs.sparse_draft (2055)
The real prepare kernel builds the reuse rows; the attention runs the CuTe program compiled the way the plan compiles
it, launched the way _qsa_attention_op launches it. A zero query makes the softmax uniform, so each output row is the
mean of the V rows the kernel attended (with multiplicity): the closest candidate set says what was read.
Expected: served row 1 = shifted (request 0's chain + own columns 0..2046), k67a = own, k67b = own + chain.
Exit 0 when every row matches its variant's expected set, 3 otherwise (1 = an error).
"""
import dataclasses
import sys

import torch
import triton

from b12x.attention.qsa import _contract as C
from b12x.attention.qsa._draft_selection import _prepare_kernel
from b12x.attention.qsa._sparse_gqa import compile_sparse_paged_gqa, launch_sparse_paged_gqa
from b12x.attention.qsa._sparse_gqa_cute_config import BLOCK_N

W, T, PAGE, QH, KVH, D = 2051, 4, 16, 24, 2, 256
dev = torch.device("cuda")
store = W + T if "WIDTH + TAIL, tl.int64) + column" in getattr(_prepare_kernel, "src", "") else W
read = W + T if "sparse_draft" in {f.name for f in dataclasses.fields(C.QsaPrograms)} else W
variant = {(W + T, W): "served", (W, W): "k67a", (W + T, W + T): "k67b"}.get((store, read), f"unknown {store}/{read}")

g = torch.Generator().manual_seed(113)
reqs = []  # (anchor, query position at draft step 3, anchor selection ending in the newest tokens)
for anchor in (5000, 6000):
    older = torch.randperm(anchor - 2, generator=g)[: W - 3].sort().values
    reqs.append((anchor, anchor + 2, torch.cat((older, torch.arange(anchor - 2, anchor + 1))).to(torch.int32)))
ppr = (max(p for _, p, _ in reqs) + PAGE) // PAGE + 1
table = torch.arange(2 * ppr, dtype=torch.int32).view(2, ppr).to(dev)
k = torch.randn(2 * ppr, PAGE, KVH, D, generator=g).to(torch.bfloat16).to(dev)
v = torch.randn(2 * ppr, PAGE, KVH, D, generator=g)
for r, (anchor, pos, sel) in enumerate(reqs):  # loud V rows where the candidate sets differ
    for p in (*range(anchor - 2, pos + 1), *range(5001, 5003)):
        v[r * ppr + p // PAGE, p % PAGE] *= 30
v = v.to(torch.bfloat16).to(dev)

src_pos = torch.tensor([a for a, _, _ in reqs], dtype=torch.int64, device=dev)
src_sel = torch.stack([s for _, _, s in reqs]).contiguous().to(dev)
source_rows = torch.tensor([0, 1, -1, -1], dtype=torch.int64, device=dev)
num_src = torch.tensor([2], dtype=torch.int32, device=dev)
rids = torch.tensor([0, 1], dtype=torch.int32, device=dev)
qpos = torch.tensor([p for _, p, _ in reqs], dtype=torch.int64, device=dev)
flat = torch.full((4 * (W + T),), -1, dtype=torch.int32, device=dev)
_prepare_kernel[(2,)](src_pos, src_sel, source_rows, num_src, rids, qpos, flat, 2, 2, 4,
                      WIDTH=W, TAIL=T, BLOCK=triton.next_power_of_2(W + T), num_warps=4)
selected = flat[: 4 * store].view(4, store)

q = torch.zeros(2, QH, D, dtype=torch.bfloat16, device=dev)
prog = compile_sparse_paged_gqa(query=q, key_cache=k, value_cache=v, request_ids=rids,
                                selected_positions=torch.empty((2, read), dtype=torch.int32, device=dev),
                                direct_kv_warps=2)
splits = C._qwen_row_splits(2)
out = torch.empty_like(q)
po = torch.empty(2 * splits, QH, D, dtype=torch.float32, device=dev)
pl = torch.empty(2 * splits, QH, dtype=torch.float32, device=dev)
launch_sparse_paged_gqa(query=q, key_cache=k, value_cache=v, block_table=table, request_ids=rids,
                        selected_positions=selected[:2], query_positions=qpos, output=out,
                        partial_output=po.view(2, splits, QH, D), partial_lse=pl.view(2, splits, QH),
                        softmax_scale=1 / 16, block_n=BLOCK_N, splits=splits, direct_kv_warps=2, _prepared=prog)
torch.cuda.synchronize()


def mean_v(r, positions):
    idx = torch.tensor(positions, dtype=torch.int64)
    rows = v.float().cpu()[r * ppr + idx // PAGE, idx % PAGE]  # [n, KVH, D]
    return rows.mean(0).repeat_interleave(QH // KVH, 0)        # [QH, D]


ok = True
print(f"variant {variant}: prepare row stride {store}, reuse program width {read}")
for r, (anchor, pos, sel) in enumerate(reqs):
    own = sel.tolist()
    cand = {"own": own, "own+chain": own + list(range(anchor + 1, pos + 1))}
    if r == 1:  # served layout, row 1 read at r * 2051: request 0's 4-column tail, then own columns 0..2046
        a0, p0, _ = reqs[0]
        cand["shifted"] = [p for p in range(a0 + 1, a0 + 1 + T) if p <= p0 and p <= pos] + own[: W - T]
    err = {n: (out[r].float().cpu() - mean_v(r, s)).abs().max().item() for n, s in cand.items()}
    best = min(err, key=err.get)
    want = {"served": "shifted" if r else "own", "k67a": "own", "k67b": "own+chain"}.get(variant)
    good = best == want and err[best] < 0.25 * min(e for n, e in err.items() if n != best)
    ok &= good
    print(f"row {r}: read = {best} ({'as expected' if good else f'EXPECTED {want}'}); max abs err "
          + ", ".join(f"{n} {e:.5f}" for n, e in err.items()))
sys.exit(0 if ok else 3)
