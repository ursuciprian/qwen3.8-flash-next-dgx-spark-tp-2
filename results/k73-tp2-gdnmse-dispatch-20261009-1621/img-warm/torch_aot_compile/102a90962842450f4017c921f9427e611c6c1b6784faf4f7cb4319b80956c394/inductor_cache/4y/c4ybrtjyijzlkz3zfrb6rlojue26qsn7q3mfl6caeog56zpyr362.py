
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'kernel_num_gb': 0.004390912, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp9 = tl.load(in_ptr0 + (x1 + 2*ks0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (ks0 + x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = ((((x2 % 32)) % 3)).to(tl.int64)
    tmp1 = (tmp0).to(tl.int64)
    tmp2 = tl.full([1], 2, tl.int64)
    tmp3 = tmp1 == tmp2
    tmp4 = (x0).to(tl.int64)
    tmp5 = (tmp4).to(tl.int64)
    tmp6 = tl.full([1], 30, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp3 & tmp7
    tmp10 = (tl.full([XBLOCK], 1048576, tl.int32)).to(tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tl.device_assert(((0 <= tmp13) & (tmp13 < 1048576)) | ~(xmask), "index out of bounds: 0 <= tmp13 < 1048576")
    tmp15 = tl.load(in_ptr1 + (x0 + 64*tmp13), xmask).to(tl.float32)
    tmp16 = tl.full([1], 1, tl.int64)
    tmp17 = tmp1 == tmp16
    tmp18 = tl.full([1], 33, tl.int64)
    tmp19 = tmp5 < tmp18
    tmp20 = tmp17 & tmp19
    tmp22 = tmp21 + tmp10
    tmp23 = tmp21 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp21)
    tl.device_assert(((0 <= tmp24) & (tmp24 < 1048576)) | ~(xmask), "index out of bounds: 0 <= tmp24 < 1048576")
    tmp26 = tl.load(in_ptr1 + (x0 + 64*tmp24), xmask).to(tl.float32)
    tmp28 = tmp27 + tmp10
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tl.device_assert(((0 <= tmp30) & (tmp30 < 1048576)) | ~(xmask), "index out of bounds: 0 <= tmp30 < 1048576")
    tmp32 = tl.load(in_ptr1 + (x0 + 64*tmp30), xmask).to(tl.float32)
    tmp33 = tl.where(tmp20, tmp26, tmp32)
    tmp34 = tl.where(tmp8, tmp15, tmp33)
    tmp35 = tl.load(in_ptr1 + (32 + x0 + 64*tmp13), xmask).to(tl.float32)
    tmp36 = tl.load(in_ptr1 + (32 + x0 + 64*tmp24), xmask).to(tl.float32)
    tmp37 = tl.load(in_ptr1 + (32 + x0 + 64*tmp30), xmask).to(tl.float32)
    tmp38 = tl.where(tmp20, tmp36, tmp37)
    tmp39 = tl.where(tmp8, tmp35, tmp38)
    tl.store(out_ptr0 + (x2), tmp34, xmask)
    tl.store(out_ptr1 + (x2), tmp39, xmask)


def get_args():
    arg_0 = rand_strided((3, 8192), (8193, 1), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((1048576, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = 8193
    return arg_0, arg_1, arg_2, arg_3, arg_4, 262144,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.004390912
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
