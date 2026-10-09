
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
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'kernel_num_gb': 0.083918848, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 2560
    x0 = (xindex % 2560)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 124160, tl.int32)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 248320, tl.int32)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp0 >= tmp3
    tmp7 = tmp6 & tmp4
    tmp8 = tmp5 | tmp7
    tmp9 = tmp8 == 0
    tmp10 = tmp8.to(tl.int64)
    tmp11 = tmp0.to(tl.int64)
    tmp12 = tmp5.to(tl.int64)
    tmp13 = tl.full([1], 124160, tl.int64)
    tmp14 = tmp12 * tmp13
    tmp15 = tmp7.to(tl.int64)
    tmp16 = tmp15 * tmp13
    tmp17 = tmp14 + tmp16
    tmp18 = tmp11 - tmp17
    tmp19 = tmp10 * tmp18
    tl.device_assert(((0 <= tmp19) & (tmp19 < 124160)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 124160")
    tmp21 = tl.load(in_ptr1 + (x0 + 2560*tmp19), xmask).to(tl.float32)
    tmp22 = tl.full([1], 0.0, tl.float32)
    tmp23 = tl.where(tmp9, tmp22, tmp21)
    tl.store(out_ptr0 + (x2), tmp23, xmask)


def get_args():
    arg_0 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int32)
    arg_1 = rand_strided((124160, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, 20971520,


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
    num_gb = 0.083918848
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
