
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'ks0': 'i64', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x1 = xindex // 2560
        x0 = (xindex % 2560)
        x2 = xindex
        tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
        tmp1 = tl.full([1], 0, tl.int32)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1], 124160, tl.int32)
        tmp4 = tmp0 < tmp3
        tmp5 = tmp2 & tmp4
        tmp6 = tl.full([1], 248320, tl.int32)
        tmp7 = tmp0 >= tmp6
        tmp8 = tmp0 < tmp6
        tmp9 = tmp7 & tmp8
        tmp10 = tmp5 | tmp9
        tmp11 = tmp10 == 0
        tmp12 = tmp10.to(tl.int64)
        tmp13 = tmp0.to(tl.int64)
        tmp14 = tmp9.to(tl.int64)
        tmp15 = tl.full([1], 124160, tl.int64)
        tmp16 = tmp14 * tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp12 * tmp17
        tl.device_assert(((0 <= tmp18) & (tmp18 < 124160)) | ~(xmask), "index out of bounds: 0 <= tmp18 < 124160")
        tmp20 = tl.load(in_ptr1 + (x0 + 2560*tmp18), xmask).to(tl.float32)
        tmp21 = tl.full([1], 0.0, tl.float32)
        tmp22 = tl.where(tmp11, tmp21, tmp20)
        tl.store(out_ptr0 + (x2), tmp22, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x5 = xindex
        x3 = (xindex % 32)
        x4 = xindex // 32
        tmp32 = tl.load(in_ptr2 + (x4 + 2*ks0), xmask, eviction_policy='evict_last')
        tmp44 = tl.load(in_ptr2 + (ks0 + x4), xmask, eviction_policy='evict_last')
        tmp50 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
        tmp23 = ((((x5 % 32)) % 3)).to(tl.int64)
        tmp24 = (tmp23).to(tl.int64)
        tmp25 = tl.full([1], 2, tl.int64)
        tmp26 = tmp24 == tmp25
        tmp27 = (x3).to(tl.int64)
        tmp28 = (tmp27).to(tl.int64)
        tmp29 = tl.full([1], 30, tl.int64)
        tmp30 = tmp28 < tmp29
        tmp31 = tmp26 & tmp30
        tmp33 = (tl.full([XBLOCK], 1048576, tl.int32)).to(tl.int32)
        tmp34 = tmp32 + tmp33
        tmp35 = tmp32 < 0
        tmp36 = tl.where(tmp35, tmp34, tmp32)
        tl.device_assert(((0 <= tmp36) & (tmp36 < 1048576)) | ~(xmask), "index out of bounds: 0 <= tmp36 < 1048576")
        tmp38 = tl.load(in_ptr3 + (x3 + 64*tmp36), xmask).to(tl.float32)
        tmp39 = tl.full([1], 1, tl.int64)
        tmp40 = tmp24 == tmp39
        tmp41 = tl.full([1], 33, tl.int64)
        tmp42 = tmp28 < tmp41
        tmp43 = tmp40 & tmp42
        tmp45 = tmp44 + tmp33
        tmp46 = tmp44 < 0
        tmp47 = tl.where(tmp46, tmp45, tmp44)
        tl.device_assert(((0 <= tmp47) & (tmp47 < 1048576)) | ~(xmask), "index out of bounds: 0 <= tmp47 < 1048576")
        tmp49 = tl.load(in_ptr3 + (x3 + 64*tmp47), xmask).to(tl.float32)
        tmp51 = tmp50 + tmp33
        tmp52 = tmp50 < 0
        tmp53 = tl.where(tmp52, tmp51, tmp50)
        tl.device_assert(((0 <= tmp53) & (tmp53 < 1048576)) | ~(xmask), "index out of bounds: 0 <= tmp53 < 1048576")
        tmp55 = tl.load(in_ptr3 + (x3 + 64*tmp53), xmask).to(tl.float32)
        tmp56 = tl.where(tmp43, tmp49, tmp55)
        tmp57 = tl.where(tmp31, tmp38, tmp56)
        tmp58 = tl.load(in_ptr3 + (32 + x3 + 64*tmp36), xmask).to(tl.float32)
        tmp59 = tl.load(in_ptr3 + (32 + x3 + 64*tmp47), xmask).to(tl.float32)
        tmp60 = tl.load(in_ptr3 + (32 + x3 + 64*tmp53), xmask).to(tl.float32)
        tmp61 = tl.where(tmp43, tmp59, tmp60)
        tmp62 = tl.where(tmp31, tmp58, tmp61)
        tl.store(out_ptr1 + (x5), tmp57, xmask)
        tl.store(out_ptr2 + (x5), tmp62, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int32)
    arg_1 = rand_strided((124160, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((3, 8192), (8193, 1), device='cuda:0', dtype=torch.int64)
    arg_3 = rand_strided((1048576, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_6 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_7 = 8193
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 20971520, 262144,


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
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.08830976
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
