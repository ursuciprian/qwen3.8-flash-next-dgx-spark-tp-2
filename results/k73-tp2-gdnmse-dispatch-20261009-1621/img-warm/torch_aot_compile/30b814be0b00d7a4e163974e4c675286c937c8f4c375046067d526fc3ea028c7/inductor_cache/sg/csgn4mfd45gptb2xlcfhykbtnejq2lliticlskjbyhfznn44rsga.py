r"""
Compile-time auto-tuning block: 

import torch
from math import inf, nan
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.async_compile import AsyncCompile

async_compile = AsyncCompile()
generate_example_value = AlgorithmSelectorCache.generate_example_value
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
get_raw_stream = torch._C._cuda_getCurrentRawStream


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/30b814be0b00d7a4e163974e4c675286c937c8f4c375046067d526fc3ea028c7/inductor_cache/47/c47uzmva6n6lnrg6ftt7qmvl7l3s64cbm34rv2f4e666x2dnz7sw.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
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
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
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
        triton_poi_fused_0.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.08830976
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/30b814be0b00d7a4e163974e4c675286c937c8f4c375046067d526fc3ea028c7/inductor_cache/jd/cjd2yjbgtxgcrfs3lrwrcartjnyqqkjtxtvdmrlrhurwyry67wmg.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'kernel_num_gb': 0.100663296, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 12)
    x2 = xindex // 3072
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + 512*x1 + 6656*x2), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/30b814be0b00d7a4e163974e4c675286c937c8f4c375046067d526fc3ea028c7/inductor_cache/vm/cvm2z5r6tm4rq2liv7ea35z3wropqzzi7uspwke3y5czmz6vrkut.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_2 = async_compile.triton('triton_red_fused_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_red_fused_2', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_2(in_ptr0, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x0 = (xindex % 12)
        x1 = xindex // 12
        _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x3 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_2 + 512*x0 + 6656*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tmp1 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tl.store(out_ptr0 + (x3), tmp4, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x4 = xindex
        _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_5 = r0_index
            tmp6 = tl.load(in_ptr0 + (6144 + r0_5 + 6656*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp7 = tmp6.to(tl.float32)
            tmp8 = tmp7 * tmp7
            tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        tl.store(out_ptr1 + (x4), tmp10, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 6656), (6656, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 12, 1), (12, 1, 98304), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192, 1, 1), (1, 8192, 8192), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 98304, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_red_fused_2.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.054951936
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/30b814be0b00d7a4e163974e4c675286c937c8f4c375046067d526fc3ea028c7/inductor_cache/js/cjsj5t7plfuovif7sc44pfyzn624nzznmoftcg6x723aephgnxwr.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'xnumel_3': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 4, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None, 'no_x_dim_3': False, 'xnumel_3': None}, 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel_0, xnumel_1, xnumel_2, xnumel_3, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(xnumel_2, XBLOCK)
    num_xblocks_3 = num_xblocks_2 + tl.cdiv(xnumel_3, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 64)
        x1 = ((xindex // 64) % 12)
        x2 = xindex // 768
        x3 = xindex // 64
        tmp0 = (x0).to(tl.int32)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = (x0).to(tl.int64)
        tmp4 = (tmp3).to(tl.int64)
        tmp5 = tl.full([1], 32, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr0 + (512*x1 + 6656*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (x3), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full([1], 256.0, tl.float32)
        tmp11 = (tmp9 / tmp10)
        tmp12 = tl.full([1], 1e-06, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp8 * tmp14
        tmp16 = tl.load(in_ptr2 + (x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tl.full([1], 1.0, tl.float32)
        tmp19 = tmp17 + tmp18
        tmp20 = tmp15 * tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tl.load(in_ptr3 + (32*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = tl.load(in_ptr0 + (32 + 512*x1 + 6656*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp25 * tmp14
        tmp27 = tl.load(in_ptr2 + (32 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 + tmp18
        tmp30 = tmp26 * tmp29
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tl.load(in_ptr4 + (32*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tmp31 * tmp32
        tmp34 = tmp23 - tmp33
        tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
        tmp36 = tl.where(tmp6, tmp34, tmp35)
        tmp37 = tmp0 >= tmp5
        tmp38 = tl.full([1], 64, tl.int64)
        tmp39 = tmp0 < tmp38
        tmp40 = tl.load(in_ptr0 + (32 + 512*x1 + 6656*x2 + ((-32) + x0)), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tl.load(in_ptr1 + (x3), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
        tmp43 = tl.full([1], 256.0, tl.float32)
        tmp44 = (tmp42 / tmp43)
        tmp45 = tl.full([1], 1e-06, tl.float32)
        tmp46 = tmp44 + tmp45
        tmp47 = libdevice.rsqrt(tmp46)
        tmp48 = tmp41 * tmp47
        tmp49 = tl.load(in_ptr2 + (32 + ((-32) + x0)), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp50 = tmp49.to(tl.float32)
        tmp51 = tl.full([1], 1.0, tl.float32)
        tmp52 = tmp50 + tmp51
        tmp53 = tmp48 * tmp52
        tmp54 = tmp53.to(tl.float32)
        tmp55 = tl.load(in_ptr3 + (32*x2 + ((-32) + x0)), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp56 = tmp54 * tmp55
        tmp57 = tl.load(in_ptr0 + (512*x1 + 6656*x2 + ((-32) + x0)), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp58 = tmp57.to(tl.float32)
        tmp59 = tmp58 * tmp47
        tmp60 = tl.load(in_ptr2 + ((-32) + x0), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp61 = tmp60.to(tl.float32)
        tmp62 = tmp61 + tmp51
        tmp63 = tmp59 * tmp62
        tmp64 = tmp63.to(tl.float32)
        tmp65 = tl.load(in_ptr4 + (32*x2 + ((-32) + x0)), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp66 = tmp64 * tmp65
        tmp67 = tmp56 + tmp66
        tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
        tmp69 = tl.where(tmp37, tmp67, tmp68)
        tmp70 = tl.where(tmp6, tmp36, tmp69)
        tl.store(out_ptr0 + (x0 + 256*x3), tmp70, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x4 = (xindex % 192)
        x5 = ((xindex // 192) % 12)
        x6 = xindex // 2304
        x7 = xindex // 192
        tmp71 = tl.load(in_ptr0 + (64 + x4 + 512*x5 + 6656*x6), xmask).to(tl.float32)
        tmp73 = tl.load(in_ptr1 + (x7), xmask, eviction_policy='evict_last')
        tmp80 = tl.load(in_ptr2 + (64 + x4), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp72 = tmp71.to(tl.float32)
        tmp74 = tl.full([1], 256.0, tl.float32)
        tmp75 = (tmp73 / tmp74)
        tmp76 = tl.full([1], 1e-06, tl.float32)
        tmp77 = tmp75 + tmp76
        tmp78 = libdevice.rsqrt(tmp77)
        tmp79 = tmp72 * tmp78
        tmp81 = tmp80.to(tl.float32)
        tmp82 = tl.full([1], 1.0, tl.float32)
        tmp83 = tmp81 + tmp82
        tmp84 = tmp79 * tmp83
        tmp85 = tmp84.to(tl.float32)
        tl.store(out_ptr1 + (x4 + 256*x7), tmp85, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_2
        x8 = (xindex % 64)
        x9 = xindex // 64
        tmp86 = (x8).to(tl.int32)
        tmp87 = tl.full([1], 0, tl.int64)
        tmp88 = tmp86 >= tmp87
        tmp89 = (x8).to(tl.int64)
        tmp90 = (tmp89).to(tl.int64)
        tmp91 = tl.full([1], 32, tl.int64)
        tmp92 = tmp90 < tmp91
        tmp93 = tl.load(in_ptr0 + (6144 + 6656*x9 + (x8)), tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp94 = tmp93.to(tl.float32)
        tmp95 = tl.load(in_ptr5 + (x9), tmp92 & xmask, eviction_policy='evict_last', other=0.0)
        tmp96 = tl.full([1], 256.0, tl.float32)
        tmp97 = (tmp95 / tmp96)
        tmp98 = tl.full([1], 1e-06, tl.float32)
        tmp99 = tmp97 + tmp98
        tmp100 = libdevice.rsqrt(tmp99)
        tmp101 = tmp94 * tmp100
        tmp102 = tl.load(in_ptr6 + (x8), tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp103 = tmp102.to(tl.float32)
        tmp104 = tl.full([1], 1.0, tl.float32)
        tmp105 = tmp103 + tmp104
        tmp106 = tmp101 * tmp105
        tmp107 = tmp106.to(tl.float32)
        tmp108 = tl.load(in_ptr3 + (32*x9 + (x8)), tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp109 = tmp107 * tmp108
        tmp110 = tl.load(in_ptr0 + (6176 + 6656*x9 + (x8)), tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp111 = tmp110.to(tl.float32)
        tmp112 = tmp111 * tmp100
        tmp113 = tl.load(in_ptr6 + (32 + (x8)), tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp114 = tmp113.to(tl.float32)
        tmp115 = tmp114 + tmp104
        tmp116 = tmp112 * tmp115
        tmp117 = tmp116.to(tl.float32)
        tmp118 = tl.load(in_ptr4 + (32*x9 + (x8)), tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp119 = tmp117 * tmp118
        tmp120 = tmp109 - tmp119
        tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
        tmp122 = tl.where(tmp92, tmp120, tmp121)
        tmp123 = tmp86 >= tmp91
        tmp124 = tl.full([1], 64, tl.int64)
        tmp125 = tmp86 < tmp124
        tmp126 = tl.load(in_ptr0 + (6176 + 6656*x9 + ((-32) + x8)), tmp123 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp127 = tmp126.to(tl.float32)
        tmp128 = tl.load(in_ptr5 + (x9), tmp123 & xmask, eviction_policy='evict_last', other=0.0)
        tmp129 = tl.full([1], 256.0, tl.float32)
        tmp130 = (tmp128 / tmp129)
        tmp131 = tl.full([1], 1e-06, tl.float32)
        tmp132 = tmp130 + tmp131
        tmp133 = libdevice.rsqrt(tmp132)
        tmp134 = tmp127 * tmp133
        tmp135 = tl.load(in_ptr6 + (32 + ((-32) + x8)), tmp123 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp136 = tmp135.to(tl.float32)
        tmp137 = tl.full([1], 1.0, tl.float32)
        tmp138 = tmp136 + tmp137
        tmp139 = tmp134 * tmp138
        tmp140 = tmp139.to(tl.float32)
        tmp141 = tl.load(in_ptr3 + (32*x9 + ((-32) + x8)), tmp123 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp142 = tmp140 * tmp141
        tmp143 = tl.load(in_ptr0 + (6144 + 6656*x9 + ((-32) + x8)), tmp123 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp144 = tmp143.to(tl.float32)
        tmp145 = tmp144 * tmp133
        tmp146 = tl.load(in_ptr6 + ((-32) + x8), tmp123 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp147 = tmp146.to(tl.float32)
        tmp148 = tmp147 + tmp137
        tmp149 = tmp145 * tmp148
        tmp150 = tmp149.to(tl.float32)
        tmp151 = tl.load(in_ptr4 + (32*x9 + ((-32) + x8)), tmp123 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp152 = tmp150 * tmp151
        tmp153 = tmp142 + tmp152
        tmp154 = tl.full(tmp153.shape, 0.0, tmp153.dtype)
        tmp155 = tl.where(tmp123, tmp153, tmp154)
        tmp156 = tl.where(tmp92, tmp122, tmp155)
        tl.store(out_ptr2 + (x8 + 256*x9), tmp156, xmask)
    elif pid < num_xblocks_3:
        pid_offset = pid - num_xblocks_2
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_3
        x10 = (xindex % 192)
        x11 = xindex // 192
        tmp157 = tl.load(in_ptr0 + (6208 + x10 + 6656*x11), xmask).to(tl.float32)
        tmp159 = tl.load(in_ptr5 + (x11), xmask, eviction_policy='evict_last')
        tmp166 = tl.load(in_ptr6 + (64 + x10), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp158 = tmp157.to(tl.float32)
        tmp160 = tl.full([1], 256.0, tl.float32)
        tmp161 = (tmp159 / tmp160)
        tmp162 = tl.full([1], 1e-06, tl.float32)
        tmp163 = tmp161 + tmp162
        tmp164 = libdevice.rsqrt(tmp163)
        tmp165 = tmp158 * tmp164
        tmp167 = tmp166.to(tl.float32)
        tmp168 = tl.full([1], 1.0, tl.float32)
        tmp169 = tmp167 + tmp168
        tmp170 = tmp165 * tmp169
        tmp171 = tmp170.to(tl.float32)
        tl.store(out_ptr3 + (x10 + 256*x11), tmp171, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 6656), (6656, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 12, 1), (12, 1, 98304), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((8192, 1, 1), (1, 8192, 8192), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_7 = rand_strided((8192, 12, 64), (3072, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_8 = rand_strided((8192, 12, 192), (3072, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_9 = rand_strided((8192, 1, 64), (256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_10 = rand_strided((8192, 1, 192), (256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 6291456, 18874368, 524288, 1572864,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.152897536
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')

async_compile.wait(globals())
del async_compile

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
with torch.cuda._DeviceGuard(0):
    raw_stream0 = get_raw_stream(0)
raw_stream0 = get_raw_stream(0)
arg0_1 = generate_example_value((8192,), (1,), 'cuda:0', torch.int32, 0, (8192,))
arg2_1 = generate_example_value((124160, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (124160, 2560))
arg14_1 = generate_example_value((3, 8192), (8193, 1), 'cuda:0', torch.int64, 0, (3, 8192))
arg20_1 = generate_example_value((1048576, 64), (64, 1), 'cuda:0', torch.bfloat16, 0, (1048576, 64))
buf0 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
buf19 = generate_example_value((8192, 32), (32, 1), 'cuda:0', torch.bfloat16, 0, (8192, 32))
buf20 = generate_example_value((8192, 32), (32, 1), 'cuda:0', torch.bfloat16, 0, (8192, 32))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_0.run(arg0_1, arg2_1, arg14_1, arg20_1, buf0, buf19, buf20, 8193, 20971520, 262144, stream=raw_stream0)
del arg0_1, arg2_1, arg14_1, arg20_1, buf0

raw_stream0 = get_raw_stream(0)
buf17 = generate_example_value((8192, 6656), (6656, 1), 'cuda:0', torch.bfloat16, 0, (8192, 6656))
buf28 = generate_example_value((8192, 12, 256), (3072, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 12, 256))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_1.run(buf17, buf28, 25165824, stream=raw_stream0)
del buf28

raw_stream0 = get_raw_stream(0)
buf18 = generate_example_value((8192, 12, 1), (12, 1, 98304), 'cuda:0', torch.float32, 0, (8192, 12, 1))
buf24 = generate_example_value((8192, 1, 1), (1, 8192, 8192), 'cuda:0', torch.float32, 0, (8192, 1, 1))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_2.run(buf17, buf18, buf24, 98304, 8192, stream=raw_stream0)

raw_stream0 = get_raw_stream(0)
arg18_1 = generate_example_value((256,), (1,), 'cuda:0', torch.bfloat16, 0, (256,))
arg19_1 = generate_example_value((256,), (1,), 'cuda:0', torch.bfloat16, 0, (256,))
buf21 = generate_example_value((8192, 12, 64), (3072, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 12, 64))
buf22 = generate_example_value((8192, 12, 192), (3072, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 12, 192))
buf25 = generate_example_value((8192, 1, 64), (256, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 1, 64))
buf26 = generate_example_value((8192, 1, 192), (256, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 1, 192))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_3.run(buf17, buf18, arg18_1, buf19, buf20, buf24, arg19_1, buf21, buf22, buf25, buf26, 6291456, 18874368, 524288, 1572864, stream=raw_stream0)
del buf19, buf20, buf17, buf18, buf24, arg18_1, arg19_1, buf21, buf22, buf25, buf26

"""
# AOT ID: ['53_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/30b814be0b00d7a4e163974e4c675286c937c8f4c375046067d526fc3ea028c7/inductor_cache/47/c47uzmva6n6lnrg6ftt7qmvl7l3s64cbm34rv2f4e666x2dnz7sw.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
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
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
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
        triton_poi_fused_0.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.08830976
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/30b814be0b00d7a4e163974e4c675286c937c8f4c375046067d526fc3ea028c7/inductor_cache/jd/cjd2yjbgtxgcrfs3lrwrcartjnyqqkjtxtvdmrlrhurwyry67wmg.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'kernel_num_gb': 0.100663296, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 12)
    x2 = xindex // 3072
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + 512*x1 + 6656*x2), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/30b814be0b00d7a4e163974e4c675286c937c8f4c375046067d526fc3ea028c7/inductor_cache/vm/cvm2z5r6tm4rq2liv7ea35z3wropqzzi7uspwke3y5czmz6vrkut.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_2 = async_compile.triton('triton_red_fused_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_red_fused_2', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_2(in_ptr0, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x0 = (xindex % 12)
        x1 = xindex // 12
        _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x3 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_2 + 512*x0 + 6656*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tmp1 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tl.store(out_ptr0 + (x3), tmp4, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x4 = xindex
        _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_5 = r0_index
            tmp6 = tl.load(in_ptr0 + (6144 + r0_5 + 6656*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp7 = tmp6.to(tl.float32)
            tmp8 = tmp7 * tmp7
            tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        tl.store(out_ptr1 + (x4), tmp10, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 6656), (6656, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 12, 1), (12, 1, 98304), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192, 1, 1), (1, 8192, 8192), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 98304, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_red_fused_2.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.054951936
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/30b814be0b00d7a4e163974e4c675286c937c8f4c375046067d526fc3ea028c7/inductor_cache/js/cjsj5t7plfuovif7sc44pfyzn624nzznmoftcg6x723aephgnxwr.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'xnumel_3': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 4, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None, 'no_x_dim_3': False, 'xnumel_3': None}, 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel_0, xnumel_1, xnumel_2, xnumel_3, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(xnumel_2, XBLOCK)
    num_xblocks_3 = num_xblocks_2 + tl.cdiv(xnumel_3, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 64)
        x1 = ((xindex // 64) % 12)
        x2 = xindex // 768
        x3 = xindex // 64
        tmp0 = (x0).to(tl.int32)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = (x0).to(tl.int64)
        tmp4 = (tmp3).to(tl.int64)
        tmp5 = tl.full([1], 32, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr0 + (512*x1 + 6656*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (x3), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full([1], 256.0, tl.float32)
        tmp11 = (tmp9 / tmp10)
        tmp12 = tl.full([1], 1e-06, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp8 * tmp14
        tmp16 = tl.load(in_ptr2 + (x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tl.full([1], 1.0, tl.float32)
        tmp19 = tmp17 + tmp18
        tmp20 = tmp15 * tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tl.load(in_ptr3 + (32*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = tl.load(in_ptr0 + (32 + 512*x1 + 6656*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp25 * tmp14
        tmp27 = tl.load(in_ptr2 + (32 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 + tmp18
        tmp30 = tmp26 * tmp29
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tl.load(in_ptr4 + (32*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tmp31 * tmp32
        tmp34 = tmp23 - tmp33
        tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
        tmp36 = tl.where(tmp6, tmp34, tmp35)
        tmp37 = tmp0 >= tmp5
        tmp38 = tl.full([1], 64, tl.int64)
        tmp39 = tmp0 < tmp38
        tmp40 = tl.load(in_ptr0 + (32 + 512*x1 + 6656*x2 + ((-32) + x0)), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tl.load(in_ptr1 + (x3), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
        tmp43 = tl.full([1], 256.0, tl.float32)
        tmp44 = (tmp42 / tmp43)
        tmp45 = tl.full([1], 1e-06, tl.float32)
        tmp46 = tmp44 + tmp45
        tmp47 = libdevice.rsqrt(tmp46)
        tmp48 = tmp41 * tmp47
        tmp49 = tl.load(in_ptr2 + (32 + ((-32) + x0)), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp50 = tmp49.to(tl.float32)
        tmp51 = tl.full([1], 1.0, tl.float32)
        tmp52 = tmp50 + tmp51
        tmp53 = tmp48 * tmp52
        tmp54 = tmp53.to(tl.float32)
        tmp55 = tl.load(in_ptr3 + (32*x2 + ((-32) + x0)), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp56 = tmp54 * tmp55
        tmp57 = tl.load(in_ptr0 + (512*x1 + 6656*x2 + ((-32) + x0)), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp58 = tmp57.to(tl.float32)
        tmp59 = tmp58 * tmp47
        tmp60 = tl.load(in_ptr2 + ((-32) + x0), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp61 = tmp60.to(tl.float32)
        tmp62 = tmp61 + tmp51
        tmp63 = tmp59 * tmp62
        tmp64 = tmp63.to(tl.float32)
        tmp65 = tl.load(in_ptr4 + (32*x2 + ((-32) + x0)), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp66 = tmp64 * tmp65
        tmp67 = tmp56 + tmp66
        tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
        tmp69 = tl.where(tmp37, tmp67, tmp68)
        tmp70 = tl.where(tmp6, tmp36, tmp69)
        tl.store(out_ptr0 + (x0 + 256*x3), tmp70, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x4 = (xindex % 192)
        x5 = ((xindex // 192) % 12)
        x6 = xindex // 2304
        x7 = xindex // 192
        tmp71 = tl.load(in_ptr0 + (64 + x4 + 512*x5 + 6656*x6), xmask).to(tl.float32)
        tmp73 = tl.load(in_ptr1 + (x7), xmask, eviction_policy='evict_last')
        tmp80 = tl.load(in_ptr2 + (64 + x4), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp72 = tmp71.to(tl.float32)
        tmp74 = tl.full([1], 256.0, tl.float32)
        tmp75 = (tmp73 / tmp74)
        tmp76 = tl.full([1], 1e-06, tl.float32)
        tmp77 = tmp75 + tmp76
        tmp78 = libdevice.rsqrt(tmp77)
        tmp79 = tmp72 * tmp78
        tmp81 = tmp80.to(tl.float32)
        tmp82 = tl.full([1], 1.0, tl.float32)
        tmp83 = tmp81 + tmp82
        tmp84 = tmp79 * tmp83
        tmp85 = tmp84.to(tl.float32)
        tl.store(out_ptr1 + (x4 + 256*x7), tmp85, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_2
        x8 = (xindex % 64)
        x9 = xindex // 64
        tmp86 = (x8).to(tl.int32)
        tmp87 = tl.full([1], 0, tl.int64)
        tmp88 = tmp86 >= tmp87
        tmp89 = (x8).to(tl.int64)
        tmp90 = (tmp89).to(tl.int64)
        tmp91 = tl.full([1], 32, tl.int64)
        tmp92 = tmp90 < tmp91
        tmp93 = tl.load(in_ptr0 + (6144 + 6656*x9 + (x8)), tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp94 = tmp93.to(tl.float32)
        tmp95 = tl.load(in_ptr5 + (x9), tmp92 & xmask, eviction_policy='evict_last', other=0.0)
        tmp96 = tl.full([1], 256.0, tl.float32)
        tmp97 = (tmp95 / tmp96)
        tmp98 = tl.full([1], 1e-06, tl.float32)
        tmp99 = tmp97 + tmp98
        tmp100 = libdevice.rsqrt(tmp99)
        tmp101 = tmp94 * tmp100
        tmp102 = tl.load(in_ptr6 + (x8), tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp103 = tmp102.to(tl.float32)
        tmp104 = tl.full([1], 1.0, tl.float32)
        tmp105 = tmp103 + tmp104
        tmp106 = tmp101 * tmp105
        tmp107 = tmp106.to(tl.float32)
        tmp108 = tl.load(in_ptr3 + (32*x9 + (x8)), tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp109 = tmp107 * tmp108
        tmp110 = tl.load(in_ptr0 + (6176 + 6656*x9 + (x8)), tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp111 = tmp110.to(tl.float32)
        tmp112 = tmp111 * tmp100
        tmp113 = tl.load(in_ptr6 + (32 + (x8)), tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp114 = tmp113.to(tl.float32)
        tmp115 = tmp114 + tmp104
        tmp116 = tmp112 * tmp115
        tmp117 = tmp116.to(tl.float32)
        tmp118 = tl.load(in_ptr4 + (32*x9 + (x8)), tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp119 = tmp117 * tmp118
        tmp120 = tmp109 - tmp119
        tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
        tmp122 = tl.where(tmp92, tmp120, tmp121)
        tmp123 = tmp86 >= tmp91
        tmp124 = tl.full([1], 64, tl.int64)
        tmp125 = tmp86 < tmp124
        tmp126 = tl.load(in_ptr0 + (6176 + 6656*x9 + ((-32) + x8)), tmp123 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp127 = tmp126.to(tl.float32)
        tmp128 = tl.load(in_ptr5 + (x9), tmp123 & xmask, eviction_policy='evict_last', other=0.0)
        tmp129 = tl.full([1], 256.0, tl.float32)
        tmp130 = (tmp128 / tmp129)
        tmp131 = tl.full([1], 1e-06, tl.float32)
        tmp132 = tmp130 + tmp131
        tmp133 = libdevice.rsqrt(tmp132)
        tmp134 = tmp127 * tmp133
        tmp135 = tl.load(in_ptr6 + (32 + ((-32) + x8)), tmp123 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp136 = tmp135.to(tl.float32)
        tmp137 = tl.full([1], 1.0, tl.float32)
        tmp138 = tmp136 + tmp137
        tmp139 = tmp134 * tmp138
        tmp140 = tmp139.to(tl.float32)
        tmp141 = tl.load(in_ptr3 + (32*x9 + ((-32) + x8)), tmp123 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp142 = tmp140 * tmp141
        tmp143 = tl.load(in_ptr0 + (6144 + 6656*x9 + ((-32) + x8)), tmp123 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp144 = tmp143.to(tl.float32)
        tmp145 = tmp144 * tmp133
        tmp146 = tl.load(in_ptr6 + ((-32) + x8), tmp123 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp147 = tmp146.to(tl.float32)
        tmp148 = tmp147 + tmp137
        tmp149 = tmp145 * tmp148
        tmp150 = tmp149.to(tl.float32)
        tmp151 = tl.load(in_ptr4 + (32*x9 + ((-32) + x8)), tmp123 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp152 = tmp150 * tmp151
        tmp153 = tmp142 + tmp152
        tmp154 = tl.full(tmp153.shape, 0.0, tmp153.dtype)
        tmp155 = tl.where(tmp123, tmp153, tmp154)
        tmp156 = tl.where(tmp92, tmp122, tmp155)
        tl.store(out_ptr2 + (x8 + 256*x9), tmp156, xmask)
    elif pid < num_xblocks_3:
        pid_offset = pid - num_xblocks_2
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_3
        x10 = (xindex % 192)
        x11 = xindex // 192
        tmp157 = tl.load(in_ptr0 + (6208 + x10 + 6656*x11), xmask).to(tl.float32)
        tmp159 = tl.load(in_ptr5 + (x11), xmask, eviction_policy='evict_last')
        tmp166 = tl.load(in_ptr6 + (64 + x10), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp158 = tmp157.to(tl.float32)
        tmp160 = tl.full([1], 256.0, tl.float32)
        tmp161 = (tmp159 / tmp160)
        tmp162 = tl.full([1], 1e-06, tl.float32)
        tmp163 = tmp161 + tmp162
        tmp164 = libdevice.rsqrt(tmp163)
        tmp165 = tmp158 * tmp164
        tmp167 = tmp166.to(tl.float32)
        tmp168 = tl.full([1], 1.0, tl.float32)
        tmp169 = tmp167 + tmp168
        tmp170 = tmp165 * tmp169
        tmp171 = tmp170.to(tl.float32)
        tl.store(out_ptr3 + (x10 + 256*x11), tmp171, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 6656), (6656, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 12, 1), (12, 1, 98304), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((8192, 1, 1), (1, 8192, 8192), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_7 = rand_strided((8192, 12, 64), (3072, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_8 = rand_strided((8192, 12, 192), (3072, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_9 = rand_strided((8192, 1, 64), (256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_10 = rand_strided((8192, 1, 192), (256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 6291456, 18874368, 524288, 1572864,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.152897536
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1 = args
        args.clear()
        s72 = arg1_1
        s47 = arg4_1
        s18 = arg7_1
        s7 = arg15_1
        assert_size_stride(arg0_1, (s18, ), (1, ), 'input')
        assert_size_stride(arg2_1, (124160, 2560), (2560, 1), 'input')
        assert_size_stride(arg14_1, (3, s18), (s7, 1), 'input')
        assert_size_stride(arg20_1, (1048576, 64), (64, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            arg14_1 = copy_if_misaligned(arg14_1)
            arg20_1 = copy_if_misaligned(arg20_1)
            buf0 = empty_strided_cuda((s18, 2560), (2560, 1), torch.bfloat16)
            buf19 = empty_strided_cuda((s18, 32), (32, 1), torch.bfloat16)
            buf20 = empty_strided_cuda((s18, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [ge, lt, and_, ge_1, lt_1, and__1, or_, invert, unsqueeze, masked_fill_, add, sub, mul_2, embedding], Original ATen: [aten.ge, aten.lt, aten.bitwise_and, aten.bitwise_or, aten.bitwise_not, aten.unsqueeze, aten.masked_fill, aten.add, aten.sub, aten.mul, aten.embedding]
            triton_poi_fused_0_xnumel_0 = 2560*s18
            triton_poi_fused_0_xnumel_1 = 32*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_0.run(arg0_1, arg2_1, arg14_1, arg20_1, buf0, buf19, buf20, s7, triton_poi_fused_0_xnumel_0, triton_poi_fused_0_xnumel_1, stream=raw_stream0)
            del arg0_1
            del arg20_1
            del arg2_1
            # Topologically Sorted Source Nodes: [ge, lt, and_, ge_1, lt_1, and__1, or_, invert, unsqueeze, masked_fill_, add, sub, mul_2, embedding, all_reduce], Original ATen: [aten.ge, aten.lt, aten.bitwise_and, aten.bitwise_or, aten.bitwise_not, aten.unsqueeze, aten.masked_fill, aten.add, aten.sub, aten.mul, aten.embedding, vllm.all_reduce]
            buf1 = torch.ops.vllm.all_reduce.default(buf0, 'tp:0')
            del buf0
            buf2 = buf1
            assert_size_stride(buf2, (s18, 2560), (2560, 1), 'torch.ops.vllm.all_reduce.default')
            assert_alignment(buf2, 16, 'torch.ops.vllm.all_reduce.default')
            del buf1
            assert_size_stride(arg3_1, (s47, 10240), (10240, 1), 'input')
            assert_size_stride(arg5_1, (8192, 4, 2560), (10240, 2560, 1), 'input')
            arg3_1 = copy_if_misaligned(arg3_1)
            # Topologically Sorted Source Nodes: [qwen3_8_flash_next_mtp_feedback], Original ATen: [vllm.qwen3_8_flash_next_mtp_feedback]
            torch.ops.vllm.qwen3_8_flash_next_mtp_feedback.default(buf2, arg3_1, arg5_1, arg6_1)
            del arg3_1
            del arg6_1
            del buf2
            assert_size_stride(arg8_1, (10240, ), (1, ), 'input')
            assert_size_stride(arg9_1, (8192, 10240), (10240, 1), 'input')
            arg8_1 = copy_if_misaligned(arg8_1)
            # Topologically Sorted Source Nodes: [hyperconnection_grouped_rmsnorm], Original ATen: [aten.slice, aten.view]
            torch.ops.b12x.hyperconnection_grouped_rmsnorm.default(reinterpret_tensor(arg5_1, (s18, 10240), (10240, 1), 0), arg8_1, arg9_1, 1e-06, 18708, zero_centered=True)
            del arg8_1
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear], Original ATen: [aten.slice, vllm.b12x_blockscaled_linear]
            buf7 = torch.ops.vllm.b12x_blockscaled_linear.default(reinterpret_tensor(arg9_1, (s18, 10240), (10240, 1), 0), None, 336, arg10_1)
            del arg10_1
            buf8 = buf7
            assert_size_stride(buf8, (s18, 336), (336, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf8, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf7
            assert_size_stride(arg11_1, (8192, 320), (320, 1), 'input')
            # Topologically Sorted Source Nodes: [getitem_2, hyperconnection_scaled_silu], Original ATen: [aten.slice, b12x.hyperconnection_scaled_silu]
            torch.ops.b12x.hyperconnection_scaled_silu.default(reinterpret_tensor(buf8, (s18, 320), (336, 1), 0), arg11_1, 18709)
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear_1], Original ATen: [aten.slice, vllm.b12x_blockscaled_linear]
            buf11 = torch.ops.vllm.b12x_blockscaled_linear.default(reinterpret_tensor(arg11_1, (s18, 320), (320, 1), 0), None, 10240, arg12_1)
            del arg11_1
            del arg12_1
            buf12 = buf11
            assert_size_stride(buf12, (s18, 10240), (10240, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf12, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf11
            assert_size_stride(arg13_1, (8192, 2560), (2560, 1), 'input')
            # Topologically Sorted Source Nodes: [reshape, hyperconnection_gate_mean], Original ATen: [aten.slice, b12x.hyperconnection_gate_mean]
            torch.ops.b12x.hyperconnection_gate_mean.default(reinterpret_tensor(arg9_1, (s18, 10240), (10240, 1), 0), buf12, arg13_1, 18710)
            del arg9_1
            del buf12
            assert_size_stride(arg16_1, (8192, 2051), (2051, 1), 'input')
            # Topologically Sorted Source Nodes: [qwen3_8_flash_next_qsa_project_inputs], Original ATen: [aten.slice]
            buf15 = torch.ops.vllm.qwen3_8_flash_next_qsa_project_inputs.default(arg14_1, reinterpret_tensor(arg13_1, (s18, 2560), (2560, 1), 0), arg16_1, 6656, arg17_1)
            del arg14_1
            del arg16_1
            del arg17_1
            buf17 = buf15
            assert_size_stride(buf17, (s18, 6656), (6656, 1), 'torch.ops.vllm.qwen3_8_flash_next_qsa_project_inputs.default')
            assert_alignment(buf17, 16, 'torch.ops.vllm.qwen3_8_flash_next_qsa_project_inputs.default')
            del buf15
            buf28 = empty_strided_cuda((s18, 12, 256), (3072, 256, 1), torch.bfloat16)
            buf18 = empty_strided_cuda((s18, 12, 1), (12, 1, 12*s18), torch.float32)
            buf24 = empty_strided_cuda((s18, 1, 1), (1, s18, s18), torch.float32)
            # Topologically Sorted Source Nodes: [split, unflatten, chunk, rms_norm_default, unflatten_1, rms_norm_default_1, flatten_3], Original ATen: [aten.split_with_sizes, aten.view, aten.split, vllm_ir.rms_norm, aten.clone]
            triton_poi_fused_1_xnumel = 3072*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_1.run(buf17, buf28, triton_poi_fused_1_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [split, unflatten, chunk, rms_norm_default, unflatten_1, rms_norm_default_1, flatten_3], Original ATen: [aten.split_with_sizes, aten.view, aten.split, vllm_ir.rms_norm, aten.clone]
            triton_red_fused_2_xnumel_0 = 12*s18
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_2.run(buf17, buf18, buf24, triton_red_fused_2_xnumel_0, s18, stream=raw_stream0)
            assert_size_stride(arg18_1, (256, ), (1, ), 'input')
            assert_size_stride(arg19_1, (256, ), (1, ), 'input')
            arg18_1 = copy_if_misaligned(arg18_1)
            arg19_1 = copy_if_misaligned(arg19_1)
            buf23 = empty_strided_cuda((s18, 12, 256), (3072, 256, 1), torch.bfloat16)
            buf21 = reinterpret_tensor(buf23, (s18, 12, 64), (3072, 256, 1), 0)  # alias
            buf22 = reinterpret_tensor(buf23, (s18, 12, 192), (3072, 256, 1), 64)  # alias
            buf27 = empty_strided_cuda((s18, 1, 256), (256, 256, 1), torch.bfloat16)
            buf25 = reinterpret_tensor(buf27, (s18, 1, 64), (256, 256, 1), 0)  # alias
            buf26 = reinterpret_tensor(buf27, (s18, 1, 192), (256, 256, 1), 64)  # alias
            # Topologically Sorted Source Nodes: [split, unflatten, chunk, float_1, add_1, rms_norm_default, getitem_20, chunk_2, unsqueeze_1, mul_3, unsqueeze_2, mul_4, sub_1, mul_5, mul_6, add_3, cat, getitem_21, cat_1, unflatten_1, float_2, add_2, rms_norm_default_1, getitem_24, chunk_3, unsqueeze_3, mul_7, unsqueeze_4, mul_8, sub_2, mul_9, mul_10, add_4, cat_2, getitem_25, cat_3], Original ATen: [aten.split_with_sizes, aten.view, aten.split, aten._to_copy, aten.add, vllm_ir.rms_norm, aten.slice, aten.unsqueeze, aten.mul, aten.sub, aten.cat]
            triton_poi_fused_3_xnumel_0 = 768*s18
            triton_poi_fused_3_xnumel_1 = 2304*s18
            triton_poi_fused_3_xnumel_2 = 64*s18
            triton_poi_fused_3_xnumel_3 = 192*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_3.run(buf17, buf18, arg18_1, buf19, buf20, buf24, arg19_1, buf21, buf22, buf25, buf26, triton_poi_fused_3_xnumel_0, triton_poi_fused_3_xnumel_1, triton_poi_fused_3_xnumel_2, triton_poi_fused_3_xnumel_3, stream=raw_stream0)
            del arg18_1
            del arg19_1
            del buf18
            del buf19
            del buf20
            del buf24
            buf29 = empty_strided_cuda((s18, 12, 256), (3072, 256, 1), torch.bfloat16)
        return (reinterpret_tensor(arg13_1, (s18, 2560), (2560, 1), 0), buf23, buf27, reinterpret_tensor(buf17, (s18, 1, 256), (6656, 256, 1), 6400), buf29, reinterpret_tensor(buf28, (s18, 3072), (3072, 1), 0), reinterpret_tensor(arg5_1, (s18, 10240), (10240, 1), 0), reinterpret_tensor(buf8, (s18, 4), (336, 1), 320), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg1_1 = 8192
    arg2_1 = rand_strided((124160, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((8192, 10240), (10240, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = 8192
    arg5_1 = rand_strided((8192, 4, 2560), (10240, 2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = None
    arg7_1 = 8192
    arg8_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((8192, 10240), (10240, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = None
    arg11_1 = rand_strided((8192, 320), (320, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = None
    arg13_1 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((3, 8192), (8193, 1), device='cuda:0', dtype=torch.int64)
    arg15_1 = 8193
    arg16_1 = rand_strided((8192, 2051), (2051, 1), device='cuda:0', dtype=torch.int32)
    arg17_1 = None
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((1048576, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
