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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/6ae13a9bbce434c673241fcde54ba9f2b6cd5af3a26521600639df215ef08c73/inductor_cache/el/cel7s5sxlnrkn4wka6eyvols4ihf2x4qt4a4deuqvn2u44pmag2j.py
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add => add_60
# Graph fragment:
#   %getitem_6 : Tensor "bf16[s18, 2560][2560, 1]cuda:0" = PlaceHolder[target=getitem_6]
#   %getitem_7 : Tensor "bf16[s18, 2560][2560, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %add_60 : Tensor "bf16[s18, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, %getitem_7), kwargs = {})
#   return %add_60
triton_poi_fused_add_0 = async_compile.triton('triton_poi_fused_add_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 167772160}, 'kernel_num_gb': 0.12582912, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/6ae13a9bbce434c673241fcde54ba9f2b6cd5af3a26521600639df215ef08c73/inductor_cache/jd/cjd2yjbgtxgcrfs3lrwrcartjnyqqkjtxtvdmrlrhurwyry67wmg.py
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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/6ae13a9bbce434c673241fcde54ba9f2b6cd5af3a26521600639df215ef08c73/inductor_cache/vm/cvm2z5r6tm4rq2liv7ea35z3wropqzzi7uspwke3y5czmz6vrkut.py
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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/6ae13a9bbce434c673241fcde54ba9f2b6cd5af3a26521600639df215ef08c73/inductor_cache/w4/cw4sz32f2nkkka5bf4qdwlkobbbehnersxill6f6yf7wm7jncabc.py
# Topologically Sorted Source Nodes: [getitem_19, chunk_1, arange, mod_1, eq_1, lt_1, and__1, getitem_24, mod, eq, lt, and_, getitem_22, getitem_23, where, where_1, arange_1, mod_3, eq_3, lt_3, and__3, getitem_27, mod_2, eq_2, lt_2, and__2, getitem_25, getitem_26, where_2, where_3], Original ATen: [aten.index, aten.split, aten.arange, aten.remainder, aten.eq, aten.lt, aten.bitwise_and, aten.select, aten.where]
# Source node to ATen node mapping:
#   and_ => bitwise_and
#   and__1 => bitwise_and_1
#   and__2 => bitwise_and_2
#   and__3 => bitwise_and_3
#   arange => iota
#   arange_1 => iota_1
#   chunk_1 => split_1
#   eq => eq_76
#   eq_1 => eq_77
#   eq_2 => eq_83
#   eq_3 => eq_84
#   getitem_19 => index
#   getitem_22 => select
#   getitem_23 => select_1
#   getitem_24 => select_2
#   getitem_25 => select_3
#   getitem_26 => select_4
#   getitem_27 => select_5
#   lt => lt
#   lt_1 => lt_1
#   lt_2 => lt_2
#   lt_3 => lt_3
#   mod => remainder
#   mod_1 => remainder_1
#   mod_2 => remainder_2
#   mod_3 => remainder_3
#   where => where
#   where_1 => where_1
#   where_2 => where_2
#   where_3 => where_3
# Graph fragment:
#   %arg15_1 : Tensor "i64[3, s18][s7, 1]cuda:0" = PlaceHolder[target=arg15_1]
#   %arg21_1 : Tensor "bf16[1048576, 64][64, 1]cuda:0" = PlaceHolder[target=arg21_1]
#   %index : Tensor "bf16[3, s18, 64][64*s18, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg21_1, [%arg15_1]), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 32, -1), kwargs = {})
#   %iota : Tensor "i64[32][1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %remainder_1 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%iota, 3), kwargs = {})
#   %eq_77 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%remainder_1, 2), kwargs = {})
#   %lt_1 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota, 30), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_77, %lt_1), kwargs = {})
#   %select_2 : Tensor "bf16[s18, 32][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_21, 0, 2), kwargs = {})
#   %remainder : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%iota, 3), kwargs = {})
#   %eq_76 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%remainder, 1), kwargs = {})
#   %lt : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota, 33), kwargs = {})
#   %bitwise_and : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_76, %lt), kwargs = {})
#   %select : Tensor "bf16[s18, 32][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_21, 0, 1), kwargs = {})
#   %select_1 : Tensor "bf16[s18, 32][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_21, 0, 0), kwargs = {})
#   %where : Tensor "bf16[s18, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and, %select, %select_1), kwargs = {})
#   %where_1 : Tensor "bf16[s18, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_1, %select_2, %where), kwargs = {})
#   %iota_1 : Tensor "i64[32][1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %remainder_3 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%iota_1, 3), kwargs = {})
#   %eq_84 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%remainder_3, 2), kwargs = {})
#   %lt_3 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota_1, 30), kwargs = {})
#   %bitwise_and_3 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_84, %lt_3), kwargs = {})
#   %select_5 : Tensor "bf16[s18, 32][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_22, 0, 2), kwargs = {})
#   %remainder_2 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%iota_1, 3), kwargs = {})
#   %eq_83 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%remainder_2, 1), kwargs = {})
#   %lt_2 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota_1, 33), kwargs = {})
#   %bitwise_and_2 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_83, %lt_2), kwargs = {})
#   %select_3 : Tensor "bf16[s18, 32][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_22, 0, 1), kwargs = {})
#   %select_4 : Tensor "bf16[s18, 32][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_22, 0, 0), kwargs = {})
#   %where_2 : Tensor "bf16[s18, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_2, %select_3, %select_4), kwargs = {})
#   %where_3 : Tensor "bf16[s18, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_3, %select_5, %where_2), kwargs = {})
#   return %where_1,%where_3
triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_3 = async_compile.triton('triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 2293760}, 'kernel_num_gb': 0.004390912, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/6ae13a9bbce434c673241fcde54ba9f2b6cd5af3a26521600639df215ef08c73/inductor_cache/x6/cx66si2rdcs4kzmuu4w6rf5m3ksvj72avsnjglciuanaocej2qfk.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
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
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 4, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None, 'no_x_dim_3': False, 'xnumel_3': None}, 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel_0, xnumel_1, xnumel_2, xnumel_3, XBLOCK : tl.constexpr):
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
        triton_poi_fused_4.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_4.benchmark_all_configs(*args)


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
buf18 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
buf17 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_add_0.run(buf18, buf17, 20971520, stream=raw_stream0)
del buf18, buf17

raw_stream0 = get_raw_stream(0)
buf34 = generate_example_value((8192, 6656), (6656, 1), 'cuda:0', torch.bfloat16, 0, (8192, 6656))
buf45 = generate_example_value((8192, 12, 256), (3072, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 12, 256))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_1.run(buf34, buf45, 25165824, stream=raw_stream0)
del buf45

raw_stream0 = get_raw_stream(0)
buf35 = generate_example_value((8192, 12, 1), (12, 1, 98304), 'cuda:0', torch.float32, 0, (8192, 12, 1))
buf41 = generate_example_value((8192, 1, 1), (1, 8192, 8192), 'cuda:0', torch.float32, 0, (8192, 1, 1))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_2.run(buf34, buf35, buf41, 98304, 8192, stream=raw_stream0)

raw_stream0 = get_raw_stream(0)
arg15_1 = generate_example_value((3, 8192), (8193, 1), 'cuda:0', torch.int64, 0, (3, 8192))
arg21_1 = generate_example_value((1048576, 64), (64, 1), 'cuda:0', torch.bfloat16, 0, (1048576, 64))
buf36 = generate_example_value((8192, 32), (32, 1), 'cuda:0', torch.bfloat16, 0, (8192, 32))
buf37 = generate_example_value((8192, 32), (32, 1), 'cuda:0', torch.bfloat16, 0, (8192, 32))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_3.run(arg15_1, arg21_1, buf36, buf37, 8193, 262144, stream=raw_stream0)
del arg15_1, arg21_1

raw_stream0 = get_raw_stream(0)
arg19_1 = generate_example_value((256,), (1,), 'cuda:0', torch.bfloat16, 0, (256,))
arg20_1 = generate_example_value((256,), (1,), 'cuda:0', torch.bfloat16, 0, (256,))
buf38 = generate_example_value((8192, 12, 64), (3072, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 12, 64))
buf39 = generate_example_value((8192, 12, 192), (3072, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 12, 192))
buf42 = generate_example_value((8192, 1, 64), (256, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 1, 64))
buf43 = generate_example_value((8192, 1, 192), (256, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 1, 192))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_4.run(buf34, buf35, arg19_1, buf36, buf37, buf41, arg20_1, buf38, buf39, buf42, buf43, 6291456, 18874368, 524288, 1572864, stream=raw_stream0)
del buf34, buf35, buf41, buf36, buf37, arg19_1, arg20_1, buf38, buf39, buf42, buf43

"""
# AOT ID: ['26_inference']
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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/6ae13a9bbce434c673241fcde54ba9f2b6cd5af3a26521600639df215ef08c73/inductor_cache/el/cel7s5sxlnrkn4wka6eyvols4ihf2x4qt4a4deuqvn2u44pmag2j.py
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add => add_60
# Graph fragment:
#   %getitem_6 : Tensor "bf16[s18, 2560][2560, 1]cuda:0" = PlaceHolder[target=getitem_6]
#   %getitem_7 : Tensor "bf16[s18, 2560][2560, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %add_60 : Tensor "bf16[s18, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, %getitem_7), kwargs = {})
#   return %add_60
triton_poi_fused_add_0 = async_compile.triton('triton_poi_fused_add_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 167772160}, 'kernel_num_gb': 0.12582912, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/6ae13a9bbce434c673241fcde54ba9f2b6cd5af3a26521600639df215ef08c73/inductor_cache/jd/cjd2yjbgtxgcrfs3lrwrcartjnyqqkjtxtvdmrlrhurwyry67wmg.py
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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/6ae13a9bbce434c673241fcde54ba9f2b6cd5af3a26521600639df215ef08c73/inductor_cache/vm/cvm2z5r6tm4rq2liv7ea35z3wropqzzi7uspwke3y5czmz6vrkut.py
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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/6ae13a9bbce434c673241fcde54ba9f2b6cd5af3a26521600639df215ef08c73/inductor_cache/w4/cw4sz32f2nkkka5bf4qdwlkobbbehnersxill6f6yf7wm7jncabc.py
# Topologically Sorted Source Nodes: [getitem_19, chunk_1, arange, mod_1, eq_1, lt_1, and__1, getitem_24, mod, eq, lt, and_, getitem_22, getitem_23, where, where_1, arange_1, mod_3, eq_3, lt_3, and__3, getitem_27, mod_2, eq_2, lt_2, and__2, getitem_25, getitem_26, where_2, where_3], Original ATen: [aten.index, aten.split, aten.arange, aten.remainder, aten.eq, aten.lt, aten.bitwise_and, aten.select, aten.where]
# Source node to ATen node mapping:
#   and_ => bitwise_and
#   and__1 => bitwise_and_1
#   and__2 => bitwise_and_2
#   and__3 => bitwise_and_3
#   arange => iota
#   arange_1 => iota_1
#   chunk_1 => split_1
#   eq => eq_76
#   eq_1 => eq_77
#   eq_2 => eq_83
#   eq_3 => eq_84
#   getitem_19 => index
#   getitem_22 => select
#   getitem_23 => select_1
#   getitem_24 => select_2
#   getitem_25 => select_3
#   getitem_26 => select_4
#   getitem_27 => select_5
#   lt => lt
#   lt_1 => lt_1
#   lt_2 => lt_2
#   lt_3 => lt_3
#   mod => remainder
#   mod_1 => remainder_1
#   mod_2 => remainder_2
#   mod_3 => remainder_3
#   where => where
#   where_1 => where_1
#   where_2 => where_2
#   where_3 => where_3
# Graph fragment:
#   %arg15_1 : Tensor "i64[3, s18][s7, 1]cuda:0" = PlaceHolder[target=arg15_1]
#   %arg21_1 : Tensor "bf16[1048576, 64][64, 1]cuda:0" = PlaceHolder[target=arg21_1]
#   %index : Tensor "bf16[3, s18, 64][64*s18, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg21_1, [%arg15_1]), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 32, -1), kwargs = {})
#   %iota : Tensor "i64[32][1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %remainder_1 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%iota, 3), kwargs = {})
#   %eq_77 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%remainder_1, 2), kwargs = {})
#   %lt_1 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota, 30), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_77, %lt_1), kwargs = {})
#   %select_2 : Tensor "bf16[s18, 32][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_21, 0, 2), kwargs = {})
#   %remainder : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%iota, 3), kwargs = {})
#   %eq_76 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%remainder, 1), kwargs = {})
#   %lt : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota, 33), kwargs = {})
#   %bitwise_and : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_76, %lt), kwargs = {})
#   %select : Tensor "bf16[s18, 32][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_21, 0, 1), kwargs = {})
#   %select_1 : Tensor "bf16[s18, 32][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_21, 0, 0), kwargs = {})
#   %where : Tensor "bf16[s18, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and, %select, %select_1), kwargs = {})
#   %where_1 : Tensor "bf16[s18, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_1, %select_2, %where), kwargs = {})
#   %iota_1 : Tensor "i64[32][1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %remainder_3 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%iota_1, 3), kwargs = {})
#   %eq_84 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%remainder_3, 2), kwargs = {})
#   %lt_3 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota_1, 30), kwargs = {})
#   %bitwise_and_3 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_84, %lt_3), kwargs = {})
#   %select_5 : Tensor "bf16[s18, 32][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_22, 0, 2), kwargs = {})
#   %remainder_2 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%iota_1, 3), kwargs = {})
#   %eq_83 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%remainder_2, 1), kwargs = {})
#   %lt_2 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota_1, 33), kwargs = {})
#   %bitwise_and_2 : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_83, %lt_2), kwargs = {})
#   %select_3 : Tensor "bf16[s18, 32][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_22, 0, 1), kwargs = {})
#   %select_4 : Tensor "bf16[s18, 32][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_22, 0, 0), kwargs = {})
#   %where_2 : Tensor "bf16[s18, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_2, %select_3, %select_4), kwargs = {})
#   %where_3 : Tensor "bf16[s18, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_3, %select_5, %where_2), kwargs = {})
#   return %where_1,%where_3
triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_3 = async_compile.triton('triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 2293760}, 'kernel_num_gb': 0.004390912, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/6ae13a9bbce434c673241fcde54ba9f2b6cd5af3a26521600639df215ef08c73/inductor_cache/x6/cx66si2rdcs4kzmuu4w6rf5m3ksvj72avsnjglciuanaocej2qfk.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
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
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 4, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None, 'no_x_dim_3': False, 'xnumel_3': None}, 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel_0, xnumel_1, xnumel_2, xnumel_3, XBLOCK : tl.constexpr):
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
        triton_poi_fused_4.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_4.benchmark_all_configs(*args)


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1 = args
        args.clear()
        s59 = arg1_1
        s18 = arg3_1
        s72 = s18
        s7 = arg16_1
        assert_size_stride(arg0_1, (s18, 24, 128), (3072, 128, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            # Topologically Sorted Source Nodes: [flatten, b12x_blockscaled_linear], Original ATen: [aten.view, vllm.b12x_blockscaled_linear]
            buf0 = torch.ops.vllm.b12x_blockscaled_linear.default(reinterpret_tensor(arg0_1, (s18, 3072), (3072, 1), 0), None, 2560, arg2_1)
            del arg0_1
            del arg2_1
            buf1 = buf0
            assert_size_stride(buf1, (s18, 2560), (2560, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf1, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf0
            # Topologically Sorted Source Nodes: [all_reduce], Original ATen: [vllm.all_reduce]
            buf2 = torch.ops.vllm.all_reduce.default(buf1, 'tp:0')
            del buf1
            buf3 = buf2
            assert_size_stride(buf3, (s18, 2560), (2560, 1), 'torch.ops.vllm.all_reduce.default')
            assert_alignment(buf3, 16, 'torch.ops.vllm.all_reduce.default')
            del buf2
            assert_size_stride(arg4_1, (s18, 10240), (10240, 1), 'input')
            assert_size_stride(arg5_1, (s18, 4), (336, 1), 'input')
            assert_size_stride(arg6_1, (10240, ), (1, ), 'input')
            arg4_1 = copy_if_misaligned(arg4_1)
            arg5_1 = copy_if_misaligned(arg5_1)
            arg6_1 = copy_if_misaligned(arg6_1)
            # Topologically Sorted Source Nodes: [hyperconnection_combine_norm], Original ATen: [b12x.hyperconnection_combine_norm]
            buf4 = torch.ops.b12x.hyperconnection_combine_norm.default(arg4_1, buf3, arg5_1, arg6_1, 1e-06, 8991)
            del arg4_1
            del arg5_1
            del arg6_1
            del buf3
            buf5 = buf4[0]
            assert_size_stride(buf5, (s18, 10240), (10240, 1), 'torch.ops.b12x.hyperconnection_combine_norm.default')
            assert_alignment(buf5, 16, 'torch.ops.b12x.hyperconnection_combine_norm.default')
            buf6 = buf4[1]
            assert_size_stride(buf6, (s18, 10240), (10240, 1), 'torch.ops.b12x.hyperconnection_combine_norm.default')
            assert_alignment(buf6, 16, 'torch.ops.b12x.hyperconnection_combine_norm.default')
            del buf4
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear_1], Original ATen: [vllm.b12x_blockscaled_linear]
            buf7 = torch.ops.vllm.b12x_blockscaled_linear.default(buf6, None, 336, arg7_1)
            del arg7_1
            buf8 = buf7
            assert_size_stride(buf8, (s18, 336), (336, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf8, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf7
            assert_size_stride(arg8_1, (8192, 320), (320, 1), 'input')
            # Topologically Sorted Source Nodes: [getitem_2, hyperconnection_scaled_silu], Original ATen: [aten.slice, b12x.hyperconnection_scaled_silu]
            torch.ops.b12x.hyperconnection_scaled_silu.default(reinterpret_tensor(buf8, (s18, 320), (336, 1), 0), arg8_1, 8988)
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear_2], Original ATen: [aten.slice, vllm.b12x_blockscaled_linear]
            buf11 = torch.ops.vllm.b12x_blockscaled_linear.default(reinterpret_tensor(arg8_1, (s18, 320), (320, 1), 0), None, 10240, arg9_1)
            del arg9_1
            buf12 = buf11
            assert_size_stride(buf12, (s18, 10240), (10240, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf12, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf11
            assert_size_stride(arg10_1, (8192, 2560), (2560, 1), 'input')
            # Topologically Sorted Source Nodes: [hyperconnection_gate_mean], Original ATen: [b12x.hyperconnection_gate_mean]
            torch.ops.b12x.hyperconnection_gate_mean.default(buf6, buf12, arg10_1, 8989)
            del buf12
            del buf6
            # Topologically Sorted Source Nodes: [moe_forward_shared], Original ATen: [aten.slice, vllm.moe_forward_shared]
            buf15 = torch.ops.vllm.moe_forward_shared.default(reinterpret_tensor(arg10_1, (s18, 2560), (2560, 1), 0), reinterpret_tensor(arg10_1, (s18, 2560), (2560, 1), 0), reinterpret_tensor(arg10_1, (s18, 2560), (2560, 1), 0), None, arg11_1, 0, torch.bfloat16)
            del arg11_1
            buf16 = buf15[0]
            assert_size_stride(buf16, (s18, 2560), (2560, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf16, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf17 = buf15[1]
            assert_size_stride(buf17, (s18, 2560), (2560, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf17, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf15
            buf18 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
            triton_poi_fused_add_0_xnumel = 2560*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_0.run(buf18, buf17, triton_poi_fused_add_0_xnumel, stream=raw_stream0)
            del buf17
            # Topologically Sorted Source Nodes: [add, all_reduce_1], Original ATen: [aten.add, vllm.all_reduce]
            buf19 = torch.ops.vllm.all_reduce.default(buf18, 'tp:0')
            del buf18
            buf20 = buf19
            assert_size_stride(buf20, (s18, 2560), (2560, 1), 'torch.ops.vllm.all_reduce.default')
            assert_alignment(buf20, 16, 'torch.ops.vllm.all_reduce.default')
            del buf19
            assert_size_stride(arg12_1, (10240, ), (1, ), 'input')
            arg12_1 = copy_if_misaligned(arg12_1)
            # Topologically Sorted Source Nodes: [getitem_3, hyperconnection_combine_norm_1], Original ATen: [aten.slice, b12x.hyperconnection_combine_norm]
            buf21 = torch.ops.b12x.hyperconnection_combine_norm.default(buf5, buf20, reinterpret_tensor(buf8, (s18, 4), (336, 1), 320), arg12_1, 1e-06, 9268)
            del arg12_1
            del buf20
            del buf5
            del buf8
            buf22 = buf21[0]
            assert_size_stride(buf22, (s18, 10240), (10240, 1), 'torch.ops.b12x.hyperconnection_combine_norm.default')
            assert_alignment(buf22, 16, 'torch.ops.b12x.hyperconnection_combine_norm.default')
            buf23 = buf21[1]
            assert_size_stride(buf23, (s18, 10240), (10240, 1), 'torch.ops.b12x.hyperconnection_combine_norm.default')
            assert_alignment(buf23, 16, 'torch.ops.b12x.hyperconnection_combine_norm.default')
            del buf21
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear_3], Original ATen: [vllm.b12x_blockscaled_linear]
            buf24 = torch.ops.vllm.b12x_blockscaled_linear.default(buf23, None, 336, arg13_1)
            del arg13_1
            buf25 = buf24
            assert_size_stride(buf25, (s18, 336), (336, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf25, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf24
            # Topologically Sorted Source Nodes: [getitem_10, hyperconnection_scaled_silu_1], Original ATen: [aten.slice, b12x.hyperconnection_scaled_silu]
            torch.ops.b12x.hyperconnection_scaled_silu.default(reinterpret_tensor(buf25, (s18, 320), (336, 1), 0), arg8_1, 9265)
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear_4], Original ATen: [aten.slice, vllm.b12x_blockscaled_linear]
            buf28 = torch.ops.vllm.b12x_blockscaled_linear.default(reinterpret_tensor(arg8_1, (s18, 320), (320, 1), 0), None, 10240, arg14_1)
            del arg14_1
            del arg8_1
            buf29 = buf28
            assert_size_stride(buf29, (s18, 10240), (10240, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf29, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf28
            # Topologically Sorted Source Nodes: [hyperconnection_gate_mean_1], Original ATen: [b12x.hyperconnection_gate_mean]
            torch.ops.b12x.hyperconnection_gate_mean.default(buf23, buf29, arg10_1, 9266)
            del buf23
            del buf29
            assert_size_stride(arg15_1, (3, s18), (s7, 1), 'input')
            assert_size_stride(arg17_1, (8192, 2051), (2051, 1), 'input')
            arg15_1 = copy_if_misaligned(arg15_1)
            # Topologically Sorted Source Nodes: [qwen3_8_flash_next_qsa_project_inputs], Original ATen: [aten.slice]
            buf32 = torch.ops.vllm.qwen3_8_flash_next_qsa_project_inputs.default(arg15_1, reinterpret_tensor(arg10_1, (s18, 2560), (2560, 1), 0), arg17_1, 6656, arg18_1)
            del arg17_1
            del arg18_1
            buf34 = buf32
            assert_size_stride(buf34, (s18, 6656), (6656, 1), 'torch.ops.vllm.qwen3_8_flash_next_qsa_project_inputs.default')
            assert_alignment(buf34, 16, 'torch.ops.vllm.qwen3_8_flash_next_qsa_project_inputs.default')
            del buf32
            buf45 = empty_strided_cuda((s18, 12, 256), (3072, 256, 1), torch.bfloat16)
            buf35 = empty_strided_cuda((s18, 12, 1), (12, 1, 12*s18), torch.float32)
            buf41 = empty_strided_cuda((s18, 1, 1), (1, s18, s18), torch.float32)
            # Topologically Sorted Source Nodes: [split, unflatten, chunk, rms_norm_default, unflatten_1, rms_norm_default_1, flatten_3], Original ATen: [aten.split_with_sizes, aten.view, aten.split, vllm_ir.rms_norm, aten.clone]
            triton_poi_fused_1_xnumel = 3072*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_1.run(buf34, buf45, triton_poi_fused_1_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [split, unflatten, chunk, rms_norm_default, unflatten_1, rms_norm_default_1, flatten_3], Original ATen: [aten.split_with_sizes, aten.view, aten.split, vllm_ir.rms_norm, aten.clone]
            triton_red_fused_2_xnumel_0 = 12*s18
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_2.run(buf34, buf35, buf41, triton_red_fused_2_xnumel_0, s18, stream=raw_stream0)
            assert_size_stride(arg21_1, (1048576, 64), (64, 1), 'input')
            arg21_1 = copy_if_misaligned(arg21_1)
            buf36 = empty_strided_cuda((s18, 32), (32, 1), torch.bfloat16)
            buf37 = empty_strided_cuda((s18, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_19, chunk_1, arange, mod_1, eq_1, lt_1, and__1, getitem_24, mod, eq, lt, and_, getitem_22, getitem_23, where, where_1, arange_1, mod_3, eq_3, lt_3, and__3, getitem_27, mod_2, eq_2, lt_2, and__2, getitem_25, getitem_26, where_2, where_3], Original ATen: [aten.index, aten.split, aten.arange, aten.remainder, aten.eq, aten.lt, aten.bitwise_and, aten.select, aten.where]
            triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_3_xnumel = 32*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_3.run(arg15_1, arg21_1, buf36, buf37, s7, triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_3_xnumel, stream=raw_stream0)
            del arg15_1
            del arg21_1
            assert_size_stride(arg19_1, (256, ), (1, ), 'input')
            assert_size_stride(arg20_1, (256, ), (1, ), 'input')
            arg19_1 = copy_if_misaligned(arg19_1)
            arg20_1 = copy_if_misaligned(arg20_1)
            buf40 = empty_strided_cuda((s18, 12, 256), (3072, 256, 1), torch.bfloat16)
            buf38 = reinterpret_tensor(buf40, (s18, 12, 64), (3072, 256, 1), 0)  # alias
            buf39 = reinterpret_tensor(buf40, (s18, 12, 192), (3072, 256, 1), 64)  # alias
            buf44 = empty_strided_cuda((s18, 1, 256), (256, 256, 1), torch.bfloat16)
            buf42 = reinterpret_tensor(buf44, (s18, 1, 64), (256, 256, 1), 0)  # alias
            buf43 = reinterpret_tensor(buf44, (s18, 1, 192), (256, 256, 1), 64)  # alias
            # Topologically Sorted Source Nodes: [split, unflatten, chunk, float_1, add_1, rms_norm_default, getitem_28, chunk_2, unsqueeze, mul, unsqueeze_1, mul_1, sub, mul_2, mul_3, add_3, cat, getitem_29, cat_1, unflatten_1, float_2, add_2, rms_norm_default_1, getitem_32, chunk_3, unsqueeze_2, mul_4, unsqueeze_3, mul_5, sub_1, mul_6, mul_7, add_4, cat_2, getitem_33, cat_3], Original ATen: [aten.split_with_sizes, aten.view, aten.split, aten._to_copy, aten.add, vllm_ir.rms_norm, aten.slice, aten.unsqueeze, aten.mul, aten.sub, aten.cat]
            triton_poi_fused_4_xnumel_0 = 768*s18
            triton_poi_fused_4_xnumel_1 = 2304*s18
            triton_poi_fused_4_xnumel_2 = 64*s18
            triton_poi_fused_4_xnumel_3 = 192*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_4.run(buf34, buf35, arg19_1, buf36, buf37, buf41, arg20_1, buf38, buf39, buf42, buf43, triton_poi_fused_4_xnumel_0, triton_poi_fused_4_xnumel_1, triton_poi_fused_4_xnumel_2, triton_poi_fused_4_xnumel_3, stream=raw_stream0)
            del arg19_1
            del arg20_1
            del buf35
            del buf36
            del buf37
            del buf41
            buf46 = empty_strided_cuda((s18, 12, 256), (3072, 256, 1), torch.bfloat16)
        return (reinterpret_tensor(arg10_1, (s18, 2560), (2560, 1), 0), buf40, buf44, reinterpret_tensor(buf34, (s18, 1, 256), (6656, 256, 1), 6400), buf46, reinterpret_tensor(buf45, (s18, 3072), (3072, 1), 0), buf22, reinterpret_tensor(buf25, (s18, 4), (336, 1), 320), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 24, 128), (3072, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = 8192
    arg2_1 = None
    arg3_1 = 8192
    arg4_1 = rand_strided((8192, 10240), (10240, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((8192, 4), (336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = None
    arg8_1 = rand_strided((8192, 320), (320, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = None
    arg10_1 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = None
    arg12_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = None
    arg14_1 = None
    arg15_1 = rand_strided((3, 8192), (8193, 1), device='cuda:0', dtype=torch.int64)
    arg16_1 = 8193
    arg17_1 = rand_strided((8192, 2051), (2051, 1), device='cuda:0', dtype=torch.int32)
    arg18_1 = None
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((1048576, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
