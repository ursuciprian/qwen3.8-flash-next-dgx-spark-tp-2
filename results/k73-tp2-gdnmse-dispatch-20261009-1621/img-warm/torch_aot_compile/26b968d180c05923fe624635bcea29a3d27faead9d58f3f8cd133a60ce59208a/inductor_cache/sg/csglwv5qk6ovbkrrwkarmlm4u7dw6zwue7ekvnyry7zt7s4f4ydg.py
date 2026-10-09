# AOT ID: ['5_inference']
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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/26b968d180c05923fe624635bcea29a3d27faead9d58f3f8cd133a60ce59208a/inductor_cache/wf/cwfog2f5dfco5i2ppznn66pexl4bl7pomptrib2xozdbwqxpbupn.py
# Topologically Sorted Source Nodes: [ge, lt, org_vocab_mask, ge_1, lt_1, added_vocab_mask, vocab_mask, mul, mul_1, valid_offset, sub_3, input_], Original ATen: [aten.ge, aten.lt, aten.bitwise_and, aten.bitwise_or, aten.mul, aten.add, aten.sub]
# Source node to ATen node mapping:
#   added_vocab_mask => bitwise_and_1
#   ge => ge
#   ge_1 => ge_1
#   input_ => mul_2
#   lt => lt
#   lt_1 => lt_1
#   mul => mul
#   mul_1 => mul_1
#   org_vocab_mask => bitwise_and
#   sub_3 => sub_3
#   valid_offset => add
#   vocab_mask => bitwise_or
# Graph fragment:
#   %arg0_1 : Tensor "i32[1][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %ge : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg0_1, %arg1_1), kwargs = {})
#   %lt : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%arg0_1, %arg2_1), kwargs = {})
#   %bitwise_and : Tensor "b8[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge, %lt), kwargs = {})
#   %ge_1 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg0_1, %arg3_1), kwargs = {})
#   %lt_1 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%arg0_1, %arg4_1), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_1, %lt_1), kwargs = {})
#   %bitwise_or : Tensor "b8[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_and, %bitwise_and_1), kwargs = {})
#   %mul : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bitwise_and, %arg1_1), kwargs = {})
#   %mul_1 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bitwise_and_1, %sub_2), kwargs = {})
#   %add : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %sub_3 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %add), kwargs = {})
#   %mul_2 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bitwise_or, %sub_3), kwargs = {})
#   return %mul_2
triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0 = async_compile.triton('triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i64', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'ks3': 'i64', 'ks4': 'i64', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = (ks0).to(tl.int32)
    tmp3 = tmp1 >= tmp2
    tmp4 = (ks1).to(tl.int32)
    tmp5 = tmp1 < tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = (ks2).to(tl.int32)
    tmp8 = tmp1 >= tmp7
    tmp9 = (ks3).to(tl.int32)
    tmp10 = tmp1 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tmp6 | tmp11
    tmp13 = tmp12.to(tl.int64)
    tmp14 = tmp1.to(tl.int64)
    tmp15 = tmp6.to(tl.int64)
    tmp16 = (ks0).to(tl.int64)
    tmp17 = (tmp16).to(tl.int64)
    tmp18 = tmp15 * tmp17
    tmp19 = tmp11.to(tl.int64)
    tmp20 = (ks0 + ks2 + ((-1)*ks1) + ((-1)*ks4)).to(tl.int64)
    tmp21 = (tmp20).to(tl.int64)
    tmp22 = tmp19 * tmp21
    tmp23 = tmp18 + tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp13 * tmp24
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp25, None)
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/26b968d180c05923fe624635bcea29a3d27faead9d58f3f8cd133a60ce59208a/inductor_cache/bu/cbucfc4vh3omjipzubpgc6jwnxiiymxpsy4nkgvfx3wypafreukn.py
# Topologically Sorted Source Nodes: [ge, lt, org_vocab_mask, ge_1, lt_1, added_vocab_mask, vocab_mask, invert], Original ATen: [aten.ge, aten.lt, aten.bitwise_and, aten.bitwise_or, aten.bitwise_not]
# Source node to ATen node mapping:
#   added_vocab_mask => bitwise_and_1
#   ge => ge
#   ge_1 => ge_1
#   invert => bitwise_not
#   lt => lt
#   lt_1 => lt_1
#   org_vocab_mask => bitwise_and
#   vocab_mask => bitwise_or
# Graph fragment:
#   %arg0_1 : Tensor "i32[1][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %ge : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg0_1, %arg1_1), kwargs = {})
#   %lt : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%arg0_1, %arg2_1), kwargs = {})
#   %bitwise_and : Tensor "b8[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge, %lt), kwargs = {})
#   %ge_1 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg0_1, %arg3_1), kwargs = {})
#   %lt_1 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%arg0_1, %arg4_1), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_1, %lt_1), kwargs = {})
#   %bitwise_or : Tensor "b8[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_and, %bitwise_and_1), kwargs = {})
#   %bitwise_not : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%bitwise_or,), kwargs = {})
#   return %bitwise_not
triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1 = async_compile.triton('triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i1', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'ks3': 'i64', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = (ks0).to(tl.int32)
    tmp3 = tmp1 >= tmp2
    tmp4 = (ks1).to(tl.int32)
    tmp5 = tmp1 < tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = (ks2).to(tl.int32)
    tmp8 = tmp1 >= tmp7
    tmp9 = (ks3).to(tl.int32)
    tmp10 = tmp1 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tmp6 | tmp11
    tmp13 = tmp12 == 0
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp13, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
        args.clear()
        s90 = arg1_1
        s87 = arg2_1
        s68 = arg3_1
        s53 = arg4_1
        s88 = arg5_1
        assert_size_stride(arg0_1, (1, ), (1, ), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_cuda((1, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [ge, lt, org_vocab_mask, ge_1, lt_1, added_vocab_mask, vocab_mask, mul, mul_1, valid_offset, sub_3, input_], Original ATen: [aten.ge, aten.lt, aten.bitwise_and, aten.bitwise_or, aten.mul, aten.add, aten.sub]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0.run(arg0_1, buf0, s90, s87, s68, s53, s88, 1, stream=raw_stream0)
            buf1 = empty_strided_cuda((1, ), (1, ), torch.bool)
            # Topologically Sorted Source Nodes: [ge, lt, org_vocab_mask, ge_1, lt_1, added_vocab_mask, vocab_mask, invert], Original ATen: [aten.ge, aten.lt, aten.bitwise_and, aten.bitwise_or, aten.bitwise_not]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1.run(arg0_1, buf1, s90, s87, s68, s53, 1, stream=raw_stream0)
            del arg0_1
        return (buf0, buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg1_1 = 0
    arg2_1 = 124160
    arg3_1 = 248320
    arg4_1 = 248320
    arg5_1 = 0
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
