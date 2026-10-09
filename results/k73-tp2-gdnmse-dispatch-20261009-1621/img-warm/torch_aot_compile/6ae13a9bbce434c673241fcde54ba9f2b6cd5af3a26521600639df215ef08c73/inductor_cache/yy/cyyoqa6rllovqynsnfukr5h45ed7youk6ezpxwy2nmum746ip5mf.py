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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/6ae13a9bbce434c673241fcde54ba9f2b6cd5af3a26521600639df215ef08c73/inductor_cache/gd/cgd5l6eeum7segqzfvfxr5opihrygenyejssgbtpiuheil4qi42p.py
# Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   zeros => full_default
# Graph fragment:
#   %full_default : Tensor "bf16[s18, 24, 128][3072, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg1_1, 24, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default
triton_poi_fused_zeros_0 = async_compile.triton('triton_poi_fused_zeros_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_zeros_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 100663296}, 'kernel_num_gb': 0.050331648, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
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
buf13 = generate_example_value((8192, 24, 128), (3072, 128, 1), 'cuda:0', torch.bfloat16, 0, (8192, 24, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_zeros_0.run(buf13, 25165824, stream=raw_stream0)
del buf13

"""
# AOT ID: ['1_inference']
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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/6ae13a9bbce434c673241fcde54ba9f2b6cd5af3a26521600639df215ef08c73/inductor_cache/gd/cgd5l6eeum7segqzfvfxr5opihrygenyejssgbtpiuheil4qi42p.py
# Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   zeros => full_default
# Graph fragment:
#   %full_default : Tensor "bf16[s18, 24, 128][3072, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg1_1, 24, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default
triton_poi_fused_zeros_0 = async_compile.triton('triton_poi_fused_zeros_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_zeros_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 100663296}, 'kernel_num_gb': 0.050331648, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
        args.clear()
        s59 = arg1_1
        s18 = arg4_1
        s72 = s18
        assert_size_stride(arg0_1, (s18, 10240), (10240, 1), 'input')
        assert_size_stride(arg2_1, (10240, ), (1, ), 'input')
        assert_size_stride(arg3_1, (8192, 10240), (10240, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            # Topologically Sorted Source Nodes: [hyperconnection_grouped_rmsnorm], Original ATen: [b12x.hyperconnection_grouped_rmsnorm]
            torch.ops.b12x.hyperconnection_grouped_rmsnorm.default(arg0_1, arg2_1, arg3_1, 1e-06, 467, zero_centered=True)
            del arg0_1
            del arg2_1
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear], Original ATen: [aten.slice, vllm.b12x_blockscaled_linear]
            buf2 = torch.ops.vllm.b12x_blockscaled_linear.default(reinterpret_tensor(arg3_1, (s18, 10240), (10240, 1), 0), None, 336, arg5_1)
            del arg5_1
            buf3 = buf2
            assert_size_stride(buf3, (s18, 336), (336, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf3, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf2
            assert_size_stride(arg6_1, (8192, 320), (320, 1), 'input')
            # Topologically Sorted Source Nodes: [getitem_1, hyperconnection_scaled_silu], Original ATen: [aten.slice, b12x.hyperconnection_scaled_silu]
            torch.ops.b12x.hyperconnection_scaled_silu.default(reinterpret_tensor(buf3, (s18, 320), (336, 1), 0), arg6_1, 468)
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear_1], Original ATen: [aten.slice, vllm.b12x_blockscaled_linear]
            buf6 = torch.ops.vllm.b12x_blockscaled_linear.default(reinterpret_tensor(arg6_1, (s18, 320), (320, 1), 0), None, 10240, arg7_1)
            del arg6_1
            del arg7_1
            buf7 = buf6
            assert_size_stride(buf7, (s18, 10240), (10240, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf7, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf6
            assert_size_stride(arg8_1, (8192, 2560), (2560, 1), 'input')
            # Topologically Sorted Source Nodes: [reshape, hyperconnection_gate_mean], Original ATen: [aten.slice, b12x.hyperconnection_gate_mean]
            torch.ops.b12x.hyperconnection_gate_mean.default(reinterpret_tensor(arg3_1, (s18, 10240), (10240, 1), 0), buf7, arg8_1, 469)
            del arg3_1
            del buf7
            # Topologically Sorted Source Nodes: [qwen_gdn_input_projections], Original ATen: [aten.slice, vllm.qwen_gdn_input_projections]
            buf10 = torch.ops.vllm.qwen_gdn_input_projections.default(reinterpret_tensor(arg8_1, (s18, 2560), (2560, 1), 0), 8192, 48, arg9_1)
            del arg8_1
            del arg9_1
            buf11 = buf10[0]
            assert_size_stride(buf11, (s18, 8192), (8192, 1), 'torch.ops.vllm.qwen_gdn_input_projections.default')
            assert_alignment(buf11, 16, 'torch.ops.vllm.qwen_gdn_input_projections.default')
            buf12 = buf10[1]
            assert_size_stride(buf12, (s18, 48), (48, 1), 'torch.ops.vllm.qwen_gdn_input_projections.default')
            assert_alignment(buf12, 16, 'torch.ops.vllm.qwen_gdn_input_projections.default')
            del buf10
            buf13 = empty_strided_cuda((s18, 24, 128), (3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
            triton_poi_fused_zeros_0_xnumel = 3072*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_zeros_0.run(buf13, triton_poi_fused_zeros_0_xnumel, stream=raw_stream0)
        return (buf11, buf12, buf13, reinterpret_tensor(buf3, (s18, 4), (336, 1), 320), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 10240), (10240, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = 8192
    arg2_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((8192, 10240), (10240, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = 8192
    arg5_1 = None
    arg6_1 = rand_strided((8192, 320), (320, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = None
    arg8_1 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = None
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
