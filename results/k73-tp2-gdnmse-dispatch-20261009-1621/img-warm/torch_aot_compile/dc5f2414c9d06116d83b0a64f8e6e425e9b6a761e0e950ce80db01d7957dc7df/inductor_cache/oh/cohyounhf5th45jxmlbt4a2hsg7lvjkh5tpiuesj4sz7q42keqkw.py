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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/dc5f2414c9d06116d83b0a64f8e6e425e9b6a761e0e950ce80db01d7957dc7df/inductor_cache/kj/ckjneqslfpvn2ysojkbkow7uvlfoejsdrlrudxqktjzrocc66hmn.py
# Topologically Sorted Source Nodes: [getitem, flatten, add], Original ATen: [aten.slice, aten.view, aten.add]
# Source node to ATen node mapping:
#   add => add_7
#   flatten => view
#   getitem => slice_1
# Graph fragment:
#   %arg3_1 : Tensor "bf16[s18, 10240][10240, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg0_1 : Tensor "bf16[8192, 4, 2560][10240, 2560, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %slice_1 : Tensor "bf16[s18, 4, 2560][10240, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg0_1, 0, 0, %arg1_1), kwargs = {})
#   %view : Tensor "bf16[s18, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%slice_1, [%arg2_1, 10240]), kwargs = {})
#   %add_7 : Tensor "bf16[s18, 10240][10240, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %view), kwargs = {})
#   return %add_7
triton_poi_fused_add_slice_view_0 = async_compile.triton('triton_poi_fused_add_slice_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_slice_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 671088640}, 'kernel_num_gb': 0.50331648, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_slice_view_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/dc5f2414c9d06116d83b0a64f8e6e425e9b6a761e0e950ce80db01d7957dc7df/inductor_cache/kf/ckff7igbo2v6dmhzm22ftqxv3bqf4stviqseyt6vuvqu2m5n3wiu.py
# Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   zeros => full_default
# Graph fragment:
#   %full_default : Tensor "bf16[s18, 24, 128][3072, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg1_1, 24, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default
triton_poi_fused_zeros_1 = async_compile.triton('triton_poi_fused_zeros_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_zeros_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 100663296}, 'kernel_num_gb': 0.050331648, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
arg3_1 = generate_example_value((8192, 10240), (10240, 1), 'cuda:0', torch.bfloat16, 0, (8192, 10240))
arg0_1 = generate_example_value((8192, 4, 2560), (10240, 2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4, 2560))
buf0 = generate_example_value((8192, 10240), (10240, 1), 'cuda:0', torch.bfloat16, 0, (8192, 10240))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_add_slice_view_0.run(arg3_1, arg0_1, buf0, 83886080, stream=raw_stream0)
del arg3_1, arg0_1, buf0

raw_stream0 = get_raw_stream(0)
buf14 = generate_example_value((8192, 24, 128), (3072, 128, 1), 'cuda:0', torch.bfloat16, 0, (8192, 24, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_zeros_1.run(buf14, 25165824, stream=raw_stream0)
del buf14

"""
# AOT ID: ['4_inference']
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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/dc5f2414c9d06116d83b0a64f8e6e425e9b6a761e0e950ce80db01d7957dc7df/inductor_cache/kj/ckjneqslfpvn2ysojkbkow7uvlfoejsdrlrudxqktjzrocc66hmn.py
# Topologically Sorted Source Nodes: [getitem, flatten, add], Original ATen: [aten.slice, aten.view, aten.add]
# Source node to ATen node mapping:
#   add => add_7
#   flatten => view
#   getitem => slice_1
# Graph fragment:
#   %arg3_1 : Tensor "bf16[s18, 10240][10240, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg0_1 : Tensor "bf16[8192, 4, 2560][10240, 2560, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %slice_1 : Tensor "bf16[s18, 4, 2560][10240, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg0_1, 0, 0, %arg1_1), kwargs = {})
#   %view : Tensor "bf16[s18, 10240][10240, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%slice_1, [%arg2_1, 10240]), kwargs = {})
#   %add_7 : Tensor "bf16[s18, 10240][10240, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %view), kwargs = {})
#   return %add_7
triton_poi_fused_add_slice_view_0 = async_compile.triton('triton_poi_fused_add_slice_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_slice_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 671088640}, 'kernel_num_gb': 0.50331648, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_slice_view_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/dc5f2414c9d06116d83b0a64f8e6e425e9b6a761e0e950ce80db01d7957dc7df/inductor_cache/kf/ckff7igbo2v6dmhzm22ftqxv3bqf4stviqseyt6vuvqu2m5n3wiu.py
# Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   zeros => full_default
# Graph fragment:
#   %full_default : Tensor "bf16[s18, 24, 128][3072, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg1_1, 24, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default
triton_poi_fused_zeros_1 = async_compile.triton('triton_poi_fused_zeros_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_zeros_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 100663296}, 'kernel_num_gb': 0.050331648, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1 = args
        args.clear()
        s18 = arg1_1
        s72 = s18
        s59 = s18
        assert_size_stride(arg3_1, (s18, 10240), (10240, 1), 'input')
        assert_size_stride(arg0_1, (8192, 4, 2560), (10240, 2560, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg3_1 = copy_if_misaligned(arg3_1)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_cuda((s18, 10240), (10240, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem, flatten, add], Original ATen: [aten.slice, aten.view, aten.add]
            triton_poi_fused_add_slice_view_0_xnumel = 10240*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_view_0.run(arg3_1, arg0_1, buf0, triton_poi_fused_add_slice_view_0_xnumel, stream=raw_stream0)
            del arg0_1
            del arg3_1
            assert_size_stride(arg4_1, (10240, ), (1, ), 'input')
            assert_size_stride(arg5_1, (8192, 10240), (10240, 1), 'input')
            arg4_1 = copy_if_misaligned(arg4_1)
            # Topologically Sorted Source Nodes: [hyperconnection_grouped_rmsnorm], Original ATen: [b12x.hyperconnection_grouped_rmsnorm]
            torch.ops.b12x.hyperconnection_grouped_rmsnorm.default(buf0, arg4_1, arg5_1, 1e-06, 892, zero_centered=True)
            del arg4_1
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear], Original ATen: [aten.slice, vllm.b12x_blockscaled_linear]
            buf3 = torch.ops.vllm.b12x_blockscaled_linear.default(reinterpret_tensor(arg5_1, (s18, 10240), (10240, 1), 0), None, 336, arg6_1)
            del arg6_1
            buf4 = buf3
            assert_size_stride(buf4, (s18, 336), (336, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf4, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf3
            assert_size_stride(arg7_1, (8192, 320), (320, 1), 'input')
            # Topologically Sorted Source Nodes: [getitem_2, hyperconnection_scaled_silu], Original ATen: [aten.slice, b12x.hyperconnection_scaled_silu]
            torch.ops.b12x.hyperconnection_scaled_silu.default(reinterpret_tensor(buf4, (s18, 320), (336, 1), 0), arg7_1, 893)
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear_1], Original ATen: [aten.slice, vllm.b12x_blockscaled_linear]
            buf7 = torch.ops.vllm.b12x_blockscaled_linear.default(reinterpret_tensor(arg7_1, (s18, 320), (320, 1), 0), None, 10240, arg8_1)
            del arg7_1
            del arg8_1
            buf8 = buf7
            assert_size_stride(buf8, (s18, 10240), (10240, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf8, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf7
            assert_size_stride(arg9_1, (8192, 2560), (2560, 1), 'input')
            # Topologically Sorted Source Nodes: [reshape, hyperconnection_gate_mean], Original ATen: [aten.slice, b12x.hyperconnection_gate_mean]
            torch.ops.b12x.hyperconnection_gate_mean.default(reinterpret_tensor(arg5_1, (s18, 10240), (10240, 1), 0), buf8, arg9_1, 894)
            del arg5_1
            del buf8
            # Topologically Sorted Source Nodes: [qwen_gdn_input_projections], Original ATen: [aten.slice, vllm.qwen_gdn_input_projections]
            buf11 = torch.ops.vllm.qwen_gdn_input_projections.default(reinterpret_tensor(arg9_1, (s18, 2560), (2560, 1), 0), 8192, 48, arg10_1)
            del arg10_1
            del arg9_1
            buf12 = buf11[0]
            assert_size_stride(buf12, (s18, 8192), (8192, 1), 'torch.ops.vllm.qwen_gdn_input_projections.default')
            assert_alignment(buf12, 16, 'torch.ops.vllm.qwen_gdn_input_projections.default')
            buf13 = buf11[1]
            assert_size_stride(buf13, (s18, 48), (48, 1), 'torch.ops.vllm.qwen_gdn_input_projections.default')
            assert_alignment(buf13, 16, 'torch.ops.vllm.qwen_gdn_input_projections.default')
            del buf11
            buf14 = empty_strided_cuda((s18, 24, 128), (3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
            triton_poi_fused_zeros_1_xnumel = 3072*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_zeros_1.run(buf14, triton_poi_fused_zeros_1_xnumel, stream=raw_stream0)
        return (buf12, buf13, buf14, buf0, reinterpret_tensor(buf4, (s18, 4), (336, 1), 320), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 4, 2560), (10240, 2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = 8192
    arg2_1 = 8192
    arg3_1 = rand_strided((8192, 10240), (10240, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((8192, 10240), (10240, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = None
    arg7_1 = rand_strided((8192, 320), (320, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = None
    arg9_1 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = None
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
