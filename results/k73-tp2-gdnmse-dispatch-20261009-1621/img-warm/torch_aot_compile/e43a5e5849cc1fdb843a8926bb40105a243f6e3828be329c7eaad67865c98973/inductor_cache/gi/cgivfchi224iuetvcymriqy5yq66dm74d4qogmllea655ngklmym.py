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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/e43a5e5849cc1fdb843a8926bb40105a243f6e3828be329c7eaad67865c98973/inductor_cache/el/cel7s5sxlnrkn4wka6eyvols4ihf2x4qt4a4deuqvn2u44pmag2j.py
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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/e43a5e5849cc1fdb843a8926bb40105a243f6e3828be329c7eaad67865c98973/inductor_cache/kf/ckff7igbo2v6dmhzm22ftqxv3bqf4stviqseyt6vuvqu2m5n3wiu.py
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
buf18 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
buf17 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_add_0.run(buf18, buf17, 20971520, stream=raw_stream0)
del buf18, buf17

raw_stream0 = get_raw_stream(0)
buf35 = generate_example_value((8192, 24, 128), (3072, 128, 1), 'cuda:0', torch.bfloat16, 0, (8192, 24, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_zeros_1.run(buf35, 25165824, stream=raw_stream0)
del buf35

"""
# AOT ID: ['25_inference']
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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/e43a5e5849cc1fdb843a8926bb40105a243f6e3828be329c7eaad67865c98973/inductor_cache/el/cel7s5sxlnrkn4wka6eyvols4ihf2x4qt4a4deuqvn2u44pmag2j.py
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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/e43a5e5849cc1fdb843a8926bb40105a243f6e3828be329c7eaad67865c98973/inductor_cache/kf/ckff7igbo2v6dmhzm22ftqxv3bqf4stviqseyt6vuvqu2m5n3wiu.py
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1 = args
        args.clear()
        s59 = arg1_1
        s18 = arg3_1
        s72 = s18
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
            buf4 = torch.ops.b12x.hyperconnection_combine_norm.default(arg4_1, buf3, arg5_1, arg6_1, 1e-06, 8701)
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
            torch.ops.b12x.hyperconnection_scaled_silu.default(reinterpret_tensor(buf8, (s18, 320), (336, 1), 0), arg8_1, 8698)
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear_2], Original ATen: [aten.slice, vllm.b12x_blockscaled_linear]
            buf11 = torch.ops.vllm.b12x_blockscaled_linear.default(reinterpret_tensor(arg8_1, (s18, 320), (320, 1), 0), None, 10240, arg9_1)
            del arg9_1
            buf12 = buf11
            assert_size_stride(buf12, (s18, 10240), (10240, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf12, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf11
            assert_size_stride(arg10_1, (8192, 2560), (2560, 1), 'input')
            # Topologically Sorted Source Nodes: [hyperconnection_gate_mean], Original ATen: [b12x.hyperconnection_gate_mean]
            torch.ops.b12x.hyperconnection_gate_mean.default(buf6, buf12, arg10_1, 8699)
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
            buf21 = torch.ops.b12x.hyperconnection_combine_norm.default(buf5, buf20, reinterpret_tensor(buf8, (s18, 4), (336, 1), 320), arg12_1, 1e-06, 9026)
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
            torch.ops.b12x.hyperconnection_scaled_silu.default(reinterpret_tensor(buf25, (s18, 320), (336, 1), 0), arg8_1, 9023)
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear_4], Original ATen: [aten.slice, vllm.b12x_blockscaled_linear]
            buf28 = torch.ops.vllm.b12x_blockscaled_linear.default(reinterpret_tensor(arg8_1, (s18, 320), (320, 1), 0), None, 10240, arg14_1)
            del arg14_1
            del arg8_1
            buf29 = buf28
            assert_size_stride(buf29, (s18, 10240), (10240, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf29, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf28
            # Topologically Sorted Source Nodes: [hyperconnection_gate_mean_1], Original ATen: [b12x.hyperconnection_gate_mean]
            torch.ops.b12x.hyperconnection_gate_mean.default(buf23, buf29, arg10_1, 9024)
            del buf23
            del buf29
            # Topologically Sorted Source Nodes: [qwen_gdn_input_projections], Original ATen: [aten.slice, vllm.qwen_gdn_input_projections]
            buf32 = torch.ops.vllm.qwen_gdn_input_projections.default(reinterpret_tensor(arg10_1, (s18, 2560), (2560, 1), 0), 8192, 48, arg15_1)
            del arg10_1
            del arg15_1
            buf33 = buf32[0]
            assert_size_stride(buf33, (s18, 8192), (8192, 1), 'torch.ops.vllm.qwen_gdn_input_projections.default')
            assert_alignment(buf33, 16, 'torch.ops.vllm.qwen_gdn_input_projections.default')
            buf34 = buf32[1]
            assert_size_stride(buf34, (s18, 48), (48, 1), 'torch.ops.vllm.qwen_gdn_input_projections.default')
            assert_alignment(buf34, 16, 'torch.ops.vllm.qwen_gdn_input_projections.default')
            del buf32
            buf35 = empty_strided_cuda((s18, 24, 128), (3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
            triton_poi_fused_zeros_1_xnumel = 3072*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_zeros_1.run(buf35, triton_poi_fused_zeros_1_xnumel, stream=raw_stream0)
        return (buf33, buf34, buf35, buf22, reinterpret_tensor(buf25, (s18, 4), (336, 1), 320), )

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
    arg15_1 = None
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
