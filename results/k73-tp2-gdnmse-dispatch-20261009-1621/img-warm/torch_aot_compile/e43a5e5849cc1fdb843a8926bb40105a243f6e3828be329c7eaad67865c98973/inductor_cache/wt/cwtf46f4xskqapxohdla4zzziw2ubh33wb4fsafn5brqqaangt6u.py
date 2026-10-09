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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/e43a5e5849cc1fdb843a8926bb40105a243f6e3828be329c7eaad67865c98973/inductor_cache/3v/c3vmhuslrrsxc2toj3kyrlpmuzl3mydjtvxaqfhg735yhapgvvu7.py
# Topologically Sorted Source Nodes: [flatten, sigmoid, mul, b12x_blockscaled_linear], Original ATen: [aten.view, aten.sigmoid, aten.mul, vllm.b12x_blockscaled_linear]
# Source node to ATen node mapping:
#   b12x_blockscaled_linear => b12x_blockscaled_linear
#   flatten => view
#   mul => mul_8
#   sigmoid => sigmoid
# Graph fragment:
#   %arg0_1 : Tensor "bf16[s18, 12, 256][3072, 256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %view : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [%arg4_1, 3072]), kwargs = {})
#   %sigmoid : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg2_1,), kwargs = {})
#   %mul_8 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %sigmoid), kwargs = {})
#   %b12x_blockscaled_linear : Tensor "bf16[s18, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.vllm.b12x_blockscaled_linear.default](args = (%mul_8, None, 2560, %arg3_1), kwargs = {})
#   return %buf0
triton_poi_fused_b12x_blockscaled_linear_mul_sigmoid_view_0 = async_compile.triton('triton_poi_fused_b12x_blockscaled_linear_mul_sigmoid_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_b12x_blockscaled_linear_mul_sigmoid_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 201326592}, 'kernel_num_gb': 0.150994944, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_b12x_blockscaled_linear_mul_sigmoid_view_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/e43a5e5849cc1fdb843a8926bb40105a243f6e3828be329c7eaad67865c98973/inductor_cache/cr/ccrx74hg4vnqfhg7rw5hdmoexbitjxsu2lnzzcqztjxjm2rswbrz.py
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add => add_66
# Graph fragment:
#   %getitem_6 : Tensor "bf16[s18, 2560][2560, 1]cuda:0" = PlaceHolder[target=getitem_6]
#   %getitem_7 : Tensor "bf16[s18, 2560][2560, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %add_66 : Tensor "bf16[s18, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, %getitem_7), kwargs = {})
#   return %add_66
triton_poi_fused_add_1 = async_compile.triton('triton_poi_fused_add_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 167772160}, 'kernel_num_gb': 0.12582912, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/e43a5e5849cc1fdb843a8926bb40105a243f6e3828be329c7eaad67865c98973/inductor_cache/by/cbymdfx3oqh5cua3prtnav4qlhfw4memc33wwos2om3a53gl3gjj.py
# Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   zeros => full_default
# Graph fragment:
#   %full_default : Tensor "bf16[s18, 24, 128][3072, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg1_1, 24, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default
triton_poi_fused_zeros_2 = async_compile.triton('triton_poi_fused_zeros_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_zeros_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 100663296}, 'kernel_num_gb': 0.050331648, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
arg0_1 = generate_example_value((8192, 12, 256), (3072, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 12, 256))
arg2_1 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
buf0 = generate_example_value((8192, 3072), (3072, 1), 'cuda:0', torch.bfloat16, 0, (8192, 3072))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_b12x_blockscaled_linear_mul_sigmoid_view_0.run(arg0_1, arg2_1, buf0, 25165824, stream=raw_stream0)
del arg0_1, arg2_1, buf0

raw_stream0 = get_raw_stream(0)
buf19 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
buf18 = generate_example_value((8192, 2560), (2560, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2560))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_add_1.run(buf19, buf18, 20971520, stream=raw_stream0)
del buf19, buf18

raw_stream0 = get_raw_stream(0)
buf36 = generate_example_value((8192, 24, 128), (3072, 128, 1), 'cuda:0', torch.bfloat16, 0, (8192, 24, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_zeros_2.run(buf36, 25165824, stream=raw_stream0)
del buf36

"""
# AOT ID: ['15_inference']
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


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/e43a5e5849cc1fdb843a8926bb40105a243f6e3828be329c7eaad67865c98973/inductor_cache/3v/c3vmhuslrrsxc2toj3kyrlpmuzl3mydjtvxaqfhg735yhapgvvu7.py
# Topologically Sorted Source Nodes: [flatten, sigmoid, mul, b12x_blockscaled_linear], Original ATen: [aten.view, aten.sigmoid, aten.mul, vllm.b12x_blockscaled_linear]
# Source node to ATen node mapping:
#   b12x_blockscaled_linear => b12x_blockscaled_linear
#   flatten => view
#   mul => mul_8
#   sigmoid => sigmoid
# Graph fragment:
#   %arg0_1 : Tensor "bf16[s18, 12, 256][3072, 256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[s18, 3072][3072, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %view : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [%arg4_1, 3072]), kwargs = {})
#   %sigmoid : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg2_1,), kwargs = {})
#   %mul_8 : Tensor "bf16[s18, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %sigmoid), kwargs = {})
#   %b12x_blockscaled_linear : Tensor "bf16[s18, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.vllm.b12x_blockscaled_linear.default](args = (%mul_8, None, 2560, %arg3_1), kwargs = {})
#   return %buf0
triton_poi_fused_b12x_blockscaled_linear_mul_sigmoid_view_0 = async_compile.triton('triton_poi_fused_b12x_blockscaled_linear_mul_sigmoid_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_b12x_blockscaled_linear_mul_sigmoid_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 201326592}, 'kernel_num_gb': 0.150994944, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_b12x_blockscaled_linear_mul_sigmoid_view_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/e43a5e5849cc1fdb843a8926bb40105a243f6e3828be329c7eaad67865c98973/inductor_cache/cr/ccrx74hg4vnqfhg7rw5hdmoexbitjxsu2lnzzcqztjxjm2rswbrz.py
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add => add_66
# Graph fragment:
#   %getitem_6 : Tensor "bf16[s18, 2560][2560, 1]cuda:0" = PlaceHolder[target=getitem_6]
#   %getitem_7 : Tensor "bf16[s18, 2560][2560, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %add_66 : Tensor "bf16[s18, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, %getitem_7), kwargs = {})
#   return %add_66
triton_poi_fused_add_1 = async_compile.triton('triton_poi_fused_add_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 167772160}, 'kernel_num_gb': 0.12582912, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /cache/runtime/vllm/torch_compile_cache/torch_aot_compile/e43a5e5849cc1fdb843a8926bb40105a243f6e3828be329c7eaad67865c98973/inductor_cache/by/cbymdfx3oqh5cua3prtnav4qlhfw4memc33wwos2om3a53gl3gjj.py
# Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   zeros => full_default
# Graph fragment:
#   %full_default : Tensor "bf16[s18, 24, 128][3072, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg1_1, 24, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default
triton_poi_fused_zeros_2 = async_compile.triton('triton_poi_fused_zeros_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_zeros_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 100663296}, 'kernel_num_gb': 0.050331648, 'kernel_flop': 0, 'backend_hash': 'A7039D76A7EBE2D08A464FA07E305C18F99407C3C0F9BC9F8E8BC888C188E274', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1 = args
        args.clear()
        s59 = arg1_1
        s18 = arg4_1
        s72 = s18
        assert_size_stride(arg0_1, (s18, 12, 256), (3072, 256, 1), 'input')
        assert_size_stride(arg2_1, (s18, 3072), (3072, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf0 = empty_strided_cuda((s18, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [flatten, sigmoid, mul, b12x_blockscaled_linear], Original ATen: [aten.view, aten.sigmoid, aten.mul, vllm.b12x_blockscaled_linear]
            triton_poi_fused_b12x_blockscaled_linear_mul_sigmoid_view_0_xnumel = 3072*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_b12x_blockscaled_linear_mul_sigmoid_view_0.run(arg0_1, arg2_1, buf0, triton_poi_fused_b12x_blockscaled_linear_mul_sigmoid_view_0_xnumel, stream=raw_stream0)
            del arg0_1
            del arg2_1
            # Topologically Sorted Source Nodes: [flatten, sigmoid, mul, b12x_blockscaled_linear], Original ATen: [aten.view, aten.sigmoid, aten.mul, vllm.b12x_blockscaled_linear]
            buf1 = torch.ops.vllm.b12x_blockscaled_linear.default(buf0, None, 2560, arg3_1)
            del arg3_1
            del buf0
            buf2 = buf1
            assert_size_stride(buf2, (s18, 2560), (2560, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf2, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf1
            # Topologically Sorted Source Nodes: [all_reduce], Original ATen: [vllm.all_reduce]
            buf3 = torch.ops.vllm.all_reduce.default(buf2, 'tp:0')
            del buf2
            buf4 = buf3
            assert_size_stride(buf4, (s18, 2560), (2560, 1), 'torch.ops.vllm.all_reduce.default')
            assert_alignment(buf4, 16, 'torch.ops.vllm.all_reduce.default')
            del buf3
            assert_size_stride(arg5_1, (s18, 10240), (10240, 1), 'input')
            assert_size_stride(arg6_1, (s18, 4), (336, 1), 'input')
            assert_size_stride(arg7_1, (10240, ), (1, ), 'input')
            arg5_1 = copy_if_misaligned(arg5_1)
            arg6_1 = copy_if_misaligned(arg6_1)
            arg7_1 = copy_if_misaligned(arg7_1)
            # Topologically Sorted Source Nodes: [hyperconnection_combine_norm], Original ATen: [b12x.hyperconnection_combine_norm]
            buf5 = torch.ops.b12x.hyperconnection_combine_norm.default(arg5_1, buf4, arg6_1, arg7_1, 1e-06, 4817)
            del arg5_1
            del arg6_1
            del arg7_1
            del buf4
            buf6 = buf5[0]
            assert_size_stride(buf6, (s18, 10240), (10240, 1), 'torch.ops.b12x.hyperconnection_combine_norm.default')
            assert_alignment(buf6, 16, 'torch.ops.b12x.hyperconnection_combine_norm.default')
            buf7 = buf5[1]
            assert_size_stride(buf7, (s18, 10240), (10240, 1), 'torch.ops.b12x.hyperconnection_combine_norm.default')
            assert_alignment(buf7, 16, 'torch.ops.b12x.hyperconnection_combine_norm.default')
            del buf5
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear_1], Original ATen: [vllm.b12x_blockscaled_linear]
            buf8 = torch.ops.vllm.b12x_blockscaled_linear.default(buf7, None, 336, arg8_1)
            del arg8_1
            buf9 = buf8
            assert_size_stride(buf9, (s18, 336), (336, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf9, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf8
            assert_size_stride(arg9_1, (8192, 320), (320, 1), 'input')
            # Topologically Sorted Source Nodes: [getitem_2, hyperconnection_scaled_silu], Original ATen: [aten.slice, b12x.hyperconnection_scaled_silu]
            torch.ops.b12x.hyperconnection_scaled_silu.default(reinterpret_tensor(buf9, (s18, 320), (336, 1), 0), arg9_1, 4814)
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear_2], Original ATen: [aten.slice, vllm.b12x_blockscaled_linear]
            buf12 = torch.ops.vllm.b12x_blockscaled_linear.default(reinterpret_tensor(arg9_1, (s18, 320), (320, 1), 0), None, 10240, arg10_1)
            del arg10_1
            buf13 = buf12
            assert_size_stride(buf13, (s18, 10240), (10240, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf13, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf12
            assert_size_stride(arg11_1, (8192, 2560), (2560, 1), 'input')
            # Topologically Sorted Source Nodes: [hyperconnection_gate_mean], Original ATen: [b12x.hyperconnection_gate_mean]
            torch.ops.b12x.hyperconnection_gate_mean.default(buf7, buf13, arg11_1, 4815)
            del buf13
            del buf7
            # Topologically Sorted Source Nodes: [moe_forward_shared], Original ATen: [aten.slice, vllm.moe_forward_shared]
            buf16 = torch.ops.vllm.moe_forward_shared.default(reinterpret_tensor(arg11_1, (s18, 2560), (2560, 1), 0), reinterpret_tensor(arg11_1, (s18, 2560), (2560, 1), 0), reinterpret_tensor(arg11_1, (s18, 2560), (2560, 1), 0), None, arg12_1, 0, torch.bfloat16)
            del arg12_1
            buf17 = buf16[0]
            assert_size_stride(buf17, (s18, 2560), (2560, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf17, 16, 'torch.ops.vllm.moe_forward_shared.default')
            buf18 = buf16[1]
            assert_size_stride(buf18, (s18, 2560), (2560, 1), 'torch.ops.vllm.moe_forward_shared.default')
            assert_alignment(buf18, 16, 'torch.ops.vllm.moe_forward_shared.default')
            del buf16
            buf19 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
            triton_poi_fused_add_1_xnumel = 2560*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_1.run(buf19, buf18, triton_poi_fused_add_1_xnumel, stream=raw_stream0)
            del buf18
            # Topologically Sorted Source Nodes: [add, all_reduce_1], Original ATen: [aten.add, vllm.all_reduce]
            buf20 = torch.ops.vllm.all_reduce.default(buf19, 'tp:0')
            del buf19
            buf21 = buf20
            assert_size_stride(buf21, (s18, 2560), (2560, 1), 'torch.ops.vllm.all_reduce.default')
            assert_alignment(buf21, 16, 'torch.ops.vllm.all_reduce.default')
            del buf20
            assert_size_stride(arg13_1, (10240, ), (1, ), 'input')
            arg13_1 = copy_if_misaligned(arg13_1)
            # Topologically Sorted Source Nodes: [getitem_3, hyperconnection_combine_norm_1], Original ATen: [aten.slice, b12x.hyperconnection_combine_norm]
            buf22 = torch.ops.b12x.hyperconnection_combine_norm.default(buf6, buf21, reinterpret_tensor(buf9, (s18, 4), (336, 1), 320), arg13_1, 1e-06, 5142)
            del arg13_1
            del buf21
            del buf6
            del buf9
            buf23 = buf22[0]
            assert_size_stride(buf23, (s18, 10240), (10240, 1), 'torch.ops.b12x.hyperconnection_combine_norm.default')
            assert_alignment(buf23, 16, 'torch.ops.b12x.hyperconnection_combine_norm.default')
            buf24 = buf22[1]
            assert_size_stride(buf24, (s18, 10240), (10240, 1), 'torch.ops.b12x.hyperconnection_combine_norm.default')
            assert_alignment(buf24, 16, 'torch.ops.b12x.hyperconnection_combine_norm.default')
            del buf22
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear_3], Original ATen: [vllm.b12x_blockscaled_linear]
            buf25 = torch.ops.vllm.b12x_blockscaled_linear.default(buf24, None, 336, arg14_1)
            del arg14_1
            buf26 = buf25
            assert_size_stride(buf26, (s18, 336), (336, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf26, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf25
            # Topologically Sorted Source Nodes: [getitem_10, hyperconnection_scaled_silu_1], Original ATen: [aten.slice, b12x.hyperconnection_scaled_silu]
            torch.ops.b12x.hyperconnection_scaled_silu.default(reinterpret_tensor(buf26, (s18, 320), (336, 1), 0), arg9_1, 5139)
            # Topologically Sorted Source Nodes: [b12x_blockscaled_linear_4], Original ATen: [aten.slice, vllm.b12x_blockscaled_linear]
            buf29 = torch.ops.vllm.b12x_blockscaled_linear.default(reinterpret_tensor(arg9_1, (s18, 320), (320, 1), 0), None, 10240, arg15_1)
            del arg15_1
            del arg9_1
            buf30 = buf29
            assert_size_stride(buf30, (s18, 10240), (10240, 1), 'torch.ops.vllm.b12x_blockscaled_linear.default')
            assert_alignment(buf30, 16, 'torch.ops.vllm.b12x_blockscaled_linear.default')
            del buf29
            # Topologically Sorted Source Nodes: [hyperconnection_gate_mean_1], Original ATen: [b12x.hyperconnection_gate_mean]
            torch.ops.b12x.hyperconnection_gate_mean.default(buf24, buf30, arg11_1, 5140)
            del buf24
            del buf30
            # Topologically Sorted Source Nodes: [qwen_gdn_input_projections], Original ATen: [aten.slice, vllm.qwen_gdn_input_projections]
            buf33 = torch.ops.vllm.qwen_gdn_input_projections.default(reinterpret_tensor(arg11_1, (s18, 2560), (2560, 1), 0), 8192, 48, arg16_1)
            del arg11_1
            del arg16_1
            buf34 = buf33[0]
            assert_size_stride(buf34, (s18, 8192), (8192, 1), 'torch.ops.vllm.qwen_gdn_input_projections.default')
            assert_alignment(buf34, 16, 'torch.ops.vllm.qwen_gdn_input_projections.default')
            buf35 = buf33[1]
            assert_size_stride(buf35, (s18, 48), (48, 1), 'torch.ops.vllm.qwen_gdn_input_projections.default')
            assert_alignment(buf35, 16, 'torch.ops.vllm.qwen_gdn_input_projections.default')
            del buf33
            buf36 = empty_strided_cuda((s18, 24, 128), (3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
            triton_poi_fused_zeros_2_xnumel = 3072*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_zeros_2.run(buf36, triton_poi_fused_zeros_2_xnumel, stream=raw_stream0)
        return (buf34, buf35, buf36, buf23, reinterpret_tensor(buf26, (s18, 4), (336, 1), 320), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 12, 256), (3072, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = 8192
    arg2_1 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = None
    arg4_1 = 8192
    arg5_1 = rand_strided((8192, 10240), (10240, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((8192, 4), (336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = None
    arg9_1 = rand_strided((8192, 320), (320, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = None
    arg11_1 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = None
    arg13_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = None
    arg15_1 = None
    arg16_1 = None
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
