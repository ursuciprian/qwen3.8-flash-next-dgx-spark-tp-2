
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
