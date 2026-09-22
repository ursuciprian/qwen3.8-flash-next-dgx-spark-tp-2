#!/usr/bin/env python3
"""vllm-gdn-deferred: teach this image's vLLM to drive b12x deferred GDN
checkpoints, behind VLLM_GDN_DEFERRED_CHECKPOINTS (default off).

Exact-block replacement, fail closed: every edit must match its pre-image
verbatim or nothing is written. Every touched file is ast-parsed before it is
written back. Idempotent: re-running on a patched tree is a no-op (run.sh
gates on the marker and on the post-image sha256 before calling this).

Net effect == ursuciprian/vllm feat/gdn-deferred-checkpoints 29bf8477f, cut
against 8e1f1e587f. The new module vllm/v1/worker/gdn_deferred_commit.py is
copied in by run.sh, not patched in here. Generated from that diff, so the
blocks below are verbatim pre-images; do not hand-edit them.
"""

import ast
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages"


# ---- vllm/envs.py ----
EDITS_0 = [
    (
        """\
    VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE: bool = True
    VLLM_GDN_DECODE_KERNEL: Literal["b12x", "cuda", "triton"] = "cuda"
    VLLM_DISABLE_PYNCCL: bool = False
    VLLM_USE_OINK_OPS: bool = False
""",
        """\
    VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE: bool = True
    VLLM_GDN_DECODE_KERNEL: Literal["b12x", "cuda", "triton"] = "cuda"
    VLLM_GDN_DEFERRED_CHECKPOINTS: bool = False
    VLLM_DISABLE_PYNCCL: bool = False
    VLLM_USE_OINK_OPS: bool = False
""",
    ),
    (
        """\
        ["b12x", "cuda", "triton"],
        case_sensitive=False,
    ),
    # Disable pynccl (using torch.distributed instead)
""",
        """\
        ["b12x", "cuda", "triton"],
        case_sensitive=False,
    ),
    # Trade the b12x GDN decode kernel's per-verified-token state checkpoints
    # for one base checkpoint plus compact per-token records, replaying the
    # accepted prefix instead of selecting a checkpoint column. Requires the
    # b12x branch that implements Caps(deferred_checkpoints=...), the b12x GDN
    # decode kernel, align mamba cache mode, and no request-boundary
    # checkpoints. Off by default.
    "VLLM_GDN_DEFERRED_CHECKPOINTS": lambda: (
        os.getenv("VLLM_GDN_DEFERRED_CHECKPOINTS", "0").lower()
        in ("true", "1", "yes", "on")
    ),
    # Disable pynccl (using torch.distributed instead)
""",
    ),
]

# ---- vllm/v1/worker/mamba_utils.py ----
EDITS_1 = [
    (
        """\
    src_col,
    dst_col,
    token_bias,
    block_table_ptrs_ptr,
    block_table_stride_req,
""",
        """\
    src_col,
    dst_col,
    conv_bias,
    temporal_bias,
    block_table_ptrs_ptr,
    block_table_stride_req,
""",
    ),
    (
        """\
    ``precopy_mamba_align_fused_kernel``, mirroring the V1 copy specs
    (``get_conv_copy_spec`` / ``get_temporal_copy_spec``):
    - conv state (conv_width > 0): shift the window by ``token_bias`` tokens,
      ``state[bt[src_col], token_bias:] ->
      state[bt[dst_col], :conv_width - token_bias]``
    - temporal state: ``token_bias`` selects the accepted speculative column,
      ``state[bt[src_col + token_bias]] -> state[bt[dst_col]]``

    The caller owns the decision logic (which columns, whether to copy); this
""",
        """\
    ``precopy_mamba_align_fused_kernel``, mirroring the V1 copy specs
    (``get_conv_copy_spec`` / ``get_temporal_copy_spec``):
    - conv state (conv_width > 0): shift the window by ``conv_bias`` tokens,
      ``state[bt[src_col], conv_bias:] ->
      state[bt[dst_col], :conv_width - conv_bias]``
    - temporal state: ``temporal_bias`` selects the accepted speculative
      column, ``state[bt[src_col + temporal_bias]] -> state[bt[dst_col]]``

    The two biases are separate because they are not always the same number.
    With deferred GDN checkpoints (``VLLM_GDN_DEFERRED_CHECKPOINTS``) the
    speculative columns hold per-token records rather than checkpoints, so
    there is no column to select: the b12x commit has already replayed the
    accepted prefix into ``bt[src_col]`` and the temporal half must copy it
    with ``temporal_bias == 0``. The conv half is unaffected and keeps shifting
    its window by the accepted-token bias.

    The caller owns the decision logic (which columns, whether to copy); this
""",
    ),
    (
        """\
        # state_elem_size alignment: tensor strides and token offsets are
        # measured in whole elements before conversion to bytes.
        num_dst_tokens = conv_width - token_bias
        for token_idx in range(0, num_dst_tokens):
            for row_base in range(0, dim_rows, COPY_BLOCK_SIZE):
""",
        """\
        # state_elem_size alignment: tensor strides and token offsets are
        # measured in whole elements before conversion to bytes.
        num_dst_tokens = conv_width - conv_bias
        for token_idx in range(0, num_dst_tokens):
            for row_base in range(0, dim_rows, COPY_BLOCK_SIZE):
""",
    ),
    (
        """\
                    src_block_addr
                    + rows * row_stride
                    + (token_idx + token_bias) * state_elem_size
                )
                dst_byte_addr = (
""",
        """\
                    src_block_addr
                    + rows * row_stride
                    + (token_idx + conv_bias) * state_elem_size
                )
                dst_byte_addr = (
""",
    ),
    (
        """\
            return
        # SD conv: copy
        #   state[bt[src_col], token_bias:] ->
        #   state[bt[dst_col], :conv_width - token_bias]
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        src_block_addr = state_base_addr + src_block_id * state_block_stride
""",
        """\
            return
        # SD conv: copy
        #   state[bt[src_col], conv_bias:] ->
        #   state[bt[dst_col], :conv_width - conv_bias]
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        src_block_addr = state_base_addr + src_block_id * state_block_stride
""",
    ),
    (
        """\
        src_block_addr = state_base_addr + src_block_id * state_block_stride
        token_bytes = state_inner_size * state_elem_size
        num_dst_tokens = conv_width - token_bias

        # Distinct blocks and exact self-copies cannot have a destructive
""",
        """\
        src_block_addr = state_base_addr + src_block_id * state_block_stride
        token_bytes = state_inner_size * state_elem_size
        num_dst_tokens = conv_width - conv_bias

        # Distinct blocks and exact self-copies cannot have a destructive
""",
    ),
    (
        """\
        # Distinct blocks and exact self-copies cannot have a destructive
        # overlap, so retain the u64-vectorized single-CTA copy.
        if src_block_id != dest_block_id or token_bias == 0:
            src_addr = src_block_addr + token_bias.to(tl.int64) * token_bytes
            copy_size = num_dst_tokens.to(tl.int64) * token_bytes
            _memcpy_u64_tiled(
""",
        """\
        # Distinct blocks and exact self-copies cannot have a destructive
        # overlap, so retain the u64-vectorized single-CTA copy.
        if src_block_id != dest_block_id or conv_bias == 0:
            src_addr = src_block_addr + conv_bias.to(tl.int64) * token_bytes
            copy_size = num_dst_tokens.to(tl.int64) * token_bytes
            _memcpy_u64_tiled(
""",
    ),
    (
        """\
        # without a barrier.
        for token_idx in range(0, num_dst_tokens):
            src_token = src_block_addr + (token_idx + token_bias) * token_bytes
            dst_token = dst_addr + token_idx * token_bytes
            _memcpy_u64_tiled(
""",
        """\
        # without a barrier.
        for token_idx in range(0, num_dst_tokens):
            src_token = src_block_addr + (token_idx + conv_bias) * token_bytes
            dst_token = dst_addr + token_idx * token_bytes
            _memcpy_u64_tiled(
""",
    ),
    (
        """\
        return

    # Temporal state: copy state[bt[src_col + token_bias]] -> state[bt[dst_col]]
    # Body u64 range is partitioned across TEMPORAL_TILES CTAs to keep the
    # SMs filled at small batch.
""",
        """\
        return

    # Temporal: copy state[bt[src_col + temporal_bias]] -> state[bt[dst_col]]
    # Body u64 range is partitioned across TEMPORAL_TILES CTAs to keep the
    # SMs filled at small batch.
""",
    ),
    (
        """\
    # Body u64 range is partitioned across TEMPORAL_TILES CTAs to keep the
    # SMs filled at small batch.
    actual_src_block_id = tl.load(block_table_base + src_col + token_bias).to(tl.int64)
    src_addr = state_base_addr + actual_src_block_id * state_block_stride
    # Use natural block data size (inner_size * elem_size), NOT
""",
        """\
    # Body u64 range is partitioned across TEMPORAL_TILES CTAs to keep the
    # SMs filled at small batch.
    actual_src_block_id = tl.load(
        block_table_base + src_col + temporal_bias
    ).to(tl.int64)
    src_addr = state_base_addr + actual_src_block_id * state_block_stride
    # Use natural block data size (inner_size * elem_size), NOT
""",
    ),
    (
        """\
    if destination <= 0:
        return
    _copy_mamba_state_block(
        state_idx,
""",
        """\
    if destination <= 0:
        return
    # Boundary checkpoints keep the checkpoint-column contract: deferred GDN
    # checkpoints are refused when request-boundary checkpointing is on,
    # because one capture can ask for several distinct biases per request and
    # a single accepted-prefix commit cannot express that.
    _copy_mamba_state_block(
        state_idx,
""",
    ),
    (
        """\
        src_col,
        0,
        token_bias,
        block_table_ptrs_ptr,
""",
        """\
        src_col,
        0,
        token_bias,
        token_bias,
        block_table_ptrs_ptr,
""",
    ),
    (
        '''\
    # the existing 2D-grid contract.
    TEMPORAL_TILES: tl.constexpr = 1,
):
    """
''',
        '''\
    # the existing 2D-grid contract.
    TEMPORAL_TILES: tl.constexpr = 1,
    DEFERRED_TEMPORAL: tl.constexpr = False,
    # Deferred GDN checkpoints need the copy decision before the copy runs, so
    # the b12x commit can replay the accepted prefix into bt[src_col] first.
    # DECISION_ONLY emits that decision and returns without copying, which
    # keeps the decision logic in exactly one place.
    DECISION_ONLY: tl.constexpr = False,
    commit_src_col_ptr=None,
    commit_accepted_ptr=None,
):
    """
''',
    ),
    (
        """\
        return

    bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx
    _copy_mamba_state_block(
""",
        """\
        return

    if DECISION_ONLY:
        if state_idx == 0 and tile_idx == 0:
            tl.store(commit_src_col_ptr + req_idx, src_block_idx)
            tl.store(commit_accepted_ptr + req_idx, accept_token_bias + 1)
        return

    bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx
    _copy_mamba_state_block(
""",
    ),
    (
        """\

    bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx
    _copy_mamba_state_block(
        state_idx,
""",
        """\

    bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx
    # With deferred checkpoints the speculative columns hold records, so the
    # accepted prefix has already been replayed into bt[src_block_idx] by
    # GdnDeferredCommit and the temporal half is a plain block copy.
    temporal_bias = 0 if DEFERRED_TEMPORAL else accept_token_bias
    _copy_mamba_state_block(
        state_idx,
""",
    ),
    (
        """\
        dest_block_idx,
        accept_token_bias,
        block_table_ptrs_ptr,
        block_table_stride_req,
""",
        """\
        dest_block_idx,
        accept_token_bias,
        temporal_bias,
        block_table_ptrs_ptr,
        block_table_stride_req,
""",
    ),
    (
        '''\
    # the 2D-grid contract; > 1 requires a 3D grid.
    TEMPORAL_TILES: tl.constexpr = 1,
):
    """Pre-copy mamba "align" state across block boundaries.
''',
        '''\
    # the 2D-grid contract; > 1 requires a 3D grid.
    TEMPORAL_TILES: tl.constexpr = 1,
    DEFERRED_TEMPORAL: tl.constexpr = False,
    DECISION_ONLY: tl.constexpr = False,
    commit_src_col_ptr=None,
    commit_accepted_ptr=None,
):
    """Pre-copy mamba "align" state across block boundaries.
''',
    ),
    (
        """\

    token_bias = tl.load(token_bias_ptr + req_idx)
    _copy_mamba_state_block(
        state_idx,
""",
        """\

    token_bias = tl.load(token_bias_ptr + req_idx)
    if DECISION_ONLY:
        if state_idx == 0 and tile_idx == 0:
            tl.store(commit_src_col_ptr + req_idx, src_col)
            tl.store(commit_accepted_ptr + req_idx, token_bias + 1)
        return
    temporal_bias = 0 if DEFERRED_TEMPORAL else token_bias
    _copy_mamba_state_block(
        state_idx,
""",
    ),
    (
        """\
        dst_col,
        token_bias,
        block_table_ptrs_ptr,
        block_table_stride_req,
""",
        """\
        dst_col,
        token_bias,
        temporal_bias,
        block_table_ptrs_ptr,
        block_table_stride_req,
""",
    ),
    (
        """\
    precopy_token_bias_buf: CpuGpuBuffer | None = None

    # Flag to track if metadata has been populated
    is_initialized: bool = False
""",
        """\
    precopy_token_bias_buf: CpuGpuBuffer | None = None

    # Deferred GDN checkpoints (VLLM_GDN_DEFERRED_CHECKPOINTS). Both stay None
    # unless the feature resolved on, which keeps every driver below on its
    # shipped path. ``gdn_block_table`` is the GDN group's block table in
    # request-slot order, the rows the commit metadata is gathered from.
    gdn_deferred_commit: Any | None = None
    gdn_block_table: torch.Tensor | None = None

    # Flag to track if metadata has been populated
    is_initialized: bool = False
""",
    ),
    (
        """\
            self.block_table_ptrs[i] = _reinterpret_u64_as_i64(bt.data_ptr())

        self.is_initialized = True

""",
        """\
            self.block_table_ptrs[i] = _reinterpret_u64_as_i64(bt.data_ptr())

        # Deferred GDN checkpoints gather their commit window out of the GDN
        # group's table. Hybrid models put every GDN layer in one mamba group,
        # so a second group here means the feature has nothing well-defined to
        # commit and is refused rather than guessed at.
        self._initialize_gdn_deferred_commit(forward_context, block_tables)

        self.is_initialized = True

""",
    ),
    (
        """\

        self.is_initialized = True

    def compute_aligned_state_indices(
""",
        """\

        self.is_initialized = True

    def _initialize_gdn_deferred_commit(
        self,
        forward_context: dict[str, Any],
        block_tables: list[torch.Tensor],
    ) -> None:
        from vllm.v1.worker.gdn_deferred_commit import GdnDeferredCommit

        layers = [
            layer
            for layer in forward_context.values()
            if getattr(layer, "b12x_gdn_deferred_checkpoints", False)
        ]
        if not layers:
            return
        if len(block_tables) != 1:
            raise ValueError(
                "deferred GDN checkpoints require a single mamba block-table "
                f"group, got {len(block_tables)}"
            )
        block_table = block_tables[0]
        columns = int(layers[0].b12x_gdn_state_index_columns)
        commit = GdnDeferredCommit(
            max_num_reqs=int(block_table.shape[0]),
            state_index_columns=columns,
            device=block_table.device,
        )
        commit.bind_layers(layers)
        commit.precompile()
        self.gdn_deferred_commit = commit
        self.gdn_block_table = block_table

    def compute_aligned_state_indices(
""",
    ),
    (
        """\
            return
        grid = (num_reqs, self.total_states, _TEMPORAL_TILES)
        precopy_mamba_align_fused_kernel[grid](
            state_idx_gpu,
""",
        """\
            return
        grid = (num_reqs, self.total_states, _TEMPORAL_TILES)
        deferred = self.gdn_deferred_commit
        if deferred is not None and deferred.active:
            deferred.reset(num_reqs)
            precopy_mamba_align_fused_kernel[(num_reqs, 1, 1)](
                state_idx_gpu,
                src_col_gpu,
                token_bias_gpu,
                self.block_table_ptrs,
                self.block_table_stride_req,
                self.state_base_addrs,
                self.state_block_strides,
                self.state_elem_sizes,
                self.state_inner_sizes,
                self.state_conv_widths,
                self.state_group_indices,
                self.state_dim_row_count,
                self.state_dim_row_stride,
                idx_mapping,
                num_reqs,
                COPY_BLOCK_SIZE=1024,
                CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
                HAS_IDX_MAPPING=idx_mapping is not None,
                TEMPORAL_TILES=1,
                DECISION_ONLY=True,
                commit_src_col_ptr=deferred.src_col,
                commit_accepted_ptr=deferred.accepted,
            )
            deferred.commit(
                num_reqs=num_reqs, block_table=self.gdn_block_table
            )
        precopy_mamba_align_fused_kernel[grid](
            state_idx_gpu,
""",
    ),
    (
        """\
            HAS_IDX_MAPPING=idx_mapping is not None,
            TEMPORAL_TILES=_TEMPORAL_TILES,
        )

""",
        """\
            HAS_IDX_MAPPING=idx_mapping is not None,
            TEMPORAL_TILES=_TEMPORAL_TILES,
            DEFERRED_TEMPORAL=deferred is not None and deferred.active,
        )

""",
    ),
    (
        """\

        grid = (num_reqs, self.total_states, _TEMPORAL_TILES)
        postprocess_mamba_fused_kernel[grid](
            num_accepted_tokens_snapshot,
""",
        """\

        grid = (num_reqs, self.total_states, _TEMPORAL_TILES)
        deferred = self.gdn_deferred_commit
        if deferred is not None and deferred.active:
            # Decide, commit, then copy. See vllm/v1/worker/gdn_deferred_commit.
            deferred.reset(num_reqs)
            postprocess_mamba_fused_kernel[(num_reqs, 1, 1)](
                num_accepted_tokens_snapshot,
                state_idx_gpu,
                None,
                new_num_computed_tokens_gpu,
                None,
                self.block_table_ptrs,
                self.block_table_stride_req,
                self.state_base_addrs,
                self.state_block_strides,
                self.state_elem_sizes,
                self.state_inner_sizes,
                self.state_conv_widths,
                self.state_group_indices,
                self.state_dim_row_count,
                self.state_dim_row_stride,
                num_accepted_tokens_gpu,
                idx_mapping,
                num_reqs,
                block_size=self.block_size,
                COPY_BLOCK_SIZE=1024,
                CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
                HAS_IDX_MAPPING=True,
                PRECOMPUTED_NEW_COMPUTED=True,
                TEMPORAL_TILES=1,
                DECISION_ONLY=True,
                commit_src_col_ptr=deferred.src_col,
                commit_accepted_ptr=deferred.accepted,
            )
            deferred.commit(
                num_reqs=num_reqs, block_table=self.gdn_block_table
            )
        postprocess_mamba_fused_kernel[grid](
            num_accepted_tokens_snapshot,
""",
    ),
    (
        """\
            PRECOMPUTED_NEW_COMPUTED=True,
            TEMPORAL_TILES=_TEMPORAL_TILES,
        )

""",
        """\
            PRECOMPUTED_NEW_COMPUTED=True,
            TEMPORAL_TILES=_TEMPORAL_TILES,
            DEFERRED_TEMPORAL=deferred is not None and deferred.active,
        )

""",
    ),
    (
        """\
        fused.token_bias.np[:num_reqs] = 0

    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
""",
        """\
        fused.token_bias.np[:num_reqs] = 0

    # Deferred GDN checkpoints turn state-cell uniqueness from a freshness
    # property into a correctness one: a speculative column that aliases the
    # running block stops being a redundant checkpoint write and becomes a
    # per-token record overwriting the base state. Checking it costs a handful
    # of host comparisons per step, which is worth paying for a failure mode
    # that would otherwise be silent corrupted output.
    deferred_columns = 0
    if fused is not None and fused.ctx.gdn_deferred_commit is not None:
        deferred_columns = fused.ctx.gdn_deferred_commit.state_index_columns

    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
""",
    ),
    (
        """\
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            accept_token_bias = int(input_batch.num_accepted_tokens_cpu[i]) - 1
            if fused is not None:
                assert accept_token_bias >= 0
""",
        """\
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            accept_token_bias = int(input_batch.num_accepted_tokens_cpu[i]) - 1
            if deferred_columns:
                window = req_state.block_ids[mamba_group_ids[0]][
                    prev_state_idx : prev_state_idx + deferred_columns
                ]
                if len(set(window)) != len(window):
                    raise ValueError(
                        "deferred GDN checkpoints require a request's state "
                        "window to hold distinct blocks; the running block "
                        f"would be overwritten by a record. window={window}"
                    )
            if fused is not None:
                assert accept_token_bias >= 0
""",
    ),
]

# ---- vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py ----
EDITS_2 = [
    (
        """\
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.worker.workspace import (
    retain_cuda_graph_capture_resource,
""",
        """\
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.worker import gdn_deferred_commit
from vllm.v1.worker.workspace import (
    retain_cuda_graph_capture_resource,
""",
    ),
    (
        """\
        self._b12x_local_key_heads = local_key_heads
        self._b12x_local_value_heads = local_value_heads
        # Caps are immutable declaration metadata.  Inspect geometry directly
        # instead of constructing an executable declaration.
""",
        """\
        self._b12x_local_key_heads = local_key_heads
        self._b12x_local_value_heads = local_value_heads
        # Raises when the env var asks for something this configuration cannot
        # do, rather than silently running the shipped checkpoint path.
        self._b12x_gdn_deferred_checkpoints = gdn_deferred_commit.resolve(
            vllm_config
        )
        # Caps are immutable declaration metadata.  Inspect geometry directly
        # instead of constructing an executable declaration.
""",
    ),
    (
        """\
        self._b12x_packed_qkv_width = caps.packed_qkv_width
        self._b12x_decode_staging = None

    def _make_b12x_gdn_caps(self, max_state_slots: int):
""",
        """\
        self._b12x_packed_qkv_width = caps.packed_qkv_width
        self._b12x_decode_staging = None

    # Read by vllm.v1.worker.gdn_deferred_commit through the forward context,
    # so it must exist on every GDN layer, not only the b12x ones.
    _b12x_gdn_deferred_checkpoints: bool = False

    @property
    def b12x_gdn_deferred_checkpoints(self) -> bool:
        return bool(self._b12x_gdn_deferred_checkpoints)

    @property
    def b12x_gdn_state_index_columns(self) -> int:
        return int(self._b12x_state_index_columns)

    def _make_b12x_gdn_caps(self, max_state_slots: int):
""",
    ),
    (
        """\
            gate_activation=self.norm.activation,
            qk_l2norm=True,
        )

""",
        """\
            gate_activation=self.norm.activation,
            qk_l2norm=True,
            deferred_checkpoints=self._b12x_gdn_deferred_checkpoints,
        )

""",
    ),
    (
        """\
        z: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
    ):
        plan = self._b12x_decode_plan
""",
        """\
        z: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        # Metadata overrides. The deferred-checkpoint commit runs outside the
        # forward pass, where the staged metadata still describes the previous
        # step, so it supplies its own window instead.
        state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        num_seqs: torch.Tensor | None = None,
    ):
        plan = self._b12x_decode_plan
""",
    ),
    (
        """\
            recurrent_state=self.kv_cache[1],
            query_start_loc=staging.query_start_loc,
            num_accepted_tokens=staging.num_accepted_tokens,
            state_indices=staging.state_indices,
            num_seqs=staging.num_seqs,
            num_tokens=staging.num_tokens,
            output=staging.output if output is None else output,
""",
        """\
            recurrent_state=self.kv_cache[1],
            query_start_loc=staging.query_start_loc,
            num_accepted_tokens=(
                staging.num_accepted_tokens
                if num_accepted_tokens is None
                else num_accepted_tokens
            ),
            state_indices=(
                staging.state_indices if state_indices is None else state_indices
            ),
            num_seqs=staging.num_seqs if num_seqs is None else num_seqs,
            num_tokens=staging.num_tokens,
            output=staging.output if output is None else output,
""",
    ),
    (
        """\
            output=staging.output if output is None else output,
        )

    def unbind_kv_cache(self) -> None:
""",
        '''\
            output=staging.output if output is None else output,
        )

    def commit_b12x_gdn_deferred(
        self,
        *,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_seqs: torch.Tensor,
        destination_indices: torch.Tensor,
    ) -> None:
        """Materialize this layer's accepted-prefix state in place.

        ``state_indices[r, 0]`` is the base checkpoint and ``[r, 1:]`` are that
        step's record blocks; the destination is column 0 again, so the commit
        is in place and the caller's block copy then moves it with a zero
        temporal bias.
        """
        if not self._b12x_gdn_deferred_checkpoints:
            return
        self._b12x_gdn_api.commit_deferred_checkpoints(
            self._bind_b12x_gdn_decode(
                state_indices=state_indices,
                num_accepted_tokens=num_accepted_tokens,
                num_seqs=num_seqs,
            ),
            destination_indices,
        )

    def precompile_b12x_gdn_deferred_commit(self) -> None:
        """Best-effort warm-up; the commit compiles lazily on first use."""
        if not self._b12x_gdn_deferred_checkpoints:
            return
        try:
            self._b12x_gdn_api.precompile_deferred_commit(
                self._bind_b12x_gdn_decode()
            )
        except PreparationResourceUnavailableError:
            logger.debug(
                "b12x GDN deferred commit not precompiled yet; it will "
                "compile on first use."
            )

    def unbind_kv_cache(self) -> None:
''',
    ),
]

EDITS = {
    'envs.py': EDITS_0,
    'v1/worker/mamba_utils.py': EDITS_1,
    'model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py': EDITS_2,
}


def fail(msg):
    print(f"FATAL: vllm-gdn-deferred: {msg}", file=sys.stderr)
    raise SystemExit(1)


staged = {}
for rel, pairs in EDITS.items():
    path = f"{ROOT}/vllm/{rel}"
    try:
        src = open(path).read()
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    for old, new in pairs:
        n = src.count(old)
        if n != 1:
            fail(
                f"{rel}: expected exactly 1 occurrence of the pre-image block, "
                f"found {n} (source drift). First line of block: "
                f"{old.splitlines()[0]!r}"
            )
        src = src.replace(old, new, 1)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        fail(f"{rel}: patched source does not parse: {exc}")
    staged[path] = src

# Only write once every file has matched and parsed.
for path, src in staged.items():
    open(path, "w").write(src)
    print(f"  vllm-gdn-deferred: patched {path}")
