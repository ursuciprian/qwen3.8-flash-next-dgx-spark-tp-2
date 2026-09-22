# `vllm-gdn-meta-fuse`

Cherry-pick of `local-inference-lab/vllm@a18246b06` ("Fuse GDN metadata copies
and state refresh across cache groups", `integration/karmic-kraken-beta`,
2026-09-22, Luke Alonso) onto our serving fork base `8e1f1e587f`.

## Does it apply to 8e1f1e58?

Yes, cleanly. `vllm/v1/attention/backends/b12x_gdn_metadata.py` is
**byte-identical** between `8e1f1e58` and `a18246b06^`
(`82811f63bf58f70bd8afd870f8decef6300331c863c664a468ad54879703b900`), and
`gdn_attn.py` diverges by only 4 lines, none of them near the touched
`update_block_table` hunk. `git cherry-pick a18246b06` onto `8e1f1e58` applies
with **zero conflicts**, and the image
`spark-vllm-b12x:local-20260918-a8333658` carries exactly those pre-image
bytes.

## What it changes

`GDNAttentionMetadataBuilder.update_block_table` runs when a second KV-cache
group reuses a captured sibling group's metadata (`cached_attn_metadata` in
`gpu_model_runner._build_attn_group_metadata`, and the V2 equivalent in
`gpu/attn_utils.py`). It used to do:

```
mixed.copy_worklists_from(src)     # 3 x torch._foreach_copy_ over 14 tensors
mixed.refresh_state_indices(...)   # 3 x zero_() + gather/copy_ + torch.where
```

roughly ten device launches plus a host-side `torch.where`, every step, per
extra group. The commit adds one Triton kernel,
`_copy_worklists_and_refresh_states`, that does the worklist copies **and** the
recurrent-state / checkpoint index refresh in a single launch. It reads the
worklists from the *source* group so there is no cross-CTA dependency, keeps
the destination buffer addresses stable (CUDA-graph replay still holds), and
clears the padded rows.

Additive and self-guarding: `copy_and_refresh_from` falls back to the old
two-call sequence when `source is self` or the buffers are not CUDA, both old
methods stay, and the capacity check is unchanged. One call site moves
(`gdn_attn.py:904`).

## Expected effect

KK's own number on this exact arch (TP2 Qwen serving, warmed prompts, verifier
step): **41.19 -> 40.16 ms, -2.5%**, i.e. ~1.03 ms of device work removed per
step. The saving is a *fixed* per-step launch cost (it scales with
`max_seqs`/`state_columns`/worklist sizes, not with the batch), so against our
measured steps (`results/profiling/README.md` on dgx-01) ~1 ms lands as:

| our step | wall | expected after | delta |
|---|---:|---:|---:|
| c1 rank0 | 55.03 ms | ~54.0 ms | ~-1.9% |
| c8 rank0 | 87.11 ms | ~86.1 ms | ~-1.2% |

It lands **inside the busy 94%**, not in the idle window: GPU idle fraction is
5.4% at c1 and 5.6% at c8, so this is real device work removed, not host
overlap.

**Gate before believing it:** the fused path only fires when a second GDN cache
group reuses the captured sibling's metadata. Our boot
(`mamba_cache_mode=align`, MTP k=4, so the drafter carries its own GDN layers)
is the geometry KK measured, but we have not confirmed the group count in a
boot log. If our config resolves to a single GDN group, this mod is a
functional no-op and the A/B will read flat. Check the A/B, not the commit
message.

## Not shipped from the upstream commit

`tests/v1/attention/test_gdn_metadata_builder.py` (+88) and
`benchmarks/kernels/benchmark_gdn_uniform_metadata.py` (+113). Neither
directory is installed in the image, so the patch carries the two `vllm/` files
only. For the GPU pass, the upstream test to run against a built wheel is
`pytest tests/v1/attention/test_gdn_metadata_builder.py` (KK: 56 GDN metadata
tests pass, including graph-replay coverage for mixed / speculative / empty
batches, strided indices, stable pointers and allocation-free updates).

## Checks

CPU-tested in `spark-vllm-b12x:local-20260918-a8333658` (`docker run --rm`, no
GPU): SHA256 pre-image on both files, dry-run, SHA256 post-image, AST parse,
import smoke, a CPU equivalence check that the fallback branch of
`copy_and_refresh_from` produces byte-equal buffers to the old two-call path
(mixed spec/non-spec rows, a zero checkpoint offset, strided indices), and the
capacity-mismatch `ValueError`. Re-running exits 0 ("already applied").

The Triton kernel itself is **not** exercised on CPU (no driver in the
container). It needs a GPU boot.

## Test

```yaml
mods: [b12x-startup-boundedwait, vllm-gdn-meta-fuse]
```

i.e. `recipes/eugr/eugr-agents-serve-local16-la.yaml` with that line, A/B
against the unmodified `la` recipe on `bench_sweep.py --levels 1,8`.
