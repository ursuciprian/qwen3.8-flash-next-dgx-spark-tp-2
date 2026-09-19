# vllm-qwen-scratch-isolation

Candidate fix for the c5-c7/c9/c12 MTP straggler (`project-qwen-straggler-batch5-7`):
one request per round loses ~97% of its speculative drafts (17-21s instead of
5.5-7.5s), root-caused to a shared-scratch collision, not batch composition or
tile-of-4 rows.

## Commits

Fork `dev/jovian-judgement`, three commits ahead of our shipped image
(`b40673cd0`, 2026-09-13):

- **`9c27ec0`** "b12x: prepare model-owned GPU plans through coordinated
  startup stages" — 94 files, +13004/-5901. A wholesale rewrite of the
  preparation/workspace API (`B12xPreparationUnit`, per-layer
  `get_b12x_preparation_units`, `PreparationSession`). Its own commit message
  admits incomplete qualification (6 failing tests, unmigrated DeepSeek V4.1
  callers, GPU serving not rerun after cleanup). **Not applied**: too large to
  patch cleanly against our image, and `200b6e2` is the only commit here that
  needs it (`context.prepared_plan`, `get_b12x_preparation_units`,
  `get_b12x_projection_workspace_sizes`, `b12x._lib.scratch.scratch_buffer_spec`
  all come from this commit, and none of them exist in our image).
- **`98cd717`** "Fix shared b12x scratch in concurrent Qwen projections" —
  GDN's `in_proj_qkvz`/`in_proj_ba` and QSA's `qkv_proj`/`indexer.index_qk_proj`
  each get their own preallocated workspace (`get_b12x_projection_workspaces`,
  built on `use_preallocated_workspace`, which our image already ships in
  `vllm/v1/worker/workspace.py`). Before this, both projections on the aux
  stream and the main stream shared whatever scratch the underlying
  `b12x_linear` split-K/activation kernel picked, so two projections launched
  concurrently (aux stream for one, main stream for the other, every decode
  step) could alias the same physical buffer. **Applied** — self-contained,
  does not need `9c27ec0`.
- **`200b6e2`** "Isolate QSA selector and projection scratch during overlapped
  decode" — extends the same idea to the QSA draft-selector's scratch, so
  selection writes can't overwrite the QKV/index projection scratch mid-flight.
  **Not applied**: depends on `9c27ec0`'s preparation API (see above).

## Hypothesis

Our victim row is **positional** (row index 4, or {5,8}, or {8,9} — the same
lane every run regardless of request id), which fits a fixed scratch-buffer
offset colliding with a fixed decode-batch row range far better than a
tile-of-4 kernel assumption (already refuted: c=12, three full tiles, still
stalls). `98cd717` is exactly this class of bug for the *projection* inputs
(GDN and QSA qkv/index projections launched concurrently on two streams every
decode step, sharing kernel-picked scratch). It is upstream of, and
independent from, the QSA *selector* scratch that `200b6e2` isolates — so it
is worth testing alone even though `200b6e2` (the commit that reads as a
closer match for "selector corrupts projection during MTP decode") can't be
applied without the `9c27ec0` prerequisite we're declining to pull in.

## Files

- `01-b12x-projection-workspaces.diff` — adds `get_b12x_projection_workspaces`
  to `vllm/utils/b12x.py`.
- `02-gdn-projection-isolation.diff` — `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`,
  `qwen_gdn_input_projections`.
- `03-qsa-projection-isolation.diff` — `vllm/models/qwen3_8_flash_next/nvidia/qsa.py`,
  `_qsa_input_projections`.

All three are hand-rebased against the file contents actually installed in
our image (`98cd717`'s raw diff has one failing hunk: our image's `qwen_gdn_linear_attn.py`
imports `get_b12x_gdn_decode`/`get_b12x_scratch_buffers` on one line, not the
multi-name block the upstream diff's context expects), then verified with
`patch -p0 --dry-run --forward` against the real installed files inside the
serving container — 0 failed hunks.

## Applying

`run.sh` applies the three diffs in order with `patch -p0 --forward` from `/`
inside the container, then AST-checks all three files. Idempotent: `patch
--forward` no-ops (exit 1, treated as already-applied) if the target already
has the change; `run.sh` treats that as success. Fail-closed: any other
`patch` failure aborts before AST-check.
