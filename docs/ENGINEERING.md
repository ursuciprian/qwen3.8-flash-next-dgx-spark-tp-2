# Engineering notes

Profiling, rejected arms, known-issue detail and build provenance, moved out of
the [README](../README.md) so it stays short. Every measurement and verdict:
[results/README.md](../results/README.md).

Contents: [Profile](#profile-rank-local-torchprofiler-resultsprofilingreadmemd) ·
[Rejected arms](#rejected-arms) · [Known issues / fixes](#known-issues--fixes) ·
[Build provenance](#build-provenance)

## Profile (rank-local `torch.profiler`, `results/profiling/README.md`)

Mod `archive/mods/vllm-decode-profiler/`, recipe
`qwen3.8-flash-next-nvfp4-tp2-profiler-local-rank.yaml`, summarized by
`scripts/prof_summary.py`. ~4% profiler overhead. Load: bench_sweep counting
prompt (temp 0, thinking off) at c1 and c8, so the step mix reflects ~4
accepted drafts per step.

| | c1 (55 ms/step) | c8 (87 ms/step) |
|---|---:|---:|
| GEMM (incl. MoE NVFP4 `siluMoEDynamicKer` ~27-34%) | 83.6% | 65.6% |
| GDN / SSM | 2.3% | 13.5% |
| all-reduce (ROCE) | 4.0% | 6.0% |
| attention | 1.6% | 2.7% |
| sampler | 1.0% | 1.6% |
| MTP head | 0.04% | 0.02% |
| idle | 5.4% | 5.6% |

Reading: decode is compute-bound in the NVFP4 MoE GEMM at both concurrencies,
not communication (all-reduce ≤6%) and not the lm_head (MTP head 0.04% of
step, which is why the reduced-draft-vocab arm below has no speed upside even
where it works). GDN's growing time share at c8 tracks growing routed token
volume, not an unbatched-launch problem — that lead was traced and closed
(`results/kernel-pass/arms.md`, "Correction to survey.md").

## Rejected arms

Screened against the `la` baseline on the bench_sweep counting diagnostic
(temp 0, thinking off, aggregate tok/s); gate bar was +3% at c1 or +5% at c8
with nothing else worse than -2%. All from `results/kernel-pass/*.json`,
`results/arms/fwd57f3572-fix/sweep.json`, `results/arms/dv/`.

| label | change | c1 tok/s | c8 tok/s | verdict |
|---|---|---:|---:|---|
| fusear | `fuse_allreduce_rms: true` | 86.7 | 428.2 | reject — c1 regression |
| spec3 | `num_speculative_tokens: 3` (was 4) | 84.1 | 375.2 | reject — both regress |
| noat | `B12X_AUTOTUNE: 0` (diagnostic) | 81.1 | 414.8 | reject — confirms autotune worth ~18% at c1 |
| fwd57f3572 | forward-port b12x's native W4A16 MoE-autotune fix onto the old image | 88.1 | 419.8 | reject — route wins only 4/38 candidate races, repeatable regression |
| occ MICRO=32 | `B12X_MICRO_MAX_ACTIVE_CLUSTERS=32` | 97.1 | 429.9 | reject — noise, kernel clamps to 48 SMs anyway |
| occ DYNAMIC=32 | `B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS=32` | 94.7 | 426.2 | reject — noise, same clamp |
| gdnbf16 | `--mamba-ssm-cache-dtype bfloat16` | — | — | cannot boot: b12x requires `state_dtype == torch.float32` |
| dv | reduced draft vocab (47,149-id table) for the MTP head | boots | — | 0% MTP acceptance (7 of 151k drafts); not a speed lever anyway, MTP head is 0.04% of step |
| b12x HEAD `0f3a8cb` | rebuild at b12x master | — | — | deadlocks TP2 preparation, see below |
| RadixArk checkpoint, BF16 KV, old PTQ rev | — | — | — | rejected 2026-09-16, superseded by QAD `7c4f1bc1` |

Typo hypothesis ("quantized GDN / fp8 KV causes long-session typos") tested
2026-09-16 on both checkpoints at 8k-128k: 0 typos on either. Dead for this
stack.

## Known issues / fixes

- **Batch 5-7 straggler (fixed).** One request per decode round lost ~97% of
  its MTP drafts and stalled ~18 s at c5-7 (also 9, 12) on the fork revision
  eugr's image shipped. Traced to a shared-scratch collision in the QSA/GDN
  projection path; fixed by the fork's scratch-isolation commits, baked into
  our own image. `archive/mods/vllm-qwen-scratch-isolation/` documents the fix.
- **`B12X_AUTOTUNE=0` in eugr's Dockerfile.** Truncates kernel-selection
  tuning; a fresh boot does 81 tok/s instead of 96-98 at c1 on the
  bench_sweep counting diagnostic (`results/kernel-pass/arms.md`, noat).
  The recipe overrides it to `1`. The plan cache then persists across boots
  at `~/.cache/sparkrun/runtime-cache/vllm/<model>/b12x/` (168-169 MB).
- **logind `RemoveIPC` kills shm.** vLLM died with `'ShmRingBuffer' object
  has no attribute 'shared_memory'` when systemd-logind wiped shm on ssh
  session end. Fix: `loginctl enable-linger nvidia` on both nodes.
- **TP2 preparation hangs from an empty plan cache.** Any b12x commit past
  `a8333658` that grows the MoE-retune candidate contract (222 → 349-837
  candidates) enters a multi-batch candidate-racing path in
  `b12x/preparation/session.py` that deadlocks a from-empty-cache TP=2
  retune. The proximate trigger is the RoCE one-shot collective's default
  spin limit (`B12X_ROCE_SPIN_LIMIT`, ~20 s) expiring before the two ranks'
  MoE candidate racing converges, which poisons the runtime instead of just
  waiting longer. Fix: `B12X_ROCE_SPIN_LIMIT: "300000000"` (~300 s) in the
  recipe env plus `archive/mods/b12x-startup-boundedwait/` (bounded `Store.wait` in
  the fork's `B12xPreparationCoordinator._exchange()`, fails fast instead of
  parking forever). Both are in the default recipe. Full trace:
  `results/kernel-pass/prep-deadlock/mechanism.md`.
- **TC-45 parser bug.** Qwen3's reasoning and tool parsers collapse onto one
  shared `ParserEngine` (`vllm/parser/parser_manager.py`,
  `vllm/parser/qwen3.py`) whose `adjust_request()` never builds a tool-choice
  grammar, so `tool_choice=required` is silently unconstrained. Same gap for
  every model on the shared engine (Kimi K2, GLM-4.7-MoE, DeepSeek variants,
  Gemma4, Mistral, SeedOss, NemotronV3, Minimax M2). Fix exists
  (`archive/mods/vllm-tc45-reasoning-structag-fix/`, hardmode 93/100) but is not
  enabled — see [Quality](../README.md#quality).
- **`scripts/gate_arm.sh`** needs `--hardmode` to run all 88 scenarios; the
  first `la` gate accidentally ran the 69-scenario default set and gave a
  score that wasn't comparable — fixed, always pass `--hardmode` for a real
  promotion decision.
- lm_head online MXFP8 (W8A16) via `VLLM_MXFP8_LM_HEAD=1` — a fork feature
  (`local-inference-lab/vllm` `dev/jovian-judgement` `8e1f1e58`,
  `_supports_default_lm_head_quantization`), enabled here. Halves the bytes
  read for the verify-head lm_head matmul (previously BF16-only on b12x at
  M=1, cuBLAS fallback); see `results/arms/la-lmq/verdict.md`.
- **The two TP ranks loaded different checkpoint revisions (fixed 2026-09-23).**
  sparkrun 0.3.6 serves with `HF_HUB_OFFLINE=1`, so vLLM on each node resolves
  `refs/main` from that node's own HF cache. sparkrun's head-to-worker copy
  (sparkrun's own `model_distribute.sh`) is `rsync -a --size-only`. `refs/main` holds
  a 40-byte commit hash on both nodes, so it is never recopied: only its mtime
  follows the head. `model_sync.sh` also skips the download when any
  safetensors file is present, even when a revision is set. After the head
  moved to the QAD revision `7c4f1bc1`, the worker kept `ada4da32` (the old
  PTQ checkpoint). Worker plan caches and rank-1 serve logs show `ada4da32`
  from at least 2026-09-18 19:31 to 2026-09-23. Fix: every recipe for this
  checkpoint sets both `model_revision:` and `--revision 7c4f1bc1…`. sparkrun
  does not pass `model_revision` through to vLLM, so the flag is what pins the
  ranks. `scripts/validate_recipes.py` warns about any `vllm serve` without
  `--revision`.

## Build provenance

### Default image (b1, since 2026-09-25)

`ghcr.io/ursuciprian/spark-vllm-b12x:b1-20260925-b7fbaf96-14077fb3-warm`
(`sha256:57c2fbd8cd811a5d22a7f2e547453f97b875f1fb4c7de60a0c3ff9fba3a79e5c`,
arm64, public) is `spark-vllm-b12x:candidate-b12x-b7fbaf96-vllm-14077fb3`
(built on dgx-01 by `build.sh`, empty `draft_vocab` context) plus the
`docker/b0-warm/` layer (b12x plan seed `8ccf4799…json`; the bounded-wait
patch skips itself because this vLLM already has it). Pushed by
spark-vllm-b12x workflow `build-b0-warm` (run 36098908444).

Sources, tagged `shipped-b1-20260925` in both forks:

- vLLM [`ursuciprian/vllm@14077fb35`](https://github.com/ursuciprian/vllm/tree/shipped-b1-20260925)
  (branch `feat/candidate-old-b12x`): the b0 source plus the TC-45
  `tool_choice` parser fix (cheap variant, gates on `required`/named only),
  deferred GDN checkpoints wiring (`VLLM_GDN_DEFERRED_CHECKPOINTS`),
  request-boundary export for deferred, and the PLE per-state fix; the
  b12x-master startup cache exchange is reverted.
- b12x [`ursuciprian/b12x@b7fbaf96`](https://github.com/ursuciprian/b12x/tree/shipped-b1-20260925)
  (branch `feat/gdn-deferred-p1-tests`): b0's `a8333658` plus the deferred
  GDN decode kernels. Same io_uring-free loader and schema-5 selection cache
  as b0, so the b0 plan seed still hits.

### b0 image (default 2026-09-22 to 2026-09-25, recipe `qwen3.8-flash-next-2x-dgx-spark-previous`)

`ghcr.io/ursuciprian/spark-vllm-b12x:b0-20260918-a8333658-warm`
(`sha256:a3d5d90d1312edc9a79c86add6fdf72d50b2a73558e295fdf4ea110488fb615d`,
arm64, public) is `spark-vllm-b12x:local-20260918-a8333658` plus the
`docker/b0-warm/` layer (b12x plan seed, bounded-wait fix). Its vLLM reports
`0.1.dev187+g8e1f1e587.d20260918`: built on dgx-01 by `build.sh` from a clean
local clone at `local-inference-lab/vllm@8e1f1e587f`, then marked dirty
because eugr's Dockerfile runs its `docker/patch_vllm_*.py` scripts over the
source tree before building the wheel (build metadata: `vllm_repo:
local-source`, `vllm_ref: 8e1f1e587f`, b12x `a83336581a`, build date
2026-09-18T15:09Z; eugr/spark-vllm-docker was a fresh `--depth 1` clone,
most likely `53bd8e034a`).

Exact source, recovered 2026-09-23 from the image itself (no rebuild):
[`ursuciprian/vllm` tag `shipped-b0-20260918`](https://github.com/ursuciprian/vllm/tree/shipped-b0-20260918)
(commit `557deb55b`, branch `shipped/b0-20260918`):

1. `8e1f1e587f` (upstream `dev/jovian-judgement`).
2. `63f265599`: the six files eugr's scripts patch, copied from the image:
   `flashinfer_b12x_swigluoai` (`fused_moe/experts/flashinfer_b12x_moe.py`,
   `fused_moe/oracle/nvfp4.py`, `utils/flashinfer.py`),
   `disable_minimax_qk_rmsnorm_ipc` (`minimax_rms_norm/rms_norm_tp.py`),
   `wsl_cuda_uma` (`utils/mem_utils.py`), `spark_kv_cache_cleanup`
   (`v1/worker/gpu_worker.py`).
3. `557deb55b`: cherry-pick of the bounded-wait fix `8b0934c73`
   (`v1/worker/b12x_startup.py`), the only file the warm layer changes.

Check: all 2350 tracked `vllm/*.py` files at the tag are byte-identical
to the warm image's installed package. The other 200 `.py` files in the image
are generated at build time (`vllm_flash_attn/`, `third_party/`,
`_version.py`). Compiled extensions (`csrc/`) cannot be compared from the image;
they were built from `8e1f1e587f` with eugr's build-stage patches.

Relation to `ursuciprian/vllm@dgx-spark` (`77bdd1070`): both branch from
`8e1f1e587f`. `dgx-spark` adds the TC-45 parser fix (`5d1df69f4`, not
shipped), the bounded-wait fix (`8b0934c73`, shipped as commit 3 above) and
the optional reduced-vocab draft head (`77bdd1070`, not shipped). It does not
carry eugr's build-time patches, which the tag does.

### Pull-based wheels image (archived alternate)

Own image `spark-vllm-b12x:local-20260918-a8333658` was built locally from
eugr's Dockerfile plus the fork pins above. To make the build reproducible
off this hardware, wheels for vLLM, FlashInfer and b12x were built on the
dgx-01 self-hosted GitHub Actions runner and published as an immutable
release, then a public hosted (arm64) runner assembled and pushed the ghcr
image from those wheels:

- Build repo: https://github.com/ursuciprian/spark-vllm-b12x
- Wheel release: `wheels-20260919-77bdd10-a833365` (vLLM `77bdd10`, b12x
  `a833365`), each asset with a `.sha256` and `build-metadata.yaml`.
- Image: `ghcr.io/ursuciprian/spark-vllm-b12x:wheels-20260919-77bdd10-a833365`,
  digest `sha256:c0314d7c…`.
- Forks: `ursuciprian/vllm@dgx-spark` (`8e1f1e58` + the TC-45 parser fix +
  the bounded-wait startup fix + an optional reduced-vocab draft head, not
  used by the default recipe), `ursuciprian/b12x@dgx-spark` (`a8333658`) with
  `exp/fwd-57f3572` for the rejected forward-port arm.
- The recipe that serves this image, `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-ghcr-image.yaml`,
  has been pulled and gated on the nodes (`ghcr2`, cache-mount fix applied):
  TC-45 passes for the first time on this stack, but bench_sweep counting
  c1 is ~-10% vs `la`. Kept as an alternate pull-based path, not promoted to default — see
  `results/README.md`.


## Moved from the README (2026-09-23)

Configuration detail, recipe/archive notes, the known-issues table and the full
credits, kept here when the README became a short guide.

### Configuration

Key serving flags and env of the default recipe, from
[`qwen3.8-flash-next-2x-dgx-spark.yaml`](../recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml) (written for the b0 image; the b1 recipe adds `VLLM_GDN_DEFERRED_CHECKPOINTS` and `B12X_COMPILE_WORKERS`, see the README).

Image identity: eugr `spark-vllm-docker` Dockerfile `798528a2` + fork
`local-inference-lab/vllm` `dev/jovian-judgement` `8e1f1e58` + b12x
`a8333658`. Checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` QAD
revision `7c4f1bc1`.

#### Serving flags

| Flag | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | `2` | One rank per Spark |
| `--speculative-config` | `method: mtp`, `num_speculative_tokens: 4`, `draft_sample_method: probabilistic` | MTP width 4; probabilistic drafts keep acceptance up for default-temperature (1.0) clients |
| `--kv-cache-dtype` | `fp8` | fp8 KV |
| `--quantization` | `modelopt_mixed` | NVFP4 checkpoint |
| `--load-format` | `b12x` | b12x weight loader |
| `--gdn-decode-kernel` / `--linear-backend` / `--moe-backend` | `b12x` | b12x kernels for GDN, linear and MoE |
| `--enable-prefix-caching` | on | Reuses cached context across turns |
| `--mamba-cache-mode` | `align` | GDN (mamba) state cache mode |
| `--enable-chunked-prefill`, `--max-parallel-prefills` | on, `4` | Bounded concurrent prefills |
| `--prefill-policy` / `--decode-refill-target` | `decode-aware` / `auto` | Keeps a decode lane instead of letting long prefills starve decode |
| `--max-model-len` | `262144` | Per-request context ceiling |
| `--max-num-seqs` | `16` | Concurrent sequences |
| `--max-num-batched-tokens` | `8192` | Prefill tokens per step |
| `--block-size` | `16` | KV block size |
| `--gpu-memory-utilization` | `0.80` | Share of unified memory for weights + KV |
| `--reasoning-parser` / `--tool-call-parser` | `qwen3` / `qwen3_xml`, `--enable-auto-tool-choice` | Thinking and tool calls |
| `--compilation-config` | `fuse_act_quant: true` | Fused activation quant pass |
| `--no-enable-flashinfer-autotune` | set | FlashInfer autotune off; b12x kernel tuning is `B12X_AUTOTUNE` |
| `--served-model-name` | `qwen3.8-flash-next` (+ HF id) | Model name clients send |

#### Environment

| Variable | Value | Why |
|---|---|---|
| `B12X_AUTOTUNE` | `1` | eugr's Dockerfile sets `0`, which truncates kernel selection (81 vs 96-98 tok/s at c1 on the counting diagnostic) |
| `B12X_ROCE_SPIN_LIMIT` | `300000000` | ~300 s instead of ~20 s, so the TP2 MoE retune from an empty cache does not deadlock |
| `VLLM_MXFP8_LM_HEAD` | `1` | lm_head online MXFP8 (W8A16), halves bytes read by the verify-head lm_head matmul |
| `VLLM_ENABLE_ROCE_ALLREDUCE` / `VLLM_ROCE_ALLREDUCE_MAX_SIZE` | `1` / `2MB` | RoCE all-reduce between the two ranks |
| `B12X_POLICY_MODE` | `auto` | b12x kernel policy |
| `CUTE_DSL_ARCH` | `sm_121a` | GB10 target |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | V2 model runner |
| `VLLM_USE_AOT_COMPILE`, `VLLM_USE_MEGA_AOT_ARTIFACT` | `1` | AOT compile cache, warm boots |
| `VLLM_SSM_CONV_STATE_LAYOUT` | `DS` | GDN conv state layout |
| `SAFETENSORS_FAST_GPU` | `1` | Faster weight load |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Worker start method |

No mods at run time. The `b12x-startup-boundedwait` patch (bounded `Store.wait` in
TP2 preparation, fails fast instead of parking forever) is baked into the image
([`docker/b0-warm/Dockerfile`](../docker/b0-warm/Dockerfile)).

### Recipes

`sparkrun recipe list` / `sparkrun recipe search qwen3.8` against this registry
shows two recipes (renamed 2026-09-25, see [RENAMES.md](../recipes/RENAMES.md); the one-hot draft fallback moved to the archive). Both are pinned (`--revision 7c4f1bc1…`), use a public
warm image, and need no mods, no host mounts and no `--trust`.

| Recipe | Image | Use |
|---|---|---|
| [`qwen3.8-flash-next-2x-dgx-spark`](../recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml) | `ghcr.io/ursuciprian/spark-vllm-b12x:b1-20260925-b7fbaf96-14077fb3-warm` (`sha256:57c2fbd8…`) | **Default, serving.** Probabilistic MTP drafts, deferred GDN checkpoints (`VLLM_GDN_DEFERRED_CHECKPOINTS=1`), TC-45 fix. Cold boot ~9.4 min, warm restart ~3.8 min (2026-09-25). |
| [`qwen3.8-flash-next-2x-dgx-spark-previous`](../recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark-previous.yaml) | `ghcr.io/ursuciprian/spark-vllm-b12x:b0-20260918-a8333658-warm` (`sha256:a3d5d90d…`) | Fallback: the default from 2026-09-22 to 2026-09-25, unchanged. Roll back here. |
| [`qwen3.8-flash-next-nvfp4-tp2-argmax-drafts`](../archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-argmax-drafts.yaml) | b0 image | Archived 2026-09-25 (`archive/recipes/`). The default before 2026-09-22, one-hot drafts + `use_local_argmax_reduction: true`. For temperature-0 clients or a rollback. Moved onto the warm image 2026-09-23; not boot-tested on it yet (same build, same baked-in fix as its old local image + mod). |

Everything else (bisection arms, rejected experiments, other checkpoints, the
SGLang-era route, the alternate `wheels-20260919` ghcr image) is in
[`archive/recipes/`](../archive/recipes/README.md) with its mods in
[`archive/mods/`](../archive/mods/README.md). sparkrun does not clone or scan
`archive/`; run an archived recipe by path from a clone, e.g.
`sparkrun run ./archive/recipes/qwen3.8-flash-next/<file>.yaml --hosts <head>,<worker>`
(many need a local-only image or host paths from the pair; see the archive index).
Where each file went: [recipes/RENAMES.md](../recipes/RENAMES.md).

### Known issues / limits

| Issue | Status | Detail |
|---|---|---|
| Checkpoint revision split: rank 1 loaded `ada4da32` while rank 0 loaded `7c4f1bc1` (2026-09-18 to 09-23) | **Fixed**: recipes pin `model_revision` + `--revision` | sparkrun's `rsync --size-only` never refreshes the worker's `refs/main`; see [Checkpoint revision split](BENCHMARKS.md#checkpoint-revision-split-fixed-2026-09-23) |
| Batch 5-7 straggler: one request per round stalled ~18 s at c5-7 (also 9, 12) | **Fixed** by our own image | Fork's QSA/GDN scratch-isolation commits; `archive/mods/vllm-qwen-scratch-isolation/` documents the fix |
| `B12X_AUTOTUNE=0` in eugr's Dockerfile: fresh boot 81 tok/s instead of 96-98 at c1 (counting diagnostic) | **Worked around**: recipe sets `1` | Plan cache persists at `~/.cache/sparkrun/runtime-cache/vllm/<model>/b12x/` (168-169 MB); [`results/kernel-pass/arms.md`](../results/kernel-pass/arms.md) |
| logind `RemoveIPC` kills shm (`'ShmRingBuffer' object has no attribute 'shared_memory'`) | **Host fix** | `loginctl enable-linger nvidia` on both nodes |
| TP2 preparation hangs from an empty plan cache on b12x commits past `a8333658` | **Fixed in default recipe** | `B12X_ROCE_SPIN_LIMIT: "300000000"` + `archive/mods/b12x-startup-boundedwait/` (baked into the image); [`results/kernel-pass/prep-deadlock/mechanism.md`](../results/kernel-pass/prep-deadlock/mechanism.md) |
| TC-45: `tool_choice=required` silently unconstrained (shared Qwen3 `ParserEngine`) | **Fixed in b1** (default since 2026-09-25); still open on `-b0` / `-argmax-drafts` | Cheap variant in vLLM `14077fb35` gates only `required`/named, no measured speed cost; hardmode TC-45 passes, targeted re-runs 5/5. History: `archive/mods/vllm-tc45-reasoning-structag-fix/` |
| `scripts/gate_arm.sh` without `--hardmode` runs only 69 of 88 scenarios | **Usage** | Always pass `--hardmode` for a real promotion decision |
| Prose 64k-cached c16 regresses vs old la (38.15 -> 36.78) | **Known** | See the comparison in [At a glance](BENCHMARKS.md#default-vs-previous-default-old-la-one-hot-argmax-drafts) |
| Raw `results/arms/` files | **Not all mirrored** | Some live on dgx-01 only, see [results/README.md](../results/README.md) |

Full write-ups (deadlock mechanism, parser detail, lm_head MXFP8):
[docs/ENGINEERING.md](ENGINEERING.md#known-issues--fixes).
Profiling, rejected arms and build provenance also live in
[docs/ENGINEERING.md](ENGINEERING.md).

### Credits

The vLLM route served by default (`recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml`,
image `spark-vllm-b12x:local-20260918-a8333658`) is built on other people's work.
Exact pins:

- [eugr](https://github.com/eugr) (Eugene Rakhmatulin):
  [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) at `798528a2`
  (2026-09-16) is the Dockerfile our image is built from, unchanged except for the
  build-plumbing fixes in `~/GEN-AI/build/apply_submodule_fix.py`;
  [sparkrun](https://github.com/eugr/sparkrun) 0.3.6 launches every recipe here and
  defines the recipe/mod format; [llama-benchy](https://github.com/eugr/llama-benchy)
  at `e9be344` is the base of our fork
  [ursuciprian/llama-benchy](https://github.com/ursuciprian/llama-benchy)
  (`0d4de42`, adds `--prompt-mode task`) that measures the agent-task grids, and
  release 0.4.0 measures the prose grids; the recipe family (formerly
  `eugr-agents`) started as his `eugr-agents.yaml`.
- [local-inference-lab](https://github.com/local-inference-lab) (Luke Alonso):
  the vLLM fork [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm)
  branch `dev/jovian-judgement` at `8e1f1e58` (2026-09-16), which carries the
  Qwen3.8-Flash-Next model code, MTP drafter and the QSA scratch-isolation fix that
  removed our batch 5-7 straggler; the [b12x](https://github.com/local-inference-lab/b12x)
  kernels at `a8333658` (2026-09-17): GDN prefill/decode, NVFP4 GEMM, kernel
  autotune and plan cache; the checkpoint
  [local-inference-lab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)
  QAD revision `7c4f1bc1` (2026-09-16).
- [MiaAI-Lab](https://github.com/MiaAI-Lab): the fast sparse-attention SGLang
  profile the SGLang recipes grew from, the llama-benchy measurement spec, the
  expert-parallel launch shape, and the 47,149-id draft-vocab table
  `files/draft_vocab_en_code_47k.txt` from
  [Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)
  at `3f99abc2` (2026-09-16), used to test the reduced-vocab MTP head (AGPL-3.0;
  used locally, not redistributed in this repo or in our image tags).
- [RadixArk](https://huggingface.co/RadixArk): the NVFP4 checkpoint the SGLang
  recipes serve, and the day-0 SGLang engine work for this model.
- [tonyd2wild](https://github.com/tonyd2wild): the vLLM SM121 overlays and the
  TP2 profile the first vLLM recipe grew from, and the 40-prompt category harness
  (`tools/tony-bench`: `bench_sweep.py` counting diagnostic, `bench_categories.py`).
- [Weschera](https://github.com/Weschera): spark-bench, the 76-scenario graded
  eval behind the fp8 quality gate.
- [SeraphimSerapis](https://github.com/SeraphimSerapis): tool-eval-bench
  2.6.1 (`--hardmode`, 88 scenarios), the tool-calling gate.
- Upstream: sgl-project/sglang#36845 and #38855, vllm-project/vllm#53945 and
  #55557, whose authors fixed the kernels these recipes depend on.

Their work made this run faster on my hardware. The measurements, the
cache-reuse diagnosis, the SM121 fp8 KV fix, the straggler root-cause and own
image build, the `B12X_AUTOTUNE=1` finding, the quality gates and the
draft-vocab experiments are mine, and so are any mistakes.
