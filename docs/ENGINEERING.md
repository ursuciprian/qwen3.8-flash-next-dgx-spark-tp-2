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

### Shipped image (default recipe)

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
