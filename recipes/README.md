# Recipes index

Every sparkrun recipe in this repo. `recipes/eugr/` is the served vLLM/b12x
family and the one that matters day to day; everything else is either an
earlier SGLang-era route or a bisection/debug script kept for history. See
the top-level `README.md` for how to boot the default recipe, and
`results/README.md` for the numbers behind each verdict below.

## `recipes/eugr/` — served recipe

| recipe | image | status | notes |
|---|---|---|---|
| `eugr-agents-serve-local16-la.yaml` | `spark-vllm-b12x:local-20260918-a8333658` | **default, serving** | local16 + `use_local_argmax_reduction: true` + `VLLM_MXFP8_LM_HEAD: "1"` (promoted 2026-09-21). See `results/arms/la-lmq/verdict.md`. |
| `eugr-agents-serve-local16.yaml` | same image | fallback | same as the default minus `use_local_argmax_reduction`. |
| `eugr-agents-serve-local16-la-ghcr.yaml` | `ghcr.io/ursuciprian/spark-vllm-b12x:wheels-20260919-77bdd10-a833365` | not promoted | pull-based build from published wheels instead of `build.sh`. `ghcr2` gate: TC-45 passes (first time on this stack) but c1 ~-10% vs `la`. See `results/RESULTS.md` §"Cache-mount fix and ghcr image gate". |
| `eugr-agents-serve-local16-la-kk.yaml` | `ghcr.io/ursuciprian/spark-vllm-b12x:wheels-20260920-kk-a229c7a-e9ce547` | in progress | karmic-kraken-beta build (vllm `57a80980` + b12x `e9ce5477`) instead of dev/jovian-judgement; not yet gated against `la`. See `results/README.md`. |
| `eugr-agents-serve-local16-la-tc.yaml` | `spark-vllm-b12x:local-20260918-a8333658` | reverted | + `vllm-tc45-reasoning-structag-fix`. Fixes TC-45 (hardmode 93/100) but costs ~-11% c1. `results/arms/tooleval-fix/notes.md`. |
| `eugr-agents-serve-local16-la-lmq.yaml` | same image | superseded | the `la-lmq` arm's own recipe copy; its change (`VLLM_MXFP8_LM_HEAD`) is now folded into the default `la` recipe directly. |
| `eugr-agents-serve-local16-la-gemv.yaml` | same image + `vllm-qwen38-bf16-gemv` mod | rejected | boot failure, `results/arms/gemv/verdict.md`. |
| `eugr-agents-serve-local16-la-fusear.yaml` | same image | rejected | `fuse_allreduce_rms: true`, c1 regression. `results/kernel-pass/arms.md`. |
| `eugr-agents-serve-local16-la-spec3.yaml` | same image | rejected | `num_speculative_tokens: 3`, both concurrencies regress. |
| `eugr-agents-serve-local16-la-b12x0f3a8cb.yaml` | `spark-vllm-b12x:local-20260919-0f3a8cbf-b12x0f3a8cb` | rejected | b12x rebuilt at HEAD (6 commits ahead); deadlocks TP2 preparation from an empty plan cache (hang #1). |
| `eugr-agents-serve-local16-la-b12x0f3a8cb-noat.yaml` | same image | diagnostic only | `B12X_AUTOTUNE: "0"` isolates the hang to the autotune path; boots and serves, confirms the image itself is fine. |
| `eugr-agents-serve-local16-la-b12x0f3a8cb-rev06809d5.yaml` | same image | rejected | + mod reverting b12x commit `06809d5`; fixes hang #1, deadlocks again at a second barrier (hang #2). |
| `eugr-agents-serve-local16-la-fwd57f3572.yaml` | `spark-vllm-b12x:local-20260918-a8333658` | rejected | + mod forward-porting b12x commit `57f3572` (MoE-only) onto the old image; reproduces the deadlock (hang #3) — confirms it's a general scale-dependent bug, not version skew. |
| `eugr-agents-serve-local16-la-fwd57f3572-fix.yaml` | same image | rejected | fwd57f3572 with the fix attempt; still loses 4/38 candidate races, repeatable regression. `results/arms/fwd57f3572-fix/`. |
| `eugr-agents-serve-local16-la-fwd57f3572-trace.yaml` | same image | diagnostic | tracing variant of the fwd57f3572 arm, not a promotion candidate. |
| `eugr-agents-serve-local16-la-occ_dynhigh.yaml`, `-occ_dynlow.yaml`, `-occ_microhigh.yaml`, `-occ_microlow.yaml` | same image | rejected | `B12X_MICRO_MAX_ACTIVE_CLUSTERS` / `B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS` sweeps; all within noise, kernel clamps occupancy to 48 SMs regardless. |
| `eugr-agents-serve-local16-la-lprof.yaml` | same image | profiling only | + `vllm-decode-profiler` mod, rank-local `torch.profiler`. Not for serving. Source of `results/profiling/README.md`. |
| `eugr-agents-serve-local16-la-prof.yaml` | same image | profiling only | `--profiler-config` variant (deadlocks under TP=2, superseded by `-lprof`). Not for serving. |
| `eugr-agents-serve-local16-autotune.yaml` | same image | experimental | exhaustive-autotune arm; forces a multi-batch retune, hits the same candidate-racing deadlock class documented in `results/kernel-pass/prep-deadlock/mechanism.md`. |
| `eugr-agents-serve-local-dv.yaml` | `spark-vllm-b12x:local-20260918-a8333658-dv` | rejected | reduced-vocab MTP draft head; 0% acceptance. `results/arms/dv/`. |
| `eugr-agents-serve-local-ep.yaml` | `spark-vllm-b12x:local-20260918-a8333658` | rejected (expected) | `--enable-expert-parallel`; b12x's EP path is W4A16-only, this checkpoint is NVFP4. Boots only to confirm the failure mode. |
| `eugr-agents-serve-local-gdnbf16.yaml` | same image | rejected | `--mamba-ssm-cache-dtype bfloat16`; cannot boot, b12x requires float32 GDN state. |
| `eugr-agents-serve-local.yaml` | same image | superseded | predecessor of `-local16`, `max_num_seqs` not yet raised to 16. |
| `eugr-agents-serve-capsizes.yaml` | `vllm-node-b12x` | debug arm | `cudagraph_capture_sizes` padded at 3/5/6/7/12, straggler diagnostic. |
| `eugr-agents-serve-mtp3.yaml` | `vllm-node-b12x` | probe | `num_speculative_tokens=3`, straggler geometry probe. |
| `eugr-agents-serve-noshare16.yaml` | `vllm-node-b12x` | probe | `max_num_seqs 16` + `index_share_for_mtp_iteration=false`, QSA top-k index-reuse probe. |
| `eugr-agents-serve-nospec.yaml` | `vllm-node-b12x` | probe | MTP speculative decoding removed entirely, straggler isolation. |
| `eugr-agents-serve-scratchfix16.yaml` | `vllm-node-b12x` | superseded | `vllm-qwen-scratch-isolation` mod applied as a live mod; the fix is now baked into the served image instead. |
| `eugr-agents-serve-spectrace.yaml`, `-spectrace16.yaml` | `vllm-node-b12x` | debug arms | `vllm-spec-trace` mod, per-request draft/accept logging for the straggler investigation. |
| `eugr-agents-serve.yaml` | `vllm-node-b12x` | superseded | pre-b12x-own-image serving default; agents-tuned (8 seqs, 8192-token prefill chunks, util 0.85) + decode-aware prefill scheduling. |
| `eugr-agents.yaml` | `vllm-node-b12x` | superseded | the base agents-tuned recipe `eugr-agents-serve.yaml` grew from. |
| `eugr-agents-kvbf16.yaml` | `vllm-node-b12x` | quality arm (A1) | BF16 KV instead of fp8. |
| `eugr-agents-radixark.yaml` | `vllm-node-b12x` | quality arm (R0) | RadixArk checkpoint, fp8 KV. |
| `eugr-agents-radixark-kvbf16.yaml` | `vllm-node-b12x` | quality arm (R1) | RadixArk checkpoint + BF16 KV. |
| `eugr-agents-nvidia-kvbf16.yaml` | `vllm-node-b12x` | quality arm (N1, fallback) | nvidia checkpoint + BF16 KV, only used if R0/R1 both fail to boot. |
| `eugr-agents-oldckpt.yaml` | `vllm-node-b12x` | superseded | old PTQ checkpoint revision, superseded by the QAD revision `7c4f1bc1`. |
| `eugr-agents-mtp3.yaml`, `-mtp5.yaml`, `-mtp6.yaml` | `vllm-node-b12x` | bisect | MTP draft-token-count sweep; MTP 5/6 fail to boot (`QSA currently supports at most four speculative tokens`). |
| `eugr-agents-roce1m.yaml`, `-roce4m.yaml` | `vllm-node-b12x` | bisect | RoCE all-reduce threshold sweep (1MB/4MB vs default 2MB); did not confirm outside the noise band on re-check. |
| `eugr-agents-decode-aware.yaml`, `eugr-agents-lpt4096.yaml`, `batched4096.yaml` | `vllm-node-b12x` | superseded | decode-aware prefill scheduling / long-prefill-threshold variants from the 2026-09-15/16 bisection that led to the fix now in `eugr-agents.yaml`. |
| `eugr-b12x-cluster.yaml` | `vllm-node-b12x` | superseded | earliest cluster-serving recipe copied from eugr's upstream, pre-agents-tuning. |
| `eugr-gdn-triton.yaml`, `eugr-kv-auto.yaml`, `eugr-linear-default.yaml`, `eugr-moe-default.yaml`, `eugr-no-roce-ar.yaml`, `eugr-nomtp.yaml` | `vllm-node-b12x` | bisect | one-flag-off-from-eugr-stock arms (Triton GDN, default KV dtype, no b12x linear/MoE backend, RoCE all-reduce off, no MTP) used to isolate which flags matter. |
| `gputil80.yaml`, `mamba-bf16.yaml`, `prefill2.yaml`, `seqs16.yaml` | `vllm-node-b12x` | bisect | further one-flag concurrency/memory/prefill sweep arms from the same family; `gpu_memory_utilization` above 0.80 was later shown structurally unviable on Spark's unified memory (`results/arms/gpu_mem_util_verdict.md`). |

## `recipes/dflash2/` — DFlash2 speculative-decoding route

Not the served route. `flashnext-dflash2-serve.yaml` / `-harvest.yaml`
explore a DFlash2 drafter instead of MTP; kept for a possible future
re-evaluation, not gated against the current `la` baseline.

## `recipes/flashnext-*.yaml`, `recipes/arms/`, `recipes/retired/` — SGLang-era

The pre-vLLM/b12x serving route and its bisection arms, from before the
2026-09-15 switch to eugr's b12x vLLM stack:

- `recipes/flashnext-*.yaml` — the SGLang recipe family (`-bigkv*`,
  `-fp8kv-1m8*`, `-vllm*`, `-deep-concurrency`, `-c16-short-candidate`, …),
  measured in `results/RESULTS.md` §"Reference: earlier engine comparison"
  and `results/fp8-gate/`.
- `recipes/retired/` — copies of the SGLang recipes no longer maintained,
  kept only so old result files still resolve to a recipe on disk.
- `recipes/arms/` — the SGLang one-flag bisection ladder (`fn-*.yaml`, ~50
  recipes) plus its driver scripts (`ab_ladder.sh`, `run_fn_ladder.sh`,
  `vllm_ladder*.sh`, `night_watchdog.sh`) and `node-guard/` (an OOM-guard
  systemd unit for the bench host). Raw run data lives on dgx-01, not
  mirrored into this repo; see `results/README.md` §"SGLang-era bisection
  arms".

None of these serve the model today — see the top-level `README.md` Quick
start for the recipe that does.
