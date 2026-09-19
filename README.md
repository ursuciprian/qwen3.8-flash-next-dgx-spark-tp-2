# Qwen3.8-Flash-Next NVFP4 on two DGX Sparks

Serving recipes for `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` at tensor
parallel 2 across two NVIDIA DGX Spark (GB10, SM121) over ConnectX-7 RoCE,
packaged for [sparkrun](https://sparkrun.dev). The served route is vLLM on a
b12x-kernel fork, built as our own container image. All numbers below come
from fresh boots, measured 2026-09-18/19, with the same harnesses.

## Quick start

```sh
sparkrun registry add https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2
sparkrun run recipes/eugr/eugr-agents-serve-local16-la.yaml --cluster <your-cluster> --tp 2 --trust
```

`--trust` accepts the mod hook that patches files inside the container before
serve (see [Why the mods](#why-the-mods)). Cold boot 20-30 minutes (kernel
autotune from an empty plan cache), warm boot with a populated
`~/.cache/sparkrun/runtime-cache/vllm/<model>/b12x/` a few minutes. The server
answers on port 8000 with the OpenAI API, model name `qwen3.8-flash-next`.

## Which recipe

| Recipe | Image | Status |
|---|---|---|
| `recipes/eugr/eugr-agents-serve-local16-la.yaml` | `spark-vllm-b12x:local-20260918-a8333658` | **Default, serving.** `use_local_argmax_reduction: true` in the MTP spec config, startup robustness mod, `B12X_AUTOTUNE=1`. |
| `recipes/eugr/eugr-agents-serve-local16.yaml` | same image | Fallback: same recipe without `use_local_argmax_reduction`. |
| `recipes/eugr/eugr-agents-serve-local16-la-ghcr.yaml` | `ghcr.io/ursuciprian/spark-vllm-b12x:wheels-20260919-77bdd10-a833365` (`sha256:c0314d7c…`) | Built from a public wheel release (see [Build provenance](#build-provenance)). **Not yet gated on the nodes** — pull and boot were interrupted by a shutdown before this recipe was measured. Do not treat it as the default until it has a screen against `la`. |

Image identity for the default: eugr `spark-vllm-docker` Dockerfile `798528a2`
+ fork `local-inference-lab/vllm` `dev/jovian-judgement` `8e1f1e58` + b12x
`a8333658`. Checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` QAD
revision `7c4f1bc1`. fp8 KV, MTP width 4, prefix caching on, `max_num_seqs 16`.

## Headline numbers

`tools/tony-bench/bench_sweep.py`, 3 rounds x 300 tokens, counting workload,
default (`la`) recipe. Boot-to-boot noise band: c1 85-99 tok/s, c8 412-442
tok/s — nothing inside that band is a result; promotion decisions used 3-boot
medians. Sources: `results/arms/la/sweep.json`, `results/arms/la-final/sweep.json`,
`results/kernel-pass/prep-deadlock/sweep_la_boundedwait_restore_r2.json`.

| concurrency | agg tok/s | per-stream tok/s | TTFT (c1 only) |
|---|---:|---:|---:|
| c1 | 95.7-98.8 (boot band 85-99) | same | 0.74-0.79 s |
| c4 | 285.2 | 72.3 | 2.2 s |
| c8 | 412-442 | 52.5-56.0 | 4.3 s |
| c16 | 629.9 | 40.3 | 8.3 s |

Cold prefill 2236-2303 tok/s (c1). Decode-probe workload mix (single stream,
peak of 3 repeats, `results/arms/la/decode_probe.txt`): code 59.6, structured
90.5, counting 98.0, prose 47.9 tok/s.

### Category harness (tonyd2wild, 40 prompts, concurrency 1)

Old = eugr's original nightly image before this repo's own build; local16 =
own image, plain recipe; la = own image with `use_local_argmax_reduction`.

| | median | json | html | reasoning | coding | summary | format | prose | narrative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old eugr image | 65.8 | 90.1 | 86.1 | 74.9 | 69.4 | 47.5 | 46.4 | 43.8 | 40.4 |
| local16 (own image) | 76.4 | 90.4 | 97.0 | 77.5 | 85.8 | 50.1 | 47.6 | 43.4 | 44.4 |
| la (own image, argmax) | not re-run on this harness | 94.6 | 95.1 | 78.8 | 82.4 | 51.5 | 60.0 | 49.1 | 45.0 |

`decode_probe.py` workload mix (mean of 3, `code/structured/counting/prose`):
old 56/73/94/46, local16 58.1/71.7/87.2/45.9, la 53.5/72.3/89.6/45.5.

Real-prompt concurrency lane, per-stream/aggregate tok/s, old image only
(2026-09-17, not re-measured on the own image): x2 61.9/83.2, x4 47.8/81.0,
x6 37.7/112.6, x8 33.7/124.8.

## Quality gates (`la`)

- **Fidelity probe** (`results/arms/la/fidelity.json`, `fidelity_probe.txt`):
  100% exact retrieval at 8k/32k/64k/128k context, 0 typos, thinking on.
- **tool-eval-bench 2.6.1 `--hardmode`** (88 scenarios): 88-91/100 across
  boots; baseline image scored 90. Two scenarios fail consistently:
  - **TC-45** (`tool_choice=required`): parser bug, see
    [Known issues / fixes](#known-issues--fixes). Fixed by
    `mods/vllm-tc45-reasoning-structag-fix/` → 93/100, but that mod costs
    c1 throughput about -12% (`results/arms/la-tc/sweep.json`, 85.0 vs 95.7).
    Not enabled by default.
  - **TC-68**: model wraps JSON in a code fence with commentary; the scenario
    intentionally sends no `response_format`, so this is model compliance,
    not a server bug — not fixable without defeating the test.
- **Straggler probe**: batch 5-16 clean (was one request per round stalling
  ~18 s at c5-7/9/12 on the prior fork revision; fixed by the own image).
  Accepted/draft ratio ~3.9 of 4.
- **MTP acceptance** on real prompts, per draft position: ~90/81/75/70%
  (`results/arms/la/mtp_metrics.txt`: overall 76767/24344 draft-tokens*4
  positions, per-position 21911/19751/18173/16932 accepted). On the counting
  workload used for the throughput sweep, acceptance is far higher — 98.9%
  overall, 99.9/99.1/98.6/97.9% by position
  (`results/profiling/README.md`).

## Profile (rank-local `torch.profiler`, `results/profiling/README.md`)

Mod `mods/vllm-decode-profiler/`, recipe
`eugr-agents-serve-local16-la-lprof.yaml`, summarized by
`scripts/prof_summary.py`. ~4% profiler overhead.

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

Screened against the `la` baseline; gate bar was +3% at c1 or +5% at c8 with
nothing else worse than -2%. All from `results/kernel-pass/*.json`,
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
  our own image. `mods/vllm-qwen-scratch-isolation/` documents the fix.
- **`B12X_AUTOTUNE=0` in eugr's Dockerfile.** Truncates kernel-selection
  tuning; a fresh boot serves single-stream at 81 tok/s instead of 96-98.
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
  recipe env plus `mods/b12x-startup-boundedwait/` (bounded `Store.wait` in
  the fork's `B12xPreparationCoordinator._exchange()`, fails fast instead of
  parking forever). Both are in the default recipe. Full trace:
  `results/kernel-pass/prep-deadlock/mechanism.md`.
- **TC-45 parser bug.** Qwen3's reasoning and tool parsers collapse onto one
  shared `ParserEngine` (`vllm/parser/parser_manager.py`,
  `vllm/parser/qwen3.py`) whose `adjust_request()` never builds a tool-choice
  grammar, so `tool_choice=required` is silently unconstrained. Same gap for
  every model on the shared engine (Kimi K2, GLM-4.7-MoE, DeepSeek variants,
  Gemma4, Mistral, SeedOss, NemotronV3, Minimax M2). Fix exists
  (`mods/vllm-tc45-reasoning-structag-fix/`, hardmode 93/100) but is not
  enabled — see Quality gates above.
- **`scripts/gate_arm.sh`** needs `--hardmode` to run all 88 scenarios; the
  first `la` gate accidentally ran the 69-scenario default set and gave a
  score that wasn't comparable — fixed, always pass `--hardmode` for a real
  promotion decision.

## Build provenance

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
- The recipe that serves this image, `eugr-agents-serve-local16-la-ghcr.yaml`,
  is on an open PR (#8) and has not been pulled or gated on the nodes yet —
  do not switch the default to it until it has a `la`-comparable screen.

## Repo map

| Path | What |
|---|---|
| `recipes/eugr/` | The served recipe family. `eugr-agents-serve-local16-la.yaml` (default), `-local16.yaml` (fallback), `-la-ghcr.yaml` (ungated, PR #8). Everything else under here is an arm tried and either promoted (baked into the default) or rejected — see `results/kernel-pass/arms.md` for the record. |
| `recipes/arms/`, `recipes/dflash2/`, `recipes/retired/`, `recipes/flashnext-*.yaml` | Earlier SGLang-era recipes and bisection arms, kept for history; not the served route. |
| `mods/` | Engine patches, one directory per mod, `mods/README.md` has details for the SGLang-era mods (stale for the newer vLLM/b12x ones — see below). |
| `scripts/` | `gate_arm.sh` (full quality gate, needs `--hardmode`), `fidelity_probe.py`, `prof_summary.py`, `needle_ladder.py`, `decode_probe.py`, `validate_recipes.py`, `run.sh`, and the arm-bisection scripts (`vllm_ladder.sh`, `qwen_ladder*.sh`, …). |
| `results/kernel-pass/`, `results/profiling/`, `results/arms/` | The 2026-09-18/19 arms, hang evidence, and per-kernel profile behind the tables above. |
| `results/eugr-b12x/`, `results/fp8-gate/`, `results/sglang-*`, `results/vllm-cached-*` | Earlier (pre-09-18) SGLang/vLLM-nightly measurements, kept for history. |
| `misc/` | Working notes not promoted into a result file. |

### Mods, one line each

Current as of 2026-09-19; `mods/README.md` only documents the SGLang-era mods
(`sglang-*`, `vllm-flashnext-nightly-8a728663`, `vllm-qsa-fp8kv-pr55557`) and
needs a follow-up pass to cover the rest:

- `b12x-startup-boundedwait` — **in the default recipe.** Bounded-wait fix for
  the TP2 preparation deadlock (see Known issues).
- `b12x-startup-trace` — diagnostic instrumentation for the same handshake,
  kept for future debugging, not in the served recipe.
- `b12x-revert-06809d5` — reverse-applies a b12x commit to unblock the first
  hang barrier; superseded (the forward-ported fix it enabled was itself
  rejected), kept for reference.
- `b12x-fwd-57f3572` — forward-ports b12x's native W4A16 MoE-autotune fix onto
  the old image; the resulting arm was rejected (see table above).
- `vllm-qwen-scratch-isolation` — documents the fork's QSA/GDN scratch-isolation
  fix that removed the batch 5-7 straggler; baked into the image, not applied
  as a live mod.
- `vllm-tc45-reasoning-structag-fix` — fixes TC-45 (`tool_choice=required`);
  not enabled by default, costs ~-12% c1 throughput (see Quality gates).
- `vllm-decode-profiler` — rank-local `torch.profiler` wrapper used for the
  per-kernel profile above; no cross-rank RPC, avoids the `--profiler-config`
  endpoint deadlock.
- `vllm-dv-devicefix` — device-placement fixes for the reduced-draft-vocab MTP
  head experiment; boots clean but the arm is rejected (0% acceptance).
- `vllm-spec-trace` — diagnostic per-step spec-decode tracer used to chase the
  batch 5-7 straggler; not in the served recipe.
- `sglang-gdn-b12x-decode` — SGLang-era: routes SGLang's GDN decode to the
  b12x CuTeDSL kernel; not used by the vLLM route this repo now serves.
- `sglang-sm121-qsa-fp8kv`, `vllm-flashnext-nightly-8a728663`,
  `vllm-qsa-fp8kv-pr55557`, `sglang-radix-chunked-insert-fix` — SGLang/vLLM-
  nightly era, documented in `mods/README.md`.

## Hardware

- 2x DGX Spark: GB10, SM121, 128 GB LPDDR5X unified memory.
- ConnectX-7 RoCE between the nodes.

## Credits

The vLLM route now served by default (`recipes/eugr/eugr-agents-serve-local16-la.yaml`,
image `spark-vllm-b12x:local-20260918-a8333658`) is built on other people's work.
Exact pins:

- [eugr](https://github.com/eugr) (Eugene Rakhmatulin):
  [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) at `798528a2`
  (2026-09-16) is the Dockerfile our image is built from, unchanged except for the
  build-plumbing fixes in `~/GEN-AI/build/apply_submodule_fix.py`;
  [sparkrun](https://github.com/eugr/sparkrun) 0.3.6 launches every recipe here and
  defines the recipe/mod format; [llama-benchy](https://github.com/eugr/llama-benchy)
  at `e9be344` measured the concurrency sweeps; the `eugr-agents` recipe family
  started as his `eugr-agents.yaml`.
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
  (`tools/tony-bench`) behind every decode, TTFT and cold-prefill figure here.
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

## License

Apache-2.0, see `LICENSE`. The vLLM overlays under `mods/` are modified copies
of Apache-2.0 vLLM files and keep their upstream headers. The spark-bench
reports under `results/fp8-gate/` are output of that tool, reproduced with its
run labels intact.
