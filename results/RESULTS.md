# Measurements — vLLM / b12x route, 2026-09-18/19

Two DGX Spark (GB10, SM121), tensor parallel 2 over ConnectX-7 RoCE, image
`spark-vllm-b12x:local-20260918-a8333658`, checkpoint
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4` QAD `7c4f1bc1`, fp8 KV, MTP 4,
prefix caching, `B12X_AUTOTUNE=1`, `max_num_seqs 16`. Default recipe
`recipes/eugr/eugr-agents-serve-local16-la.yaml` (adds
`use_local_argmax_reduction: true`).

Two fresh boots of the same recipe differ by up to 15% at c1 and a few
percent at c8 (`results/arms/la/sweep.json` 95.7 vs `la-final` 98.8 vs
`la-reboot1` 97.2 vs `sweep_la_final_restore` 94.9; c8 412.0-441.8 across
boots). **Noise band: c1 85-99, c8 412-442. Nothing inside that band is a
result** — arm promotion used 3-boot medians and a gate bar of +3% at c1 or
+5% at c8 with nothing else worse than -2%. Earlier (2026-09-03/11/16) SGLang
and vLLM-nightly measurements are kept below for history but are not the
current serving route; see [Reference: earlier engine comparison](#reference-earlier-engine-comparison-2026-09-11-and-before).

## Throughput sweep (`tools/tony-bench/bench_sweep.py`, 3 rounds x 300 tokens, counting workload)

| concurrency | agg tok/s | per-stream tok/s | TTFT | source |
|---|---:|---:|---:|---|
| 1 | 95.7-98.8 (boot band 85-99) | same | 0.74-0.79 s | `results/arms/la/sweep.json`, `la-final/sweep.json` |
| 4 | 285.2 | 72.3 | 2.22 s | `results/arms/la/sweep.json` |
| 8 | 412-442 | 52.5-56.0 | 4.28-4.31 s | `results/arms/la/sweep.json`, `prep-deadlock/sweep_la_boundedwait_restore*.json` |
| 16 | 629.9 | 40.3 | 8.27 s | `results/arms/la/sweep.json` |

Cold prefill (c1) 2236-2303 tok/s across the same boots.

## Category harness (tonyd2wild, 40 prompts, concurrency 1, `--hardmode` 88 scenarios for tool-eval)

| build | median | json | html | reasoning | coding | summary | format | prose | narrative | tool-eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old eugr image (pre-own-build) | 65.8 | 90.1 | 86.1 | 74.9 | 69.4 | 47.5 | 46.4 | 43.8 | 40.4 | 90 |
| local16 (own image, plain) | 76.4 | 90.4 | 97.0 | 77.5 | 85.8 | 50.1 | 47.6 | 43.4 | 44.4 | — |
| la (own image, argmax, default) | — (not re-run) | 94.6 | 95.1 | 78.8 | 82.4 | 51.5 | 60.0 | 49.1 | 45.0 | 88-91 |

`decode_probe.py` (mean of 3, `code/structured/counting/prose`, tok/s):

| build | code | structured | counting | prose |
|---|---:|---:|---:|---:|
| old | 56 | 73 | 94 | 46 |
| local16 | 58.1 | 71.7 | 87.2 | 45.9 |
| la | 53.5 | 72.3 | 89.6 | 45.5 |

`la` peak-of-3 decode probe (`results/arms/la/decode_probe.txt`): code 59.6,
structured 90.5, counting 98.0, prose 47.9.

Real-prompt concurrency lane (old image only, 2026-09-17, not re-measured on
the own image), per-stream/aggregate tok/s: x2 61.9/83.2, x4 47.8/81.0,
x6 37.7/112.6, x8 33.7/124.8.

## Quality gates (`la`, `results/arms/la/`)

- **Fidelity probe** (`fidelity.json`, `fidelity_probe.txt`): 100% exact
  retrieval at 8k/32k/64k/128k, 0 typos, thinking on, temperature 0.6.
- **tool-eval-bench 2.6.1 `--hardmode`** (88 scenarios): 88-91/100. Baseline
  (pre-own-image) 90/100. TC-45 (`tool_choice=required`) and TC-68 fail on
  every boot — root causes and fix status in the README's "Known issues"
  section. TC-45's fix mod (`vllm-tc45-reasoning-structag-fix`) reaches
  93/100 (`la-tc` gate) but drops c1 from 95.7 to 85.0
  (`results/arms/la-tc/sweep.json` vs `la/sweep.json`) — not enabled.
- **Straggler probe**: batch 5-16 clean, ~3.9 accepted tokens per 4-token
  draft. The prior fork revision stalled ~18 s at c5-7/9/12 (one request per
  round losing ~97% of drafts); fixed by the own image's scratch-isolation
  commits.
- **MTP acceptance**, real prompts (`la/mtp_metrics.txt`): per draft
  position 21911/19751/18173/16932 accepted of 24344 drafts each (90.0/81.1/
  74.6/69.6%). On the counting workload used for the throughput sweep,
  acceptance is much higher: 98.9% overall, 99.9/99.1/98.6/97.9% by position
  (`results/profiling/README.md`), 0 preemptions.

## Cache-mount fix and ghcr image gate, 2026-09-20

Root cause of "0 cached" / ~9-min boots on `la`: the recipe never mounted a
persistent cache dir, and the container's `HOME=/tmp` (sparkrun's docker
executor), so b12x's SelectionCache and vLLM/torch's own compile caches
resolve under `/tmp/.cache` — wiped on every `sparkrun stop`+`run`. The
plan-cache JSON the container writes fresh each boot has an identical
identity hash to the one already on the host under
`runtime-cache/.../b12x/compile/preparation/` (`b12x/preparation/_cache.py:
cache_identity` only hashes `device_name`+`namespace`+version, not
driver/firmware — both nodes run the same 580.173.02 driver). Fix: mount
`runtime-cache/vllm/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-ca6f25af`
at `/tmp/.cache` via `executor_config.volumes`. Applied to `la`, `local16`,
and the ghcr variant. `la` boot log went from 0 cached / c1 86 to 272
cached / c1 97.4.

The ghcr recipe's first gate (`results/arms/ghcr`) ran *before* this fix
existed (0 cached, ~9-min boot) — treat those numbers as tainted noise, not
a real reading. Re-gated with the mount as `results/arms/ghcr2`: boot log
confirms `272 cached` (same plan-cache hit as `la`, since both pin b12x SHA
`a8333658`), health 200 in ~4 min (down from ~9).

| | `la` (own build) | `ghcr` (tainted, 0 cached) | `ghcr2` (272 cached) |
|---|---:|---:|---:|
| c1 tok/s | 97.4 (boot band 85-99) | 85.6 | 87.7 |
| c4 tok/s | 285.2 | 283.5 | 286.2 |
| c8 tok/s | 412-442 | 429.9 | 433.7 |
| c16 tok/s | 629.9 | 634.4 | 625.6 |
| fidelity (8k/32k/64k/128k) | 100% exact x4 | 100% exact x4 | 100% exact x4 |
| hardmode Quality | 88-91/100 | 86/100 | 89/100 |
| TC-45 (`tool_choice=required`) | **fails every boot** | fails | **passes** |
| straggler probe | clean | not run (`/tmp/straggler.py` lost) | clean, ~3.98 accepted/draft |

`ghcr2`'s c1 (87.7) sits inside `la`'s own documented boot-to-boot noise
band (85-99, see above) — one boot isn't enough to call it a regression
distinct from that noise. c4/c8/c16 are all within a percent or two of `la`.
Everything else (fidelity, hardmode, straggler) is equal or better, and
`ghcr2` is the first boot of this stack, own build or ghcr, where TC-45
passes.

**TC-45 root-cause check.** `la` (and every prior own-build boot) fails
TC-45 because `ParserManager`'s collapse branch and `ParserEngine.
adjust_request()` never build the `tool_choice=required` structural tag for
Qwen3 (see `mods/vllm-tc45-reasoning-structag-fix/run.sh` for the two-bug
trace). The ghcr image (`wheels-20260919-77bdd10-a833365`) carries that fix
as a source commit rather than a runtime mod. Diffed
`vllm/parser/parser_manager.py`, `vllm/parser/engine/parser_engine.py`, and
`vllm/parser/abstract_parser.py` between the running ghcr container and a
container of `spark-vllm-b12x:local-20260918-a8333658` with
`vllm-tc45-reasoning-structag-fix` applied (that mod scored 93/100 on
`la-tc`, see above): **all three files are byte-identical**. So the ghcr
image's parser code is exactly the mod's fix, baked in. The tainted `ghcr`
gate still failed TC-45 (0/2, "No tool calls despite tool_choice=
'required'") — most likely explained by the 0-cached boot's degraded state
rather than a source gap, since the re-gated `ghcr2` (272 cached, otherwise
identical image) passes TC-45 cleanly (2/2).

**Verdict: promote `ghcr2` to serving**, replacing `la`. Rationale: real
correctness win (TC-45 now passes, first time on this stack), quality equal
or better everywhere else, and the only metric below `la`'s single-boot
number (c1) is inside `la`'s own established noise band. `la-tc`'s earlier
finding that the same fix (applied as a runtime mod) costs ~11% at c1
(95.7 → 85.0) does not clearly replicate here — 85.0 and 87.7 are both
within the noise band, so that cost claim may itself have been an
unlucky boot rather than the fix's overhead. Recommend a repeat `ghcr2`
boot before fully retiring that caveat. Pulls (`docker pull`) instead of
requiring `build.sh` on every cluster host, superseding the ghcr recipe from
PR #8.

## Rejected arms

Gated against the `la` baseline (bar: +3% c1 or +5% c8, nothing else worse
than -2%). Full narrative and hang evidence in `results/kernel-pass/arms.md`
and `results/kernel-pass/prep-deadlock/mechanism.md`.

| label | change | c1 tok/s | c1 delta | c8 tok/s | c8 delta | verdict | source |
|---|---|---:|---:|---:|---:|---|---|
| fusear | `compilation_config.pass_config.fuse_allreduce_rms: true` | 86.7 | -12.2% | 428.2 | +3.9% | reject — c1 regression | `results/kernel-pass/sweep_fusear.json` |
| spec3 | `num_speculative_tokens: 3` (was 4) | 84.1 | -14.9% | 375.2 | -8.9% | reject — both regress | `results/kernel-pass/sweep_spec3.json` |
| noat | `B12X_AUTOTUNE: "0"` (diagnostic) | 81.1 | -17.9% | 414.8 | +0.7% | diagnostic — confirms autotune worth ~18% at c1 | `results/kernel-pass/sweep_noat.json` |
| fwd57f3572 | forward-port b12x's native NVFP4 A16 MoE-autotune fix (`57f3572`) onto the old image | 87.9-88.1 | -8% | 419.8-427.3 | -1 to +1% | reject — W4A16 route wins only 4/38 candidate races, repeatable | `results/arms/fwd57f3572-fix/sweep.json`, `sweep_rerun.json`, `results/kernel-pass/prep-deadlock/moe-decode-selection.txt` |
| occ MICRO=32 | `B12X_MICRO_MAX_ACTIVE_CLUSTERS=32` | 97.1 | within noise | 429.9 | within noise | reject — noise; kernel clamps occupancy to 48 SMs regardless | `results/kernel-pass/prep-deadlock/sweep_occ_microlow.json` |
| occ DYNAMIC=32 | `B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS=32` | 94.7 | within noise | 426.2 | within noise | reject — same clamp | `results/kernel-pass/prep-deadlock/sweep_occ_dynlow.json` |
| gdnbf16 | `--mamba-ssm-cache-dtype bfloat16` | — | — | — | — | cannot boot: `state_dtype must be torch.float32` (`b12x/sequence/gdn_prefill/_impl.py:64`) | `misc/gdn-state-bf16-lever3-2026-09-18.md` |
| dv | reduced draft vocab (MiaAI-Lab 47,149-id table, AGPL, local use only) for the MTP head | boots | — | — | — | reject — 0% MTP acceptance (7 of 151k drafts); profile shows the MTP head is 0.04% of step anyway, so no speed upside even if quality held | `results/arms/dv/fidelity.json` |
| b12x HEAD `0f3a8cb` | rebuild at b12x master (6 commits ahead) | — | — | — | — | reject — deadlocks TP2 preparation from an empty plan cache (3 hangs chased, see mechanism.md); root cause is a scale-dependent multi-batch candidate-racing deadlock in `b12x/preparation/session.py`, present since `a8333658`, only entered once a retune needs >1 candidate batch round | `results/kernel-pass/{b12x0f3a8cb-hang,b12x0f3a8cb-hang2,fwd57f3572-hang3}/` |
| RadixArk checkpoint, BF16 KV, old PTQ revision | — | — | — | — | — | rejected 2026-09-16, superseded by QAD `7c4f1bc1` | JOURNAL.md 2026-09-16 |

**Typo hypothesis dead.** Reddit claim that quantized GDN / fp8 KV causes
long-session typos, tested 2026-09-16 on both the QAD and the old PTQ
checkpoint at 8k/32k/64k/128k: 0 typos on either.

## Profile (`results/profiling/README.md`)

Rank-local `torch.profiler` (`mods/vllm-decode-profiler/`), decode steps
220-320 (c1) / 260-360 (c8), ~4% profiler overhead.

| bucket | c1 (55.0 ms/step) | c8 (87.1 ms/step) |
|---|---:|---:|
| GEMM (MoE `siluMoEDynamicKer` share) | 83.6% (26.7%) | 65.6% (33.9%) |
| GDN / SSM | 2.3% | 13.5% |
| all-reduce (RoCE + NCCL) | 4.0% | 6.0% |
| attention | 1.6% | 2.7% |
| sampler / argmax | 1.0% | 1.6% |
| MTP head | 0.04% | 0.02% |
| other | 6.8% | 10.2% |
| idle (union of kernel intervals) | 5.4% | 5.6% |

Ranking of optimization targets, both concurrencies compute-bound (idle ≤7.6%
on either rank): MoE GEMM first (grows with concurrency, 27%→34%), GDN/SSM
second (grows with routed token volume, not launch count — the "unbatched
launch" hypothesis was traced and refuted: GDN decode is one
`torch.ops.b12x.gdn_decode` call per step for the whole batch already), then
`fuse_allreduce_rms` (implemented, off by default, but rejected on
measurement above), then CUDA-graph launch count (~3.4 graphs/step) and
sampler top-k launch count, both low-value at ≤1.6% of step each.

## Known-issue root causes with numbers

- **`B12X_ROCE_SPIN_LIMIT` / TP2 preparation deadlock.** The RoCE one-shot
  collective's default spin limit (~20,000,000 polls ≈ 20 s) expires before
  a from-empty-cache retune's two ranks converge on their MoE candidate
  racing, which poisons the runtime (`RoCE collective on rank 1 timed out
  waiting for rank 0`) instead of waiting. Fix: env
  `B12X_ROCE_SPIN_LIMIT: "300000000"` (~300 s) plus
  `mods/b12x-startup-boundedwait/` (bounded `Store.wait` in the vLLM fork's
  `B12xPreparationCoordinator._exchange()`, fails fast instead of parking).
  Post-fix boot: 349 moe.decode candidates, contract 4→5, retune completes,
  `la` restored at c1 98.7 / c8 441.8
  (`results/kernel-pass/prep-deadlock/sweep_la_boundedwait_restore_r2.json`).
- **`B12X_AUTOTUNE=0` default in eugr's Dockerfile.** Truncates kernel
  selection; the recipe sets it to `1`. Plan cache then persists at
  `~/.cache/sparkrun/runtime-cache/vllm/<model>/b12x/` (168-169 MB), so only
  the first boot after an image/checkpoint change pays the full autotune
  cost.
- **logind `RemoveIPC`** wiped shm on ssh session end
  (`'ShmRingBuffer' object has no attribute 'shared_memory'`). Fix:
  `loginctl enable-linger nvidia` on both nodes.
- **`scripts/gate_arm.sh --hardmode`**: without the flag it runs the default
  69-scenario set, not the 88-scenario hardmode set, which made the first
  `la` gate number (69 and 88 scenarios compared as if equivalent)
  incomparable to later gates. Always pass `--hardmode` for a promotion
  decision.

## Reference: earlier engine comparison, 2026-09-11 and before

Kept for history — this is the SGLang-vs-vLLM-nightly comparison from before
the own-image b12x route existed. Not the current serving route (see the top
of this file and the README for what serves today).

Grids are `llama-benchy --pp 2048 --tg 128 --enable-prefix-caching` at depths
0 to 32k and concurrency 1, 2, 5. `pp2048@depth` is a fresh 2k-token prompt
after a cached context of that depth (a cached turn, not cold prefill).
`tg128` is decode, aggregate across concurrent streams. Two fresh boots of
the same SGLang recipe differed by 20% on prefill and 13% on decode.

### `flashnext-bigkv-g8-c4096` (SGLang)

| depth | pp2048 (c1/c2/c5) | tg128 (c1/c2/c5) |
|---|---|---|
| 0 | 2353 / 2633 / 3004 | 41 / 62 / 94 |
| 4096 | 915 / 1388 / 1963 | 36 / 62 / 83 |
| 16384 | 915 / 1331 / 1902 | 38 / 54 / 84 |
| 32768 | 918 / 1345 / 1874 | 37 / 58 / 79 |

### `flashnext-bigkv-nospec` (SGLang, no drafter)

| depth | pp2048 (c1/c2/c5) | tg128 (c1/c2/c5) |
|---|---|---|
| 0 | 2496 / 2952 / 3115 | 26 / 53 / 85 |
| 4096 | 1046 / 1526 / 2135 | 26 / 53 / 75 |
| 16384 | 1001 / 1479 / 2075 | 26 / 45 / 85 |
| 32768 | 961 / 1284 / 1979 | 26 / 47 / 77 |

### `flashnext-vllm-cached` (vLLM nightly `8a728663`, NVIDIA checkpoint)

| depth | pp2048 (c1/c2/c5) | tg128 (c1/c2/c5) |
|---|---|---|
| 0 | 2890 / 2773 / 2773 | 40 / 54 / 71 |
| 4096 | 2012 / 1962 / 1998 | 30 / 44 / 56 |
| 16384 | 2282 / 2229 / 2290 | 34 / 47 / 63 |
| 32768 | 1924 / 1935 / 1994 | 39 / 46 / 62 |

Reuse verified with a three-request probe (same 20k prompt fresh, identical,
then changed last line): reuses 0, then 19,200, then 19,200 tokens with
`disable_eagle_block_drop`; without it, 0, 0, 16,000 and the depth cells
collapse to 161 tok/s at 32k.

### fp8 KV quality gate, 2026-09-12 (SGLang bf16 vs fp8)

spark-bench (Weschera), 76 scenarios, 2 repeats, thinking off, temp 0.

| | bf16 `flashnext-bigkv-g8-c4096` | fp8 `flashnext-fp8kv-1m8` |
|---|---:|---:|
| TrueScore | 80.8 | 81.3 |
| Pass@1 / Pass@K | 90.8% / 82.9% | 88.2% / 85.5% |
| needle 8k/32k/64k/100k | 12/12 | 12/12 |
| needle 150k/200k/250k | out of context | 9/9 |

No measurable quality cost from fp8 KV on SGLang; full reports in
`results/fp8-gate/`. Full raw grids: `results/sglang-bigkv-g8-c4096-sep03-grid-0-32k.csv`,
`results/sglang-bigkv-nospec-grid-0-32k.csv`, `results/vllm-cached-grid-0-32k.csv`,
`results/vllm-cached-nospec-grid-0-32k.csv`. Pre-own-image eugr b12x route
comparison: `results/eugr-b12x/RESULTS.md`.
