# Results index

Every measurement, arm and verdict in this repo, newest ideas first within
each group. Ground truth for the current serving numbers is the top-level
`README.md` ("Headline numbers" and "Default recipe … measured numbers"),
backed by `results/benchy/*.md` and `results/arms/la-mtpprob/` — this file is
the index, not a second source of numbers.

Workload key used below — every number carries one of these tags:

- **[count]** `tools/tony-bench/bench_sweep.py`: "List the numbers from 1 to
  300…", temp 0, thinking off, non-streaming, 320 max tokens, fresh context,
  3 rounds; aggregate tok/s (at c1 = per-stream). MTP accepts ~4 of 4 drafts
  here, so it is a speculative-decoding ceiling diagnostic, not the speed of
  coding or chat.
- **[task]** llama-benchy fork `--prompt-mode task`: agent coding turn, 2048
  new prompt tokens on a cached context (depth 0/16k/64k), up to 512 output
  tokens, thinking on, prefix caching; aggregate gen tok/s. Temperature
  stated per entry.
- **[prose]** llama-benchy 0.4.0 default book continuation, 2048 new prompt
  tokens, 128 output tokens, default temperature 1.0; aggregate gen tok/s. "File" points at the primary
doc for that entry; most arms also have raw JSON/logs alongside it that
aren't summarized here.

## Current default recipe: la (probabilistic MTP drafts + MXFP8 lm_head)

| date | recipe / arm | key numbers | verdict | file |
|---|---|---|---|---|
| 2026-09-22 | `la-mtpprob` — `draft_sample_method: probabilistic` | [task] temp 1.0: c1 45.6 ± 13.7 (old la 46.1, tie), c16 221.9 (186.3), 64k c1 58.4 (42.4), 16k c16 115.4 ± 24.3 (120.3); [prose]: c1 56.1 (41.1), c16 126.7 (117.8), 64k c16 36.78 (38.15, regression); [count]: c1 100.9-102.1, c8 434.6-439.4, c16 635.0-641.3; hardmode 86 then 90, fidelity 8k-64k 20/20, 128k 19/20 then 20/20 x2, straggler clean | **promoted, now default** (PR #25) | `results/arms/la-mtpprob/verdict.md`, `results/benchy/la-mtpprob-{task16,prose16}.md` vs `la-{task16,prose16}.md` |
| 2026-09-22 | old la (argmax) at temp 0, [task] | c1 55.3, c16 217.7; 16k c1 55.3; 64k c1 55.9; temp 0.6 grid pending | reference for temperature effect | `results/benchy/la-task16-t0.md` |
| 2026-09-21 | `la-lmq` — online MXFP8 verify-head lm_head (`VLLM_MXFP8_LM_HEAD=1`) | [count] c1 99.5 median/101.8 max (was 95.6/99.5), c8 440.7 (was 433.0), hardmode 89/100, fidelity 100%x4, straggler clean | **promoted, now default** | `results/arms/la-lmq/verdict.md` |
| 2026-09-21 | c1 "decline" investigation | [count] 25 runs, bimodal 94-99 (fast) / 84-88 (slow), no monotonic decline; GPU-internal or async-scheduling suspected, not host/thermal/RoCE/CPU-placement | root-caused as measurement noise, not a regression — report median of >=5 + max | `results/arms/c1-decline/verdict.md` |
| 2026-09-20/21 | ghcr / ghcr2 — pull-based image from published wheels | [count] ghcr (0 cached, tainted) c1 85.6; ghcr2 (272 cached) c1 87.7 (~-10% vs la's 97.4), TC-45 passes (first time), quality/fidelity/straggler equal or better | **not promoted** — quality win, throughput regression vs `la`; kept as an alternate pull path (PR #8) | `results/RESULTS.md` §"Cache-mount fix and ghcr image gate" |
| 2026-09-19/20 | cache-mount fix (`executor_config.volumes` -> `/tmp/.cache`) | root cause of "0 cached" boots: `HOME=/tmp` in container, plan cache never persisted; la boot log 0 cached / [count] c1 86 -> 272 cached / c1 97.4 | fixed, applied to `la`/`local16`/ghcr variants | `results/RESULTS.md` §"Cache-mount fix and ghcr image gate" |
| 2026-09-19 | `la-tc` — `vllm-tc45-reasoning-structag-fix` (Phase B) | TC-45 fixed, hardmode 93/100 (was 91), but [count] c1 85.0 vs baseline 95.7 (~-11%) | **reverted** — quality win not worth the throughput cost | `results/arms/tooleval-fix/notes.md` |
| 2026-09-20/21 | `la-tcc` — `vllm-tc45-cheap` (v2, gates on `required`/named only, not `auto`) | mechanism: old fix's grammar gate fired on `tool_choice=auto` too, serializing async sampling every step for any tool-bearing request; new gate narrows to `required`/named | validation recipe written, GPU validation pending — see file for the 6-step promotion checklist | `results/arms/tooleval-fix/cheap-design.md` |
| 2026-09-20 | `la-kk` — KK (karmic-kraken-beta) image, vllm `57a80980` + b12x `e9ce5477` | KK's `ParserManager` never collapses Qwen3 onto a shared engine (design difference from jovian), so the TC-45 bug class doesn't exist there; still needs the `abstract_parser.py` `reasoning=False` + `auto`-gate fixes (`vllm-tc45-cheap` variant `0002`) | in progress, not gated against `la` yet | `mods/vllm-tc45-reasoning-structag-fix/`, `results/arms/tooleval-fix/cheap-design.md` §"Karmic-kraken differs" |

## Rejected arms (screened against `la` on [count], bar +3% c1 / +5% c8, nothing else worse than -2%)

| date | arm | change | result | verdict | file |
|---|---|---|---|---|---|
| 2026-09-19 | fusear | `fuse_allreduce_rms: true` | c1 86.7 (-12.2%), c8 428.2 (+3.9%) | reject — c1 regression | `results/kernel-pass/arms.md`, `results/kernel-pass/sweep_fusear.json` |
| 2026-09-19 | spec3 | `num_speculative_tokens: 3` (was 4) | c1 84.1 (-14.9%), c8 375.2 (-8.9%) | reject — both regress | `results/kernel-pass/arms.md`, `results/kernel-pass/sweep_spec3.json` |
| 2026-09-19 | noat | `B12X_AUTOTUNE: "0"` (diagnostic) | c1 81.1 (-17.9%), c8 414.8 (+0.7%) | diagnostic — confirms autotune worth ~18% at c1, not a promotion candidate | `results/kernel-pass/arms.md`, `results/kernel-pass/sweep_noat.json` |
| 2026-09-19/20 | fwd57f3572 | forward-port b12x's native W4A16 MoE-autotune fix (`57f3572`) onto the old image | c1 87.9-88.1 (-8%), c8 419.8-427.3 | reject — wins only 4/38 candidate races, repeatable regression | `results/arms/fwd57f3572-fix/sweep.json`, `results/kernel-pass/arms.md` |
| 2026-09-19 | occ MICRO=32 / DYNAMIC=32 (`B12X_MICRO_MAX_ACTIVE_CLUSTERS` / `B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS`) | c1/c8 within noise of baseline | reject — noise, kernel clamps occupancy to 48 SMs regardless | `results/kernel-pass/prep-deadlock/sweep_occ_*.json` |
| 2026-09-18 | gdnbf16 | `--mamba-ssm-cache-dtype bfloat16` | cannot boot | reject — b12x requires `state_dtype == torch.float32` | `misc/gdn-state-bf16-lever3-2026-09-18.md` |
| 2026-09-20/21 | dv | reduced draft vocab (MiaAI-Lab 47,149-id table, AGPL, local use only) for the MTP head | boots, 0% MTP acceptance (7 of 151k drafts) | reject — not a speed lever anyway, MTP head is 0.04% of step | `results/arms/dv/fidelity.json`, `mods/vllm-dv-devicefix/` |
| 2026-09-19 | b12x HEAD `0f3a8cb` rebuild | 6 commits ahead of `a8333658` | deadlocks TP2 preparation | reject — scale-dependent multi-batch candidate-racing deadlock in `b12x/preparation/session.py` (hang #1); reverting one commit only moved the hang to a second barrier (hang #2) | `results/kernel-pass/arms.md`, `results/kernel-pass/{b12x0f3a8cb-hang,b12x0f3a8cb-hang2}/SUMMARY.md` |
| 2026-09-19 | fwd57f3572 hang (hang #3) | same MoE-retune-forcing change on the OLD, otherwise-working image | deadlocks at the same barrier | confirms the deadlock is a general scale-dependent bug in b12x's candidate-racing handshake, not a version-skew issue | `results/kernel-pass/fwd57f3572-hang3/SUMMARY.md`, `results/kernel-pass/prep-deadlock/mechanism.md` |
| 2026-09-20/21 | gemv — `vllm-qwen38-bf16-gemv` (b12x `gemm.bf16_gemv` for the verify-head lm_head) | boot failed, `ValueError: candidate races require an activation-producing context` | reject / do not ship — b12x's candidate-racing prep path doesn't support the new target's call shape yet | `results/arms/gemv/verdict.md` |
| 2026-09-21 | gpu_memory_utilization 0.88 / 0.84 (Lever C, concurrency ceiling, `max_num_seqs=24`) | 0.88 crashed 9s into b12x state prep (host-RAM OOM, no CUDA OOM trace); 0.84 aborted live near an earlyoom threshold; 0.80 (default) forced a retune that deadlocked the same way as the other multi-batch cases | reject — structural: DGX Spark's unified memory means `gpu_memory_utilization` doesn't reserve host RAM for b12x's candidate-measurement subprocesses; anything above 0.80 is not viable regardless of `max_num_seqs` | `results/arms/gpu_mem_util_verdict.md`, `results/arms/leverC_verdict.md`, `results/arms/seqs24_mem88_CRASHED.md` |
| 2026-09-16 | RadixArk checkpoint, BF16 KV, old PTQ revision | — | superseded by QAD checkpoint `7c4f1bc1` | rejected | `results/RESULTS.md` |
| 2026-09-16 | typo hypothesis ("quantized GDN / fp8 KV causes long-session typos") | 0 typos at 8k-128k on either checkpoint | dead — does not reproduce on this stack | `results/RESULTS.md`, `results/arms/la-lmq/verdict.md` |

## Quality-gated / open

| arm | what | status | file |
|---|---|---|---|
| `vllm-qwen38-lmhead-b12x` (BF16 verify head via a b12x kernel path) | superseded by `VLLM_MXFP8_LM_HEAD` unless a BF16 head is needed for quality reasons | superseded, kept for reference | `results/kernel-pass/lmhead-hc-design.md` §A |
| `vllm-qwen38-hc-mxfp8` (hyper-connection/router online MXFP8) | patched (`patches/vllm-qwen38-hc-mxfp8.patch`), expected +7.8% from bytes alone, mutually exclusive with the bf16-gemv mod | quality-gated, not yet promoted (raw sweep data on dgx-01 under `results/arms/la-hcq/`, no verdict written yet) | `results/kernel-pass/lmhead-hc-design.md` §B2 |
| HC TP=2 sharding (item B) | best case +2.2%, realistic 0 to negative | not sound, no patch written | `results/kernel-pass/lmhead-hc-design.md` §B |

## Kernel-pass technical deep dives (2026-09-19/21, no GPU runs unless noted)

| doc | subject |
|---|---|
| `results/kernel-pass/survey.md` | upstream commit survey (vLLM fork 0 commits ahead; b12x 6 commits ahead, evaluated one by one) that set up the arms above |
| `results/kernel-pass/arms.md` | the 2026-09-19 arm sweep (fusear/spec3/noat/fwd57f3572/occ), the three b12x-HEAD/forward-port hang investigations, and the GDN-batching lead correction |
| `results/kernel-pass/dense-gemm-fusion.md` | non-MoE decode GEMM inventory from checkpoint headers + a saved decode trace; the "fuse the 5 GDN projections" lead is dead (fork already merges them to 2, and they cost ~0 wall time); real target is the BF16 verify-head/HC GEMV path |
| `results/kernel-pass/lmhead-hc-design.md` | verify-head and hyper-connection weight-stream design note; A (b12x lm_head kernel) not reachable above M=1, A2 (MXFP8 verify head) is what got promoted as `la-lmq`, B (HC TP=2 sharding) not sound, B2 (HC/router MXFP8) quality-gated |
| `results/kernel-pass/prep-deadlock/mechanism.md` | full trace of the TP2-preparation deadlock and the `B12X_ROCE_SPIN_LIMIT` + `b12x-startup-boundedwait` fix |
| `results/kernel-pass/b12x0f3a8cb-hang/`, `b12x0f3a8cb-hang2/`, `fwd57f3572-hang3/` | wchan/nvidia-smi evidence for each of the three hangs above |

## Profiling

| doc | subject |
|---|---|
| `results/profiling/README.md` | rank-local `torch.profiler` decode-step kernel breakdown at c1/c8 on `la` under the [count] prompt: GEMM 83.6%/65.6%, GDN/SSM 2.3%/13.5%, all-reduce 4.0%/6.0%, both concurrencies compute-bound (idle <=7.6%); ranked list of optimization targets |

## Earlier (pre-2026-09-18) engine comparisons — kept for history, not the current serving route

| date | subject | file |
|---|---|---|
| 2026-09-15 | eugr's b12x vLLM route vs the SGLang recipes, first measurements on this checkpoint; agents-ladder MTP/RoCE-threshold arms; LPT4096 exploratory benchmark | `results/eugr-b12x/RESULTS.md` |
| 2026-09-16 | decode-aware prefill scheduling fixes the c5/c9/c12 depth collapse (root cause: `max_parallel_prefills=1` starving decode); promoted into `recipes/eugr/eugr-agents.yaml` | `results/eugr-b12x/RESULTS.md` §"decode-aware prefill scheduling" |
| 2026-09-16 | SGLang b12x GDN kernel port (`mods/sglang-gdn-b12x-decode`) — decode-only kernel matches Triton; decode+verify kernel breaks acceptance in the live server, root cause not found | `results/eugr-b12x/RESULTS.md` §"SGLang b12x GDN kernel port" |
| 2026-09-11 and before | SGLang vs vLLM-nightly comparison (bigkv/nospec/vllm-cached grids), fp8 KV quality gate (spark-bench, 76 scenarios) | `results/RESULTS.md` §"Reference: earlier engine comparison", `results/sglang-bigkv-*.csv`, `results/vllm-cached-*.csv`, `results/fp8-gate/` |

## SGLang-era bisection arms (`recipes/arms/fn-*.yaml`)

A large family of one-flag-at-a-time SGLang bisection recipes and their
`.csv`/`.log`/`.txt` outputs under `results/arms/` on dgx-01 (not mirrored
into this repo — raw data only, no written verdict beyond what's folded into
`results/RESULTS.md`'s SGLang-era section and `results/eugr-b12x/RESULTS.md`).
Not the current serving route; kept on the bench host for reference if the
SGLang path is revisited.

## Raw data not in this repo

`results/arms/` on dgx-01 holds the raw JSON/log/CSV evidence behind every
entry above. This repo's `results/arms/` is gitignored except for the curated
`.md` verdicts listed in the tables — force-added deliberately, one per
finding, never the raw sweep/log/JSON files themselves. If a number above
needs to be re-derived, ssh to dgx-01 and read the file the table cites.
