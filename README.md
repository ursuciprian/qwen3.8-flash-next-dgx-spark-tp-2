# Qwen3.8-Flash-Next NVFP4 on two DGX Sparks

Serving recipes for `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` at tensor
parallel 2 across two NVIDIA DGX Spark (GB10, SM121) over ConnectX-7 RoCE,
packaged for [sparkrun](https://sparkrun.dev). The served route is vLLM on a
b12x-kernel fork, built as our own container image. Numbers below come from
fresh boots measured 2026-09-18 to 2026-09-22; each one states its workload
and source file.

## Quick start

```sh
sparkrun registry add https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2
sparkrun run recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml --cluster <your-cluster> --tp 2 --trust
```

`--trust` accepts the mod hook that patches files inside the container before
serve (see [Why the mods](#why-the-mods)). Cold boot 20-30 minutes (kernel
autotune from an empty plan cache), warm boot with a populated
`~/.cache/sparkrun/runtime-cache/vllm/<model>/b12x/` a few minutes. The server
answers on port 8000 with the OpenAI API, model name `qwen3.8-flash-next`.

## Which recipe

| Recipe | Image | Status |
|---|---|---|
| `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml` | `spark-vllm-b12x:local-20260918-a8333658` | **Default, serving.** Probabilistic MTP draft sampling (`draft_sample_method: probabilistic`) in the speculative config, startup robustness mod, `B12X_AUTOTUNE=1`. |
| `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-argmax-drafts.yaml` | same image | Fallback: previous default — one-hot drafts + `use_local_argmax_reduction: true` instead of probabilistic sampling. |
| `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-local-build-seqs-16.yaml` | same image | Fallback: same recipe without `use_local_argmax_reduction`. |
| `recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-ghcr-image.yaml` | `ghcr.io/ursuciprian/spark-vllm-b12x:wheels-20260919-77bdd10-a833365` (`sha256:c0314d7c…`) | Gated (`ghcr2`, 272 cached): TC-45 passes for the first time on this stack, but bench_sweep counting diagnostic c1 is ~-10% vs `la` (87.7 vs 97.4). Not promoted — see [Known issues / fixes](#known-issues--fixes) and `results/README.md`. |

Every recipe in `recipes/qwen3.8-flash-next/`, including the rejected/experimental arms, is
tabled with status and a verdict pointer in `recipes/README.md`.

Image identity for the default: eugr `spark-vllm-docker` Dockerfile `798528a2`
+ fork `local-inference-lab/vllm` `dev/jovian-judgement` `8e1f1e58` + b12x
`a8333658`. Checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` QAD
revision `7c4f1bc1`. fp8 KV, MTP width 4, prefix caching on, `max_num_seqs 16`.

## Headline numbers

Every figure in this repo names its workload. Three workloads are used, and
their numbers are not comparable with each other (see
[How to read these numbers](#how-to-read-these-numbers)). "Old la" is the
previous default (one-hot drafts + `use_local_argmax_reduction`, now
`-la-argmax.yaml`); "probabilistic" is the current default
(`draft_sample_method: probabilistic`). Aggregate = generated tokens per
second summed over all concurrent streams; per-stream = one request's rate.

### Agent coding task (primary number)

Our llama-benchy fork ([ursuciprian/llama-benchy](https://github.com/ursuciprian/llama-benchy) `0d4de42`, `--prompt-mode
task --no-force-length`, command shape in `scripts/run_b_arm.sh`, with
`--concurrency 1 4 10 16` and `--temperature 0` for the temp-0 grid): chat-shaped agent
coding turn (short fixed system prompt, file excerpt, coding instruction),
2048 new prompt tokens on top of a cached context of the given depth (prefix
caching on), up to 512 output tokens (stops naturally), thinking on (server
default; reasoning tokens counted), 3 runs per cell. Values are **aggregate
gen tok/s**; at c1 aggregate = per-stream.

| cached depth | conc. | old la, temp 1.0 (default) | old la, temp 0 | probabilistic, temp 1.0 (default) | old la, temp 0.6 |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 46.1 ± 1.5 | 55.3 ± 3.6 | 45.6 ± 13.7 | pending |
| 0 | 4 | 104.4 | 128.1 | 115.2 | pending |
| 0 | 10 | 155.8 | 184.1 | 171.1 | pending |
| 0 | 16 | 186.3 | 217.7 | 221.9 | pending |
| 16k | 1 | 41.2 | 55.3 | 55.4 | pending |
| 16k | 16 | 120.3 | 134.2 | 115.4 ± 24.3 | pending |
| 64k | 1 | 42.4 | 55.9 | 58.4 | pending |
| 64k | 16 | 100.0 | 111.4 | 106.8 | pending |
| source | | `results/benchy/la-task16.md` | `results/benchy/la-task16-t0.md` | `results/benchy/la-mtpprob-task16.md` | `la-task16-t06` still running on dgx-01 |

Temperature 1.0 is the checkpoint default (1.0 / top-p 0.95 / top-k 20),
i.e. what a client that sends no temperature gets. Probabilistic at temp 0 was
not measured on this workload.

## Default recipe (la, probabilistic MTP drafts) — measured numbers

All three tables are from the current default recipe
(`qwen3.8-flash-next-nvfp4-tp2.yaml`, `draft_sample_method: probabilistic`),
measured 2026-09-21/22. ± is the spread across runs as llama-benchy reports it.

**Agent coding.** `results/benchy/la-mtpprob-task16.md`: llama-benchy fork
`--prompt-mode task --no-force-length`, 2048 new prompt tokens on a cached
context of the given depth, up to 512 output tokens, thinking on, default
temperature (1.0 / top-p 0.95 / top-k 20), prefix caching, 3 runs per cell.

| cached depth | conc. | gen agg tok/s | gen per-stream tok/s | prompt tok/s (agg) | TTFT (ms) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 45.57 ± 13.67 | 45.57 ± 13.67 | 3010.03 ± 114.52 | 698.46 ± 27.32 |
| 0 | 4 | 115.24 ± 22.77 | 30.68 ± 6.80 | 3230.00 ± 77.69 | 2027.22 ± 539.05 |
| 0 | 10 | 171.11 ± 19.49 | 19.69 ± 2.97 | 3196.54 ± 38.95 | 4058.79 ± 1753.61 |
| 0 | 16 | 221.91 ± 2.05 | 17.09 ± 1.59 | 3280.83 ± 7.19 | 5968.70 ± 2934.10 |
| 16k | 1 | 55.42 ± 3.20 | 55.42 ± 3.20 | 779.44 ± 4.34 | 2630.36 ± 14.96 |
| 16k | 4 | 102.47 ± 1.81 | 32.45 ± 3.57 | 852.46 ± 1.39 | 7588.48 ± 2095.66 |
| 16k | 10 | 122.32 ± 0.58 | 18.01 ± 3.77 | 861.12 ± 0.48 | 15065.82 ± 6497.38 |
| 16k | 16 | 115.40 ± 24.27 | 10.97 ± 4.01 | 814.81 ± 56.80 | 23850.35 ± 11577.43 |
| 64k | 1 | 58.42 ± 3.45 | 58.42 ± 3.45 | 570.51 ± 5.98 | 3596.16 ± 39.15 |
| 64k | 4 | 91.66 ± 0.93 | 30.10 ± 4.46 | 635.86 ± 1.16 | 10129.08 ± 2788.80 |
| 64k | 10 | 93.70 ± 12.18 | 14.70 ± 4.22 | 602.19 ± 61.42 | 21165.63 ± 10081.95 |
| 64k | 16 | 106.82 ± 5.89 | 10.89 ± 3.54 | 644.61 ± 1.14 | 29218.89 ± 14327.02 |

Temperature-0 and temperature-0.6 agent-coding grids exist only for the old
argmax config so far (`results/benchy/la-task16-t0.md`: c1 55.3, c16 217.7;
temp 0.6 still running) and are pending for this recipe.

**Prose continuation.** `results/benchy/la-mtpprob-prose16.md`: llama-benchy
0.4.0 default book corpus, 2048 new prompt tokens on a cached context of the
given depth, 128 output tokens, default temperature (1.0), thinking not set
(server default), prefix caching, 2 runs per cell.

| cached depth | conc. | gen agg tok/s | gen per-stream tok/s | prompt tok/s (agg) | TTFT (ms) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 56.05 ± 2.56 | 56.05 ± 2.56 | 3012.77 ± 65.69 | 683.13 ± 14.84 |
| 0 | 4 | 100.29 ± 14.00 | 29.90 ± 3.45 | 3279.76 ± 87.54 | 1972.35 ± 596.46 |
| 0 | 10 | 116.86 ± 5.94 | 17.19 ± 3.71 | 3296.24 ± 36.55 | 3902.41 ± 1719.24 |
| 0 | 16 | 126.72 ± 1.56 | 12.86 ± 3.85 | 3284.74 ± 32.87 | 5802.02 ± 2824.02 |
| 16k | 1 | 58.64 ± 7.35 | 58.64 ± 7.35 | 805.47 ± 1.96 | 2545.33 ± 6.20 |
| 16k | 4 | 55.50 ± 0.15 | 25.76 ± 7.54 | 859.38 ± 0.89 | 7531.04 ± 2086.44 |
| 16k | 10 | 50.49 ± 0.05 | 11.95 ± 6.10 | 866.41 ± 0.05 | 14976.77 ± 6459.43 |
| 16k | 16 | 49.50 ± 0.97 | 8.17 ± 5.45 | 861.38 ± 0.36 | 21843.41 ± 10786.61 |
| 64k | 1 | 51.63 ± 4.85 | 51.63 ± 4.85 | 584.03 ± 0.76 | 3509.35 ± 4.56 |
| 64k | 4 | 44.63 ± 0.80 | 22.72 ± 7.81 | 640.82 ± 3.27 | 10052.57 ± 2772.32 |
| 64k | 10 | 38.39 ± 1.41 | 10.19 ± 6.35 | 654.04 ± 0.52 | 19792.30 ± 8539.95 |
| 64k | 16 | 36.78 ± 2.68 | 6.85 ± 5.43 | 619.90 ± 30.22 | 30082.58 ± 15450.70 |

**Counting ceiling (not user throughput).** `results/arms/la-mtpprob/decode_c*.json`
and `sweep.json` (dgx-01): `tools/tony-bench/bench_sweep.py`, "List the
numbers from 1 to 300 separated by commas…", temperature 0, thinking off,
non-streaming, fresh context, 320 max tokens, 3 rounds per level.

| conc. | agg tok/s | per-stream tok/s | source |
|---:|---:|---:|---|
| 1 | 100.9 / 102.1 / 101.8 | = aggregate | `decode_c1_run{1,2,3}.json` |
| 1 | 101.6 | 101.7 | `sweep.json` (gate sweep) |
| 4 | 288.3 | 72.7 | `sweep.json` |
| 8 | 439.4 / 434.6 | 56.3 / 55.0 | `decode_c8.json` / `sweep.json` |
| 12 | 545.3 | 46.6 | `decode_c12.json` |
| 16 | 641.3 / 635.0 | 41.2 / 40.9 | `decode_c16_run{1,2}.json` |

The gate's own c16 sweep read 399.5 once (`sweep.json`) and did not reproduce
in the two reruns above.

**Quality.** tool-eval-bench `--hardmode` 86/100, rerun 90/100; fidelity
8k/32k/64k 20/20, 128k 19/20 then 20/20 and 20/20; straggler c5-16 no
one-request stall, 3.96-4.00 accepted/draft. Details in
[Quality gates](#quality-gates).

**Against the old argmax config** (same workloads, tables below): sampled
throughput improves mostly at concurrency and long context (agent coding c16
186.3 -> 221.9, 64k c1 42.4 -> 58.4; prose c1 41.1 -> 56.1). Exceptions:
fresh agent coding c1 is a tie (46.1 vs 45.6 ± 13.7); prose 64k c16
regresses (38.15 -> 36.78); agent coding 16k c16 is lower (120.3 vs
115.4 ± 24.3). Counting ceiling unchanged within noise.

## Old la vs probabilistic, by workload

### Prose continuation (comparison)

llama-benchy 0.4.0 default book corpus (raw text continuation), 2048 new
prompt tokens on a cached context of the given depth, 128 output tokens,
default temperature (1.0), thinking not set (server default), 2 runs per
cell, `--enable-prefix-caching`. Aggregate gen tok/s.

| cached depth | conc. | old la | probabilistic |
|---:|---:|---:|---:|
| 0 | 1 | 41.1 | 56.1 |
| 0 | 16 | 117.8 | 126.7 |
| 16k | 1 | 38.1 | 58.6 |
| 16k | 16 | 43.1 | 49.5 |
| 64k | 1 | 48.9 | 51.6 |
| 64k | 16 | 38.15 | 36.78 (regression) |
| source | | `results/benchy/la-prose16.md` | `results/benchy/la-mtpprob-prose16.md` |

### Time to first token

From the agent coding task files above (end-to-end TTFT for the 2048 new
prompt tokens; at c>1 all requests arrive at once, so TTFT includes queueing
behind the other prefills). Old la (`la-task16.md`) / probabilistic
(`la-mtpprob-task16.md`):

| cached depth | c1 | c10 |
|---:|---:|---:|
| 0 | 0.69 / 0.70 s | 4.07 / 4.06 s |
| 16k | 2.60 / 2.63 s | 15.2 / 15.1 s |
| 64k | 3.48 / 3.60 s | 20.1 / 21.2 s |

Multi-turn continuation (`scripts/multiturn_ttft.py`, old la, c1, temp 0,
turn 1 = depth-token prose context + one question, 128 output tokens; turn 2 =
turn 1 + its answer + ~200 new tokens; median of 3,
`results/benchy/la-multiturn.txt`): turn-2 TTFT 1.89 s at 16k (turn 1
5.39 s), 2.92 s at 64k (turn 1 23.3 s).

### Speculative-decoding ceiling (not user throughput)

`tools/tony-bench/bench_sweep.py` (tonyd2wild): prompt "List the numbers from
1 to 300 separated by commas. Output only the numbers, nothing else, no
commentary.", temperature 0, thinking off, non-streaming, 320 max tokens
(~300 generated), fresh context, 3 rounds per level. This is a counting task
the MTP drafter predicts almost perfectly (~3.96-4.00 of 4 drafts accepted,
`results/arms/*/straggler.log`), so it measures how fast the stack can go when
speculation never misses. It is a regression diagnostic, **not** the speed of
coding, chat or agent work. Raw files are on dgx-01 (`results/arms/` is not
mirrored here, see `results/README.md`).

| conc. | old la, agg (per-stream) | probabilistic, agg (per-stream) |
|---:|---:|---:|
| 1 | 99.5 median / 101.8 max of 6 sweeps (boot band 85-102) | 100.9-102.1 (3 runs), sweep 101.6 |
| 8 | 440.7-454.7 (56.2-57.9) | 434.6-439.4 (55.0-56.3) |
| 12 | 547.6 (46.8) | 545.3 (46.6) |
| 16 | 645.1 (41.4); 641.6-645.9 on 2026-09-22 | 635.0-641.3 (40.9-41.2) |
| source | `results/arms/la-lmq/verdict.md`, `results/arms/la/decode_c{12,16}*.json` | `results/arms/la-mtpprob/decode_c*.json`, `sweep.json` |

c1 is bimodal boot to boot (fast mode 94-102, slow mode 84-88,
`results/arms/c1-decline/verdict.md`): report the median of >=5 sweeps and
the max, not one run. One probabilistic c16 sweep read 399.5
(`la-mtpprob/sweep.json`) and did not reproduce in two reruns.

### How to read these numbers

- **Draft acceptance.** MTP proposes 4 tokens per step. On the counting
  prompt nearly all 4 are accepted (98.9% overall, ~3.96 per step,
  `results/profiling/README.md`). On the sampled (temp 1.0) prose and agent
  task grids with probabilistic drafts, 39-41% are accepted, ~1.6 per step
  (`results/arms/la-mtpprob/mtp_{before,after}_{prose,task}.txt` on dgx-01:
  prose 10523/26872, task 87368/213924). The same engine therefore shows
  ~100 tok/s on counting and ~45-58 on real single-stream work.
- **Temperature.** At temp 0 the target and draft agree more often, so
  acceptance and throughput rise (task c1 55.3 at temp 0 vs 46.1 at 1.0,
  old la). Clients that send no temperature get 1.0.
- **Thinking tokens.** The task and prose runs have thinking on; reasoning
  tokens are generated and counted like answer tokens. The counting
  diagnostic and the category harness run with thinking off.
- **Cached-context prefill.** Depth runs prefill 2048 new tokens on top of a
  cached context; attention and GDN cost grow with depth, so both TTFT and
  per-token decode slow down, and concurrency at depth queues prefills.

### Other single-stream probes

`scripts/decode_probe.py` (streaming, temp 0, thinking not set (server
default), 512 max tokens, fresh context, c1, decode rate excludes TTFT;
prompts: code = "Write a complete Python implementation of a red-black
tree…", structured = 40-object JSON array, counting = 1 to 200, prose =
reflective essay). Old la + MXFP8 lm_head, peak of 3
(`results/arms/la-lmq/decode_probe.txt`): code 69.0 (mean 57.1), structured
86.4, counting 100.3, prose 50.2 tok/s. Mean of 3 across builds: old eugr
image 56/73/94/46, local16 58.1/71.7/87.2/45.9, la 53.5/72.3/89.6/45.5
(code/structured/counting/prose, `results/RESULTS.md`).

Category harness (tonyd2wild `tools/tony-bench/bench_categories.py`, 40 real
prompts, 5 per category, temp 0, thinking off, streaming, max 900 tokens,
fresh context, c1, per-stream decode tok/s excluding TTFT; the coding column
is its 5 coding prompts with hidden tests, not bench_sweep). Old = eugr's
nightly image before this repo's own build; local16 = own image, plain
recipe; la = own image with `use_local_argmax_reduction`. Source
`results/RESULTS.md`.

| | median | json | html | reasoning | coding | summary | format | prose | narrative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old eugr image | 65.8 | 90.1 | 86.1 | 74.9 | 69.4 | 47.5 | 46.4 | 43.8 | 40.4 |
| local16 (own image) | 76.4 | 90.4 | 97.0 | 77.5 | 85.8 | 50.1 | 47.6 | 43.4 | 44.4 |
| la (own image, argmax) | not re-run on this harness | 94.6 | 95.1 | 78.8 | 82.4 | 51.5 | 60.0 | 49.1 | 45.0 |

Real-prompt concurrency lane (prompt set, temperature and thinking setting
not recorded; 2026-09-17 journal entry only), per-stream/aggregate tok/s, old
image only (not re-measured on the own image, `results/RESULTS.md`): x2 61.9/83.2, x4
47.8/81.0, x6 37.7/112.6, x8 33.7/124.8.

## Quality gates

Default recipe (probabilistic drafts), `results/arms/la-mtpprob/`:

- **tool-eval-bench 2.6.1 `--hardmode`** (88 scenarios, thinking on): 86/100
  (`hardmode.log`), rerun 90/100 (`hardmode2.log`); historical band across
  la-family arms 86-93.
- **Fidelity probe** (`fidelity_probe.txt`, `fidelity_128k_rerun{1,2}.json`;
  20 tool-call retrievals per depth, thinking on, temperature 0.6): 20/20 at
  8k/32k/64k; 128k 19/20, then 20/20 and 20/20 on two reruns; 0 typos.
- **Straggler probe** (`straggler.log`, bench_sweep counting prompt, batches
  5/6/7/8/12/16): no preemptions, 3.96-4.00 accepted per draft, no
  one-request stall. The c6 round took 13.7 s with all six requests equally
  slow (other rounds 4.7-7.5 s).

Previous default (old la + MXFP8 lm_head), `results/arms/la-lmq/`:

- **Fidelity probe** (`fidelity.json`, `fidelity_probe.txt`): 20/20 exact
  retrieval at 8k/32k/64k/128k, 0 typos, thinking on.
- **tool-eval-bench `--hardmode`**: 89/100 (la band across arms: 86-93/100).
- **Straggler probe**: batch 5-16 clean, 3.96-3.99 accepted/draft on the
  counting prompt (was one request per round stalling ~18 s at c5-7/9/12 on
  the prior fork revision; fixed by the own image).

Two hardmode scenarios fail on every la-family boot:

- **TC-45** (`tool_choice=required`): parser bug, see
  [Known issues / fixes](#known-issues--fixes). Fixed by
  `mods/vllm-tc45-reasoning-structag-fix/` → 93/100, but that mod costs
  about -12% on the bench_sweep counting diagnostic at c1
  (`results/arms/la-tc/sweep.json`, 85.0 vs 95.7). Not enabled by default.
- **TC-68**: model wraps JSON in a code fence with commentary; the scenario
  intentionally sends no `response_format`, so this is model compliance, not
  a server bug — not fixable without defeating the test.

**MTP acceptance** on real prompts (old la), per draft position:
~90/81/75/70% (`results/arms/la/mtp_metrics.txt`: overall 76767/24344
draft-tokens*4 positions, per-position 21911/19751/18173/16932 accepted). On
the bench_sweep counting prompt acceptance is far higher — 98.9% overall,
99.9/99.1/98.6/97.9% by position (`results/profiling/README.md`).

## Profile (rank-local `torch.profiler`, `results/profiling/README.md`)

Mod `mods/vllm-decode-profiler/`, recipe
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
  our own image. `mods/vllm-qwen-scratch-isolation/` documents the fix.
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
- lm_head online MXFP8 (W8A16) via `VLLM_MXFP8_LM_HEAD=1` — a fork feature
  (`local-inference-lab/vllm` `dev/jovian-judgement` `8e1f1e58`,
  `_supports_default_lm_head_quantization`), enabled here. Halves the bytes
  read for the verify-head lm_head matmul (previously BF16-only on b12x at
  M=1, cuBLAS fallback); see `results/arms/la-lmq/verdict.md`.

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
- The recipe that serves this image, `qwen3.8-flash-next-nvfp4-tp2-ghcr-image.yaml`,
  has been pulled and gated on the nodes (`ghcr2`, cache-mount fix applied):
  TC-45 passes for the first time on this stack, but bench_sweep counting
  c1 is ~-10% vs `la`. Kept as an alternate pull-based path, not promoted to default — see
  `results/README.md`.

## Repo map

| Path | What |
|---|---|
| `recipes/qwen3.8-flash-next/` | The served recipe family. `qwen3.8-flash-next-nvfp4-tp2.yaml` (default), `-local16.yaml` (fallback), `-la-ghcr.yaml` (alternate pull-based image, gated but not promoted). Full table with status and verdict pointers: `recipes/README.md`. |
| `recipes/arms/`, `recipes/dflash2/`, `recipes/retired/`, `recipes/flashnext-*.yaml` | Earlier SGLang-era recipes and bisection arms, kept for history; not the served route. See `recipes/README.md`. |
| `mods/` | Engine patches, one directory per mod. `mods/README.md` has a one-line summary of every mod, current and SGLang-era. |
| `scripts/` | `gate_arm.sh` (full quality gate, needs `--hardmode`), `fidelity_probe.py`, `prof_summary.py`, `needle_ladder.py`, `decode_probe.py`, `validate_recipes.py`, `run.sh`, and the arm-bisection scripts (`vllm_ladder.sh`, `qwen_ladder*.sh`, …). |
| `results/benchy/` | llama-benchy agent-task and prose grids (`*-task16*.md`, `*-prose16.md`) and the multi-turn TTFT probe output behind the headline tables. |
| `results/kernel-pass/`, `results/profiling/`, `results/arms/` | The 2026-09-18/21 arms, hang evidence, and per-kernel profile behind the tables above. Full index: `results/README.md`. |
| `results/eugr-b12x/`, `results/fp8-gate/`, `results/sglang-*`, `results/vllm-cached-*` | Earlier (pre-09-18) SGLang/vLLM-nightly measurements, kept for history. |
| `misc/` | Working notes not promoted into a result file. |

For every recipe's status (default/fallback/experimental/rejected) and every
mod's one-line summary, see `recipes/README.md` and `mods/README.md`. For
every measurement and verdict this repo has produced, see `results/README.md`.

## Hardware

- 2x DGX Spark: GB10, SM121, 128 GB LPDDR5X unified memory.
- ConnectX-7 RoCE between the nodes.

## Credits

The vLLM route now served by default (`recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml`,
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
  release 0.4.0 measures the prose grids; the `eugr-agents` recipe family
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

## License

Apache-2.0, see `LICENSE`. The vLLM overlays under `mods/` are modified copies
of Apache-2.0 vLLM files and keep their upstream headers. The spark-bench
reports under `results/fp8-gate/` are output of that tool, reproduced with its
run labels intact.
