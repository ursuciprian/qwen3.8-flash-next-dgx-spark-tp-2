# Benchmarks: full tables

Full measurement tables behind the [README](../README.md) headline numbers.
Every figure names its workload and source file. Paths under `results/arms/`
are partly on dgx-01 only; see [results/README.md](../results/README.md).

Contents: [Headline grid](#headline-numbers) ·
[Default recipe tables](#default-recipe-la-probabilistic-mtp-drafts--measured-numbers) ·
[Old la vs probabilistic](#old-la-vs-probabilistic-by-workload) ·
[How to read these numbers](#how-to-read-these-numbers) ·
[Other single-stream probes](#other-single-stream-probes) ·
[MTP acceptance](#mtp-acceptance-per-draft-position)

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
[Quality](../README.md#quality).

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

## MTP acceptance per draft position

**MTP acceptance** on real prompts (old la), per draft position:
~90/81/75/70% (`results/arms/la/mtp_metrics.txt`: overall 76767/24344
draft-tokens*4 positions, per-position 21911/19751/18173/16932 accepted). On
the bench_sweep counting prompt acceptance is far higher — 98.9% overall,
99.9/99.1/98.6/97.9% by position (`results/profiling/README.md`).
