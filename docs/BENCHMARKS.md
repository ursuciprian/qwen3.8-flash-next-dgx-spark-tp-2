# Benchmarks: full tables

## Single Spark v3a (2026-10-04)

Recipe `qwen3.8-flash-next-1x-dgx-spark` (one GB10, TP=1), image
`ghcr.io/ursuciprian/spark-vllm-b12x:tp1-v3a-20261004-5bf24021-7fa812b3-warm`, checkpoint revision `7c4f1bc1`.
v3a = v2 + `VLLM_PLE_MMAP_KEEPALIVE_MS=50` (keeps the NVMe drive out of its power-saving state between decode steps).
Raw files: [`results/tp1-v3a-20261004/`](../results/tp1-v3a-20261004/).

**Coding**: llama-benchy `--prompt-mode task`, 2048 new prompt tokens, up to 512 out, thinking on, temperature 1.0 /
top-p 0.95 / top-k 20, prefix caching, 3 runs, one boot. Total tok/s, mean ± sd over runs.

| depth | test | c1 | c2 | c4 | c8 |
|---|---|---|---|---|---|
| 0 | pp2048 | 1201 ± 20 | 1583 ± 155 | 1666 ± 96 | 1947 ± 59 |
| 0 | tg512 | 50.0 ± 1.3 | 73.3 ± 7.8 | 103.7 ± 2.4 | 117.6 ± 4.8 |
| 16k | pp2048 | 1010 ± 40 | 1116 ± 30 | 1231 ± 3 | 99 ± 20 |
| 16k | tg512 | 51.2 ± 4.8 | 77.6 ± 10.9 | 102.5 ± 3.4 | 19.7 ± 2.6 |

16k c8: the 6 GiB KV pool fills and requests are deferred (same in v2); a fix is in progress.

**Paired A/B vs v2** (ABBA, 4 passes over 2 Sparks, temperature 0, same prompts per pair).
Change in % with 95% CI; negative step time is faster.

| cell | n | step time | tok/s |
|---|---|---|---|
| fresh c1 | 32 | -0.7 [-1.0, -0.5] | +2.3 [+0.0, +4.1] |
| fresh c2 | 32 | -0.6 [-1.0, -0.1] | +1.5 [-0.6, +3.5] |
| fresh c4 | 48 | -8.4 [-8.8, -8.0] | +7.6 [+5.8, +9.4] |
| fresh c8 | 32 | -5.1 [-5.6, -4.7] | +4.9 [+3.1, +6.8] |
| 16k c1 | 32 | -0.9 [-1.3, -0.5] | +2.3 [+0.2, +4.4] |
| 16k c2 | 32 | -0.6 [-1.1, -0.1] | +2.8 [+1.2, +4.4] |
| 16k c4 | 48 | -8.0 [-8.5, -7.4] | +7.7 [+5.9, +9.4] |
| counting c1-c4 | 32-48 | -0.6 to +0.1, within CI | +0.0 to +0.6, within CI |
| counting c8 | 32 | -3.1 [-3.5, -2.8] | +3.6 [+3.0, +4.2] |

16k c8 (n=12) is not measurable in either arm (KV pool churn). MTP acceptance per draft position is unchanged in every
cell (e.g. fresh c4 0.80/0.63/0.49/0.39 in both).

**Quality gate**: hardmode 92/100, TC-45 100/100, fidelity 20/20 at 8k/32k/64k/128k plus 128k seeds 11 and 13 at 20/20,
batch stragglers c8-c16 with 0 preemptions (3.98-3.99 accepted per draft), min MemAvailable 13.97 GiB.

## b1.4 image (2026-10-01)

The current default build.

Terms used: **concurrency (c)** = requests running at the same time;
**depth** = tokens of earlier conversation already cached before the new prompt; **MTP** (multi-token prediction) = the
model drafts up to 4 next tokens that are checked in one step, which is what makes single-user speed high.

Image `ghcr.io/ursuciprian/spark-vllm-b12x:b1.4-20261001-b7fbaf96-a7e649d8-warm`
(`sha256:3b2f26080addadafe675f31227d6dacec3716064cbc0c7fd376b643bc34183fd`), checkpoint revision `7c4f1bc1`.
A/B against the previous build (2026-09-29) on 2026-10-01, two separate boots per build, means of both boots.
Raw files and verdict: [`results/b1.4-20261001/`](../results/b1.4-20261001/). The 2026-09-29 build's tables follow below.
What changed: the MTP draft head scores 131,072 of the 248,320 vocab ids (lossless via rejection sampling), the vllm#923
QSA prefill-flag race fix, the vllm#914 Triton recompile fix, and the server default thinking effort `medium` (a recipe
flag; the speed and gate numbers below were measured without it, at the template default).

**Coding**: [llama-benchy fork](https://github.com/ursuciprian/llama-benchy) `--prompt-mode task` (agent coding turn,
2048 new prompt tokens, up to 512 out, thinking on, temperature 1.0 / top-p 0.95 / top-k 20, prefix caching, 3 runs per boot).
Total tok/s; `*` = beyond run-to-run noise.

| depth | c1 | c2 | c4 | c5 | c8 | c10 | c16 |
|---|---|---|---|---|---|---|---|
| 0 | 62.2 | 105.1 | 150.9 | 160.6 * | 195.5 | 199.5 | 242.8 |
| 0, previous | 63.9 | 99.1 | 147.0 | 154.1 | 190.3 | 202.0 | 240.9 |
| 16k | 63.8 | 91.9 | 117.5 | 128.1 | 145.9 | 155.0 | 176.4 |
| 16k, previous | 64.8 | 93.0 | 117.3 | 126.0 | 145.3 | 152.6 | 175.8 |

Depth 0: c1 -2.6% (noise 9.6%), c2 +6.0% (noise 6.8%), c4 +2.6%, c5 +4.2%, c8 +2.8%, c10 -1.2%, c16 +0.8%.
16k: c1 -1.5% (noise 11.8%), c2 -1.1% (noise 12.4%), c4 +0.2%, c5 +1.6%, c8 +0.4%, c10 +1.5%, c16 +0.4%.
Beyond noise: d0 c5 up. No cell is worse beyond noise. d0 c1 and 16k c1/c2 split boot 1 vs boot 2 on both builds, so their
noise band is wide; the paired probe below is the low-variance reading for them.

**Paired temperature-0 decode-step probe** (same prompts on both builds, pooled 2 boots, 95% CI):

| cell | step time | tokens/step | tok/s |
|---|---|---|---|
| fresh c1 / c2 / c4 / c8 / c16 | -2.9 / -3.5 / -2.1 (CI incl. 0) / -2.4 / -0.4% | +3.1 / -2.2 / -2.8 / +1.0 / +4.8% | +6.4 / +2.4 / +2.7 / +3.6 / +7.4% |
| 16k c1 / c2 / c4 / c8 / c16 | -3.5 / -3.5 / -1.0 / -3.3 / +1.3% | +4.4 / -1.7 / +5.1 / +2.6 / +1.1% | +9.6 / +2.6 / +6.9 / +7.4 / +1.3% |
| counting c1 / c2 / c4 / c8 / c16 | -3.4 / -4.9 / -2.2 / -1.5 / -0.7% | -0.6 / -0.5 / -0.1 / +0.2 / +0.1% | +2.9 / +4.6 / +1.5 / +2.4 / +0.5% |

MTP acceptance per draft position within ±0.03 everywhere (16k c1 0.71/0.50/0.36/0.26 -> 0.71/0.50/0.35/0.26; fresh c1
0.75/0.55/0.43/0.35 -> 0.75/0.58/0.45/0.37). The step gain is the smaller draft-head GEMM (65,536 instead of 124,160 rows
per rank, four passes per step). Only 16k c16 has a slower step (+1.3%), offset by more tokens per step (tok/s +1.3%, CI incl. 0).

**Counting** (`tools/tony-bench/bench_sweep.py`: "list the numbers from 1 to 300", temperature 0, thinking off, 300 tokens).
Nearly every draft token is accepted, so this is the stack's upper bound, not coding speed. Aggregate tok/s:

| | c1 | c2 | c4 | c5 | c8 | c10 | c16 |
|---|---|---|---|---|---|---|---|
| current | 119.8 * | 214.3 * | 356.6 | 407.7 * | 524.8 | 586.0 | 778.0 |
| previous | 114.2 | 207.8 | 349.8 | 390.4 | 517.8 | 578.5 | 764.8 |

c1 +4.9%, c2 +3.1%, c4 +1.9%, c5 +4.4%, c8 +1.4%, c10 +1.3%, c16 +1.7%; `*` = beyond noise. c1 is the median of 5 single
runs per boot (c1 is bimodal on this pair).

### Quality (b1.4)

| Check | Result |
|---|---|
| tool-eval-bench `--hardmode` (88 tool-use scenarios, thinking on, temperature 0) | 92 and 92/100 on the two A/B boots (previous build 88-90; run-to-run band 86-93). TC-45 (`tool_choice=required`) 5/5 |
| Long-context recall (`scripts/fidelity_probe.py`, 20 tool-call retrievals per depth) | A/B gate boot: 20/20 at 8k, 64k, 128k and **19/20 at 32k** (one `no_call` on the first cold-prefill trial). Re-gate on a fresh boot: 20/20 at 8k, 32k, 64k and 128k, plus 32k seeds 21 and 22 20/20 each. Cold-32k repro (20 fresh-prefix trials per build): b1.3 20/20, b1.4 20/20, cold TTFT ~23 s on both. 128k seeds 11 and 13: 20/20 each |
| Batch stragglers, c5-c16 (`scripts/straggler_probe.py`) | none |

Verdict: `arm_verdict.py` v2 forced a reject on the 32k 19/20 alone; after the re-gate passed, Jev on the same fact sheet
with the re-gate result: **ship** (0.78; distribution ship 0.83 / reject 0.10 / ship_with_caveat 0.05 / rerun 0.02).
Logits check (20 prompts x 16 tokens, 2 captures per boot): cross-build mean |dlogprob| 0.033-0.041, within the
0.038-0.046 self-noise. Files: [`results/b1.4-20261001/`](../results/b1.4-20261001/).

**Thinking effort default** (DevOps 14 prompts x 3, same grader as 2026-09-28, T=1.0, max 16,384 tokens):

| | Mean checks passed | Clean runs | Runaway thinking | Median time per task |
|---|:---:|:---:|:---:|:---:|
| b1.2, `xhigh` (template default) | 42.5% | 17 / 42 | 23 / 42 | 246 s |
| b1.4 candidate, `medium` | 95.9% | 29 / 42 | 0 / 42 | 33 s |
| same weights at TP=1, `medium` | 97.8% | 35 / 42 | 0 / 42 | 105 s |

Total wall time for the 42 runs 1,498 s at `medium` vs 8,982 s at `xhigh`. Jev on making `medium` the server default:
**medium_default 1.00**. Per-task table: [`results/b1.4-20261001/devops-b14-medium.md`](../results/b1.4-20261001/devops-b14-medium.md).

## b1.3 image (2026-09-29)

Recommended from 2026-09-29 to 2026-10-01, now the `-previous` fallback; superseded by b1.4 (above).

Terms used: **concurrency (c)** = requests running at the same time;
**depth** = tokens of earlier conversation already cached before the new prompt; **MTP** (multi-token prediction) = the
model drafts up to 4 next tokens that are checked in one step, which is what makes single-user speed high.

Image `ghcr.io/ursuciprian/spark-vllm-b12x:b1.3-20260929-b7fbaf96-7344a997-warm`
(`sha256:32cb8bd8800e413726b4cfe3d9947f80d4eb01d92dbe12405e3c763db7306a02`), checkpoint revision `7c4f1bc1`.
A/B against the previous build (2026-09-27) on 2026-09-29, two separate boots per build, means of both boots.
Raw files and verdict: [`results/b1.3-20260929/`](../results/b1.3-20260929/). The 2026-09-27 build's tables follow below.

**Coding**: [llama-benchy fork](https://github.com/ursuciprian/llama-benchy) `--prompt-mode task` (agent coding turn,
2048 new prompt tokens, up to 512 out, thinking on, temperature 1.0 / top-p 0.95 / top-k 20, prefix caching, 3 runs per boot).
Total tok/s; `*` = beyond run-to-run noise.

| depth | c1 | c2 | c4 | c5 | c8 | c10 | c16 |
|---|---|---|---|---|---|---|---|
| 0 | 62.2 | 101.8 * | 147.3 | 144.7 * | 188.0 | 197.0 | 241.1 |
| 0, previous | 58.2 | 97.8 | 142.8 | 151.9 | 187.5 | 196.0 | 240.2 |
| 16k | 63.5 * | 85.7 | 113.8 | 123.7 | 143.3 | 153.4 | 178.5 |
| 16k, previous | 60.4 | 85.8 | 111.8 | 122.2 | 146.4 | 152.6 | 176.1 |

Depth 0: c1 +6.9%, c2 +4.0%, c4 +3.1%, c5 -4.7%, c8 +0.2%, c10 +0.5%, c16 +0.4%. 16k: c1 +5.3%, c2 -0.2%, c4 +1.7%, c5 +1.2%, c8 -2.1%, c10 +0.6%, c16 +1.4%.
Beyond noise: d0 c2 and 16k c1 up; d0 c5 down 4.7% (the same change measured +4.7% beyond noise at d0 c5 in its first
A/B on 2026-09-28, so this cell is run-to-run spread of the temperature-1.0 grid at c5). Everything else within noise.
Paired temperature-0 decode-step probe (same prompt, pooled 2 boots): step time -3.1% counting / -1.9% fresh / -2.7% 16k
at c1, -3.0/-2.5% at c2, -2.4% counting c4, -1.4..-2.8% at c8/c16; MTP acceptance per position unchanged.
The gain is from skipping 35 per-group GDN metadata refreshes (~350 small launches) on every uniform decode step.

**Counting** (`tools/tony-bench/bench_sweep.py`: "list the numbers from 1 to 300", temperature 0, thinking off, 300 tokens).
Nearly every draft token is accepted, so this is the stack's upper bound, not coding speed. Aggregate tok/s:

| | c1 | c2 | c4 | c5 | c8 | c10 | c16 |
|---|---|---|---|---|---|---|---|
| current | 114.0 | 194.7 | 325.9 | 374.0 | 525.4 * | 570.0 | 754.4 |
| previous | 109.7 | 198.8 | 295.3 | 392.4 | 499.1 | 574.0 | 751.2 |

c1 +4.0%, c2 -2.1%, c4 +10.3%, c5 -4.7%, c8 +5.3%, c10 -0.7%, c16 +0.4%; `*` = beyond noise. c1 is the median of 5 single runs per boot (c1 is bimodal on this pair).

**Single request by workload** (`scripts/decode_probe.py` after the cold-boot test, temperature 0, 512 tokens, 5 runs, mean tok/s):
code 56.5, structured 81.6, counting 102.5, prose 48.2 on the 2026-09-25 build; not re-run for this build (decode step time
change is already captured by the paired probe above).

### Quality (b1.3)

| Check | Result |
|---|---|
| tool-eval-bench `--hardmode` (88 tool-use scenarios, thinking on, temperature 0) | 88 and 90/100 on the two A/B boots (91 and 92 in the 2026-09-28 A/B of the same change; previous build 90-92; run-to-run band 86-93). TC-45 (`tool_choice=required`) 5/5 |
| Long-context recall (`scripts/fidelity_probe.py`, 20 tool-call retrievals per depth) | 20/20 exact at 8k, 32k, 64k and 128k; 128k also on two more seeds (11, 13): 20/20 each |
| Batch stragglers, c5-c16 (`scripts/straggler_probe.py`) | none |

Gate run on the A/B boots of the same build (`scripts/gate_arm.sh`); the published image adds only the plan-seed layer.
Verdict: `arm_verdict.py` v2 + Jev, **ship with caveat** (confidence 1.00; the caveat is the d0 c5 coding cell); the same change as a mod on the previous image: ship 0.99 (2026-09-28). Logits check (20 prompts x 16 tokens, 2 captures per boot): cross-build mean |dlogprob| 0.031-0.035, within the 0.037-0.042 self-noise. Files: [`results/b1.3-20260929/`](../results/b1.3-20260929/).

**Task evals (2026-09-28, previous build b1.2, thinking on, card sampling):** a DevOps set of 14 prompts x 3 (Terraform, Kubernetes,
GitHub Actions, IAM, bash, Helm, Dockerfile, Prometheus, incident triage), graded by terraform/kubeconform/actionlint/
shellcheck/helm/hadolint/promtool plus rubric checks: **17/42 runs clean, mean check score 42.5%**. The main failure is
**runaway thinking**: 23/42 runs (55%) spent the whole 16k-token budget thinking and never answered, mostly on prompts that
say "must pass <validator>". When the model did answer, 17 of 19 runs were clean. MMLU-Pro (2000 subset) was stopped at 770/2000
with no score; GSM8K, IFEval, LiveCodeBench and the comparison with the previous build are still to run. Details:
[results/evals-20260928/](../results/evals-20260928/README.md).

## b1.2 image (2026-09-27)

Recommended from 2026-09-27 to 2026-09-29, then the `-previous` fallback until 2026-10-01 (archived).

Image `ghcr.io/ursuciprian/spark-vllm-b12x:b1.2-20260927-b7fbaf96-a9aa81b2-warm`
(`sha256:ed5520eb037ceaadb02c9325d05ad55a37dc7cf972e0ef25a1f24dcddef624cb`), checkpoint revision `7c4f1bc1`.
A/B against the previous build (2026-09-26) on 2026-09-27, two separate boots per build, means of both boots.
Raw files and verdict: [`../results/b1.2-20260927/`](../results/b1.2-20260927/). The 2026-09-26 build's tables are in
this file.

**Coding**: [llama-benchy fork](https://github.com/ursuciprian/llama-benchy) `--prompt-mode task` (agent coding turn,
2048 new prompt tokens, up to 512 out, thinking on, temperature 1.0 / top-p 0.95 / top-k 20, prefix caching, 3 runs per boot).
Total tok/s; `*` = beyond run-to-run noise.

| depth | c1 | c2 | c4 | c5 | c8 | c10 | c16 |
|---|---|---|---|---|---|---|---|
| 0 | 61.8 * | 96.3 * | 142.2 * | 151.7 | 187.4 | 198.2 | 239.5 |
| 0, previous | 53.4 | 87.5 | 135.6 | 147.3 | 183.1 | 195.0 | 242.2 |
| 16k | 57.6 * | 87.2 * | 114.1 * | 127.4 | 142.9 | 152.9 | 173.3 |
| 16k, previous | 55.0 | 82.6 | 109.8 | 120.2 | 141.4 | 151.5 | 175.9 |

Depth 0: c1 +15.7%, c2 +9.9%, c4 +4.9% beyond noise; c5-c16 within noise (no cell worse).
16k: c1 +4.9%, c2 +5.6%, c4 +3.9% beyond noise; c5-c16 within noise (no cell worse).
Paired temperature-0 decode-step probe (same prompt, pooled 2 boots): step time -9.9% at c1 (fresh), -10.0% at c1 (16k),
-7.6/-7.4% at c2, -4.4/-5.7% at c4, -2..-5% at c8/c16 (every CI excludes 0); MTP acceptance per position unchanged.
The gain is from online MXFP8 on the hyper-connection mixers (router gate stays BF16): half the bytes read per mixer per step.

**Counting** (`tools/tony-bench/bench_sweep.py`: "list the numbers from 1 to 300", temperature 0, thinking off, 300 tokens).
Nearly every draft token is accepted, so this is the stack's upper bound, not coding speed. Aggregate tok/s:

| | c1 | c2 | c4 | c5 | c8 | c10 | c16 |
|---|---|---|---|---|---|---|---|
| current | 111.9 | 199.9 | 329.1 | 386.1 | 503.3 | 572.9 | 751.5 |
| previous | 100.7 | 176.4 | 305.9 | 349.7 | 487.3 | 533.5 | 737.4 |

c1 +11.1%, c2 +13.3%, c5 +10.4%, c8 +3.3%, c10 +7.4% beyond noise; c4 +7.6% and c16 +1.9% within noise (no cell worse).
c1 is the median of 5 single runs per boot (c1 is bimodal on this pair).

## b1.1 image (2026-09-26)

Recommended from 2026-09-26 to 2026-09-27, then the `-previous` fallback until 2026-09-29 (archived).

Image `ghcr.io/ursuciprian/spark-vllm-b12x:b1.1-20260926-b7fbaf96-6d232f16-warm`
(`sha256:91a60ebce422db8a80f58ce998a3e9bd847814c8579ac33fa4df99e62351b3a8`), checkpoint revision `7c4f1bc1`.
A/B against the previous build (2026-09-25) on 2026-09-26, two separate boots per build, means of both boots.
Raw files and verdict: [`../results/b1.1-20260926/`](../results/b1.1-20260926/).

**Coding**: [llama-benchy fork](https://github.com/ursuciprian/llama-benchy) `--prompt-mode task` (agent coding turn,
2048 new prompt tokens, up to 512 out, thinking on, temperature 1.0 / top-p 0.95 / top-k 20, prefix caching, 3 runs per boot).
Total tok/s; `*` = beyond run-to-run noise.

| depth | c1 | c2 | c4 | c5 | c8 | c10 | c16 |
|---|---|---|---|---|---|---|---|
| 0 | 55.1 | 88.7 | 136.2 | 143.6 | 188.1 | 191.4 | 237.6 |
| 0, previous | 54.2 | 90.6 | 137.3 | 147.8 | 183.5 | 197.2 | 236.3 |
| 16k | 56.1 * | 82.8 | 108.8 * | 121.1 * | 139.5 * | 151.6 * | 176.5 * |
| 16k, previous | 52.7 | 80.8 | 101.0 | 111.6 | 119.6 | 125.8 | 139.4 |

At 16k depth: c1 +6.5%, c4 +7.7%, c5 +8.5%, c8 +16.7%, c10 +20.5%, c16 +26.6%. Depth 0 is within noise (-2.9% to +2.5%); no cell is worse.
The gain comes from the prefix cache: with MTP it used to stop one 2864-token block short, so every follow-up turn re-read
~2.9K extra tokens and that prefill held up decoding for everyone. Decode step time itself is unchanged (paired probe, ±3% in all cells).

**Time to first token on a follow-up turn** (same conversation, cached context + 2048 new tokens, single request, mean seconds):
16k 2.58 -> 1.63 (-37%), 64k 3.43 -> 2.27 (-34%); at 16k with c1-c16 running, -29% to -53%. A cold first prompt is unchanged
(16k 5.4 s, 64k 23.7 s). Prefix-cache hit at 16k: 11,456 -> 14,320 tokens; at 64k: 60,144 -> 63,008.

**Counting** (`tools/tony-bench/bench_sweep.py`: "list the numbers from 1 to 300", temperature 0, thinking off, 300 tokens).
Nearly every draft token is accepted, so this is the stack's upper bound, not coding speed. Aggregate tok/s:

| | c1 | c2 | c4 | c5 | c8 | c10 | c16 |
|---|---|---|---|---|---|---|---|
| current | 101.5 | 180.7 | 302.5 | 359.1 | 483.5 | 546.6 | 728.3 |
| previous | 100.1 | 181.4 | 308.6 | 359.8 | 478.7 | 546.9 | 724.2 |

All within noise. c1 is the median of 5 single runs per boot (c1 is bimodal on this pair). c2/c4/c5 are the mean of 4 boots
(two with 10-round sweeps): the 3-round c4 cell swings ±5%; a paired temperature-0 probe on the same prompt measured c4 step time
+0.4% (CI -0.3..+1.0) and tokens per step +0.3%.

**Single request by workload** (`scripts/decode_probe.py` after the cold-boot test, temperature 0, 512 tokens, 5 runs, mean tok/s):
code 56.5, structured 81.6, counting 102.5, prose 48.2 on the 2026-09-25 build; not re-run for this build (decode step time is unchanged).

### Quality (b1.1)

| Check | Result |
|---|---|
| tool-eval-bench `--hardmode` (88 tool-use scenarios, thinking on, temperature 0) | 91 and 89/100 on the two A/B boots (previous build: 88-91; run-to-run band 86-93). TC-45 (`tool_choice=required`) 5/5 |
| Long-context recall (`scripts/fidelity_probe.py`, 20 tool-call retrievals per depth) | 20/20 exact at 8k, 32k, 64k and 128k; 128k also on two more seeds (11, 13): 20/20 each |
| Batch stragglers, c5-c16 (`scripts/straggler_probe.py`) | none; 3.97-4.00 tokens accepted per 4-token draft |

## b1 image (2026-09-25)

Recommended from 2026-09-25 to 2026-09-26, now the `-previous` fallback; superseded by b1.1 (README).

Image `ghcr.io/ursuciprian/spark-vllm-b12x:b1-20260925-b7fbaf96-14077fb3-warm`
(`sha256:57c2fbd8cd811a5d22a7f2e547453f97b875f1fb4c7de60a0c3ff9fba3a79e5c`), checkpoint revision `7c4f1bc1`.
A/B against the previous build on 2026-09-24/25, two separate boots per build, means of both boots.
Raw files and verdict: [`../results/b1-20260925/`](../results/b1-20260925/).

**Coding**: [llama-benchy fork](https://github.com/ursuciprian/llama-benchy) `--prompt-mode task` (agent coding turn,
2048 new prompt tokens, up to 512 out, thinking on, temperature 1.0 / top-p 0.95 / top-k 20, prefix caching, 3 runs per boot).
Cells: total tok/s / per-request tok/s (decode).

| depth | c1 | c2 | c4 | c5 | c8 | c10 | c16 |
|---|---|---|---|---|---|---|---|
| 0 | 54.1 | 91.7 / 47.4 | 135.3 / 36.8 | 147.0 / 32.8 | 186.6 / 26.7 | 196.8 / 23.3 | 241.1 / 18.7 |
| 0, previous | 53.2 | 85.5 / 44.2 | 131.6 / 35.2 | 137.3 / 30.1 | 166.8 / 24.2 | 180.0 / 21.3 | 218.6 / 17.1 |
| 16k | 55.3 | 81.6 / 45.5 | 103.5 / 33.2 | 112.9 / 30.2 | 120.2 / 22.1 | 127.2 / 19.2 | 138.8 / 14.2 |
| 16k, previous | 54.0 | 72.3 / 39.7 | 98.6 / 31.2 | 103.6 / 28.0 | 112.8 / 20.5 | 119.1 / 17.6 | 130.9 / 12.9 |

Beyond run-to-run noise: depth 0 c5-c16 +7 to +12%, 16k c2-c16 +5 to +13%. c1 and depth-0 c2/c4 are within noise; no cell is worse.
Depth 64k and the prose workload were not re-measured; the previous build's tables (2026-09-23, incl. 64k and prose) are in
[docs/BENCHMARKS.md](#b0-shipped-image-2026-09-23).

**Time to first token, coding** (mean seconds): depth 0: c1 0.73, c4 2.0, c16 6.1; depth 16k: c1 2.6, c4 7.7, c16 22.2.
Prefill: ~2,900-3,300 tok/s at depth 0, ~780-860 at 16k.

**Counting** (`tools/tony-bench/bench_sweep.py`: "list the numbers from 1 to 300", temperature 0, thinking off, 300 tokens).
Nearly every draft token is accepted, so this is the stack's upper bound, not coding speed. Aggregate tok/s:

| | c1 | c2 | c4 | c5 | c8 | c10 | c16 |
|---|---|---|---|---|---|---|---|
| current | 101.0 | 174.8 | 307.5 | 360.1 | 477.4 | 549.5 | 720.4 |
| previous | 100.9 | 181.2 | 288.4 | 319.8 | 443.7 | 486.4 | 643.0 |

c1 is the median of 10 single runs (5 per boot; range 85-103, c1 is bimodal on this pair). c5-c16 +8 to +13%.

**Single request by workload** (`scripts/decode_probe.py` after the cold-boot test, temperature 0, 512 tokens, 5 runs, mean tok/s):
code 56.5, structured 81.6, counting 102.5, prose 48.2 (previous build: 61.3 / 88.9 / 96.9 / 49.0; single runs vary ±15%).

## b0 shipped image (2026-09-23)

Previous default, superseded 2026-09-25 by b1 (README). Pinned checkpoint on both ranks; not hybrid.
Image `ghcr.io/ursuciprian/spark-vllm-b12x:b0-20260918-a8333658-warm`
(`sha256:a3d5d90d1312edc9a79c86add6fdf72d50b2a73558e295fdf4ea110488fb615d`), checkpoint revision `7c4f1bc1`,
measured 2026-09-23. Raw files: [`results/shipped-20260923/`](../results/shipped-20260923/).
Cells are **total tok/s / per-request tok/s** (decode). Depth = cached context before the new prompt.

**Coding** ([llama-benchy fork](https://github.com/ursuciprian/llama-benchy) `--prompt-mode task`: agent coding turn,
2048 new prompt tokens, up to 512 out, thinking on, temperature 1.0 / top-p 0.95 / top-k 20, prefix caching, 3 runs)

| depth | c1 | c2 | c4 | c5 | c8 | c10 | c16 |
|---|---|---|---|---|---|---|---|
| 0 | 59.2 | 91.6 / 47.5 | 135.3 / 36.2 | 138.7 / 31.2 | 175.7 / 25.0 | 179.2 / 21.6 | 224.1 / 17.2 |
| 16k | 56.8 | 81.5 / 45.0 | 100.4 / 31.9 | 106.6 / 28.7 | 115.1 / 20.7 | 120.2 / 17.9 | 132.8 / 13.1 |
| 64k | 56.8 | 73.3 / 41.1 | 88.9 / 31.1 | 93.1 / 27.8 | 99.9 / 19.1 | 102.2 / 16.0 | 110.5 / 11.6 |

**Prose** (llama-benchy default continue mode, book text, 2048 in, 128 out, server-default sampling).
Only 128 tokens per request, so at depth the prefills of other requests fill most of the window and totals stay flat or fall as concurrency rises.

| depth | c1 | c2 | c4 | c5 | c8 | c10 | c16 |
|---|---|---|---|---|---|---|---|
| 0 | 51.0 | 92.9 / 48.7 | 97.9 / 31.8 | 98.9 / 25.9 | 123.1 / 21.0 | 112.2 / 16.9 | 126.1 / 12.8 |
| 16k | 56.8 | 55.8 / 38.0 | 54.9 / 25.7 | 52.5 / 21.9 | 50.8 / 14.9 | 49.8 / 11.9 | 49.9 / 8.3 |
| 64k | 53.6 | 49.6 / 35.6 | 45.9 / 23.7 | 43.5 / 21.0 | 40.7 / 13.3 | 39.8 / 10.5 | 39.0 / 7.1 |

**Time to first token, coding** (mean seconds; under concurrency requests queue behind each other's prefill)

| depth | c1 | c4 | c16 |
|---|---|---|---|
| 0 | 0.69 | 2.0 | 6.1 |
| 16k | 2.6 | 7.7 | 22.3 |
| 64k | 3.5 | 10.1 | 29.4 |

Prefill: ~3,000-3,200 tok/s at depth 0.

**Counting: a speculative-decoding ceiling, not coding speed.** bench_sweep "list the numbers from 1 to 300",
temperature 0, thinking off, 300 tokens. Nearly every draft is accepted, so this shows the stack's upper bound and catches regressions.

| | c1 | c2 | c4 | c5 | c8 | c10 | c16 |
|---|---|---|---|---|---|---|---|
| total tok/s | 100.2 | 183.9 | 304.2 | 340.6 | 456.1 | 500.7 | 634.9 |
| per request | 100.2 | 92.0 | 76.3 | 68.7 | 57.3 | 50.5 | 41.1 |

Five extra c1 runs: 100.3-102.1 tok/s.

**Single request by workload** (`scripts/decode_probe.py`, temperature 0, 512 tokens, 5 runs, mean tok/s):
code 61.3, structured 88.9, counting 96.9, prose 49.0.

> [!WARNING]
> **Hybrid checkpoint.** Every table here was measured before 2026-09-23 14:12
> EEST. In that window rank 1 loaded checkpoint revision `ada4da32` while rank 0
> loaded `7c4f1bc1` (at least 2026-09-18 onward, probably 2026-09-17 too). All
> numbers below are therefore (hybrid). Pinned-checkpoint results are in the
> [b0 shipped image](#b0-shipped-image-2026-09-23) and the README: [Results](../README.md#results-default-b1-image) and
> [Checkpoint revision split](#checkpoint-revision-split-fixed-2026-09-23).

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


## Moved from the README (2026-09-23)

The README became a short guide on 2026-09-23. These sections were in it before;
they are kept here unchanged apart from link paths. Numbers marked (hybrid) had
rank 1 on the old checkpoint revision and are superseded by the shipped-image
results in the [README](../README.md#results-default-b1-image) and [b0 shipped image](#b0-shipped-image-2026-09-23).

### Winning recipe

[`recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml`](../recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml)
(`B0`). What sets it apart from the other arms:

- **Checkpoint:** `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` QAD revision
  `7c4f1bc1`, pinned with `model_revision` plus `--revision` so both ranks load
  exactly that revision.
- **Speculative decoding:** MTP with 4 draft tokens and
  `draft_sample_method: probabilistic`. Drafts are sampled from the draft
  distribution, so acceptance holds up for clients at the default temperature
  (agent coding c16 221.9 vs 186.3 agg tok/s for one-hot drafts, (hybrid)).
- **lm_head:** the verify-head lm_head runs as online MXFP8
  (`VLLM_MXFP8_LM_HEAD=1`); the MTP draft head is NVFP4.
- **Image:** `ghcr.io/ursuciprian/spark-vllm-b12x:b0-20260918-a8333658-warm`.
  It is vLLM fork `8e1f1e58` with b12x `a8333658`, plus a warm layer carrying the
  b12x kernel-selection cache and the TP2 startup bounded-wait patch.

#### Sampling settings

The server uses the checkpoint's generation defaults: **temperature 1.0,
top-p 0.95, top-k 20**. Agent clients that send no temperature (omp, pi,
hermes) get these defaults, and that is the setting the recipe is tuned for.
Temperature 0.6 measured the same speed within noise. Temperature 0 is only
used for the counting diagnostic, not for real work.

#### Quality gate on the pinned checkpoint (2026-09-23)

Both ranks on `7c4f1bc1`. Files: `results/arms/pinned-7c4f1bc1/` (dgx-01).

| Gate | Result |
|---|---|
| tool-eval-bench `--hardmode` (88 scenarios, thinking on) | **90/100**; fails TC-45, TC-49, TC-68, TC-74 |
| Fidelity probe (20 tool-call retrievals per depth) | **20/20** exact at 8k, 32k, 64k and 128k; typo rate 0.00 |
| Straggler probe (batches 5-16) | no stragglers |
| Counting diagnostic (bench_sweep, temp 0, thinking off; not user throughput) | c1 102.0, c4 305.6, c8 443.3, c16 650.6 agg tok/s |
| decode_probe c1, temp 0 | code 56.2, structured 93.4, counting 102.9, prose 49.8 tok/s |
| decode_probe c1, temp 0, shipped warm image, cold boot (5 runs, mean / peak) | code 61.3 / 65.6, structured 88.9 / 96.1, counting 96.9 / 102.7, prose 49.0 / 51.2 tok/s |
| Agent coding and prose grids (default temperature) on the shipped image | **PENDING**: to be measured on the warm image; not measured yet |

#### Checkpoint revision split (fixed 2026-09-23)

sparkrun runs vLLM with `HF_HUB_OFFLINE=1`, so each node loads whatever its own
`refs/main` points at. sparkrun copies the model head-to-worker with
`rsync --size-only`, and a 40-byte commit hash always has the same size, so the
worker's `refs/main` was never updated. From at least 2026-09-18 until
2026-09-23 14:12 EEST, rank 0 loaded the QAD revision `7c4f1bc1` and rank 1
loaded the old PTQ revision `ada4da32`. Every number measured in that window
is labelled (hybrid) below: the agent coding and prose grids, the counting
ceilings, the old-vs-new comparison, and the la-family quality gates. The
09-17 numbers were probably affected too. The recipes now pin the revision;
details are in [docs/ENGINEERING.md](ENGINEERING.md#known-issues--fixes).

### At a glance

#### Agent coding task, default temperature (primary number) (hybrid)

Default recipe (probabilistic MTP drafts), measured 2026-09-21/22 with rank 1 on `ada4da32` (hybrid).
Source: [`results/benchy/la-mtpprob-task16.md`](../results/benchy/la-mtpprob-task16.md).

Workload: our llama-benchy fork
([ursuciprian/llama-benchy](https://github.com/ursuciprian/llama-benchy)
`0d4de42`, `--prompt-mode task --no-force-length`, command shape in
[`scripts/run_b_arm.sh`](../scripts/run_b_arm.sh), `--concurrency 1 4 10 16`).
Chat-shaped agent coding turn (short fixed system prompt, file excerpt,
coding instruction), 2048 new prompt tokens on top of a cached context of the
given depth (prefix caching on), up to 512 output tokens (stops naturally),
thinking on (server default; reasoning tokens counted), default temperature
(1.0 / top-p 0.95 / top-k 20), 3 runs per cell. ± is the spread across runs as
llama-benchy reports it.

| cached depth | conc. | gen agg tok/s | gen per-stream tok/s | prompt tok/s (agg) | TTFT (ms) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 45.57 ± 13.67 | 45.57 ± 13.67 | 3010.03 ± 114.52 | 698.46 ± 27.32 |
| 0 | 4 | 115.24 ± 22.77 | 30.68 ± 6.80 | 3230.00 ± 77.69 | 2027.22 ± 539.05 |
| 0 | 10 | 171.11 ± 19.49 | 19.69 ± 2.97 | 3196.54 ± 38.95 | 4058.79 ± 1753.61 |
| 0 | 16 | **221.91 ± 2.05** | 17.09 ± 1.59 | 3280.83 ± 7.19 | 5968.70 ± 2934.10 |
| 16k | 1 | 55.42 ± 3.20 | 55.42 ± 3.20 | 779.44 ± 4.34 | 2630.36 ± 14.96 |
| 16k | 4 | 102.47 ± 1.81 | 32.45 ± 3.57 | 852.46 ± 1.39 | 7588.48 ± 2095.66 |
| 16k | 10 | 122.32 ± 0.58 | 18.01 ± 3.77 | 861.12 ± 0.48 | 15065.82 ± 6497.38 |
| 16k | 16 | 115.40 ± 24.27 | 10.97 ± 4.01 | 814.81 ± 56.80 | 23850.35 ± 11577.43 |
| 64k | 1 | 58.42 ± 3.45 | 58.42 ± 3.45 | 570.51 ± 5.98 | 3596.16 ± 39.15 |
| 64k | 4 | 91.66 ± 0.93 | 30.10 ± 4.46 | 635.86 ± 1.16 | 10129.08 ± 2788.80 |
| 64k | 10 | 93.70 ± 12.18 | 14.70 ± 4.22 | 602.19 ± 61.42 | 21165.63 ± 10081.95 |
| 64k | 16 | 106.82 ± 5.89 | 10.89 ± 3.54 | 644.61 ± 1.14 | 29218.89 ± 14327.02 |

Aggregate = generated tokens per second summed over all concurrent streams;
per-stream = one request's rate. At c1 they are the same.

#### Other workloads (not comparable with the table above) (hybrid)

| Workload | Headline | Source |
|---|---|---|
| Agent coding, **temperature 0** (old argmax config only; not measured on the default recipe) | c1 55.3, c16 217.7 agg tok/s | [`results/benchy/la-task16-t0.md`](../results/benchy/la-task16-t0.md) |
| Agent coding, temperature 0.6 | same speed as temperature 1.0 within noise (2026-09-23, B0, (hybrid)) | `results/benchy/` task-t06 run on dgx-01 |
| **Prose continuation**, default recipe, default temperature (1.0), 128 output tokens | c1 56.05 ± 2.56, c16 126.72 ± 1.56 agg tok/s; 64k-cached c16 36.78 ± 2.68 | [`results/benchy/la-mtpprob-prose16.md`](../results/benchy/la-mtpprob-prose16.md) |
| **Counting ceiling — not user throughput** (bench_sweep "List the numbers from 1 to 300", temp 0, thinking off) | c1 100.9 / 102.1 / 101.8, c8 439.4 / 434.6, c16 641.3 / 635.0 agg tok/s | `results/arms/la-mtpprob/decode_c*.json`, `sweep.json` (dgx-01) |

The counting ceiling measures how fast the stack goes when speculation never
misses (~3.96-4.00 of 4 drafts accepted). It is a regression diagnostic,
**not** the speed of coding, chat or agent work. Full prose grid, counting
table and old-vs-new comparisons: [docs/BENCHMARKS.md](BENCHMARKS.md).

#### Default vs previous default (old la, one-hot argmax drafts) (hybrid)

| Workload (aggregate gen tok/s, temp 1.0) | old la | probabilistic (default) |
|---|---:|---:|
| Agent coding c1, fresh | 46.1 | 45.6 ± 13.7 (tie) |
| Agent coding c16, fresh | 186.3 | 221.9 |
| Agent coding 64k-cached c1 | 42.4 | 58.4 |
| Agent coding 16k-cached c16 | 120.3 | 115.4 ± 24.3 (lower) |
| Prose c1 | 41.1 | 56.1 |
| Prose 64k-cached c16 | 38.15 | 36.78 (regression) |
| Counting ceiling | — | unchanged within noise |

Sources: `results/benchy/la-task16.md` vs `la-mtpprob-task16.md`,
`la-prose16.md` vs `la-mtpprob-prose16.md` (all in
[`results/benchy/`](../results/benchy/)).

### Quality

Pinned-checkpoint gate: [Winning recipe](#quality-gate-on-the-pinned-checkpoint-2026-09-23).
The tables below are older (hybrid) gates. Default recipe (probabilistic drafts), files in `results/arms/la-mtpprob/`.

| Gate | Result | Setting | File |
|---|---|---|---|
| tool-eval-bench 2.6.1 `--hardmode` (88 scenarios) | **86/100**, rerun **90/100** (historical band across la-family arms 86-93) | thinking on | `hardmode.log`, `hardmode2.log` |
| Fidelity probe (20 tool-call retrievals per depth) | **20/20** at 8k/32k/64k; 128k **19/20**, then 20/20 and 20/20 on two reruns; 0 typos | thinking on, temperature 0.6 | `fidelity_probe.txt`, `fidelity_128k_rerun{1,2}.json` |
| Straggler probe (bench_sweep counting prompt, batches 5/6/7/8/12/16) | No preemptions, **3.96-4.00** accepted per draft, no one-request stall | temp 0, thinking off | `straggler.log` |

The straggler c6 round took 13.7 s with all six requests equally slow (other
rounds 4.7-7.5 s).

Previous default (old la + MXFP8 lm_head, `results/arms/la-lmq/`): fidelity
20/20 exact retrieval at 8k/32k/64k/128k, 0 typos, thinking on; hardmode
89/100 (la band across arms: 86-93/100); straggler batch 5-16 clean,
3.96-3.99 accepted/draft on the counting prompt.

**Two hardmode scenarios fail on every la-family boot:**

| Scenario | Cause | Status |
|---|---|---|
| TC-45 (`tool_choice=required`) | Parser bug, see [Known issues](ENGINEERING.md#known-issues--limits) | Fixed by `archive/mods/vllm-tc45-reasoning-structag-fix/` → 93/100, but that mod costs about -12% on the bench_sweep counting diagnostic at c1 (`results/arms/la-tc/sweep.json`, 85.0 vs 95.7). Not enabled by default. |
| TC-68 | Model wraps JSON in a code fence with commentary; the scenario intentionally sends no `response_format` | Model compliance, not a server bug — not fixable without defeating the test |

### How to read the numbers

Three workloads are used, and their numbers are not comparable with each other.

| Factor | What it means here |
|---|---|
| **Workload** | Agent coding task (primary), prose continuation, and the counting ceiling. The same engine shows ~100 tok/s on counting and ~45-58 on real single-stream work. |
| **Draft acceptance** | MTP proposes 4 tokens per step. On the counting prompt nearly all 4 are accepted (98.9% overall, ~3.96 per step, [`results/profiling/README.md`](../results/profiling/README.md)). On the sampled (temp 1.0) prose and agent task grids with probabilistic drafts, 39-41% are accepted, ~1.6 per step. |
| **Temperature** | At temp 0 the target and draft agree more often, so acceptance and throughput rise (task c1 55.3 at temp 0 vs 46.1 at 1.0, old la). Clients that send no temperature get 1.0. |
| **Thinking tokens** | Task and prose runs have thinking on; reasoning tokens are generated and counted like answer tokens. The counting diagnostic and the category harness run with thinking off. |
| **Cached-context depth** | Depth runs prefill 2048 new tokens on top of a cached context; TTFT and per-token decode both slow with depth, and concurrency at depth queues prefills. |
| **c1 bimodality** | c1 is bimodal boot to boot on the counting diagnostic (fast mode 94-102, slow mode 84-88, [`results/arms/c1-decline/verdict.md`](../results/arms/c1-decline/verdict.md)): report the median of >=5 sweeps and the max, not one run. |

Details and raw acceptance counts: [docs/BENCHMARKS.md](BENCHMARKS.md#how-to-read-these-numbers).
