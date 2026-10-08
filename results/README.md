# Results index

Every measurement, arm and verdict in this repo, newest ideas first within
each group. Ground truth for the current serving numbers is the top-level
`README.md` ("Benchmark results"), backed by `results/b1.4-20261001/`
(b1.3: `results/b1.3-20260929/`; b1.2: `results/b1.2-20260927/`; b1.1: `results/b1.1-20260926/`; b1: `results/b1-20260925/`; b0, 2026-09-23:
`results/shipped-20260923/`)
— this file is the index, not a second source of numbers.

Quality/task evals (not speed): `results/evals-20260928/` (b1.2 DevOps task set; MMLU-Pro stopped, other evals deferred).

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
- **[copy]** copy-heavy decode: 1-8 concurrent copy tasks from a shared cached
  prefix, low reasoning effort, 1,500 tokens out, 3 rounds per stream count;
  tok/s over the window where all streams decode, ~4.9 tokens/step. Files:
  `results/showcase-20261004/A/copy-streams.json` (2x Spark b1.4) and
  `B/dgx0{1,2}/copy-streams.json` (1x Spark v3a on each Spark), with the
  counting files of the same run; earlier: `results/b1.2-20260927/copy-streams-20260929.json`
  (2x b1.2), `results/tp1-v2-20261002/copy-streams.json` (1x v2); 1x v3c:
  `results/tp1-v3c-20261005/bench/copy-streams.json`; 1x v3d: `results/tp1-v3d-20261005/bench/copy-streams.json`
  (v3c rerun in the same window on the other Spark: `results/tp1-v3d-20261005/bench-v3c-dgx02/`). The README charts
  (`docs/img/`) are rendered by `scripts/make_charts.py` from `b14.json`,
  `b1.4-20261001/benchy/`, `showcase-20261004/A/`, `lib-bench-20261005/`, the
  1x build grids (`tp1-v2-*`, `tp1-v3a-*`, `tp1-v3b-*`, `tp1-v3c-*`) and `tp1-v3d-20261005/`.
- **[lib]** llm-inference-bench 0.7.6: 30 s sustained decode per cell at
  c1/c4/c8 with 0/16K/64K context, standalone prefill 8K-128K, hotel-lights x8,
  server default sampling.
- **[prose]** llama-benchy 0.4.0 default book continuation, 2048 new prompt
  tokens, 128 output tokens, default temperature 1.0; aggregate gen tok/s. "File" points at the primary
doc for that entry; most arms also have raw JSON/logs alongside it that
aren't summarized here.

## Current default recipe: b1.6 (b1.4 + the retrained MTP drafter, 2026-10-08); b1.4 = b1.3 + 131k-id MTP draft vocab + vllm#923/#914 fixes, default thinking effort medium

| date | recipe / arm | key numbers | verdict | file |
|---|---|---|---|---|
| 2026-10-08 | 2x b1.6: b1.4 + the v3e retrained MTP drafter (24 mtp.* tensors over 7c4f1bc1), b12x plans pinned to the control selections (k71; k70 without pinning lost at c1 on 33 seed-missing plans), Thunderdome on the pair + TP2 gate; then image check b16 | acceptance pos 1-4 0.819/0.670/0.546/0.445 -> 0.855/0.712/0.592/0.492; fresh c4 +10.89% (noise 6.48), fresh c8 +6.21% (3.52), 16K c8 wall +1.72% (1.14), tg512 c8 +7.15% (4.79), tg512 c1 +1.12% (15.03), no cell worse beyond noise; gate hardmode 92, TC-45 100, fidelity 20/20 x4 + 128k seeds 20/20 x2, 0 preemptions; Jev ship 0.88; check boot 0 measured, tg512 79.1 / 186.7 c1 / c8 | **promoted, 2x default (b1.6)** | `results/k71-tp2-refit-pinned-plans-20261008-0921/`, `results/b1.6-20261008/` |
| 2026-10-08 | DP=2 production gate (k72, #106): shipped 1x v3e on each Spark (plain sparkrun --solo) behind tools/dp2/pa_router.py (sticky least-sessions, dgx-01:8100) vs shipped 2x b1.6; k60 agent replay + full gate through the router | wall s TP2 / DP2: agent8 157.1 / 111.6, agent16 255.6 / 173.3, long-4 225.7 / 141.9, long-16 901.7 / 567.8 (DP2 29-37% sooner on all six); output tok/s agent8 12.2 / 20.5, agent16 10.3 / 14.3, long-16 1.2 / 2.0; follow-up TTFT mean long-16 63.4 / 8.1 s; sessions per replica even; 0 preemptions, 16 x ~129.5K clean on both; gate hardmode 93, TC-45 100, fidelity 20/20 x4 + 128k seeds 11/13, stragglers c5-c16 0 preemptions per replica; KV 993,754 per replica vs 3,650,419 TP2 | PASS; documented as the many-sessions option, TP2 stays default (README "Two Sparks") | `results/dp2-gate-k72-20261008-1135/` |
| 2026-10-07 | DP=2 candidate (k63, #106): shipped 1x v3d on each Spark behind pa_router.py with sticky least-sessions (new in k63) vs shipped 2x b1.4; k60 replay + spot check through the router | wall s TP2 / DP2: agent8 157.1 / 110.2, agent16 255.1 / 173.6, long-16 901.8 / 570.2 (30-37% sooner on all six); sessions per replica even (k60: 3/13 on agent16); TC-45 100, fidelity 20/20 at 8k-128k | candidate, gated in k72 | `results/dp2-candidate-k63-20261007-0944/` |
| 2026-10-07 | 1x v3e: v3d + retrained MTP drafter (#97 refit run 1, 72 steps on ~3.5M tokens of v3d outputs, fp8 drafter KV; HF ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE @ 16c9bd54) vs v3d, Thunderdome k56, one arm per Spark + gate | acceptance pos 1-4 dgx-01 0.819/0.661/0.535/0.431 -> 0.856/0.713/0.593/0.489, dgx-02 0.818/0.666/0.539/0.438 -> 0.854/0.708/0.589/0.488; fresh c4 +4.88% (noise 3.05) / +7.90% (2.82), fresh c8 +6.79% (3.56) / +8.08% (2.61), 16K c4 +5.47% (3.65) / +5.74% (3.23), 16K c8 wall +2.86% / +2.04% (1.00), pp2048 c1 +0.01% (1.00) / -0.89% (2.02), no cell worse beyond noise; gate hardmode 91, TC-45 100, fidelity 20/20 x4 (128k seeds 119/120 dgx-01, 120/120 dgx-02), 0 preemptions; Jev ship 0.97 | **promoted, 1x default (v3e)** | `results/thunderdome-k56-20261007/` |
| 2026-10-07 | DP=2 vs 2x TP=2 (k60, #103): shipped 1x v3d on each Spark behind pa_router.py (sha1 prefix affinity, dgx-01:8100, no load balancing) vs shipped 2x b1.4; agent replay from drive.py (agent8, agent16, long-4/8/12/16), one boot each | wall s TP2 / DP2: agent8 156.1 / 149.5, agent16 253.6 / 303.1 (router split 3/13), long-4 224.2 / 141.8, long-16 891.9 / 638.1; follow-up TTFT mean long-16 90.3 / 8.0 s; output tok/s agent8 13.1 / 20.2 (DP2 wrote 48% more tokens), long-16 1.2 / 1.9; 0 preemptions, 16 x ~129.5K sessions on both; quality through the router not checked | reference (#100) | `results/dp2-vs-tp2-k60-20261007-0248/` |
| 2026-10-07 | 2x b1.4 long-context concurrency (k59, #102): fidelity probe at 245,267 and 256,515 prompt tokens, c1/c2/c4 on one transcript, 2 and 4 separate transcripts | 240/240 exact, 0 preemptions, KV max 28% (4 x ~256K), decode per request 89.3-95.9 at c1, 30.9-34.7 with 4 at once, TTFT max 540.4 s (4 cold ~256K contexts), lowest MemAvailable 10.51 GiB | reference (#100) | `results/longctx-concurrency-k59-20261007-0206/` |
| 2026-10-07 | 1x MTP drafter KV (k58, #101): k58a BF16 drafter KV with the drafter page kept at the fp8 page size (dgx-01), k58b fp8 drafter KV with K scale 0.25 (dgx-02), Thunderdome vs v3d | k58a: pool 4568 blocks (unchanged), requests that fit 25.2 at 16K / 10.9 at 64K (control 24.2 / 11.5), acceptance pos 0-3 +0.005/+0.021/+0.033/+0.035, KILL after pass 1 by the two-sided acceptance rule, speed cells not measured; k58b: acceptance shift <= +0.004, pp2048 c1 -4.32% (noise 1.00%) KILL, tg512 c1 +7.87% and fresh-c1 +4.45% follow per-request acceptance (#101) | k58a re-screened as k62; k58b rejected | `results/thunderdome-k58-20261007/` |
| 2026-10-07 | 1x confidence-gated MTP depth r19b (k54, #94): d6-all (up to 6 drafts at any batch size, dgx-01), d6-c2 (6 at c1-c2, 4 at c3+, dgx-02), threshold 0.90, vs v3d | d6-all INCONCLUSIVE: fresh-c1 -7.51% (noise 6.78%), fresh-c4 -2.57% (2.01%), count-c8 +7.60%, tg512 c1 +12.90%, hook copy/count c1-c2 +9 to +17%; d6-c2 KILL: fresh c4 -4.31%, fresh c8 -5.56%, count c8 -7.38%, tg512 c8 -8.18% while drafting 4 at c3+; GPU test stage: 40 GDN builder tests failed on a Hub config lookup in a container without network (test setup, not the code) | not promoted | `results/thunderdome-k54-20261007-0418/` |
| 2026-10-07 | 2x TP2 port of the k58 drafter KV change (k61, #104) | skipped: no k58 arm passed the screen; re-queued behind k62 | skipped | - |
| 2026-10-06 | Coding probe (k55): 36 coding prompts (12 Python, 8 each C++/Rust/Go), c1, max_tokens 768, on shipped 2x b1.4 and 1x v3d (dgx-02), one pass per setting | T=0 thinking off, decode median (max): 2x 106.2 (120.6), 1x 72.9 (81.8); server defaults (thinking on): 2x 87.5 (97.2), 1x 59.7 (67.2); tokens/step 4.01 / 3.96 and 3.44 / 3.38; serve logs not committed | reference (#100) | `results/coding-probe-k55-20261006/` |
| 2026-10-02 | 2x b1.4 KV pool from the r9 dense-check baseline boot | 30.53 GiB, 3,673,158 tokens, 14.01x at 262,144 (max_num_seqs 32 boot: 3,615,479, 13.79x) | reference (#100) | `results/b1.4-20261001/kv-pool-2x.txt` |
| 2026-10-05 | High concurrency, max_num_seqs 32 (not the shipped cap; quality gate not run at this cap): 2x b1.4 and 1x v3d, one boot each (k46b) | 2x: [count] max of 5 c32 994.4, [copy] max of 3 at 32 streams 910.7, [task] tg512 c32 290.7, 0 preemptions, min MemAvailable 5.19 GiB; 1x v3d: [count] c32 675.9, [copy] 565.2, [task] c32 198.7, 0 preemptions, min MemAvailable 4.46 GiB; 1x s16 boot stopped by the 4 GiB guard | reported, not a recipe change (#89) | `results/high-conc-k46b-20261005/` |
| 2026-10-05 | 1x v3d (v3c + NVFP4 GDN weights for decode, MXFP8 copy for prefill, cutoff 41 rows; checkpoint ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE) vs v3c, Thunderdome screen + gate | screen: fresh c8 +4.44%, 16K c4 +5.89%, [count] c8 +7.83%, [task] tg512 c1 +19.3%, pp2048 c1 -0.89% (noise 1.00), acceptance shift <= 0.005, 72 MXFP8 copies per boot; cutoff-128 arm KILL (pp2048 -2.27%, noise 1.01); gate hardmode 91, TC-45 100, fidelity 20/20 x4 + 128k seeds 11/13, no stragglers, min MemAvailable 14.04 GiB; [task] tg512 c1 59.8, c8 139.5, 16k c8 111.0; [count] max of 5 c8 377.8; [copy] max of 3 c1/c4/c8 80.8/188.5/281.5 (v3c same window 74.8/189.6/268.4); [lib] c8 0/16K/64K 173.4/191.5/179.5 | **promoted, 1x default** | `results/tp1-v3d-20261005/` |
| 2026-10-05 | 1x v3c (v3b + shared GDN prefill staging + compact GDN records, KV 14 GiB) vs v3b, Thunderdome screen + gate | screen: fresh c4 +2.45%, [count] c8 +3.13%, 16K c8 wall 347/370 -> 125/126 s, rest within noise, acceptance unchanged, T=0 identity pass; kv16 arm INCONCLUSIVE (16K c4 -2.0%); gate hardmode 93, TC-45 100, fidelity 20/20 x4 + 128k seeds 11/13, no stragglers, min MemAvailable 15.01 GiB; [task] tg512 c1 51.8, c8 125.1, 16k c8 109.3 (v3b 22.1); [count] max of 5 c8 360.1; [copy] max of 3 c8 265.3; [lib] c8 0/16K/64K 165.7/165.6/160.4 | **promoted, 1x default** (Jev ship 0.72) | `results/tp1-v3c-20261005/` |
| 2026-10-05 | [lib] on 2x b1.4 and 1x v3b (#74), counting with every round saved (#82) | b1.4 c1/c4/c8 at 0/16K/64K: 64.8/73.3/77.8, 171.2/175.0/165.4, 248.6/254.9/257.4; v3b 46.9/45.3/44.3, 116.5/112.0/106.3, 168.4/174.8/did not fit; prefill 8K 2,857 / 2,162; hotel-lights 8/8, 7/8 | reference | `results/lib-bench-20261005/` |
| 2026-10-04 | 2x r9-pin (b12x block-scaled pinned plan 2560x3072@16) vs same image off, 2 passes ABBA | tok/s [count] c2 +2.3% [+1.1, +3.7], [count] c8 -3.1% [-5.3, -0.8] (step +2.8%), other cells within CI, acceptance unchanged | **rejected** (count c8 worse beyond noise; #69) | `results/r9-pin-20261003/report.txt` |
| 2026-10-01 | b1.4 (`b14`) vs b1.3, 2 boots each, same day | [task] temp 1.0 d0: c1 62.2 (b1.3 63.9, noise 9.6%), c2 105.1 (99.1), c5 160.6 (154.1) *; 16k c1 63.8 (64.8), c16 176.4 (175.8); [count] c1 119.8 (114.2) *, c2 214.3 (207.8) *, c5 407.7 (390.4) *; paired step time -2..-5% c1-c8, 16k c16 +1.3%; acceptance within +-0.03; hardmode 92/92, TC-45 5/5, no stragglers, fidelity 20/20 x4 on the re-gate (A/B boot 32k 19/20) + 32k seeds 21/22 + 128k seeds 11/13; logits within self-noise. DevOps 42 at medium: 95.9% checks, 29/42 clean, 0/42 runaway | **promoted, now default** (Jev ship 0.78; default effort medium 1.00) | `results/b1.4-20261001/` |
| 2026-09-29 | b1.3 (`b13`, gdnmeta baked) vs b1.2, 2 boots each, same day | [task] temp 1.0 d0: c1 62.2 (b1.2 58.2), c2 101.8 (97.8), c5 144.7 (151.9, caveat); 16k c1 63.5 (60.4); [count] c1 114.0 (109.7), c8 525.4 (499.1); paired step time c1 -1.9..-3.1%, c2 -2.5..-3.0%; hardmode 88/90, fidelity 20/20 x4 + 128k seeds 11/13, TC-45 5/5, no stragglers; logits within self-noise | **promoted** (Jev ship_with_caveat 1.00; mod A/B 2026-09-28 ship 0.99) | `results/b1.3-20260929/` |
| 2026-09-27 | b1.2 (`hcq`) vs b1.1, 2 boots each | [task] temp 1.0 d0: c1 61.8 (b1.1 53.4), c2 96.3 (87.5), c4 142.2 (135.6); 16k c1 57.6 (55.0), c2 87.2 (82.6), c4 114.1 (109.8); [count] c1 111.9 (100.7), c2 199.9 (176.4), c5 386.1 (349.7); paired step time -9.9% c1 fresh / -10.0% c1 16k / -7.6..-4.4% c2-c4; hardmode 90/92, fidelity 20/20 x4, TC-45 5/5, no stragglers | **promoted** (Jev ship 0.97) | `results/b1.2-20260927/` |
| 2026-09-26 | b1.1 (`b11pdrun`, prefix-drop) vs b1, 2 boots each | [task] temp 1.0 16k: c1 56.1 (b1 52.7), c4 108.8 (101.0), c8 139.5 (119.6), c16 176.5 (139.4); cached-prefix TTFT 16k -37%, 64k -34%; hardmode 91/89, fidelity 20/20 x4 + 128k seeds 11/13, TC-45 5/5, no stragglers | **promoted** (Jev ship 0.96) | `results/b1.1-20260926/` |
| 2026-09-25 | b1 (`oldb12x-on`) vs b0, 2 boots each | [task] temp 1.0 d0: c1 54.1 (b0 53.2), c8 186.6 (166.8), c16 241.1 (218.6); 16k c2 81.6 (72.3), c16 138.8 (130.9); [count] c1 101.0 (100.9), c16 720.4 (643.0); hardmode 89/89, fidelity 20/20 x4, no stragglers | **promoted, now default** (Jev ship 0.99) | `results/b1-20260925/` |
| 2026-09-23 | shipped warm image, both ranks on `7c4f1bc1` | [task] temp 1.0: c1 59.2, c16 224.1, 64k c1 56.8, 64k c16 110.5; [prose]: c1 51.0, c16 126.1, 64k c16 39.0; [count]: c1 100.2 (100.3-102.1 x5), c8 456.1, c16 634.9 | **shipped**; supersedes every (hybrid) number below | `results/shipped-20260923/` |
| 2026-09-22 | `la-mtpprob` — `draft_sample_method: probabilistic` | [task] temp 1.0: c1 45.6 ± 13.7 (old la 46.1, tie), c16 221.9 (186.3), 64k c1 58.4 (42.4), 16k c16 115.4 ± 24.3 (120.3); [prose]: c1 56.1 (41.1), c16 126.7 (117.8), 64k c16 36.78 (38.15, regression); [count]: c1 100.9-102.1, c8 434.6-439.4, c16 635.0-641.3; hardmode 86 then 90, fidelity 8k-64k 20/20, 128k 19/20 then 20/20 x2, straggler clean | **promoted, now default** (PR #25) | `results/arms/la-mtpprob/verdict.md`, `results/benchy/la-mtpprob-{task16,prose16}.md` vs `la-{task16,prose16}.md` |
| 2026-09-22 | old la (argmax) at temp 0, [task] | c1 55.3, c16 217.7; 16k c1 55.3; 64k c1 55.9; temp 0.6 grid pending | reference for temperature effect | `results/benchy/la-task16-t0.md` |
| 2026-09-21 | `la-lmq` — online MXFP8 verify-head lm_head (`VLLM_MXFP8_LM_HEAD=1`) | [count] c1 99.5 median/101.8 max (was 95.6/99.5), c8 440.7 (was 433.0), hardmode 89/100, fidelity 100%x4, straggler clean | **promoted, now default** | `results/arms/la-lmq/verdict.md` |
| 2026-09-21 | c1 "decline" investigation | [count] 25 runs, bimodal 94-99 (fast) / 84-88 (slow), no monotonic decline; GPU-internal or async-scheduling suspected, not host/thermal/RoCE/CPU-placement | root-caused as measurement noise, not a regression — report median of >=5 + max | `results/arms/c1-decline/verdict.md` |
| 2026-09-20/21 | ghcr / ghcr2 — pull-based image from published wheels | [count] ghcr (0 cached, tainted) c1 85.6; ghcr2 (272 cached) c1 87.7 (~-10% vs la's 97.4), TC-45 passes (first time), quality/fidelity/straggler equal or better | **not promoted** — quality win, throughput regression vs `la`; kept as an alternate pull path (PR #8) | `results/RESULTS.md` §"Cache-mount fix and ghcr image gate" |
| 2026-09-19/20 | cache-mount fix (`executor_config.volumes` -> `/tmp/.cache`) | root cause of "0 cached" boots: `HOME=/tmp` in container, plan cache never persisted; la boot log 0 cached / [count] c1 86 -> 272 cached / c1 97.4 | fixed, applied to `la`/`local16`/ghcr variants | `results/RESULTS.md` §"Cache-mount fix and ghcr image gate" |
| 2026-09-19 | `la-tc` — `vllm-tc45-reasoning-structag-fix` (Phase B) | TC-45 fixed, hardmode 93/100 (was 91), but [count] c1 85.0 vs baseline 95.7 (~-11%) | **reverted** — quality win not worth the throughput cost | `results/arms/tooleval-fix/notes.md` |
| 2026-09-20/21 | `la-tcc` — `vllm-tc45-cheap` (v2, gates on `required`/named only, not `auto`) | mechanism: old fix's grammar gate fired on `tool_choice=auto` too, serializing async sampling every step for any tool-bearing request; new gate narrows to `required`/named | validation recipe written, GPU validation pending — see file for the 6-step promotion checklist | `results/arms/tooleval-fix/cheap-design.md` |
| 2026-09-20 | `la-kk` — KK (karmic-kraken-beta) image, vllm `57a80980` + b12x `e9ce5477` | KK's `ParserManager` never collapses Qwen3 onto a shared engine (design difference from jovian), so the TC-45 bug class doesn't exist there; still needs the `abstract_parser.py` `reasoning=False` + `auto`-gate fixes (`vllm-tc45-cheap` variant `0002`) | in progress, not gated against `la` yet | `archive/mods/vllm-tc45-reasoning-structag-fix/`, `results/arms/tooleval-fix/cheap-design.md` §"Karmic-kraken differs" |

## Rejected arms (screened against `la` on [count], bar +3% c1 / +5% c8, nothing else worse than -2%)

| date | arm | change | result | verdict | file |
|---|---|---|---|---|---|
| 2026-09-19 | fusear | `fuse_allreduce_rms: true` | c1 86.7 (-12.2%), c8 428.2 (+3.9%) | reject — c1 regression | `results/kernel-pass/arms.md`, `results/kernel-pass/sweep_fusear.json` |
| 2026-09-19 | spec3 | `num_speculative_tokens: 3` (was 4) | c1 84.1 (-14.9%), c8 375.2 (-8.9%) | reject — both regress | `results/kernel-pass/arms.md`, `results/kernel-pass/sweep_spec3.json` |
| 2026-09-19 | noat | `B12X_AUTOTUNE: "0"` (diagnostic) | c1 81.1 (-17.9%), c8 414.8 (+0.7%) | diagnostic — confirms autotune worth ~18% at c1, not a promotion candidate | `results/kernel-pass/arms.md`, `results/kernel-pass/sweep_noat.json` |
| 2026-09-19/20 | fwd57f3572 | forward-port b12x's native W4A16 MoE-autotune fix (`57f3572`) onto the old image | c1 87.9-88.1 (-8%), c8 419.8-427.3 | reject — wins only 4/38 candidate races, repeatable regression | `results/arms/fwd57f3572-fix/sweep.json`, `results/kernel-pass/arms.md` |
| 2026-09-19 | occ MICRO=32 / DYNAMIC=32 (`B12X_MICRO_MAX_ACTIVE_CLUSTERS` / `B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS`) | c1/c8 within noise of baseline | reject — noise, kernel clamps occupancy to 48 SMs regardless | `results/kernel-pass/prep-deadlock/sweep_occ_*.json` |
| 2026-09-18 | gdnbf16 | `--mamba-ssm-cache-dtype bfloat16` | cannot boot | reject — b12x requires `state_dtype == torch.float32` | `misc/gdn-state-bf16-lever3-2026-09-18.md` |
| 2026-09-20/21 | dv | reduced draft vocab (MiaAI-Lab 47,149-id table, AGPL, local use only) for the MTP head | boots, 0% MTP acceptance (7 of 151k drafts) | reject — not a speed lever anyway, MTP head is 0.04% of step | `results/arms/dv/fidelity.json`, `archive/mods/vllm-dv-devicefix/` |
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
| 2026-09-16 | decode-aware prefill scheduling fixes the c5/c9/c12 depth collapse (root cause: `max_parallel_prefills=1` starving decode); promoted into `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-tuned-baseline.yaml` | `results/eugr-b12x/RESULTS.md` §"decode-aware prefill scheduling" |
| 2026-09-16 | SGLang b12x GDN kernel port (`archive/mods/sglang-gdn-b12x-decode`) — decode-only kernel matches Triton; decode+verify kernel breaks acceptance in the live server, root cause not found | `results/eugr-b12x/RESULTS.md` §"SGLang b12x GDN kernel port" |
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
