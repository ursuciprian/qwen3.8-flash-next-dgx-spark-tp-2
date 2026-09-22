# Arm la-lmq: online MXFP8 lm_head (`VLLM_MXFP8_LM_HEAD=1`) — verdict (2026-09-21)

> **Workload.** Unless a line says otherwise, every tok/s, c1/c8/c16 and
> ms/step figure in this file is the `tools/tony-bench/bench_sweep.py`
> counting diagnostic: "List the numbers from 1 to 300 separated by commas…",
> temperature 0, thinking off, non-streaming, 320 max tokens, fresh context,
> aggregate tok/s (c1 = per-stream). MTP accepts ~4 of 4 drafts on it, so it is
> a speculative-decoding ceiling, not coding or chat speed. Agent-coding and
> prose numbers: top-level `README.md`.
>
> The "bench_categories" table below is a different harness: 40 real prompts,
> temp 0, thinking off, streaming, c1; its `coding` row is 5 coding prompts.

**Verdict: PROMOTE.** Online MXFP8 quantization of the verify-head lm_head
(W8A16, activations stay BF16) is now the default in
`recipes/eugr/eugr-agents-serve-local16-la.yaml`.

## Root cause / why this exists

The verify-head lm_head is a dense `124160 x 2560` BF16 projection. On
b12x, the BF16 vocab projection only has a compiled kernel for `M=1`, so
every decode step's verify-head matmul fell back to cuBLAS: 636 MB read per
decode step, measured at 3.6 ms = 6.5% of the 55 ms c1 step. The MTP draft
head was already quantized to NVFP4 (`VLLM_MTP_NVFP4_LM_HEAD`, default 1),
but the verify head stayed BF16.

`VLLM_MXFP8_LM_HEAD=1` switches the verify-head linear to online MXFP8
(W8A16 — `Mxfp8OnlineLinearMethod(use_a16=True)`; only weights are
quantized to MXFP8, activations stay BF16), roughly halving the bytes read
per step. This is a fork feature from
`local-inference-lab/vllm@dev/jovian-judgement` (`8e1f1e58`,
`_supports_default_lm_head_quantization`) — not something built here.

With this change both lm_heads (MTP draft + verify) are quantized.

## Data (arm la-lmq vs baseline A, same boot night, 2026-09-21)

Source: dgx-01
`~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/arms/la-lmq/{sweep.log,hardmode.log,fidelity_probe.txt,straggler.log,categories_c1.log}`,
`results/arms/lmhead/lmqB*.json`, `results/arms/gemv/A*.json` (baseline).

### c1 throughput, 6 sweeps each (`bench_sweep.py --levels 1 --rounds 3`)

| run | baseline A (tok/s) | la-lmq / lmqB (tok/s) |
|---|---|---|
| 1 | 99.5 | 99.1 |
| 2 | 89.7 | 99.9 |
| 3 | 98.8 | 84.7 |
| 4 | 97.5 | 101.8 |
| 5 | 85.2 | 101.1 |
| 6 | 93.6 | 97.6 |
| **median** | **95.55** | **99.5** |
| **max** | **99.5** | **101.8** |

Baseline is bimodal per `results/arms/c1-decline/verdict.md` (fast mode
94-99, slow mode 84-88); la-lmq shows the same bimodality, so report median
of >=5 sweeps and the max (fast-mode) value for both, per that verdict's
operational recommendation.

### c8 throughput

| | agg tok/s | per-stream tok/s | w2w med | w2w p90 |
|---|---|---|---|---|
| baseline A (`A_c8.json`) | 433.0 | 55.2 | 5.80s | 6.03s |
| la-lmq (`lmqB_c8.json`) | 440.7 | 56.2 | 5.69s | 5.93s |

### Gate sweep (la-lmq, `sweep.log` / `sweep.json`)

| c | agg tok/s | per-stream | w2w med | w2w p90 | TTFT med |
|---|---|---|---|---|---|
| 1 | 100.6 | 100.6 | 3.18s | 3.87s | 0.76s |
| 4 | 293.7 | 74.0 | 4.32s | 4.70s | 2.21s |
| 8 | 454.7 | 57.9 | 5.53s | 5.63s | 4.27s |
| 16 | 645.1 | 41.4 | 7.74s | 7.87s | 8.27s |

### Hardmode (tool-eval-bench v2.6.1.dev65+g6be685f0e, full scenarios, thinking on)

- Quality: **89/100** (la band across arms: 86-93)
- Responsiveness: 51/100 (median turn 2.9s)
- Deployability: 78/100 (alpha=0.7)
- 74 passed / 9 partial / 5 failed, 157/176 points
- Weakest category: Autonomous Planning (50%, 3/6)

### Fidelity probe (8k / 32k / 64k / 128k context, 20 tool-call prompts each)

| depth | exact | near | wrong | typo_rate | ttft | lat |
|---|---|---|---|---|---|---|
| 8000 | 20/20 | 0 | 0 | 0.00 | 1.80s | 4.3s |
| 32000 | 20/20 | 0 | 0 | 0.00 | 3.10s | 5.3s |
| 64000 | 20/20 | 0 | 0 | 0.00 | 5.24s | 7.7s |
| 128000 | 20/20 | 0 | 0 | 0.00 | 9.65s | 11.5s |

20/20 at every depth, x4 depths — no logits/fidelity regression from the
verify-head quantization.

### Straggler probe (batches 5,6,7,8,12,16)

| c | round wall | preemptions | accept/draft |
|---|---|---|---|
| 5 | 4.9s | 0 | 3.98 |
| 6 | 5.3s | 0 | 3.96 |
| 7 | 5.1s | 0 | 3.98 |
| 8 | 5.5s | 0 | 3.99 |
| 12 | 6.8s | 0 | 3.99 |
| 16 | 7.4s | 0 | 3.98 |

Clean at c5-16, no preemptions, MTP acceptance stable at 3.96-3.99
accepted/draft across the whole concurrency range — the verify-head
quantization does not perturb MTP acceptance.

### bench_categories (c1, `categories_c1.log`)

Overall: auto 0.834, ttft med 0.14s, decode med 79.5 tok/s.

| category | auto | ttft | decode tok/s | tokens |
|---|---|---|---|---|
| coding | 1.0 | 0.12s | 90.1 | 154 |
| reasoning | 1.0 | 0.15s | 82.5 | 414 |
| json | 1.0 | 0.16s | 100.9 | 59 |
| html | 0.829 | 0.15s | 99.4 | 163 |
| prose | 0.4 | 0.14s | 48.8 | 201 |
| narrative | 0.693 | 0.14s | 45.4 | 324 |
| summary | 0.75 | 0.13s | 54.9 | 31 |
| format | 1.0 | 0.14s | 40.4 | 20 |

Sub-1.0 categories (prose word-count bounds, one html strict-tag miss, one
summary bullet-count miss) are known scoring-harness strictness on word/tag
counts, not tool-call or logits failures — consistent with other la-family
arms, not new with this mod.

## MTP acceptance metrics (whole gate run)

- drafts: 48,588
- accepted: 142,564
- accepted per draft position: pos0 42,323, pos1 36,916, pos2 33,086, pos3 30,239

## Conclusion

c1 median/max both improve (99.5/101.8 vs 95.6/99.5), c8 improves (440.7 vs
433.0), hardmode/fidelity/straggler all clean — no regression anywhere
measured. Promoted to `recipes/eugr/eugr-agents-serve-local16-la.yaml` as
`VLLM_MXFP8_LM_HEAD: "1"`.
