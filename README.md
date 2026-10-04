<div align="center">

# Qwen3.8-Flash-Next on DGX Spark: one or two GB10s

NVFP4 Qwen3.8-Flash-Next on vLLM V2 with b12x kernels and 4-token MTP, launched with sparkrun. 262k context, OpenAI-compatible API.<br>
Two recipes from one checkpoint: **2× Spark** (TP=2 over ConnectX-7) and **1× Spark** (TP=1, experimental).<br>
A build is promoted only if it passes the [quality gate](#quality-gate) and no c1–c4 cell is slower beyond noise.

<img src="docs/img/throughput.svg" alt="Aggregate decode tok/s by concurrent requests. 2× Spark: copy-streams 425 at 8 streams, counting 525 at c8 and 778 at c16, coding 196 at c8 and 243 at c16. 1× Spark: copy-streams 263 and counting 356 at c8, coding 118 at c8." width="900">

</div>

<!-- TODO(#65): the copy-streams cells (2× b1.2, 1× v2) and the 1× counting cells (v2) are refreshed by the pending
     showcase run (results/showcase-*): 2× copy-streams -> b1.4, 1× copy-streams and counting -> v3a. -->

| Workload | Tokens / step | 2× Spark, 1 request | 2× Spark, 8 requests | 1× Spark, 1 request | 1× Spark, 8 requests |
|---|:---:|:---:|:---:|:---:|:---:|
| **Copy-streams** (high acceptance) | 4.93–4.96 | 112<!-- TODO(#65) --> | **425**<!-- TODO(#65) --> | 75<!-- TODO(#65) --> | **263**<!-- TODO(#65) --> |
| **Counting** (high acceptance) | ~5.0 | 120 | **525** | 77<!-- TODO(#65) --> | **356**<!-- TODO(#65) --> |
| **Coding**, llama-benchy tg512 | 2.7–2.8 (2×) · 3.2–3.4 (1×) | 62 | 196 | 50 | 118 |
| **Coding** at 16k cached context | 2.6–2.7 (2×) · 3.1–3.2 (1×) | 64 | 146 | 51 | 20 ([limits](#known-limits)) |
| **Prefill**, 2,048-token prompt | | 2,784–2,855 | | 1,201 | |
| **Prefill**, filling a 16k context | | 2,895–2,920 | | 2,003 | |

Aggregate decode tok/s unless marked prefill. Builds: 2× Spark b1.4 (copy-streams b1.2); 1× Spark v3a (copy-streams and
counting v2). Workloads and raw files: [Measured](#measured).

<div align="center">

[![hardmode](https://img.shields.io/badge/hardmode-92%2F100-2ea44f)](#quality-gate)
[![TC-45](https://img.shields.io/badge/TC--45-100%2F100-2ea44f)](#quality-gate)
[![retrieval](https://img.shields.io/badge/tool--call%20retrieval-20%2F20%20up%20to%20245k%20tokens-2ea44f)](#quality-gate)
[![stragglers](https://img.shields.io/badge/batch%20stragglers-none-2ea44f)](#quality-gate)
<br>
[![2x build](https://img.shields.io/badge/2×%20Spark-b1.4%20·%202026--10--01-blue)](#changelog)
[![1x build](https://img.shields.io/badge/1×%20Spark-v3a%20·%202026--10--04-blue)](#changelog)
[![Engine](https://img.shields.io/badge/engine-vLLM%20V2%20+%20b12x-blue)](docs/REFERENCE.md#what-is-in-the-image)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)

</div>

## Quick start

Needs [sparkrun](https://github.com/eugr/sparkrun) ≥ 0.3.6.

```sh
sparkrun registry add https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2
sparkrun run qwen3.8-flash-next-2x-dgx-spark                     # 2× Spark: default cluster, or --hosts <head>,<worker>
sparkrun run qwen3.8-flash-next-1x-dgx-spark --hosts <spark> --solo   # 1× Spark
```

Then [verify](#verify) the server. Base URL `http://<head>:8000/v1`, model `qwen3.8-flash-next`.

[Measured](#measured) · [Quality gate](#quality-gate) · [Verify](#verify) · [Requirements](#requirements) · [Known limits](#known-limits) · [Recipes](#recipes) · [Changelog](#changelog) · [How we measure](#how-we-measure) · [Troubleshooting](#troubleshooting)

---

## Measured

Two kinds of workload, labelled per row, because they differ by about 2× in tokens per decode step:

- **High-acceptance** workloads (copying, counting) accept nearly every MTP draft (~4.9–5.0 tokens per step). They show
  the decode rate when drafts land, which bounds what MTP can give.
- **Coding** is an agent coding turn at temperature 1.0 with thinking on (2.6–3.4 tokens per step). It is what an agent
  sees in practice.

<!-- TODO(#65): copy-streams rows (2× b1.2, 1× v2) and the 1× counting row (v2) are refreshed by results/showcase-*. -->

### 2× Spark (TP=2), build b1.4

| Workload | Tokens/step | c1 | c4 | c8 | c16 | Build |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **High-acceptance:** copy-streams, median of 3 rounds | 4.93–4.96 | 112<!-- TODO(#65) --> | 268<!-- TODO(#65) --> | 425<!-- TODO(#65) --> | | b1.2 (2026-09-29) |
| **High-acceptance:** counting, T=0 | ~5.0 (acceptance 1.00/0.99/0.99/0.99) | 119.8 | 356.6 | 524.8 | 778.0 | b1.4 |
| **Coding:** llama-benchy tg512, depth 0 | 2.7–2.8 | 62.2 | 150.9 | 195.5 | 242.8 | b1.4 |
| **Coding:** llama-benchy tg512, 16k cached depth | 2.6–2.7 | 63.8 | 117.5 | 145.9 | 176.5 | b1.4 |

Prefill (c1): 2,784–2,855 tok/s for a 2,048-token prompt; 2,895–2,920 tok/s filling a 16k context (two boots).
TTFT at c1: 0.75 s (2k new tokens) / 1.70 s (2k new tokens on a 16k cached context). c2/c5/c10 cells:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md#b14-image-2026-10-01).

### 1× Spark (TP=1, experimental), build v3a

| Workload | Tokens/step | c1 | c4 | c8 | Build |
|---|:---:|:---:|:---:|:---:|---|
| **High-acceptance:** copy-streams, median of 3 rounds | 4.93–4.95 | 74.6<!-- TODO(#65) --> | 171.4<!-- TODO(#65) --> | 263.1<!-- TODO(#65) --> | v2 (2026-10-02) |
| **High-acceptance:** counting, T=0 | ~5.0 (3.98 accepted per 4 drafts) | 77.0<!-- TODO(#65) --> | 228.5<!-- TODO(#65) --> | 356.0<!-- TODO(#65) --> | v2 (2026-10-02) |
| **Coding:** llama-benchy tg512, depth 0 | 3.2–3.4 | 50.0 | 103.7 | 117.6 | v3a |
| **Coding:** llama-benchy tg512, 16k cached depth | 3.1–3.2 | 51.2 | 102.5 | **19.7** (see [limits](#known-limits)) | v3a |

Prefill (c1): 1,201 tok/s for a 2,048-token prompt; 2,003 tok/s filling a 16k context. TTFT at c1: 1.75 s (2k new
tokens) / 2.04 s (2k new tokens on a 16k cached context).

<img src="docs/img/prefill.svg" alt="Prefill tok/s at one request: 2× Spark 2,784–2,855 for a 2,048-token prompt and 2,895–2,920 filling a 16k context; 1× Spark 1,201 and 2,003." width="620">

**Conditions.** Coding rows: [llama-benchy](https://github.com/ursuciprian/llama-benchy) `--prompt-mode task`, 2,048 new
prompt tokens, up to 512 out, thinking on, T=1.0 / top-p 0.95 / top-k 20, prefix caching on; 2× Spark is the mean of
two boots × 3 runs, 1× Spark one boot × 3 runs. Counting: "list the numbers from 1 to 300", T=0, thinking off, 3 rounds
(2× Spark: mean of two boots). Copy-streams: 1–8 concurrent copy tasks from a shared cached prefix at low reasoning
effort, 1,500 tokens out, 3 rounds per stream count; tok/s is counted over the window where all streams decode, and the
table shows the median round. Tokens/step is 1 + 4 × accepted/draft tokens from vLLM's spec-decode counters (benchy
`accept/draft` column, copy-streams `tokens_per_step`). Raw files:
[`results/b1.4-20261001/`](results/b1.4-20261001/) ([`b14.json`](results/b1.4-20261001/b14.json), [`benchy/`](results/b1.4-20261001/benchy/)),
[`results/tp1-v3a-20261004/`](results/tp1-v3a-20261004/),
copy-streams [`results/b1.2-20260927/copy-streams-20260929.json`](results/b1.2-20260927/copy-streams-20260929.json) and
[`results/tp1-v2-20261002/copy-streams.json`](results/tp1-v2-20261002/copy-streams.json), 1× counting
[`results/tp1-v2-20261002/counting-sweep.json`](results/tp1-v2-20261002/counting-sweep.json).
Charts: `uv run scripts/make_charts.py` renders every chart on this page from those files.
<!-- TODO: vendor the copy-streams runner into scripts/ so the copy-streams rows are reproducible from this repo. -->

## Quality gate

A build ships only if it passes every check. A faster build that fails one is not promoted.

<img src="docs/img/quality-gate.svg" alt="Quality gate: hardmode 92/100 on both builds, TC-45 100/100, tool-call retrieval 20/20 at 16k, 62k, 123k and 245k tokens, no batch stragglers." width="820">

| Check | Tool | 2× Spark b1.4 | 1× Spark v3a |
|---|---|:---:|:---:|
| Hard multi-step tool use (88 scenarios, thinking on, T=0); gate ≥ 88 | [tool-eval-bench](https://github.com/SeraphimSerapis) `--hardmode` | 92/100 on both A/B boots (run-to-run band 86–93) | 92/100 |
| `tool_choice=required` compliance, 5 trials | TC-45 | 100/100 | 100/100 |
| Long-context tool-call retrieval, 20 needles per depth; actual prompt sizes ~15.7k / 61.7k / 122.6k / 245k tokens | [`scripts/fidelity_probe.py`](scripts/fidelity_probe.py) | 20/20 at every depth (re-gate boot¹); 32k seeds 21, 22 and ~245k seeds 11, 13: 20/20 | 20/20 at every depth; ~245k seeds 11, 13: 20/20 |
| Batch stragglers | [`scripts/straggler_probe.py`](scripts/straggler_probe.py) | none, c5–c16 | none, c8–c16, 0 preemptions |
| Numerics vs previous build (20 prompts × 16 tokens, top-5 logprobs) | [`scripts/logits_equiv.py`](scripts/logits_equiv.py) | mean \|Δlogprob\| 0.033–0.041 vs self-noise 0.038–0.046 | not run (host-side change only) |
| MTP acceptance per draft position (numerics canary) | paired decode probe | within ±0.03 of b1.3 | unchanged vs v2 in every cell |
| Host memory headroom during the gate | `MemAvailable` | not recorded | min 13.97 GiB |
| DevOps task set (14 prompts × 3, graded by terraform / kubeconform / actionlint / shellcheck / helm / hadolint / promtool) | own grader | 95.9% checks, 29/42 clean, 0/42 runaway thinking | 97.8% checks, 35/42 clean, 0/42 runaway (b1.4 weights at TP=1) |

¹ The A/B boot had one `no_call` at 32k (19/20). Twenty cold 32k trials per build then gave 20/20 for both b1.3 and
b1.4, and a fresh re-gate boot passed. Details: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#quality-b14).

Depth labels in the gate logs are 8k / 32k / 64k / 128k; the probe sizes transcripts by characters, and the logged
actual prompt sizes are the token counts in the table above.

Not yet measured at the `medium` thinking default: MMLU-Pro, GSM8K, IFEval, LiveCodeBench.

## Verify

`sparkrun run` returns before the engine is ready. Wait for health, then check that the reply has both reasoning and
an answer:

```sh
H=http://<head>:8000
until [ "$(curl -s -o /dev/null -w '%{http_code}' $H/health)" = 200 ]; do sleep 10; done
curl -s $H/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen3.8-flash-next","max_tokens":1024,
  "messages":[{"role":"user","content":"Write a Python function that reverses a string."}]}' \
| python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']
print('reasoning:', len(m.get('reasoning') or m.get('reasoning_content') or ''), 'chars')
print('content  :', (m.get('content') or '')[:300])"
```

Empty `content` with long reasoning means `max_tokens` ran out inside thinking; garbled text means a checkpoint
mismatch (see [Troubleshooting](#troubleshooting)).

**Optional long-context check** (stdlib Python): expect `exact 20` at both depths.

```sh
git clone https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2 && cd qwen3.8-flash-next-dgx-spark-tp-2
python3 scripts/fidelity_probe.py --base $H --model qwen3.8-flash-next --depths 8000,32000
```

The server binds 0.0.0.0 with no API key: keep it on a trusted network or put a proxy with auth in front.
Stop with `sparkrun stop --all`.
<!-- TODO: document --api-key through sparkrun. -->

> **Upgrading:** sparkrun caches registries and does not refresh them on `run`. Run `sparkrun registry update qwen38-flashnext`
> first, or the previous recipe revision boots.

## Requirements

| | 2× Spark | 1× Spark |
|---|---|---|
| **Hardware** | 2× DGX Spark (GB10, 128 GB unified), CX-7 ports cabled back-to-back | 1× DGX Spark; checkpoint on local NVMe (the PLE table is read through the page cache) |
| **Launcher** | sparkrun ≥ 0.3.6 with a two-node cluster defined | sparkrun ≥ 0.3.6 |
| **Disk** | ~125 GB per node (98.5 GiB checkpoint + ~25 GB image) | ~125 GB |
| **Kernel** | `6.17.0-1032-nvidia`. `7.0.0-1019-nvidia` breaks NCCL `ibv_reg_mr` past ~85 GB GPU-resident ([forum](https://forums.developer.nvidia.com/t/dgx-spark-regression-kernel-7-0-0-1019-nvidia-causes-nccl-roce-ibv-reg-mr-iova2-enomem-6-17-0-1032-works/383023)) | No NCCL at TP=1 |
| **Host setting** | `loginctl enable-linger nvidia` on both nodes (otherwise logind `RemoveIPC` kills the shm ring buffer) | — |
| **Boot** | ~4 min warm, ~9.5 min cold | not yet measured with the shipped plan seed |
| **Concurrency** | `max_num_seqs` 16 | `max_num_seqs` 8, KV pool 6 GiB |

<!-- TODO: 1x kernel/linger requirements, warm and seeded cold-boot time, and KV token count (~379k vs ~350k: README and recipe header disagree). -->

Checkpoint: [`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)
@ `7c4f1bc1` (NVFP4 experts; MXFP8 dense, GDN and attention; 98.5 GiB). Image and video input are not tested on this build.

## Known limits

- **1× Spark, 8 concurrent requests at 16k context: ~20 tok/s.** The 6 GiB KV pool fills and requests are deferred.
  Keep long-context concurrency at 4 or less on one Spark. A fix is in progress.
- **1× Spark, first boot without a usable plan seed** autotunes and compiles every kernel: ~30 min, with host
  MemAvailable down to 3.8 GiB for about a minute (earlyoom triggers at ~2.4 GiB). The image ships the TP=1 plan seed
  and compile cache, so a normal first boot skips this. Close other memory-heavy work during the first boot.
- **1× Spark steady state:** MemAvailable 13–14 GiB, most of it PLE page cache.
- **2× Spark, c1 is bimodal** (counting ~95–100 vs ~85–88 tok/s between boots); we report medians of several runs.
- Hardmode still fails a few multi-step scenarios (e.g. TC-30, TC-68, TC-74, TC-88) on every build.
- 64k-depth and prose throughput were not re-measured on b1.4.

## Recipes

| Recipe | Image | Use |
|---|---|---|
| [`qwen3.8-flash-next-2x-dgx-spark`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml) | `b1.4-20261001-b7fbaf96-a7e649d8-warm` | 2× Spark default |
| [`qwen3.8-flash-next-2x-dgx-spark-previous`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark-previous.yaml) | `b1.3-20260929-b7fbaf96-7344a997-warm` | Rollback: full draft vocab, no QSA race fix, thinking effort `xhigh` |
| [`qwen3.8-flash-next-1x-dgx-spark`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml) | `tp1-v3a-20261004-5bf24021-7fa812b3-warm` | 1× Spark, experimental |
| [`qwen3.8-flash-next-1x-dgx-spark-previous`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-previous.yaml) | `tp1-20261002-5bf24021-884b4ff6-warm` | Rollback: v2, no NVMe keepalive |

Each pair shares the runtime cache, so a rollback boots warm. Renames: [recipes/RENAMES.md](recipes/RENAMES.md).

## Changelog

Promoted builds only. Deltas are from that build's own A/B against the previous one; paired-probe figures are decode
step time at T=0 with 95% CIs. Every row passed the gate.

<img src="docs/img/build-history.svg" alt="2× Spark tok/s per promoted build from the 2026-09-23 shipped build to b1.4: counting at 1 request 101 to 120, at 8 requests 444 to 525; coding at 1 request 53 to 62, at 8 requests 167 to 196." width="900">

**2× Spark** (same checkpoint throughout)

| Build | Date | Coding c1 d0 | Coding c16 16k | TTFT c1 16k | What changed | Measured delta |
|---|---|:---:|:---:|:---:|---|---|
| b1 | 2026-09-25 | 54 | 139 | 2.6 s | Deferred GDN checkpoints, TC-45 fix | d0 c8 186.6 vs 166.8, c16 241.1 vs 218.6 |
| b1.1 | 2026-09-26 | 53.4 | 175.9 | — | Exact prefix hits under MTP, `NULL_BLOCK_ID` padding fix, compile-worker cap | 16k c16 176.5 vs 139.4; cached-prefix TTFT 16k −37% |
| b1.2 | 2026-09-27 | 58.2–61.8 | 173.3–176.1 | — | HC mixers in online MXFP8 | step −9.9% c1 fresh, −4.4 to −7.6% c2–c4 |
| b1.3 | 2026-09-29 | 62.2 | 178.5 | 1.71 s | GDN uniform-decode metadata skip (~350 launches/step) | step −1.9 to −3.1% c1, −2.5 to −3.0% c2 |
| **b1.4** | **2026-10-01** | **62.2** | **176.4** | **1.70 s** | 131k-id MTP draft vocab, QSA race + recompile fixes, thinking effort `medium` | step −2 to −5% c1–c8; d0 c5 +4.2%; counting c1 +4.9% |

**1× Spark**

| Build | Date | Coding c1 d0 | Coding c8 d0 | What changed | Measured delta |
|---|---|:---:|:---:|---|---|
| v2 | 2026-10-02 | 47.4 | 120.0 | PLE WILLNEED before each decode gather; 131k-id draft vocab | WILLNEED: step −16.9% fresh c1, −8 to −9% c2–c8; draft vocab: step −2.7 to −4.3% c1–c4 |
| **v3a** | **2026-10-04** | **50.0** | **117.6** | NVMe keepalive (`VLLM_PLE_MMAP_KEEPALIVE_MS=50`), compile cache in the image | step −8.4% c4, −5.1% c8, −8.0% 16k c4; c1/c2 −0.5 to −0.9% |

The benchy grid varies by up to ~10% between runs, so single cells (e.g. v3a c8 117.6 vs v2 120.0) do not resolve
differences this small; the paired probe does. Per-build tables: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## How we measure

- **Grid.** llama-benchy task mode (above) at c1–c16 and depths 0 / 16k, 3 runs per boot, two boots per build on the
  2× Spark. A cell counts as changed only if the difference is larger than its own boot-to-boot noise.
- **Paired A/B.** Candidate and previous build on the same prompts at temperature 0, booted in ABBA order (2× Spark:
  two boots per build; 1× Spark: 4 passes over 2 Sparks). We report decode step time, tokens per step and tok/s per cell with 95% CIs.
  Example: [`results/tp1-v3a-20261004/ab-report-v3a-vs-v2.txt`](results/tp1-v3a-20261004/ab-report-v3a-vs-v2.txt).
  <!-- TODO: commit the script that produces these reports. -->
- **Promotion rule.** The gate must pass; at least one coding or counting cell must be faster beyond noise; no c1–c4
  cell may be slower beyond noise, and a loss at c5–c16 is published as a caveat ([`scripts/arm_verdict.py`](scripts/arm_verdict.py)).
- **Numerics canary.** MTP acceptance per draft position is compared cell by cell. A drop means the target or draft
  numerics changed, even when the gate passes. Logprob agreement against the previous build must sit within self-noise.
- **High-acceptance rows** (counting, copy-streams) bound what MTP can give, not coding speed.

Index of every run and verdict: [results/README.md](results/README.md).

## How it works

- **2× Spark: TP=2 over RoCE.** One rank per GB10; all-reduces over the CX-7 link take ~4% of a c1 decode step.
- **1× Spark: PLE table through the page cache.** The 26.8 GiB PLE n-gram table is read from the checkpoint files
  (`VLLM_PLE_MMAP=1`) with a WILLNEED pass before each decode gather and a 50 ms NVMe keepalive, so ~72 GiB of other
  weights plus a 6 GiB KV pool fit on one GB10.
- **b12x kernels** for NVFP4 MoE, MXFP8 linears, GDN (36 layers) and QSA sparse attention (12 layers), with an
  autotuned plan cache baked into each image.
- **MTP ×4 with probabilistic drafts** over a 131k-id draft vocabulary. Rejection sampling keeps the output
  distribution unchanged.
- **Where the 2× Spark c1 step (~43 ms) goes:** MoE 30%, dense MXFP8 31%, MTP draft + head 19%, idle 6%,
  all-reduce 4%, GDN 3%. MoE reads ~175 GB/s of the ~250 GB/s the GB10 reaches.

Flags, environment variables and the reason for each: [docs/REFERENCE.md](docs/REFERENCE.md). Kernel and engine notes,
rejected experiments: [docs/ENGINEERING.md](docs/ENGINEERING.md).

## Thinking effort

Both recipes set the server default to `reasoning_effort: medium`. On our DevOps task set (2× Spark), b1.2 at the
template default `xhigh` passed 42.5% of checks with 23/42 runaway-thinking runs and a 246 s median per task; b1.4 at
`medium` passed 95.9% with 0/42 runaway and 33 s. A request can still ask for `xhigh` or `low`; `"reasoning_effort": "none"`
turns thinking off. Override table and measurements: [docs/REFERENCE.md](docs/REFERENCE.md#thinking-effort).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` silent for minutes | Expected on a cold boot. `sparkrun logs <recipe> -f` |
| OOM / earlyoom at start (2× Spark) | Unified memory: `gpu_memory_utilization` ≥ 0.84 starves host RAM. Keep 0.80 and clear other containers |
| OOM / earlyoom at first boot (1× Spark) | Plan seed not used, so it autotunes; see [Known limits](#known-limits) |
| NCCL `ibv_reg_mr_iova2 ... Cannot allocate memory` | Kernel `7.0.0-1019-nvidia`; hold `6.17.0-1032-nvidia` |
| `ShmRingBuffer ... shared_memory` crash | `loginctl enable-linger nvidia` on both nodes |
| Garbled output | Rank checkpoint mismatch: both serve logs must show `snapshots/7c4f1bc1` ([why](docs/REFERENCE.md#tuning-and-troubleshooting)) |
| Empty `content`, long reasoning | `max_tokens` ran out during thinking; raise it or send `"reasoning_effort": "low"` |
| Old build boots after an upgrade | `sparkrun registry update qwen38-flashnext` |
| Anything else | Try the `-previous` recipe, then open an issue with `sparkrun logs <recipe> -a` |

## Docs

| | |
|---|---|
| [docs/REFERENCE.md](docs/REFERENCE.md) | Configuration, image provenance, thinking effort, tuning, known limits |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Full benchmark and quality tables for every build |
| [docs/ENGINEERING.md](docs/ENGINEERING.md) | Known issues and fixes, profiling, rejected experiments |
| [results/](results/README.md) | Every raw measurement and verdict |
| [archive/](archive/recipes/README.md) | Experimental and superseded recipes (not listed by sparkrun) |

## Credits

This build stands on these projects:

- [local-inference-lab](https://github.com/local-inference-lab): the vLLM fork our branches start from, the b12x kernels (NVFP4 MoE, GDN, QSA) and the NVFP4 checkpoint.
- [eugr](https://github.com/eugr): spark-vllm-docker (our image base), sparkrun (the launcher), and llama-benchy (the base of our benchmark fork).
- [tonyd2wild](https://github.com/tonyd2wild): the bench_sweep counting harness behind the counting numbers.
- [SeraphimSerapis](https://github.com/SeraphimSerapis): tool-eval-bench, which runs the hardmode and TC-45 quality gate.

Earlier experiments drew on other projects too, and [docs/ENGINEERING.md](docs/ENGINEERING.md#credits) keeps that full history with exact pins.

## License

Apache-2.0, see [`LICENSE`](LICENSE). The vLLM overlays under `archive/mods/` keep their upstream Apache-2.0 headers.
