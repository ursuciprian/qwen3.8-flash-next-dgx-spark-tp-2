# Qwen3.8-Flash-Next NVFP4 on two DGX Sparks

Serving recipes for **`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`** at
tensor parallel 2 across two NVIDIA DGX Spark (GB10, SM121) over ConnectX-7
RoCE, packaged for [sparkrun](https://sparkrun.dev). The served route is vLLM
on a b12x-kernel fork, built as our own container image.

**Default recipe:** [`recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml)
**Default image:** `ghcr.io/ursuciprian/spark-vllm-b12x:b0-20260918-a8333658-warm` (local build `spark-vllm-b12x:local-20260918-a8333658` + [warm layer](docker/b0-warm/Dockerfile))

**Numbers:** the pinned-checkpoint gate below was measured 2026-09-23. Older
numbers come from fresh boots between 2026-09-18 and 2026-09-22, when rank 1
was loading the old checkpoint revision. They are marked **(hybrid)**; see
[Checkpoint revision split](#checkpoint-revision-split-fixed-2026-09-23). Each
number states its workload and source file. Every measurement and verdict:
[results/README.md](results/README.md). Full tables:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

**Contents:** [Winning recipe](#winning-recipe) · [At a glance](#at-a-glance) ·
[Quick start](#quick-start) · [Configuration](#configuration) ·
[Recipes](#recipes) · [Quality](#quality) ·
[How to read the numbers](#how-to-read-the-numbers) ·
[Known issues / limits](#known-issues--limits) · [Files](#files) ·
[Credits](#credits) · [License](#license)

---

## Winning recipe

[`recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml)
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

### Sampling settings

The server uses the checkpoint's generation defaults: **temperature 1.0,
top-p 0.95, top-k 20**. Agent clients that send no temperature (omp, pi,
hermes) get these defaults, and that is the setting the recipe is tuned for.
Temperature 0.6 measured the same speed within noise. Temperature 0 is only
used for the counting diagnostic, not for real work.

### Quality gate on the pinned checkpoint (2026-09-23)

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

### Checkpoint revision split (fixed 2026-09-23)

sparkrun runs vLLM with `HF_HUB_OFFLINE=1`, so each node loads whatever its own
`refs/main` points at. sparkrun copies the model head-to-worker with
`rsync --size-only`, and a 40-byte commit hash always has the same size, so the
worker's `refs/main` was never updated. From at least 2026-09-18 until
2026-09-23 14:12 EEST, rank 0 loaded the QAD revision `7c4f1bc1` and rank 1
loaded the old PTQ revision `ada4da32`. Every number measured in that window
is labelled (hybrid) below: the agent coding and prose grids, the counting
ceilings, the old-vs-new comparison, and the la-family quality gates. The
09-17 numbers were probably affected too. The recipes now pin the revision;
details are in [docs/ENGINEERING.md](docs/ENGINEERING.md#known-issues--fixes).

---

## At a glance

### Agent coding task, default temperature (primary number) (hybrid)

Default recipe (probabilistic MTP drafts), measured 2026-09-21/22 with rank 1 on `ada4da32` (hybrid).
Source: [`results/benchy/la-mtpprob-task16.md`](results/benchy/la-mtpprob-task16.md).

Workload: our llama-benchy fork
([ursuciprian/llama-benchy](https://github.com/ursuciprian/llama-benchy)
`0d4de42`, `--prompt-mode task --no-force-length`, command shape in
[`scripts/run_b_arm.sh`](scripts/run_b_arm.sh), `--concurrency 1 4 10 16`).
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

### Other workloads (not comparable with the table above) (hybrid)

| Workload | Headline | Source |
|---|---|---|
| Agent coding, **temperature 0** (old argmax config only; not measured on the default recipe) | c1 55.3, c16 217.7 agg tok/s | [`results/benchy/la-task16-t0.md`](results/benchy/la-task16-t0.md) |
| Agent coding, temperature 0.6 | same speed as temperature 1.0 within noise (2026-09-23, B0, (hybrid)) | `results/benchy/` task-t06 run on dgx-01 |
| **Prose continuation**, default recipe, default temperature (1.0), 128 output tokens | c1 56.05 ± 2.56, c16 126.72 ± 1.56 agg tok/s; 64k-cached c16 36.78 ± 2.68 | [`results/benchy/la-mtpprob-prose16.md`](results/benchy/la-mtpprob-prose16.md) |
| **Counting ceiling — not user throughput** (bench_sweep "List the numbers from 1 to 300", temp 0, thinking off) | c1 100.9 / 102.1 / 101.8, c8 439.4 / 434.6, c16 641.3 / 635.0 agg tok/s | `results/arms/la-mtpprob/decode_c*.json`, `sweep.json` (dgx-01) |

The counting ceiling measures how fast the stack goes when speculation never
misses (~3.96-4.00 of 4 drafts accepted). It is a regression diagnostic,
**not** the speed of coding, chat or agent work. Full prose grid, counting
table and old-vs-new comparisons: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### Default vs previous default (old la, one-hot argmax drafts) (hybrid)

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
[`results/benchy/`](results/benchy/)).

---

## Quick start

On a sparkrun cluster of two DGX Sparks (default cluster, or add `--cluster <name>`):

```sh
sparkrun registry add https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2
sparkrun run qwen3.8-flash-next-nvfp4-tp2
```

Or from a clone: `sparkrun run recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml`.

The server answers on port **8000** with the OpenAI API, model name
**`qwen3.8-flash-next`**. Image digest:
`ghcr.io/ursuciprian/spark-vllm-b12x@sha256:a3d5d90d1312edc9a79c86add6fdf72d50b2a73558e295fdf4ea110488fb615d`. Nothing else to place by hand: sparkrun pulls the image
(`ghcr.io/ursuciprian/spark-vllm-b12x:b0-20260918-a8333658-warm`) and downloads the
checkpoint pinned at revision `7c4f1bc1` on both nodes. The b12x kernel-selection
cache ships in the image, and compile caches persist in
`~/.cache/sparkrun/runtime-cache/vllm/`. The recipe has no hooks or mounts, so it
needs no `--trust`.

### Prerequisites

| Need | Detail |
|---|---|
| Hardware | 2x DGX Spark: GB10, SM121, 128 GB LPDDR5X unified memory |
| Link | ConnectX-7 RoCE between the nodes |
| Launcher | [sparkrun](https://github.com/eugr/sparkrun) 0.3.6, cluster of both nodes set up |
| Disk | ~100 GB for the checkpoint and ~25 GB for the image, per node |
| Hosts | `loginctl enable-linger nvidia` on both nodes (see [Known issues](#known-issues--limits)) |

> [!IMPORTANT]
> **Boot times (measured 2026-09-23 on the pair, from a fresh clone).** Cold boot,
> with no runtime cache and the image not yet on either node: **~9 min** from
> `sparkrun run` to `/health` 200. That covers the ~108 s image pull and sync, then
> about 7 min of serve start, of which the kernel compilation is most. The kernel
> selection itself ships in the image, so no autotuning runs (0 measured).
> A warm restart takes **~3.8 min** with 0 kernel compilations. The weights
> (~100 GB) were already present in this test, so a truly fresh pair also pays
> for the checkpoint download. `sparkrun run` may return before the server is
> ready, so poll `http://<head>:8000/health`.

---

## Configuration

Key serving flags and env of the default recipe, from
[`qwen3.8-flash-next-nvfp4-tp2.yaml`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml).

Image identity: eugr `spark-vllm-docker` Dockerfile `798528a2` + fork
`local-inference-lab/vllm` `dev/jovian-judgement` `8e1f1e58` + b12x
`a8333658`. Checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` QAD
revision `7c4f1bc1`.

### Serving flags

| Flag | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | `2` | One rank per Spark |
| `--speculative-config` | `method: mtp`, `num_speculative_tokens: 4`, `draft_sample_method: probabilistic` | MTP width 4; probabilistic drafts keep acceptance up for default-temperature (1.0) clients |
| `--kv-cache-dtype` | `fp8` | fp8 KV |
| `--quantization` | `modelopt_mixed` | NVFP4 checkpoint |
| `--load-format` | `b12x` | b12x weight loader |
| `--gdn-decode-kernel` / `--linear-backend` / `--moe-backend` | `b12x` | b12x kernels for GDN, linear and MoE |
| `--enable-prefix-caching` | on | Reuses cached context across turns |
| `--mamba-cache-mode` | `align` | GDN (mamba) state cache mode |
| `--enable-chunked-prefill`, `--max-parallel-prefills` | on, `4` | Bounded concurrent prefills |
| `--prefill-policy` / `--decode-refill-target` | `decode-aware` / `auto` | Keeps a decode lane instead of letting long prefills starve decode |
| `--max-model-len` | `262144` | Per-request context ceiling |
| `--max-num-seqs` | `16` | Concurrent sequences |
| `--max-num-batched-tokens` | `8192` | Prefill tokens per step |
| `--block-size` | `16` | KV block size |
| `--gpu-memory-utilization` | `0.80` | Share of unified memory for weights + KV |
| `--reasoning-parser` / `--tool-call-parser` | `qwen3` / `qwen3_xml`, `--enable-auto-tool-choice` | Thinking and tool calls |
| `--compilation-config` | `fuse_act_quant: true` | Fused activation quant pass |
| `--no-enable-flashinfer-autotune` | set | FlashInfer autotune off; b12x kernel tuning is `B12X_AUTOTUNE` |
| `--served-model-name` | `qwen3.8-flash-next` (+ HF id) | Model name clients send |

### Environment

| Variable | Value | Why |
|---|---|---|
| `B12X_AUTOTUNE` | `1` | eugr's Dockerfile sets `0`, which truncates kernel selection (81 vs 96-98 tok/s at c1 on the counting diagnostic) |
| `B12X_ROCE_SPIN_LIMIT` | `300000000` | ~300 s instead of ~20 s, so the TP2 MoE retune from an empty cache does not deadlock |
| `VLLM_MXFP8_LM_HEAD` | `1` | lm_head online MXFP8 (W8A16), halves bytes read by the verify-head lm_head matmul |
| `VLLM_ENABLE_ROCE_ALLREDUCE` / `VLLM_ROCE_ALLREDUCE_MAX_SIZE` | `1` / `2MB` | RoCE all-reduce between the two ranks |
| `B12X_POLICY_MODE` | `auto` | b12x kernel policy |
| `CUTE_DSL_ARCH` | `sm_121a` | GB10 target |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | V2 model runner |
| `VLLM_USE_AOT_COMPILE`, `VLLM_USE_MEGA_AOT_ARTIFACT` | `1` | AOT compile cache, warm boots |
| `VLLM_SSM_CONV_STATE_LAYOUT` | `DS` | GDN conv state layout |
| `SAFETENSORS_FAST_GPU` | `1` | Faster weight load |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Worker start method |

No mods at run time. The `b12x-startup-boundedwait` patch (bounded `Store.wait` in
TP2 preparation, fails fast instead of parking forever) is baked into the image
([`docker/b0-warm/Dockerfile`](docker/b0-warm/Dockerfile)).

---

## Recipes

| Recipe | Image | Status |
|---|---|---|
| [`qwen3.8-flash-next-nvfp4-tp2.yaml`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml) | `ghcr.io/ursuciprian/spark-vllm-b12x:b0-20260918-a8333658-warm` | **Default, serving.** Probabilistic MTP draft sampling (`draft_sample_method: probabilistic`) in the speculative config, startup robustness patch baked into the image, `B12X_AUTOTUNE=1`. |
| [`qwen3.8-flash-next-nvfp4-tp2-argmax-drafts.yaml`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-argmax-drafts.yaml) | same image | Fallback: previous default — one-hot drafts + `use_local_argmax_reduction: true` instead of probabilistic sampling. |
| [`qwen3.8-flash-next-nvfp4-tp2-local-build-seqs-16.yaml`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-local-build-seqs-16.yaml) | same image | Fallback: same recipe without `use_local_argmax_reduction`. |
| [`qwen3.8-flash-next-nvfp4-tp2-ghcr-image.yaml`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-ghcr-image.yaml) | `ghcr.io/ursuciprian/spark-vllm-b12x:wheels-20260919-77bdd10-a833365` (`sha256:c0314d7c…`) | Alternate pull-based image, gated (`ghcr2`, 272 cached): TC-45 passes for the first time on this stack, but bench_sweep counting diagnostic c1 is ~-10% vs `la` (87.7 vs 97.4). Not promoted. |

All recipes live in `recipes/qwen3.8-flash-next/`. Every one, including the
rejected and experimental arms, is tabled with status and a verdict pointer in
[recipes/README.md](recipes/README.md). Old `recipes/eugr/` names map to new
ones in [recipes/RENAMES.md](recipes/RENAMES.md).

---

## Quality

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
| TC-45 (`tool_choice=required`) | Parser bug, see [Known issues](#known-issues--limits) | Fixed by `mods/vllm-tc45-reasoning-structag-fix/` → 93/100, but that mod costs about -12% on the bench_sweep counting diagnostic at c1 (`results/arms/la-tc/sweep.json`, 85.0 vs 95.7). Not enabled by default. |
| TC-68 | Model wraps JSON in a code fence with commentary; the scenario intentionally sends no `response_format` | Model compliance, not a server bug — not fixable without defeating the test |

---

## How to read the numbers

Three workloads are used, and their numbers are not comparable with each other.

| Factor | What it means here |
|---|---|
| **Workload** | Agent coding task (primary), prose continuation, and the counting ceiling. The same engine shows ~100 tok/s on counting and ~45-58 on real single-stream work. |
| **Draft acceptance** | MTP proposes 4 tokens per step. On the counting prompt nearly all 4 are accepted (98.9% overall, ~3.96 per step, [`results/profiling/README.md`](results/profiling/README.md)). On the sampled (temp 1.0) prose and agent task grids with probabilistic drafts, 39-41% are accepted, ~1.6 per step. |
| **Temperature** | At temp 0 the target and draft agree more often, so acceptance and throughput rise (task c1 55.3 at temp 0 vs 46.1 at 1.0, old la). Clients that send no temperature get 1.0. |
| **Thinking tokens** | Task and prose runs have thinking on; reasoning tokens are generated and counted like answer tokens. The counting diagnostic and the category harness run with thinking off. |
| **Cached-context depth** | Depth runs prefill 2048 new tokens on top of a cached context; TTFT and per-token decode both slow with depth, and concurrency at depth queues prefills. |
| **c1 bimodality** | c1 is bimodal boot to boot on the counting diagnostic (fast mode 94-102, slow mode 84-88, [`results/arms/c1-decline/verdict.md`](results/arms/c1-decline/verdict.md)): report the median of >=5 sweeps and the max, not one run. |

Details and raw acceptance counts: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#how-to-read-these-numbers).

---

## Known issues / limits

| Issue | Status | Detail |
|---|---|---|
| Checkpoint revision split: rank 1 loaded `ada4da32` while rank 0 loaded `7c4f1bc1` (2026-09-18 to 09-23) | **Fixed**: recipes pin `model_revision` + `--revision` | sparkrun's `rsync --size-only` never refreshes the worker's `refs/main`; see [Checkpoint revision split](#checkpoint-revision-split-fixed-2026-09-23) |
| Batch 5-7 straggler: one request per round stalled ~18 s at c5-7 (also 9, 12) | **Fixed** by our own image | Fork's QSA/GDN scratch-isolation commits; `mods/vllm-qwen-scratch-isolation/` documents the fix |
| `B12X_AUTOTUNE=0` in eugr's Dockerfile: fresh boot 81 tok/s instead of 96-98 at c1 (counting diagnostic) | **Worked around**: recipe sets `1` | Plan cache persists at `~/.cache/sparkrun/runtime-cache/vllm/<model>/b12x/` (168-169 MB); [`results/kernel-pass/arms.md`](results/kernel-pass/arms.md) |
| logind `RemoveIPC` kills shm (`'ShmRingBuffer' object has no attribute 'shared_memory'`) | **Host fix** | `loginctl enable-linger nvidia` on both nodes |
| TP2 preparation hangs from an empty plan cache on b12x commits past `a8333658` | **Fixed in default recipe** | `B12X_ROCE_SPIN_LIMIT: "300000000"` + `mods/b12x-startup-boundedwait/`; [`results/kernel-pass/prep-deadlock/mechanism.md`](results/kernel-pass/prep-deadlock/mechanism.md) |
| TC-45: `tool_choice=required` silently unconstrained (shared Qwen3 `ParserEngine`) | **Open**, fix exists but off | `mods/vllm-tc45-reasoning-structag-fix/`, hardmode 93/100, costs speed (see [Quality](#quality)) |
| `scripts/gate_arm.sh` without `--hardmode` runs only 69 of 88 scenarios | **Usage** | Always pass `--hardmode` for a real promotion decision |
| Prose 64k-cached c16 regresses vs old la (38.15 -> 36.78) | **Known** | See the comparison in [At a glance](#default-vs-previous-default-old-la-one-hot-argmax-drafts) |
| Raw `results/arms/` files | **Not all mirrored** | Some live on dgx-01 only, see [results/README.md](results/README.md) |

Full write-ups (deadlock mechanism, parser detail, lm_head MXFP8):
[docs/ENGINEERING.md](docs/ENGINEERING.md#known-issues--fixes).
Profiling, rejected arms and build provenance also live in
[docs/ENGINEERING.md](docs/ENGINEERING.md).

---

## Files

| Path | What |
|---|---|
| [`recipes/qwen3.8-flash-next/`](recipes/qwen3.8-flash-next/) | The served recipe family: default, fallbacks, experimental and rejected arms. Status table: [recipes/README.md](recipes/README.md); old names: [recipes/RENAMES.md](recipes/RENAMES.md) |
| `recipes/arms/`, `recipes/dflash2/`, `recipes/retired/`, `recipes/flashnext-*.yaml` | Earlier SGLang-era recipes and bisection arms, kept for history; not the served route |
| [`mods/`](mods/) | Engine patches, one directory per mod; one-line summary of each in [mods/README.md](mods/README.md) |
| [`scripts/`](scripts/) | `gate_arm.sh` (full quality gate, needs `--hardmode`), `fidelity_probe.py`, `prof_summary.py`, `needle_ladder.py`, `decode_probe.py`, `validate_recipes.py`, `run.sh`, arm-bisection scripts (`vllm_ladder.sh`, `qwen_ladder*.sh`, …) |
| [`results/benchy/`](results/benchy/) | llama-benchy agent-task and prose grids and the multi-turn TTFT probe behind the headline tables |
| `results/kernel-pass/`, `results/profiling/`, `results/arms/` | The 2026-09-18/21 arms, hang evidence and per-kernel profile |
| `results/eugr-b12x/`, `results/fp8-gate/`, `results/sglang-*`, `results/vllm-cached-*` | Earlier (pre-09-18) SGLang/vLLM-nightly measurements, kept for history |
| [`results/README.md`](results/README.md) | Index of every measurement and verdict |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Full benchmark tables, comparisons, TTFT, single-stream probes |
| [`docs/ENGINEERING.md`](docs/ENGINEERING.md) | Profile, rejected arms, known-issue detail, build provenance |
| `misc/` | Working notes not promoted into a result file |

---

## Credits

The vLLM route served by default (`recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml`,
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
  release 0.4.0 measures the prose grids; the recipe family (formerly
  `eugr-agents`) started as his `eugr-agents.yaml`.
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

---

## License

Apache-2.0, see [`LICENSE`](LICENSE). The vLLM overlays under `mods/` are
modified copies of Apache-2.0 vLLM files and keep their upstream headers. The
spark-bench reports under `results/fp8-gate/` are output of that tool,
reproduced with its run labels intact.
