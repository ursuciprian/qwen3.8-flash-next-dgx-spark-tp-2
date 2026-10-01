<div align="center">

# Qwen3.8-Flash-Next · 2× DGX Spark · TP=2

**NVFP4 Qwen3.8-Flash-Next served over ConnectX-7 with vLLM V2, b12x kernels and 4-token MTP.**<br>
62 tok/s at c1 on agentic coding turns, 243 tok/s aggregate at c16, 262k context, OpenAI-compatible.

[![Build](https://img.shields.io/badge/build-b1.4%20·%202026--10--01-2ea44f)](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/releases/latest)
[![Hardware](https://img.shields.io/badge/hardware-2×%20GB10%20·%20TP%3D2-76b900)](#requirements)
[![Engine](https://img.shields.io/badge/engine-vLLM%20V2%20+%20b12x-blue)](docs/REFERENCE.md#what-is-in-the-image)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)

[Quick start](#quick-start) · [Performance](#performance) · [Quality gate](#quality-gate) · [Thinking effort](#thinking-effort) · [Recipes](#recipes) · [Troubleshooting](#troubleshooting) · [Docs](#docs)

</div>

---

## At a glance

| | |
|---|---|
| **Checkpoint** | [`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) @ `7c4f1bc1`: NVFP4 experts, MXFP8 dense/GDN/attention, 98.5 GiB |
| **Topology** | TP=2 across two GB10s, RoCE all-reduce over CX-7, `gpu_memory_utilization 0.80` |
| **Decode** | MTP ×4 with probabilistic drafts over a 131k-id draft vocab; per-position acceptance ~0.75 / 0.58 / 0.45 / 0.37 on fresh code |
| **Throughput** | 62.2 tok/s c1 · 151 c4 · 196 c8 · 243 c16 (task-mode coding, T=1.0, thinking on) |
| **Latency** | TTFT 0.75 s (2k fresh) / 1.70 s (16k cached depth) at c1 |
| **Context** | 262,144 tokens, fp8 KV, prefix caching incl. GDN state (`--mamba-cache-mode align`) |
| **API** | OpenAI-compatible on :8000; `qwen3` reasoning parser, `qwen3_xml` tool parser, auto tool choice |
| **Thinking** | `reasoning_effort` defaults to `medium`; a request can ask for `xhigh` or `low` ([details](#thinking-effort)) |
| **Boot** | ~4 min warm, ~9.5 min cold; b12x plan seed baked into the image, no autotune on first boot |

## Quick start

**1. Register and launch** with [sparkrun](https://github.com/eugr/sparkrun):

```sh
sparkrun registry add https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2
sparkrun run qwen3.8-flash-next-2x-dgx-spark        # default cluster, or --hosts <head>,<worker>
```

**2. Wait for health.** `sparkrun run` returns before the engine is ready:

```sh
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://<head>:8000/health)" = 200 ]; do sleep 10; done
```

**3. Smoke test:**

```sh
curl -s http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Write a Python function that reverses a string."}]}'
```

Base URL `http://<head>:8000/v1`, model `qwen3.8-flash-next`, no API key. Stop it with `sparkrun stop --all`.

> **Upgrading:** sparkrun caches registries and does not refresh them on `run`. Run `sparkrun registry update qwen38-flashnext`
> first, or the previous recipe revision boots.

### Requirements

| | |
|---|---|
| **Hardware** | 2× DGX Spark (GB10, 128 GB unified), CX-7 ports cabled back-to-back |
| **Launcher** | sparkrun ≥ 0.3.6 with a two-node cluster defined |
| **Disk** | ~125 GB per node (98.5 GiB checkpoint + ~25 GB image) |
| **Kernel** | `6.17.0-1032-nvidia`. `7.0.0-1019-nvidia` breaks NCCL `ibv_reg_mr` past ~85 GB GPU-resident ([forum](https://forums.developer.nvidia.com/t/dgx-spark-regression-kernel-7-0-0-1019-nvidia-causes-nccl-roce-ibv-reg-mr-iova2-enomem-6-17-0-1032-works/383023)) |
| **Host setting** | `loginctl enable-linger nvidia` on both nodes (otherwise logind `RemoveIPC` kills the shm ring buffer) |

## Performance

[llama-benchy](https://github.com/ursuciprian/llama-benchy) `--prompt-mode task`: 2,048 new prompt tokens, up to 512 out,
thinking on, T=1.0 / top-p 0.95 / top-k 20, prefix caching on. Means of two boots × 3 runs. Per-request is the mean decode rate of one stream, so it exceeds aggregate / c.

| Concurrency | Depth 0: aggregate · per request | Depth 16k: aggregate · per request |
|:---:|:---:|:---:|
| c1 | **62.2** · 62.2 | **63.8** · 63.8 |
| c2 | 105.1 · 54.6 | 91.9 · 51.3 |
| c4 | 150.9 · 40.5 | 117.5 · 37.8 |
| c8 | 195.5 · 28.5 | 145.9 · 25.3 |
| c16 | **242.8** · 19.6 | 176.5 · 16.5 |

| TTFT (end-to-end) | Depth 0 | Depth 16k |
|:---:|:---:|:---:|
| c1 | **0.75 s** | **1.70 s** |
| c16 | 6.4 s | 14.0 s |

**Upper bound** (counting, T=0, nearly every draft accepted): 120 tok/s c1 · 357 c4 · 525 c8 · 778 c16.

**Build history**, same checkpoint, all gates passing:

| Build | c1 depth 0 | c16 depth 16k | TTFT c1 depth 16k | What changed |
|---|:---:|:---:|:---:|---|
| 2026-09-25 | 54 | 139 | 2.6 s | Deferred GDN checkpoints, TC-45 fix |
| 2026-09-26 | 53.4 | 175.9 | — | Exact prefix hits under MTP, `NULL_BLOCK_ID` padding fix, compile-worker cap |
| 2026-09-27 | 58.2–61.8 | 173.3–176.1 | — | HC mixers in online MXFP8 |
| 2026-09-29 | 62.2 | 178.5 | 1.71 s | GDN uniform-decode metadata skip (~350 launches/step) |
| **2026-10-01** | **62.2** | **176.4** | **1.70 s** | 131k-id MTP draft vocab (decode step −2 to −5%), QSA race + recompile fixes, default thinking effort `medium` |

Each row is that build's own A/B; c1 and 16k cells swing ±5–10% between boots. Against b1.3 on the same day, b1.4 is
faster beyond noise at coding depth-0 c5 (+4.2%) and counting c1/c2/c5 (+4.9/+3.1/+4.4%), with no cell slower beyond noise.
Per-build tables, paired decode-step probes and raw JSON: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Quality gate

A build ships only if it passes all of these. A faster build that fails any check is rejected.

| Check | Tool | b1.4 |
|---|---|---|
| Hard multi-step tool use (88 scenarios, thinking on, T=0) | [tool-eval-bench](https://github.com/SeraphimSerapis) `--hardmode` | 92 / 92 (band 86–93; b1.3 88–90) |
| `tool_choice=required` compliance | TC-45 | 5 / 5 |
| Long-context retrieval, 20 tool-call needles | `scripts/fidelity_probe.py` | 20/20 at 8k / 32k / 64k / 128k (re-gate boot; the A/B boot had one 32k miss, not reproduced in 20 cold trials); 32k seeds 21, 22 and 128k seeds 11, 13 20/20 |
| Batch stragglers, c5–c16 | `scripts/straggler_probe.py` | none |
| Numerics vs previous build (20 prompts × 16 tokens) | `scripts/logits_equiv.py` | mean \|Δlogprob\| 0.033–0.041 vs 0.038–0.046 self-noise |
| MTP acceptance per position | paired decode probe | within ±0.03 of b1.3 |

## Thinking effort

The chat template knows three efforts: `xhigh` (its own default), `medium` and `low`; any other value is an HTTP 400
(vLLM's top-level `"reasoning_effort": "none"` switches thinking off instead).
At `xhigh` it adds a "think carefully, validate key assumptions" system sentence, and on prompts that say "must pass
`terraform validate`" the model keeps re-checking its draft until the token budget runs out. **This recipe sets the
server default to `medium`** (`--default-chat-template-kwargs '{"reasoning_effort":"medium"}'`), which adds no sentence.

DevOps task set (14 prompts × 3 runs: Terraform, Kubernetes, GitHub Actions, IAM, bash, Helm, Dockerfile, Prometheus,
incident triage; graded by terraform/kubeconform/actionlint/shellcheck/helm/hadolint/promtool plus rubric checks;
T=1.0, max 16,384 tokens):

| Effort | Build | Mean checks passed | Clean runs | Runaway thinking | Median time per task |
|---|---|:---:|:---:|:---:|:---:|
| `xhigh` (template default) | b1.2 | 42.5% | 17 / 42 | 23 / 42 | 246 s |
| **`medium` (this recipe)** | **b1.4** | **95.9%** | **29 / 42** | **0 / 42** | **33 s** |
| `medium`, one Spark (TP=1) | same weights | 97.8% | 35 / 42 | 0 / 42 | 105 s |

To think harder on one request, send `reasoning_effort` (top level) or `chat_template_kwargs`:

```sh
curl -s http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next", "reasoning_effort": "xhigh",
  "messages": [{"role": "user", "content": "Prove that the square root of 2 is irrational."}]}'
# or: "chat_template_kwargs": {"reasoning_effort": "xhigh"}
```

Both fields override the server default, checked live on the promoted server (3 short prompts × 2 runs per request,
server default sampling). `prompt_tokens` shows which effort the template applied: `xhigh` and `low` add a system
sentence, `medium` adds none.

| Request | Effort applied | Prompt tokens | Thinking tokens (median) |
|---|---|:---:|:---:|
| no effort field | `medium` (server default) | 30 / 28 / 44 | 58 / 357 / 185 |
| `"reasoning_effort": "medium"` | `medium` | 30 / 28 / 44 | 56 / 483 / 202 |
| `"reasoning_effort": "xhigh"` | `xhigh` | 72 / 70 / 86 | 73 / **6,144** / 67 |
| `"chat_template_kwargs": {"reasoning_effort": "xhigh"}` | `xhigh` | 72 / 70 / 86 | 73 / 1,776 / 81 |
| top-level `"low"` + `chat_template_kwargs` `"xhigh"` | `low`: the top-level field wins | 60 / 58 / 74 | 55 / 321 / 125 |
| `"reasoning_effort": "none"` | thinking off | 32 / 30 / 46 | 0 |
| `"reasoning_effort": "high"` | HTTP 400, `Unexpected reasoning effort high` | | |

The 6,144 is the request's `max_tokens`: both top-level `xhigh` runs of the "5 largest files" bash prompt thought until
the cap and never answered, the same runaway the `medium` default avoids. Script and raw rows:
[`results/b1.4-20261001/effort-override/`](results/b1.4-20261001/effort-override/).

Not yet measured at `medium`: MMLU-Pro, GSM8K, IFEval and LiveCodeBench, where long thinking may still pay off; ask for
`xhigh` there. `-previous` (b1.3) keeps the template default `xhigh`. [Eval details](results/evals-20260928/README.md),
[b1.4 run](results/b1.4-20261001/devops-b14-medium.md).

## Recipes

| Recipe | Image | Use |
|---|---|---|
| [`qwen3.8-flash-next-2x-dgx-spark`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml) | `b1.4-20261001-b7fbaf96-a7e649d8-warm` | Default |
| [`qwen3.8-flash-next-2x-dgx-spark-previous`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark-previous.yaml) | `b1.3-20260929-b7fbaf96-7344a997-warm` | Rollback: same checkpoint, full draft vocab, no QSA race fix, thinking effort `xhigh` by default |

Both share the runtime cache, so a rollback boots warm. Renames: [recipes/RENAMES.md](recipes/RENAMES.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` silent for minutes | Expected on a cold boot (~9.5 min). `sparkrun logs qwen3.8-flash-next-2x-dgx-spark -f` |
| OOM / earlyoom at start | Unified memory: `gpu_memory_utilization` ≥ 0.84 starves host RAM. Keep 0.80 and clear other containers |
| NCCL `ibv_reg_mr_iova2 ... Cannot allocate memory` | Kernel `7.0.0-1019-nvidia`; hold `6.17.0-1032-nvidia` |
| `ShmRingBuffer ... shared_memory` crash | `loginctl enable-linger nvidia` on both nodes |
| Garbled output | Rank checkpoint mismatch: both serve logs must show `snapshots/7c4f1bc1` ([why](docs/REFERENCE.md#tuning-and-troubleshooting)) |
| Old build boots after an upgrade | `sparkrun registry update qwen38-flashnext` |
| Anything else | Try `-previous`, then open an issue with `sparkrun logs qwen3.8-flash-next-2x-dgx-spark -a` |

## How it works

- **TP=2 over RoCE.** One rank per GB10. All-reduces go over the CX-7 link through vLLM's RoCE path (≤2 MB messages)
  and take ~4% of a c1 decode step.
- **b12x kernels** for NVFP4 MoE, MXFP8 linears, GDN (36 layers) and QSA sparse attention (12 layers), with an autotuned
  plan cache baked into the image.
- **MTP ×4, probabilistic drafts.** Probabilistic draft sampling keeps acceptance high at T=1.0. Exact prefix-cache hits
  under MTP (`VLLM_PREFIX_DROP_EXACT`) and deferred GDN checkpoints make cached turns cheap.
- **Where the c1 step (~43 ms) goes:** MoE 30%, dense MXFP8 31%, MTP draft + head 19%, idle 6%, all-reduce 4%, GDN 3%.
  MoE reads ~175 GB/s of the ~250 GB/s the GB10 can reach, so decode is close to bandwidth-bound.

Flags, environment variables and the reason for each: [docs/REFERENCE.md](docs/REFERENCE.md).

## Docs

| | |
|---|---|
| [docs/REFERENCE.md](docs/REFERENCE.md) | Configuration, image provenance, tuning, known limits |
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
