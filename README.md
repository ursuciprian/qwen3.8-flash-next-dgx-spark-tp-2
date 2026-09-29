<div align="center">

# Qwen3.8-Flash-Next · 2× DGX Spark · TP=2

**NVFP4 Qwen3.8-Flash-Next served over ConnectX-7 with vLLM V2, b12x kernels and 4-token MTP.**<br>
62 tok/s at c1 on agentic coding turns, 241 tok/s aggregate at c16, 262k context, OpenAI-compatible.

[![Build](https://img.shields.io/badge/build-b1.3%20·%202026--09--29-2ea44f)](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/releases/latest)
[![Hardware](https://img.shields.io/badge/hardware-2×%20GB10%20·%20TP%3D2-76b900)](#requirements)
[![Engine](https://img.shields.io/badge/engine-vLLM%20V2%20+%20b12x-blue)](docs/REFERENCE.md#what-is-in-the-image)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)

[Quick start](#quick-start) · [Performance](#performance) · [Quality gate](#quality-gate) · [Recipes](#recipes) · [Troubleshooting](#troubleshooting) · [Docs](#docs)

</div>

---

## At a glance

| | |
|---|---|
| **Checkpoint** | [`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) @ `7c4f1bc1`: NVFP4 experts, MXFP8 dense/GDN/attention, 98.5 GiB |
| **Topology** | TP=2 across two GB10s, RoCE all-reduce over CX-7, `gpu_memory_utilization 0.80` |
| **Decode** | MTP ×4 with probabilistic drafts; per-position acceptance ~0.75 / 0.57 / 0.43 / 0.35 on fresh code |
| **Throughput** | 62.2 tok/s c1 · 147 c4 · 188 c8 · 241 c16 (task-mode coding, T=1.0, thinking on) |
| **Latency** | TTFT 0.74 s (2k fresh) / 1.71 s (16k cached depth) at c1 |
| **Context** | 262,144 tokens, fp8 KV, prefix caching incl. GDN state (`--mamba-cache-mode align`) |
| **API** | OpenAI-compatible on :8000; `qwen3` reasoning parser, `qwen3_xml` tool parser, auto tool choice |
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
| c1 | **62.2** · 62.2 | **63.5** · 63.5 |
| c2 | 101.8 · 52.3 | 85.7 · 47.5 |
| c4 | 147.3 · 39.2 | 113.8 · 36.5 |
| c8 | 188.0 · 27.9 | 143.3 · 24.6 |
| c16 | **241.1** · 19.1 | 178.5 · 16.3 |

| TTFT (end-to-end) | Depth 0 | Depth 16k |
|:---:|:---:|:---:|
| c1 | **0.74 s** | **1.71 s** |
| c16 | 6.4 s | 14.0 s |

**Upper bound** (counting, T=0, nearly every draft accepted): 114 tok/s c1 · 326 c4 · 525 c8 · 754 c16.

**Build history**, same checkpoint, all gates passing:

| Build | c1 depth 0 | c16 depth 16k | TTFT c1 depth 16k | What changed |
|---|:---:|:---:|:---:|---|
| 2026-09-25 | 54 | 139 | 2.6 s | Deferred GDN checkpoints, TC-45 fix |
| 2026-09-26 | 53.4 | 175.9 | — | Exact prefix hits under MTP, `NULL_BLOCK_ID` padding fix, compile-worker cap |
| 2026-09-27 | 58.2–61.8 | 173.3–176.1 | — | HC mixers in online MXFP8 |
| **2026-09-29** | **62.2** | **178.5** | **1.71 s** | GDN uniform-decode metadata skip (~350 launches/step) |

Per-build tables, paired decode-step probes and raw JSON: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Quality gate

A build ships only if it passes all of these. A faster build that fails any check is rejected.

| Check | Tool | b1.3 |
|---|---|---|
| Hard multi-step tool use (88 scenarios, thinking on, T=0) | [tool-eval-bench](https://github.com/SeraphimSerapis) `--hardmode` | 88 / 90 (band 86–93; b1.2 90–92) |
| `tool_choice=required` compliance | TC-45 | 5 / 5 |
| Long-context retrieval, 20 tool-call needles | `scripts/fidelity_probe.py` | 20/20 at 8k / 32k / 64k / 128k; 128k seeds 11 and 13 20/20 |
| Batch stragglers, c5–c16 | `scripts/straggler_probe.py` | none |
| Numerics vs previous build (20 prompts × 16 tokens) | `scripts/logits_equiv.py` | mean \|Δlogprob\| 0.031–0.035 vs 0.037–0.042 self-noise |
| MTP acceptance per position | paired decode probe | unchanged |

> **Known weakness: runaway thinking.** On validator-gated DevOps prompts ("must pass `terraform validate`") the model
> re-checks its draft until the 16k thinking budget runs out: 23/42 runs never close `</think>`. Of the runs that answered,
> 17/19 were clean. `reasoning_effort` / `thinking_token_budget` mitigations are under test. [Eval details](results/evals-20260928/README.md).

## Recipes

| Recipe | Image | Use |
|---|---|---|
| [`qwen3.8-flash-next-2x-dgx-spark`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml) | `b1.3-20260929-b7fbaf96-7344a997-warm` | Default |
| [`qwen3.8-flash-next-2x-dgx-spark-previous`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark-previous.yaml) | `b1.2-20260927-b7fbaf96-a9aa81b2-warm` | Rollback: same checkpoint and flags, no GDN metadata skip |

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
