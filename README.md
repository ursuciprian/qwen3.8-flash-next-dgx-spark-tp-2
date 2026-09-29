<div align="center">

# Qwen3.8-Flash-Next on two DGX Sparks

**A fast, private coding model on your own desk.**<br>
Two NVIDIA DGX Sparks, one OpenAI-compatible endpoint, 62 tokens/s for one person and 241 tokens/s shared across sixteen.

[![Build](https://img.shields.io/badge/build-b1.3%20·%202026--09--29-2ea44f)](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/releases/latest)
[![Hardware](https://img.shields.io/badge/hardware-2×%20DGX%20Spark-76b900)](#what-you-need)
[![Engine](https://img.shields.io/badge/engine-vLLM%20+%20b12x-blue)](docs/REFERENCE.md#what-is-in-the-image)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)

[Quick start](#quick-start) · [Speed](#speed) · [Quality](#quality) · [Recipes](#recipes) · [Troubleshooting](#troubleshooting) · [Docs](#docs)

</div>

---

## At a glance

| | |
|---|---|
| **Model** | [Qwen3.8-Flash-Next](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4), 4-bit NVFP4, split across both Sparks |
| **One person** | **62 tokens/s** while coding, first words in **0.7 s** |
| **Sixteen people** | **241 tokens/s** in total |
| **Context** | 262,144 tokens per request (a mid-sized codebase) |
| **Works with** | Any OpenAI client, chat app or coding agent: tools, thinking and streaming |
| **Quality** | Checked on every build: tool use, long-context recall and batching ([details](#quality)) |
| **Start time** | ~4 minutes (~10 on the very first run) |

## Quick start

**1. Add this recipe to [sparkrun](https://github.com/eugr/sparkrun) and start it.**

```sh
sparkrun registry add https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2
sparkrun run qwen3.8-flash-next-2x-dgx-spark        # your default cluster, or --hosts <head>,<worker>
```

**2. Wait until it answers `200`.** `sparkrun run` returns before the model has loaded.

```sh
curl -s -o /dev/null -w '%{http_code}\n' http://<head>:8000/health
```

**3. Talk to it.**

```sh
curl -s http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Write a Python function that reverses a string."}]}'
```

Point any app at `http://<head>:8000/v1`, model `qwen3.8-flash-next`. No API key is needed. Stop it with `sparkrun stop --all`.

> **Upgrading?** sparkrun keeps its own copy of this registry. Run `sparkrun registry update qwen38-flashnext` before
> `sparkrun run`, or you get the build you had before.

### What you need

| | |
|---|---|
| **Hardware** | 2× DGX Spark (128 GB each), connected through their ConnectX-7 ports |
| **Launcher** | [sparkrun](https://github.com/eugr/sparkrun) 0.3.6+, with a cluster of both Sparks |
| **Disk** | ~125 GB free on each Spark (100 GB model + 25 GB image) |
| **Linux kernel** | `6.17.0-1032-nvidia` works; avoid `7.0.0-1019-nvidia` ([why](https://forums.developer.nvidia.com/t/dgx-spark-regression-kernel-7-0-0-1019-nvidia-causes-nccl-roce-ibv-reg-mr-iova2-enomem-6-17-0-1032-works/383023)) |
| **One-time setting** | `loginctl enable-linger nvidia` on both Sparks |

## Speed

Tokens per second is how fast the answer appears. A token is about three quarters of a word, and people read at 5–8 tokens
per second. These are real coding requests: the model thinks first, then answers.

| People at once | Short chat | Long chat (~12,000 words of history) |
|:---:|:---:|:---:|
| 1 | **62** tokens/s | **63** tokens/s |
| 4 | 39 each · 147 total | 36 each · 114 total |
| 16 | 19 each · **241 total** | 16 each · 179 total |

| First words appear after | Short chat | Long chat |
|:---:|:---:|:---:|
| Alone | **0.7 s** | **1.7 s** |
| 16 people at once | 6 s | 14 s |

**Getting faster with every build**, with the same model and every quality check passing:

| Build | One person | 16 people, long chats | First words, long chat |
|---|:---:|:---:|:---:|
| 2026-09-25 | 54 tokens/s | 139 tokens/s | 2.6 s |
| **2026-09-29** (current) | **62** tokens/s (+15%) | **179** tokens/s (+29%) | **1.7 s** (−35%) |

Full tables, methods and raw data: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Quality

Every build must pass the same gate before it ships. If a faster build fails any check, it does not ship.

| Check | Current build |
|---|---|
| **Tool use**: 88 hard multi-step scenarios ([tool-eval-bench](https://github.com/SeraphimSerapis) `--hardmode`) | 88–92 / 100, same band as the previous builds |
| **Forced tool calls** (`tool_choice=required`) | 5 / 5 |
| **Long-context recall**: 20 retrievals at 8k, 32k, 64k and 128k tokens | 20 / 20 at every depth |
| **Many users at once**: no request stalls at 5–16 concurrent | none |
| **Same answers as the previous build**: logits comparison | within run-to-run noise |

> **Known weakness: overthinking.** On long DevOps prompts that say "must pass `<validator>`", the model sometimes
> keeps checking its work and runs out of thinking budget before answering (23 of 42 test runs). When it did answer,
> 17 of 19 answers were clean. [Details](results/evals-20260928/README.md).

## Recipes

| Recipe | Use it when |
|---|---|
| [`qwen3.8-flash-next-2x-dgx-spark`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml) | **Always.** Current build (2026-09-29) |
| [`qwen3.8-flash-next-2x-dgx-spark-previous`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark-previous.yaml) | Only if the current one misbehaves for you. Previous build (2026-09-27), same model and settings |

Older names still resolve: see [recipes/RENAMES.md](recipes/RENAMES.md).

## Troubleshooting

| You see | Do this |
|---|---|
| `/health` doesn't answer yet | Normal for up to 10 minutes on the first run. Follow it with `sparkrun logs qwen3.8-flash-next-2x-dgx-spark -f` |
| Out-of-memory at start | Stop other containers or programs on both Sparks; the model uses 80% of each Spark's memory |
| NCCL or `ibv_reg_mr` error | Check `uname -r` on both Sparks (see the kernel row in [What you need](#what-you-need)) |
| Crash with `ShmRingBuffer ... shared_memory` | Run `loginctl enable-linger nvidia` on both Sparks, then start again |
| Answers look garbled | Both Sparks must load the same model files; see "Checkpoint revision" in [the reference](docs/REFERENCE.md#tuning-and-troubleshooting) |
| Still stuck | Try the `-previous` recipe, then open an issue with `sparkrun logs qwen3.8-flash-next-2x-dgx-spark -a` |

## How it works

- **Split across two machines.** Each Spark holds half of every layer (tensor parallel, TP=2), and the halves sync over the
  ConnectX-7 link after each layer.
- **Small, fast weights.** The model's experts are stored in 4-bit NVFP4 (most other layers in 8-bit MXFP8) and run on [b12x](https://github.com/local-inference-lab)
  kernels written for the Spark's GB10 chip.
- **Drafts 4 tokens at a time.** A small built-in draft head (MTP) guesses the next 4 tokens and the model checks them in
  one step. This is the biggest reason one person gets 62 tokens/s.
- **Remembers your conversation.** Prefix caching reuses the part of a chat it has already read, which is why long chats
  start answering in under 2 seconds.

Every flag and environment variable, with the reason for each: [docs/REFERENCE.md](docs/REFERENCE.md).

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
