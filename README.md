# Qwen3.8-Flash-Next NVFP4 on two DGX Sparks

Serves [`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)
with vLLM + b12x kernels across 2x DGX Spark (tensor parallel 2 over ConnectX-7 RoCE),
with MTP speculative decoding. Packaged as a [sparkrun](https://github.com/eugr/sparkrun) recipe.

## Quick start

```sh
sparkrun registry add https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2
sparkrun run qwen3.8-flash-next-nvfp4-tp2          # default cluster; or --hosts <head>,<worker>
```

sparkrun pulls the image and downloads the pinned checkpoint on both nodes. No mods, mounts or `--trust`.

| Boot | Time (measured 2026-09-23) |
|---|---|
| Cold (no image, no compile cache) | ~9 min, plus the ~100 GB checkpoint download on a fresh pair |
| Warm restart | ~3.8 min |

`sparkrun run` can return before the server is ready. Poll `curl -s http://<head>:8000/health` until it returns 200, then:

```sh
curl -s http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Write a Python function that reverses a string."}],"max_tokens":512}'
```

## Requirements

| Need | Detail |
|---|---|
| Hardware | 2x DGX Spark (GB10, 128 GB unified memory each) |
| Link | ConnectX-7 RoCE between the two nodes |
| Launcher | sparkrun 0.3.6, cluster of both nodes set up |
| Disk | ~100 GB checkpoint + ~25 GB image, per node |
| Kernel | `6.17.0-1032-nvidia` known good. `7.0.0-1019-nvidia` has a [reported NCCL/RoCE regression](https://forums.developer.nvidia.com/t/dgx-spark-regression-kernel-7-0-0-1019-nvidia-causes-nccl-roce-ibv-reg-mr-iova2-enomem-6-17-0-1032-works/383023) |
| Host setting | `loginctl enable-linger nvidia` on both nodes (otherwise logind removes vLLM's shared memory) |

## Results (shipped image)

Image `ghcr.io/ursuciprian/spark-vllm-b12x:b0-20260918-a8333658-warm`
(`sha256:a3d5d90d1312edc9a79c86add6fdf72d50b2a73558e295fdf4ea110488fb615d`), checkpoint revision `7c4f1bc1`,
measured 2026-09-23. Raw files: [`results/shipped-20260923/`](results/shipped-20260923/).
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

## Quality

| Check | Result |
|---|---|
| tool-eval-bench `--hardmode` (88 scenarios, thinking on) | **90/100**; fails TC-45, TC-49, TC-68, TC-74. TC-45 (`tool_choice=required`) needs a parser fix that is not in this build |
| Long-context recall (20 tool-call retrievals per depth) | **20/20** exact at 8k, 32k, 64k and 128k |
| Batch stragglers, c5-c16 | none |

These ran on the pinned build of the same source one hour before the image was published (`results/arms/pinned-7c4f1bc1/` on the test pair).

## Settings

Sampling comes from the checkpoint: **temperature 1.0, top-p 0.95, top-k 20**. Clients that send no temperature get these.
Temperature 0.6 measured the same speed.

What the recipe sets ([`qwen3.8-flash-next-nvfp4-tp2.yaml`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml)):

| Setting | Value |
|---|---|
| Tensor parallel | 2 (one rank per Spark) |
| Speculative decoding | MTP, 4 draft tokens, `draft_sample_method: probabilistic` |
| KV cache | fp8 |
| lm_head | online MXFP8 (`VLLM_MXFP8_LM_HEAD=1`) |
| Checkpoint pin | `model_revision` + `--revision 7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd` |
| `gpu_memory_utilization` | 0.80 |
| `max_num_seqs` / `max_model_len` | 16 / 262144 |
| Prefix caching | on (`--mamba-cache-mode align`) |

Full flag and env tables: [docs/ENGINEERING.md](docs/ENGINEERING.md#configuration).

## Troubleshooting

| Check | How |
|---|---|
| Both nodes on the same checkpoint revision | `cat ~/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/refs/main` on each node. The recipe pins `--revision`; before that pin, sparkrun's copy left the worker on an older revision ([details](docs/ENGINEERING.md#known-issues--fixes)) |
| Kernel | `uname -r`: `6.17.0-1032-nvidia` works; see the 7.0.0-1019 note above |
| Memory headroom | The recipe uses 0.80 of unified memory; stop other containers if the boot runs out of memory |
| RoCE link | Launch through sparkrun, which sets `NCCL_IB_HCA` / `NCCL_IB_*` for the ConnectX-7 RoCE ports. By hand: check `NCCL_IB_HCA` names the CX7 RoCE devices (`ls /sys/class/infiniband`); `NCCL_SOCKET_IFNAME` only affects TCP bootstrap |
| Server not answering right after `sparkrun run` | It returns before the model is ready; poll `/health` |
| Fallback recipe `-argmax-drafts` | Previous default (one-hot drafts); not yet boot-tested on the warm image |

## Recipes

| Recipe | Use |
|---|---|
| [`qwen3.8-flash-next-nvfp4-tp2`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml) | Default |
| [`qwen3.8-flash-next-nvfp4-tp2-argmax-drafts`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-argmax-drafts.yaml) | Fallback: one-hot drafts, for temperature-0 clients or rollback |

## Files

| Path | What |
|---|---|
| [docs/ENGINEERING.md](docs/ENGINEERING.md) | Known issues and fixes, full configuration, image build provenance, rejected experiments |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Older benchmark tables and comparisons (before 2026-09-23) |
| [results/](results/README.md) | Every measurement and verdict |
| [archive/](archive/recipes/README.md) | Experimental and superseded recipes and mods (not scanned by sparkrun) |

## Credits

- [eugr](https://github.com/eugr): spark-vllm-docker (image base), sparkrun, llama-benchy (base of our fork).
- [local-inference-lab](https://github.com/local-inference-lab): the vLLM fork, the b12x kernels and the NVFP4 checkpoint.
- [MiaAI-Lab](https://github.com/MiaAI-Lab): early SGLang profile, benchmark spec and draft-vocab table.
- [RadixArk](https://huggingface.co/RadixArk): NVFP4 checkpoint and day-0 SGLang work.
- [tonyd2wild](https://github.com/tonyd2wild): vLLM SM121 overlays and the bench_sweep counting harness.
- [Weschera](https://github.com/Weschera): spark-bench graded eval.
- [SeraphimSerapis](https://github.com/SeraphimSerapis): tool-eval-bench.

Exact pins and contributions: [docs/ENGINEERING.md](docs/ENGINEERING.md#credits).

## License

Apache-2.0, see [`LICENSE`](LICENSE). The vLLM overlays under `mods/` keep their upstream Apache-2.0 headers.
