# Qwen3.8-Flash-Next on two DGX Sparks

Runs the [Qwen3.8-Flash-Next](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) model split across two
NVIDIA DGX Sparks, and serves it as an OpenAI-compatible API on port 8000 (any OpenAI client or chat app works).

**Start here:** [What you need](#what-you-need) · [Quick start](#quick-start) · [What speed to expect](#what-speed-to-expect) ·
[Which recipe](#which-recipe) · [If something goes wrong](#if-something-goes-wrong)
**Details:** [Benchmark results](#benchmark-results) · [Quality](#quality) · [Configuration](#configuration) ·
[What is in the image](#what-is-in-the-image) · [Tuning and troubleshooting](#tuning-and-troubleshooting) ·
[Known limits](#known-limits) · [Files](#files) · [Credits](#credits)

## What you need

| Need | Detail |
|---|---|
| Hardware | 2x DGX Spark (128 GB memory each) |
| Cable | The two Sparks connected by their ConnectX-7 ports (the fast link between them) |
| Launcher | [sparkrun](https://github.com/eugr/sparkrun) 0.3.6, with a cluster of both Sparks set up |
| Disk | ~100 GB for the model + ~25 GB for the container image, on each Spark |
| Linux kernel | `6.17.0-1032-nvidia` works. Avoid `7.0.0-1019-nvidia` ([known problem](https://forums.developer.nvidia.com/t/dgx-spark-regression-kernel-7-0-0-1019-nvidia-causes-nccl-roce-ibv-reg-mr-iova2-enomem-6-17-0-1032-works/383023)) |
| One setting | Run `loginctl enable-linger nvidia` once on both Sparks (otherwise the server can crash while starting) |

## Quick start

```sh
sparkrun registry add https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2
sparkrun run qwen3.8-flash-next-2x-dgx-spark          # uses your default cluster; or --hosts <head>,<worker>
```

sparkrun downloads the container image and the model on both Sparks. The first start takes about **9-10 minutes**
(plus the ~100 GB model download on a new pair); later starts take about **4 minutes**.
`sparkrun run` can return before the model is ready, so wait until this prints `200`:

```sh
curl -s -o /dev/null -w '%{http_code}\n' http://<head>:8000/health
```

Then send a request (`<head>` is the first Spark's address):

```sh
curl -s http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Write a Python function that reverses a string."}],"max_tokens":512}'
```

Point a chat app or coding agent at `http://<head>:8000/v1` with model name `qwen3.8-flash-next` (no API key needed).
Stop the server with `sparkrun stop --all`.

## What speed to expect

Speed is measured in **tokens per second**: how fast the answer text appears. A token is roughly three quarters of a word;
reading speed is about 5-8 tokens per second. Numbers below are for coding requests (the model thinks first, then answers).

| People using it at once | Short conversation | Long conversation (~12,000 words already in the chat) |
|---|---|---|
| 1 | 54 tokens/s | 55 tokens/s |
| 4 | 37 tokens/s each (135 in total) | 33 each (104 in total) |
| 16 | 19 tokens/s each (241 in total) | 14 each (139 in total) |

**Time until the answer starts:** under 1 second for a short conversation, about 2.6 seconds for a long one when you are
the only user. With 16 people at once, about 6 seconds (short) and 22 seconds (long), because each new request waits for
the others' prompts to be read. Exact tables: [Benchmark results](#benchmark-results).

## Which recipe

| Recipe | Pick it when |
|---|---|
| [`qwen3.8-flash-next-2x-dgx-spark`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml) | **Always, by default.** The current build (2026-09-25). |
| [`qwen3.8-flash-next-2x-dgx-spark-previous`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark-previous.yaml) | Only if the recommended one misbehaves for you. The previous build (2026-09-23): same model and settings, about 10% slower with several users. Its first start takes ~9 minutes again. |

Renamed on 2026-09-25: `qwen3.8-flash-next-nvfp4-tp2` is now `qwen3.8-flash-next-2x-dgx-spark` ([all renames](recipes/RENAMES.md)).

## If something goes wrong

| What you see | What to check |
|---|---|
| `/health` does not answer yet | Normal for up to 10 minutes on the first start. Watch progress with `sparkrun logs qwen3.8-flash-next-2x-dgx-spark -f` |
| Start fails with an out-of-memory error | Stop other containers or programs on both Sparks; the model uses 80% of each Spark's memory |
| Start fails with an NCCL or `ibv_reg_mr` error | Check `uname -r` on both Sparks; see the kernel row in [What you need](#what-you-need) |
| Crash with `ShmRingBuffer ... shared_memory` | Run `loginctl enable-linger nvidia` on both Sparks, then start again |
| Answers look wrong or garbled | Both Sparks must have the same model files: see "Checkpoint revision" in [Tuning and troubleshooting](#tuning-and-troubleshooting) |
| Still stuck | Try the `-previous` recipe, then open an issue with the output of `sparkrun logs qwen3.8-flash-next-2x-dgx-spark -a` |

---

## Benchmark results

Everything below is for experienced users. Terms used: **concurrency (c)** = requests running at the same time;
**depth** = tokens of earlier conversation already cached before the new prompt; **MTP** (multi-token prediction) = the
model drafts up to 4 next tokens that are checked in one step, which is what makes single-user speed high.

Image `ghcr.io/ursuciprian/spark-vllm-b12x:b1-20260925-b7fbaf96-14077fb3-warm`
(`sha256:57c2fbd8cd811a5d22a7f2e547453f97b875f1fb4c7de60a0c3ff9fba3a79e5c`), checkpoint revision `7c4f1bc1`.
A/B against the previous build on 2026-09-24/25, two separate boots per build, means of both boots.
Raw files and verdict: [`results/b1-20260925/`](results/b1-20260925/).

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
[docs/BENCHMARKS.md](docs/BENCHMARKS.md#b0-shipped-image-2026-09-23).

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

## Quality

| Check | Result |
|---|---|
| tool-eval-bench `--hardmode` (88 tool-use scenarios, thinking on, temperature 0) | 89/100 on both A/B boots (previous build: 90); 87 then 91 on the published image (run-to-run band 86-93). TC-45 (`tool_choice=required`) now passes |
| Long-context recall (`scripts/fidelity_probe.py`, 20 tool-call retrievals per depth) | 20/20 exact at 8k, 32k, 64k and 128k, on both A/B boots and on the published image |
| Batch stragglers, c5-c16 (`scripts/straggler_probe.py`) | none; 3.97-4.00 tokens accepted per 4-token draft |

The published image was re-gated after promotion (`scripts/gate_arm.sh`, results in `results/b1-20260925/gate/`).

## Configuration

Recipe: [`qwen3.8-flash-next-2x-dgx-spark.yaml`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml).
Sampling comes from the checkpoint: **temperature 1.0, top-p 0.95, top-k 20** for clients that send none (0.6 measured the same speed).

| Flag | Value | Why |
|---|---|---|
| `--revision` (+ recipe `model_revision`) | `7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd` | Pins the same checkpoint on both ranks (see [Checkpoint revision](#tuning-and-troubleshooting)) |
| `--tensor-parallel-size` | `2` | One rank per Spark |
| `--speculative-config` | `method: mtp`, `num_speculative_tokens: 4`, `draft_sample_method: probabilistic` | MTP width 4; probabilistic drafts keep acceptance up at temperature 1.0 |
| `--kv-cache-dtype` | `fp8` | Halves KV memory |
| `--quantization` / `--load-format` | `modelopt_mixed` / `b12x` | NVFP4 checkpoint, b12x weight loader |
| `--gdn-decode-kernel` / `--linear-backend` / `--moe-backend` | `b12x` | b12x kernels for GDN (linear attention), linear layers and MoE |
| `--enable-prefix-caching`, `--mamba-cache-mode align` | on | Reuses cached conversation across turns, including GDN state |
| `--enable-chunked-prefill`, `--max-parallel-prefills` | on, `4` | Bounded concurrent prefills |
| `--prefill-policy` / `--decode-refill-target` | `decode-aware` / `auto` | Keeps decoding going while long prompts are read |
| `--max-model-len` / `--max-num-seqs` / `--max-num-batched-tokens` | `262144` / `16` / `8192` | Context per request / concurrent requests / prefill tokens per step |
| `--block-size` | `16` | KV block size |
| `--gpu-memory-utilization` | `0.80` | Share of unified memory for weights + KV; higher starves host RAM (see below) |
| `--reasoning-parser` / `--tool-call-parser` | `qwen3` / `qwen3_xml`, `--enable-auto-tool-choice` | Thinking and tool calls |
| `--compilation-config` | `fuse_act_quant: true` | Fused activation quantization |
| `--no-enable-flashinfer-autotune` | set | Kernel tuning is b12x's (`B12X_AUTOTUNE`) |
| `--served-model-name` | `qwen3.8-flash-next` (+ the HF id) | Model name clients send |

| Environment | Value | Why |
|---|---|---|
| `VLLM_GDN_DEFERRED_CHECKPOINTS` | `1` | New in this build: deferred GDN checkpoints (see [What is in the image](#what-is-in-the-image)) |
| `B12X_AUTOTUNE` | `1` | The base Dockerfile sets `0`, which truncates kernel selection (81 vs 96-98 tok/s c1 counting) |
| `B12X_COMPILE_WORKERS` | `2` | Caps kernel compile parallelism (~1.2 GB RAM each) so a cold first boot stays clear of the OOM killer |
| `B12X_ROCE_SPIN_LIMIT` | `300000000` | ~300 s instead of ~20 s, so an empty-cache TP=2 kernel retune cannot deadlock |
| `VLLM_MXFP8_LM_HEAD` | `1` | Output projection in online MXFP8 (W8A16): half the bytes per decode step |
| `VLLM_ENABLE_ROCE_ALLREDUCE` / `VLLM_ROCE_ALLREDUCE_MAX_SIZE` | `1` / `2MB` | RoCE all-reduce between the two ranks |
| `B12X_POLICY_MODE` / `CUTE_DSL_ARCH` | `auto` / `sm_121a` | b12x kernel policy; GB10 target |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | V2 model runner |
| `VLLM_USE_AOT_COMPILE`, `VLLM_USE_MEGA_AOT_ARTIFACT` | `1` | Compile cache, fast warm boots |
| `VLLM_SSM_CONV_STATE_LAYOUT` / `SAFETENSORS_FAST_GPU` / `VLLM_WORKER_MULTIPROC_METHOD` | `DS` / `1` / `spawn` | GDN conv state layout; faster weight load; worker start method |

The first line of the recipe's `command` copies the b12x plan-selection cache shipped in the image (`/opt/b12x-seed`) into
sparkrun's runtime cache when it is missing, so even a first boot does no kernel autotuning. No mods, host mounts or `--trust`.

## What is in the image

`ghcr.io/ursuciprian/spark-vllm-b12x:b1-20260925-b7fbaf96-14077fb3-warm`, digest
`sha256:57c2fbd8cd811a5d22a7f2e547453f97b875f1fb4c7de60a0c3ff9fba3a79e5c` (arm64, public). The recipe keeps the tag, not the
digest, because sparkrun 0.3.6 copies the image to the worker with `docker save | docker load`, which drops digests.

| Part | Source |
|---|---|
| Dockerfile | [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) `798528a2`, built by [spark-vllm-b12x](https://github.com/ursuciprian/spark-vllm-b12x) `build.sh` |
| vLLM | [`ursuciprian/vllm` tag `shipped-b1-20260925`](https://github.com/ursuciprian/vllm/tree/shipped-b1-20260925) (`14077fb35`), on local-inference-lab `dev/jovian-judgement` `8e1f1e58` |
| b12x kernels | [`ursuciprian/b12x` tag `shipped-b1-20260925`](https://github.com/ursuciprian/b12x/tree/shipped-b1-20260925) (`b7fbaf96`), on local-inference-lab `a8333658` |
| Warm layer | [`docker/b0-warm/`](docker/b0-warm/Dockerfile): b12x plan seed; pushed by the `build-b0-warm` workflow (run 36098908444) |

Changes against the previous build (`b0-20260918-a8333658-warm`):

- **Deferred GDN checkpoints.** With MTP, the GDN (linear-attention) layers used to write a full recurrent-state checkpoint
  for every draft position. Now the b12x decode kernel writes one base checkpoint plus small per-token records and replays
  only the accepted prefix. Most of the gain shows at 5+ concurrent requests.
- **TC-45 fix.** `tool_choice=required` (and named tool choice) was silently unconstrained because Qwen3's reasoning and
  tool parsers share one engine that never built the tool grammar. The fix gates only on `required`/named, so
  `tool_choice=auto` pays nothing.

Full provenance, including the previous image: [docs/ENGINEERING.md](docs/ENGINEERING.md#build-provenance).

## Tuning and troubleshooting

| Topic | Detail |
|---|---|
| Boot times | Measured 2026-09-25 from a fresh clone of `main`: cold 9 min 22 s (image pull + worker sync + empty compile cache), warm restart 3 min 50 s. Head log shows `0 measured` (no autotune); a cold boot compiles ~260 kernels once (they are keyed by GPU UUID), a warm one shows `0 compilations` |
| Logs | `sparkrun logs <recipe> -a` (head and worker), or `/tmp/sparkrun_serve.log` inside each node's container. `docker logs` shows only the banner |
| Checkpoint revision | Serve logs on both ranks must name only `snapshots/7c4f1bc1`. sparkrun's head-to-worker copy (`rsync --size-only`) never refreshes the worker's `refs/main`, so without `--revision` the ranks can load different revisions ([details](docs/ENGINEERING.md#known-issues--fixes)) |
| Caches | `~/.cache/sparkrun/runtime-cache/vllm/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-<hash>/` on each node, mounted at `/cache/runtime` (b12x, triton, inductor, vLLM compile caches). Deleting it makes the next boot cold, not broken |
| NCCL / RoCE | Launch through sparkrun: it sets `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX` and `UCX_NET_DEVICES` for the ConnectX-7 RoCE ports. `NCCL_SOCKET_IFNAME` only picks the TCP bootstrap interface. By hand, check `ls /sys/class/infiniband` |
| Kernel | `7.0.0-1019-nvidia` fails NCCL memory registration (`ibv_reg_mr_iova2 ... Cannot allocate memory`) once ~85-90 GB per node is GPU-resident; hold `6.17.0-1032-nvidia` |
| Memory | Memory is unified (CPU and GPU share 121 GB). `gpu_memory_utilization` 0.84+ drives host RAM to ~2 GB and earlyoom kills the server; 0.88 kills kernel preparation. Keep 0.80 |
| Shared memory | systemd-logind `RemoveIPC=yes` deletes the `nvidia` user's shared memory when an ssh session ends; `loginctl enable-linger nvidia` prevents it |
| Rolling back | `sparkrun stop --all`, then `sparkrun run qwen3.8-flash-next-2x-dgx-spark-previous`. Both builds share the runtime cache |

## Known limits

- Hardmode tool use still fails a handful of multi-step scenarios (e.g. TC-30, TC-68, TC-74, TC-88) on both builds.
- Depth-64k and prose throughput were not re-measured on this build; see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
- Single-request (c1) speed is bimodal on this pair (fast ~95-100 / slow ~85-88 tok/s counting); report medians of several runs.
- At most 16 requests run at once (`max_num_seqs`); more are queued.

## Files

| Path | What |
|---|---|
| [docs/ENGINEERING.md](docs/ENGINEERING.md) | Known issues and fixes, build provenance, profiling, rejected experiments |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | The previous build's full tables (2026-09-23, incl. 64k and prose) and older comparisons |
| [results/](results/README.md) | Every measurement and verdict |
| [recipes/RENAMES.md](recipes/RENAMES.md) | Recipe renames and the archive move |
| [archive/](archive/recipes/README.md) | Experimental and superseded recipes and mods (not listed by sparkrun) |

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

Apache-2.0, see [`LICENSE`](LICENSE). The vLLM overlays under `archive/mods/` keep their upstream Apache-2.0 headers.
