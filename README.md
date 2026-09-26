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
| [`qwen3.8-flash-next-2x-dgx-spark`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml) | **Always, by default.** The current build (2026-09-26). |
| [`qwen3.8-flash-next-2x-dgx-spark-previous`](recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark-previous.yaml) | Only if the recommended one misbehaves for you. The previous build (2026-09-25): same model and settings, same speed on fresh prompts, slower on follow-up turns of long chats. |

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

Image `ghcr.io/ursuciprian/spark-vllm-b12x:b1.1-20260926-b7fbaf96-6d232f16-warm`
(`sha256:91a60ebce422db8a80f58ce998a3e9bd847814c8579ac33fa4df99e62351b3a8`), checkpoint revision `7c4f1bc1`.
A/B against the previous build (2026-09-25) on 2026-09-26, two separate boots per build, means of both boots.
Raw files and verdict: [`results/b1.1-20260926/`](results/b1.1-20260926/). The 2026-09-25 build's tables are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

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

## Quality

| Check | Result |
|---|---|
| tool-eval-bench `--hardmode` (88 tool-use scenarios, thinking on, temperature 0) | 91 and 89/100 on the two A/B boots (previous build: 88-91; run-to-run band 86-93). TC-45 (`tool_choice=required`) 5/5 |
| Long-context recall (`scripts/fidelity_probe.py`, 20 tool-call retrievals per depth) | 20/20 exact at 8k, 32k, 64k and 128k; 128k also on two more seeds (11, 13): 20/20 each |
| Batch stragglers, c5-c16 (`scripts/straggler_probe.py`) | none; 3.97-4.00 tokens accepted per 4-token draft |

Gate run on the A/B boots of the same build (`scripts/gate_arm.sh`); the published image adds only the plan-seed layer.

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
| `VLLM_PREFIX_DROP_EXACT` | `1` | New in this build: exact prefix-cache hits with MTP (see [What is in the image](#what-is-in-the-image)) |
| `VLLM_GDN_DEFERRED_CHECKPOINTS` | `1` | Deferred GDN checkpoints (since 2026-09-25) |
| `B12X_AUTOTUNE` | `1` | The base Dockerfile sets `0`, which truncates kernel selection (81 vs 96-98 tok/s c1 counting) |
| `B12X_COMPILE_WORKERS` | `2` | Caps kernel compile parallelism (~1.7 GB RAM each) so a cold first boot stays clear of the OOM killer. Honoured from this build on (earlier builds always used 16) |
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

`ghcr.io/ursuciprian/spark-vllm-b12x:b1.1-20260926-b7fbaf96-6d232f16-warm`, digest
`sha256:91a60ebce422db8a80f58ce998a3e9bd847814c8579ac33fa4df99e62351b3a8` (arm64, public). The recipe keeps the tag, not the
digest, because sparkrun 0.3.6 copies the image to the worker with `docker save | docker load`, which drops digests.

| Part | Source |
|---|---|
| Dockerfile | [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) `798528a2`, built by [spark-vllm-b12x](https://github.com/ursuciprian/spark-vllm-b12x) `build.sh` |
| vLLM | [`ursuciprian/vllm` tag `shipped-b1.1-20260926`](https://github.com/ursuciprian/vllm/tree/shipped-b1.1-20260926) (`6d232f16f`), on local-inference-lab `dev/jovian-judgement` `8e1f1e58` |
| b12x kernels | [`ursuciprian/b12x` tag `shipped-b1-20260925`](https://github.com/ursuciprian/b12x/tree/shipped-b1-20260925) (`b7fbaf96`, unchanged), on local-inference-lab `a8333658` |
| Warm layer | [`docker/b0-warm/`](docker/b0-warm/Dockerfile): b12x plan seed; pushed by the `build-b0-warm` workflow (run 36247635045) |

Changes against the previous build (`b1-20260925-b7fbaf96-14077fb3-warm`), all in vLLM:

- **Exact prefix-cache hits with MTP** (`VLLM_PREFIX_DROP_EXACT=1`). The aligned GDN checkpoint lookup dropped the last
  cached block to leave room for the MTP drafter, one 2864-token block more than needed. Follow-up turns now reuse it.
- **Mamba/GDN padding fix** (vllm#887): padded block-table slots use `NULL_BLOCK_ID`.
- **Compile-worker cap honoured.** vLLM's startup compile pool ignored `B12X_COMPILE_WORKERS` and always used 16 workers;
  a cold boot now stays at 2 (cold boot 14 min, lowest free memory 4 GB head / 6 GB worker).

Earlier changes (2026-09-25 build): deferred GDN checkpoints and the TC-45 `tool_choice=required` fix.

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
