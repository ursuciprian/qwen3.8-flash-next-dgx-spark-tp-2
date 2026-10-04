# Reference

Configuration, image provenance, tuning and known limits for the current build (b1.4, 2026-10-01).
Back to the [README](../README.md).

## Configuration

Recipe: [`qwen3.8-flash-next-2x-dgx-spark.yaml`](../recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml).
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
| `--default-chat-template-kwargs` | `{"reasoning_effort":"medium"}` | Thinking effort for requests that set none (the template's own default is `xhigh`, which runs away on validator-gated tasks); a request overrides it, see the README [Thinking effort](../README.md#thinking-effort) (since 2026-10-01) |

| Environment | Value | Why |
|---|---|---|
| `VLLM_PREFIX_DROP_EXACT` | `1` | Exact prefix-cache hits with MTP (since 2026-09-26; see [What is in the image](#what-is-in-the-image)) |
| `VLLM_GDN_DEFERRED_CHECKPOINTS` | `1` | Deferred GDN checkpoints (since 2026-09-25) |
| `VLLM_GDN_UNIFORM_DECODE_META_SKIP` | `1` | Skips ~350 GDN metadata launches per uniform decode step (since 2026-09-29) |
| `VLLM_MTP_DRAFT_VOCAB` | `/opt/mtp-vocab/ids-v2-K131072.txt.gz` | MTP draft head scores 131,072 of 248,320 vocab ids (lossless via rejection sampling); the list ships in the image (since 2026-10-01) |
| `VLLM_QWEN38_HC_MXFP8` | `hc` | Hyper-connection mixers in online MXFP8 (since 2026-09-27) |
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

`ghcr.io/ursuciprian/spark-vllm-b12x:b1.4-20261001-b7fbaf96-a7e649d8-warm`, digest
`sha256:3b2f26080addadafe675f31227d6dacec3716064cbc0c7fd376b643bc34183fd` (arm64, public). The recipe keeps the tag, not the
digest, because sparkrun 0.3.6 copies the image to the worker with `docker save | docker load`, which drops digests.

| Part | Source |
|---|---|
| Dockerfile | [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) `798528a2`, built by [spark-vllm-b12x](https://github.com/ursuciprian/spark-vllm-b12x) `build.sh` |
| vLLM | [`ursuciprian/vllm` tag `shipped-b1.4-20261001`](https://github.com/ursuciprian/vllm/tree/shipped-b1.4-20261001) (`a7e649d8`), on local-inference-lab `dev/jovian-judgement` `8e1f1e58` |
| b12x kernels | [`ursuciprian/b12x` tag `shipped-b1.4-20261001`](https://github.com/ursuciprian/b12x/tree/shipped-b1.4-20261001) (`b7fbaf96`, unchanged since 2026-09-25), on local-inference-lab `a8333658` |
| Warm layer | [`docker/b0-warm/`](../docker/b0-warm/Dockerfile): b12x plan seed + the MTP draft-vocab list (`/opt/mtp-vocab`); pushed by the `build-b0-warm` workflow (run 36866001482) |

Changes against the previous build (`b1.3-20260929-b7fbaf96-7344a997-warm`):

- **MTP draft vocab** (`VLLM_MTP_DRAFT_VOCAB`). The draft head scores 131,072 of the 248,320 vocab ids (65,536 rows
  per rank) instead of all of them, which cuts the draft-head GEMM per pass; rejection sampling keeps the output
  distribution unchanged (an id outside the list gets draft probability 0). The list is ranked on our own corpora and
  eval outputs and always holds every special token, all 256 byte symbols and all single-character tokens, so a copied
  rare string can still be drafted byte by byte ([provenance](../archive/mods/r7-dvocab/README.md)). A borrowed
  65,536-id list cost 3-5 acceptance points at 16k and was rejected on 2026-09-30; this one keeps acceptance within
  0.2 points.
- **vllm#923 port**: the QSA prefill flag sat in a pinned host buffer that the next step could overwrite before the
  GPU read it (a race that can give NaN state and token loops); it is now a plain host tensor.
- **vllm#914 port**: a Triton restore kernel no longer specialises on block/slot values, so it stops recompiling on
  the request path.
- **Default thinking effort `medium`** (`--default-chat-template-kwargs`), a recipe change, not an image change.

Earlier changes (2026-09-29 build): GDN uniform-decode metadata skip (`VLLM_GDN_UNIFORM_DECODE_META_SKIP=1`).
Earlier changes (2026-09-27 build): HC (hyper-connection) mixers in online MXFP8 (`VLLM_QWEN38_HC_MXFP8=hc`).
Earlier (2026-09-26 build): exact prefix-cache hits with MTP (`VLLM_PREFIX_DROP_EXACT`), the Mamba/GDN
`NULL_BLOCK_ID` padding fix (vllm#887), and the compile-worker cap. Earlier still (2026-09-25 build): deferred GDN
checkpoints and the TC-45 `tool_choice=required` fix.

Full provenance, including the previous image: [docs/ENGINEERING.md](ENGINEERING.md#build-provenance).

## Tuning and troubleshooting

| Topic | Detail |
|---|---|
| Boot times | Measured 2026-09-25 from a fresh clone of `main`: cold 9 min 22 s (image pull + worker sync + empty compile cache), warm restart 3 min 50 s. Head log shows `0 measured` (no autotune); a cold boot compiles ~260 kernels once (they are keyed by GPU UUID), a warm one shows `0 compilations` |
| Logs | `sparkrun logs <recipe> -a` (head and worker), or `/tmp/sparkrun_serve.log` inside each node's container. `docker logs` shows only the banner |
| Checkpoint revision | Serve logs on both ranks must name only `snapshots/7c4f1bc1`. sparkrun's head-to-worker copy (`rsync --size-only`) never refreshes the worker's `refs/main`, so without `--revision` the ranks can load different revisions ([details](ENGINEERING.md#known-issues--fixes)) |
| Caches | `~/.cache/sparkrun/runtime-cache/vllm/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-<hash>/` on each node, mounted at `/cache/runtime` (b12x, triton, inductor, vLLM compile caches). Deleting it makes the next boot cold, not broken |
| NCCL / RoCE | Launch through sparkrun: it sets `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX` and `UCX_NET_DEVICES` for the ConnectX-7 RoCE ports. `NCCL_SOCKET_IFNAME` only picks the TCP bootstrap interface. By hand, check `ls /sys/class/infiniband` |
| Kernel | `7.0.0-1019-nvidia` fails NCCL memory registration (`ibv_reg_mr_iova2 ... Cannot allocate memory`) once ~85-90 GB per node is GPU-resident; hold `6.17.0-1032-nvidia` |
| Memory | Memory is unified (CPU and GPU share 121 GB). `gpu_memory_utilization` 0.84+ drives host RAM to ~2 GB and earlyoom kills the server; 0.88 kills kernel preparation. Keep 0.80 |
| Shared memory | systemd-logind `RemoveIPC=yes` deletes the `nvidia` user's shared memory when an ssh session ends; `loginctl enable-linger nvidia` prevents it |
| Rolling back | `sparkrun stop --all`, then `sparkrun run qwen3.8-flash-next-2x-dgx-spark-previous`. Both builds share the runtime cache |

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
[`results/b1.4-20261001/effort-override/`](../results/b1.4-20261001/effort-override/).

Not yet measured at `medium`: MMLU-Pro, GSM8K, IFEval and LiveCodeBench, where long thinking may still pay off; ask for
`xhigh` there. `-previous` (b1.3) keeps the template default `xhigh`. [Eval details](../results/evals-20260928/README.md),
[b1.4 run](../results/b1.4-20261001/devops-b14-medium.md).

## Known limits

- Hardmode tool use still fails a handful of multi-step scenarios (e.g. TC-30, TC-68, TC-74, TC-88) on both builds.
- Depth-64k and prose throughput were not re-measured on this build; see [docs/BENCHMARKS.md](BENCHMARKS.md).
- Single-request (c1) speed is bimodal on this pair (fast ~95-100 / slow ~85-88 tok/s counting); report medians of several runs.
- At most 16 requests run at once (`max_num_seqs`); more are queued.
