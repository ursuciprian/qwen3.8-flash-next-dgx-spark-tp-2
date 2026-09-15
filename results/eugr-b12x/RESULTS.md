# Qwen3.8-Flash-Next on eugr's b12x vLLM route, two DGX Sparks, 2026-09-15

Recipe: eugr/spark-vllm-docker `recipes/qwen3.8-flash-next-nvfp4-cluster.yaml` (repo 3e1578b), run unchanged through `sparkrun run <recipe> --cluster dgx-cluster-cx7 --tp 2 --trust --no-follow`. Image `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x:latest` (79ecfeff, built 2026-09-13), vLLM fork `local-inference-lab/vllm` v0.1.dev20759, b12x kernel library. Checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (98.6 GiB, 37 shards, ada4da32). Model load 45 s with the b12x loader, 52 GiB per rank, healthy in about 9 minutes.

All decode numbers below are `scripts/decode_probe.py` (Tony's method: streaming, temp 0, decode = (completion-1)/(t_last-t_first), three repeats, mean). The probe times `content` deltas; on vLLM the thinking phase streams as `reasoning_content`, so the node copy of the probe was patched to time reasoning deltas too. Before that patch the code lane reported FAILED on vLLM. Structured-lane spread inside one arm reaches +-15%; treat deltas under 10 tok/s as noise.

## eugr route vs our SGLang recipe

| lane | SGLang `flashnext-fp8kv-1m8` (Triton GDN, NEXTN 3) | SGLang + `--linear-attn-decode-backend cutedsl` | SGLang + flashinfer GDN | eugr b12x vLLM, first boot 08:37 | eugr b12x vLLM, later clean boots |
|---|---|---|---|---|---|
| code | 44.9 | probe failed (old probe) | 42.7 | probe failed (old probe) | 49-55 |
| structured | 54.7 | 92.5 (one valid run) | 55.1 | 113.1 | 70-85 |
| counting | 63.9 | 79.2 | 62.9 | 112.3 | 84-94 |
| prose | 37.7 | 47.2 | 36.6 | 53.2 | 43-47 |
| llama-benchy tg128 c1 | 37.5 | 36.5 | 37.3 | 39.1 | 42-46 |
| llama-benchy pp2048 c1 | | | | 2612 | 2127-2368 |
| tool-eval short | 93-95 (2026-09-14) | | | 100 | 100 on every arm |
| essays T 0 to 1.0, 24 tries | | | | 0 loops, 42-47 tok/s | 0 loops |

The 08:37 structured/counting figures (113/112) did not reproduce on six later boots of the same image and recipe (70-85 structured). Same image ID on both nodes for all runs. The bisect that ran between 10:06 and 10:52 was partly contaminated by numeric kernel checks running on the head GPU in a second container, so its one-flag deltas are not reported as findings; MTP off is the only clean result there: plain decode 32 tok/s on every lane, so MTP-4 is worth about 2.5x.

## Agents ladder (recipes/eugr/eugr-agents*.yaml)

Base changes for agent use: `max_num_seqs 8`, `max_num_batched_tokens 8192`, `gpu_memory_utilization 0.85`. Clean boots, no other GPU load.

| arm | code | structured | counting | prose | tg128 c1 | tool-eval short |
|---|---|---|---|---|---|---|
| eugr-agents (MTP 4, RoCE 2MB) | 53.8 | 70.2 | 93.4 | 43.6 | 45.9 | 100 |
| MTP 3 | 53.4 | 72.0 | 83.8 | 46.6 | 37.0 | 100 |
| MTP 5, MTP 6 | boot failed: `QSA currently supports at most four speculative tokens` | | | | | |
| RoCE all-reduce threshold 4MB | 55.4 | 85.3 | 90.5 | 44.4 | 42.3 | 100 |
| RoCE all-reduce threshold 1MB | 49.0 | 70.9 | 94.4 | 43.0 | 42.3 | 100 |

RoCE 4MB looked like the only arm outside the noise band on structured (83-87 vs 62-77) but the A/B/A re-check with five-repeat probes did not confirm it: RoCE 4MB 76.1 (70-85), base 79.0 (68-86). The threshold stays at 2MB. Five-repeat base numbers: code 51.6, structured 79.0, counting 89.6, prose 44.0. Tool-eval hardmode on the base recipe (T=1.0, thinking medium, 32 turns): 91/100, 77 passed, 6 partial, 5 failed, median turn 2.7 s. Prefix caching works: over the ladder 946k prefix-cache queries and 834k hits (88%); an identical 4k-token prompt goes 1.55 s, 0.21 s, 0.21 s. The usage field `prompt_tokens_details` is not filled by this fork, so client-side cached-token accounting reads None; the multi-turn prefix gate (68 requests, 5 concurrent growing chats) had 0 failures.

## SGLang b12x GDN kernel port (mods/sglang-gdn-b12x-decode)

b12x installs into the SGLang nightly image (`pip install git+https://github.com/local-inference-lab/b12x@40bcdf82a03b`, Apache-2.0). The mod adds `B12xGDNKernel` behind SGLang's linear-attention kernel contract: `packed_decode` (v1) and `target_verify` for the NEXTN chain (v2), routed by two anchors in `GDNKernelDispatcher` when `SGLANG_GDN_B12X=1`. Numeric checks on GB10 against the Triton kernels passed (decode: |rmsnorm(triton) - b12x| <= 1e-3, states <= 1e-3; verify over 4 chained tokens: output <= 0.03 bf16, intermediate checkpoints <= 0.004, committed states untouched). The v1 arm (decode kernel only) measured the same as Triton under NEXTN, as expected: the target runs `target_verify`. The v2 arm (decode + verify on b12x) is broken in the live server although both isolated checks and a CUDA-graph replay check pass: acceptance length collapses to 1.0-1.5, output is repeated garbage, tool-eval 0/100. The call-site mismatch is not identified yet; an instrumented boot (`SGLANG_GDN_B12X_DEBUG=1`, in-situ Triton A/B on the real tensors) is queued. The cutedsl re-run with the fixed probe measured the same as Triton (44.8 / 56.7 / 62.8 / 39.5, tool-eval 97), so with NEXTN the GDN decode kernel choice does not move SGLang; only a working verify port could.
