# Single-request perf with tool-eval-bench --perf (2026-10-06)

Issue #92. This follows a published single-request methodology, so the numbers can be read on the same terms.

## Method

- tool-eval-bench 2.6.1.dev65 `--perf-only`, which drives llama-benchy 0.4.0 over the OpenAI chat-completions endpoint.
- pp 2048, tg 128, one request at a time, 3 runs per cell. The cells are the tool's default depths 0, 4096 and 8192. Each value is the mean of the 3 runs.
- `temperature=0` and `reasoning_effort=low` are sent in every request body, both at the top level and in `chat_template_kwargs`. Every build confirmed that `low` reached the template: the prompt took 46 tokens at `low` and 16 at `medium`, because `low` adds an instruction.
- Speculative decoding as shipped: MTP with 4 drafts per step, fixed.
- One request of about 2k tokens ran before the first cell of each build, and its result was discarded.
- Tokens per step and acceptance come from the `vllm:spec_decode_*` counters in `/metrics`, read before and after each cell. Tokens per step = 1 + accepted drafts / steps. Acceptance per position is the fraction of steps whose draft at that position was accepted. The deltas include a few tokens from the tool's warm-up and latency probes.

Builds, each booted fresh on the pair:

| build | recipe | image |
|---|---|---|
| v3d | single Spark, `qwen3.8-flash-next-1x-dgx-spark` v3d (Hugging Face checkpoint) | `tp1-v3d-hf-20261005-21e0b201-5dad364d-warm` |
| v3c | single Spark, published 1x recipe v3c | `tp1-v3c-20261005-21e0b201-50330171-warm` |
| tp2 | both Sparks, `qwen3.8-flash-next-2x-dgx-spark` b1.4 | `b1.4-20261001-b7fbaf96-a7e649d8-warm` |

## Results

| build | depth | prefill tok/s | decode tok/s | TTFT ms | tokens/step | acceptance | acceptance per position | steps/s |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| tp2 | 0 | 3,389 | 84.0 | 731 | 3.43 | 60.7% | 0.84 / 0.70 / 0.50 / 0.39 | 24.5 |
| tp2 | 4096 | 2,848 | 92.8 | 2,275 | 3.68 | 67.1% | 0.89 / 0.74 / 0.59 / 0.47 | 25.2 |
| tp2 | 8192 | 3,011 | 91.2 | 3,533 | 3.77 | 69.2% | 0.90 / 0.76 / 0.62 / 0.49 | 24.2 |
| v3d | 0 | 1,878 | 62.3 | 1,197 | 3.47 | 61.8% | 0.83 / 0.69 / 0.54 / 0.41 | 18.0 |
| v3d | 4096 | 1,933 | 65.9 | 3,280 | 3.75 | 68.6% | 0.89 / 0.74 / 0.62 / 0.50 | 17.6 |
| v3d | 8192 | 2,003 | 63.0 | 5,217 | 3.55 | 63.9% | 0.82 / 0.68 / 0.58 / 0.47 | 17.7 |
| v3c | 0 | 1,928 | 50.7 | 1,172 | 3.19 | 54.6% | 0.79 / 0.59 / 0.46 / 0.35 | 15.9 |
| v3c | 4096 | 1,910 | 53.6 | 3,325 | 3.26 | 56.5% | 0.84 / 0.62 / 0.44 / 0.36 | 16.4 |
| v3c | 8192 | 2,056 | 59.1 | 5,088 | 3.68 | 67.1% | 0.89 / 0.73 / 0.56 / 0.50 | 16.1 |

Steps/s is decode tok/s divided by tokens per step. TTFT at depth > 0 includes prefill of the context.

## Reading

- Each build runs at a nearly constant number of steps per second: about 41 ms per step on TP2, 56 ms on v3d and 62 ms on v3c. Almost all of the decode difference between cells of one build comes from tokens per step, which ranges from 3.2 to 3.8 with 4 drafts. So in this single-request setting, draft acceptance is the main lever. A step that accepted every draft would give 5 tokens.
- Acceptance falls with draft position: the first draft is accepted in 79-90% of steps, the fourth in 35-50%. Adaptive draft depth is tracked in a separate issue.
- Acceptance changes by several points between cells with the same settings, because each cell has a different prompt. With 3 runs of 128 tokens per cell, a difference of a few percent in decode tok/s between builds is within what acceptance alone moves. Compare step times instead.

## Files

- `summary.md`: table from `summary.py` (the k50 job).
- `<build>/build.txt`, `effort.txt`, `effort-*.json`: recipe, image, tool versions and the effort check.
- `<build>/d<depth>/`: the tool's report (`2026/10/*.md`), its console log (`teb.log`) and the `/metrics` snapshots.
- `job.log`: job timeline.
