DP=2 against TP=2 on the same agent-style replay, measured 2026-10-08.

- 2x TP=2: the shipped `qwen3.8-flash-next-2x-dgx-spark` (b1.6), `max_num_seqs` 16.
- DP=2: the shipped `qwen3.8-flash-next-1x-dgx-spark` (v3e) on each Spark, `max_num_seqs` 8 each, behind a minimal prefix-affinity router (`tools/dp2/pa_router.py` in the repo: a new conversation goes to the replica with the fewest conversations, keyed by the sha1 of its first two non-system messages, and every later turn sticks to the replica that holds its prefix cache).

Workload (`drive.py` in the results dir, same seeds on both setups): each session is a synthetic agent transcript of tool calls; every turn sends the whole conversation with tools on, thinking off, temperature 0.6, then appends the reply and a tool result. Sessions start together.

- `agent8`: 8 sessions x 6 turns, ~32K-token start, +2K tokens per turn, up to 512 output tokens.
- `agent16`: 16 sessions x 4 turns, same sizes.
- `long-N`: N sessions x 2 turns, ~128K-token start, +1K per turn, up to 256 output tokens, N = 4, 8, 12, 16. The ladder stops after the first N with a preemption or a request error.

| workload | setup | turns done | output tok/s total | TTFT first turn mean / max (s) | TTFT follow-up mean / p90 (s) | decode tok/s per request | prefix hit | preemptions | KV max per replica | sessions per replica | min MemAvailable dgx-01 / dgx-02 (GiB) |
|---|---|---:|---:|---|---|---:|---:|---:|---|---|---|
| agent8 | 2x TP=2 | 48/48 | 12.2 | 64.9 / 113.5 | 5.0 / 7.2 | 8.0 | 78% | 0 | 15% | - | 10.83 / 14.21 |
| agent8 | DP=2 (2 x 1x v3e) | 48/48 | 20.5 | 49.5 / 65.8 | 4.5 / 6.8 | 16.5 | 77% | 0 | 20% / 24% | 4/4 | 14.08 / 13.84 |
| agent16 | 2x TP=2 | 64/64 | 10.3 | 123.8 / 237.5 | 9.4 / 21.2 | 7.3 | 70% | 0 | 18% | - | 10.34 / 14.10 |
| agent16 | DP=2 (2 x 1x v3e) | 64/64 | 14.3 | 88.1 / 148.7 | 8.7 / 11.0 | 8.3 | 70% | 0 | 33% / 38% | 8/8 | 13.71 / 13.82 |
| long-4 | 2x TP=2 | 8/8 | 1.3 | 166.3 / 221.1 | 5.6 / 9.3 | 27.6 | 49% | 0 | 13% | - | 10.09 / 14.05 |
| long-4 | DP=2 (2 x 1x v3e) | 8/8 | 2.3 | 136.6 / 137.4 | 2.4 / 3.1 | 39.5 | 49% | 0 | 29% / 29% | 2/2 | 13.71 / 13.76 |
| long-8 | 2x TP=2 | 16/16 | 1.3 | 285.8 / 449.6 | 8.4 / 9.6 | 14.5 | 49% | 0 | 20% | - | 10.40 / 14.04 |
| long-8 | DP=2 (2 x 1x v3e) | 16/16 | 2.4 | 210.8 / 280.1 | 6.0 / 9.0 | 25.8 | 49% | 0 | 43% / 40% | 4/4 | 13.65 / 13.73 |
| long-12 | 2x TP=2 | 24/24 | 1.2 | 411.0 / 675.7 | 32.5 / 152.3 | 9.9 | 49% | 0 | 23% | - | 10.34 / 14.02 |
| long-12 | DP=2 (2 x 1x v3e) | 24/24 | 2.1 | 284.0 / 423.7 | 7.3 / 9.3 | 15.9 | 49% | 0 | 43% / 46% | 6/6 | 13.61 / 13.70 |
| long-16 | 2x TP=2 | 32/32 | 1.2 | 519.4 / 897.7 | 63.4 / 154.9 | 7.7 | 49% | 0 | 29% | - | 10.25 / 13.81 |
| long-16 | DP=2 (2 x 1x v3e) | 32/32 | 2.0 | 360.7 / 564.9 | 8.1 / 9.7 | 11.4 | 49% | 0 | 58% / 55% | 8/8 | 13.55 / 13.70 |

Max concurrent long contexts:
- 2x TP=2: 16 sessions of ~129,531 tokens ran with no preemption or error (the largest step of the ladder).
- DP=2: 16 sessions of ~129,531 tokens ran with no preemption or error (the largest step of the ladder).

KV pools from the boot logs:
```
2x TP=2 (dgx-01 rank): Available KV cache memory: 30.34 GiB
2x TP=2 (dgx-01 rank): GPU KV cache size: 3,650,419 tokens, Maximum concurrency for 262,144 tokens per request: 13.93x
1x v3e on <cx7-ip-a>: GPU KV cache size: 993,754 tokens, Maximum concurrency for 262,144 tokens per request: 3.79x
1x v3e on <cx7-ip-b>: GPU KV cache size: 993,754 tokens, Maximum concurrency for 262,144 tokens per request: 3.79x
```

Columns: prefix hit = cache hit tokens / queried tokens, summed over replicas; preemptions = `vllm:num_preemptions_total` delta; KV max = highest `vllm:kv_cache_usage_perc` sample (every 5 s) per replica; sessions per replica = distinct conversations the router sent to each replica; decode tok/s per request = completion tokens / time after the first token.

Raw data: {RESULTS_URL} (per-turn JSON per workload, router stats, boot log KV lines, guard logs).
