DP=2 against TP=2 on the same agent-style replay, measured 2026-10-07.

- 2x TP=2: the shipped `qwen3.8-flash-next-2x-dgx-spark` (b1.4), `max_num_seqs` 16.
- DP=2: the shipped `qwen3.8-flash-next-1x-dgx-spark` (v3d) on each Spark, `max_num_seqs` 8 each, behind a minimal prefix-affinity router (`scripts/pa_router.py`: sha1 of the first two non-system messages picks the replica, so all turns of a session go to the replica that holds its prefix cache).

Workload (`drive.py` in the results dir, same seeds on both setups): each session is a synthetic agent transcript of tool calls; every turn sends the whole conversation with tools on, thinking off, temperature 0.6, then appends the reply and a tool result. Sessions start together.

- `agent8`: 8 sessions x 6 turns, ~32K-token start, +2K tokens per turn, up to 512 output tokens.
- `agent16`: 16 sessions x 4 turns, same sizes.
- `long-N`: N sessions x 2 turns, ~128K-token start, +1K per turn, up to 256 output tokens, N = 4, 8, 12, 16. The ladder stops after the first N with a preemption or a request error.

| workload | setup | turns done | output tok/s total | TTFT first turn mean / max (s) | TTFT follow-up mean / p90 (s) | decode tok/s per request | prefix hit | preemptions | KV max per replica | sessions per replica | min MemAvailable dgx-01 / dgx-02 (GiB) |
|---|---|---:|---:|---|---|---:|---:|---:|---|---|---|
| agent8 | 2x TP=2 | 48/48 | 11.3 | 68.3 / 104.6 | 4.9 / 7.6 | 6.9 | 78% | 0 | 15% | - | 10.97 / 14.35 |
| agent8 | DP=2 (2 x 1x v3d) | 48/48 | 18.7 | 49.5 / 65.7 | 4.0 / 5.5 | 13.8 | 78% | 0 | 21% / 20% | 4/4 | 14.16 / 14.03 |
| agent16 | 2x TP=2 | 64/64 | 11.0 | 121.9 / 233.5 | 9.7 / 20.5 | 6.5 | 70% | 0 | 21% | - | 10.30 / 14.13 |
| agent16 | DP=2 (2 x 1x v3d) | 64/64 | 15.7 | 87.3 / 144.1 | 6.7 / 10.4 | 8.1 | 70% | 0 | 35% / 35% | 8/8 | 13.84 / 13.91 |
| long-4 | 2x TP=2 | 8/8 | 1.5 | 165.8 / 220.0 | 5.2 / 7.4 | 33.0 | 49% | 0 | 14% | - | 10.69 / 14.10 |
| long-4 | DP=2 (2 x 1x v3d) | 8/8 | 2.1 | 136.3 / 137.5 | 2.9 / 3.1 | 49.2 | 49% | 0 | 29% / 29% | 2/2 | 13.81 / 13.84 |
| long-8 | 2x TP=2 | 16/16 | 1.3 | 283.7 / 448.6 | 7.3 / 8.7 | 16.3 | 49% | 0 | 19% | - | 10.64 / 14.08 |
| long-8 | DP=2 (2 x 1x v3d) | 16/16 | 2.2 | 210.6 / 279.4 | 5.9 / 9.2 | 27.3 | 49% | 0 | 43% / 40% | 4/4 | 13.75 / 13.82 |
| long-12 | 2x TP=2 | 24/24 | 1.4 | 409.3 / 670.6 | 44.6 / 150.0 | 9.2 | 49% | 0 | 27% | - | 10.61 / 13.92 |
| long-12 | DP=2 (2 x 1x v3d) | 24/24 | 2.1 | 283.4 / 423.0 | 7.3 / 9.2 | 18.5 | 49% | 0 | 45% / 46% | 6/6 | 13.69 / 13.77 |
| long-16 | 2x TP=2 | 32/32 | 1.2 | 520.0 / 897.2 | 62.4 / 157.2 | 7.1 | 49% | 0 | 29% | - | 10.44 / 14.00 |
| long-16 | DP=2 (2 x 1x v3d) | 32/32 | 2.8 | 359.9 / 563.4 | 7.7 / 9.5 | 10.4 | 49% | 0 | 58% / 58% | 8/8 | 13.65 / 13.75 |

Max concurrent long contexts:
- 2x TP=2: 16 sessions of ~129,531 tokens ran with no preemption or error (the largest step of the ladder).
- DP=2: 16 sessions of ~129,531 tokens ran with no preemption or error (the largest step of the ladder).

KV pools from the boot logs:
```
2x TP=2 (dgx-01 rank): Available KV cache memory: 30.43 GiB
2x TP=2 (dgx-01 rank): GPU KV cache size: 3,661,881 tokens, Maximum concurrency for 262,144 tokens per request: 13.97x
1x v3d on <cx7-ip-a>: GPU KV cache size: 993,754 tokens, Maximum concurrency for 262,144 tokens per request: 3.79x
1x v3d on <cx7-ip-b>: GPU KV cache size: 993,754 tokens, Maximum concurrency for 262,144 tokens per request: 3.79x
```

Columns: prefix hit = cache hit tokens / queried tokens, summed over replicas; preemptions = `vllm:num_preemptions_total` delta; KV max = highest `vllm:kv_cache_usage_perc` sample (every 5 s) per replica; sessions per replica = distinct conversations the router sent to each replica; decode tok/s per request = completion tokens / time after the first token.

Raw data: {RESULTS_URL} (per-turn JSON per workload, router stats, boot log KV lines, guard logs).
