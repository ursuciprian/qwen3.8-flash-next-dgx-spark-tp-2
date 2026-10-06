# k46 high-concurrency runs

Results dir: results/high-conc-k46b-20261005 (logs not committed). Every arm is one fresh boot. 'max of N' = the best of N rounds of that cell.

## tp2-s32: TP2 b1.4 (2x recipe), max_num_seqs 32, CUDA graphs up to 160 rows

node(s) dgx-01+dgx-02; image ghcr.io/ursuciprian/spark-vllm-b12x:b1.4-20261001-b7fbaf96-a7e649d8-warm; boot 602s; served capture sizes [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160]; KV tokens 3615479; status done

Counting (bench_sweep.py: list 1..300, temp 0, thinking off, 320 tokens out, fresh context; 5 rounds per level, each level a separate call). TTFT = the script's prefill probe: c concurrent ~1.5K-token prompts, 8 tokens out.

| c | agg tok/s, max round | median | tokens/step max round / median | per-request tok/s p50 / min | TTFT probe p50 / p99 s |
|---|---|---|---|---|---|
| 1 | 121.9 (max of 5) | 118.9 | 5.00 / 4.92 | 118.9 / 116.3 | 0.75 / 0.75 |
| 8 | 552.4 (max of 5) | 535.3 | 4.98 / 4.97 | 68.6 / 66.1 | 4.46 / 4.51 |
| 16 | 781.9 (max of 5) | 768.8 | 4.97 / 4.96 | 49.1 / 47.5 | 8.69 / 8.76 |
| 32 | 994.4 (max of 5) | 988.3 | 4.97 / 4.96 | 32.0 / 30.6 | 14.45 / 17.58 |

Copy-heavy (bench_copy_streams.py, generated public copy tasks, effort low, max_tokens 1500, 3 rounds per stream count after one warmup; tok/s = completion tokens in the all-decoding window / window).

| streams | window tok/s, max round | median | tokens/step median | per-stream tok/s p50 / min | TTFT p50 / p99 s | running / waiting peak | failed |
|---|---|---|---|---|---|---|---|
| 1 | 116.7 (max of 3) | 115.4 | 4.91 | 115.5 / 114.4 | 1.57 / 1.59 | 1 / 0 | 0 |
| 2 | 189.8 (max of 3) | 188.0 | 4.95 | 92.0 / 90.3 | 0.66 / 1.27 | 2 / 0 | 0 |
| 3 | 230.8 (max of 3) | 222.9 | 4.96 | 73.9 / 68.6 | 1.56 / 1.61 | 3 / 0 | 0 |
| 4 | 305.1 (max of 3) | 291.8 | 4.94 | 72.6 / 66.0 | 1.95 / 2.06 | 4 / 0 | 0 |
| 5 | 331.4 (max of 3) | 323.9 | 4.93 | 64.8 / 60.4 | 2.40 / 2.47 | 5 / 0 | 0 |
| 6 | 353.2 (max of 3) | 348.6 | 4.93 | 56.9 / 53.2 | 2.35 / 3.05 | 6 / 1 | 0 |
| 7 | 402.1 (max of 3) | 397.3 | 4.93 | 55.0 / 51.6 | 2.34 / 3.47 | 7 / 2 | 0 |
| 8 | 439.4 (max of 3) | 430.2 | 4.92 | 52.6 / 48.6 | 2.33 / 3.77 | 8 / 3 | 0 |
| 16 | 644.9 (max of 3) | 644.9 | 4.93 | 37.4 / 34.4 | 4.22 / 7.53 | 16 / 11 | 0 |
| 32 | 910.7 (max of 3) | 909.6 | 4.93 | 25.6 / 23.0 | 7.94 / 15.10 | 32 / 27 | 0 |

Coding (llama-benchy fork, --prompt-mode task, pp2048 tg512, depth 0, T=1.0 top-p 0.95 top-k 20, prefix caching, 3 runs per level). tokens/step = 1 + 4 x accept/draft of the cell.

| c | tg tok/s total, max run | mean | per-request tg tok/s p50 / min | TTFR p50 / p99 ms (pp rows) | tokens/step |
|---|---|---|---|---|---|
| 16 | 232.0 (max of 3) | 227.2 | 21.2 / 13.4 | 6703 / 11004 | 3.28 |
| 32 | 290.7 (max of 3) | 285.5 | 12.9 / 7.9 | 12834 / 22091 | 3.32 |

Straggler probe (counting prompt, 320 tokens, temp 0, ignore_eos, one round per level):

```
c=8 round wall 4.7s | per request (submit order): #0:4.42s/320t #1:4.55s/320t #2:4.55s/320t #3:4.55s/320t #4:4.62s/320t #5:4.62s/320t #6:4.67s/320t #7:4.67s/320t
      preemptions +0 | queue time total 0.12s over 8 req | prefill total 2.10s | decode total 34.6s | accept 3.98/draft
c=16 round wall 6.3s | per request (submit order): #0:5.99s/320t #1:5.99s/320t #2:6.09s/320t #3:6.09s/320t #4:6.09s/320t #5:6.09s/320t #6:6.09s/320t #7:6.08s/320t #8:6.17s/320t #9:6.17s/320t #10:6.17s/320t #11:6.24s/320t #12:6.24s/320t #13:6.29s/320t #14:6.24s/320t #15:6.29s/320t
      preemptions +0 | queue time total 0.76s over 16 req | prefill total 5.39s | decode total 92.9s | accept 3.99/draft
c=32 round wall 10.0s | per request (submit order): #0:9.10s/320t #1:9.10s/320t #2:9.24s/320t #3:9.10s/320t #4:9.10s/320t #5:9.50s/320t #6:9.37s/320t #7:9.37s/320t #8:9.50s/320t #9:9.50s/320t #10:9.50s/320t #11:9.50s/320t #12:9.61s/320t #13:9.61s/320t #14:9.61s/320t #15:9.61s/320t #16:9.72s/320t #17:9.61s/320t #18:9.81s/320t #19:9.72s/320t #20:9.72s/320t #21:9.81s/320t #22:9.81s/320t #23:9.81s/320t #24:9.81s/320t #25:9.89s/320t #26:9.96s/320t #27:9.89s/320t #28:9.96s/320t #29:9.96s/320t #30:9.96s/320t #31:10.01s/320t
      preemptions +0 | queue time total 12.22s over 32 req | prefill total 27.37s | decode total 280.4s | accept 3.99/draft
```

min MemAvailable: mem-dgx02.log 6.58 GiB, mem-dgx01.log 5.19 GiB; preemptions over the arm: metrics.log +0

## v3d-s16: single Spark v3d (checkpoint files at local path f4..03), max_num_seqs 16, CUDA graphs up to 80 rows

node(s) dgx-01; image ghcr.io/ursuciprian/spark-vllm-b12x:tp1-v3d-20261005-21e0b201-5dad364d-warm; boot -; served capture sizes -; KV tokens -; status dropped: MemAvailable < 4 GiB during boot (guard)

min MemAvailable: mem.log 4.00 GiB; preemptions over the arm: -

## v3d-s32: single Spark v3d (checkpoint files at local path f4..03), max_num_seqs 32, CUDA graphs up to 160 rows

node(s) dgx-02; image ghcr.io/ursuciprian/spark-vllm-b12x:tp1-v3d-20261005-21e0b201-5dad364d-warm; boot 685s; served capture sizes [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160]; KV tokens 993754; status done

Counting (bench_sweep.py: list 1..300, temp 0, thinking off, 320 tokens out, fresh context; 5 rounds per level, each level a separate call). TTFT = the script's prefill probe: c concurrent ~1.5K-token prompts, 8 tokens out.

| c | agg tok/s, max round | median | tokens/step max round / median | per-request tok/s p50 / min | TTFT probe p50 / p99 s |
|---|---|---|---|---|---|
| 1 | 84.6 (max of 5) | 82.2 | 5.00 / 4.95 | 82.2 / 79.9 | 1.25 / 1.25 |
| 8 | 366.2 (max of 5) | 361.2 | 4.94 / 4.94 | 46.1 / 43.8 | 6.02 / 6.12 |
| 16 | 522.2 (max of 5) | 510.3 | 4.98 / 4.96 | 33.0 / 31.9 | 11.75 / 11.86 |
| 32 | 675.9 (max of 5) | 666.6 | 4.96 / 4.96 | 21.6 / 20.6 | 22.16 / 23.48 |

Copy-heavy (bench_copy_streams.py, generated public copy tasks, effort low, max_tokens 1500, 3 rounds per stream count after one warmup; tok/s = completion tokens in the all-decoding window / window).

| streams | window tok/s, max round | median | tokens/step median | per-stream tok/s p50 / min | TTFT p50 / p99 s | running / waiting peak | failed |
|---|---|---|---|---|---|---|---|
| 1 | 80.3 (max of 3) | 80.2 | 4.97 | 80.3 / 79.7 | 2.06 / 2.07 | 1 / 0 | 0 |
| 2 | 121.5 (max of 3) | 121.0 | 4.94 | 60.4 / 60.1 | 1.08 / 1.29 | 2 / 0 | 0 |
| 3 | 150.7 (max of 3) | 150.0 | 4.95 | 49.8 / 47.6 | 1.71 / 1.72 | 3 / 0 | 0 |
| 4 | 184.7 (max of 3) | 183.7 | 4.96 | 45.9 / 44.3 | 2.13 / 2.13 | 4 / 0 | 0 |
| 5 | 208.7 (max of 3) | 207.0 | 4.94 | 41.3 / 40.0 | 2.55 / 2.56 | 5 / 0 | 0 |
| 6 | 225.8 (max of 3) | 220.7 | 4.95 | 36.8 / 35.8 | 2.96 / 3.20 | 6 / 1 | 0 |
| 7 | 257.9 (max of 3) | 256.3 | 4.95 | 36.4 / 33.9 | 3.22 / 3.63 | 7 / 2 | 0 |
| 8 | 282.6 (max of 3) | 275.6 | 4.94 | 34.3 / 32.4 | 3.48 / 4.03 | 8 / 3 | 0 |
| 16 | 405.2 (max of 3) | 403.7 | 4.94 | 24.7 / 23.1 | 5.67 / 7.91 | 16 / 11 | 0 |
| 32 | 565.2 (max of 3) | 560.3 | 4.94 | 16.8 / 15.5 | 9.59 / 15.91 | 32 / 27 | 0 |

Coding (llama-benchy fork, --prompt-mode task, pp2048 tg512, depth 0, T=1.0 top-p 0.95 top-k 20, prefix caching, 3 runs per level). tokens/step = 1 + 4 x accept/draft of the cell.

| c | tg tok/s total, max run | mean | per-request tg tok/s p50 / min | TTFR p50 / p99 ms (pp rows) | tokens/step |
|---|---|---|---|---|---|
| 16 | 152.1 (max of 3) | 141.3 | 13.8 / 8.8 | 11714 / 15819 | 3.28 |
| 32 | 198.7 (max of 3) | 189.3 | 8.9 / 5.6 | 19880 / 30982 | 3.37 |

Straggler probe (counting prompt, 320 tokens, temp 0, ignore_eos, one round per level):

```
c=8 round wall 6.9s | per request (submit order): #0:6.65s/320t #1:6.75s/320t #2:6.84s/320t #3:6.75s/320t #4:6.84s/320t #5:6.91s/320t #6:6.84s/320t #7:6.91s/320t
      preemptions +0 | queue time total 0.37s over 8 req | prefill total 3.21s | decode total 51.3s | accept 3.98/draft
c=16 round wall 10.0s | per request (submit order): #0:9.52s/320t #1:9.37s/320t #2:9.66s/320t #3:9.66s/320t #4:9.66s/320t #5:9.66s/320t #6:9.79s/320t #7:9.79s/320t #8:9.66s/320t #9:9.79s/320t #10:9.79s/320t #11:9.79s/320t #12:9.97s/320t #13:9.89s/320t #14:9.89s/320t #15:9.97s/320t
      preemptions +0 | queue time total 3.44s over 16 req | prefill total 10.82s | decode total 145.0s | accept 3.98/draft
c=32 round wall 15.1s | per request (submit order): #0:13.74s/320t #1:13.74s/320t #2:13.74s/320t #3:13.96s/320t #4:13.96s/320t #5:13.96s/320t #6:14.18s/320t #7:14.18s/320t #8:14.18s/320t #9:14.18s/320t #10:14.18s/320t #11:14.38s/320t #12:14.56s/320t #13:14.38s/320t #14:14.38s/320t #15:14.55s/320t #16:14.55s/320t #17:14.71s/320t #18:14.71s/320t #19:14.71s/320t #20:14.85s/320t #21:14.71s/320t #22:14.85s/320t #23:14.71s/320t #24:14.97s/320t #25:14.85s/320t #26:14.85s/320t #27:14.97s/320t #28:15.05s/320t #29:14.97s/320t #30:14.97s/320t #31:15.05s/320t
      preemptions +0 | queue time total 20.21s over 32 req | prefill total 39.45s | decode total 424.3s | accept 3.99/draft
```

min MemAvailable: mem.log 4.46 GiB; preemptions over the arm: metrics.log +0

