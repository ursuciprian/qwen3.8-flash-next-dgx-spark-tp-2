## bench

kv_tokens 993754

llm-inference-bench decode (30 s cells): aggregate tok/s, tokens per step (server), effective concurrency

| ctx | c | agg tok/s | per-request tok/s | tokens/step | eff. conc | ttft p50 s | note |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 43.7 | 43.7 | 2.67 | 1.0 | 0.19 |  |
| 16384 | 1 | 43.0 | 43.0 | 2.66 | 1.0 | 0.73 |  |
| 65536 | 1 | 43.3 | 43.3 | 2.68 | 1.0 | 0.82 |  |
| 0 | 4 | 106.2 | 26.5 | 2.87 | 4.0 | 0.43 |  |
| 0 | 8 | 165.7 | 20.7 | 3.12 | 8.0 | 0.66 |  |
| 16384 | 4 | 116.9 | 29.2 | 3.16 | 4.0 | 2.32 |  |
| 16384 | 8 | 165.6 | 20.7 | 3.28 | 8.0 | 3.73 |  |
| 65536 | 4 | 113.9 | 28.5 | 3.12 | 4.0 | 2.41 |  |
| 65536 | 8 | 160.4 | 20.1 | 3.18 | 8.0 | 3.87 |  |

prefill: {"8192": {"ttft_seconds": 3.8, "prefill_seconds": 3.8, "tok_per_sec": 2157.0, "client_ttft_seconds": 3.8, "client_tok_per_sec": 2157.0, "prompt_tokens": 8194, "samples": 3, "method": "client", "server_validation": {"method": "", "tok_per_sec": 0.0, "prefill_seconds": 0.0, "prompt_tokens": 0, "request_prompt_tokens": 0, "cached_tokens": 0, "token_source": "", "samples": 0, "invalid_reason": ""}, "hardware_summary": {"samples": 6, "duration_seconds": 10.564, "gpu_count": 1, "cpu_util_avg_pct": 5.07, "cpu_temp_max_c": 0.0, "gpu_util_avg_pct": 96.0, "gpu_util_max_pct": 96.0, "mem_util_avg_pct": 0.0, "mem_util_max_pct": 0.0, "temp_avg_c": 50.67, "temp_max_c": 51.0, "power_total_avg_w": 40.84, "power_total_max_w": 41.93, "power_limit_total_w": 0.0, "vram_used_avg_mb": 0.0, "vram_used_max_mb": 0.0, "vram_total_mb": 0.0, "vram_used_avg_pct": 0.0, "vram_used_max_pct": 0.0, "pcie_rx_avg_mb_s": 0.0, "pcie_rx_max_mb_s": 0.0, "pcie_tx_avg_mb_s": 0.0, "pcie_tx_max_mb_s": 0.0}}, "16384": {"ttft_seconds": 7.343, "prefill_seconds": 7.343, "tok_per_sec": 2209.0, "client_ttft_seconds": 7.343, "client_tok_per_sec": 2209.0, "prompt_tokens": 16224, "samples": 2, "method": "client", "server_validation": {"method": "", "tok_per_sec": 0.0, "prefill_seconds": 0.0, "prompt_tokens": 0, "request_prompt_tokens": 0, "cached_tokens": 0, "token_source": "", "samples": 0, "invalid_reason": ""}, "hardware_summary": {"samples": 8, "duration_seconds": 14.732, "gpu_count": 1, "cpu_util_avg_pct": 4.96, "cpu_temp_m

whole decode run tokens/step 3.10, hotel run 4.08

hotel-lights x8: correct_regex=""; correct=null; correct=5; correct_rate=0.625; correct_rate_completed=0.625; correct=5; correct_rate=0.625; correct_rate_completed=0.625

counting sweep (5 rounds each; median / max agg tok/s; per-round agg/tokens per step)

| c | median | max | rounds | tokens/step (whole level) |
|---|---|---|---|---|
| 1 | 76.4 | 76.9 | 76.4/5.0 76.9/5.0 75.4/4.954 75.1/4.864 76.6/5.0 | 4.97 |
| 2 | 135.1 | 142.4 | 135.1/4.977 134.0/4.977 128.9/4.931 137.6/4.969 142.4/5.0 | 4.97 |
| 4 | 232.5 | 243.8 | 216.0/4.988 231.1/4.988 243.8/5.0 232.5/4.988 234.7/4.942 | 4.98 |
| 8 | 357.8 | 360.1 | 330.2/4.942 333.8/4.954 357.8/4.977 360.1/4.971 360.1/4.988 | 4.97 |

/metrics sampler: KV use max 0.281 mean 0.084 over 2105 samples; preemptions +0

min MemAvailable GiB 14.70
