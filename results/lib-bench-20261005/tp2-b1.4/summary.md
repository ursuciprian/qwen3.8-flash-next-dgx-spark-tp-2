## tp2-b1.4

kv_tokens 3662251

llm-inference-bench decode (30 s cells): aggregate tok/s, tokens per step (server), effective concurrency

| ctx | c | agg tok/s | per-request tok/s | tokens/step | eff. conc | ttft p50 s | note |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 64.8 | 64.8 | 2.69 | 1.0 | 0.12 |  |
| 16384 | 1 | 73.3 | 73.3 | 2.86 | 1.0 | 0.76 |  |
| 65536 | 1 | 77.8 | 77.8 | 3.08 | 1.0 | 0.93 |  |
| 0 | 4 | 171.2 | 42.8 | 2.79 | 4.0 | 0.26 |  |
| 0 | 8 | 248.6 | 31.1 | 2.82 | 8.0 | 0.45 |  |
| 16384 | 4 | 175.0 | 43.8 | 2.92 | 4.0 | 2.67 |  |
| 16384 | 8 | 254.9 | 31.9 | 2.97 | 8.0 | 3.21 |  |
| 65536 | 4 | 165.4 | 41.3 | 2.85 | 4.0 | 2.73 |  |
| 65536 | 8 | 257.4 | 32.2 | 3.01 | 8.0 | 3.25 |  |

prefill: {"8192": {"ttft_seconds": 2.868, "prefill_seconds": 2.868, "tok_per_sec": 2857.0, "client_ttft_seconds": 2.868, "client_tok_per_sec": 2857.0, "prompt_tokens": 8194, "samples": 4, "method": "client", "server_validation": {"method": "", "tok_per_sec": 0.0, "prefill_seconds": 0.0, "prompt_tokens": 0, "request_prompt_tokens": 0, "cached_tokens": 0, "token_source": "", "samples": 0, "invalid_reason": ""}, "hardware_summary": {"samples": 6, "duration_seconds": 10.54, "gpu_count": 1, "cpu_util_avg_pct": 15.6, "cpu_temp_max_c": 0.0, "gpu_util_avg_pct": 96.0, "gpu_util_max_pct": 96.0, "mem_util_avg_pct": 0.0, "mem_util_max_pct": 0.0, "temp_avg_c": 46.0, "temp_max_c": 47.0, "power_total_avg_w": 35.42, "power_total_max_w": 36.71, "power_limit_total_w": 0.0, "vram_used_avg_mb": 0.0, "vram_used_max_mb": 0.0, "vram_total_mb": 0.0, "vram_used_avg_pct": 0.0, "vram_used_max_pct": 0.0, "pcie_rx_avg_mb_s": 0.0, "pcie_rx_max_mb_s": 0.0, "pcie_tx_avg_mb_s": 0.0, "pcie_tx_max_mb_s": 0.0}}, "16384": {"ttft_seconds": 5.536, "prefill_seconds": 5.536, "tok_per_sec": 2931.0, "client_ttft_seconds": 5.536, "client_tok_per_sec": 2931.0, "prompt_tokens": 16224, "samples": 2, "method": "client", "server_validation": {"method": "", "tok_per_sec": 0.0, "prefill_seconds": 0.0, "prompt_tokens": 0, "request_prompt_tokens": 0, "cached_tokens": 0, "token_source": "", "samples": 0, "invalid_reason": ""}, "hardware_summary": {"samples": 6, "duration_seconds": 10.589, "gpu_count": 1, "cpu_util_avg_pct": 13.93, "cpu_t

whole decode run tokens/step 2.96, hotel run 4.09

hotel-lights x8: correct_regex=""; correct=null; correct=8; correct_rate=1.0; correct_rate_completed=1.0; correct=8; correct_rate=1.0; correct_rate_completed=1.0

counting sweep (5 rounds each; median / max agg tok/s; per-round agg/tokens per step)

| c | median | max | rounds | tokens/step (whole level) |
|---|---|---|---|---|
| 1 | 119.5 | 122.4 | 119.5/4.954 119.0/4.954 119.3/4.954 122.4/5.0 120.7/5.0 | 4.97 |
| 2 | 211.1 | 221.7 | 221.6/4.954 221.7/5.0 211.1/4.977 210.8/4.977 204.1/4.931 | 4.96 |
| 4 | 350.0 | 389.8 | 348.6/4.912 370.0/4.961 350.0/4.95 389.8/5.0 349.8/4.916 | 4.95 |
| 8 | 544.1 | 558.0 | 558.0/4.994 544.1/4.94 532.5/4.965 538.4/4.971 549.2/4.961 | 4.96 |
| 16 | 787.3 | 795.3 | 776.2/4.934 795.3/4.965 788.9/4.968 777.0/4.962 787.3/4.937 | 4.96 |

/metrics sampler: KV use max 0.160 mean 0.054 over 716 samples; preemptions +0

