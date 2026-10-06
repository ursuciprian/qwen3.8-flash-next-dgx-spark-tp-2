## bench

kv_tokens 993754

llm-inference-bench decode (30 s cells): aggregate tok/s, tokens per step (server), effective concurrency

| ctx | c | agg tok/s | per-request tok/s | tokens/step | eff. conc | ttft p50 s | note |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 47.4 | 47.4 | 2.63 | 1.0 | 0.19 |  |
| 16384 | 1 | 51.2 | 51.2 | 2.85 | 1.0 | 0.73 |  |
| 65536 | 1 | 56.9 | 56.9 | 3.21 | 1.0 | 0.82 |  |
| 0 | 4 | 116.6 | 29.1 | 2.99 | 4.0 | 0.43 |  |
| 0 | 8 | 173.4 | 21.7 | 3.14 | 8.0 | 0.68 |  |
| 16384 | 4 | 112.6 | 28.1 | 2.95 | 4.0 | 2.34 |  |
| 16384 | 8 | 191.5 | 23.9 | 3.45 | 8.0 | 3.78 |  |
| 65536 | 4 | 111.8 | 28.0 | 2.89 | 4.0 | 2.42 |  |
| 65536 | 8 | 179.5 | 22.4 | 3.27 | 8.0 | 3.89 |  |

prefill: {"8192": {"ttft_seconds": 3.835, "prefill_seconds": 3.835, "tok_per_sec": 2137.0, "client_ttft_seconds": 3.835, "client_tok_per_sec": 2137.0, "prompt_tokens": 8193, "samples": 3, "method": "client", "server_validation": {"method": "", "tok_per_sec": 0.0, "prefill_seconds": 0.0, "prompt_tokens": 0, "request_prompt_tokens": 0, "cached_tokens": 0, "token_source": "", "samples": 0, "invalid_reason": ""}, "hardware_summary": {"samples": 6, "duration_seconds": 10.487, "gpu_count": 1, "cpu_util_avg_pct": 5.18, "cpu_temp_max_c": 0.0, "gpu_util_avg_pct": 94.0, "gpu_util_max_pct": 96.0, "mem_util_avg_pct": 0.0, "mem_util_max_pct": 0.0, "temp_avg_c": 50.33, "temp_max_c": 52.0, "power_total_avg_w": 38.98, "power_total_max_w": 40.01, "power_limit_total_w": 0.0, "vram_used_avg_mb": 0.0, "vram_used_max_mb": 0.0, "vram_total_mb": 0.0, "vram_used_avg_pct": 0.0, "vram_used_max_pct": 0.0, "pcie_rx_avg_mb_s": 0.0, "pcie_rx_max_mb_s": 0.0, "pcie_tx_avg_mb_s": 0.0, "pcie_tx_max_mb_s": 0.0}}, "16384": {"ttft_seconds": 7.435, "prefill_seconds": 7.435, "tok_per_sec": 2182.0, "client_ttft_seconds": 7.435, "client_tok_per_sec": 2182.0, "prompt_tokens": 16223, "samples": 2, "method": "client", "server_validation": {"method": "", "tok_per_sec": 0.0, "prefill_seconds": 0.0, "prompt_tokens": 0, "request_prompt_tokens": 0, "cached_tokens": 0, "token_source": "", "samples": 0, "invalid_reason": ""}, "hardware_summary": {"samples": 8, "duration_seconds": 14.726, "gpu_count": 1, "cpu_util_avg_pct": 4.94, "cpu_

whole decode run tokens/step 3.13, hotel run 4.05

hotel-lights x8: correct_regex=""; correct=null; correct=6; correct_rate=0.75; correct_rate_completed=0.75; correct=6; correct_rate=0.75; correct_rate_completed=0.75

counting sweep (5 rounds each; median / max agg tok/s; per-round agg/tokens per step)

| c | median | max | rounds | tokens/step (whole level) |
|---|---|---|---|---|
| 1 | 83.0 | 84.8 | 84.0/5.0 83.0/4.954 82.8/4.954 80.9/4.821 84.8/5.0 | 4.96 |
| 2 | 145.1 | 154.9 | 142.3/4.961 153.7/5.0 145.1/4.977 141.2/4.946 154.9/5.0 | 4.96 |
| 4 | 235.3 | 239.8 | 237.9/4.908 230.5/4.916 232.5/4.935 235.3/4.954 239.8/4.95 | 4.93 |
| 8 | 368.3 | 377.8 | 352.4/4.918 368.3/4.971 362.6/4.906 368.7/4.954 377.8/4.971 | 4.95 |

/metrics sampler: KV use max 0.281 mean 0.078 over 2051 samples; preemptions +0

min MemAvailable GiB 13.30
