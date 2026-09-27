# c1-c4 decode step profile, b1.1 (2026-09-26, opus-executor-3)

Image b1.1-20260926-b7fbaf96-6d232f16-warm + mods/vllm-decode-profiler, rank0, 60-step windows,
scripts/step_breakdown.py. Values are ms per decode step and % of the step wall.
Kernel groups are split by CUDA graph (target forward / MTP draft graphs / eager).

| window | step ms | NVFP4 MoE | dense MXFP8/NVFP4 | BF16 GEMM (HC mixers, router) | MTP draft passes | MTP lm_head | main lm_head | RoCE all-reduce | GDN | QSA select | attention | idle/launch gaps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fresh c1 | 47.8 | 13.94 (29%) | 9.75 (20%) | 9.13 (19%) | 5.59 (12%) | 2.89 (6%) | 1.38 (3%) | 1.70 (4%) | 1.26 (3%) | 0.12 (0.3%) | 0.36 | 2.19 (5%) |
| fresh c4 | 75.9 | 33.08 (44%) | 10.28 (14%) | 10.16 (13%) | 7.02 (9%) | 3.22 (4%) | 1.86 (2%) | 2.81 (4%) | 2.44 (3%) | 0.32 (0.4%) | 1.01 | 2.96 (4%) |
| 16k c1 | 48.0 | 13.14 (27%) | 9.74 (20%) | 9.12 (19%) | 5.64 (12%) | 2.89 (6%) | 1.38 (3%) | 1.40 (3%) | 1.22 (3%) | 0.19 (0.4%) | 0.40 | 3.24 (7%) |
| 16k c2 | 57.8 | 21.14 (37%) | 11.19 (19%) | 9.31 (16%) | 6.27 (11%) | 2.97 (5%) | 1.42 (2%) | 1.87 (3%) | 1.50 (3%) | 0.30 (0.5%) | 0.63 | 2.96 (5%) |
| 16k c4 | 76.4 | 33.13 (43%) | 10.38 (14%) | 10.08 (13%) | 7.03 (9%) | 3.22 (4%) | 1.87 (2%) | 2.80 (4%) | 2.45 (3%) | 0.57 (0.7%) | 1.09 | 3.11 (4%) |
| 64k c1 | 48.7 | 13.06 (27%) | 9.77 (20%) | 9.11 (19%) | 5.68 (12%) | 2.90 (6%) | 1.39 (3%) | 1.39 (3%) | 1.24 (3%) | 0.66 (1.4%) | 0.45 | 3.41 (7%) |
| 128k c1 | 48.5 | 12.80 (26%) | 9.80 (20%) | 9.08 (19%) | 5.79 (12%) | 2.90 (6%) | 1.39 (3%) | 1.46 (3%) | 1.23 (3%) | 1.33 (2.7%) | 0.53 | 2.80 (6%) |

fresh c2 (77 ms) is excluded: the window caught the second request's prefill (all-reduce 19 ms of peer wait).
Sampling/rejection is 0.3-0.6 ms and norm 0.8-1.0 ms at every point.
