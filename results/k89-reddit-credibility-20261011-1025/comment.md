Reference stack against my builds, same Sparks, same harnesses (k89-reddit-credibility, with k84b and k86).

Reference: the checkpoint publisher's vLLM image for DGX Spark, `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-spark-20261007-d071fb3dde0ba979` (`sha256:af14f1a90e141ad7a1b4951f0fd37d695101ebd5887fcddb499f2f427eff5319`), with its two-Spark preset `qwen38-dgx-spark-x2` unchanged (TP2 over RoCE, MTP 3 probabilistic drafts, 16 sequences, utilization 0.75), on `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `7c4f1bc1`. I set only the checkpoint revision, the RoCE interface settings, the API name and the container names. Three boots: one from k84b and two from this job (2 of 2 came up).

Mine: 2x v2.0.0 and 1x v2.2.0 from k86 (3 boots of each on each Spark set). Cells: llama-benchy pp2048/tg512 task mode, 3 runs per boot; the 36-prompt coding probe, aggregate output tok/s (one pass per reference boot, 3 passes per boot in k86). Values are tok/s, mean of the per-boot means, with the standard deviation between boots in parentheses.

| Cell | Reference 2x, mean (sd), boots | My 2x v2.0.0, mean (sd), 3 boots | 2x / reference | My 1x v2.2.0, mean (sd), 6 boots | 1x / reference |
|---|---|---|---|---|---|
| tg512 c1, T=1.0 thinking on | 51.8 (0.91), 3 | 86.3 (2.37) | 1.67 | 60.4 (3.22) | 1.17 |
| tg512 c8, T=1.0 thinking on | 96.6 (2.33), 3 | 195.3 (5.12) | 2.02 | 136.7 (3.30) | 1.41 |
| tg512 c1, T=0 thinking off | 67.9 (2.76), 3 | 108.3 (3.05) | 1.60 | 72.8 (2.07) | 1.07 |
| tg512 c8, T=0 thinking off | 89.5 (6.19), 3 | 227.4 (5.02) | 2.54 | 149.2 (7.52) | 1.67 |
| coding-36 c1, T=0 | 63.1 (0.43), 3 | 112.8 (0.02) | 1.79 | 74.9 (0.31) | 1.19 |
| coding-36 c8, T=0 | 226.4 (0.71), 3 | 317.4 (1.34) | 1.40 | 192.0 (1.02) | 0.85 |
| GPU power idle, W (both Sparks) | 19.2 | 19.5 | | | |

IFEval, all 541 prompts, lm-eval 0.4.13, 16 requests at a time, T=1.0 top-p 0.95 top-k 20, thinking on, reasoning effort medium sent in every request (the reference preset has no server default). Paired per prompt; McNemar p is exact and two-sided; instruction-level accuracy is pooled over all instructions, so it has no paired test.

```
eval:metric                                        n       mine-2x     reference   discordant (mine-2x only / reference only)   McNemar p
ifeval_think_med:prompt_level_loose_acc          541        90.20%        89.83%       19 / 17                      0.868
ifeval_think_med:prompt_level_strict_acc         541        87.43%        87.06%       21 / 19                      0.875
ifeval_think_med:inst_level_strict_acc           834        91.61%        91.13%                                    -
ifeval_think_med:inst_level_loose_acc            834        93.41%        93.05%                                    -
```

Full cell tables with GPU power are in `pooled/cred.txt`.

Results: {RESULTS_URL}
