Reference baseline (k84b-reference-baseline): the checkpoint publisher's reference vLLM image for DGX Spark, `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-spark-20261007-d071fb3dde0ba979` (`sha256:af14f1a90e141ad7a1b4951f0fd37d695101ebd5887fcddb499f2f427eff5319`, linux/arm64, the newest Spark tag on 2026-10-10), with its own two-Spark preset `qwen38-dgx-spark-x2` (TP2 over RoCE, one container per Spark, MTP 3 probabilistic drafts, 16 sequences, utilization 0.75), on the unmodified `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `7c4f1bc1`.

What I set: the checkpoint revision (`MODEL_REVISION=7c4f1bc1`, offline; the preset leaves it unpinned and the Hub main is now a newer revision), the per-deployment RoCE settings the hardware profile asks for (`NCCL_IB_HCA`, socket interfaces, host IP), the API name (`qwen3.8-flash-next`, which my harness sends; the launcher takes one name) and container names. No serving option was tuned. The `mtp4` setup adds only `--draft-tokens 4`, the draft count of my recipes. There is no single-Spark Qwen preset from the publisher (its preset notes say one box cannot hold the weights), so this is 2x only. The resolved launcher config of each setup is in `print-config.txt`.

Boot attempts:
```
2026-10-11 01:04:33 GPU check: cuda 13.4 True NVIDIA GB10
2026-10-11 01:10:43 ref-2x-mtp3: booted (boot_s 368 pong pong )
2026-10-11 01:32:18 ref-2x-mtp4: booted (boot_s 222 pong pong )
```

Cells and metrics are the same as k86 and k84: llama-benchy pp2048/tg512 task mode, 3 runs, c1 and c8, at T=1.0 top-p 0.95 top-k 20 with thinking on (`tgdef`) and at T=0 with thinking off (`tgt0`); the 36-prompt coding probe at T=0 with thinking off, c1 and c8, one pass. GPU power is nvidia-smi power.draw at 1 Hz on both Sparks (GPU only, not wall power). The last table lists my shipped builds from k86 (3 boots each) next to these runs.

```
Results: /home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k84b-reference-baseline-20261011-0104

== ref-2x-mtp3
cell                      hosts        units  tok/s mean  sd boots   cv % sd within   GPU W  tok/s/W
coding-t0-nothink-c1      dgx01+dgx02      1        62.7      0.00    0.0         -    47.2    1.328
coding-t0-nothink-c8      dgx01+dgx02      1       225.8      0.00    0.0         -    44.7    5.057
idle                      dgx01+dgx02      1           -         -      -         -    18.8        -
tgdef-c1                  dgx01+dgx02      1        51.6      0.00    0.0      2.94    46.0    1.121
tgdef-c8                  dgx01+dgx02      1        95.7      0.00    0.0      9.57    47.7    2.006
tgt0-c1                   dgx01+dgx02      1        71.1      0.00    0.0      4.78    48.1    1.478
tgt0-c8                   dgx01+dgx02      1        88.5      0.00    0.0      5.74    48.2    1.837

== ref-2x-mtp4
cell                      hosts        units  tok/s mean  sd boots   cv % sd within   GPU W  tok/s/W
coding-t0-nothink-c1      dgx01+dgx02      1        64.5      0.00    0.0         -    49.2    1.311
coding-t0-nothink-c8      dgx01+dgx02      1       233.6      0.00    0.0         -    46.5    5.027
idle                      dgx01+dgx02      1           -         -      -         -    19.4        -
tgdef-c1                  dgx01+dgx02      1        50.2      0.00    0.0      1.67    47.6    1.056
tgdef-c8                  dgx01+dgx02      1        93.7      0.00    0.0     11.11    48.6    1.928
tgt0-c1                   dgx01+dgx02      1        65.7      0.00    0.0      8.40    49.5    1.328
tgt0-c8                   dgx01+dgx02      1        88.7      0.00    0.0     11.58    49.1    1.805

== same cells, this run against /home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k86-boot-noise-power-20261010-1704
coding-t0-nothink-c1      1x-v2.2.0/dgx01: 75.2 (sd 0.17, 3 boots) | 1x-v2.2.0/dgx02: 74.7 (sd 0.23, 3 boots) | 1x-v2.2.0/pooled: 74.9 (sd 0.31, 6 boots) | 2x-v2.0.0/dgx01+dgx02: 112.8 (sd 0.02, 3 boots) | ref-2x-mtp3/dgx01+dgx02: 62.7 | ref-2x-mtp4/dgx01+dgx02: 64.5
coding-t0-nothink-c8      1x-v2.2.0/dgx01: 191.7 (sd 1.07, 3 boots) | 1x-v2.2.0/dgx02: 192.2 (sd 1.10, 3 boots) | 1x-v2.2.0/pooled: 192.0 (sd 1.02, 6 boots) | 2x-v2.0.0/dgx01+dgx02: 317.4 (sd 1.34, 3 boots) | ref-2x-mtp3/dgx01+dgx02: 225.8 | ref-2x-mtp4/dgx01+dgx02: 233.6
tgdef-c1                  1x-v2.2.0/dgx01: 61.4 (sd 2.55, 3 boots) | 1x-v2.2.0/dgx02: 59.4 (sd 4.05, 3 boots) | 1x-v2.2.0/pooled: 60.4 (sd 3.22, 6 boots) | 2x-v2.0.0/dgx01+dgx02: 86.3 (sd 2.37, 3 boots) | ref-2x-mtp3/dgx01+dgx02: 51.6 | ref-2x-mtp4/dgx01+dgx02: 50.2
tgdef-c8                  1x-v2.2.0/dgx01: 139.2 (sd 1.71, 3 boots) | 1x-v2.2.0/dgx02: 134.2 (sd 2.39, 3 boots) | 1x-v2.2.0/pooled: 136.7 (sd 3.30, 6 boots) | 2x-v2.0.0/dgx01+dgx02: 195.3 (sd 5.12, 3 boots) | ref-2x-mtp3/dgx01+dgx02: 95.7 | ref-2x-mtp4/dgx01+dgx02: 93.7
tgt0-c1                   1x-v2.2.0/dgx01: 72.4 (sd 3.16, 3 boots) | 1x-v2.2.0/dgx02: 73.3 (sd 0.27, 3 boots) | 1x-v2.2.0/pooled: 72.8 (sd 2.07, 6 boots) | 2x-v2.0.0/dgx01+dgx02: 108.3 (sd 3.05, 3 boots) | ref-2x-mtp3/dgx01+dgx02: 71.1 | ref-2x-mtp4/dgx01+dgx02: 65.7
tgt0-c8                   1x-v2.2.0/dgx01: 150.7 (sd 2.83, 3 boots) | 1x-v2.2.0/dgx02: 147.8 (sd 11.26, 3 boots) | 1x-v2.2.0/pooled: 149.2 (sd 7.52, 6 boots) | 2x-v2.0.0/dgx01+dgx02: 227.4 (sd 5.02, 3 boots) | ref-2x-mtp3/dgx01+dgx02: 88.5 | ref-2x-mtp4/dgx01+dgx02: 88.7

tok/s = llama-benchy tg_throughput (aggregate over the concurrent requests, reasoning tokens counted) or coding.py aggregate output tok/s per pass; mean of the runs/passes of a boot, then of the boots. GPU W = nvidia-smi power.draw mean over the cell window (both Sparks added for 2x); idle = 120 s with the server up.
```

Results: {RESULTS_URL}
