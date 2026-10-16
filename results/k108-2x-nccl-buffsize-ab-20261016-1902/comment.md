k108 result: I tested two NCCL settings on the 2x, `NCCL_BUFFSIZE=1048576` and `NCCL_MAX_CTAS=8`, against the 2x v2.2.0 on the same image, TP=2, 4 boots per side in the order ctl arm arm ctl ctl arm arm ctl. Only prefill all-reduces go through NCCL on this stack (above the 2MB RoCE limit, so every pp2048 and cold-prompt chunk); decode stays on RoCE, so its cells are guards. Measure only; nothing ships from this job.

`RULE2=FLAT`. Main cells: pp2048 at c1 and cold 16K TTFT.
Nothing is worse beyond noise, but neither pp2048 nor cold 16K is faster beyond noise in this run.

```
k108 2026-10-16 21:02:28 EEST: NCCL_BUFFSIZE=1048576 NCCL_MAX_CTAS=8 on the 2x v2.2.0 (ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm) vs the same image without them, TP=2, 4 boots per side
RULE2=FLAT
RoCE all-reduce live line in 8 of 8 screen boots (decode stays on RoCE, prefill above 2MB on NCCL)

k108 2x 2026-10-16 19:02:04 EEST: arm ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm NCCL_BUFFSIZE=1048576 NCCL_MAX_CTAS=8 vs 2x v2.2.0 (ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm)
bake: prebake boot 172s Directly load AOT compilation; full boot 172s Directly load AOT compilation; rank 0 measured 0
screen (ctl:1 arm:1 arm:2 ctl:2 ctl:3 arm:3 arm:4 ctl:4):
main cells (can make WIN): pp2048 (c1), 16K
== cold TTFT (s, mean of 3 per boot); noise = larger boot-to-boot spread of either side (floor 0.5%); CI guard: whole boot-level 95% CI above +1% is worse
  16K: ctl 5.47 5.51 5.48 5.53 | arm 5.49 5.54 5.50 5.58 -> +0.5% (+0.03 s; noise 1.6%; CI [-0.7,+1.7]) flat
  64K: ctl 23.89 23.98 23.92 24.17 | arm 23.86 24.01 23.93 24.02 -> -0.2% (-0.04 s; noise 1.2%; CI [-1.0,+0.7]) flat
  128K: ctl 53.68 53.88 53.79 54.18 | arm 53.66 53.96 53.76 54.23 -> +0.0% (+0.02 s; noise 1.1%; CI [-0.8,+0.8]) flat
== llama-benchy pp2048 + tg512 (tok/s, side means; band = max(k92 band, side spread); CI guard: whole boot-level 95% CI below -1% is worse)
  pp2048 (c1): ctl 2839.3 arm 2841.6 +0.1% (band 6.9%; CI [-4.81,+4.97])
  tg512 (c1): ctl 79.5 arm 84.4 +6.2% (band 11.9%; CI [-3.02,+15.32])
  tg512 (c4): ctl 168.9 arm 167.9 -0.6% (band 19.0%; CI [-14.69,+13.50])
  tg512 (c8): ctl 195.2 arm 193.9 -0.7% (band 5.6%; CI [-4.70,+3.38])
== fresh-c1 T=0 probe, pass K of the arm paired with pass K of the control (decode tok/s, stalls excluded)
  fresh-c1: -1.33% (noise 7.76%; n=16 CI [-4.2,+2.0] control boot-to-boot -7.76%, step +0.19%) flat
== MTP acceptance per draft position (fresh c8, T=0), pooled per side
  ctl: 0.853 0.704 0.575 0.468
  arm: 0.849 0.699 0.571 0.466
== path: arm pairs NCCL_BUFFSIZE=1048576 NCCL_MAX_CTAS=8 on both ranks, images; KV pool per boot
  ctl-p1: pairs 0/2 on both ranks; image ok; KV 3531734 tokens, 29.35 GiB available for KV
  ctl-p2: pairs 0/2 on both ranks; image ok; KV 3518793 tokens, 29.24 GiB available for KV
  ctl-p3: pairs 0/2 on both ranks; image ok; KV 3528591 tokens, 29.32 GiB available for KV
  ctl-p4: pairs 0/2 on both ranks; image ok; KV 3523045 tokens, 29.28 GiB available for KV
  arm-p1: pairs 2/2 on both ranks; image ok; KV 3532843 tokens, 29.36 GiB available for KV
  arm-p2: pairs 2/2 on both ranks; image ok; KV 3539868 tokens, 29.42 GiB available for KV
  arm-p3: pairs 2/2 on both ranks; image ok; KV 3532288 tokens, 29.36 GiB available for KV
  arm-p4: pairs 2/2 on both ranks; image ok; KV 3539868 tokens, 29.42 GiB available for KV
  KV pool arm vs ctl: +0.3% tokens, +0.09 GiB per rank
better: none; in the main cells: none
RULE2=FLAT
```

Results: {RESULTS_URL}
