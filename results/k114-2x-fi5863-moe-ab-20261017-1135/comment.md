k114 result: I tested FlashInfer's SM12x CuTe DSL W4A4 MoE (flashinfer-ai/flashinfer#5863) for the routed experts in place of b12x on the 2x against the 2x v2.2.0, TP=2, 4 boots per side in the order ctl arm arm ctl ctl arm arm ctl. The arm image is the v2.2.0 image with vLLM `0aefd5eef6` (`VLLM_FI_SM12X_W4A4_MOE=1`, trunk layers only; the MTP drafter stays on b12x) and FlashInfer `38e6b8944f` installed next to the image's FlashInfer. The k109 layer probe measured x1.13 decode and x1.26 prefill for the MoE layer at this shape; the outputs differ from b12x's by about each one's own rounding error, so acceptance per draft position is the canary and the release gate runs only after a WIN. Measure only; nothing ships from this job.

`RULE2=FLAT`. Main cells: tg512 at c1, c4 and c8, the fresh-c1 paired probe, pp2048 and cold TTFT at 16K, 64K and 128K. KV rule: a KV pool more than 3% smaller than the control's is a loss unless all three tg512 cells are better with a mean gain of at least 10%.
Nothing is worse beyond noise, but no decode cell is faster beyond noise in this run.

```
k114 2026-10-17 13:40:10 EEST: FlashInfer SM12x W4A4 routed MoE (#5863) on the 2x (ghcr.io/ursuciprian/spark-vllm-b12x:k114-20261017-d717da5a-38e6b894-0aefd5ee: vLLM 0aefd5ee, flashinfer_k114 38e6b894, b12x d717da5a, VLLM_FI_SM12X_W4A4_MOE=1) vs 2x v2.2.0 (ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm), TP=2, 4 boots per side
RULE2=FLAT

k114 2x 2026-10-17 11:37:19 EEST: arm ghcr.io/ursuciprian/spark-vllm-b12x:k114-20261017-d717da5a-38e6b894-0aefd5ee VLLM_FI_SM12X_W4A4_MOE=1 vs 2x v2.2.0 (ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm)
bake: prebake pong: pong; full boot 181s Directly load AOT compilation; rank 0 measured 0
screen (ctl:1 arm:1 arm:2 ctl:2 ctl:3 arm:3 arm:4 ctl:4):
main cells (can make WIN): tg512 (c1), tg512 (c4), tg512 (c8), fresh-c1, pp2048 (c1), 16K, 64K, 128K
== cold TTFT (s, mean of 3 per boot); noise = larger boot-to-boot spread of either side (floor 0.5%); CI guard: whole boot-level 95% CI above +1% is worse
  16K: ctl 5.44 5.45 5.44 5.49 | arm 5.50 5.63 5.53 5.46 -> +1.4% (+0.08 s; noise 3.1%; CI [-0.8,+3.6]) flat
  64K: ctl 24.13 24.13 23.93 23.90 | arm 24.00 24.05 24.02 24.12 -> +0.1% (+0.03 s; noise 1.0%; CI [-0.7,+0.9]) flat
  128K: ctl 53.72 53.92 53.94 53.96 | arm 53.87 54.07 53.95 54.14 -> +0.2% (+0.12 s; noise 0.5%; CI [-0.2,+0.6]) flat
== llama-benchy pp2048 + tg512 (tok/s, side means; band = max(k92 band, side spread); CI guard: whole boot-level 95% CI below -1% is worse)
  pp2048 (c1): ctl 2813.6 arm 2861.6 +1.7% (band 5.8%; CI [-3.13,+6.54])
  tg512 (c1): ctl 82.3 arm 77.6 -5.7% (band 20.0%; CI [-20.42,+8.98])
  tg512 (c4): ctl 154.4 arm 149.3 -3.3% (band 10.7%; CI [-10.19,+3.59])
  tg512 (c8): ctl 188.5 arm 188.1 -0.2% (band 5.8%; CI [-4.68,+4.25])
== fresh-c1 T=0 probe, pass K of the arm paired with pass K of the control (decode tok/s, stalls excluded)
  fresh-c1: -1.30% (noise 10.35%; n=16 CI [-5.8,+4.2] control boot-to-boot -10.35%, step +2.51%) flat
== MTP acceptance per draft position (fresh c8, T=0), pooled per side
  ctl: 0.848 0.700 0.574 0.467
  arm: 0.849 0.701 0.576 0.469
== path: arm pairs VLLM_FI_SM12X_W4A4_MOE=1 on both ranks, images; KV pool per boot
  ctl-p1: pairs 0/1 on both ranks; image ok; KV 3521011 tokens, 29.26 GiB available for KV
    FlashInfer MoE layers per rank none; load check rel L2 none; boot 171 s
  ctl-p2: pairs 0/1 on both ranks; image ok; KV 3520457 tokens, 29.26 GiB available for KV
    FlashInfer MoE layers per rank none; load check rel L2 none; boot 171 s
  ctl-p3: pairs 0/1 on both ranks; image ok; KV 3506407 tokens, 29.14 GiB available for KV
    FlashInfer MoE layers per rank none; load check rel L2 none; boot 181 s
  ctl-p4: pairs 0/1 on both ranks; image ok; KV 3530624 tokens, 29.34 GiB available for KV
    FlashInfer MoE layers per rank none; load check rel L2 none; boot 172 s
  arm-p1: pairs 1/1 on both ranks; image ok; KV 3463332 tokens, 28.78 GiB available for KV
    FlashInfer MoE layers per rank {0: 48, 1: 48}; load check rel L2 {0: 0.1493, 1: 0.1536}; boot 182 s
  arm-p2: pairs 1/1 on both ranks; image ok; KV 3455937 tokens, 28.72 GiB available for KV
    FlashInfer MoE layers per rank {0: 48, 1: 48}; load check rel L2 {0: 0.1493, 1: 0.1536}; boot 181 s
  arm-p3: pairs 1/1 on both ranks; image ok; KV 3416191 tokens, 28.39 GiB available for KV
    FlashInfer MoE layers per rank {0: 48, 1: 48}; load check rel L2 {0: 0.1493, 1: 0.1536}; boot 182 s
  arm-p4: pairs 1/1 on both ranks; image ok; KV 3447433 tokens, 28.65 GiB available for KV
    FlashInfer MoE layers per rank {0: 48, 1: 48}; load check rel L2 {0: 0.1493, 1: 0.1536}; boot 181 s
  KV pool arm vs ctl: -2.1% tokens (ctl 3,519,625, arm 3,445,723), -0.62 GiB per rank
better: none; in the main cells: none
RULE2=FLAT
```

Results: {RESULTS_URL}
