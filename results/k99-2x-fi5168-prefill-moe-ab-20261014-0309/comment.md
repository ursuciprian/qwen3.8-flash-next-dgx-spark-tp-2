k99-2x-fi5168-prefill-moe-ab result: I routed the 2x's large prefill chunks to FlashInfer's SM12x fused MoE at `5c4058bd9b` (merge of FlashInfer PR 5168) and kept b12x for decode and for chunks below 4096 rows, on the shipped 2x v2.1.0 image with the same weights and plan selections. The 2x prefills a long prompt in 5,728-row chunks (2 x its 2,864-token attention block) at budget 8192, so every full chunk takes the FlashInfer path; decode steps (at most 80 rows) stay on b12x. On the arm boots FlashInfer served calls of 27 distinct row counts from 4185 to 8192 (rank 0). FlashInfer reads b12x's prepared expert storage in place; the image's own flashinfer stays in use for sampling. The operator probe behind this (k98, I320 rows) is on ursuciprian/qwen3.8-flash-next-1x-dgx-spark#27.

Op check before any boot (one Spark, routed-MoE layers 5 and 47 at the I320 rank shard, real routing from a cold 16K prompt): `OPCHECK PASS`.
```
  layer      M   err FI  err b12x  diff   ms FI  ms b12x   x
      5   2864   0.1138    0.1138 0.007   5.316    5.380 x1.01
      5   5728   0.1137    0.1138 0.009   6.998    8.271 x1.18
      5   6048   0.1137    0.1138 0.009   7.122    8.574 x1.20
      5   9072   0.1137    0.1138 0.009   9.754   11.884 x1.22
     47   2864   0.1193    0.1193 0.005     nan      nan xnan
     47   5728   0.1193    0.1193 0.008     nan      nan xnan
     47   6048   0.1193    0.1193 0.008     nan      nan xnan
     47   9072   0.1193    0.1193 0.008     nan      nan xnan
  projection at M 5728 (operator only): +1.273 ms per layer x 49 layers = +62.4 ms per chunk; per cold prompt 16K 2 chunks +0.12 s, 64K 11 chunks +0.69 s, 128K 22 chunks +1.37 s
  memory kept by FlashInfer: layer 5 260 MiB (with the shared workspace), layer 47 90 MiB -> per rank ~4.47 GiB (49 layers + workspace 170 MiB at M 5728; grows with the largest M seen)
    layer 5 FlashInfer caches (MiB): _WEIGHT_CACHE 75, _PADDED_SCALE_CACHE 90, _PADDED_FP4_CACHE 0, _WORKSPACE_CACHE 170, _W4A16_WEIGHT_CACHE 0
    layer 47 FlashInfer caches (MiB): _WEIGHT_CACHE 75, _PADDED_SCALE_CACHE 90, _PADDED_FP4_CACHE 0, _WORKSPACE_CACHE 0, _W4A16_WEIGHT_CACHE 0
  OPCHECK PASS
```

Screen on the pair, boots in the order ctl:1 arm:1 arm:2 ctl:2 (2 per side), cold prompts identical on every boot: `RULE=FLAT`. Gate on one arm boot: PASS. hardmode 93; TC-45 100.0; long-context retrieval 8000:20/20 32000:20/20 64000:20/20 128000:20/20 128000:20/20 128000:20/20; straggler preemptions 0 0 0 0 0 0.
```
== cold TTFT (s, mean of 3 per boot); noise = larger boot-to-boot spread of either side (floor 0.5%)
  16K: ctl 5.64 5.64 | arm 5.59 5.64 -> -0.5% (-0.03 s; noise 1.0%) flat
  64K: ctl 24.65 24.62 | arm 24.50 24.58 -> -0.4% (-0.10 s; noise 0.5%) flat
  128K: ctl 55.38 55.18 | arm 54.99 55.20 -> -0.3% (-0.18 s; noise 0.5%) flat
== llama-benchy pp2048 + tg512 (tok/s, side means; band = max(k92 band, side spread))
  pp2048 (c1): ctl 2798.5 arm 2765.4 -1.2% (band 4.7%)
  tg512 (c1): ctl 83.7 arm 87.2 +4.2% (band 11.9%)
  tg512 (c4): ctl 150.2 arm 165.3 +10.0% (band 8.2%)
  tg512 (c8): ctl 198.8 arm 193.0 -2.9% (band 6.5%)
== MTP acceptance per draft position (fresh c8, T=0), pooled per side
  ctl: 0.843 0.694 0.567 0.467
  arm: 0.851 0.705 0.577 0.470
== path: K99FI calls (rank 0 / rank 1) and KV pool per boot
  ctl-p1: FlashInfer calls 0 / 0; KV 3519347 tokens, 29.25 GiB available for KV
  ctl-p2: FlashInfer calls 0 / 0; KV 3520087 tokens, 29.25 GiB available for KV
  arm-p1: FlashInfer calls 1000 / 1000; KV 2987480 tokens, 24.83 GiB available for KV
  arm-p2: FlashInfer calls 1000 / 1000; KV 2994690 tokens, 24.89 GiB available for KV
  KV pool arm vs ctl: -15.0% tokens, -4.39 GiB (the FlashInfer path's memory cost per rank)
```

The FlashInfer prefill path does not move cold TTFT beyond boot noise on the 2x in this run, so the 2x stays on b12x for prefill.

Results: {RESULTS_URL}
