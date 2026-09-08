# Measurements

All runs: two DGX Spark (GB10), tensor parallel 2 over ConnectX-7, temperature 0.
Grids are llama-benchy `--pp 2048 --tg 128 --enable-prefix-caching`. `pp2048` is a
fresh 2k-token prompt after a cached context of the given depth; `tg128` is
decode, aggregate tok/s across the concurrent streams. The default corpus is
prose, so decode figures are prose figures; predictable text (code, JSON,
counting) decodes 1.5-2x faster on the same server because the drafter accepts
more tokens.

Two fresh boots of the same SGLang recipe differ by 10-17% median on decode;
two boots of the vLLM recipe by under 2% on prefill. Differences smaller than
that between rows are noise.

## Recipes published on Spark Arena

### `qwen38-flash-next-nvfp4-fastqsa4096bigkv-g8-sglang` (SGLang)

Measured 2026-08-30, depths 0-65k, `sglang-fastqsa-bigkv-g8-grid-0-65k.csv`.

| depth | pp2048 c1 / c2 / c5 | tg128 c1 / c2 / c5 |
|---|---|---|
| 0 | 2521 / 2647 / 3023 | 40 / 62 / 95 |
| 4096 | 989 / 1447 / 2029 | 38 / 53 / 89 |
| 16384 | 946 / 1382 / 1939 | 37 / 60 / 81 |
| 32768 | 942 / 1332 / 1850 | 30 / 55 / 81 |
| 65536 | 853 / 1213 / 1700 | 31 / 57 / 65 |

The only configuration in that grid that kept five streams alive at 65k. KV pool
900k tokens, 110k context per request.

### `qwen3.8-flash-next-nvfp4-tp2` (vLLM)

Same profile measured with 4 draft tokens, depths 0-65k,
`vllm-tp2-mtp4-grid-0-65k.csv`; the published file ships with 3, which trades
a little predictable-text speed for a little prose speed (single stream, 512
tokens: counting 56.8 vs 66.7, code 46.7 vs 45.0, prose 39.9 vs 37.8).

| depth | pp2048 c1 / c2 / c5 | tg128 c1 / c2 / c5 |
|---|---|---|
| 0 | 1613 / 2115 / 2683 | 34 / 49 / 74 |
| 4096 | 875 / 899 / 1062 | 31 / 43 / 71 |
| 16384 | 634 / 656 / 664 | 34 / 53 / 43 |
| 32768 | 596 / 605 / 628 | 33 / 40 / 39 |
| 65535 | 526 / - / - | 31 / 38 / 36 |

Single-stream decode nearly flat to 64k. Prefill at depth halves with every
depth doubling because, on this build, prefix caching was not reusing anything
across turns; the current vLLM recipe fixes that (below). 262k context, 1.1M
KV tokens.

## Current recipes

### `flashnext-bigkv-g8-c4096` (SGLang, 2026-09-03 image)

`sglang-bigkv-g8-c4096-sep03-grid-0-32k.csv`, one of four boots.

| depth | pp2048 c1 / c2 / c5 | tg128 c1 / c2 / c5 |
|---|---|---|
| 0 | 2378 / 2580 / 2913 | 38 / 63 / 78 |
| 4096 | 905 / 1431 / 1946 | 31 / 56 / 79 |
| 16384 | 845 / 1392 / 1871 | 38 / 47 / 70 |
| 32768 | 892 / 1306 / 1737 | 36 / 59 / 64 |

Community 40-prompt category harness, single stream: 62.5 tok/s median
(coding 65.4, reasoning 63.0, JSON 67.6, prose 38.7). Versus the published
recipe: fresh shallow prefill about 10% lower at c1, decode unchanged, in
exchange for the image whose sparse-attention kernel is correct on SM121 at
long context.

### `flashnext-bigkv-nospec` (SGLang, no drafter)

`sglang-bigkv-nospec-grid-0-32k.csv`.

| depth | pp2048 c1 / c2 / c5 | tg128 c1 / c2 / c5 |
|---|---|---|
| 0 | 2496 / 2952 / 3115 | 26 / 53 / 85 |
| 4096 | 1046 / 1526 / 2135 | 26 / 53 / 75 |
| 16384 | 1001 / 1479 / 2075 | 26 / 45 / 85 |
| 32768 | 961 / 1284 / 1979 | 26 / 47 / 77 |

Prefill +20% everywhere and c5 decode +6 to +43% at depth; single-stream
decode drops from ~36 to 26. Use at five or more concurrent streams.

### `flashnext-vllm-cached` (vLLM, drafter on, prefix caching that reuses)

`vllm-cached-grid-0-32k.csv`.

| depth | pp2048 c1 / c2 / c5 | tg128 c1 / c2 / c5 |
|---|---|---|
| 0 | 2736 / 2644 / 2647 | 43 / 47 / 67 |
| 4096 | 1889 / 1868 / 1897 | 29 / 46 / 53 |
| 16384 | 2176 / 2154 / 2180 | 31 / 45 / 61 |
| 32768 | 1862 / 1840 / 1896 | 33 / 44 / 52 |

Cached prefill at depth is 2x the SGLang recipe and 3-12x the published vLLM
recipe; c5 decode at 32k is 52 where the published recipe fell to 39 and, on
the newer build without the fix, to 8.6. Category harness single stream:
55.8 coding, 62.2 JSON, 37.5 prose, 50.6 median. Cold prefill 2770-3281 tok/s
from 7k to 88k tokens. Reuse verified on every boot with a three-request probe
(same 20k prompt: 0 / 19,200 / 19,200 tokens hit). 262k context, 2.0M KV
tokens. Same recipe without the drafter (`vllm-cached-nospec-grid-0-32k.csv`):
identical cached prefill, single-stream decode 26, KV pool 2.7M.

## Reading the two engines

| workload | pick |
|---|---|
| chat, one or two agents, cached history | SGLang `flashnext-bigkv-g8-c4096` |
| five or more streams, batch prefill | SGLang `flashnext-bigkv-nospec` |
| long documents revisited across turns, many agents, capacity | vLLM `flashnext-vllm-cached` |

SGLang decodes faster on every one-shot workload (coding 65 vs 56 tok/s at
c1). vLLM wins wherever a large context is reused: every cached turn prefills
2x faster and concurrent decode at depth does not collapse.
