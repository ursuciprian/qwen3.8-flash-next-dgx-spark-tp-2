# Measurements

Two DGX Spark (GB10), tensor parallel 2 over ConnectX-7, temperature 0.

Grids are `llama-benchy --pp 2048 --tg 128 --enable-prefix-caching` at depths 0
to 32k and concurrency 1, 2, 5. `pp2048@depth` is a fresh 2k-token prompt after
a cached context of that depth, so it measures a cached turn, not cold prefill.
`tg128` is decode, aggregate across the concurrent streams. The default corpus
is prose; predictable text decodes far faster on the same server.

Two fresh boots of the same SGLang recipe differ by 20% on prefill and 13% on
decode. Differences smaller than that are not results.

## `flashnext-bigkv-g8-c4096` (SGLang)

`sglang-bigkv-g8-c4096-sep03-grid-0-32k.csv`, c1 / c2 / c5.

| depth | pp2048 | tg128 |
|---|---|---|
| 0 | 2353 / 2633 / 3004 | 41 / 62 / 94 |
| 4096 | 915 / 1388 / 1963 | 36 / 62 / 83 |
| 16384 | 915 / 1331 / 1902 | 38 / 54 / 84 |
| 32768 | 918 / 1345 / 1874 | 37 / 58 / 79 |

40-prompt category mix, single stream: 58-61 tok/s median, coding 63, JSON 66,
reasoning 62, prose 39. Speculative acceptance 2.2-2.4 of a maximum 4.0 on
prose, 4.0 on counting.

## `flashnext-bigkv-nospec` (SGLang, no drafter)

`sglang-bigkv-nospec-grid-0-32k.csv`, c1 / c2 / c5.

| depth | pp2048 | tg128 |
|---|---|---|
| 0 | 2496 / 2952 / 3115 | 26 / 53 / 85 |
| 4096 | 1046 / 1526 / 2135 | 26 / 53 / 75 |
| 16384 | 1001 / 1479 / 2075 | 26 / 45 / 85 |
| 32768 | 961 / 1284 / 1979 | 26 / 47 / 77 |

Prefill about 20% higher everywhere and decode at five streams 6-43% higher at
depth, in exchange for single-stream decode falling from 37-41 to 26. The
drafter accepts roughly 2 of 4 tokens on prose; past five streams the
verification costs more than it returns.

## `flashnext-vllm-cached` (vLLM)

`vllm-cached-grid-0-32k.csv`, c1 / c2 / c5.

| depth | pp2048 | tg128 |
|---|---|---|
| 0 | 2890 / 2773 / 2773 | 40 / 54 / 71 |
| 4096 | 2012 / 1962 / 1998 | 30 / 44 / 56 |
| 16384 | 2282 / 2229 / 2290 | 34 / 47 / 63 |
| 32768 | 1924 / 1935 / 1994 | 39 / 46 / 62 |

Cached turns prefill at roughly twice the SGLang recipe, and decode at depth
holds up at five streams. Category mix single stream: 52 tok/s median, coding
58. Cold prefill 2770-3280 tok/s from 7k to 88k tokens with a needle check
correct at every size.

Reuse is verified on every boot with a three-request probe: the same 20k prompt
sent fresh, identical, then with a changed last line reuses 0, then 19,200,
then 19,200 tokens, the second request falling from 10 s to 1.7 s. Without the
`disable_eagle_block_drop` flag the same probe reads 0, 0, 16,000 and the depth
cells collapse to 161 tok/s at 32k. Same recipe without the drafter
(`vllm-cached-nospec-grid-0-32k.csv`): identical cached prefill, single-stream
decode 26, KV pool 2.7M tokens.

## Choosing between the engines

| workload | recipe |
|---|---|
| chat, one or two agents, cached history | SGLang `flashnext-bigkv-g8-c4096` |
| five or more streams, batch prefill | SGLang `flashnext-bigkv-nospec` |
| long documents reused across turns, many agents, capacity | vLLM `flashnext-vllm-cached` |

SGLang decodes faster on one-shot work: 63 tok/s on coding prompts against 58.
vLLM wins wherever a large context is reused, because every cached turn
prefills about twice as fast and concurrent decode at depth does not collapse.
