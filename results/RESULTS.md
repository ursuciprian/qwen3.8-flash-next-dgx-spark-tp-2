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

## Reference harnesses, 2026-09-11

Fresh boot per arm, page cache dropped on both nodes. Three lanes per boot:
llama-benchy 0.4.0 in MiaAI-Lab's standard spec (pp 128/4096, tg 256/1024,
depth 0/32768, concurrency 1/4), tonyd2wild's 40-prompt category harness
(concurrency 1 to 6, cold prefill ladder to 88k with a needle), and
`tool-eval-bench --short` (15 core scenarios, thinking off).

| streams | SGLang shipped, bf16 KV | vLLM cached + EP | SGLang fp8 KV, 900k | SGLang fp8 KV, 1.8M |
|---|---|---|---|---|
| 1 | 67.0 | 65.2 | 64.7 | 64.2 |
| 2 | 121.9 | 118.7 | 126.6 | - |
| 4 | 229.2 | 206.6 | 218.1 | - |
| 6 | **314.3** | 296.0 | 308.6 | - |
| TTFT at 6 streams | **0.40 s** | 3.50 s | 0.41 s | - |
| cold prefill at 88k | 4077 | 3436 | 4197 | 4097 |
| tool-eval, 15 core | 97 | **100** | 97 | 97 |

Four bf16 boots of the shipped recipe gave 67.0, 66.5, 66.3 and 63.7 tok/s at
one stream on this harness, so the envelope is about 5%. Both fp8 arms sit
inside it.

Against the published references on the same hardware class, Tony's own
harness against his TP2 profile: 53.7 vs **67.0** at one stream, 97.9 vs
**314.3** at six, 2093 vs **3119** tok/s cold prefill at 28k. MiaAI-Lab's
standard spec against their published T1: 24.1 vs **36.7** tg256 at one stream
on vLLM + EP (their cluster caps the GPU clock at 2200 MHz, which accounts for
part of the gap).

### fp8 KV cache (`flashnext-fp8kv-1m8`)

`--kv-cache-dtype fp8_e4m3` cannot serve this model on GB10 unpatched: the SM121
QSA decode kernel is BF16-only and the scheduler dies on the first decode with
`unsupported SM121 QSA call: expected BF16 D=256 ...`. The mod
`sglang-sm121-qsa-fp8kv` allocates the gather scratch in the query dtype so the
selected keys convert on store. Result: KV pool 899,968 to 1,800,000 tokens,
context 110k to 262k, decode within control drift. Quality is checked only by
the 15-scenario tool-eval (97, same as bf16, same single partial) and the
needle to 88k. bf16 stays the default until a 76-scenario run and a full-window
needle ladder are measured on both.

### Expert parallelism in `flashnext-vllm-cached`

`--enable-expert-parallel --all2all-backend allgather_reducescatter` measured
+3-6% prefill and +4-19% decode at depth on 2026-09-09 and no regression on any
lane on 2026-09-11, with the best tool-eval score of the day. Promoted into the
recipe.

### Corruption gate (sglang#37111)

Upstream reports QSA + NEXTN + decode CUDA graphs silently corrupting output on
GB10 TP2. `scripts/gate_37111.py` runs both reported cases against a live
server: a 1024-token essay and a 25k prompt with four ordered markers. On the
shipped recipe: coherent prose with `finish_reason: length`, 4/4 markers in
order. The report pins a day-zero image; the shipped digest carries upstream's
own SM121 kernel.

## Choosing between the engines

| workload | recipe |
|---|---|
| chat, one or two agents, cached history | SGLang `flashnext-bigkv-g8-c4096` |
| the same, needing more than 110k context or a bigger pool, quality gate accepted | SGLang `flashnext-fp8kv-1m8` |
| five or more streams, batch prefill | SGLang `flashnext-bigkv-nospec` |
| long documents reused across turns, many agents, capacity | vLLM `flashnext-vllm-cached` |

SGLang decodes faster on one-shot work: 63 tok/s on coding prompts against 58.
vLLM wins wherever a large context is reused, because every cached turn
prefills about twice as fast and concurrent decode at depth does not collapse.
