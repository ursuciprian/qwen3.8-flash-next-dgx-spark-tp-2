Correction to the earlier k84 failure on this issue: it came from my image check, which looked up only the alias `Qwen3_8FlashNextForConditionalGeneration`. Both stock images register `Qwen4ExpForConditionalGeneration`, the first architecture in the checkpoint's config, and vLLM uses that one. This rerun checks every listed architecture.

Stock baseline (k84-stock-baseline): upstream vLLM on the unmodified checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `7c4f1bc1`, on the same two Sparks, with the harnesses of the capability matrix (k76).

Images (digest and vLLM version per Spark):
```
dgx01 vllm/vllm-openai:v0.31.0 vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3 sha256:bbe7045055707d1027079ed01cca40812578a235c56032d85c16108ad473384c
dgx02 vllm/vllm-openai:v0.31.0 vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3 sha256:bbe7045055707d1027079ed01cca40812578a235c56032d85c16108ad473384c
vllm/vllm-openai:v0.31.0: ARCHCHECK 0.31.0 Qwen4ExpForConditionalGeneration 2
dgx01 vllm/vllm-openai:nightly-7d0b4e57aac4c4323225b0eee9ae960ce75ffffe vllm/vllm-openai@sha256:2b3b0fe155011cbca03828e35fe1a483c1b5c9cf44c807cdee6ab51db734d553 sha256:7d928d3c9b8a4027cf58d3c50c861ed381cf322ab34dc06a2363fc88e08d3ead
dgx02 vllm/vllm-openai:nightly-7d0b4e57aac4c4323225b0eee9ae960ce75ffffe vllm/vllm-openai@sha256:2b3b0fe155011cbca03828e35fe1a483c1b5c9cf44c807cdee6ab51db734d553 sha256:7d928d3c9b8a4027cf58d3c50c861ed381cf322ab34dc06a2363fc88e08d3ead
vllm/vllm-openai:nightly-7d0b4e57aac4c4323225b0eee9ae960ce75ffffe: ARCHCHECK 0.31.1rc1.dev253+g7d0b4e57a Qwen4ExpForConditionalGeneration 2
```

Boot attempts:
```
2026-10-11 00:52:58 stock-2x-mtp3-v0.31.0: did not boot; 
2026-10-11 00:57:31 stock-2x-mtp3-nightly-7d0b4e57: did not boot; 
```

Recipe: upstream recipe flags (prefix caching, fp8 KV and indexer KV, no flashinfer autotune, qwen3 reasoning parser) with the memory settings of my 2x recipe; MTP on means 3 draft tokens, the upstream setting. The only addition is the server default reasoning effort "medium", which my recipes also set. The generated recipes are in each setup folder (`recipe.yaml`).

Setup: TP=2 on both Sparks with MTP on (3 draft tokens). I dropped MTP off and the single-Spark attempt to keep the job short.

Cells, the same as k86 so the numbers compare cell for cell: llama-benchy pp2048/tg512 task mode, 3 runs, c1 and c8, at T=1.0 top-p 0.95 top-k 20 with thinking on (`tgdef`) and at T=0 with thinking off (`tgt0`); the 36-prompt coding probe at T=0 with thinking off, c1 and c8, one pass (aggregate output tok/s of the pass). GPU power is nvidia-smi power.draw at 1 Hz (GPU only, not wall power); 2x adds both Sparks. The last table lists the shipped builds from k86 (3 boots each) next to stock.

```
no measured cells
```

Results: {RESULTS_URL}
