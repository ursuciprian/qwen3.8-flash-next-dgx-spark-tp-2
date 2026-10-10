Quality of the GDN-MSE requant (k85-quality-kld-evals), one engine for both arms: the shipped 1x recipe v2.2.0 (vLLM 5dad364d, b12x 21e0b201) on `ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE` @ `03f4a057`, where `VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS` selects which copy of the GDN projection weights runs: `1` = the MXFP8 tensors of `7c4f1bc1` (shard 35, byte-identical to the unmodified checkpoint) on every call, `0` = the NVFP4 GDN-MSE tensors on every call, `41` = the shipped dispatch. Every other tensor is byte-identical between the two checkpoints, so the arms differ only in the GDN weights.

Reference: `7c4f1bc1`, not BF16. The BF16 base (336 GB) is on disk but does not fit in the 2 x 128 GB of the pair, so it cannot be served as a reference.

IFEval, all 541 prompts, at the served sampling (T=1.0, top-p 0.95, top-k 20, thinking on, server default effort medium), lm-eval 0.4.13, c8, one Spark per arm at the same time: `7c4f1bc1` = v2.2.0 with `MIN_TOKENS=1`, `GDN-MSE` = v2.2.0 unchanged. Paired per prompt; McNemar p is exact and two-sided.

```
eval:metric                                        n      7c4f1bc1       GDN-MSE   discordant (7c4f1bc1 only / GDN-MSE only)   McNemar p
ifeval_think:prompt_level_loose_acc              541        91.13%        91.13%       20 / 20                      1.000
ifeval_think:prompt_level_strict_acc             541        87.62%        88.91%       17 / 24                      0.349
ifeval_think:inst_level_strict_acc            834/834 instructions        91.61%        92.57%                                    -
```

KLD and top-1 agreement, one round: 44 held-out sequences of the model's own outputs (agentic 14, chat 5, code 10, math 5, tools 10; 32,351 scored tokens), teacher-forced at the output positions, top-64 prompt logprobs, MTP off. KLD is over the reference's top 64 plus a rest bucket. Prompt scoring is prefill, where the shipped dispatch runs the MXFP8 copy, so the `0` arm shows the NVFP4 GDN error at every position; in the shipped recipe it only reaches decode steps (below 41 rows). Noise floor: the reference server scored twice.

```
Reference first. mxfp8 = 7c4f1bc1's GDN tensors (MIN 1), nvfp4 = GDN-MSE's GDN tensors (MIN 0); same image, same checkpoint folder, MTP off, top-64 prompt logprobs on 44 sequences.
GDN-MSE vs 7c4f1bc1: not measured (missing mxfp8-dgx01-a.jsonl.gz or nvfp4-dgx02-a.jsonl.gz)
floor: 7c4f1bc1, same server twice: not measured (missing mxfp8-dgx01-a.jsonl.gz or mxfp8-dgx01-b.jsonl.gz)
```

I trimmed this round to IFEval and one KLD round. I did not run MMLU-Pro: my 2000-question run on 2026-09-28 was stopped at 770/2000 after 3 h 10 min (about 4000 thinking tokens per question at effort xhigh) and left no score, because lm-eval writes samples only at the end and that run had no `--use_cache`.

Results: {RESULTS_URL}
