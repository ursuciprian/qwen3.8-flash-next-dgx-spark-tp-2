# Further speed levers, ranked — 2026-09-18 (not implemented, list only)

Ranked by expected gain/effort. Sources: kernel-sweep-2026-09-18.md §3/§5
(already-researched), plus this prep task's own reading of the fork/b12x.

1. **`use_local_argmax_reduction:true` alone, no vocab table (zero patch).**
   Effort: none (flag only, already in `eugr-agents-serve-local-dv.yaml`'s
   speculative-config even before the patch/table exist). Gain: cross-rank
   comm only (O(vocab)->O(2*tp_size)), TP2/RoCE-relevant; smaller than the
   full lever-1 patch but free — try before the patch lands.
2. **`--mamba-ssm-cache-dtype bfloat16`** — this prep's lever 3, effectively
   free (one flag, no rebuild). See `gdn-state-bf16-lever3-2026-09-18.md`.
3. **Port upstream vLLM #53945** (Mamba state cache at EAGLE-resume
   block-grid position) — fixes align-mode cold-prefix-cache-reuse defect.
   Effort: small (3 commits, scheduler + mamba-manager only, no kernel/CUDA).
   Gain: unmeasured but directly explains the known "no first-pass reuse"
   symptom; likely the highest gain/effort ratio for anyone hitting repeated
   cold turns, but doesn't apply to already-warm sessions.
4. **`disable_eagle_block_drop: true`** in `--speculative-config` — zero-
   patch flag. Gain (third-party, `jschmied/qwen38-flash-next-gb10`,
   2026-09-18): cached tokens/warm-turn 4,800->6,400, warm-turn latency
   2.05s->1.52s, MTP acceptance 53-56%->57-60%. Complementary to #3, not a
   substitute (doesn't fix the cold first pass).
5. **`B12X_POLICY_MODE=heuristic-only`** A/B vs `auto` — zero patch, one
   env var. Gain: unmeasured; tests whether mispriced MoE warmup profiles
   contribute to the open c5-c7 straggler and to general decode cost;
   cheapest untried MoE lever.
6. **`VLLM_B12X_NVFP4_ACTIVATION_MODE=quantized`** for dense (non-MoE)
   layers — zero patch, one env var. Gain: unmeasured, could speed small-
   batch (c1-c4) decode; accuracy risk unquantified, needs the same fidelity
   gate as the numbered levers before trusting.
7. **`VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13`** — zero patch, coarser
   per-expert->layer-max NVFP4 MoE activation scale for gate/up proj. Gain:
   unmeasured, low-risk one-boot A/B.
8. **fp8 lm_head for the 5 verify tokens per step** (not just the MTP draft
   head) — would shrink the 1.18 GiB/rank full-vocab BF16 lm_head read on
   the *target* model's verify pass too, on top of lever 1's draft-only
   fix. Effort: medium-large (needs an NVFP4/fp8 target lm_head + accuracy
   validation on the actual output path, higher risk than the draft head
   since target lm_head errors are NOT re-verified by anything). Not traced
   into the fork's target-model file within this task's budget.
9. **Fusing/batching the lm_head projection across the 5 per-step verify
   tokens** (single wider GEMM instead of 5 narrow ones) — effort: medium,
   needs a read of the verify-path call site in
   `vllm/v1/worker/gpu/spec_decode/speculator.py` / rejection sampler to
   see whether it's already batched (plausible it already is, since verify
   naturally processes all draft positions together) — check before
   assuming this is unclaimed work.
10. **Sequence-parallel norms to cut per-layer all-reduces** — effort:
    large (touches every decoder layer's residual/norm wiring under TP2);
    gain plausible (removes 1 of ~2 all-reduces/layer over RoCE) but not
    confirmed the fork's norm implementation supports it; lowest
    effort/gain ratio of this list without a source read first.
11. **CUDA-graph FULL vs PIECEWISE for decode** — effort: small (one
    `--compilation-config` flag A/B), but our recipe's
    `cudagraph_capture_sizes` A/B already ran during the straggler
    investigation (`project-qwen-straggler-batch5-7.md`) with no effect
    on the bug; a fresh FULL-vs-PIECEWISE decode-latency A/B (as distinct
    from the straggler fix) hasn't been run and is cheap, but expected
    gain is speculative — no third-party number for this model.
12. **Pull MiaAI-Lab's `cached_tokens` reporting fix** (single-Spark repo,
    2026-09-17, commit `850a39338`) — not a speed lever, a measurement
    fix: our own harness currently can't trust
    `usage.prompt_tokens_details.cached_tokens` (reads 0 even on provable
    hits, kernel-sweep §5). Low effort, no serving-path risk (harness-only
    change), worth doing regardless so future A/Bs on #3/#4 above are
    measured correctly instead of via the Prometheus-counter workaround.
