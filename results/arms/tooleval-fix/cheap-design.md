# TC-45 without the decode tax -- mechanism, fix, validation

Author pass 2026-09-20, read-only on dgx-01 (no GPU work).

> **v2 correction, 2026-09-21.** The la-tcc validation boot failed TC-45, and
> chasing that turned up two things that change this document. Read this
> banner before the sections below; where they disagree, this banner wins.
>
> 1. **TC-45 is not deterministic under this grammar.** Same `ghcr` image, two
>    boots: `results/arms/ghcr/hardmode.log` TC-45 **FAIL** (47.2 s turn,
>    score 86) and `results/arms/ghcr2/hardmode.log` TC-45 **PASS** (48.0 s,
>    89). `la-tc` PASS (48.8 s, 93). The la-tcc run FAILED at 42.5 s. With no
>    grammar at all (`results/arms/la/hardmode*.log`) TC-45 fails in
>    **0.8-1.3 s**. So the turn time, not the pass/fail, tells you whether the
>    grammar was built: 42-49 s means it was built and the model still talked
>    its way out of the call; sub-second means no grammar. la-tcc at 42.5 s
>    with `finish_reason=length` is the *grammar-applied* failure mode, same as
>    the ghcr boot -- **not** a regression introduced by the cheap variant. The
>    A/B that flagged it was n=1 against n=1 on a ~50/50 scenario.
>    (Confirmed independently: the la-tcc boot log line 483 shows
>    `Triton kernel JIT compilation during inference:
>    _apply_grammar_bitmask_kernel` -- a grammar was compiled and applied.)
>
> 2. **The "auto" cost claim below is overstated.**
>    `get_model_structural_tag()` returns `None` for `tool_choice="auto"`
>    unless a tool sets `strict: true`
>    (`vllm/tool_parsers/structural_tag_registry.py:111-113`). So plain
>    agentic `auto` traffic never became a structured-output request, and the
>    four per-step costs listed under **Mechanism** only ever armed for
>    `required`/named/strict-`auto` requests. What `auto` did cost is the
>    wasted construction of a tool parser (a `ParserEngineToolAdapter`, so a
>    second parser engine with its tokenizer vocab) on every request, just to
>    be handed `None`. That is a frontend per-request cost, not a decode cost.
>    Consequently **the ~10% c1 sweep regression is very unlikely to be
>    explained by the patch at all**, and hypothesis **H2** (gate ordering /
>    workload confound) below is now the strong favourite. The distinguishing
>    experiment is still worth its three minutes.
>
> 3. **v1 dropped `auto` outright; v2 puts it back for strict tools.** That was
>    a real behaviour regression in v1 -- a client asking for `strict: true`
>    schema enforcement lost it. `0001b`/`0002b` gate on the same rule the
>    registry uses. Behaviour is now identical to upstream, and the wasted
>    tool-parser construction is still gone.
>
> **Current artefacts:** apply `0001` **then** `0001b` (jovian), or `0002`
> **then** `0002b` (KK). `mods/vllm-tc45-cheap/` already produces the
> `0001+0001b` end state in one shot -- verified byte-identical, post-image
> sha256 `fdc94d71...` for `abstract_parser.py`.
>
> **What the mod now proves at boot, without a GPU:** the self-check builds a
> real `ChatCompletionRequest` (tools + `tool_choice="required"`), runs it
> through `attach_structural_tag()`, and asserts `structured_outputs
> .structural_tag` is populated and names the tool; asserts plain `auto`
> builds nothing and `auto` + `strict` still does; asserts the collapsed
> parser is a per-call subclass whose `adjust_request` is the patched
> `ParserEngine.adjust_request` and that `Qwen3Parser` itself was not mutated.
> A broken gate now fails the boot instead of costing a 45-minute benchmark.
Follows the "Phase B result: TC-45 fixed, but the fix regresses c1 throughput
-- REVERTED" section of `notes.md`.

Artefacts:

- `patches/vllm-tc45-cheap/0001-tc45-cheap-jovian.patch` -- on top of
  `ursuciprian/vllm@5d1df69f` (which is on top of jovian `8e1f1e58`).
- `patches/vllm-tc45-cheap/0002-tc45-cheap-kk.patch` -- on top of
  `integration/karmic-kraken-beta@57a80980`.
- `mods/vllm-tc45-cheap/` -- applies the *net* jovian change (8e1f1e58 ->
  0001) to the installed vLLM in `spark-vllm-b12x:local-20260918-a8333658`.
  Verified: running `apply.py` against a clean 8e1f1e58 checkout reproduces
  the 0001 result byte-for-byte (all three sha256 match).
- `recipes/eugr/eugr-agents-serve-local16-la-tcc.yaml` -- `la` + this mod.

## What 5d1df69f actually turns on

`5d1df69f` fixed two real bugs (`ParserEngine.adjust_request()` was a no-op
stub at `vllm/parser/engine/parser_engine.py:204`, and the collapse branch at
`vllm/parser/parser_manager.py:143` returned the engine class still carrying
the generic adapters `make_adapters()` pinned on it at import time, which have
no `structural_tag_model`). TC-45 passes because of it, and hardmode went
91 -> 93.

But it inherited `DelegatingParser._apply_structural_tag()`'s gate, and that
gate fires on **`tool_choice == "auto"`** as well as required/named:

    vllm/parser/abstract_parser.py:536-543
        need_tool_calling = (
            request.tool_choice == "auto"        <-- the expensive one
            or request.tool_choice == "required"
            or isinstance(request.tool_choice, (Named..., ToolChoiceFunction))
        )

`auto` is what every agentic client sends. So after the fix, *every request
that carries tools* becomes a structured-output request
(`vllm/v1/request.py:297`), and that is a property of the request for its
whole lifetime, not a one-off cost at admission.

## Mechanism: what a live grammar costs per engine step

Per-request, once:

- `vllm/renderers/online_renderer.py:481` constructs the parser per request
  anyway; with the tag gate open it now also constructs
  `Qwen3EngineToolParser(tokenizer, tools)`, which is a
  `ParserEngineToolAdapter` and so builds a **second** `Qwen3Parser` engine
  (`vllm/parser/engine/adapters.py:168`), tokenizer vocab included.
- `vllm/v1/engine/core.py:1119` `grammar_init()` -> xgrammar compile; the
  request parks in `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`
  (`vllm/v1/request.py:117`).

Per engine step, for as long as such a request is in flight -- and this is the
part that shows up as decode rate:

1. **Sampling stops being pipelined.**
   `vllm/v1/core/sched/async_scheduler.py:31` sets
   `scheduler_output.pending_structured_output_tokens` whenever a scheduled
   request `use_structured_output` and has output placeholders. That flag
   makes `EngineCore.step_with_batch_queue()` take the deferred branch:

       vllm/v1/engine/core.py:757-772   -> defer instead of sample_tokens(non_block=True)
       vllm/v1/engine/core.py:826-847   -> sample only after the *previous*
                                           step's output has been processed

   This is the big one. Async scheduling's whole point is that step N+1 is
   dispatched while step N's output is still being consumed; the deferred
   branch serialises them. It is a property of the *step*, so it penalises
   every request in the batch, not just the one holding the grammar.

2. **A draft-token D2H copy that async scheduling otherwise skips entirely.**
   `vllm/v1/worker/gpu_model_runner.py:5070` returns early unless
   `scheduler_output.has_structured_output_requests`; with a grammar live it
   does the copy + event sync, and `core.py:828` calls
   `take_draft_token_ids()` before the deferred bitmask.

3. **Bitmask work.** `Scheduler.get_grammar_bitmask()`
   (`vllm/v1/core/sched/scheduler.py:2576`, armed by the `|=` at
   `scheduler.py:2267`) -> host-side `fill_bitmask` ->
   `StructuredOutputsWorker.apply_grammar_bitmask()`
   (`vllm/v1/worker/gpu/structured_outputs.py:65`): H2D copy of the mask, a
   python-level mapping build, and a triton kernel over the logits, per step.

4. **MTP drafts get grammar-rejected.** Visible in the evidence already
   captured: accepted tokens per draft is
   `76767/24344 = 3.15` for `la` and `116582/42086 = 2.77` for `la-tc`,
   `116048/41906 = 2.77` for `ghcr2` (`results/arms/*/mtp_metrics.txt`) --
   a 12% acceptance loss, the same order as the throughput loss.

All four are latency, not throughput: at c8/c16 the GPU step is long enough to
hide them, which is exactly the measured shape (c4/c8/c16 within noise, c1 down
~10%).

## Honest caveat about the c1 sweep number

The `-10%` was measured by `bench_sweep.py`, which sends **no tools**, so with
the code above it should early-return and cost nothing. Two things are worth
knowing before treating 85.0 as the patch's steady-state price:

- `scripts/gate_arm.sh` runs `tool-eval-bench --hardmode` **before** the
  sweep, so the sweep always runs in a process that has already served
  hundreds of grammar requests. The la-tc/ghcr2 hardmode runs were also ~73%
  heavier than la's (42086 vs 24344 drafts).
- In the *same* boots, `bench_categories --concurrency 1` and `decode_probe`
  -- both single-stream real-prompt probes (temp 0), unlike the bench_sweep
  counting diagnostic, both run after hardmode and before the sweep -- show
  no regression at all (html2 100.1 / 100.4 / 99.8 tok/s for la / la-tc /
  ghcr2; json/coding/prose likewise within noise).

So the two candidate mechanisms for the *sweep* delta specifically are:

- **H1 (sticky):** the first structural-tag request leaves the engine process
  in a slower steady state -- xgrammar backend + `StructuredOutputsWorker`
  GPU bitmask tensor allocated after CUDA-graph capture, threadpools, or a
  path that stays warm. Predicts the drop persists with zero grammars in
  flight.
- **H2 (confound):** the patch costs nothing for tool-free requests (as the
  code says) and the sweep number is ordering/variance -- a heavier preceding
  hardmode run in those two boots. Supported by categories-c1 and decode_probe
  being identical across arms.

Either way, the mechanism that costs *real agentic traffic* (the per-step list
above, live on every `auto` request) is proven from the code and from the MTP
counters, and the fix removes it. H1/H2 only decides whether the tool-free
sweep number was ever the patch's fault.

### 5-minute experiment that distinguishes them (GPU worker)

Boot the patched image once and reverse the gate order:

1. Boot `la-tcc`, wait healthy. **Do not run hardmode.**
2. `bench_sweep.py <base> qwen3.8-flash-next presweep --levels 1 --rounds 3`
3. One forced-tool request, to arm every structured-output path in the
   process:

       curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
         "model":"qwen3.8-flash-next","messages":[{"role":"user","content":"What is 7 times 8?"}],
         "tools":[{"type":"function","function":{"name":"calculator","description":"eval",
           "parameters":{"type":"object","properties":{"expression":{"type":"string"}},
           "required":["expression"]}}}],
         "tool_choice":"required","temperature":0,"max_tokens":128}' | jq '.choices[0].message.tool_calls'

4. `bench_sweep.py <base> qwen3.8-flash-next postsweep --levels 1 --rounds 3`

Read-out:

- step 2 ~= step 4 ~= 95-97  -> **H2**. The patch is free for tool-free
  traffic; the old 85.0 was a gate-ordering confound. Record it and move on.
- step 2 ~= 96, step 4 ~= 85 -> **H1**. Sticky. Then bisect with
  `/metrics` before/after step 3 and with `VLLM_LOGGING_LEVEL=DEBUG`, and
  look first at whether `StructuredOutputsWorker`'s GPU bitmask
  (`vllm/v1/worker/gpu/structured_outputs.py:57`) is being allocated after
  CUDA-graph capture.
- step 2 ~= 85 already -> the cost is at import/startup, not per-request;
  bisect the three files (parser_manager alone, then parser_engine).

Cost: one boot, ~3 minutes of benching on top of it. It does not need its own
boot if it is run as the *first* thing after the la-tcc boot, before the gate.

## What the fix changes

`needs_structural_tag()` (new, `vllm/parser/abstract_parser.py`) gates on
`tools` **and** `tool_choice in {"required", named}`. `auto` no longer builds
a grammar, so:

- no `structured_outputs` on the request -> `use_structured_output` stays
  False -> none of the four per-step costs above ever arm for ordinary
  agentic traffic;
- the grammar machinery stays lazily initialised and strictly per-request --
  nothing is turned on process-wide, and no capability flag is widened;
- the tool parser (and the second parser engine it builds) is constructed
  only after the gate passes, instead of on every tools request;
- `ParserManager.get_parser()` no longer mutates `tool_parser_cls` /
  `reasoning_parser_cls` on the shared engine class; it returns a per-call
  subclass, the pattern the `DelegatingParser` branch in the same function
  already uses. Idempotent, and invisible to other models backed by the same
  engine.

TC-45 sends `tool_choice: "required"`, so it still gets its grammar -- now
with `reasoning=True` so xgrammar admits the `<think>` prefix
`--reasoning-parser qwen3` always emits.

Quality expectation: `auto` requests go back to the la baseline's behaviour
(no grammar, parser extracts the call after the fact) -- la scored 91 that way
-- plus TC-45 now passing. So >= 91 is the expectation, 93 only if the `auto`
grammars were also buying quality. The gate below is >= 90 either way.

## Karmic-kraken differs

`57a80980` **deleted the collapse branch** from `ParserManager.get_parser()`:
Qwen3 (and Kimi K2, GLM-4.7-MoE, ...) now compose through `DelegatingParser`,
so `_apply_structural_tag()` is reachable and both jovian bugs are already
gone there. KK's own design is the better one -- do not port the jovian
`parser_manager`/`parser_engine` edits forward.

What KK still has wrong, and what `0002-tc45-cheap-kk.patch` fixes, is the
same two things in `abstract_parser.py`:

- `reasoning=False` is still hardcoded (`abstract_parser.py:587` at KK), so a
  forced tool call under thinking-on is constrained against a grammar that
  forbids the `<think>` the model opens with -- the original Phase A
  hypothesis, still live on KK;
- the gate still fires on `auto`, so KK will hit this exact decode tax the
  moment it serves agentic traffic. On KK it is worse: `Parser.__init__`
  (`abstract_parser.py:127`) instantiates `tool_parser_cls` for *every*
  request regardless, so KK already pays two parser-engine constructions per
  request before any grammar is involved. Not in scope here.

## Validation recipe for the GPU worker

Fail fast, in this order. Anything that misses, stop and report -- do not
promote.

1. **Boot** `recipes/eugr/eugr-agents-serve-local16-la-tcc.yaml` (= `la` plus
   `mods/vllm-tc45-cheap`). The mod prints
   `mod vllm-tc45-cheap: self-check ok -> Qwen3Parser Qwen3EngineToolParser`
   during startup; if it prints FATAL the boot must fail -- that is the mod
   working, not a flake. Confirm the B12X plan cache was reused (mtime
   unchanged) as in the la-tc run.

2. **Throughput gate, first thing after boot, before any tool request:**

       cd tools/tony-bench && python3 bench_sweep.py http://localhost:8000 \
         qwen3.8-flash-next la-tcc --levels 1 --rounds 3 --out /tmp/la-tcc_c1.json

   Run it **three times**. Every c1 `agg_tok_s` must be **>= 95**. Baseline is
   95.7 (`results/arms/la/sweep.json`) / 97.2 (the fresh la reboot). If any
   run lands at 85-88 the fix has not removed the cost -- stop, and run the
   H1/H2 experiment above before anything else.

   While here, this *is* step 2 of the distinguishing experiment: record the
   number, then run the single forced-tool curl from step 3, then repeat the
   sweep. Two extra minutes, and it settles H1 vs H2 for good.

3. **TC-45 alone:**

       tool-eval-bench run --hardmode --temperature 0.0 --backend vllm \
         --timeout 600 --max-turns 32 --base-url http://localhost:8000 \
         --model qwen3.8-flash-next --scenarios TC-45

   **Run it three times.** TC-45 is ~50/50 under this grammar (see the v2
   banner), so a single FAIL proves nothing. The gate is: **at least one PASS
   in three**, and *every* run -- pass or fail -- must have a turn time of
   **40 s or more**. A sub-second failure is the real regression signal: it
   means no grammar was built at all, i.e. the gate or the collapse subclass
   is broken. (The mod's boot self-check already covers that case, so a
   sub-second failure here would mean something further downstream.)

4. **Full hardmode:** `scripts/gate_arm.sh la-tcc`. Quality must be **>= 90**
   (la 91, la-tc 93). TC-68 is expected to keep failing -- by design, see
   `notes.md`; it is not a regression. Check no *other* scenario regressed
   against `results/arms/la/hardmode.log`.

5. **Post-gate sanity:** the gate's own sweep (which runs after hardmode, the
   way la-tc's did) must also be >= 95 at c1. That is the apples-to-apples
   comparison against the 85.0 that caused the revert.

6. Promote only if 2, 3, 4 and 5 all hold. Then update `notes.md` with the
   numbers and retire `mods/vllm-tc45-reasoning-structag-fix` and
   `recipes/eugr/eugr-agents-serve-local16-la-tc.yaml`. The two mods are
   mutually exclusive; the new mod refuses to run on top of the old one.
