# TC-45 / TC-68 root-cause diagnosis (Phase A, read-only, 2026-09-19)

Repo: qwen3.8-flash-next-dgx-spark-tp-2
vLLM fork: ~/GEN-AI/build/vllm (dev/jovian-judgement, 8e1f1e58)
Serving recipe under test: recipes/eugr/eugr-agents-serve-local16-la.yaml
  (--tool-call-parser qwen3_xml --enable-auto-tool-choice --reasoning-parser qwen3)

## TC-45 "tool_choice=required Compliance" -> "No tool calls despite tool_choice=required"

Scenario source: tool_eval_bench/evals/scenarios/agentic/tc45.py
  user_message = "What is 7 times 8?", tool_choice_override = "required".
The harness first probes `supports_tool_choice_required()` (orchestrator.py:744) with a
trivial forced `probe_ping` call; TC-45 is only scored (not excluded) when that probe
succeeds, so the endpoint DOES honor tool_choice=required at least some of the time.

Trace through the fork:
- vllm/entrypoints/openai/chat_completion/protocol.py `extract_structured_outputs()` /
  `to_sampling_params()` build `SamplingParams.structured_outputs` ONLY from
  `response_format`; tool_choice is never consulted there.
- The actual enforcement path lives in vllm/parser/abstract_parser.py:
  `Parser.adjust_request()` -> `_apply_structural_tag()` (called from
  vllm/renderers/online_renderer.py:487, the mainline chat-completions render path,
  gated only by `tool_choice != "none"` -- so it does run for "required").
- `_apply_structural_tag()` requires `tool_parser.structural_tag_model` to be set.
  `Qwen3EngineToolParser.structural_tag_model = "qwen_3_coder"`
  (vllm/tool_parsers/qwen3_engine_tool_parser.py), and "qwen_3_coder" IS one of the
  xgrammar builtin structural-tag models
  (XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS in structural_tag_registry.py), so the
  lookup succeeds and delegates to xgrammars `get_model_structural_tag()`.
- Because `structural_tag_model is not None` and `VLLM_ENFORCE_STRICT_TOOL_CALLING`
  defaults to True, `ToolParser.__init_subclass__` auto-sets
  `Qwen3EngineToolParser.supports_required_and_named = False`. That flag only affects
  how a *finished* completion is parsed after the fact (falls back to normal
  auto-style extract_tool_calls instead of the naive "parse whole content as JSON tool
  call list" path) -- it does not gate whether the structural tag is applied.

**Root cause found:** `_apply_structural_tag()` calls
`self._tool_parser.get_structural_tag(request, reasoning=False)` with a **hardcoded
`reasoning=False`**, which is forwarded straight into xgrammars
`get_model_structural_tag(..., reasoning=False, ...)`. That builds a grammar whose
completion must begin directly with a tool call / plain-text tag -- it does NOT permit
a leading `<think>...</think>` block. But the recipe runs with
`--reasoning-parser qwen3` and hardmode gate ("thinking on", matches baseline
methodology), so the model is expected to open every completion with `<think>`.
The grammar the server actually constrains against therefore conflicts with the
reasoning mode the recipe runs in: xgrammar has to mask out the models trained
`<think>` opening token for every request where tool_choice is required/named, which
is consistent with a model that, for a "should I even bother with a tool" trivial
question like "What is 7 times 8?", ends up emitting no usable tool call at all
(unlike the harnesss own probe, which asks the model to do something
unambiguous and unrelated to reasoning-heavy trivial math). The probe passing while
the real scenario fails is explained by the same grammar+thinking mismatch: probe_ping
is a single unambiguous instruction, TC-45s math question is exactly the class of
prompt where a reasoning model without its `<think>` scratchpad degrades.

Fix (smallest): make `reasoning=` in `_apply_structural_tag()` reflect whether a
reasoning parser is actually active for the request instead of a hardcoded False,
e.g.:
```python
structure_tag = self._tool_parser.get_structural_tag(
    request,
    reasoning=self._reasoning_parser is not None,
)
```
(vllm/parser/abstract_parser.py, function `_apply_structural_tag`, ~line 547).
xgrammars `get_model_structural_tag` accepts `reasoning: bool | Literal["enabled",
"disabled","auto"]` (default `"enabled"`), so passing True (or "auto") lets the
grammar admit a `<think>...</think>` prefix before the forced tool-call tag, matching
how the model is actually being run. This is a 1-line change in the vLLM fork; ship it
as a small patch/mod (mods/ dir convention) rather than a recipe flag, since no CLI
flag exposes this.

Verification plan for Phase B: rebuild/patch, re-run TC-45 alone
(`tool-eval-bench run --hardmode ... ` with scenario filtering if supported, else full
hardmode), confirm calculator tool call appears with expression 7*8.

## TC-68 "Schema Violation Resistance" -> "Output is not valid JSON"

Scenario source: tool_eval_bench/evals/scenarios/structured/tc68.py.
The scenario **deliberately sets no `response_format_override`** -- the code comment
says sending the schema via response_format would let the server enforce
`additionalProperties: false` server-side and make the test "trivially passable". So
this is intentionally a pure-prompting test of whether the model, completely
unconstrained by the server, honors "no extra fields" and emits clean JSON.

Given that, TC-68 failing "Output is not valid JSON" is NOT a structured-output
backend bug (nothing server-side is asked to enforce the schema) -- xgrammar is
irrelevant here by design. `state.final_answer` is `result.content` (the parsed
non-reasoning content from the qwen3 reasoning-parser adapter,
vllm/reasoning/qwen3_engine_reasoning_parser.py -> Qwen3ParserReasoningAdapter, an
engine-based parser); the harness does `json.loads(answer.strip())` with **no
fence-stripping / extraction**. The most likely failure mode is the model wrapping its
JSON answer in a ```json ... ``` code fence or adding a short prose
preamble/postamble under the "also include priority/due date/hours" pressure from the
user turn -- both are legal completions of "output as JSON matching this schema" that
a raw `json.loads` will reject.

This is not fixable via `--structured-outputs-config` / `--guided-decoding-backend`
without defeating the scenarios intent (would trivially force compliance and
misrepresent whether the *model* resists the pressure). The legitimate fix vectors are
model-behavior ones:
1. A system-prompt addendum (via a recipe `--chat-template-kwargs` default system
   message, or a lightweight mod that injects one) instructing the model to emit raw
   JSON only, no markdown fences, when asked to "output as JSON" -- verify this does
   not also change tool-choice-driven scenarios via A/B.
2. Confirm the qwen3 reasoning parser is fully stripping `<think>` from `content` for
   this checkpoint/quantization (fp8 KV + modelopt_mixed) -- if the closing `</think>`
   tag is ever mis-detected (e.g. truncated by max-token limits or b12x decode
   quirks), partial reasoning text would leak into `content` and break `json.loads`
   independent of fences. Recommend capturing the raw TC-68 transcript in Phase B
   (`--hardmode` log already includes per-scenario traces) to see the literal
   `final_answer` string before proposing (1) vs (2).

No code/flag fix proposed for TC-68 pending that raw-transcript check in Phase B --
this is the one item that needs a live request to nail down definitively.

## Phase B result: TC-45 fixed, but the fix regresses c1 throughput -- REVERTED

### Root cause was deeper than the Phase A hypothesis

Live testing (once GPUs were free) falsified the Phase A hypothesis (hardcoded
`reasoning=False` in `DelegatingParser._apply_structural_tag()`). A raw TC-45
transcript against the unmodified server showed the model reasoning explicitly
("I dont need to use a calculator tool for this basic multiplication") and then
answering in free text -- proof that generation was completely unconstrained,
which a grammar-forcing bug (as opposed to a grammar-missing bug) could not
produce.

Actual root cause: `ParserManager.get_parser()` (vllm/parser/parser_manager.py)
detects that Qwen3s reasoning parser and tool parser both back onto the same
`ParserEngine` subclass (`Qwen3Parser`, vllm/parser/qwen3.py) and "collapses"
them, returning the raw engine class directly instead of a `DelegatingParser`.
`ParserEngine.adjust_request()` (vllm/parser/engine/parser_engine.py) is a
no-op stub (`request.skip_special_tokens = False; return request`) -- it never
builds or applies an xgrammar structural tag. `DelegatingParser._apply_structural_tag()`
(the code that does build one) is therefore **unreachable for every model that
collapses onto a shared ParserEngine**: Qwen3, Kimi K2, GLM-4.7-MoE, DeepSeek
engine variants, Gemma4, Mistral, SeedOss, Inkling, NemotronV3, Minimax M2.
Confirmed live: `ParserManager.get_parser(tool_parser_name="qwen3_xml",
reasoning_parser_name="qwen3", enable_auto_tools=True, ...)` returns
`vllm.parser.qwen3.Qwen3Parser` directly.

A second, compounding bug: even the collapsed engine classs `tool_parser_cls`
class attribute was stale -- `make_adapters()` (vllm/parser/engine/adapters.py)
sets `Qwen3Parser.tool_parser_cls` once at import time to the generic
`ParserEngineToolAdapter` subclass, which never carries `structural_tag_model`;
the actually-resolved tool parser for "qwen3_xml" is a *further* subclass,
`Qwen3EngineToolParser` (vllm/tool_parsers/qwen3_engine_tool_parser.py), which
does set `structural_tag_model = "qwen_3_coder"`.

### Fix applied (mods/vllm-tc45-reasoning-structag-fix)

Three files patched inside the image (see `mods/vllm-tc45-reasoning-structag-fix/`):
- `abstract_parser.py`: `reasoning=False` -> `reasoning=self._reasoning_parser is not None`
  in `DelegatingParser._apply_structural_tag()` (latent fix for non-collapsed models).
- `parser_manager.py`: the collapse branch now re-points
  `reasoning_engine_cls.tool_parser_cls`/`.reasoning_parser_cls` to the
  actually-resolved classes instead of leaving the stale generic ones.
- `parser_engine.py`: `ParserEngine.adjust_request()` gained an
  `_apply_structural_tag()` (mirrors `DelegatingParser`s), using
  `self._has_reasoning` (correctly computed per-model from
  `parser_engine_config`) instead of a hardcoded `False`.

Recipe copy: `recipes/eugr/eugr-agents-serve-local16-la-tc.yaml` (la + this mod).

### Verification

- Live-verified parser resolution after the fix:
  `tool_parser_cls -> Qwen3EngineToolParser`, `structural_tag_model ->
  "qwen_3_coder"`, `_apply_structural_tag` present on the returned class.
- `tool-eval-bench run --scenarios TC-45 TC-68 --hardmode`: **TC-45 now PASSES**
  ("Used calculator with correct expression -- honored tool_choice=required").
  TC-68 still fails as expected/by-design (see above).
- Full hardmode gate (`scripts/gate_arm.sh la-tc`): **Quality 93/100** (vs la
  baseline 91/100 the day before) -- meets and slightly beats the gate, no
  scenario regressed other than TC-68 (already failing on baseline too, not a
  new failure).

### But: c1 throughput regressed below the gate -- reverted to plain `la`

`results/arms/la-tc/sweep.json`: **c1 = 85.0 tok/s** (agg == per-stream, single
stream) vs `results/arms/la/sweep.json` baseline **c1 = 95.7 tok/s** (~11% drop).
c4/c8/c16 aggregate throughput were within noise of baseline (276/417/631
tok/s vs 285/424/630), so this is a single-stream-latency-bound regression,
not a batching/scheduler regression.

Isolation test (per runbook): stopped la-tc, rebooted plain `la` (recipe
unchanged, same image, same B12X plan cache -- confirmed reused, `.../b12x/compile`
mtime still Sep 18 19:33, untouched by either boot). Fresh la reboot measured
**c1 = 97.2 tok/s** (`/tmp/la_reboot1_sweep.json`) on the first attempt --
matches the historical baseline, so this is **not boot-to-boot variance**; the
regression is specific to the -tc mod. Resolved `vllm serve ...` command line
(read from `/proc/<pid>/cmdline` inside the running container) is
byte-identical between la and la-tc -- the only difference is the three
patched python files.

Root-cause hypothesis for the throughput hit (not yet confirmed, flagged for
follow-up): `tools/tony-bench/bench_sweep.py` sends no `tools`/`tool_choice`
in its payload, so `_apply_structural_tag()`s own early-return
(`if not tools: return`) should make it a no-op for sweep requests -- a
per-request setup cost couldnt explain a sustained *decode*-rate regression
at single-stream concurrency anyway. The more likely mechanism: this is the
first time any request in this fork exercises xgrammars *structural_tag*
enforcement path (as opposed to the pre-existing, response_format-driven
plain JSON-schema guided decoding, which was already exercised by both las
and la-tcs own hardmode runs without any c1 regression). If the structural-tag
backend forces the sampler off the CUDA-graph-only decode path for the engine
process once triggered (e.g. during the TC-45/hardmode run that precedes the
sweep in the same boot), that would show up as exactly this kind of sticky,
single-stream-latency-bound slowdown for the rest of the process lifetime.
Not confirmed -- would need an nsys/py-spy trace comparing a request before
vs. after the first tool_choice=required call in the same process to nail
down. Out of scope for this pass.

### Decision

Per the Phase B gate ("if it holds, keep the -tc recipe serving; else revert
to the previous recipe"): **reverted**. Serving recipe is plain
`recipes/eugr/eugr-agents-serve-local16-la.yaml` (TC-45 unfixed, c1 healthy).
The `-tc` recipe and mod are kept in the repo for follow-up (fix TC-45s root
cause without the throughput side effect), but are NOT the serving recipe.
