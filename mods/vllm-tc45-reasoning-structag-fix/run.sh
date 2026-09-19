#!/usr/bin/env bash
# TC-45 fix ("tool_choice=required Compliance"): tool_choice=required/named is
# supposed to force an xgrammar structural tag so the sampler is constrained to
# emit a tool call. Two independent bugs make that never happen for Qwen3 (and
# every other model whose reasoning + tool parser collapse onto one
# ParserEngine: Kimi K2, GLM-4.7-MoE, DeepSeek engine variants, Gemma4,
# Mistral, SeedOss, Inkling, NemotronV3, Minimax M2):
#
#   1. DelegatingParser._apply_structural_tag() (vllm/parser/abstract_parser.py)
#      hardcodes reasoning=False, which would forbid the <think> opening this
#      recipe's --reasoning-parser qwen3 always emits -- but this path is
#      unreachable for Qwen3 anyway (see #2), so it's a latent bug for models
#      that *do* go through DelegatingParser.
#   2. ParserManager.get_parser() (vllm/parser/parser_manager.py) detects that
#      Qwen3's reasoning parser and tool parser both back onto the same
#      ParserEngine class (Qwen3Parser) and "collapses" them, returning the
#      raw engine class directly instead of a DelegatingParser. ParserEngine's
#      own adjust_request() (vllm/parser/engine/parser_engine.py) is a no-op
#      stub (only sets skip_special_tokens=False) -- it never calls into any
#      structural-tag logic at all. Confirmed live: a TC-45 request against
#      the server (tool_choice=required, "What is 7 times 8?") produced a
#      free-text answer with the model's own <think> reasoning saying "I don't
#      need to use a calculator tool for this basic multiplication" -- i.e.
#      generation was completely unconstrained, proving no grammar was ever
#      applied.
#
# Fix (two files):
#   - parser_manager.py: the collapse branch re-points the returned engine
#     class's `tool_parser_cls` to the actually-resolved tool_parser_cls
#     (e.g. Qwen3EngineToolParser, which carries structural_tag_model =
#     "qwen_3_coder") instead of leaving the generic adapter that
#     make_adapters() set once at import time (which lacks
#     structural_tag_model).
#   - parser_engine.py: ParserEngine.adjust_request() gains an
#     _apply_structural_tag() mirroring DelegatingParser's, using
#     self._has_reasoning (already computed correctly from the model's own
#     parser_engine_config) instead of a hardcoded False.
#   - abstract_parser.py: same reasoning=False -> reasoning-aware fix for the
#     DelegatingParser path, for models that don't hit the collapse branch.
#
# See results/arms/tooleval-fix/notes.md for the full trace.
#
# Fail-closed: each edit does an exact-block match before writing, and every
# touched file is ast-parsed after editing.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P=/usr/local/lib/python3.12/dist-packages/vllm/parser
ABSTRACT="$P/abstract_parser.py"
MANAGER="$P/parser_manager.py"
ENGINE="$P/engine/parser_engine.py"

for f in "$ABSTRACT" "$MANAGER" "$ENGINE"; do
  [ -f "$f" ] || { echo "mod vllm-tc45-reasoning-structag-fix: $f not in this image, skipping"; exit 0; }
done

if grep -q "reasoning=self._reasoning_parser is not None" "$ABSTRACT"; then
  echo "mod vllm-tc45-reasoning-structag-fix: abstract_parser.py already patched, skipping that file"
else
  python3 "$HERE/apply_fix.py" "$ABSTRACT"
  python3 -c "import ast; ast.parse(open('$ABSTRACT').read())"
  echo "mod vllm-tc45-reasoning-structag-fix: abstract_parser.py patched (reasoning=True when a reasoning parser is active)"
fi

if grep -q "tc45-structag-fix" "$MANAGER" && grep -q "tc45-structag-fix" "$ENGINE"; then
  echo "mod vllm-tc45-reasoning-structag-fix: engine collapse fix already applied, skipping"
  exit 0
fi

python3 "$HERE/apply_engine_fix.py" "$MANAGER" "$ENGINE"
python3 -c "import ast; ast.parse(open('$MANAGER').read()); ast.parse(open('$ENGINE').read())"
echo "mod vllm-tc45-reasoning-structag-fix: parser_manager.py + parser_engine.py patched (collapsed ParserEngine now builds the tool_choice=required/named structural tag)"
