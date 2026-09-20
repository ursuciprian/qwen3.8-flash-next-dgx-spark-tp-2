#!/usr/bin/env bash
# vllm-tc45-cheap-kk -- TC-45 (tool_choice=required compliance) without the
# decode tax, for the KK image (built from vllm.git a229c7a1).
#
# a229c7a1 already carries the cherry-picked 5d1df69f fix (commit
# 9e0825c968: honor tool_choice=required for models on the shared
# ParserEngine, reasoning= passed through instead of hardcoded False) but
# still has the cost bugs mods/vllm-tc45-cheap (the jovian image's fix)
# addresses:
#
#   - abstract_parser.py / parser_engine.py gated on tool_choice="auto"
#     unconditionally. get_model_structural_tag() (structural_tag_registry.py)
#     already returns None for plain auto (no strict tool), so plain auto
#     never put a request on the structured-output path in the first place --
#     the only waste there was constructing a tool parser (a second parser
#     engine, tokenizer vocab included) just to be handed None back. Gate on
#     needs_structural_tag(): required/named always, auto only when a tool
#     sets strict: true -- identical to what the registry decides anyway.
#   - parser_manager.py's collapse branch mutated reasoning_engine_cls in
#     place -- that class is process-wide state every other model backed by
#     the same shared ParserEngine would inherit. Point a per-call subclass
#     at the resolved classes instead.
#   - ParserManager.get_parser() called cls._get_parser_engine_cls(), which
#     was never ported into KK's parser_manager.py (upstream 1a20d23dab) --
#     any server booting with both --reasoning-parser and --tool-call-parser
#     set (e.g. Qwen3) crashed at startup before reaching either bug above.
#     Restored verbatim from jovian.
#
# See patches/vllm-tc45-cheap/0003-tc45-cheap-kk-a229c7a1.patch,
# patches/vllm-tc45-cheap/0003b-tc45-cheap-kk-a229c7a1.patch and
# results/arms/tooleval-fix/cheap-design.md.
#
# Fail closed: every edit must match its pre-image verbatim, nothing is
# written unless all of them do, every touched file is ast-parsed, and the
# mod then imports the patched modules and asserts the gate and the parser
# resolution.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V=/usr/local/lib/python3.12/dist-packages/vllm

FILES=(parser/abstract_parser.py parser/parser_manager.py parser/engine/parser_engine.py)

# sha256 of each file as built from vllm.git a229c7a1 (this image's base).
declare -A PRE=(
  [parser/abstract_parser.py]=34d3edd8f2606dc297518942185662f9cbea389e4ebb99458311086bc3ea5542
  [parser/parser_manager.py]=6116adcd47f09256a459c8e5e4b0a880564fa1ba490295211f5f25ed39e2f3bb
  [parser/engine/parser_engine.py]=0dd96a1182955b8ed8356289f26ffc0198cdfa64c5f65430dc19fd959a9dfb3a
)
# ... and after this mod has applied.
declare -A POST=(
  [parser/abstract_parser.py]=9217ebfe90f5d58fad14d644a3dafa751aeb04ebab99db34a1dd44d253d1ac60
  [parser/parser_manager.py]=08572e274726d04bc0b1057eae5897b732571529f32485748b0b212c60d2bc1f
  [parser/engine/parser_engine.py]=9453a3c0a1b9c3d8ad42515fdcfa66bccb1f2ddd2ad4d36db845e9327d8abff0
)

for f in "${FILES[@]}"; do
  [ -f "$V/$f" ] || { echo "mod vllm-tc45-cheap-kk: $V/$f not in this image, skipping"; exit 0; }
done

# vllm-tc45-cheap (the jovian mod) is mutually exclusive with this one --
# its helper docstring carries a Sphinx :mod: reference this mod's does not.
if /bin/grep -q ':mod:`vllm.v1.worker.gpu.structured_outputs`' "$V/parser/abstract_parser.py" 2>/dev/null; then
  echo "FATAL: mod vllm-tc45-cheap-kk: vllm-tc45-cheap (jovian) is already" \
       "applied to this image. The two mods are mutually exclusive -- drop" \
       "vllm-tc45-cheap from the recipe." >&2
  exit 1
fi

for f in "${FILES[@]}"; do
  have=$(sha256sum "$V/$f" | cut -d' ' -f1)
  if [ "$have" = "${POST[$f]}" ]; then
    echo "mod vllm-tc45-cheap-kk: $f already at the post-image sha, skipping"
    exit 0
  elif [ "$have" != "${PRE[$f]}" ]; then
    echo "mod vllm-tc45-cheap-kk: WARN $f sha $have is not the recorded a229c7a1" \
         "pre-image ${PRE[$f]} -- relying on the exact-block match below" >&2
  fi
done

if /bin/grep -q "_any_tool_strict" "$V/parser/abstract_parser.py"; then
  echo "mod vllm-tc45-cheap-kk: already applied, skipping"
  exit 0
fi

python3 "$HERE/apply.py" "$V"

for f in "${FILES[@]}"; do
  have=$(sha256sum "$V/$f" | cut -d' ' -f1)
  [ "$have" = "${POST[$f]}" ] || echo "mod vllm-tc45-cheap-kk: note $f -> $have (expected ${POST[$f]})" >&2
done

# Import smoke + behaviour check. Fails the boot rather than serving a server
# whose tool_choice=required is silently a no-op again, whose gate has
# quietly widened back to plain "auto", or whose strict-auto carve-out is
# missing.
python3 - <<'PY'
import json

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.parser.abstract_parser import attach_structural_tag, needs_structural_tag
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.parser_manager import ParserManager
from vllm.parser.qwen3 import Qwen3Parser
from vllm.tool_parsers.abstract_tool_parser import ToolParser

TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


def req(**kw):
    return ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "What is 7 times 8?"}], **kw
    )


# --- gate --------------------------------------------------------------
assert not needs_structural_tag(req(tool_choice="none")), "no tools -> no grammar"
assert not needs_structural_tag(req(tools=[TOOL], tool_choice="none"))
assert not needs_structural_tag(req(tools=[TOOL], tool_choice="auto")), (
    "plain auto must not build a grammar"
)
assert needs_structural_tag(req(tools=[TOOL], tool_choice="required")), (
    "tool_choice=required MUST build a grammar -- this is TC-45"
)
named = {"type": "function", "function": {"name": "calculator"}}
assert needs_structural_tag(req(tools=[TOOL], tool_choice=named)), "named tool choice"
strict = json.loads(json.dumps(TOOL))
strict["function"]["strict"] = True
assert needs_structural_tag(req(tools=[strict], tool_choice="auto")), (
    "auto + strict tool must keep its grammar (upstream behaviour)"
)

# --- the tag itself, end to end, no tokenizer and no GPU needed ---------
# ToolParser.get_structural_tag() only reads self.structural_tag_model and
# the request, so a probe stands in for Qwen3EngineToolParser here.
class _Probe:
    structural_tag_model = "qwen_3_coder"
    get_structural_tag = ToolParser.get_structural_tag


r = req(tools=[TOOL], tool_choice="required", response_format={"type": "text"})
attach_structural_tag(r, _Probe(), reasoning=True)
assert r.structured_outputs is not None, (
    "attach_structural_tag produced NO structured_outputs for tool_choice=required"
)
tag = r.structured_outputs.structural_tag
assert tag and "calculator" in tag, f"structural tag missing the tool: {tag!r:.200}"
assert r.response_format is None, "response_format must be cleared"

# --- the collapsed engine actually routes through the patched gate ------
parser = ParserManager.get_parser(
    tool_parser_name="qwen3_xml",
    reasoning_parser_name="qwen3",
    enable_auto_tools=True,
)
tp = parser.tool_parser_cls
assert getattr(tp, "structural_tag_model", None) == "qwen_3_coder", (
    f"collapsed engine resolved tool_parser_cls={tp!r} with "
    f"structural_tag_model={getattr(tp, 'structural_tag_model', None)!r}"
)
assert issubclass(parser, Qwen3Parser) and parser is not Qwen3Parser, (
    "collapse branch must return a per-call subclass, not the shared class"
)
assert Qwen3Parser.tool_parser_cls is not tp, "shared engine class was mutated"
assert parser.adjust_request is ParserEngine.adjust_request, (
    "the collapsed parser does not use the patched ParserEngine.adjust_request"
)
assert "needs_structural_tag" in ParserEngine.adjust_request.__code__.co_names, (
    "ParserEngine.adjust_request is not the patched one"
)
print(
    "mod vllm-tc45-cheap-kk: self-check ok ->",
    parser.__name__, tp.__name__, "tag", len(tag), "bytes",
)
PY

echo "mod vllm-tc45-cheap-kk: applied (tool_choice=required/named/strict-auto builds the xgrammar structural tag; plain auto and tool-free requests stay off the structured-output path)"
