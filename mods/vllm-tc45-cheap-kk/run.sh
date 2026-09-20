#!/usr/bin/env bash
# vllm-tc45-cheap-kk -- TC-45 (tool_choice=required compliance) without the
# decode tax, for the KK image (built from vllm.git a229c7a1).
#
# a229c7a1 already carries the cherry-picked 5d1df69f fix (commit
# 9e0825c968: honor tool_choice=required for models on the shared
# ParserEngine, reasoning= passed through instead of hardcoded False) but
# still has both cost bugs mods/vllm-tc45-cheap (the jovian image's fix)
# addresses:
#
#   - abstract_parser.py / parser_engine.py gate on tool_choice="auto" as
#     well as required/named. "auto" leaves the model free to answer in
#     prose, so the tag constrains nothing it was not already allowed to
#     emit, while making every agentic request a structured-output request
#     for its whole lifetime (deferred sampling in
#     EngineCore.step_with_batch_queue, the draft-token D2H copy async
#     scheduling otherwise skips, the grammar bitmask fill/copy/kernel, and
#     grammar rejection of MTP drafts). Gate on needs_structural_tag()
#     (required/named only) instead.
#   - parser_manager.py's collapse branch mutates reasoning_engine_cls
#     in place -- that class is process-wide state every other model backed
#     by the same shared ParserEngine would inherit. Point a per-call
#     subclass at the resolved classes instead.
#
# See patches/vllm-tc45-cheap/0003-tc45-cheap-kk-a229c7a1.patch and
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
  [parser/abstract_parser.py]=50a718d2554c63cef445f383830e3abe63736ae009f21ec1f1ac991702eb8bf5
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

if grep -q "def needs_structural_tag(" "$V/parser/abstract_parser.py"; then
  echo "mod vllm-tc45-cheap-kk: already applied, skipping"
  exit 0
fi

python3 "$HERE/apply.py" "$V"

for f in "${FILES[@]}"; do
  have=$(sha256sum "$V/$f" | cut -d' ' -f1)
  [ "$have" = "${POST[$f]}" ] || echo "mod vllm-tc45-cheap-kk: note $f -> $have (expected ${POST[$f]})" >&2
done

# Import smoke + behaviour check. Fails the boot rather than serving a server
# whose tool_choice=required is silently a no-op again, or whose gate has
# quietly widened back to "auto".
python3 - <<'PY'
import sys
from vllm.parser.abstract_parser import needs_structural_tag
from vllm.parser.parser_manager import ParserManager


class Req:
    tools = None
    tool_choice = "required"


assert not needs_structural_tag(Req()), "no tools must not build a grammar"
Req.tools = [{"type": "function", "function": {"name": "f"}}]
for choice in (None, "none", "auto"):
    Req.tool_choice = choice
    assert not needs_structural_tag(Req()), f"tool_choice={choice!r} must stay grammar-free"
Req.tool_choice = "required"
assert needs_structural_tag(Req()), "tool_choice=required must build a grammar"

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
from vllm.parser.qwen3 import Qwen3Parser

assert issubclass(parser, Qwen3Parser) and parser is not Qwen3Parser, (
    "collapse branch must return a per-call subclass, not the shared class"
)
assert Qwen3Parser.tool_parser_cls is not tp, "shared engine class was mutated"
print("mod vllm-tc45-cheap-kk: self-check ok ->", parser.__name__, tp.__name__)
PY

echo "mod vllm-tc45-cheap-kk: applied (tool_choice=required/named builds the xgrammar structural tag; auto and tool-free requests stay off the structured-output path)"
