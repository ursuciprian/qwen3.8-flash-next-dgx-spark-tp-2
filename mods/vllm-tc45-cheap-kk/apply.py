#!/usr/bin/env python3
"""vllm-tc45-cheap-kk: apply the TC-45 cheap-path structural-tag fix to the
KK image's installed vLLM (built from a229c7a1).

Exact-block replacement, fail closed: every edit must match its pre-image
verbatim or nothing is written. Every touched file is ast-parsed before it
is written back. Idempotent: re-running on a patched tree is a no-op.

Net effect == vllm.git a229c7a1 -> patches/vllm-tc45-cheap/0003-tc45-cheap-kk-a229c7a1.patch
"""

import ast
import sys

MARKER = "def needs_structural_tag("

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages/vllm"

HELPERS = '''def needs_structural_tag(
    request: "ChatCompletionRequest | ResponsesRequest",
) -> bool:
    """Whether the request must be grammar-constrained into a tool call.

    Only ``required`` and named tool choices force one. ``auto`` leaves the
    model free to answer in prose, so a structural tag changes nothing it is
    allowed to emit -- but it does put the request on the engine's
    structured-output path for its whole lifetime: a per-step grammar bitmask
    fill, H2D copy and mask kernel, deferred sampling in
    EngineCore.step_with_batch_queue() (which drops async-scheduling overlap
    for the *whole* batch), and grammar rejection of speculative draft
    tokens. Keep the grammar for the case that actually needs it.
    """
    if not getattr(request, "tools", None):
        return False
    tool_choice = getattr(request, "tool_choice", None)
    return tool_choice == "required" or isinstance(
        tool_choice, (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction)
    )


def attach_structural_tag(
    request: "ChatCompletionRequest | ResponsesRequest",
    tool_parser,
    *,
    reasoning: bool,
) -> None:
    """Build ``tool_parser``'s xgrammar structural tag onto ``request``."""
    structure_tag = tool_parser.get_structural_tag(request, reasoning=reasoning)
    if structure_tag is None:
        return
    request.structured_outputs = StructuredOutputsParams(
        structural_tag=json.dumps(structure_tag.model_dump()),
    )
    if isinstance(request, ResponsesRequest):
        request.text = None
    else:
        request.response_format = None


'''

# ---- abstract_parser.py ----

OLD_TAG = '''    def _apply_structural_tag(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if (
            self._tool_parser is None
            or self._tool_parser.structural_tag_model is None
            or not request.tools
        ):
            return request

        need_tool_calling = (
            request.tool_choice == "auto"
            or request.tool_choice == "required"
            or isinstance(
                request.tool_choice,
                (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
            )
        )
        if not need_tool_calling:
            return request

        structure_tag = self._tool_parser.get_structural_tag(
            request,
            reasoning=self._reasoning_parser is not None,
        )
        if structure_tag is None:
            return request

        structural_tag = json.dumps(structure_tag.model_dump())
        request.structured_outputs = StructuredOutputsParams(
            structural_tag=structural_tag,
        )
        if isinstance(request, ResponsesRequest):
            request.text = None
        else:
            request.response_format = None
        return request
'''

NEW_TAG = '''    def _apply_structural_tag(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if (
            self._tool_parser is not None
            and self._tool_parser.structural_tag_model is not None
            and needs_structural_tag(request)
        ):
            attach_structural_tag(
                request,
                self._tool_parser,
                reasoning=self._reasoning_parser is not None,
            )
        return request
'''

# ---- parser_manager.py ----
#
# a229c7a1's ParserManager.get_parser() (via the 9e0825c968 cherry-pick of
# 5d1df69f) calls cls._get_parser_engine_cls(), but that static method was
# never ported into KK's parser_manager.py -- it is upstream commit
# 1a20d23dab ([PARSER][Mistral] unified engine-based parser for reasoning
# and tool calls), which KK's own parser_manager.py history never picked up,
# even though the ParserEngine/adapters machinery it reads
# (vllm/parser/engine/adapters.py's _parser_engine_cls) is present. Every
# server boot with both --reasoning-parser and --tool-call-parser set (e.g.
# Qwen3) crashes at startup with AttributeError before ever reaching the
# get_parser() call our TC-45 fix touches. Restore it verbatim from jovian
# (self-contained, no other dependency) so get_parser() -- and this mod's
# self-check -- can run at all.

OLD_MGR_CLASS_HEAD = '''class ParserManager:
    """
    Provides a unified Parser by composing reasoning and tool parser adapters.
    """

    @classmethod
    def get_tool_parser(
'''

NEW_MGR_CLASS_HEAD = '''class ParserManager:
    """
    Provides a unified Parser by composing reasoning and tool parser adapters.
    """

    @staticmethod
    def _get_parser_engine_cls(parser_cls):
        # Restored from vllm.git 8e1f1e58 (upstream 1a20d23dab) -- dropped
        # somewhere in KK's integration history despite the adapters
        # machinery it reads still being present. Without it,
        # get_parser() below raises AttributeError for every model that
        # sets both a reasoning and a tool parser.
        if parser_cls is None:
            return None
        parser_engine_cls = getattr(parser_cls, "_parser_engine_cls", None)
        if parser_engine_cls is None:
            return None

        from vllm.parser.engine.parser_engine import ParserEngine

        if not isinstance(parser_engine_cls, type) or not issubclass(
            parser_engine_cls, ParserEngine
        ):
            return None
        return parser_engine_cls

    @classmethod
    def get_tool_parser(
'''

OLD_MGR = '''        reasoning_engine_cls = cls._get_parser_engine_cls(reasoning_parser_cls)
        tool_engine_cls = cls._get_parser_engine_cls(tool_parser_cls)
        if reasoning_engine_cls is not None and reasoning_engine_cls is tool_engine_cls:
            # tc45-structag-fix: the collapsed engine class's tool_parser_cls
            # was set once at import time by make_adapters() to the generic
            # ParserEngineToolAdapter subclass, which does not carry
            # structural_tag_model. Re-point it to the actually-resolved
            # tool_parser_cls (e.g. Qwen3EngineToolParser) so
            # ParserEngine.adjust_request() can build a tool_choice=required
            # xgrammar structural tag.
            if tool_parser_cls is not None:
                reasoning_engine_cls.tool_parser_cls = tool_parser_cls
            if reasoning_parser_cls is not None:
                reasoning_engine_cls.reasoning_parser_cls = reasoning_parser_cls
            return reasoning_engine_cls
'''

NEW_MGR = '''        reasoning_engine_cls = cls._get_parser_engine_cls(reasoning_parser_cls)
        tool_engine_cls = cls._get_parser_engine_cls(tool_parser_cls)
        if reasoning_engine_cls is not None and reasoning_engine_cls is tool_engine_cls:
            # tc45-cheap: the collapsed engine class's tool_parser_cls was set
            # once at import time by make_adapters() to the generic
            # ParserEngineToolAdapter subclass, which does not carry
            # structural_tag_model. Point a per-call SUBCLASS at the
            # actually-resolved classes instead of mutating
            # reasoning_engine_cls in place -- that class is process-wide
            # state every other model backed by the same shared engine would
            # inherit.
            class _CollapsedParserEngine(reasoning_engine_cls):  # type: ignore[misc,valid-type]
                pass

            _CollapsedParserEngine.__name__ = reasoning_engine_cls.__name__
            _CollapsedParserEngine.__qualname__ = reasoning_engine_cls.__qualname__
            if tool_parser_cls is not None:
                _CollapsedParserEngine.tool_parser_cls = tool_parser_cls
            if reasoning_parser_cls is not None:
                _CollapsedParserEngine.reasoning_parser_cls = reasoning_parser_cls
            return _CollapsedParserEngine
'''

# ---- parser_engine.py ----

OLD_IMPORT = "from vllm.parser.abstract_parser import Parser, StreamState\n"
NEW_IMPORT = (
    "from vllm.parser.abstract_parser import (\n"
    "    Parser,\n"
    "    StreamState,\n"
    "    attach_structural_tag,\n"
    "    needs_structural_tag,\n"
    ")\n"
)

OLD_ENGINE = '''    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request.skip_special_tokens = False
        self._apply_structural_tag(request)
        return request

    def _apply_structural_tag(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> None:
        # tc45-structag-fix: mirrors DelegatingParser._apply_structural_tag(),
        # which this engine-based (collapsed) Parser path never inherited
        # (ParserEngine.__init__ always sets self._tool_parser = None).
        tool_parser_cls = type(self).tool_parser_cls
        if tool_parser_cls is None or getattr(
            tool_parser_cls, "structural_tag_model", None
        ) is None:
            return
        tools = getattr(request, "tools", None) or self._tools
        if not tools:
            return
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionNamedToolChoiceParam,
        )
        from openai.types.responses import ToolChoiceFunction

        tool_choice = getattr(request, "tool_choice", None)
        need_tool_calling = (
            tool_choice == "auto"
            or tool_choice == "required"
            or isinstance(
                tool_choice,
                (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
            )
        )
        if not need_tool_calling:
            return
        if getattr(request, "structured_outputs", None) is not None and getattr(
            request.structured_outputs, "structural_tag", None
        ) is not None:
            return
        tool_parser = tool_parser_cls(self.model_tokenizer, tools)
        structure_tag = tool_parser.get_structural_tag(
            request, reasoning=self._has_reasoning
        )
        if structure_tag is None:
            return
        request.structured_outputs = StructuredOutputsParams(
            structural_tag=json.dumps(structure_tag.model_dump())
        )
        if hasattr(request, "response_format"):
            request.response_format = None
'''

NEW_ENGINE = '''    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request.skip_special_tokens = False
        # This engine-based ("collapsed") Parser path never inherited
        # DelegatingParser's structural-tag step, because ParserEngine
        # always leaves self._tool_parser None. Check the shared gate first
        # -- tool_choice="auto" must stay grammar-free -- and only then
        # instantiate the tool parser: instantiating it constructs a second
        # parser engine, and the grammar itself is not free at decode time
        # (see needs_structural_tag).
        tool_parser_cls = type(self).tool_parser_cls
        if (
            tool_parser_cls is not None
            and getattr(tool_parser_cls, "structural_tag_model", None) is not None
            and needs_structural_tag(request)
        ):
            tools = getattr(request, "tools", None) or self._tools
            if tools and not (
                getattr(request, "structured_outputs", None) is not None
                and getattr(request.structured_outputs, "structural_tag", None)
                is not None
            ):
                attach_structural_tag(
                    request,
                    tool_parser_cls(self.model_tokenizer, tools),
                    reasoning=self._has_reasoning,
                )
        return request
'''

EDITS = {
    "parser/abstract_parser.py": [
        ("class DelegatingParser(Parser):", HELPERS + "class DelegatingParser(Parser):"),
        (OLD_TAG, NEW_TAG),
    ],
    "parser/engine/parser_engine.py": [
        (OLD_IMPORT, NEW_IMPORT),
        (OLD_ENGINE, NEW_ENGINE),
    ],
    "parser/parser_manager.py": [
        (OLD_MGR_CLASS_HEAD, NEW_MGR_CLASS_HEAD),
        (OLD_MGR, NEW_MGR),
    ],
}


def fail(msg):
    print(f"FATAL: vllm-tc45-cheap-kk: {msg}", file=sys.stderr)
    raise SystemExit(1)


staged = {}
for rel, edits in EDITS.items():
    path = f"{ROOT}/{rel}"
    try:
        src = open(path).read()
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            fail(
                f"{rel}: expected exactly 1 occurrence of the pre-image block, "
                f"found {n} (source drift). First line of block: {old.splitlines()[0]!r}"
            )
        src = src.replace(old, new, 1)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        fail(f"{rel}: patched source does not parse: {exc}")
    staged[path] = src

# Only write once every file has matched and parsed.
for path, src in staged.items():
    open(path, "w").write(src)
    print(f"  vllm-tc45-cheap-kk: patched {path}")
