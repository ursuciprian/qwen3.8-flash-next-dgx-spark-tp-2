import sys

manager_path, engine_path = sys.argv[1], sys.argv[2]

# --- Fix A: parser_manager.py collapse branch keeps the stale generic
# tool_parser_cls (set once at import time by make_adapters()) instead of the
# actually-resolved subclass (e.g. Qwen3EngineToolParser, which carries
# structural_tag_model). Re-point both class attrs to the resolved classes.
src = open(manager_path).read()
old = (
    "        if reasoning_engine_cls is not None and reasoning_engine_cls is tool_engine_cls:\n"
    "            return reasoning_engine_cls\n"
)
new = (
    "        if reasoning_engine_cls is not None and reasoning_engine_cls is tool_engine_cls:\n"
    "            # tc45-structag-fix: the collapsed engine class's tool_parser_cls\n"
    "            # was set once at import time by make_adapters() to the generic\n"
    "            # ParserEngineToolAdapter subclass, which does not carry\n"
    "            # structural_tag_model. Re-point it to the actually-resolved\n"
    "            # tool_parser_cls (e.g. Qwen3EngineToolParser) so\n"
    "            # ParserEngine.adjust_request() can build a tool_choice=required\n"
    "            # xgrammar structural tag.\n"
    "            if tool_parser_cls is not None:\n"
    "                reasoning_engine_cls.tool_parser_cls = tool_parser_cls\n"
    "            if reasoning_parser_cls is not None:\n"
    "                reasoning_engine_cls.reasoning_parser_cls = reasoning_parser_cls\n"
    "            return reasoning_engine_cls\n"
)
if old not in src:
    print("FATAL: parser_manager.py collapse-branch block not found (source drift)", file=sys.stderr)
    sys.exit(1)
src = src.replace(old, new, 1)
open(manager_path, "w").write(src)

# --- Fix B: ParserEngine.adjust_request() is a no-op stub (only sets
# skip_special_tokens=False). It never builds a structural tag, so
# tool_choice=required/named has zero effect on generation for any model
# collapsed onto a single ParserEngine (Qwen3, Kimi K2, GLM-4.7-MoE,
# DeepSeek engine variants, Gemma4, Mistral, SeedOss, Inkling, NemotronV3,
# Minimax M2). Add the same structural-tag construction that
# DelegatingParser._apply_structural_tag() does, using self._has_reasoning
# (already computed correctly from the model's parser_engine_config) instead
# of a hardcoded reasoning=False.
esrc = open(engine_path).read()
old_import = "from vllm.parser.abstract_parser import Parser, StreamState"
new_import = (
    "from vllm.parser.abstract_parser import Parser, StreamState\n"
    "from vllm.sampling_params import StructuredOutputsParams"
)
if old_import not in esrc:
    print("FATAL: parser_engine.py import line not found (source drift)", file=sys.stderr)
    sys.exit(1)

old_method = (
    "    def adjust_request(\n"
    "        self, request: ChatCompletionRequest | ResponsesRequest\n"
    "    ) -> ChatCompletionRequest | ResponsesRequest:\n"
    "        request.skip_special_tokens = False\n"
    "        return request\n"
)
new_method = (
    "    def adjust_request(\n"
    "        self, request: ChatCompletionRequest | ResponsesRequest\n"
    "    ) -> ChatCompletionRequest | ResponsesRequest:\n"
    "        request.skip_special_tokens = False\n"
    "        self._apply_structural_tag(request)\n"
    "        return request\n"
    "\n"
    "    def _apply_structural_tag(\n"
    "        self, request: ChatCompletionRequest | ResponsesRequest\n"
    "    ) -> None:\n"
    "        # tc45-structag-fix: mirrors DelegatingParser._apply_structural_tag(),\n"
    "        # which this engine-based (collapsed) Parser path never inherited\n"
    "        # (ParserEngine.__init__ always sets self._tool_parser = None).\n"
    "        tool_parser_cls = type(self).tool_parser_cls\n"
    "        if tool_parser_cls is None or getattr(\n"
    "            tool_parser_cls, \"structural_tag_model\", None\n"
    "        ) is None:\n"
    "            return\n"
    "        tools = getattr(request, \"tools\", None) or self._tools\n"
    "        if not tools:\n"
    "            return\n"
    "        from vllm.entrypoints.openai.chat_completion.protocol import (\n"
    "            ChatCompletionNamedToolChoiceParam,\n"
    "        )\n"
    "        from openai.types.responses import ToolChoiceFunction\n"
    "\n"
    "        tool_choice = getattr(request, \"tool_choice\", None)\n"
    "        need_tool_calling = (\n"
    "            tool_choice == \"auto\"\n"
    "            or tool_choice == \"required\"\n"
    "            or isinstance(\n"
    "                tool_choice,\n"
    "                (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),\n"
    "            )\n"
    "        )\n"
    "        if not need_tool_calling:\n"
    "            return\n"
    "        if getattr(request, \"structured_outputs\", None) is not None and getattr(\n"
    "            request.structured_outputs, \"structural_tag\", None\n"
    "        ) is not None:\n"
    "            return\n"
    "        tool_parser = tool_parser_cls(self.model_tokenizer, tools)\n"
    "        structure_tag = tool_parser.get_structural_tag(\n"
    "            request, reasoning=self._has_reasoning\n"
    "        )\n"
    "        if structure_tag is None:\n"
    "            return\n"
    "        request.structured_outputs = StructuredOutputsParams(\n"
    "            structural_tag=json.dumps(structure_tag.model_dump())\n"
    "        )\n"
    "        if hasattr(request, \"response_format\"):\n"
    "            request.response_format = None\n"
)
if old_method not in esrc:
    print("FATAL: parser_engine.py adjust_request method body not found (source drift)", file=sys.stderr)
    sys.exit(1)
esrc = esrc.replace(old_import, new_import, 1)
esrc = esrc.replace(old_method, new_method, 1)
open(engine_path, "w").write(esrc)
