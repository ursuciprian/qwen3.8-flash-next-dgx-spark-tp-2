#!/usr/bin/env python3
"""vllm-tool-grammar-all: apply the VLLM_TOOL_GRAMMAR_ALL patch to the
installed vLLM (built from 869138f26).

Exact-block replacement, fail closed: every edit must match its pre-image
verbatim or nothing is written. Every touched file is ast-parsed before it
is written back. Idempotent: re-running on a patched tree is a no-op.

Net effect == vllm.git 869138f26 -> 98136849 (vllm/envs.py,
vllm/parser/abstract_parser.py, vllm/tool_parsers/structural_tag_registry.py).
"""

import ast
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages/vllm"

# (relative path, old block, new block) -- one entry per hunk, in file order.
EDITS = [
    (
        "envs.py",
        '''    VLLM_ENFORCE_STRICT_TOOL_CALLING: bool = True
    VLLM_MQ_MAX_CHUNK_BYTES_MB: int = 16''',
        '''    VLLM_ENFORCE_STRICT_TOOL_CALLING: bool = True
    VLLM_TOOL_GRAMMAR_ALL: bool = False
    VLLM_MQ_MAX_CHUNK_BYTES_MB: int = 16''',
    ),
    (
        "envs.py",
        '''    "VLLM_ENFORCE_STRICT_TOOL_CALLING": lambda: (
        os.getenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "True").lower() in ("true", "1")
    ),
    # Control the max chunk bytes (in MB) for the rpc message queue.''',
        '''    "VLLM_ENFORCE_STRICT_TOOL_CALLING": lambda: (
        os.getenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "True").lower() in ("true", "1")
    ),
    # Constrain tool-call arguments to each tool's schema under
    # tool_choice="auto" even when no tool sets strict=true. The grammar is an
    # xgrammar triggered tag, so free text and reasoning stay unconstrained,
    # but every tools request becomes a structured-output request (no async
    # scheduling overlap for its batch while it runs). Off by default.
    "VLLM_TOOL_GRAMMAR_ALL": lambda: (
        os.getenv("VLLM_TOOL_GRAMMAR_ALL", "0").lower() in ("true", "1", "yes", "on")
    ),
    # Control the max chunk bytes (in MB) for the rpc message queue.''',
    ),
    (
        "envs.py",
        '''        "VLLM_SKIP_MODEL_NAME_VALIDATION",
        "LOCAL_RANK",''',
        '''        "VLLM_SKIP_MODEL_NAME_VALIDATION",
        # Request-level tool grammar; never reaches compiled graphs.
        "VLLM_TOOL_GRAMMAR_ALL",
        "LOCAL_RANK",''',
    ),
    (
        "parser/abstract_parser.py",
        '''    strict schema enforcement. Answering that here means an ordinary agentic''',
        '''    strict schema enforcement (or for every tool with VLLM_TOOL_GRAMMAR_ALL,
    at the cost below). Answering that here means an ordinary agentic''',
    ),
    (
        "parser/abstract_parser.py",
        '''    from vllm.tool_parsers.structural_tag_registry import _any_tool_strict

    return _any_tool_strict(tools)''',
        '''    from vllm.tool_parsers.structural_tag_registry import auto_tools_need_grammar

    return auto_tools_need_grammar(tools)''',
    ),
    (
        "tool_parsers/structural_tag_registry.py",
        '''    TriggeredTagsFormat,
)

from vllm.entrypoints.openai.chat_completion.protocol import (''',
        '''    TriggeredTagsFormat,
)

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import (''',
    ),
    (
        "tool_parsers/structural_tag_registry.py",
        '''        if isinstance(tool, ChatCompletionToolsParam) and tool.function.strict is True:
            return True
    return False


def get_model_structural_tag(''',
        '''        if isinstance(tool, ChatCompletionToolsParam) and tool.function.strict is True:
            return True
    return False


def auto_tools_need_grammar(
    tools: Sequence[ChatCompletionToolsParam | ResponsesTool],
) -> bool:
    """Whether tool_choice="auto" gets a tool-call grammar for these tools.

    Upstream: only when a tool opts into strict=true. VLLM_TOOL_GRAMMAR_ALL
    extends it to every tool (strict=false tools still get syntax only).
    """
    return envs.VLLM_TOOL_GRAMMAR_ALL or _any_tool_strict(tools)


def get_model_structural_tag(''',
    ),
    (
        "tool_parsers/structural_tag_registry.py",
        '''    if tool_choice == "auto" and not _any_tool_strict(tools):
        return None''',
        '''    if tool_choice == "auto" and not auto_tools_need_grammar(tools):
        return None''',
    ),
    (
        "tool_parsers/structural_tag_registry.py",
        '''    return get_xgrammar_model_structural_tag(
        model=model,
        tools=dumped_tools,
        tool_choice=dumped_tool_choice,
        reasoning=reasoning,
    )''',
        '''    return get_xgrammar_model_structural_tag(
        model=model,
        tools=dumped_tools,
        tool_choice=dumped_tool_choice,
        reasoning=reasoning,
        # Non-strict tools never promised a parameter order.
        any_order=envs.VLLM_TOOL_GRAMMAR_ALL,
    )''',
    ),
]


def fail(msg):
    print(f"FATAL: mod vllm-tool-grammar-all: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    # Group edits by file, read once, verify every block for every file
    # before writing anything back (fail closed).
    files = {}
    for rel, old, new in EDITS:
        files.setdefault(rel, {"path": f"{ROOT}/{rel}", "edits": []})
        files[rel]["edits"].append((old, new))

    contents = {}
    for rel, info in files.items():
        path = info["path"]
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            fail(f"{path} not found -- this image does not match 869138f26")

        already_patched = "def auto_tools_need_grammar(" in text if rel == "tool_parsers/structural_tag_registry.py" else False
        if already_patched:
            print(f"mod vllm-tool-grammar-all: {rel} already patched, skipping")
            contents = None
            break

        for old, new in info["edits"]:
            count = text.count(old)
            if count == 0:
                fail(
                    f"{rel}: expected pre-image block not found -- the installed "
                    "vllm does not match 869138f26 (or a prior patch already "
                    "changed this region). Refusing to guess."
                )
            if count > 1:
                fail(f"{rel}: pre-image block is not unique ({count} matches)")
            text = text.replace(old, new, 1)

        try:
            ast.parse(text, filename=path)
        except SyntaxError as e:
            fail(f"{rel}: patched file fails to parse: {e}")

        contents[rel] = (path, text)

    if contents is None:
        return

    for rel, (path, text) in contents.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"mod vllm-tool-grammar-all: patched {rel}")


if __name__ == "__main__":
    main()
