import sys

path = sys.argv[1]
src = open(path).read()
old = (
    "        structure_tag = self._tool_parser.get_structural_tag(\n"
    "            request,\n"
    "            reasoning=False,\n"
    "        )"
)
new = (
    "        structure_tag = self._tool_parser.get_structural_tag(\n"
    "            request,\n"
    "            reasoning=self._reasoning_parser is not None,\n"
    "        )"
)
if old not in src:
    print(
        "FATAL: vllm-tc45-reasoning-structag-fix: expected exact block not found "
        "(source drift) in " + path,
        file=sys.stderr,
    )
    sys.exit(1)
src = src.replace(old, new, 1)
open(path, "w").write(src)
