#!/usr/bin/env bash
# vllm-tool-grammar-all -- adds VLLM_TOOL_GRAMMAR_ALL, an opt-in env var that
# extends the tool_choice="auto" structural-tag grammar (vllm-tc45-cheap-kk's
# needs_structural_tag gate) to every tool, not just tools with strict=true.
# Off by default (VLLM_TOOL_GRAMMAR_ALL=0); the recipe using this mod sets it
# to "1" in env to turn the cost probe/hardmode arm on.
#
# Net effect == vllm.git 869138f26 -> 98136849 (single commit,
# feat(parser): VLLM_TOOL_GRAMMAR_ALL holds auto tool arguments to the
# schema) on vllm/envs.py, vllm/parser/abstract_parser.py and
# vllm/tool_parsers/structural_tag_registry.py.
#
# Fail closed: every edit must match its pre-image verbatim, nothing is
# written unless all of them do, every touched file is ast-parsed, and the
# mod then imports the patched modules and asserts VLLM_TOOL_GRAMMAR_ALL
# actually changes auto_tools_need_grammar()'s answer.
#
# Runs via `docker exec --user root` (sparkrun pre-serve hook). Patches
# files by path (importlib.util.find_spec("vllm"), never imports vllm
# itself before patching) per sparkrun-gotchas-dgx-pair: importing engine
# modules here can leave root-owned caches the unprivileged server can't
# read.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

V=$(python3 -c "
import importlib.util
spec = importlib.util.find_spec('vllm')
if spec is None or not spec.submodule_search_locations:
    raise SystemExit('vllm not found')
print(list(spec.submodule_search_locations)[0])
")

FILES=(envs.py parser/abstract_parser.py tool_parsers/structural_tag_registry.py)

# sha256 of each file as built from vllm.git 869138f26 (this image's base:
# spark-vllm-b12x:candidate-b12x-e6b6b93e-vllm-869138f2).
declare -A PRE=(
  [envs.py]=3edca9f88575d8e1c2b983828d43209a1f28ee57b4e920f2a7684a3956d7b478
  [parser/abstract_parser.py]=552bde34a798025949895a563bcb95579ae5a32f0355a108e4f734c58df66dfc
  [tool_parsers/structural_tag_registry.py]=e1df5a8c31976e2a6f6f4ed08dc8eca79caf718bc0393670f7ec4f3f4b16c56c
)
# ... and after this mod has applied (vllm.git 98136849).
declare -A POST=(
  [envs.py]=20ca5a0bec0217c544f31fad93735cc1783e72ef6fa14afce67c0ec490a35bdf
  [parser/abstract_parser.py]=9d0a1dc4527fff00b8cad3dea77d034fb0892d43d449df6712ccd3d0a1f2f59a
  [tool_parsers/structural_tag_registry.py]=86993791b342d8e396dc25a6724916ec68f569fe58b92d702be947d99b479489
)

for f in "${FILES[@]}"; do
  [ -f "$V/$f" ] || {
    echo "FATAL: mod vllm-tool-grammar-all: $V/$f not in this image -- expected" \
         "vllm built from 869138f26" >&2
    exit 1
  }
done

for f in "${FILES[@]}"; do
  have=$(sha256sum "$V/$f" | cut -d' ' -f1)
  if [ "$have" = "${POST[$f]}" ]; then
    echo "mod vllm-tool-grammar-all: $f already at the post-image sha, skipping"
    exit 0
  elif [ "$have" != "${PRE[$f]}" ]; then
    echo "FATAL: mod vllm-tool-grammar-all: $f sha $have is not the recorded" \
         "869138f26 pre-image ${PRE[$f]} -- this image does not match, refusing" \
         "to guess" >&2
    exit 1
  fi
done

if /bin/grep -q "def auto_tools_need_grammar(" "$V/tool_parsers/structural_tag_registry.py"; then
  echo "mod vllm-tool-grammar-all: already applied, skipping"
  exit 0
fi

python3 "$HERE/apply.py" "$V"

for f in "${FILES[@]}"; do
  have=$(sha256sum "$V/$f" | cut -d' ' -f1)
  [ "$have" = "${POST[$f]}" ] || {
    echo "FATAL: mod vllm-tool-grammar-all: $f -> $have (expected ${POST[$f]}) --" \
         "patch did not land as expected" >&2
    exit 1
  }
done

# Import smoke + behaviour check. Fails the boot rather than serving a
# server whose VLLM_TOOL_GRAMMAR_ALL knob is silently a no-op.
python3 - <<'PY'
import importlib
import os

os.environ["VLLM_TOOL_GRAMMAR_ALL"] = "0"
import vllm.envs as envs
importlib.reload(envs)
assert envs.VLLM_TOOL_GRAMMAR_ALL is False, "default must be off"

from vllm.tool_parsers.structural_tag_registry import (
    auto_tools_need_grammar,
    _any_tool_strict,
)

plain_tool = [{"type": "function", "function": {"name": "f", "strict": False}}]

assert auto_tools_need_grammar is not None
assert not _any_tool_strict(plain_tool), "sanity: plain tool is not strict"
assert not auto_tools_need_grammar(plain_tool), (
    "VLLM_TOOL_GRAMMAR_ALL=0 must behave exactly like upstream _any_tool_strict"
)

os.environ["VLLM_TOOL_GRAMMAR_ALL"] = "1"
importlib.reload(envs)
assert envs.VLLM_TOOL_GRAMMAR_ALL is True

import vllm.tool_parsers.structural_tag_registry as m
importlib.reload(m)
assert m.auto_tools_need_grammar(plain_tool), (
    "VLLM_TOOL_GRAMMAR_ALL=1 must force a grammar even for a non-strict tool"
)

from vllm.parser.abstract_parser import needs_structural_tag

print(
    "mod vllm-tool-grammar-all: self-check ok -> auto_tools_need_grammar",
    "off/on both verified, needs_structural_tag importable:",
    needs_structural_tag.__name__,
)
PY

echo "mod vllm-tool-grammar-all: applied (VLLM_TOOL_GRAMMAR_ALL, default off, extends the tool_choice=auto structural-tag grammar to every tool when set to 1)"
