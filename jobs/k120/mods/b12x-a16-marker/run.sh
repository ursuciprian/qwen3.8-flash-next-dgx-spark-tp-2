#!/usr/bin/env bash
# k120 path marker mod (sparkrun mod, runs in each rank's container before vLLM starts). Appends a logging wrapper to
# b12x's A16 BF16 path module so serve.log shows, per rank, the plan decision for each qualifying query and the first
# serving call that took the BF16 path ("k120 marker: ..."). Used on the prebake boot only. Fail-closed: aborts when the
# module is missing (an image without the BF16 path must not boot as if it had it). Idempotent.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
F=/usr/local/lib/python3.12/dist-packages/b12x/gemm/blockscaled/_a16_dequant.py
MARKER="k120 path marker (mods/b12x-a16-marker)"
[ -s "$F" ] || { echo "mod b12x-a16-marker: $F missing, aborting" >&2; exit 1; }
grep -q "^def engaged" "$F" && grep -q "^def run" "$F" || { echo "mod b12x-a16-marker: engaged/run not in $F, aborting" >&2; exit 1; }
if grep -q "$MARKER" "$F"; then echo "mod b12x-a16-marker: already applied"; exit 0; fi
cat "$HERE/marker_snippet.py" >> "$F"
python3 -c "import ast; ast.parse(open('$F').read())"
echo "mod b12x-a16-marker: applied ($MARKER)"
