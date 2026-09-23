#!/usr/bin/env bash
# fp8_e4m3 main KV cache on the Qwen4Exp QSA path: upstream vllm-project/vllm#55557, applied as a
# patch because it is still open.
#
# Without it, a stock vLLM nightly refuses the recipe at engine init:
#   NotImplementedError: Qwen4Exp QSA requires a BF16 main KV cache
# raised in models/qwen4_exp/nvidia/qsa.py, with three further BF16 assertions behind it (the
# backend's supported_kv_cache_dtypes list, a Q/K/V dtype check in forward_qsa, and a cache-storage
# check in the attention owner's __init__).
#
# The patch teaches the Triton kernel in ops/qsa.py to dequantize fp8 pages as it gathers them, so
# only the selected pages are converted and the cache itself stays fp8. That is upstream's design,
# not ours; carrying it here is strictly a time shift until it merges.
#
# Why this rather than our own overlays: the six whole-file overlays taken from build 8a728663 do
# not survive on a newer engine. They crash with
#   RuntimeError: split_with_sizes expects split_sizes to sum exactly to 2560 ... got [512, 128]
# which is the same failure that blocked the DeepSeek-V4-Flash nightly move. A patch against the
# build's own files has no such drift.
#
# Fail-closed: a dry run must succeed before anything is written.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P=/usr/local/lib/python3.12/dist-packages
[ -f "$P/vllm/models/qwen4_exp/nvidia/qsa.py" ] || { echo "mod vllm-qsa-fp8kv-pr55557: qwen4_exp not in this image, skipping"; exit 0; }

if ! grep -q "QSA requires a BF16 main KV cache" "$P/vllm/models/qwen4_exp/nvidia/qsa.py"; then
  echo "mod vllm-qsa-fp8kv-pr55557: BF16 gate absent, already patched or merged upstream; skipping"
  exit 0
fi

if ! patch -p1 -d "$P" --dry-run < "$HERE/pr55557.diff" > /tmp/pr55557-dryrun.txt 2>&1; then
  echo "FATAL: vllm-project/vllm#55557 does not apply to this build:" >&2
  cat /tmp/pr55557-dryrun.txt >&2
  exit 1
fi
patch -p1 -d "$P" < "$HERE/pr55557.diff" > /dev/null
python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" \
  "$P/vllm/models/qwen4_exp/nvidia/qsa.py" "$P/vllm/models/qwen4_exp/nvidia/ops/qsa.py"
echo "mod vllm-qsa-fp8kv-pr55557: applied (fp8_e4m3 KV enabled on the QSA path)"
