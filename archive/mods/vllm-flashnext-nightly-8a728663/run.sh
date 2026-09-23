#!/usr/bin/env bash
# Six overlays for vLLM nightly 8a728663 serving nvidia/Qwen3.8-Flash-Next-NVFP4 (fc694b54) on GB10.
# See patches/README.md in the repository for what each one fixes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P=/usr/local/lib/python3.12/dist-packages/vllm
declare -A MAP=(
  [ops_ple.py]="$P/models/qwen4_exp/nvidia/ops/ple.py"
  [ops_qsa.py]="$P/models/qwen4_exp/nvidia/ops/qsa.py"
  [qsa.py]="$P/models/qwen4_exp/nvidia/qsa.py"
  [platforms_interface.py]="$P/platforms/interface.py"
  [modelopt.py]="$P/model_executor/layers/quantization/modelopt.py"
  [kv_cache_utils.py]="$P/v1/core/kv_cache_utils.py"
)
for f in "${!MAP[@]}"; do
  t="${MAP[$f]}"
  [ -f "$t" ] || { echo "mod vllm-flashnext-nightly-8a728663: $t missing, is this the 8a728663 nightly?"; exit 1; }
  cp "$t" "$t.orig"; cp "$HERE/$f" "$t"; echo "mod vllm-flashnext-nightly-8a728663: replaced $t"
done
