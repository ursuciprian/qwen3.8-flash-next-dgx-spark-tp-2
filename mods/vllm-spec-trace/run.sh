#!/usr/bin/env bash
# Instrumentation for the c5-c7 MTP straggler bug (kernel-sweep-2026-09-18.md §1/1a).
# Patches two installed vLLM files by path (no vllm import) to log, when
# VLLM_SPEC_TRACE=1, one line per spec-decode step to /tmp/spec_trace.log inside
# the container:
#   - autoregressive/speculator.py propose(): drafts proposed per request
#     (uniform across the batch unless the num_speculative_tokens==1 early exit)
#   - rejection_sampler.py __call__(): accepted/rejected counts per request,
#     straight from the same tensors the round-level "accept X/draft" metric sums.
# No-op (source unchanged) when the env var is unset — both insertions are gated
# at runtime, not at patch time, so this mod has zero cost when off.
# Idempotent and fail-closed: skips if already patched, aborts if anchors don't match.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P=/usr/local/lib/python3.12/dist-packages
SPEC="$P/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
REJ="$P/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py"

for f in "$SPEC" "$REJ"; do
  [ -f "$f" ] || { echo "mod vllm-spec-trace: $f not in this image, skipping"; exit 0; }
done

if grep -q VLLM_SPEC_TRACE "$SPEC" && grep -q VLLM_SPEC_TRACE "$REJ"; then
  echo "mod vllm-spec-trace: already patched, skipping"
  exit 0
fi

python3 "$HERE/patch.py" "$SPEC" "$REJ"

python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$SPEC" "$REJ"
echo "mod vllm-spec-trace: applied (propose()/rejection_sampler tracing, gated on VLLM_SPEC_TRACE=1)"
