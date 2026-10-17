#!/usr/bin/env bash
# k109 FlashInfer container entry (throwaway container of the 2x v2.2.0 image): install the staged nvidia-cutlass-dsl
# 4.7.1 wheels (the image has 4.7.0; FlashInfer's b12x extra pins 4.7.1), put the FlashInfer main source at $FISHA on
# PYTHONPATH (the SM12x CuTe DSL kernels compile at first call), then run moe109.py with the given arguments (k98).
set -uo pipefail
pip install -q --no-index --no-deps /wheels/*.whl 2>&1 | grep -v "Running pip as the 'root' user" | tail -3
export PYTHONPATH=/src PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
import importlib.metadata as md, flashinfer, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
for p in ("nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base", "nvidia-cutlass-dsl-libs-core", "nvidia-cutlass-dsl-libs-cu13", "apache-tvm-ffi"):
    try: print(p, md.version(p))
    except Exception as e: print(p, "MISSING", e)
print("flashinfer from", flashinfer.__file__)
PY
exec python3 /k109/moe109.py "$@"
