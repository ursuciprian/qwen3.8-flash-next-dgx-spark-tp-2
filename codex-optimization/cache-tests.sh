#!/usr/bin/env bash
# Inside the pinned container with repository mounted at /work, writable test tree.
set -euo pipefail
cd /work/codex-optimization
python3 install.py cache
cd cache-backport
touch tests/__init__.py tests/v1/__init__.py tests/v1/core/__init__.py tests/v1/core/prefix_cache/__init__.py
# This one newer SWA test fails identically on the original image: its API is absent.
PYTHONPATH=. python3 -m pytest --noconftest -q \
  tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py \
  tests/v1/core/test_prefix_caching.py \
  -k 'not test_swa_reachable_block_mask_final_partial_segment'
