# Mods

Empty on purpose. The registry-visible recipes (`recipes/qwen3.8-flash-next/`)
need no mods: every fix they rely on is baked into the
`ghcr.io/ursuciprian/spark-vllm-b12x:b0-20260918-a8333658-warm` image.

Every mod used by an experiment, bisection arm or older route lives in
[`archive/mods/`](../archive/mods/README.md), next to the recipes that use it
in `archive/recipes/`. sparkrun never checks out or scans `archive/`.
