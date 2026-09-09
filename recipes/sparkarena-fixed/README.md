# Corrected replacements for the two Spark Arena entries

Upload these over the published recipes. The SGLang one has a correctness
problem as published; the vLLM one is unchanged and only listed here so both
files sit together.

## `qwen38-flash-next-nvfp4-fastqsa4096bigkv-g8-sglang` — must be replaced

The published recipe pulls `lmsysorg/sglang:qwen38flashnext` and bind-mounts
`qsa_upstream_fix.py` over the engine's sparse-attention backend. That was
correct when it was uploaded: the tag then pointed at the day-0 build, whose
backend selects a kernel path that does not exist on SM121, and the mounted
file was upstream's guard fix (sgl-project/sglang#36556).

**The tag has since moved.** Since 2026-09-03 `lmsysorg/sglang:qwen38flashnext`
points at a build that ships upstream's own dedicated SM121 kernel
(sgl-project/sglang#36845). Mounting the old file over it replaces that kernel
with the earlier trtllm path, which upstream found silently corrupts context
above roughly 95k tokens on SM121: token-ID-0 output while HTTP still returns
200, one run in four at 120k, four in four at 210k (sgl-project/sglang#36806).

Anyone who pulls the published recipe today gets that combination.

Changes in the replacement:

| | published | replacement |
|---|---|---|
| image | `lmsysorg/sglang:qwen38flashnext` (mutable) | pinned by digest `sha256:5ae58167…`, the 2026-09-03 build |
| QSA patch | bind-mounted | none; the image carries upstream's kernel |
| `--allow-auto-truncate` | present | removed, so over-length requests fail instead of being silently cut |
| absolute host path | `/home/nvidia/GEN-AI/flashnext/patches/...` | gone with the mount |

Serving flags are otherwise identical, so the published performance figures
still describe it. Measured on this pair, the digest-pinned build costs about
10% of fresh shallow prefill at concurrency 1 against the day-0 build and
leaves decode unchanged; that is the price of the correct kernel.

If you would rather keep the day-0 build and the patch, pin that instead:
`lmsysorg/sglang@sha256:12d3392bdc8be8d3…`, and do not use it above ~95k
context.

## `qwen3.8-flash-next-nvfp4-tp2` — unchanged

Its image tag `vllm/vllm-openai:qwen38-flash-next` is model-specific and has
not moved, and the PLE loader patch is still required for the RadixArk
checkpoint on that build. Two things worth knowing, neither a defect in the
recipe:

- Prefix caching is enabled but does not reuse a prompt's first pass on that
  build, so every deep turn re-prefills. vLLM #53504 describes it; the fix
  merged as #53945 on 2026-09-08 and will appear in later builds.
- The mounted path `/home/nvidia/GEN-AI/qwen38-opt/patches/vllm_ple_layer.py`
  only resolves on the machine it was written on. In this repository the same
  patch ships as the mod `vllm-ple-fp8`, which needs no path.

## Fabric

Both recipes pin NCCL to this cluster's `enp1s0f1np1` and
`rocep1s0f1,roceP2p1s0f1`. Other DGX Spark pairs are often wired on `f0`.
`scripts/detect-fabric.sh` prints the right values for a given pair and can
patch them in.
