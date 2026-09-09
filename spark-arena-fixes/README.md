# Spark Arena submissions: what is broken and how to fix it

Five entries are published. **None of them runs on anyone else's hardware**, and
the four SGLang ones are now unsafe as well. This folder holds a corrected YAML
per submission, ready to paste into the admin console, and the reasoning.

| Submission | Runtime | Copies | Status | Replacement |
|---|---|---|---|---|
| `sub1788106678543` | SGLang | 27 | broken + unsafe | `sub1788106678543-sglang.yaml` |
| `sub1788030960415` | SGLang | 0 | broken + unsafe | `sub1788030960415-sglang.yaml` |
| `sub1788030837243` | SGLang | 0 | broken + unsafe | `sub1788030837243-sglang.yaml` |
| `sub1788030743002` | SGLang | 0 | broken + unsafe | `sub1788030743002-sglang.yaml` |
| `sub1787862385932` | vLLM | 84 | broken | `sub1787862385932-vllm.yaml` |

## Fault 1: every entry mounts a path from the author's machine

All five carry an `executor_config.volumes` entry pointing at an absolute path
that exists only on the machine they were written on:

```
/home/nvidia/GEN-AI/flashnext/patches/qsa_upstream_fix.py      # the four SGLang entries
/home/nvidia/GEN-AI/qwen38-opt/patches/vllm_ple_layer.py       # the vLLM entry
```

On any other node that file does not exist, so Docker creates the path as an
empty directory and the container dies at creation:

```
error mounting "..." to rootfs at "...": not a directory:
Are you trying to mount a directory onto a file (or vice-versa)?
```

There is no way for a user to recover from this except by editing the recipe.
With 84 and 27 copies taken, this has been failing for people.

## Fault 2: the SGLang image tag moved under its patch

The SGLang entries pull `lmsysorg/sglang:qwen38flashnext` and mount a patched
sparse-attention backend over it. That was correct when they were published: the
tag pointed at the day-0 build, whose backend selects a kernel path that does not
exist on GB10, and the mounted file was upstream's guard fix
(sgl-project/sglang#36556).

Since 2026-09-03 that tag points at a build shipping upstream's own SM121 kernel
(sgl-project/sglang#36845). Mounting the old file over it puts the earlier path
back, and upstream found that path silently corrupts context above roughly 95k
tokens on SM121: token-ID-0 output while HTTP still returns 200
(sgl-project/sglang#36806). So a user who did supply the patch file would get a
server that looks healthy and quietly damages long conversations.

## The fix, per runtime

**SGLang, all four entries.** No patch is needed any more. Pin the image by
digest to the 2026-09-03 build and delete the `volumes:` block. Serving flags are
unchanged, so the published figures still describe the recipe.

```yaml
container: lmsysorg/sglang@sha256:5ae5816783d58e2e56e84d2e863f5441425056f500b7fbd7448c4aae017a2521
# executor_config.volumes removed entirely
```

If you would rather keep the day-0 build and its patch, pin
`lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1`
instead and use the `sglang-sm121-qsa-guard` mod, but do not run that above ~95k
context.

**vLLM.** The PLE loader patch is still required: the RadixArk checkpoint's PLE
shards are global-scale FP8 and the checkpoint lists `*.ple.*` under `ignore`, so
the stock gate never selects the FP8 embedding path and weight loading fails on
`ngram_embedding.weight_scale`. The patch now ships as a mod, so the recipe
references it instead of mounting a path:

```yaml
mods:
  - '@qwen38-flashnext/vllm-ple-fp8'
```

That reference resolves once the user has added the registry, which is one
command and should go in the entry's description:

```sh
sparkrun registry add https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2
```

The image is also pinned by digest
(`vllm/vllm-openai@sha256:fc120ece…`), so the build cannot move underneath it the
way the SGLang tag did.

## Still the user's job

Both replacements pin NCCL to this cluster's `enp1s0f1np1` and
`rocep1s0f1,roceP2p1s0f1`. Many DGX Spark pairs are wired on `f0`, and sparkrun's
`-o` overrides do not reach `env:`, so the YAML has to be edited.
`scripts/detect-fabric.sh` in this repository prints the right values and can
patch them in. Worth saying in the entry description; left wrong, NCCL falls back
to TCP over the management link.

## Two things the replacements deliberately do not change

The SGLang entries keep `--allow-auto-truncate` and decode graphs `1 2 4 8`, and
the vLLM entry keeps its 3 draft tokens and capture sizes `[1,2,3,4]`. Those are
tuning choices the published numbers were measured with. The current recipes in
`recipes/` improve on all of them, but that is a different recipe, not a fix to
these. Point users at `recipes/` from the entry description if you want them on
the faster configuration.

One known limitation of the vLLM entry, unchanged because it is upstream's, not
the recipe's: prefix caching is enabled but does not reuse a prompt's first pass
on that build, so deep turns re-prefill. vLLM #53504 is the report and #53945 the
fix; later builds have it.

## Verifying before you paste

```sh
scripts/validate_recipes.py spark-arena-fixes
```

Clean except two expected warnings: the registry-scoped mod, which cannot be
resolved without the registry, and the first-pass caching note above.
