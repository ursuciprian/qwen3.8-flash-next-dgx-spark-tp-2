# Recipes

What `sparkrun recipe list` shows for this registry. Both recipes are pinned to
checkpoint revision `7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`, use the public
image `ghcr.io/ursuciprian/spark-vllm-b12x:b0-20260918-a8333658-warm`, and need
no mods, no host mounts and no `--trust`. Both are TP=2 across two DGX Sparks.

| recipe | use |
|---|---|
| [`qwen3.8-flash-next-nvfp4-tp2`](qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml) | **Default.** Probabilistic MTP drafts; best for clients that send no temperature (checkpoint default 1.0). |
| [`qwen3.8-flash-next-nvfp4-tp2-argmax-drafts`](qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-argmax-drafts.yaml) | Fallback: previous default, one-hot drafts + `use_local_argmax_reduction`. For temperature-0 clients or a rollback. |

```bash
sparkrun run qwen3.8-flash-next-nvfp4-tp2 --hosts <head-ip>,<worker-ip>
```

Every other recipe (bisection arms, rejected experiments, other checkpoints,
the SGLang-era route) is in [`archive/recipes/`](../archive/recipes/README.md);
sparkrun does not scan it. Where each file moved: [RENAMES.md](RENAMES.md).
`scripts/validate_recipes.py` fails any recipe here that lacks `--revision`,
uses an image without a registry host, or needs mods or volumes.
