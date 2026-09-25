# Recipes

What `sparkrun recipe list` shows for this registry. All three recipes are
pinned to checkpoint revision `7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`, use a
public ghcr image, and need no mods, no host mounts and no `--trust`. All are
TP=2 across two DGX Sparks.

| recipe | image | use |
|---|---|---|
| [`qwen3.8-flash-next-nvfp4-tp2`](qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2.yaml) | `b1-20260925-b7fbaf96-14077fb3-warm` | **Default** (2026-09-25). Probabilistic MTP drafts, deferred GDN checkpoints, TC-45 `tool_choice` fix. |
| [`qwen3.8-flash-next-nvfp4-tp2-b0`](qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-b0.yaml) | `b0-20260918-a8333658-warm` | Fallback: the default from 2026-09-22 to 2026-09-25. Roll back here. |
| [`qwen3.8-flash-next-nvfp4-tp2-argmax-drafts`](qwen3.8-flash-next/qwen3.8-flash-next-nvfp4-tp2-argmax-drafts.yaml) | `b0-20260918-a8333658-warm` | Fallback: one-hot drafts + `use_local_argmax_reduction`, for temperature-0 clients. |

```bash
sparkrun run qwen3.8-flash-next-nvfp4-tp2 --hosts <head-ip>,<worker-ip>
```

Every other recipe (bisection arms, rejected experiments, other checkpoints,
the SGLang-era route) is in [`archive/recipes/`](../archive/recipes/README.md);
sparkrun does not scan it. Where each file moved: [RENAMES.md](RENAMES.md).
`scripts/validate_recipes.py` fails any recipe here that lacks `--revision`,
uses an image without a registry host, or needs mods or volumes.
