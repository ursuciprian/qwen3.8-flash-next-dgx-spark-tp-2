# Recipes

What `sparkrun recipe list` shows for this registry. Both recipes are pinned
to checkpoint revision `7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`, use a public
ghcr image, and need no mods, no host mounts and no `--trust`. Both run the
model across two DGX Sparks (tensor parallel 2).

| recipe | image | use |
|---|---|---|
| [`qwen3.8-flash-next-2x-dgx-spark`](qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml) | `b1-20260925-b7fbaf96-14077fb3-warm` | **Recommended.** Current build (2026-09-25). |
| [`qwen3.8-flash-next-2x-dgx-spark-previous`](qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark-previous.yaml) | `b0-20260918-a8333658-warm` | Fallback: the previous build (2026-09-23), about 10% slower with several users. |

Renamed 2026-09-25; old names and where the one-hot draft recipe went: [RENAMES.md](RENAMES.md).

```bash
sparkrun run qwen3.8-flash-next-2x-dgx-spark --hosts <head-ip>,<worker-ip>
```

Every other recipe (bisection arms, rejected experiments, other checkpoints,
the SGLang-era route) is in [`archive/recipes/`](../archive/recipes/README.md);
sparkrun does not scan it. Where each file moved: [RENAMES.md](RENAMES.md).
`scripts/validate_recipes.py` fails any recipe here that lacks `--revision`,
uses an image without a registry host, or needs mods or volumes.
