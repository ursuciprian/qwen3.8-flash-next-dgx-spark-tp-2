# Qwen3.8-Flash-Next NVFP4 on two DGX Sparks (TP=2)

Recipes, patches and measurements for serving `Qwen3.8-Flash-Next-NVFP4`
(180B MoE) across two NVIDIA DGX Spark (GB10, SM121) over ConnectX-7, with
sparkrun, on SGLang and on vLLM. Neither engine supports this hardware out of
the box; the recipes here boot, and the numbers behind them are in
[results/RESULTS.md](results/RESULTS.md).

Two sets of recipes:

- **`recipes/sparkarena/`**: the two recipes published on Spark Arena, same
  flags as uploaded, patches applied as mods.
- **`recipes/sparkarena-fixed/`**: replacements for those two. The published
  SGLang recipe pins a tag that has since moved, so its bind-mounted SM121
  patch now overwrites the newer image's correct kernel and can corrupt long
  context. Read `recipes/sparkarena-fixed/README.md` before running the
  published copy.
- **`recipes/latest/`**: the three recipes I run today. Faster, on newer images,
  and with a correctness fix the published SGLang recipe does not have.

## Which recipe

| You are serving | Recipe | Engine | What you get |
|---|---|---|---|
| Chat, one or two agents, cached history | `latest/flashnext-bigkv-g8-c4096.yaml` | SGLang | Best single-stream decode: 37-41 tok/s on prose, 58-61 tok/s median on a 40-prompt category harness (coding 63). 110k context, 0.9M-token KV pool. |
| Five or more streams, batch prefill | `latest/flashnext-bigkv-nospec.yaml` | SGLang | Same recipe without the drafter: prefill +20%, c5 decode 75-85 tok/s at depth. Single stream drops to 26; use at c5 and above. |
| Long documents revisited across turns, many agents, capacity | `latest/flashnext-vllm-cached.yaml` | vLLM | Prefix caching that really reuses, drafter on: a fresh 2k turn after a cached 16k context prefills at 2176 tok/s (SGLang 946). Decode 43 tok/s single stream, 52-61 at c5 and depth. 262k context, 2.0M-token KV pool. |
| Reproduce the Spark Arena entries | `sparkarena/*.yaml` | both | The published configurations; patches as mods. |

Decode on this model depends more on how predictable the output is than on
engine or flags: the same server decodes prose at 38 tok/s and JSON at 62.
Measure your own workload before trusting one figure.

## Use as a sparkrun registry

This repository is a sparkrun recipe registry (`.sparkrun/registry.yaml`).
The overlays ship as mods, so nothing needs a path edit:

```sh
sparkrun registry add https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2
sparkrun recipe search flashnext
sparkrun run @qwen38-flashnext/flashnext-bigkv-g8-c4096 --cluster <your-cluster> --tp 2
sparkrun run @qwen38-flashnext/flashnext-vllm-cached     --cluster <your-cluster> --tp 2
```

The vLLM recipe declares `mods: [vllm-flashnext-nightly-8a728663]`; the mod's
`run.sh` copies the six overlays over the image's files inside every container
before serve. sparkrun will ask you to confirm the hook the first time unless
you pass `--trust`. The Spark Arena copies sit in a second, hidden registry
entry (`@qwen38-flashnext-sparkarena/...`). In the registry they carry their
patches as mods too (`sglang-sm121-qsa-guard`, `vllm-ple-fp8`) instead of the
absolute bind-mount lines of the uploaded files; flags and everything else are
unchanged.

**Before the first launch on different hardware**, fix the fabric names. The
recipes carry this cluster's (`enp1s0f1np1`, `rocep1s0f1,roceP2p1s0f1`); many
DGX Spark pairs are wired on `f0`. sparkrun's `-o` overrides do not reach
`env:`, so the YAML has to be edited:

```sh
WORKER_IP=<worker ip on the fast link> scripts/detect-fabric.sh          # print
WORKER_IP=<worker ip on the fast link> scripts/detect-fabric.sh --write  # patch
```

Also check the `taskset -c 5-9,15-19` CPU list in the SGLang recipes; it pins
the server to this machine's fast cores. Images are pinned by digest, so a
moved tag cannot silently change the build under you.

## Quick start

On the head node, with a two-host sparkrun cluster on the CX-7 addresses:

```sh
git clone https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2 ~/GEN-AI/qwen3.8-flash-next
cd ~/GEN-AI/qwen3.8-flash-next

scripts/run.sh sglang --check       # validate cluster and nodes, launch nothing
scripts/run.sh sglang               # latest SGLang, interactive
scripts/run.sh sglang-nospec        # latest SGLang, 5+ streams
scripts/run.sh vllm                 # latest vLLM, cached long context
scripts/run.sh sglang-sparkarena    # published SGLang recipe
scripts/run.sh vllm-sparkarena      # published vLLM recipe

DEPTHS="0 16384" CONCURRENCY="1 2 5" scripts/run.sh vllm --bench
```

`run.sh` checks both nodes, syncs the checkout to the worker, fetches the checkpoint and mirrors
it over CX-7, drops the page cache on both nodes, launches, and waits for the
port. Loads take 10-15 minutes. Stop with `sparkrun stop --all`. Launch by hand
with `sparkrun run <recipe> --cluster <name> --tp 2` after editing the mount
paths in the recipe.

Before every launch, on both nodes:

```sh
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

GB10 shares one 128 GB pool between CPU and GPU. A warm page cache starves
the GPU allocator about 20 minutes into the load; `free -g` misreports it.

## The recipes, briefly

**Published SGLang** (`qwen38-flash-next-nvfp4-fastqsa4096bigkv-g8-sglang`,
2026-08-30): day-0 image with an SM121 guard fix bind-mounted, NEXTN
speculative decoding (3 steps, 4 draft tokens), BF16 Mamba state, chunked
prefill 4096, decode CUDA graphs at batch 1/2/4/8, KV pool enlarged to 900k
tokens. The only configuration in the August grid that kept five streams alive
at 65k depth. Known issue: on SM121 the guard-fixed kernel path silently
corrupts contexts above roughly 95k tokens (sgl-project/sglang #36806), and
the image tag is not pinned. Use the latest recipe.

**Latest SGLang** (`flashnext-bigkv-g8-c4096`): same tuning on the
2026-09-03 image, which carries upstream's dedicated SM121 kernel
(sgl-project/sglang #36845), so nothing is mounted. Decode graphs cover every
batch 1 to 10. `--allow-auto-truncate` removed: over-length requests fail
explicitly. `--no-ple-offload-embedding` since 2026-09-09: on unified memory
the n-gram offload is a pinned host copy out of the same pool, so keeping the
table GPU-resident and sharded is free speed (8 of 36 grid cells above a
three-boot control envelope, none below, repeated on a second boot). Fresh shallow prefill about 10% lower than the published recipe,
decode unchanged, long context correct.

**Latest SGLang without drafter** (`flashnext-bigkv-nospec`): the same file
minus the five speculative flags. The drafter accepts about 2 of 4 tokens on
prose; at five or more streams verification costs more than it returns.

**Published vLLM** (`qwen3.8-flash-next-nvfp4-tp2`): RadixArk checkpoint on
`vllm/vllm-openai:qwen38-flash-next`, MTP with 3 draft tokens, compilation
mode 0 (Inductor compile on this model consumed 40 GB outside the memory
budget and locked both nodes; CUDA graph capture itself is fine), decode
graphs at batch 1-4, patched PLE loader. Single-stream decode nearly flat to
64k context. Prefix caching was on but not reusing anything across turns on
that build, so deep turns re-prefilled the whole context.

**Latest vLLM** (`flashnext-vllm-cached`): NVIDIA checkpoint
`nvidia/Qwen3.8-Flash-Next-NVFP4` at `fc694b54` on nightly `8a728663`, MTP
with 3 draft tokens, fp8 KV, six overlays (see
[patches/README.md](patches/README.md)), and one speculative-config flag,
`disable_eagle_block_drop`. Why: vLLM's Mamba "align" caching keeps only the
GDN state at the prompt's last block boundary, and the MTP block drop looks one
block earlier, so a prompt's first pass is never reusable. Disabling the drop
makes the lookup land (same 20k prompt three times: 0 / 19,200 / 19,200
tokens reused, second request 1.7 s instead of 10 s). The drop guards against
reusing KV written under the next-token lookahead; on six 6k-18k prompts,
cached and fresh greedy outputs diverged at the same rate with and without it,
with the same two near-tied continuations swapping sides, so that divergence
is batch-shape numerics rather than the flag. The stock-semantics alternative,
`--prefix-cache-retention-interval 3200`, keeps the drop and reuses one block
less per turn, halving cached prefill. Written up for upstream; the drafter
group annotation is vllm-project/vllm #55390.

## Hardware

- 2x DGX Spark: GB10, SM121, 128 GB LPDDR5X unified, about 273 GB/s per node.
- CX-7 RoCE between the nodes on `192.168.100.x` (interface `enp1s0f1np1`,
  HCAs `rocep1s0f1,roceP2p1s0f1`, GID index 3). The recipes pin NCCL to it;
  on defaults NCCL falls back to TCP over WiFi and adds 18-65 ms per decode
  step. Adjust the interface and HCA names to your nodes.
- CPUs 5-9 and 15-19 run at 3900 MHz, the rest at 2808; recipes pin the
  server to the fast ones with `taskset`.
- Checkpoints: `RadixArk/Qwen3.8-Flash-Next-NVFP4` at `7b719225` (SGLang and
  published vLLM), `nvidia/Qwen3.8-Flash-Next-NVFP4` at `fc694b54` (latest
  vLLM). About 135 GB each on disk, 73 GB of weights per node at TP=2.

## Layout

| Path | What |
|---|---|
| `recipes/sparkarena/` | The two published recipes, verbatim |
| `recipes/latest/` | The three maintained recipes |
| `patches/` | The overlay files with a README explaining each |
| `mods/` | The same overlays as sparkrun mods (`run.sh` copies them into the container); referenced by the registry recipes |
| `.sparkrun/registry.yaml` | Registry manifest: `qwen38-flashnext` (recipes/latest) and the hidden `qwen38-flashnext-sparkarena` |
| `scripts/run.sh` | One launcher, one option per recipe |
| `scripts/recipe_metadata.py` | Reads recipes, rewrites mount paths to this checkout |
| `scripts/validate_recipes.py` | Checks recipes against the failures this project hit (moving tags, missing overlays, unsupported flags, fabric, secrets) |
| `results/` | Grids as CSV and `RESULTS.md` with the tables |

## Credits

RadixArk for the NVFP4 checkpoint and the day-0 engine work. tonyd2wild for
the vLLM SM121 overlays, the SPEED profile the latest vLLM recipe grew from,
and the 40-prompt category harness used for the coding/JSON/prose figures.
MiaAI for the fast sparse-attention SGLang profile the SGLang recipes grew
from. Their work made this run faster on my setup and my workloads; the
measurements, the cache-reuse diagnosis and the fixes here are mine, and so
are any mistakes.
