# Qwen3.8-Flash-Next NVFP4 on two DGX Sparks

Three serving recipes for `Qwen3.8-Flash-Next-NVFP4` (180B MoE) at tensor
parallel 2 across two NVIDIA DGX Spark (GB10, SM121) over ConnectX-7, for
[sparkrun](https://sparkrun.dev). Two on SGLang, one on vLLM. Neither engine
supports this hardware out of the box; these boot, and every number below comes
from a fresh boot measured with the same grid.

## Which recipe

| You are serving | Recipe | Engine | What you get |
|---|---|---|---|
| Chat, one or two agents, cached history | `flashnext-bigkv-g8-c4096` | SGLang | Best single-stream decode: 37-41 tok/s on prose, 58-61 tok/s median across a 40-prompt category mix, 63 on coding. 110k context, 0.9M-token KV pool. |
| Five or more concurrent streams, batch prefill | `flashnext-bigkv-nospec` | SGLang | Same recipe without the drafter: prefill +20%, decode 75-85 tok/s at five streams. Single stream drops to 26, so do not use it below five. |
| Long documents reused across turns, many agents, capacity | `flashnext-vllm-cached` | vLLM | Prefix caching that actually reuses a prompt's first pass: a fresh 2k turn after a cached 16k context prefills at 2176 tok/s against SGLang's 946. 262k context, 2.0M-token KV pool. |

Full grids in [results/RESULTS.md](results/RESULTS.md).

Decode on this model depends more on how predictable the output is than on the
engine or the flags: the same server decodes prose at 38 tok/s and JSON at 62.
Measure your own workload before trusting one figure.

## Use

```sh
sparkrun registry add https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2
sparkrun run @qwen38-flashnext/flashnext-bigkv-g8-c4096 --cluster <your-cluster> --tp 2
sparkrun run @qwen38-flashnext/flashnext-vllm-cached    --cluster <your-cluster> --tp 2 --trust
```

The vLLM recipe needs six engine overlays; they ship as the mod
`vllm-flashnext-nightly-8a728663`, which sparkrun copies into every container
before the serve command, so nothing depends on a clone path. `--trust` skips
the confirmation prompt for that hook. The SGLang recipes need no patch.

Or clone and use the launcher, which also checks the cluster, mirrors the
checkpoint over the fast link and drops the page cache on both nodes:

```sh
git clone https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2
cd qwen3.8-flash-next-dgx-spark-tp-2
scripts/run.sh sglang --check     # validate, launch nothing
scripts/run.sh sglang             # flashnext-bigkv-g8-c4096
scripts/run.sh sglang-nospec
scripts/run.sh vllm               # flashnext-vllm-cached
DEPTHS="0 16384" CONCURRENCY="1 2 5" scripts/run.sh vllm --bench
```

Loads take 10-15 minutes. Stop with `sparkrun stop --all`.

### Before the first launch on your hardware

The recipes pin NCCL to this cluster's fabric (`enp1s0f1np1`,
`rocep1s0f1,roceP2p1s0f1`). Many DGX Spark pairs are wired on `f0`. sparkrun's
`-o` overrides do not reach `env:`, so the YAML has to be edited:

```sh
WORKER_IP=<worker ip on the fast link> scripts/detect-fabric.sh          # print
WORKER_IP=<worker ip on the fast link> scripts/detect-fabric.sh --write  # patch
```

Left wrong, NCCL falls back to TCP over the management link and adds 18-65 ms
of jitter per decode step. Also check the `taskset -c 5-9,15-19` list in the
SGLang recipes; it pins the server to this machine's fast cores.

On unified memory a warm page cache starves the GPU allocator about 20 minutes
into the load, and `free -g` misreports it, so drop it on both nodes first:

```sh
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

`scripts/validate_recipes.py` checks a recipe against the failures this project
hit: moving image tags, missing overlays, unsupported flags, unset fabric,
credentials in `env`.

## What is in the recipes

**SGLang** (`flashnext-bigkv-g8-c4096`, and `-nospec` without the drafter).
The 2026-09-03 build, pinned by digest, which carries upstream's own SM121
sparse-attention kernel (sgl-project/sglang#36845), so nothing is patched.
NEXTN speculative decoding with 3 steps and 4 draft tokens, BF16 Mamba state,
chunked prefill 4096, decode CUDA graphs for every batch size 1 to 10, KV pool
raised to 900k tokens, 97 Mamba slots. The n-gram table stays GPU-resident
(`--no-ple-offload-embedding`): on unified memory the "offload" is a pinned
host copy from the same 128 GB pool, so it frees nothing and costs a gather per
decode step. Removing it measured above a three-boot control envelope on 8 of
36 grid cells with none below, repeated on a second boot.

**vLLM** (`flashnext-vllm-cached`). NVIDIA's checkpoint at revision `fc694b54`
on nightly `8a728663`, pinned by digest. MTP with 3 draft tokens, fp8 KV,
expert parallelism, six overlays, and one speculative-config flag,
`disable_eagle_block_drop`.

That flag exists because of a real defect. vLLM's Mamba "align" caching keeps
only the GDN state at the prompt's last block boundary, while the MTP block
drop looks one block earlier, so a prompt's first pass is never reusable and
every deep turn re-prefills the whole context: a fresh 2k prefill at 32k depth
runs at 161 tok/s instead of 1862. Disabling the drop makes the lookup land
(the same 20k prompt sent three times reuses 0, then 19,200, then 19,200
tokens; the second request drops from 10 s to 1.7 s).

The drop guards against reusing KV written under the next-token lookahead. On
six prompts of 6k-18k tokens, cached and fresh greedy outputs diverged at the
same rate with the drop kept and with it disabled, and the same two near-tied
continuations swapped sides, so that divergence is batch-shape numerics rather
than the flag. A 68-request multi-turn correctness gate passes either way. This
is evidence, not proof.

Upstream has since fixed the underlying defect: vllm-project/vllm#53504 is the
report, #53945 merged on 2026-09-08. On a build containing it the flag should
be unnecessary; this repository will drop it once that is measured here.

## Known issue with the published Spark Arena entries

If you came from Spark Arena, read this before running the published SGLang
recipe. It pulls `lmsysorg/sglang:qwen38flashnext` and bind-mounts a patched
sparse-attention backend. That was correct when uploaded: the tag pointed at
the day-0 build, whose backend selects a kernel path that does not exist on
SM121, and the mounted file was upstream's guard fix (sgl-project/sglang#36556).

**The tag has since moved.** Since 2026-09-03 it points at a build that ships
upstream's own SM121 kernel (#36845). Mounting the old file over it restores
the earlier path, which upstream found silently corrupts context above roughly
95k tokens on SM121: token-ID-0 output while HTTP still returns 200
(sgl-project/sglang#36806). Use `flashnext-bigkv-g8-c4096` here instead; it is
the same tuning on the correct build, pinned by digest.

The published vLLM entry is sound. Note only that its prefix caching does not
reuse a first pass on that build, for the reason above, and that its patch path
is absolute.

## Hardware

- 2x DGX Spark: GB10, SM121, 128 GB LPDDR5X unified, about 273 GB/s per node.
- ConnectX-7 RoCE between the nodes, GID index 3.
- Checkpoints: `RadixArk/Qwen3.8-Flash-Next-NVFP4` at `7b719225` (SGLang) and
  `nvidia/Qwen3.8-Flash-Next-NVFP4` at `fc694b54` (vLLM). About 135 GB on disk,
  73 GB of weights per node at TP=2.

## Layout

| Path | What |
|---|---|
| `recipes/` | The three recipes |
| `mods/` | The vLLM engine overlays, with a README explaining each and why |
| `scripts/` | `run.sh`, `detect-fabric.sh`, `validate_recipes.py`, `recipe_metadata.py` |
| `results/` | `RESULTS.md` and the grids behind it |
| `.sparkrun/registry.yaml` | Registry manifest |

## Credits

RadixArk for the NVFP4 checkpoint and the day-0 engine work. tonyd2wild for the
vLLM SM121 overlays, the profile the vLLM recipe grew from, and the 40-prompt
category harness used for the coding, JSON and prose figures. MiaAI for the
fast sparse-attention SGLang profile the SGLang recipes grew from. Their work
made this run faster on my hardware and my workloads; the measurements, the
cache-reuse diagnosis and the fixes here are mine, and so are any mistakes.

Apache-2.0. The engine overlays are modified copies of vLLM files and keep
their upstream headers.
