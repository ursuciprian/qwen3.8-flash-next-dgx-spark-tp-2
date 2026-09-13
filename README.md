# Qwen3.8-Flash-Next NVFP4 on two DGX Sparks

Serving recipes for `Qwen3.8-Flash-Next-NVFP4` (180B MoE) at tensor parallel 2
across two NVIDIA DGX Spark (GB10, SM121) over ConnectX-7, packaged for
[sparkrun](https://sparkrun.dev). Three recipes on SGLang, one on vLLM. Neither
engine serves this model on this hardware out of the box; these boot, and every
number here comes from a fresh boot with the page cache dropped, measured with
the same harnesses.

## Quick start

```sh
sparkrun registry add https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2
sparkrun run @qwen38-flashnext/flashnext-fp8kv-1m8      --cluster <your-cluster> --tp 2 --trust
sparkrun run @qwen38-flashnext/flashnext-vllm-cached    --cluster <your-cluster> --tp 2 --trust
```

`--trust` accepts the mod hook that patches the engine inside the container
(see [Why the mods](#why-the-mods)). Loads take 10-15 minutes warm, 20-30 cold.
Stop with `sparkrun stop --all`. The server answers on port 8000 with the
OpenAI API, model name `qwen3.8-flash-next`, thinking off by default.

Or clone and use the launcher, which checks the cluster, mirrors the checkpoint
over the fast link and drops the page cache on both nodes:

```sh
git clone https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2
cd qwen3.8-flash-next-dgx-spark-tp-2
scripts/run.sh sglang --check     # validate, launch nothing
scripts/run.sh sglang             # flashnext-fp8kv-1m8, the default
scripts/run.sh sglang-bf16        # flashnext-bigkv-g8-c4096
scripts/run.sh sglang-nospec      # flashnext-bigkv-nospec
scripts/run.sh vllm               # flashnext-vllm-cached
DEPTHS="0 16384" CONCURRENCY="1 2 5" scripts/run.sh vllm --bench
```

Before the first launch on your own pair, read [Gotchas](#gotchas): the fabric
interface names are pinned to this cluster.

### Which recipe

| You are serving | Recipe | Engine | What you get |
|---|---|---|---|
| Chat, one or two agents, cached history | `flashnext-fp8kv-1m8` | SGLang | Best single-stream decode, 262k context, 1.8M-token KV pool in fp8. Quality gate against bf16 passed (TrueScore 81.3 vs 80.8, needle exact to 250k). |
| Same, without a mod | `flashnext-bigkv-g8-c4096` | SGLang | Identical flags with bf16 KV on the 2026-09-03 digest: 110k context, 0.9M-token pool. |
| Five or more streams, batch prefill | `flashnext-bigkv-nospec` | SGLang | No drafter: prefill +20%, decode 75-85 tok/s at five streams. Single stream drops to 26, so not below five. |
| Long documents reused across turns, many agents | `flashnext-vllm-cached` | vLLM | Prefix caching that reuses a prompt's first pass: a fresh 2k turn after a cached 16k context prefills at 2176 tok/s against SGLang's 946. 262k context, 2.0M-token pool. |

## How this works

A sparkrun recipe is one YAML: the container digest, the checkpoint revision,
the environment, and the serve command. sparkrun starts the same container on
both nodes, wires NCCL over the ConnectX-7 link, and runs the head's serve
command with `--tp 2`. Everything the engine needs beyond its stock image comes
from a **mod**: a directory under `mods/` that sparkrun copies into each
container and runs before the serve command. Mods here are small, idempotent
and fail closed; if a patch does not apply exactly, the boot stops rather than
serving a half-patched engine.

**SGLang recipes.** The RadixArk NVFP4 checkpoint with the NEXTN drafter (3
steps, 4 draft tokens), BF16 Mamba state, chunked prefill 4096, decode CUDA
graphs for batch sizes 1 to 10, 97 Mamba slots. The n-gram (PLE) table stays
GPU-resident: on unified memory the "offload" is a pinned host copy from the
same 128 GB pool, so it frees nothing and costs a gather per decode step.
`flashnext-fp8kv-1m8` adds `--kv-cache-dtype fp8_e4m3` on the 2026-09-11
nightly plus the `sglang-sm121-qsa-fp8kv` mod, which doubles the KV pool and
lifts context from 110k to 262k at no measured quality cost.

**vLLM recipe.** NVIDIA's checkpoint on nightly `8a728663`, pinned by digest.
MTP with 3 draft tokens, fp8 KV, expert parallelism, six engine overlays, and
`disable_eagle_block_drop` in the speculative config. That flag is what makes
prefix caching reuse a first pass: vLLM's Mamba "align" caching keeps only the
state at the prompt's last block boundary while the MTP block drop looks one
block earlier, so without it every deep turn re-prefills the whole context (a
fresh 2k prefill at 32k depth runs at 161 tok/s instead of 1862). Cached and
fresh greedy outputs were checked on six long prompts and a 68-request
multi-turn gate with the drop kept and disabled; both diverge at the same rate
from batch-shape numerics, so the flag is not the cause. Upstream fixed the
underlying defect in vllm-project/vllm#53945 (2026-09-08); the flag goes once a
build carrying it is measured here.

**Measurement.** Every recipe is measured with three independent harnesses:
tonyd2wild's 40-prompt category harness (concurrency 1 to 6, cold prefill
ladder to 88k with a needle), MiaAI-Lab's llama-benchy spec, and spark-bench's
76 graded scenarios. Four boots of the same recipe spread about 5% at one
stream, so nothing below that is reported as a result. Full tables in
[results/RESULTS.md](results/RESULTS.md).

## Why the mods

Three mods, each explained in [mods/README.md](mods/README.md).

- **`sglang-sm121-qsa-fp8kv`** (used by `flashnext-fp8kv-1m8`). The only sparse
  attention decode kernel qualified for SM121 accepts BF16 keys only, so with an
  fp8 KV cache the scheduler dies on the first decode with `unsupported SM121
  QSA call: expected BF16 D=256 ...`. The mod allocates the packed gather
  scratch in the query dtype at the two call sites, so the selected keys convert
  on store and the cache itself stays fp8. Valid while the KV scales are 1.0,
  which is the default without a calibration file.
- **`vllm-flashnext-nightly-8a728663`** (used by `flashnext-vllm-cached`). Six
  whole-file overlays: community SM121 fixes for the sparse-attention and PLE
  ops and the platform gate (which also carry fp8 KV on this build), the
  modelopt loader fixes NVIDIA's MTP head needs, and the KV-cache utility change
  that lets vLLM identify the drafter's cache group.
  Whole-file overlays are bound to one build and crash on a newer one, which is
  why the image is pinned by digest.
- **`vllm-qsa-fp8kv-pr55557`** (not in a shipped recipe yet). Upstream's own fp8
  KV patch, vllm-project/vllm#55557, applied as a diff so current vLLM nightlies
  accept fp8 without the overlays. Measured at 2.07M KV tokens and decode parity;
  one tool-eval run scored 90 against the pinned build's 100, so it stays out of
  the recipe until that repeats clean.

## Gotchas

- **Fabric names are pinned.** The recipes set `enp1s0f1np1` and
  `rocep1s0f1,roceP2p1s0f1` in `env:`. Many pairs are wired on `f0`. sparkrun's
  `-o` overrides do not reach `env:`, so patch the YAML:
  ```sh
  WORKER_IP=<worker ip on the fast link> scripts/detect-fabric.sh --write
  ```
  Left wrong, NCCL falls back to TCP over the management link and adds 18-65 ms
  of jitter per decode step. Check the `taskset -c 5-9,15-19` list in the SGLang
  recipes too; it pins the server to this machine's fast cores.
- **Drop the page cache before loading.** On unified memory a warm page cache
  starves the GPU allocator about 20 minutes into the load, and `free -g`
  misreports it. `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` on both
  nodes. `scripts/run.sh` does this for you.
- **Cold boots are slow.** 20-30 minutes from a dropped page cache or a fresh
  image, which also pays cold Triton and torch JIT. sparkrun's 120 s readiness
  window is not the limit here; the recipes set the engine's own timeout.
- **The SGLang server log is inside the container** at `/tmp/sparkrun_serve.log`.
  `docker logs` shows only the CUDA banner.
- **Worker rendezvous race.** Roughly one vLLM boot in five on this pair dies
  with `FileNotFoundError` on a worker semaphore before the engine starts.
  Relaunching succeeds; it is not recipe-dependent.
- **Mods resolve beside the recipe directory.** If you copy a recipe out of
  `recipes/`, keep `recipes/mods -> ../mods` next to it or sparkrun reports
  `Could not resolve mod`.
- **llama-benchy's `--extra-body` takes `key=value` pairs.** A JSON object is
  accepted silently and sets nothing. SGLang needs `return_token_ids=false`
  whenever the request streams.
- **Decode speed follows the text, not the flags.** The same server decodes
  prose at 38 tok/s and JSON at 62. Measure your own workload.
- `scripts/validate_recipes.py recipes` checks a recipe against the failures
  this project hit: moving image tags, missing mods, unsupported flags, unset
  fabric, credentials in `env`.

## Not working properly

- **vLLM TTFT under load.** 3.50 s at six streams against SGLang's 0.40 s on
  the same harness, same chunk size and seat count. `--long-prefill-token-threshold`
  does not help (TTFT flat, decode down 15%). Being bisected; see RESULTS.md.
- **vLLM cached turns at depth prefill at a third of cold speed** (480 tok/s vs
  1480, flat from 4k to 32k). Both sparse-attention kernels are cleared as the
  cause by microbenchmark; the remaining 2.9 s per chunk is being profiled.
- **fp8 KV on the vLLM nightly is one patch away.** Stock nightlies refuse it
  (`Qwen4Exp QSA requires a BF16 main KV cache`); the `pr55557` mod fixes that
  but its quality repeat is pending.
- **The two Spark Arena entries for this model.** The published SGLang entry
  pulls a moving tag and mounts a day-0 backend file over a build that no longer
  needs it, which restores a kernel path upstream found silently corrupts
  context above roughly 95k tokens on SM121 (sgl-project/sglang#36806). Use the
  recipes here; same tuning, correct build, pinned by digest. The published vLLM
  entry is sound but its prefix caching does not reuse a first pass and its
  patch path is absolute.
- **Not affected:** sgl-project/sglang#37111 (silent corruption with QSA, NEXTN
  and decode graphs on GB10 TP2) does not reproduce on the shipped digest.
  `scripts/gate_37111.py` runs both reported cases against a live server.

## Models tested

One model, two checkpoints, two engines.

| | SGLang `flashnext-fp8kv-1m8` | vLLM `flashnext-vllm-cached` |
|---|---|---|
| checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ `7b719225` | `nvidia/Qwen3.8-Flash-Next-NVFP4` @ `fc694b54` |
| build | `lmsysorg/sglang` 2026-09-11 nightly, digest pinned | `vllm/vllm-openai` nightly `8a728663`, digest pinned |
| KV cache | fp8_e4m3, 1,800,000 tokens | fp8_e4m3, 2,048,795 tokens |
| context | 262,144 | 262,144 |
| decode, 1 stream (40-prompt median) | **67.0 tok/s** | 65.2 |
| decode, 6 streams aggregate | **314 tok/s** | 296 |
| TTFT at 6 streams | **0.40 s** | 3.50 s |
| cold prefill, 7k / 88k tokens | 2250 / **4100 tok/s** | 2770 / 3440 |
| cached 2k turn after 16k context | 915 tok/s | **2282 tok/s** |
| decode at 32k depth, 5 streams | **79 tok/s** | 62 |
| tool-eval, 15 core scenarios | 97 | **100** |
| spark-bench TrueScore, 76 x2 | 81.3 | pending |
| needle retrieval | 21/21 to 250k | 88k rung, every size |
| load time, warm page cache | 12 min | 15 min |

SGLang wins one-shot work: faster decode at every concurrency, TTFT under load
an order of magnitude lower, faster cold prefill. vLLM wins wherever a large
context is reused across turns, because every cached turn prefills about twice
as fast. Both recipes ran the same prompts on the same afternoon from fresh
boots. The category and llama-benchy grids, and the fp8 quality gate, are in
[results/RESULTS.md](results/RESULTS.md).

Against the published dual-Spark references on the same harnesses (their
numbers, our runs of their harness on our recipes): tonyd2wild's TP2 profile
53.7 tok/s at one stream and 97.9 at six against 67.0 and 314; MiaAI-Lab's
llama-benchy T1 24.1 tok/s at one stream against 36.7 on our vLLM recipe
(their cluster caps the GPU clock at 2200 MHz).

## Hardware

- 2x DGX Spark: GB10, SM121, 128 GB LPDDR5X unified, about 273 GB/s per node.
- ConnectX-7 RoCE between the nodes, GID index 3.
- About 135 GB of checkpoints on disk, 73 GB of weights per node at TP=2.

## Layout

| Path | What |
|---|---|
| `recipes/` | The four recipes; `recipes/mods` links to `mods/` |
| `mods/` | The three engine patches, with a README explaining each |
| `scripts/` | `run.sh`, `detect-fabric.sh`, `validate_recipes.py`, `recipe_metadata.py`, `gate_37111.py`, `needle_ladder.py` |
| `results/` | `RESULTS.md`, the grids behind it, the fp8 quality gate reports |
| `.sparkrun/registry.yaml` | Registry manifest |

## Credits

- [RadixArk](https://huggingface.co/RadixArk): the NVFP4 checkpoint the SGLang
  recipes serve, and the day-0 SGLang engine work for this model.
- [tonyd2wild](https://github.com/tonyd2wild): the vLLM SM121 overlays and the
  TP2 profile the vLLM recipe grew from, and the 40-prompt category harness used
  for every decode, TTFT and cold-prefill figure here.
- [MiaAI-Lab](https://github.com/MiaAI-Lab): the fast sparse-attention SGLang
  profile the SGLang recipes grew from, the llama-benchy measurement spec, and
  the expert-parallel launch shape.
- [Weschera](https://github.com/Weschera): spark-bench, the 76-scenario graded
  eval behind the fp8 quality gate.
- [SeraphimSerapis](https://github.com/SeraphimSerapis): tool-eval-bench, the
  15-scenario tool-calling gate.
- [eugr](https://github.com/eugr): sparkrun, and the recipe and mod format this
  registry follows.
- Upstream: sgl-project/sglang#36845 and #38855, vllm-project/vllm#53945 and
  #55557, whose authors fixed the kernels these recipes depend on.

Their work made this run faster on my hardware. The measurements, the
cache-reuse diagnosis, the SM121 fp8 KV fix and the quality gate are mine, and
so are any mistakes.

## License

Apache-2.0, see `LICENSE`. The vLLM overlays under `mods/` are modified copies
of Apache-2.0 vLLM files and keep their upstream headers. The spark-bench
reports under `results/fp8-gate/` are output of that tool, reproduced with its
run labels intact.
