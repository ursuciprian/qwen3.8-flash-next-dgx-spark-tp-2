# Versions

Release names for the 2× Spark (TP=2) recipe `qwen3.8-flash-next-2x-dgx-spark`. The single-Spark recipe has its own
list in the [1× repo](https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark/blob/main/VERSIONS.md); versions are
per repo, so write the setup with the number when both appear together ("2× v1.5.0", "1× v2.1.0").

## Scheme

- **Release** `vMAJOR.MINOR.PATCH`, only for builds that became the default recipe. The date is the day the recipe PR
  that made it the default was merged.
  - MAJOR: only when users must act: new main-model weights to download (a different checkpoint, for example the
    GDN-MSE requant), a recipe rename, new required flags, a new minimum sparkrun.
  - MINOR: quality-gated speed or capability changes that need no action: drafter (even as a new HF revision),
    kernels, precision of activations, KV or the HC mixers, image, KV pool.
  - PATCH: fixes without speed claims: plan seed, recipe bug, boot fix.
  - Qwen4 will be v3.0.0 in both recipe repos.
- **Old names** (b1 to b1.6) stay valid as aliases and appear once per release as "v1.5.0 (old name b1.6)". The date
  tags `v2026.09.23-shipped` to `v2026.10.01` and their GitHub releases stay as they are.
- **Drafters**: D0 = the checkpoint's original MTP drafter, D1 = refit run 1 (shipped), D2 = refit run 2 (never
  shipped), D3 = refit run 3a, D3b = refit run 3b.
- **Experiments**: `k<NN>-<slug>`, for example `k73-2x-gdnmse-dispatch` or `k76-capability-matrix`; results folders
  start with the same name.
- **Images**: existing image tags keep their old form (`b1.6-20261008-b7fbaf96-a7e649d8-warm`). Images built from
  k76 on also get a tag with the setup and the release, for example `2x-v2.0.0`.
- Each recipe header carries a line `# Release: v1.5.0 (old name b1.6)`.

## Releases

Newest first. "Measured" is the release's own A/B against the one before it (same Sparks, alternating boots, T=0
paired probes or the llama-benchy coding grid); details in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Release | Old name | Default since | Recipe PR | What changed | Bump | Measured |
|---|---|---|---|---|---|---|
| **v1.5.0** | b1.6 | 2026-10-08 | [#120](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/pull/120) | Retrained MTP drafter D1 (24 `mtp.*` tensors over the same main weights), plan seed with all 616 plans | MINOR: drafter only, main weights unchanged | acceptance +0.036 to +0.047 per position; probe fresh c4 +10.9%, fresh c8 +6.2%, tg512 c8 +7.2%; 1 request within noise |
| v1.4.0 | b1.4 | 2026-10-01 | [#57](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/pull/57) | 131k-id MTP draft vocab, QSA race and recompile fixes, server default `reasoning_effort` `xhigh` to `medium` | MINOR: no action needed; the thinking effort is a server default each request can override | step −2 to −5% at 1–8 requests; coding c5 +4.2%; counting c1 +4.9% |
| v1.3.0 | b1.3 | 2026-09-29 | [#52](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/pull/52) | GDN uniform-decode metadata skip (~350 launches per step fewer) | MINOR: kernels only, logprobs within self-noise | step −1.9 to −3.1% at 1 request, −2.5 to −3.0% at 2 |
| v1.2.0 | b1.2 | 2026-09-27 | [#49](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/pull/49) | Hyper-connection mixers in online MXFP8 (router gate stays BF16) | MINOR: activation precision of the HC mixers, no new download | step −9.9% at 1 request, −4.4 to −7.6% at 2–4; coding c1 61.8 against 53.4 tok/s |
| v1.1.0 | b1.1 | 2026-09-26 | [#47](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/pull/47) | Exact prefix hits under MTP, `NULL_BLOCK_ID` padding fix, compile-worker cap | MINOR: faster at depth, same model | coding 16k c16 176.5 against 139.4 tok/s; cached-prefix TTFT at 16k −37% |
| v1.0.0 | b1 | 2026-09-25 | [#45](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/pull/45), [#46](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/pull/46) | Deferred GDN checkpoints, TC-45 `tool_choice` fix; #46 gave the recipes their current names | First release | coding c8 186.6 against 166.8, c16 241.1 against 218.6 tok/s |

Next release: the GDN-MSE checkpoint at TP=2 (k73-2x-gdnmse-dispatch, passed its screen and gate on 2026-10-09)
changes the main-model weights to download, so it is **v2.0.0**. After it, a drafter, kernel, precision, image or KV
change is v2.1.0 and a fix without speed claims is v2.0.1.

Every release passed the quality gate. b1.5 (GDN-MSE main weights at TP=2) failed the 128K fidelity gate and never
shipped.

Before v1.0.0 (no SemVer): the shipped build of 2026-09-23 (b0, date tag `v2026.09.23-shipped`, public image
`b0-20260918-a8333658-warm`), the `la` builds of 2026-09-19 to 2026-09-22 (locally built image, not pullable), and
the SGLang recipes of 2026-09-08 to 2026-09-12 (other engine and checkpoint).

DP=2 ([`tools/dp2/`](tools/dp2/README.md), #122 on 2026-10-08, router fix #126 on 2026-10-09) is not a recipe
release: it runs two copies of the current 1× release behind the router. This repo also carries copies of the 1×
recipes for compatibility; their releases are listed in the 1× repo.

## Tags and aliases

| Release | Tag on commit | Date tag (alias) |
|---|---|---|
| v1.5.0 | `21e616a8` (#120) | none |
| v1.4.0 | `6194f910` (#57) | `v2026.10.01` on `4afa6b95` (#58, the docs PR after it) |
| v1.3.0 | `0ff5a2f7` (#52) | `v2026.09.29` |
| v1.2.0 | `c2b39c82` (#49) | `v2026.09.27` |
| v1.1.0 | `4c58c156` (#47) | `v2026.09.26` |
| v1.0.0 | `b5d88c4d` (#46, the rename the same day as #45) | `v2026.09.25` |

## Manifests

Image tags are under `ghcr.io/ursuciprian/spark-vllm-b12x`. vLLM commits are in `ursuciprian/vllm`, b12x commits in
`ursuciprian/b12x`. The checkpoint is `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`
for every release. Plan seed = the b12x plan file baked into the image (hash prefix, record count). The base image
commit of the image is unknown for every release except v1.4.0 (`798528a2`).

| Release | Image tag | Digest | vLLM | b12x | Drafter | Plan seed | Recipe commit |
|---|---|---|---|---|---|---|---|
| v1.5.0 | `b1.6-20261008-b7fbaf96-a7e649d8-warm` | `sha256:ed8582a1a2bb7c7c173ba8ffe30f4b821bdd021b51e87f90011c8e78ec53cfb4` | `a7e649d8fede` | `b7fbaf968464` | D1: `mtp.*` of `ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE` @ `16c9bd54`, carried in the image | `15b63901` (616) | `21e616a8` |
| v1.4.0 | `b1.4-20261001-b7fbaf96-a7e649d8-warm` | `sha256:3b2f26080addadafe675f31227d6dacec3716064cbc0c7fd376b643bc34183fd` | `a7e649d8fede` | `b7fbaf968464` | D0 | `8ccf4799` (583) | `6194f910` |
| v1.3.0 | `b1.3-20260929-b7fbaf96-7344a997-warm` | `sha256:32cb8bd8800e413726b4cfe3d9947f80d4eb01d92dbe12405e3c763db7306a02` | `7344a9976077` | `b7fbaf968464` | D0 | `8ccf4799` (583) | `0ff5a2f7` |
| v1.2.0 | `b1.2-20260927-b7fbaf96-a9aa81b2-warm` | `sha256:ed5520eb037ceaadb02c9325d05ad55a37dc7cf972e0ef25a1f24dcddef624cb` | `a9aa81b23a07` | `b7fbaf968464` | D0 | `8ccf4799` (583) | `c2b39c82` |
| v1.1.0 | `b1.1-20260926-b7fbaf96-6d232f16-warm` | `sha256:91a60ebce422db8a80f58ce998a3e9bd847814c8579ac33fa4df99e62351b3a8` | `6d232f16fd44` | `b7fbaf968464` | D0 | `8ccf4799` (583) | `4c58c156` |
| v1.0.0 | `b1-20260925-b7fbaf96-14077fb3-warm` | `sha256:57c2fbd8cd811a5d22a7f2e547453f97b875f1fb4c7de60a0c3ff9fba3a79e5c` | `14077fb35ffe` | `b7fbaf968464` | D0 | `8ccf4799` (583) | `7d335d6a` / `b5d88c4d` |

Digests were read back from ghcr on 2026-10-09 and match the values recorded at release time. Seed hashes and record
counts come from `docker/b0-warm/preparation/` at each recipe commit; the image layers themselves were not inspected.
