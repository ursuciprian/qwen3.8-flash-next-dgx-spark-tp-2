# Warm layer over the B0 image (spark-vllm-b12x a8333658). A fresh pair boots it
# with plain `sparkrun run`, with no mods and no host mounts:
#  - preparation/: b12x plan-selection caches for checkpoint 7c4f1bc1 at TP=2 (8ccf4799) and TP=1
#    (54b3e768, the 1x v3c recipe; added 2026-10-02), and TP=1 for ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE @ 244cb6fe
#    (f7894ea6, the 1x v3d recipe; added 2026-10-05), and TP=2 for the b1.6 overlay path
#    /cache/runtime/mtp-refit/Qwen3.8-Flash-Next-NVFP4-7c4f1bc1-mtp-16c9bd54 (15b63901, 616 records = 8ccf4799 + the 33
#    a16 GEMM plans 8ccf4799 lacks; docker/mtp-refit; added 2026-10-08) on
#    GB10. Its identity is device name + namespace (model snapshot path, TP, dtype,
#    spec method), not host or GPU UUID, so it hits on any GB10 pair. The recipe's
#    command copies it into sparkrun's runtime cache ($XDG_CACHE_HOME) when that
#    cache does not have it yet. Compiled kernels (b12x/compile/<xx>/*.o) are keyed
#    by GPU UUID and are deliberately not shipped: they rebuild on the first boot
#    and then persist in sparkrun's runtime cache.
#  - archive/mods/b12x-startup-boundedwait applied at build time (fail-closed,
#    sha-checked; skips itself when the base vLLM already has it, as the b1
#    base candidate-b12x-b7fbaf96-vllm-14077fb3 does).
#  - mtp-vocab/ids-v2-K131072.txt.gz (b1.4, 2026-10-01): the 131,072-id MTP draft-vocab
#    list the recipe's VLLM_MTP_DRAFT_VOCAB points at (/opt/mtp-vocab). Our own ranking
#    (scripts/build_draft_vocab_v2.py, archive/mods/r7-dvocab/README.md); its tail order follows the
#    65,536-id list of dime-online/qwen3.8-Flash-DGX-UltraFast (Apache-2.0). sha256-checked
#    below. Base images without draft-vocab support simply ignore it.
# The AGPL draft_vocab table is not part of this image or its base.
#
# Build (from the repo root):
#   docker buildx build --load --build-context mods=archive/mods \
#     --build-arg BASE=<local base> -t <tag> docker/b0-warm
# CI: ursuciprian/spark-vllm-b12x workflow build-b0-warm (b0 2026-09-23,
# b1 2026-09-25).
ARG BASE=spark-vllm-b12x:local-20260918-a8333658
FROM ${BASE}
COPY preparation/ /opt/b12x-seed/preparation/
COPY mtp-vocab/ /opt/mtp-vocab/
COPY --from=mods b12x-startup-boundedwait/ /tmp/mod-boundedwait/
RUN bash /tmp/mod-boundedwait/run.sh && rm -rf /tmp/mod-boundedwait \
 && echo 'fc40970533741fc14217879a5f5385897f4126bd1d7ce23ed0268236b4b8cf7b  /opt/mtp-vocab/ids-v2-K131072.txt.gz' | sha256sum -c - \
 && chmod -R a+rX /opt/b12x-seed /opt/mtp-vocab
