# Warm layer over the B0 image (spark-vllm-b12x a8333658). A fresh pair boots it
# with plain `sparkrun run`, with no mods and no host mounts:
#  - preparation/: b12x plan-selection cache for checkpoint 7c4f1bc1 at TP=2 on
#    GB10. Its identity is device name + namespace (model snapshot path, TP, dtype,
#    spec method), not host or GPU UUID, so it hits on any GB10 pair. The recipe's
#    command copies it into sparkrun's runtime cache ($XDG_CACHE_HOME) when that
#    cache does not have it yet. Compiled kernels (b12x/compile/<xx>/*.o) are keyed
#    by GPU UUID and are deliberately not shipped: they rebuild on the first boot
#    and then persist in sparkrun's runtime cache.
#  - mods/b12x-startup-boundedwait applied at build time (fail-closed, sha-checked).
# The AGPL draft_vocab table is not part of this image or its base.
#
# Build (from the repo root):
#   docker buildx build --load --build-context mods=mods \
#     -t ghcr.io/ursuciprian/spark-vllm-b12x:b0-20260918-a8333658-warm docker/b0-warm
ARG BASE=spark-vllm-b12x:local-20260918-a8333658
FROM ${BASE}
COPY preparation/ /opt/b12x-seed/preparation/
COPY --from=mods b12x-startup-boundedwait/ /tmp/mod-boundedwait/
RUN bash /tmp/mod-boundedwait/run.sh && rm -rf /tmp/mod-boundedwait \
 && chmod -R a+rX /opt/b12x-seed
