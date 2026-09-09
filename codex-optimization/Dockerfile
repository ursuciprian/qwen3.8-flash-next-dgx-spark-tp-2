FROM vllm/vllm-openai@sha256:f5df5cc3302b5f404848c4eca88d7bf7ed5226e151c056da22816d7734644d67
ARG ARM=baseline
COPY mods/vllm-flashnext-nightly-8a728663 /opt/flashnext/mods/vllm-flashnext-nightly-8a728663
COPY codex-optimization /opt/flashnext/codex-optimization
RUN python3 /opt/flashnext/codex-optimization/install.py "$ARM"
