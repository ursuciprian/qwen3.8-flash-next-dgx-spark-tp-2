# b1.6 base layer (#115): the retrained MTP drafter for the 2x recipe, which serves
# local-inference-lab/Qwen3.8-Flash-Next-NVFP4 @ 7c4f1bc1 (b1.5's GDN-MSE requant of the main model failed the TP=2 128k
# fidelity gate, so the 2x keeps 7c4f1bc1 and only takes the drafter).
#  - mtp-refit-16c9bd54.safetensors: the 24 tensors of model-00034 that differ between 7c4f1bc1 and
#    ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE @ 16c9bd54 (the v3e checkpoint, #97 refit run 1), 178 MB. Not in git:
#      hf download ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE model-00034-of-00036.safetensors --revision 16c9bd54788d12390838a65ce4a4ecda97fa5f1d
#      python3 overlay.py patch <7c4f1bc1 model-00034> <16c9bd54 model-00034> mtp-refit-16c9bd54.safetensors
#  - overlay.py: the recipe command runs `overlay.py build` before `vllm serve` (see its docstring).
# Then docker/b0-warm on top (CI build-b0-warm) adds the b12x plan seed for the overlay path.
#   docker build --build-arg BASE=ghcr.io/ursuciprian/spark-vllm-b12x:b1.4-20261001-b7fbaf96-a7e649d8-warm -t <tag> .
ARG BASE
FROM ${BASE}
COPY overlay.py mtp-refit-16c9bd54.safetensors /opt/mtp-refit/
RUN echo 'c5816f92605f90b03629d88845fcc234ea6077175be2e36e0b397fdf78fa0884  /opt/mtp-refit/mtp-refit-16c9bd54.safetensors' | sha256sum -c - \
 && chmod -R a+rX /opt/mtp-refit
