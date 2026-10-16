# k104s (tp-2 #176, 2026-10-16): the k101s/k104 test image (spark-vllm-b12x:k101-gv-10d0acc8-d21d7ade, Id d7da780422fd) + the f509586e AOT artifacts of
# both ranks compiled for its vLLM files (k101s bake). Everything else in /opt/b12x-seed is the v2.1.0 seed.
FROM spark-vllm-b12x:k101-gv-10d0acc8-d21d7ade
COPY aot/ /opt/b12x-seed/torch_compile_cache/torch_aot_compile/
