# Correction (2026-10-13)

The attempts in this folder say "did not boot" for both stock images. That was wrong: vLLM never started. My launcher
(sparkrun) appends its own `bash -c` command to the container, and `vllm/vllm-openai` declares
`ENTRYPOINT ["vllm","serve"]`, which took that command as its own arguments, so the server was never launched. It was a
launcher problem on my side, not a vLLM problem.

The rerun with the image entrypoint cleared in the recipe (`executor_config: {entrypoint: ""}`) is
[k84c-stock-entrypoint](../k84c-stock-entrypoint-20261013-1034/). There vLLM starts, and stock vLLM at TP=2 on two Sparks fails during model init for
reasons inside vLLM: v0.31.0 refuses the TP-sharded vision MLP (`Unsupported model when in features size is not
multiple of 16`), and the 7d0b4e57 nightly with `--mm-encoder-tp-mode data` stops at the per-layer embedding table
(`Shape mismatch for PLE embedding shard 0: expected (2500012, 160), got (2500012, 80)`). No stock cell ran.
