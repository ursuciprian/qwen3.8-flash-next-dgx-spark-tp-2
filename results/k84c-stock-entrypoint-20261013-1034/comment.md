Correction to my earlier k84 result on this issue: I reported that stock vLLM did not boot on GB10. That was wrong: vLLM never started. My launcher (sparkrun) appends its own `bash -c` command to the container, and `vllm/vllm-openai` declares `ENTRYPOINT ["vllm","serve"]`, which took that command as its own arguments, so the server was never launched. It was a launcher problem on my side, not a vLLM problem.

k84c-stock-entrypoint reran it with the image entrypoint cleared in the recipe (`executor_config: {entrypoint: ""}`, which sparkrun 0.3.10 honours), the same two stock images, the same recipe flags and the same cells as k84, on the unmodified checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `7c4f1bc1`, TP=2 on both Sparks with MTP 3. No image was rebuilt or pushed. vLLM now starts, and both attempts fail during model init for reasons inside vLLM:

- `vllm/vllm-openai:v0.31.0` (`sha256:c1c9f6fd`): the vision encoder's MLP `linear_fc2` is row-sharded at TP=2 and the ModelOpt quantization refuses the shard: `ValueError: Unsupported model when in features size is not multiple of 16` (both ranks, `qwen3_vl.py` -> `modelopt.py create_weights`).
- `vllm/vllm-openai:nightly-7d0b4e57` (`sha256:2b3b0fe1`), with `--mm-encoder-tp-mode data` added, a stock flag that keeps the vision encoder unsharded (my recipes set it too): it passes the vision encoder and fails loading the per-layer embedding table: `ValueError: Shape mismatch for PLE embedding shard 0: expected (2500012, 160), got (2500012, 80)` (`qwen4_exp/nvidia/ngram_embedding.py load_weights`).

So there are still no stock numbers for the 2x setup: no cell ran. Not tried in this window: v0.31.0 with `--mm-encoder-tp-mode data`, and a single-Spark stock setup. The generated recipes, image digests, attempts and the rank logs of the nightly attempt are in the results folder (the v0.31.0 rank log is an excerpt; I removed that container by hand before the job saved its log).

Results: {RESULTS_URL}
