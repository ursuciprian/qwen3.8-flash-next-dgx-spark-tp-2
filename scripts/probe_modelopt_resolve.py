#!/usr/bin/env python3
"""Resolve the MTP experts' quant algo through a modelopt.py overlay, offline.

Run inside the pinned vLLM image with the overlay mounted and the HF cache at
/cache/huggingface. Answers "will the RoutedExperts dispatch see an algo it
handles?" before spending a 15-minute boot on it.

  docker run --rm --entrypoint python3 \
    -v <overlay>/modelopt.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py:ro \
    -v ~/.cache/huggingface:/cache/huggingface:ro \
    -v $PWD/scripts/probe_modelopt_resolve.py:/t.py:ro \
    vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c /t.py [snapshot_sha]

2026-09-06: on nvidia snapshot fc694b54 this printed algo=FP8_PB_WO for
mtp.layers.48.mlp.experts - the name no community overlay dispatched on.
"""
import json, sys
from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig

snap = sys.argv[1] if len(sys.argv) > 1 else "fc694b54fb0174e0913e6adf86691ef85a4ead47"
d = f"/cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/{snap}"
cfg = json.load(open(f"{d}/config.json"))["quantization_config"]
mp = ModelOptMixedPrecisionConfig.from_config(cfg)
print("class:", type(mp).__name__)
for p in ("mtp.layers.48.mlp.experts", "mtp.layers.0.mlp.experts", "model.language_model.layers.0.mlp.experts"):
    print("%-48s -> algo=%s excluded=%s group=%s" % (
        p, mp._resolve_quant_algo(p), mp.is_layer_excluded(p), mp._quantized_layer_group_size(p)))
