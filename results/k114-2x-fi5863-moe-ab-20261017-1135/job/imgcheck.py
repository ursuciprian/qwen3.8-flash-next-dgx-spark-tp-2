"""k114 in-image CPU check (run.sh img_ok, --network none, no GPU): the vLLM backend imports flashinfer_k114, nothing
from FlashInfer's embedded b12x tree is loaded (it registers b12x:: torch ops, the namespace of our b12x), and the
image's b12x, FlashInfer 0.7.0 and CuTe DSL 4.7.0 are the ones in use."""
import importlib.metadata as md
import os
import sys

import b12x
import flashinfer
from vllm.model_executor.layers.fused_moe.experts import flashinfer_sm12x_w4a4_moe as m

m._fi()
import flashinfer_k114  # noqa: E402
from flashinfer_k114.fused_moe.cute_dsl.blackwell_sm12x import moe_nvfp4_w4a4  # noqa: E402,F401
from flashinfer_k114.fused_moe.utils import get_hybrid_num_tokens_buckets  # noqa: E402,F401

emb = os.path.join(os.path.dirname(flashinfer_k114.__file__), "experimental", "b12x")
bad = [n for n, x in list(sys.modules.items())
       if n.startswith("flashinfer_k114.experimental.b12x") or (getattr(x, "__file__", None) or "").startswith(emb)]
assert not bad, ("embedded b12x loaded", bad[:5])
assert b12x.__file__ == "/usr/local/lib/python3.12/dist-packages/b12x/__init__.py", b12x.__file__
assert "_AliasFinder" not in open(b12x.__file__).read(), "b12x is FlashInfer's compatibility alias"
assert flashinfer.__version__ == md.version("flashinfer-python") == "0.7.0", flashinfer.__version__
assert md.version("nvidia-cutlass-dsl") == "4.7.0", md.version("nvidia-cutlass-dsl")
print("k114 image check ok", flashinfer_k114.__git_commit__, "b12x", md.version("b12x"))
