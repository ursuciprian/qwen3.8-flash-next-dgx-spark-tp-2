"""k99 mod (2026-10-13): FlashInfer 5c4058bd fused MoE for large prefill chunks on the 2x, without editing the image.

Loaded through PYTHONPATH=/cache/runtime/k99/mod (the job puts this directory into the arm's own runtime cache on
both Sparks). When VLLM_K99_FI_MOE_SRC is set, an import hook patches the source text of two vLLM modules as they are
imported (the files on disk stay untouched):
  vllm.envs                                   registers VLLM_K99_FI_MOE_SRC and VLLM_K99_FI_MOE_MIN_TOKENS (default
                                              6048), so envs.compile_factors() hashes them into the torch compile key
                                              (a warm AOT cache of another setting is never loaded)
  vllm.model_executor.layers.fused_moe.b12x  B12xExperts.apply: W4A4 NVFP4 calls (main-model experts; the W4A16 MTP
                                              drafter experts stay on b12x) with >= MIN_TOKENS rows outside CUDA graph
                                              capture go to k99fi.apply (FlashInfer); every other call stays on b12x
Every anchor must occur exactly once (vLLM d21d7ade); otherwise the import raises and the boot fails loudly.
Nothing happens when VLLM_K99_FI_MOE_SRC is not set.
"""
import importlib.abc
import importlib.util
import os
import sys

PATCHES = {
    "vllm.envs": (
        ('    "VLLM_B12X_A16_MAX_TOKENS": lambda: int(os.getenv("VLLM_B12X_A16_MAX_TOKENS", "0")),\n',
         '    "VLLM_B12X_A16_MAX_TOKENS": lambda: int(os.getenv("VLLM_B12X_A16_MAX_TOKENS", "0")),\n'
         '    "VLLM_K99_FI_MOE_SRC": lambda: os.getenv("VLLM_K99_FI_MOE_SRC", ""),\n'
         '    "VLLM_K99_FI_MOE_MIN_TOKENS": lambda: int(os.getenv("VLLM_K99_FI_MOE_MIN_TOKENS", "6048")),\n'),
    ),
    "vllm.model_executor.layers.fused_moe.b12x": (
        ("logger = init_logger(__name__)\n",
         "logger = init_logger(__name__)\n"
         "_K99_MIN = envs.VLLM_K99_FI_MOE_MIN_TOKENS if envs.VLLM_K99_FI_MOE_SRC else 0\n"),
        ("        topk_weights = _normalize_topk_weights(topk_weights)\n",
         "        topk_weights = _normalize_topk_weights(topk_weights)\n"
         "        if (_K99_MIN and self._quant_mode == \"nvfp4\" and int(hidden_states.shape[0]) >= _K99_MIN\n"
         "                and not _is_current_stream_capturing()):\n"
         '            return __import__("k99fi").apply(self, output, hidden_states, topk_weights, topk_ids, '
         "_normalize_expert_scale)\n"),
    ),
}


def apply(src, patches, name="?"):
    for old, new in patches:
        n = src.count(old)
        if n != 1:
            raise ImportError(f"k99 mod: anchor found {n} times in {name}: {old.strip()[:80]!r}")
        src = src.replace(old, new)
    return src


class _Loader(importlib.abc.Loader):
    def __init__(self, origin, name):
        self.origin, self.name = origin, name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        with open(self.origin) as f:
            src = apply(f.read(), PATCHES[self.name], self.name)
        exec(compile(src, self.origin, "exec"), module.__dict__)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            raise ImportError(f"k99 mod: no source file for {name}")
        spec.loader = _Loader(spec.origin, name)
        return spec


if os.environ.get("VLLM_K99_FI_MOE_SRC"):
    sys.meta_path.insert(0, _Finder())
    print(f"K99MOD active in pid {os.getpid()} (min {os.environ.get('VLLM_K99_FI_MOE_MIN_TOKENS', '6048')})",
          file=sys.stderr, flush=True)
