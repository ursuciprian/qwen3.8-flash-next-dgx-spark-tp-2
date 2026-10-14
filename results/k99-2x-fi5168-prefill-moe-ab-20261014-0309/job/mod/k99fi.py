"""k99 FlashInfer path (imported by the patched B12xExperts.apply, see sitecustomize.py; also by ../opcheck.py).

Calls of >= VLLM_K99_FI_MOE_MIN_TOKENS rows (prefill chunks) run FlashInfer 5c4058bd b12x_fused_moe (PR #5168,
quant_mode nvfp4); decode and small chunks stay on b12x.
  weights   zero copy: views of b12x's prepared expert storage. vLLM fuses FC1 as [gate; up]; b12x flips that storage
            in place to [up; gate] (FlashInfer's order) once, on first use. _ensure_w13_kernel_order_inplace is
            idempotent (registry keyed by storage address), so whichever backend runs first does the flip exactly once.
            Block scales: b12x's 128x4-swizzled storage, viewed in FlashInfer's 6D MMA layout (convert_sf_to_mma_layout
            returns a strided view). FlashInfer's own caches (scale copies, I 320 -> 384 padding, workspace) are its
            memory cost; opcheck.py measures them.
  scales    FlashInfer SM12x contract (k98): input_global_scale = FC1 input_scale (= 1 / b12x a1_gscale), w1_alpha =
            weight_scale_2, fc2_input_scale = FC2 input_scale, w2_alpha = weight_scale_2 x FC2 input_scale.
  modules   the image's flashinfer 0.7.0 stays the process's `flashinfer` (the top-k/top-p sampler uses it). The
            5c4058bd tree is imported once into a private module table and swapped into sys.modules around each call,
            so its call-time imports (and the CuTe DSL compile) resolve to its own modules. cutlass-dsl is the image's,
            shared with b12x. ponytail: the swap is not thread-safe; vLLM runs the forward on one thread per worker.
  counter   'K99FI calls=N M=<rows> pid=<pid>' on stderr at calls 1, 10, 100, 1000 and every 10000; 'K99FI M=<rows>
            first call' once per distinct row count per process.
"""
import contextlib
import os
import sys

import torch

_S = {"mine": {}, "old": {}, "n": -1, "fn": None, "conv": None, "where": None, "calls": 0, "ms": set()}


def _scan():
    # copy() is one C-level step under the GIL: another thread importing during a scan cannot break the iteration
    return {k: v for k, v in sys.modules.copy().items() if k == "flashinfer" or k.startswith("flashinfer.")}


def load(src=None):
    """Import the flashinfer package under src (a repo root, as k98's PYTHONPATH=/src) into the private table."""
    if _S["fn"] is None:
        src = os.path.abspath(src or os.environ["VLLM_K99_FI_MOE_SRC"])
        old = _scan()
        for k in old:
            del sys.modules[k]
        sys.path.insert(0, src)
        try:
            import flashinfer
            from flashinfer import b12x_fused_moe
            from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
            where = flashinfer.__file__
        finally:
            sys.path.remove(src)
            mine = _scan()
            for k in mine:
                del sys.modules[k]
            sys.modules.update(old)
        if not where.startswith(os.path.join(src, "")):
            raise ImportError(f"k99: flashinfer came from {where}, not {src}")
        _S.update(mine=mine, fn=b12x_fused_moe, conv=convert_sf_to_mma_layout, where=where, n=-1)
        print(f"K99FI loaded {where} ({len(mine)} modules) pid={os.getpid()}", file=sys.stderr, flush=True)
    return _S


@contextlib.contextmanager
def active():
    mods, s = sys.modules, load()
    if len(mods) != s["n"]:  # something was imported since the last call: rescan the image's flashinfer modules
        s["old"] = _scan()
    for k in s["old"]:
        mods.pop(k, None)
    mods.update(s["mine"])
    n = len(mods)
    try:
        yield
    finally:
        if len(mods) != n:  # first-time imports during the call (lazy imports, compile): keep the tree's own
            s["mine"].update(_scan())
        for k in s["mine"]:
            mods.pop(k, None)
        mods.update(s["old"])
        s["n"] = len(mods)


def views(impl, g1, g2, a1g, a2g):
    """FlashInfer operands over b12x's prepared storage (impl = PreparedExperts._impl) + four [E] scale vectors.

    g1/g2: FC1/FC2 weight_scale_2; a1g/a2g: b12x's activation multipliers (1 / input_scale), all [E] float32."""
    from b12x.moe.fused_moe._impl import _ensure_w13_kernel_order_inplace
    e, h, i = int(impl.num_experts), int(impl.hidden_size), int(impl.intermediate_size)
    _ensure_w13_kernel_order_inplace(impl.w1_fp4, impl.w1_blockscale, n=i, k=h)
    f8 = lambda t: t.reshape(-1).view(torch.uint8).view(torch.float8_e4m3fn)
    with active():
        s1 = _S["conv"](f8(impl.w1_blockscale), m=2 * i, k=h, num_groups=e)
        s2 = _S["conv"](f8(impl.w2_blockscale), m=h, k=i, num_groups=e)
    fc2 = (1.0 / a2g.float()).contiguous()
    return {"impl": impl, "E": e, "w1": impl.w1_fp4.view(torch.uint8), "s1": s1, "w2": impl.w2_fp4.view(torch.uint8),
            "s2": s2, "a1": g1.float().contiguous(), "a2": (g2.float() * fc2).contiguous(),
            "ig": (1.0 / a1g.float()).contiguous(), "fc2": fc2}


def run(st, output, x, topk_weights, topk_ids):
    _S["calls"] += 1
    n = _S["calls"]
    m = int(x.shape[0])
    if n in (1, 10, 100, 1000) or n % 10000 == 0:
        print(f"K99FI calls={n} M={m} pid={os.getpid()}", file=sys.stderr, flush=True)
    if m not in _S["ms"]:
        _S["ms"].add(m)
        print(f"K99FI M={m} first call pid={os.getpid()}", file=sys.stderr, flush=True)
    ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
    with active():
        _S["fn"](x, st["w1"], st["s1"], st["w2"], st["s2"], ids, topk_weights, st["E"], int(ids.shape[1]),
                 w1_alpha=st["a1"], w2_alpha=st["a2"], fc2_input_scale=st["fc2"], input_global_scale=st["ig"],
                 output=output, quant_mode="nvfp4")
    return output


def apply(experts, output, x, topk_weights, topk_ids, norm):
    """B12xExperts.apply replacement for one call (norm = vLLM's _normalize_expert_scale)."""
    impl = experts._prepared()._impl
    st = getattr(experts, "_k99", None)
    if st is None or st["impl"] is not impl:  # built on the first call (the profile run), rebuilt after a re-prepare
        st = experts._k99 = views(impl, norm(experts.g1_alphas), norm(experts.g2_alphas), norm(experts.a1_gscale),
                                  norm(experts.a2_gscale))
    run(st, output, x, topk_weights, topk_ids)
