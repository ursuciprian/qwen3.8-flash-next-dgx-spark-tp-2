#!/usr/bin/env python3
"""k99 CPU check (no GPU, --network none, the shipped 2x image; dry-run and preflight): run with PYTHONPATH=/k99mod
VLLM_K99_FI_MOE_SRC=/fi VLLM_K99_FI_MOE_MIN_TOKENS=<min>, the mod at /k99mod and the 5c4058bd tree at /fi.
  - our sitecustomize is the one Python loaded (the image's /usr/lib one only installs apport) and its hook is active
  - both patches apply exactly once to the image's vLLM source and compile; vllm.envs imported through the hook
    exposes both K99 variables and envs.compile_factors() hashes them (own torch compile key for the arm)
  - k99fi loads the 5c4058bd tree beside the image's flashinfer 0.7.0, the swap resolves the MoE dispatch module
    from the tree, and the process's `flashinfer` is 0.7.0 again afterwards; swap cost per call (steady state)
Prints 'CPUCHECK ok' last on success.
"""
import os
import sys
import timeit

sc = sys.modules["sitecustomize"]
assert sc.__file__.startswith("/k99mod/"), f"sitecustomize from {sc.__file__}"
assert any(type(f).__name__ == "_Finder" for f in sys.meta_path), "k99 import hook not installed"
V = "/usr/local/lib/python3.12/dist-packages/vllm"
for name, rel in (("vllm.envs", "envs.py"), ("vllm.model_executor.layers.fused_moe.b12x", "model_executor/layers/fused_moe/b12x.py")):
    compile(sc.apply(open(os.path.join(V, rel)).read(), sc.PATCHES[name], name), rel, "exec")
    print("patch applies and compiles:", rel)
import vllm.envs as envs  # noqa: E402
f = {k: v for k, v in envs.compile_factors().items() if "K99" in k}
assert f == {"VLLM_K99_FI_MOE_SRC": "/fi", "VLLM_K99_FI_MOE_MIN_TOKENS": int(os.environ["VLLM_K99_FI_MOE_MIN_TOKENS"])}, f
print("compile factors:", f)
import flashinfer  # noqa: E402
import flashinfer.sampling  # noqa: E402,F401
v0 = flashinfer
import k99fi  # noqa: E402
s = k99fi.load("/fi")
assert sys.modules["flashinfer"] is v0
with k99fi.active():
    import flashinfer as f2
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import launch_sm120_moe  # noqa: F401
    assert f2.__file__.startswith("/fi/"), f2.__file__
assert sys.modules["flashinfer"] is v0 and "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch" in s["mine"]
print(f"image flashinfer {v0.__version__} stays the process's; 5c4058bd tree {len(s['mine'])} modules in the private table")


def sw():
    with k99fi.active():
        pass


print(f"swap cost per call: {timeit.timeit(sw, number=2000) / 2000 * 1e6:.1f} us ({len(sys.modules)} modules)")
print("CPUCHECK ok")
