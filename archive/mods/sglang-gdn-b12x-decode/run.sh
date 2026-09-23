#!/usr/bin/env bash
# b12x CuTeDSL GDN decode kernel for SGLang (Qwen3.8-Flash-Next: 36 of 48 layers are GDN).
# The kernel is the one eugr's b12x vLLM route runs on GB10 (structured 113 tok/s vs 66 on the
# SGLang Triton path, 2026-09-15). Installs b12x from git at a pinned commit, drops the
# B12xGDNKernel wrapper next to SGLang's GDN kernels and routes the GDN decode dispatcher to it
# when SGLANG_GDN_B12X=1. Prefill/extend and NEXTN target-verify stay on Triton. Idempotent,
# fail-closed. See gdn_b12x.py for the norm-idempotence trick that keeps the model code untouched.
set -euo pipefail
B12X_REF="${B12X_REF:-40bcdf82a03b}"
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 - "$HERE" "$B12X_REF" <<'PY'
import importlib.util, pathlib, shutil, subprocess, sys
here, ref = pathlib.Path(sys.argv[1]), sys.argv[2]
root = pathlib.Path(importlib.util.find_spec("sglang").origin).parent / "srt/layers/attention/linear"
tag = "[gdn-b12x]"
# 1. b12x package
if importlib.util.find_spec("b12x") is None or importlib.util.find_spec("b12x.sequence") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
                    f"git+https://github.com/local-inference-lab/b12x@{ref}"], check=True)
    print(f"gdn-b12x: installed b12x@{ref}")
else:
    print("gdn-b12x: b12x already present")
# 2. kernel wrapper
dst = root / "kernels/gdn_b12x.py"
src = here / "gdn_b12x.py"
if not dst.exists() or dst.read_text() != src.read_text():
    shutil.copy(src, dst); print(f"gdn-b12x: wrote {dst}")
# 3. dispatcher anchor
p = root / "gdn_backend.py"; s = p.read_text()
if tag in s:
    print("gdn-b12x: dispatcher already patched")
else:
    old = ("        cutedsl_kernel = None\n"
           "        if decode_backend.is_triton():\n"
           "            self.decode_kernel = triton_kernel\n")
    new = ("        cutedsl_kernel = None\n"
           "        import os  # " + tag + "\n"
           "        if os.environ.get(\"SGLANG_GDN_B12X\") == \"1\":\n"
           "            from sglang.srt.layers.attention.linear.kernels.gdn_b12x import B12xGDNKernel\n"
           "            self.decode_kernel = B12xGDNKernel()\n"
           "            import logging as _lg; _lg.getLogger(__name__).info(\"GDN decode kernel: b12x CuTeDSL (mod sglang-gdn-b12x-decode)\")\n"
           "        elif decode_backend.is_triton():\n"
           "            self.decode_kernel = triton_kernel\n")
    if s.count(old) != 1:
        print(f"gdn-b12x: dispatcher anchor matched {s.count(old)} times, expected 1; refusing"); sys.exit(1)
    import ast; ast.parse(s.replace(old, new)); p.write_text(s.replace(old, new)); print(f"gdn-b12x: patched {p}")
# 4. verify anchor: route NEXTN target_verify to the same kernel unless SGLANG_GDN_B12X_VERIFY=0
old2 = ("        self.supports_packed_decode = getattr(\n"
        "            self.decode_kernel, \"supports_packed_decode\", False\n"
        "        )\n")
new2 = old2 + ("        if os.environ.get(\"SGLANG_GDN_B12X\") == \"1\" and os.environ.get(\"SGLANG_GDN_B12X_VERIFY\", \"1\") == \"1\":  # " + tag + "\n"
               "            self.verify_kernel = self.decode_kernel\n"
               "            self.verify_kernel_is_flashinfer = False\n")
s = p.read_text()
if tag + "\n            self.verify_kernel" in s:
    print("gdn-b12x: verify anchor already patched")
else:
    if s.count(old2) != 1:
        print(f"gdn-b12x: verify anchor matched {s.count(old2)} times, expected 1; refusing"); sys.exit(1)
    import ast; ast.parse(s.replace(old2, new2)); p.write_text(s.replace(old2, new2)); print("gdn-b12x: patched verify routing")
PY
