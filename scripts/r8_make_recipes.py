#!/usr/bin/env python3
"""Generate the r8 arm recipes from a base recipe (opus-kernel-11, 2026-10-01).

    r8_make_recipes.py tp1 --base <tp1-safe.yaml> --image <tag> [--out <dir>]
    r8_make_recipes.py tp2 --base <registry recipe = b1.4> [--out <dir>]

TP=1 arms start from the tp1-safe recipe (KV 6 GiB, 1 compile worker), swap in the r8
TP=1 screening image (vLLM exp/r8-tp1-screen: page-cache PLE reader + all r8 knobs,
off by default), drop the vllm-tp1-ple-mmap overlay (the code is in the image) and
turn on VLLM_PLE_MMAP_STATS=100 in every arm, control included. TP=2 arms start from
the promoted b1.4 registry recipe. Each arm changes one thing; every edit must match
exactly once or the script fails, so a changed base cannot yield a silent no-op arm.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

AD = "archive/recipes/qwen3.8-flash-next"
SPEC = '"draft_sample_method":"probabilistic"'
COMPILE = """'{{"pass_config":{{"fuse_act_quant":true}}}}'"""
# Default capture sizes from the boot logs (TP1 2026-09-30 tp1-gate, TP2 r7 2026-10-01).
CAPTURE = {
    "tp1": [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80],
    "tp2": [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
            136, 144, 152, 160],
}


def sub(text, pattern, repl, flags=re.M):
    out, n = re.subn(pattern, repl, text, flags=flags)
    if n != 1:
        sys.exit(f"edit {pattern!r} matched {n} times (want 1)")
    return out


def env(text, **pairs):
    lines = "".join(f'  {k}: "{v}"\n' for k, v in pairs.items())
    return sub(text, r"^env:\n", lambda m: m.group(0) + lines)


def drop_env(text, key):
    return sub(text, rf'^  {key}: "[^"]*"\n', "")


def spec(text, extra):
    return sub(text, re.escape(SPEC), lambda m: SPEC + "," + extra)


def capture(text, tp):
    sizes = sorted(set(CAPTURE[tp]) | {5, 10, 20})
    new = COMPILE[:-3] + ',"cudagraph_capture_sizes":' + json.dumps(sizes).replace(" ", "") + "}}'"
    return sub(text, re.escape(COMPILE), lambda m: new)


def drafts(text, n):
    return sub(text, r"^  num_speculative_tokens: 4$", f"  num_speculative_tokens: {n}")


def willneed(text):
    return env(text, VLLM_PLE_MMAP_WILLNEED_MAX="8192")


def mtpq(text):
    return sub(text, r'^  VLLM_QWEN38_HC_MXFP8: "hc"$', '  VLLM_QWEN38_HC_MXFP8: "hc,mtp"')


def profiler(text):
    text = sub(text, r"^runtime: vllm\n", "runtime: vllm\n\nmods:\n  - vllm-decode-profiler\n")
    return env(text, VLLM_LOCAL_PROF_TRIGGER_DIR="/cache/runtime/prof-trigger",
               VLLM_LOCAL_PROF_STEPS="60")


TP1_ARMS = {
    "tp1-off": ("control: r8 TP=1 image, every r8 knob off", lambda t: t),
    "tp1-prof": ("Phase 1 profile: control + mods/vllm-decode-profiler (60-step windows)",
                 lambda t: profiler(t)),
    "tp1-ple-par": ("PLE reader pool for decode-sized gathers (threshold 2048 -> 32, 8-row jobs)",
                    lambda t: env(t, VLLM_PLE_MMAP_PARALLEL_LOOKUPS="32", VLLM_PLE_MMAP_CHUNK="8")),
    "tp1-ple-willneed": ("PLE reader WILLNEED pass: fadvise every row of gathers <= 8192 lookups first",
                         lambda t: env(t, VLLM_PLE_MMAP_WILLNEED_MAX="8192")),
    "tp1-ple-prewarm": ("PLE scale plane (3 GiB) streamed into the page cache at startup",
                        lambda t: env(t, VLLM_PLE_MMAP_PREWARM="scale")),
    "tp1-ple-cpuhash": ("PLE ids hashed on the host (no GPU hash kernel / id download); "
                        "first 2000 reads checked against the GPU hash",
                        lambda t: env(t, VLLM_PLE_MMAP_CPU_HASH="1", VLLM_PLE_MMAP_CPU_HASH_CHECK="2000")),
    "tp1-trim": ("malloc_trim after load, KV init and warm-up", lambda t: env(t, VLLM_MALLOC_TRIM="1")),
    "tp1-disk": ("native b12x O_DIRECT io_uring disk table instead of the page cache "
                 "(VLLM_PLE_TABLE_MEMORY=disk, no page cache used)",
                 lambda t: env(drop_env(t, "VLLM_PLE_MMAP"), VLLM_PLE_TABLE_MEMORY="disk")),
    "tp1-block": ("block verification rejection sampling (lossless)",
                  lambda t: spec(t, '"rejection_sample_method":"block"')),
    "tp1-d3": ("3 MTP drafts instead of 4", lambda t: drafts(t, 3)),
    "tp1-cg": ("CUDA-graph capture sizes + [5,10,20]", lambda t: capture(t, "tp1")),
    # opus-kernel-17 (2026-10-02): tp1-dv128 sits on willneed (the r8 PLE winner), like the
    # drafter arms below; all run on the k16 image, control tp1-willneed.
    "tp1-dv128": ("willneed + b1.4's MTP draft vocabulary, r7 dvocab v2 K=131072 (numerics: canary)",
                  lambda t: env(willneed(t), VLLM_MTP_DRAFT_VOCAB="/cache/runtime/r7/ids-v2-K131072.txt.gz")),
    "tp1-mtpq": ("willneed + online MXFP8 on the MTP layer's BF16 linears (VLLM_QWEN38_HC_MXFP8=hc,mtp; "
                 "numerics: canary)", lambda t: mtpq(willneed(t))),
    "tp1-mtpq-d3": ("willneed + mtpq + 3 MTP drafts instead of 4 (numerics: canary)",
                    lambda t: drafts(mtpq(willneed(t)), 3)),
    "tp1-dv98": ("r7 dvocab v2 K=98304 MTP draft vocabulary (rejected at TP=2 in r7)",
                 lambda t: env(t, VLLM_MTP_DRAFT_VOCAB="/cache/runtime/r7/ids-v2-K98304.txt.gz")),
    # opus-kernel-16: the wm arms run on the r8 TP=1 wm image (b12x exp/r8-tp1-wm, I=640
    # geometry); tp1-willneed is their same-image control (= tp1-ple-willneed, the r8 winner).
    "tp1-willneed": ("control for the k16 arms (wm, b12x GEMV): PLE WILLNEED pass (gathers <= 8192 lookups)",
                     lambda t: willneed(t)),
    "tp1-wm2m8": ("willneed + weight-major fused NVFP4 MoE decode (wm) at <= 8 tokens "
                  "(c1 verify M=5); numerics: BF16 combine order, canary",
                  lambda t: env(willneed(t), B12X_MOE_DECODE_BACKEND="wm", B12X_MOE_WM_MAX_TOKENS="8")),
    "tp1-wm": ("willneed + wm MoE decode at <= 32 tokens (c1..c6 verify); numerics: canary",
               lambda t: env(willneed(t), B12X_MOE_DECODE_BACKEND="wm", B12X_MOE_WM_MAX_TOKENS="32")),
    # vLLM feat/r8-mtp-gemv: b12x SIMT GEMV (rows <= 8) for BF16 projections, same math.
    "tp1-gemv-mtp": ("willneed + b12x GEMV for the MTP draft qkv/o_proj (BF16, rows <= 8)",
                     lambda t: env(willneed(t), VLLM_QWEN38_B12X_GEMV="mtp")),
    "tp1-gemv": ("willneed + b12x GEMV for the MTP draft qkv/o_proj and every MoE router gate",
                 lambda t: env(willneed(t), VLLM_QWEN38_B12X_GEMV="mtp,gate")),
}

TP2_ARMS = {
    "r8-off": ("control: the promoted registry recipe verbatim", lambda t: t),
    "r8-prof": ("b1.4 + mods/vllm-decode-profiler, for the eager-tail re-profile (prof:r8-prof)",
                profiler),
    "r8-cg": ("CUDA-graph capture sizes + [5,10,20]", lambda t: capture(t, "tp2")),
    "r8-d3": ("per-batch drafts: 4 at 1-7 running requests, 3 at 8-16",
              lambda t: spec(t, '"num_speculative_tokens_per_batch_size":[[1,7,4],[8,16,3]]')),
    "r8-d3g": ("3 MTP drafts at every batch size", lambda t: drafts(t, 3)),
    "r8-fp4scale": ("VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13: one layer-wide w13 input "
                    "scale (a13_scale.amax()) instead of per-expert (numerics: canary)",
                    lambda t: env(t, VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE="w13")),
    "r8-block": ("block verification rejection sampling (lossless)",
                 lambda t: spec(t, '"rejection_sample_method":"block"')),
}


def make(kind, base, out, image=None, only=None):
    text = Path(base).read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    if kind == "tp1":
        arms, prefix = TP1_ARMS, "qwen3.8-flash-next-1x-dgx-spark"
        text = sub(text, r"^container: .*$", f"container: {image}")
        text = sub(text, r"^mods:\n  - vllm-tp1-ple-mmap\n\n?", "")
        text = env(text, VLLM_PLE_MMAP_STATS="100")
    else:
        arms, prefix = TP2_ARMS, "qwen3.8-flash-next-2x-dgx-spark"
        if "rejection_sample_method" in text:
            arms = {k: v for k, v in arms.items() if k != "r8-block"}
            print("base already sets rejection_sample_method: r8-block skipped (in b1.4)")
    if only:
        missing = set(only) - set(arms)
        if missing:
            sys.exit(f"unknown arms {sorted(missing)}")
        arms = {k: v for k, v in arms.items() if k in only}
    for arm, (what, edit) in arms.items():
        body = edit(text)
        body = sub(body, r"^name: .*$", f"name: {prefix}-{arm}")
        body = sub(body, r"^description: .*$", f'description: "r8 {arm}: {what}"')
        head = (f"# GENERATED by scripts/r8_make_recipes.py {kind} from {Path(base).name} "
                f"(sha256 {digest}).\n# r8 arm {arm}: {what}\n")
        path = Path(out) / f"{prefix}-{arm}.yaml"
        path.write_text(head + body)
        print(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kind", choices=("tp1", "tp2"))
    ap.add_argument("--base", required=True)
    ap.add_argument("--image", help="tp1: container tag of the r8 TP=1 screening image")
    ap.add_argument("--out", default=AD)
    ap.add_argument("--only", help="comma list: write only these arms")
    a = ap.parse_args()
    if a.kind == "tp1" and not a.image:
        ap.error("tp1 needs --image")
    make(a.kind, a.base, a.out, a.image, a.only.split(",") if a.only else None)


if __name__ == "__main__":
    main()
