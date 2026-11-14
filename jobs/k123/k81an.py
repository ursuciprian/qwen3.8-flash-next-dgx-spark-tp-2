#!/usr/bin/env python3
"""k81an.py (2026-10-11, 1x #27): op-class breakdown of vLLM torch-profiler traces of one cold 16K prefill per arm.

  k81an.py trace FILE [tokens]   breakdown of one trace (debug)
  k81an.py report RES            k81.txt (all arms under RES: <unit>/cold-ttft.json, <unit>/prof-k81-*/**/*.json.gz)
  k81an.py selftest

Kernel -> op class by the CPU ops that enclose its launch (cuda_runtime event, same thread, by correlation id):
main-model work is launched inside the `execute_context_*` step annotation; GPU work launched after it in the same step
(outside the annotation) through model ops is the MTP drafter, the rest is sampler/bookkeeping. Dense b12x GEMMs that
are not under the GDN input-projection op are named by the last layer core seen on the GPU timeline (GDN core ->
GDN out_proj, QSA attention -> attention o_proj). GPU idle = step span minus the union of kernel intervals.
ponytail: keyword table over op names; extend CLASSES when a new op family lands in "other".
"""
import glob, gzip, json, os, re, statistics, sys
from collections import defaultdict

# (class, needle in the enclosing-op chain or kernel name); first match wins
CLASSES = [
    ("routed MoE (b12x NVFP4 W4A4)", "tp_moe_dynamic_launch"),
    ("shared expert + router", "moe_forward_shared"),
    ("dense: GDN in_proj (qkvz+ba)", "qwen_gdn_input_projections"),
    ("dense: b12x blockscaled", "b12x_blockscaled_linear"),
    ("GDN core (conv, chunk delta, norm)", "qwen_gdn_attention_core"),
    ("dense: attention qkv proj", "qsa_project_inputs"),
    ("attention (QSA indexer + sparse)", "qsa_"),
    ("PLE embedding", "flash_next_ple"),
    ("norms / hyperconnection", "hyperconnection"),
    ("norms / hyperconnection", "rmsnorm"),
]
PLE_KERNELS = ("_nvfp4_lookup_kernel", "_hash_ids_kernel", "_request_ids_kernel")
GEMM_KERNELS = ("nvjet", "gemm", "gemv", "cutlass_80")
# analytic forward GFLOP per prompt token (2 x active params; config: hidden 2560, 48 layers = 36 GDN + 12 QSA,
# 512 experts top-10, expert/shared intermediate 640, GDN 16 k-heads + 48 v-heads x 128, QSA 24 q-heads x 256 + gate)
GFLOP_PER_TOKEN = {
    "routed MoE (b12x NVFP4 W4A4)": 2 * 10 * 3 * 2560 * 640 * 48 / 1e9,
    "shared expert + router": 2 * (3 * 2560 * 640 + 2560 * 512) * 48 / 1e9,
    "dense: GDN in_proj (qkvz+ba)": 2 * 2560 * (2048 + 2048 + 6144 + 6144 + 96) * 36 / 1e9,
    "dense: GDN out_proj": 2 * 6144 * 2560 * 36 / 1e9,
    "dense: attention o_proj": 2 * 6144 * 2560 * 12 / 1e9,
}


def load(path):
    with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
        d = json.load(f)
    return d.get("traceEvents", d) if isinstance(d, dict) else d


def ctx_tokens(name):
    m = re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", name)
    return (int(m.group(2)), int(m.group(4))) if m else None


def classify(kname, chain):
    c = " > ".join(chain)
    main = "execute_context" in c
    if not main:
        if any(s in c for s in ("vllm::", "b12x::", "CompiledFxGraph", "Torch-Compiled")):
            return "MTP drafter (prefill)"
        return "sampler / bookkeeping"
    for cls, needle in CLASSES:
        if needle in c:
            return cls
    if kname.startswith(PLE_KERNELS):
        return "PLE embedding"
    if "hyperconnection" in kname or "norm" in kname:
        return "norms / hyperconnection"
    if "CompiledFxGraph" in c:
        return "dense: BF16 cuBLAS in compiled graph" if any(g in kname for g in GEMM_KERNELS) else "fused elementwise (inductor)"
    return "step setup (inputs, KV, metadata)"


def union(iv):
    tot, cur = 0.0, None
    for s, e in sorted(iv):
        if cur is None or s > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    return tot + (cur[1] - cur[0] if cur else 0.0)


def analyze(path):
    ev = [e for e in load(path) if e.get("ph") == "X"]
    rt = {e["args"]["correlation"]: e for e in ev
          if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    cpu = defaultdict(list)
    for e in ev:
        if e.get("cat") in ("cpu_op", "user_annotation"):
            cpu[e["tid"]].append(e)
    for t in cpu:
        cpu[t].sort(key=lambda e: e["ts"])
    steps = sorted((e for e in ev if e.get("cat") == "user_annotation" and ctx_tokens(e["name"])), key=lambda e: e["ts"])
    kern = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])

    def chain(k):  # enclosing CPU ops of the launch, outermost first, with the innermost op's input dims
        r = rt.get(k.get("args", {}).get("correlation"))
        if r is None:
            return [], r, None, None
        enc = [c for c in cpu[r["tid"]] if c["ts"] <= r["ts"] <= c["ts"] + c["dur"]]
        dims = next((c["args"]["Input Dims"] for c in reversed(enc) if c.get("args", {}).get("Input Dims")), None)
        bl = next((c["ts"] for c in reversed(enc) if c["name"] == "vllm::b12x_blockscaled_linear"), None)
        return [c["name"] for c in enc], r, dims, bl

    out = []  # one dict per step
    st = {i: {"name": s["name"], "tokens": ctx_tokens(s["name"]), "cls": defaultdict(float), "iv": [], "kern": defaultdict(lambda: [0, 0.0, None])}
          for i, s in enumerate(steps)}
    starts = [s["ts"] for s in steps]
    last, owner = None, {}
    for k in kern:
        ch, r, dims, bl = chain(k)
        if r is None:
            continue
        i = max((j for j, t in enumerate(starts) if t <= r["ts"]), default=None)
        if i is None:
            continue
        cls = classify(k["name"], ch)
        if cls.startswith("GDN core"):
            last = "gdn"
        elif cls.startswith("attention ("):
            last = "attn"
        elif cls.startswith("routed MoE"):
            last = None
        elif cls == "dense: b12x blockscaled":
            # the first blockscaled op after a layer core is its output projection (all kernels of that op)
            if bl not in owner:
                owner[bl] = {"gdn": "dense: GDN out_proj", "attn": "dense: attention o_proj"}.get(last, "dense: b12x other")
                last = None
            cls = owner[bl]
        s = st[i]; d = k.get("dur", 0) or 0
        s["cls"][cls] += d; s["iv"].append((k["ts"], k["ts"] + d))
        kk = s["kern"][(cls, k["name"][:90])]; kk[0] += 1; kk[1] += d; kk[2] = kk[2] or dims
    for i in sorted(st):
        s = st[i]
        if not s["iv"]:
            continue
        span = max(e for _, e in s["iv"]) - min(b for b, _ in s["iv"])
        busy = union(s["iv"])
        out.append({"name": s["name"], "ctx": s["tokens"][0], "gen": s["tokens"][1], "cls": dict(s["cls"]),
                    "span_us": span, "busy_us": busy, "idle_us": span - busy,
                    "t0": min(b for b, _ in s["iv"]), "t1": max(e for _, e in s["iv"]),
                    "kern": {f"{c}|{n}": v for (c, n), v in s["kern"].items()}})
    return out


def prefill_steps(steps):
    return [s for s in steps if s["ctx"] > 0]


def arm_table(steps):
    """rows (class, [ms per chunk], us/token, % of busy) over the prefill steps, plus totals."""
    pf = prefill_steps(steps)
    toks = sum(s["ctx"] for s in pf) or 1
    classes = sorted({c for s in pf for c in s["cls"]}, key=lambda c: -sum(s["cls"].get(c, 0) for s in pf))
    busy = sum(sum(s["cls"].values()) for s in pf) or 1
    rows = [(c, [s["cls"].get(c, 0) / 1000 for s in pf], sum(s["cls"].get(c, 0) for s in pf) / toks,
             100 * sum(s["cls"].get(c, 0) for s in pf) / busy) for c in classes]
    idle = [s["idle_us"] / 1000 for s in pf]
    gaps = [(b["t0"] - a["t1"]) / 1000 for a, b in zip(pf, pf[1:])]
    return pf, toks, rows, idle, gaps


def fmt_arm(label, steps):
    pf, toks, rows, idle, gaps = arm_table(steps)
    if not pf:
        return f"== {label}: no prefill steps in the trace\n"
    L = [f"== {label}: {len(pf)} prefill chunks ({', '.join(str(s['ctx']) for s in pf)} tokens), GPU time per op class",
         f"{'op class':44s} " + " ".join(f"{'chunk' + str(i + 1) + ' ms':>11s}" for i in range(len(pf))) + f" {'us/token':>9s} {'% busy':>7s} {'TFLOPS':>7s}"]
    for c, ms, upt, pct in rows:
        base = re.sub(r" \[.*\]$", "", c)
        fl = GFLOP_PER_TOKEN.get(base)
        tf = f"{fl * 1e9 / (upt * 1e-6) / 1e12:7.1f}" if fl and upt else f"{'':7s}"
        L.append(f"{c[:44]:44s} " + " ".join(f"{m:11.1f}" for m in ms) + f" {upt:9.1f} {pct:6.1f}% {tf}")
    tb = sum(sum(s["cls"].values()) for s in pf) / toks
    L.append(f"{'GPU busy (sum of kernels)':44s} " + " ".join(f"{sum(s['cls'].values()) / 1000:11.1f}" for s in pf) + f" {tb:9.1f}")
    L.append(f"{'GPU idle inside the step':44s} " + " ".join(f"{x:11.1f}" for x in idle) + f" {sum(idle) * 1000 / toks:9.1f}")
    L.append(f"{'step span (first kernel to last)':44s} " + " ".join(f"{s['span_us'] / 1000:11.1f}" for s in pf) + f" {sum(s['span_us'] for s in pf) / toks:9.1f}")
    if gaps:
        L.append(f"GPU idle between prefill chunks (CPU scheduling under the profiler): {', '.join(f'{g:.1f}' for g in gaps)} ms")
    return "\n".join(L) + "\n"


def top_kernels(steps, n=20):
    pf = prefill_steps(steps)
    agg = defaultdict(lambda: [0, 0.0, None])
    for s in pf:
        for k, (cnt, us, dims) in s["kern"].items():
            a = agg[k]; a[0] += cnt; a[1] += us; a[2] = a[2] or dims
    L = [f"{'ms':>8s} {'calls':>6s}  class | kernel | input dims of the innermost op"]
    for k, (cnt, us, dims) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:n]:
        c, name = k.split("|", 1)
        L.append(f"{us / 1000:8.1f} {cnt:6d}  {c} | {name[:70]} | {json.dumps(dims)[:120] if dims else ''}")
    return "\n".join(L) + "\n"


# ---------------- RES-level report ----------------
def units(res):
    return sorted(d for d in glob.glob(os.path.join(res, "d0?-*")) if os.path.isdir(d) and "prebake" not in d)


def trace_of(u):
    t = sorted(glob.glob(os.path.join(u, "prof-k81-*", "**", "*.json*"), recursive=True))
    return t[0] if t else None


def ttft_of(u):
    try:
        s = json.load(open(os.path.join(u, "cold-ttft.json")))
        return s["summary"]["16384"], [r["ttft_s"] for r in s["rows"] if r.get("ttft_s")]
    except (OSError, ValueError, KeyError):
        return None, []


def facts(u):
    try:
        return open(os.path.join(u, "facts.txt")).read().strip()
    except OSError:
        return ""


def class_us(steps):
    pf, toks, rows, idle, gaps = arm_table(steps)
    d = {c: upt for c, _, upt, _ in rows}
    d["GPU busy"] = sum(sum(s["cls"].values()) for s in pf) / toks
    d["GPU idle in steps"] = sum(idle) * 1000 / toks
    return d


def compare(cols, data):
    """cols [(label, unit)], data {unit: class->us/token} -> text table with deltas vs the first column."""
    keys = sorted({k for u in data.values() for k in u}, key=lambda k: -max(u.get(k, 0) for u in data.values()))
    keys = [k for k in keys if k not in ("GPU busy", "GPU idle in steps")] + ["GPU busy", "GPU idle in steps"]
    L = [f"{'us/token':44s} " + " ".join(f"{l[:14]:>14s}" for l, _ in cols) + "   " + " ".join(f"{'d ' + l[:12]:>14s}" for l, _ in cols[1:])]
    b = data[cols[0][1]]
    for k in keys:
        v = [data[u].get(k, 0) for _, u in cols]
        L.append(f"{k[:44]:44s} " + " ".join(f"{x:14.1f}" for x in v) + "   " + " ".join(f"{x - v[0]:+14.1f}" for x in v[1:]))
    return "\n".join(L) + "\n"


def report(res, md=False):
    us = units(res); an = {}; out = []
    out.append("== TTFT, one cold 16K prompt at c1 (3 unprofiled requests per boot, prompt tokens / mean TTFT)")
    out.append(f"{'unit':16s} {'TTFT s':>8s} {'tok/s':>7s} {'us/token':>9s}  each TTFT s | boot facts")
    for u in us:
        s, each = ttft_of(u)
        n = os.path.basename(u)
        if s and s.get("n"):
            out.append(f"{n:16s} {s['ttft_mean_s']:8.3f} {s['tok_s']:7.0f} {1e6 / s['tok_s']:9.1f}  {', '.join(f'{x:.2f}' for x in each)} | {facts(u)}")
        else:
            out.append(f"{n:16s} {'n/a':>8s} | {facts(u)}")
        t = trace_of(u)
        if t:
            try:
                an[n] = analyze(t)
            except Exception as e:  # a broken trace is reported, the rest still runs
                out.append(f"  trace {t}: {e!r}"[:300])
    for n, steps in an.items():
        out.append(""); out.append(fmt_arm(n, steps).rstrip())
    data = {n: class_us(s) for n, s in an.items() if prefill_steps(s)}
    for title, cols in (("dgx-01: dense GEMM path and MTP (shipped 14 GiB pool; arms 6 GiB pool)",
                         [("ship41", "d01-ship"), ("min0", "d01-min0"), ("min1", "d01-min1"), ("mtp-off", "d01-nomtp")]),
                        ("dgx-02: b12x rebase image vs the shipped image (both 6 GiB pool, same Spark)",
                         [("ship41", "d02-ship6"), ("rebase", "d02-rb")]),
                        ("shipped image on both Sparks (dgx-01 14 GiB vs dgx-02 6 GiB pool)",
                         [("d01 ship", "d01-ship"), ("d02 ship6", "d02-ship6")])):
        cols = [(l, u) for l, u in cols if u in data]
        if len(cols) >= 2:
            out.append(""); out.append(f"== {title}"); out.append(compare(cols, data).rstrip())
    if "d01-ship" in an:
        out.append(""); out.append("== top kernels, d01-ship prefill chunks"); out.append(top_kernels(an["d01-ship"], 25).rstrip())
    if "d02-rb" in an:
        out.append(""); out.append("== top kernels, d02-rb prefill chunks"); out.append(top_kernels(an["d02-rb"], 15).rstrip())
    txt = "\n".join(out) + "\n"
    return txt


def selftest():
    def k(name, ts, dur, corr):
        return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur, "args": {"correlation": corr}}

    def r(ts, corr):
        return {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": ts, "dur": 1, "tid": 1, "args": {"correlation": corr}}

    def op(name, ts, dur):
        return {"ph": "X", "cat": "cpu_op", "name": name, "ts": ts, "dur": dur, "tid": 1, "args": {}}
    ev = [{"ph": "X", "cat": "user_annotation", "name": "execute_context_1(8192)_generation_0(0)", "ts": 0, "dur": 50, "tid": 1},
          op("vllm::qwen_gdn_attention_core_fused_norm_packed", 1, 5), r(2, 1), k("conv", 100, 10, 1),
          op("vllm::b12x_blockscaled_linear", 6, 3), r(7, 2), k("gemm_f8E4M3FN", 110, 20, 2),
          op("vllm::moe_forward_shared", 10, 10), op("b12x::tp_moe_dynamic_launch", 11, 3), r(12, 3), k("moe", 140, 100, 3),
          op("vllm::qwen3_8_flash_next_qsa_run_projected", 21, 3), r(22, 4), k("attn", 240, 30, 4),
          op("vllm::b12x_blockscaled_linear", 25, 3), r(26, 5), k("gemm_nvf4", 270, 10, 5),
          op("vllm::qwen3_8_flash_next_mtp_feedback", 60, 3), r(61, 6), k("mtpk", 290, 5, 6),
          {"ph": "X", "cat": "user_annotation", "name": "execute_context_1(100)_generation_0(0)", "ts": 100, "dur": 10, "tid": 1},
          op("vllm::moe_forward_shared", 101, 5), r(102, 7), k("nvjet_x", 400, 8, 7)]
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "t.json"); json.dump({"traceEvents": ev}, open(p, "w"))
    s = analyze(p)
    assert [x["ctx"] for x in s] == [8192, 100], s
    c = s[0]["cls"]
    assert c["GDN core (conv, chunk delta, norm)"] == 10 and c["dense: GDN out_proj"] == 20, c
    assert c["routed MoE (b12x NVFP4 W4A4)"] == 100 and c["attention (QSA indexer + sparse)"] == 30, c
    assert c["dense: attention o_proj"] == 10 and c["MTP drafter (prefill)"] == 5, c
    assert s[0]["span_us"] == 195 and s[0]["idle_us"] == 195 - 175, s[0]
    assert s[1]["cls"] == {"shared expert + router": 8}
    pf, toks, rows, idle, gaps = arm_table(s)
    assert toks == 8292 and abs(gaps[0] - (400 - 295) / 1000) < 1e-9
    assert fmt_arm("x", s) and top_kernels(s)
    print("k81an selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["trace"]:
        s = analyze(a[1]); print(fmt_arm(os.path.basename(a[1]), s)); print(top_kernels(s, 40))
    elif a[:1] == ["report"]:
        print(report(a[1]), end="")
    else:
        sys.exit(__doc__)
