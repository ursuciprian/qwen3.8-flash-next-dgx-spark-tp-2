#!/usr/bin/env python3
"""k123an.py (2026-11-14): per-rank op-class split of one cold 16K prefill on the 2x, from rank-local traces of
mods/vllm-decode-profiler. Reuses k81an.py (the 1x k81 prefill analysis, copied unchanged next to this file).

k81an needs vLLM's `execute_context_*` step annotation, which only exists under --profiler-config (not usable on the
TP=2 pair: follower-node collective_rpc assert). The mod trace has none, so this wrapper adds one synthetic
annotation spanning the whole window (the profiler window is set to the prefill's steps, VLLM_LOCAL_PROF_STEPS), and
classes kernels launched under an op whose name contains "mtp" as MTP drafter. Drafter work outside such ops lands in
the main classes. ponytail: one window = one step, so no per-chunk split; add one when the mod logs step bounds.

  k123an.py report RES      k123.txt body: per rank, op classes over the window, us/token, top kernels, coverage
  k123an.py selftest
"""
import glob, gzip, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k81an  # noqa: E402

_classify = k81an.classify


def classify(kname, chain):
    if any("mtp" in c.lower() for c in chain):
        return "MTP drafter (ops named mtp only)"
    return _classify(kname, chain)


k81an.classify = classify


def inject(src, tokens, dst):
    """copy of the trace with one execute_context_1(<tokens>) annotation over the window on every launching thread"""
    ev = k81an.load(src)
    x = [e for e in ev if e.get("ph") == "X" and "ts" in e]
    t0, t1 = min(e["ts"] for e in x), max(e["ts"] + (e.get("dur") or 0) for e in x)
    tids = {(e.get("pid"), e.get("tid")) for e in x if e.get("cat") in ("cuda_runtime", "cuda_driver")}
    ev = ev + [{"ph": "X", "cat": "user_annotation", "name": f"execute_context_1({tokens})_generation_0(0)",
                "ts": t0 - 1, "dur": t1 - t0 + 2, "pid": p, "tid": t} for p, t in tids]
    json.dump({"traceEvents": ev}, open(dst, "w"))
    return dst


def cold(res, name):
    try:
        return json.load(open(os.path.join(res, name)))["summary"]["16384"]
    except (OSError, ValueError, KeyError):
        return None


def report(res):
    out = []
    ttft, prof = cold(res, "cold-ttft.json"), cold(res, "cold-prof.json")
    tokens = int(prof["prompt_tokens"]) if prof else 16384
    out.append("== cold 16K TTFT (s): unprofiled " + (f"{ttft['ttft_mean_s']:.3f} mean of {ttft['n']}" if ttft else "n/a")
               + ", profiled request " + (f"{prof['ttft_mean_s']:.3f}" if prof else "n/a") + f", prompt tokens {tokens}")
    for r in (0, 1):
        t = sorted(glob.glob(os.path.join(res, "prof", f"rank{r}.json*")))
        if not t:
            out.append(f"\n== rank {r}: no trace"); continue
        src = t[0]
        if src.endswith(".gz"):
            plain = src[:-3]
            open(plain, "wb").write(gzip.open(src).read()); src = plain
        steps = k81an.analyze(inject(src, tokens, os.path.join(res, "prof", f"rank{r}-k123.json")))
        out.append(""); out.append(k81an.fmt_arm(f"rank {r} (one window = the prefill steps)", steps).rstrip())
        busy = sum(sum(s["cls"].values()) for s in steps) / 1e6
        if ttft:
            cov = busy / ttft["ttft_mean_s"]
            out.append(f"coverage: GPU busy {busy:.3f} s = {cov:.2f} x the unprofiled TTFT"
                       + ("  (below 0.85: the window may have missed a chunk)" if cov < 0.85 else ""))
        out.append(f"\n== rank {r}: top kernels"); out.append(k81an.top_kernels(steps, 25).rstrip())
    return "\n".join(out) + "\n"


def selftest():
    import tempfile
    d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "prof"))

    def k(name, ts, dur, c):
        return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur, "pid": 0, "tid": 9, "args": {"correlation": c}}

    def r(ts, c):
        return {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": ts, "dur": 1, "pid": 0, "tid": 1, "args": {"correlation": c}}

    def op(name, ts, dur):
        return {"ph": "X", "cat": "cpu_op", "name": name, "ts": ts, "dur": dur, "pid": 0, "tid": 1, "args": {}}
    ev = [op("b12x::tp_moe_dynamic_launch", 10, 5), r(11, 1), k("moe", 100, 400, 1),
          op("vllm::qwen_gdn_attention_core_fused", 20, 5), r(21, 2), k("conv", 600, 100, 2),
          op("vllm::qwen3_8_flash_next_mtp_feedback", 30, 5), r(31, 3), k("mtpk", 800, 50, 3)]
    for rk in (0, 1):
        with gzip.open(os.path.join(d, "prof", f"rank{rk}.json.gz"), "wt") as f:
            json.dump({"traceEvents": ev}, f)
    json.dump({"summary": {"16384": {"n": 3, "prompt_tokens": 16384, "ttft_mean_s": 0.0006}}}, open(os.path.join(d, "cold-ttft.json"), "w"))
    json.dump({"summary": {"16384": {"n": 1, "prompt_tokens": 16384, "ttft_mean_s": 0.0007}}}, open(os.path.join(d, "cold-prof.json"), "w"))
    txt = report(d)
    assert txt.count("1 prefill chunks (16384 tokens)") == 2, txt
    assert "routed MoE (b12x NVFP4 W4A4)" in txt and "GDN core" in txt and "MTP drafter (ops named mtp only)" in txt, txt
    assert "coverage: GPU busy 0.001 s" in txt and "missed a chunk" not in txt, txt
    print("k123an selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["report"] and len(a) == 2:
        print(report(a[1]), end="")
    else:
        sys.exit(__doc__)
