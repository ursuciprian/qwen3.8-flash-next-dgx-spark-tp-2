#!/usr/bin/env python3
"""k123an.py (2026-11-14): per-rank op-class split of one cold 16K prefill on the 2x, from rank-local traces of
mods/vllm-decode-profiler. Reuses k81an.py (the 1x k81 prefill analysis, copied unchanged next to this file).

k81an needs vLLM's `execute_context_*` step annotation, which only exists under --profiler-config (not usable on the
TP=2 pair: follower-node collective_rpc assert). The mod marks every execute_model call of its window with a
`local_prof_step_<n>` range, so this wrapper adds one `execute_context_1(<tokens>)` annotation per marker (same thread
and span): one k81an step per execute_model call. The 16K prompt's steps are the leading run of heavy steps (GPU busy
>= HEAVY of the heaviest step); the later light steps are the flush request and are counted, not split. Each 16K step
is given its chunk's tokens (CHUNK = max_num_batched_tokens, the last one the rest). Kernels launched under an op whose
name contains "mtp" are classed as MTP drafter; drafter work outside such ops lands in the main classes.
A trace without markers (an older mod) falls back to one synthetic step over the whole window and says so.
Coverage = GPU busy of the 16K steps / the unprofiled TTFT: below LOW the window may have missed a chunk, above HIGH
it holds work that is not the 16K prefill. Both are flagged; k123 is DONE only when both ranks are inside.

  k123an.py report RES      k123.txt body: per rank, window steps, op classes per chunk, us/token, top kernels, coverage
  k123an.py selftest
"""
import glob, gzip, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k81an  # noqa: E402

_classify = k81an.classify
HEAVY, CHUNK, LOW, HIGH = 0.10, 8192, 0.85, 1.10
MARK = re.compile(r"^local_prof_step_(\d+)$")


def classify(kname, chain):
    if any("mtp" in c.lower() for c in chain):
        return "MTP drafter (ops named mtp only)"
    return _classify(kname, chain)


k81an.classify = classify


def inject(src, dst):
    """copy of the trace with an execute_context_1(1) annotation per window step marker (or one over the whole
    window when the trace has no markers) -> (dst, number of markers)"""
    ev = k81an.load(src)
    marks = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") == "user_annotation" and MARK.match(e.get("name", ""))),
                   key=lambda e: e["ts"])
    if marks:
        add = [{"ph": "X", "cat": "user_annotation", "name": "execute_context_1(1)_generation_0(0)",
                "ts": m["ts"], "dur": m["dur"], "pid": m.get("pid"), "tid": m["tid"]} for m in marks]
    else:
        x = [e for e in ev if e.get("ph") == "X" and "ts" in e]
        t0, t1 = min(e["ts"] for e in x), max(e["ts"] + (e.get("dur") or 0) for e in x)
        tids = {(e.get("pid"), e.get("tid")) for e in x if e.get("cat") in ("cuda_runtime", "cuda_driver")}
        add = [{"ph": "X", "cat": "user_annotation", "name": "execute_context_1(1)_generation_0(0)",
                "ts": t0 - 1, "dur": t1 - t0 + 2, "pid": p, "tid": t} for p, t in tids]
    json.dump({"traceEvents": ev + add}, open(dst, "w"))
    return dst, len(marks)


def split(steps, tokens):
    """leading heavy steps = the 16K prompt; tokens assigned per chunk -> (prefill steps, other steps)"""
    if not steps:
        return [], []
    top = max(s["busy_us"] for s in steps)
    n = 0
    while n < len(steps) and steps[n]["busy_us"] >= HEAVY * top:
        n += 1
    pf, left = steps[:n], tokens
    for i, s in enumerate(pf):
        s["ctx"] = left if i == len(pf) - 1 else min(CHUNK, left); left -= s["ctx"]
    for s in steps[n:]:
        s["ctx"] = 0
    return pf, steps[n:]


def ms(steps):
    return ", ".join("%.1f" % (s["busy_us"] / 1000) for s in steps)


def cold(res, name):
    try:
        return json.load(open(os.path.join(res, name)))["summary"]["16384"]
    except (OSError, ValueError, KeyError):
        return None


def report(res):
    out, verdicts = [], []
    ttft, prof = cold(res, "cold-ttft.json"), cold(res, "cold-prof.json")
    tokens = int(prof["prompt_tokens"]) if prof else 16384
    out.append("== cold 16K TTFT (s): unprofiled " + (f"{ttft['ttft_mean_s']:.3f} mean of {ttft['n']}" if ttft else "n/a")
               + ", profiled request " + (f"{prof['ttft_mean_s']:.3f}" if prof else "n/a") + f", prompt tokens {tokens}")
    for r in (0, 1):
        t = sorted(glob.glob(os.path.join(res, "prof", f"rank{r}.json*")))
        if not t:
            out.append(f"\n== rank {r}: no trace"); verdicts.append("MISSING"); continue
        src = t[0]
        if src.endswith(".gz"):
            plain = src[:-3]
            open(plain, "wb").write(gzip.open(src).read()); src = plain
        dst, nmark = inject(src, os.path.join(res, "prof", f"rank{r}-k123.json"))
        steps = k81an.analyze(dst)
        pf, other = split(steps, tokens)
        out.append("")
        out.append(f"== rank {r}: window {nmark} marked steps" + ("" if nmark else " (no step markers: one synthetic step)")
                   + f"; 16K prefill steps {len(pf)} (GPU busy ms {ms(pf)})"
                   + f"; other steps {len(other)} (flush, busy ms {ms(other) or '-'})")
        out.append(k81an.fmt_arm(f"rank {r} (16K prefill steps of the window)", pf).rstrip())
        busy = sum(sum(s["cls"].values()) for s in pf) / 1e6
        if ttft:
            cov = busy / ttft["ttft_mean_s"]
            v = "LOW" if cov < LOW else "HIGH" if cov > HIGH else "OK"
            out.append(f"coverage rank {r}: GPU busy {busy:.3f} s = {cov:.2f} x the unprofiled TTFT {v}"
                       + {"LOW": f" (below {LOW}: the window may have missed a chunk)",
                          "HIGH": f" (above {HIGH}: work that is not the 16K prefill is in the split)", "OK": ""}[v])
            verdicts.append(v)
        else:
            out.append(f"coverage rank {r}: n/a (no unprofiled TTFT)"); verdicts.append("MISSING")
        out.append(f"\n== rank {r}: top kernels"); out.append(k81an.top_kernels(pf, 25).rstrip())
    out.append(f"COVERAGE={'OK' if verdicts == ['OK', 'OK'] else ','.join(verdicts)}")
    return "\n".join(out) + "\n"


def selftest():
    import tempfile

    def k(name, ts, dur, c):
        return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur, "pid": 0, "tid": 9, "args": {"correlation": c}}

    def r(ts, c):
        return {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": ts, "dur": 1, "pid": 0, "tid": 1, "args": {"correlation": c}}

    def op(name, ts, dur, cat="cpu_op"):
        return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "pid": 0, "tid": 1, "args": {}}

    def trace(marks=True):
        ev, c = [], 0
        # steps 0, 1: the 16K (two heavy chunks); steps 2..4: flush (light)
        for i, (moe, gdn) in enumerate(((240000, 60000), (240000, 60000), (300, 100), (300, 100), (300, 100))):
            b = i * 1_000_000
            if marks:
                ev.append(op(f"local_prof_step_{i}", b, 900, "user_annotation"))
            c += 1; ev += [op("b12x::tp_moe_dynamic_launch", b + 10, 5), r(b + 11, c), k("moe", b + 100, moe, c)]
            c += 1; ev += [op("vllm::qwen_gdn_attention_core_fused", b + 20, 5), r(b + 21, c), k("conv", b + 100 + moe, gdn, c)]
            c += 1; ev += [op("vllm::qwen3_8_flash_next_mtp_feedback", b + 30, 5), r(b + 31, c), k("mtpk", b + 100 + moe + gdn, 50, c)]
        return ev

    def run(ttft_s, marks=True):
        d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "prof"))
        for rk in (0, 1):
            with gzip.open(os.path.join(d, "prof", f"rank{rk}.json.gz"), "wt") as f:
                json.dump({"traceEvents": trace(marks)}, f)
        json.dump({"summary": {"16384": {"n": 3, "prompt_tokens": 16384, "ttft_mean_s": ttft_s}}}, open(os.path.join(d, "cold-ttft.json"), "w"))
        json.dump({"summary": {"16384": {"n": 1, "prompt_tokens": 16384, "ttft_mean_s": ttft_s}}}, open(os.path.join(d, "cold-prof.json"), "w"))
        return report(d)

    txt = run(0.65)   # 16K busy = 2 x 0.30005 s = 0.6001 s -> 0.92 x
    assert txt.count("window 5 marked steps; 16K prefill steps 2") == 2, txt
    assert txt.count("other steps 3 (flush") == 2 and txt.count("2 prefill chunks (8192, 8192 tokens)") == 2, txt
    assert "routed MoE (b12x NVFP4 W4A4)" in txt and "GDN core" in txt and "MTP drafter (ops named mtp only)" in txt, txt
    assert "= 0.92 x the unprofiled TTFT OK" in txt and "COVERAGE=OK" in txt, txt
    txt = run(0.50)   # 1.20 x: foreign work in the split
    assert "1.20 x the unprofiled TTFT HIGH" in txt and "COVERAGE=HIGH,HIGH" in txt, txt
    txt = run(0.80)   # 0.75 x: a missed chunk
    assert "COVERAGE=LOW,LOW" in txt, txt
    txt = run(0.65, marks=False)   # older mod: one synthetic step over the window (flush included)
    assert "no step markers" in txt and "16K prefill steps 1" in txt, txt
    print("k123an selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["report"] and len(a) == 2:
        print(report(a[1]), end="")
    else:
        sys.exit(__doc__)
