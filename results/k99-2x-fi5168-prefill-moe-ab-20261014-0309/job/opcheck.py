#!/usr/bin/env python3
"""k99 op check (2026-10-13): the k99 mod's FlashInfer call (mod/k99fi.py) on b12x-prepared I320 layers, before any boot.

Runs in a throwaway container of the shipped 2x image (vLLM d21d7ade, b12x 21e0b201, flashinfer 0.7.0, cutlass-dsl
4.7.0 as shipped), GPU, --network none, B12X_DYNAMIC_TILE_MN=32x128 (k98's b12x-m32 serving anchor). Layers 5 and 47
of the served checkpoint, rank-0 I320 shard (k98's moe98.load_layer/shard), real routing from the k97 capture.
  sampler   the image's flashinfer 0.7.0 top-k/top-p sampler is called before and after the 5c4058bd tree is loaded
            and used in the same process (k99fi keeps the two apart): it must work both times
  order     b12x weights are prepared (vLLM [gate; up] = W31) without a b12x run; the FlashInfer path runs first (as
            vLLM's profile run at 8192 rows will), then b12x plans and runs: its error must match k98's b12x-m32,
            i.e. the one-time [up; gate] flip happened exactly once (whether b12x's weight preparation or the first
            FlashInfer call did it is reported)
  bytes     after the flip, b12x's storage equals what k98 handed FlashInfer: FC1 [up; gate] packed weights and
            moe98.swizzle_128x4 block scales (FC1 and FC2); FlashInfer's operands are views of it (same addresses)
  numerics  M 2864 / 5728 / 6048 / 9072, rows 0..255 vs the FP32 reference: err <= 1.10 x err(b12x) + 0.005,
            difference to b12x <= 0.5 x err(b12x); at M 6048 err within 0.001 of k98 fi5168 tp2 real
  speed     layer 5, CUDA events, L2 flush, median of 15 after 3 warmups (moe98.timed), FlashInfer through k99fi.run
            (module swap included): within +-7% of k98 at M 6048 and 9072 (fi5168 7.225 / 9.717 ms, b12x-m32
            8.538 / 11.822 ms); M 5728 (the 2x's large chunk: 2 x 2864, attention block 2864) and 2864 reported
  memory    GPU memory FlashInfer keeps per layer (allocated delta around the first call of layer 47; layer 5's
            delta also holds the shared workspace), FlashInfer's module caches by name; padded FP4 weight copies
            (~0.75 GB per layer) are a FAIL; per layer above MEM_MAX_MIB is a FAIL
Usage: opcheck.py run /out --ckpt C --cap-dir /cap --fi-src /fi | selftest (stdlib)
Output: /out/opcheck.tsv, stdout report ending in 'OPCHECK PASS ...' or 'OPCHECK FAIL: ...'.
"""
import dataclasses
import os
import sys
import time
import traceback

I, LAYERS_SERVED, TIME_LAYER = 320, 49, 5   # rank shard; 48 decoder MoE layers + the MTP drafter's
MS = (2864, 5728, 6048, 9072)
K98 = {"err": {5: 0.1137, 47: 0.1193}, "fi": {6048: 7.225, 9072: 9.717}, "b12x": {6048: 8.538, 9072: 11.822},
       "b12x_err": {5: 0.1138, 47: 0.1193}}
TOL_MS, TOL_ERR, MEM_MAX_MIB = 0.07, 0.001, 200
CHUNKS_5728 = {"16K": 2, "64K": 11, "128K": 22}   # 5728-row chunks per cold prompt at budget 8192 on the 2x


def sampler():
    import torch
    import flashinfer
    try:
        g = torch.Generator(device="cuda").manual_seed(99)
        logits = torch.randn(4, 4096, device="cuda", generator=g)
        out = flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, torch.full((4,), 20, device="cuda", dtype=torch.int32),
                                                                   torch.full((4,), 0.95, device="cuda"))
        torch.cuda.synchronize()
        ok = out.shape[0] == 4 and int(out.min()) >= 0 and int(out.max()) < 4096
        return f"{'ok' if ok else 'BAD'} (flashinfer {flashinfer.__version__} from {flashinfer.__file__})"
    except Exception as e:
        return f"ERROR {type(e).__name__}: {str(e)[:200]}"


def fi_caches():
    """bytes held by the 5c4058bd MoE module caches, by cache name."""
    import torch
    import k99fi
    m = k99fi._S["mine"].get("flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch")
    out = {}
    for name in ("_WEIGHT_CACHE", "_PADDED_SCALE_CACHE", "_PADDED_FP4_CACHE", "_WORKSPACE_CACHE", "_W4A16_WEIGHT_CACHE"):
        seen, ids, n = set(), set(), 0
        stack = list(getattr(m, name, {}).values()) if m else []
        while stack:
            v = stack.pop()
            if id(v) in ids:
                continue
            ids.add(id(v))
            if isinstance(v, torch.Tensor):
                if v.untyped_storage().data_ptr() not in seen:
                    seen.add(v.untyped_storage().data_ptr()); n += v.untyped_storage().nbytes()
            elif isinstance(v, (tuple, list)):
                stack += list(v)
            elif isinstance(v, dict):
                stack += list(v.values())
            elif dataclasses.is_dataclass(v) and not isinstance(v, type):
                stack += [getattr(v, f.name, None) for f in dataclasses.fields(v)]
        out[name] = n
    return out


def run(out_dir, ckpt, cap_dir, fi_src):
    import torch
    sys.path[:0] = ["/k98", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mod")]
    import moe98
    import k99fi
    from b12x.moe.fused_moe._impl import _ensure_w13_kernel_order_inplace  # noqa: F401 (k99fi needs it)
    rows, notes, fails = [], [], []
    s0 = sampler(); notes.append(f"sampler before: {s0}")
    t0 = time.time(); k99fi.load(fi_src); notes.append(f"5c4058bd tree loaded in {time.time() - t0:.1f} s: {k99fi._S['where']}")
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    mem = {}
    for layer in (5, 47):
        lw = moe98.shard(moe98.load_layer(ckpt, layer), I)
        x, ids, w, src = moe98.make_inputs(lw, "real", 0, cap_dir, layer)
        notes.append(f"layer {layer} inputs: {src}")
        ref = moe98.reference(lw, x, ids, w)
        plan = moe98.B12x._plan
        moe98.B12x._plan = lambda self, cap: self.plans.__setitem__(cap, None)   # weights only, no b12x run yet
        try:
            b = moe98.B12x(lw)
        finally:
            moe98.B12x._plan = plan
        impl = b.experts._impl
        u8 = lambda t: t.contiguous().view(torch.uint8).reshape(-1)
        w31 = torch.equal(u8(impl.w1_fp4), u8(torch.cat([lw["gw"], lw["uw"]], dim=1)))
        # FlashInfer first (vLLM's profile run): views + one call at M 5728, memory around it
        out = torch.empty(max(MS), moe98.H, dtype=torch.bfloat16, device="cuda")
        torch.cuda.synchronize(); m0 = torch.cuda.memory_allocated(); c0 = fi_caches()
        st = k99fi.views(impl, lw["gg"], lw["dg"], 1.0 / lw["gi"], 1.0 / lw["di"])
        k99fi.run(st, out[:5728], x[:5728], w[:5728], ids[:5728]); torch.cuda.synchronize()
        m1 = torch.cuda.memory_allocated(); c1 = fi_caches()
        mem[layer] = ((m1 - m0) / 2**20, {k: (c1[k] - c0.get(k, 0)) / 2**20 for k in c1})
        w13 = torch.equal(u8(impl.w1_fp4), u8(torch.cat([lw["uw"], lw["gw"]], dim=1)))
        s1 = torch.equal(u8(impl.w1_blockscale), u8(moe98.swizzle_128x4(torch.cat([lw["us"], lw["gs"]], dim=1))))
        s2 = torch.equal(u8(impl.w2_blockscale), u8(moe98.swizzle_128x4(lw["ds"])))
        zc = (st["w1"].data_ptr() == impl.w1_fp4.data_ptr() and st["w2"].data_ptr() == impl.w2_fp4.data_ptr()
              and st["s1"].data_ptr() == impl.w1_blockscale.data_ptr() and st["s2"].data_ptr() == impl.w2_blockscale.data_ptr())
        notes.append(f"layer {layer} bytes: still [gate; up] before the first FlashInfer call {w31} (False = b12x's weight "
                     f"preparation flipped it already), [up; gate] after it {w13}, FC1 scales = swizzle_128x4 {s1}, "
                     f"FC2 scales = swizzle_128x4 {s2}, FlashInfer operands are views {zc}")
        if not (w13 and s1 and s2 and zc):
            fails.append(f"layer {layer} storage/view check")
        # now b12x: plans + runs (its flip must be a no-op)
        for cap in (8192, max(MS)):
            plan(b, cap)
        for m in MS:
            of, ob = torch.empty(m, moe98.H, dtype=torch.bfloat16, device="cuda"), torch.empty(m, moe98.H, dtype=torch.bfloat16, device="cuda")
            ffi = lambda: k99fi.run(st, of, x[:m], w[:m], ids[:m])
            fb = b.fn(x[:m], ids[:m], w[:m], ob)
            ffi(); fb(); torch.cuda.synchronize()
            ef, eb = moe98.rel(of[:moe98.NUM_ROWS], ref), moe98.rel(ob[:moe98.NUM_ROWS], ref)
            d = moe98.rel(of[:moe98.NUM_ROWS], ob[:moe98.NUM_ROWS])
            tf = tb = float("nan")
            if layer == TIME_LAYER:
                tf, tb = moe98.timed(ffi, flush), moe98.timed(fb, flush)
            rows.append({"layer": layer, "m": m, "err_fi": ef, "err_b12x": eb, "diff": d, "ms_fi": tf, "ms_b12x": tb})
            print("\t".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rows[-1].items()), flush=True)
        del b, impl, st, lw, x, ids, w, ref
        torch.cuda.empty_cache()
    s1_ = sampler(); notes.append(f"sampler after: {s1_}")
    if s0.startswith("ok") and not s1_.startswith("ok"):
        fails.append("sampler broke after the 5c4058bd tree was used")
    if not s0.startswith("ok"):
        notes.append("sampler check n/a (the image's sampler call failed before the tree was loaded)")
    with open(os.path.join(out_dir, "opcheck.tsv"), "w") as f:
        f.write("layer\tm\terr_fi\terr_b12x\tdiff\tms_fi\tms_b12x\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in ("layer", "m", "err_fi", "err_b12x", "diff", "ms_fi", "ms_b12x")) + "\n")
    print(report(rows, mem, notes, fails))


def report(rows, mem, notes, fails):
    L, fails = list(notes), list(fails)
    by = {(r["layer"], r["m"]): r for r in rows}
    L.append("layer      M   err FI  err b12x  diff   ms FI  ms b12x   x")
    for r in rows:
        x = r["ms_b12x"] / r["ms_fi"] if r["ms_fi"] == r["ms_fi"] and r["ms_fi"] else float("nan")
        L.append(f"{r['layer']:5d} {r['m']:6d} {r['err_fi']:8.4f} {r['err_b12x']:9.4f} {r['diff']:5.3f} {r['ms_fi']:7.3f} {r['ms_b12x']:8.3f} x{x:.2f}")
        if not (r["err_fi"] <= 1.10 * r["err_b12x"] + 0.005 and r["diff"] <= 0.5 * r["err_b12x"]):
            fails.append(f"numerics L{r['layer']} M{r['m']}")
    for layer, e in K98["err"].items():
        r = by.get((layer, 6048))
        if not r or abs(r["err_fi"] - e) > TOL_ERR:
            fails.append(f"L{layer} M6048 FI err {r and round(r['err_fi'], 4)} vs k98 {e}")
        if not r or abs(r["err_b12x"] - K98["b12x_err"][layer]) > TOL_ERR:
            fails.append(f"L{layer} b12x err {r and round(r['err_b12x'], 4)} vs k98 {K98['b12x_err'][layer]} (flip order?)")
    for m in (6048, 9072):
        r = by.get((TIME_LAYER, m))
        for k in ("fi", "b12x"):
            v = r and r[f"ms_{k}"]
            if not v or abs(v / K98[k][m] - 1) > TOL_MS:
                fails.append(f"{k} {m}: {v and round(v, 3)} ms vs k98 {K98[k][m]} (> {TOL_MS:.0%})")
    r = by.get((TIME_LAYER, 5728))
    if r and r["ms_fi"] == r["ms_fi"]:
        d = r["ms_b12x"] - r["ms_fi"]
        L.append(f"projection at M 5728 (operator only): {d:+.3f} ms per layer x {LAYERS_SERVED} layers = {d * LAYERS_SERVED:+.1f} ms "
                 "per chunk; per cold prompt " + ", ".join(f"{k} {n} chunks {d * LAYERS_SERVED * n / 1000:+.2f} s" for k, n in CHUNKS_5728.items()))
    if mem:
        per = mem.get(47, mem.get(5))[0]
        ws = mem.get(5, (per,))[0] - per
        L.append(f"memory kept by FlashInfer: layer 5 {mem.get(5, (float('nan'),))[0]:.0f} MiB (with the shared workspace), layer 47 "
                 f"{per:.0f} MiB -> per rank ~{(per * LAYERS_SERVED + ws) / 1024:.2f} GiB ({LAYERS_SERVED} layers + workspace "
                 f"{ws:.0f} MiB at M 5728; grows with the largest M seen)")
        for layer, (_, c) in sorted(mem.items()):
            L.append(f"  layer {layer} FlashInfer caches (MiB): " + ", ".join(f"{k} {v:.0f}" for k, v in c.items()))
        if any(c.get("_PADDED_FP4_CACHE", 0) > 0 for _, c in mem.values()):
            fails.append("FlashInfer built padded FP4 weight copies")
        if per > MEM_MAX_MIB:
            fails.append(f"FlashInfer keeps {per:.0f} MiB per layer (> {MEM_MAX_MIB})")
    if len(rows) != 2 * len(MS):
        fails.append(f"{len(rows)} of {2 * len(MS)} cells")
    L.append("OPCHECK " + ("PASS" if not fails else "FAIL: " + "; ".join(fails)))
    return "\n".join(L)


def selftest():
    def rows(fi6048=7.2, err=0.1137):
        out = []
        for layer in (5, 47):
            for m in MS:
                out.append({"layer": layer, "m": m, "err_fi": err if layer == 5 else 0.1193, "err_b12x": K98["b12x_err"][layer],
                            "diff": 0.01, "ms_fi": (fi6048 if m == 6048 else K98["fi"].get(m, 7.0)) if layer == 5 else float("nan"),
                            "ms_b12x": K98["b12x"].get(m, 8.2) if layer == 5 else float("nan")})
        return out
    mem = {5: (300.0, {"_WEIGHT_CACHE": 90.0, "_PADDED_FP4_CACHE": 0}), 47: (95.0, {"_WEIGHT_CACHE": 90.0, "_PADDED_FP4_CACHE": 0})}
    ok = report(rows(), mem, [], [])
    assert ok.endswith("OPCHECK PASS"), ok
    assert "projection at M 5728" in ok and "16K 2 chunks" in ok, ok
    assert "FAIL: fi 6048" in report(rows(fi6048=8.0), mem, [], [])
    assert "M6048 FI err" in report(rows(err=0.2), mem, [], [])
    assert "padded FP4" in report(rows(), {47: (95.0, {"_PADDED_FP4_CACHE": 700.0})}, [], [])
    assert "per layer (> 200)" in report(rows(), {47: (400.0, {})}, [], [])
    assert "2 of 8 cells" in report(rows()[:2], mem, [], [])
    print("opcheck selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k: a[a.index(k) + 1] if k in a else None
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["run"]:
        os.makedirs(a[1], exist_ok=True)
        try:
            run(a[1], opt("--ckpt"), opt("--cap-dir"), opt("--fi-src"))
        except Exception:
            traceback.print_exc()
            print("OPCHECK FAIL: exception (traceback above)")
            sys.exit(1)
    else:
        sys.exit(__doc__)
