#!/usr/bin/env python3
"""k109 fi5863-moe-layer-probe (tp-2 #174): one routed-MoE layer of Qwen3.8-Flash-Next (H 2560, 512 experts, top-10,
checkpoint NVFP4 weights of layer 5 timed, layers 5 and 47 for numerics), our pinned b12x against FlashInfer main's
SM12x CuTe DSL MoE backends (#5863 W4A4 SM12xNvfp4Config and #5735 W4A16 SM12xNvfp4Bf16Config, unified MoE API), in
two geometries:
  tp1  I 640, the single-Spark layer
  tp2  I 320, the rank-0 shard of the 2x (gate/up rows 0..319, down columns 0..319; same experts, same routing)
at M 1 / 5 / 10 / 20 / 40 (decode: MTP verify rows, 5 per sequence) and 2048 / 3024 / 6048 / 8192 (prefill chunks).
Copy of k98/moe98.py (load, shard, reference, numerics gate, capture, report) with these changes:
  variants  b12x-m32 (anchor: dynamic tile 32x128, the k91 serving anchor), b12x-auto (no tile override), b12x-wm
            (B12X_MOE_DECODE_BACKEND=wm: the weight-major decode kernel for M <= B12X_MOE_WM_MAX_TOKENS, default 32;
            dynamic above), fi-w4a4 (SM12xNvfp4Config), fi-w4a16 (SM12xNvfp4Bf16Config). b12x variants run A4 (the
            shipped contract, per-expert FC1/FC2 activation scales), one execution plan of capacity 8192 with a
            prepared variant per M, heuristic selection (autotune off, as k98; the anchor checks it against k81).
  fi layout w13 = [up; gate] (FlashInfer's reference splits up then gate), logical E4M3 block scales and per-expert
            FP32 global scales (weight_scale_2), prepared by the config's prepare_weights. W4A4 gets the activation
            multipliers in the view keys w1_input_global_scale / w2_input_global_scale = 1 / max(input_scale) over the
            layer's experts (PR #5863: the keys take 1 / input_scale; k98's fi5168 path took input_scale itself).
  routing   "real" = the k97 capture (layer 5 / 47 routed-MoE inputs with their top-10 ids and weights from a cold 16K
            prompt) when present, "uniform" = synthetic. Decode M takes windows of 5 consecutive capture rows spaced
            1/8 of the capture apart (one window per sequence of an MTP verify batch); prefill M takes the first M
            rows. Unique experts per layer are reported for each M.
  timing    one CUDA graph per variant and cell (captured after 3 eager warm-ups, eager when capture fails, labelled),
            a 256 MB buffer rewritten before every replay to evict L2, median of 30 replays.
  memory    GPU memory each implementation keeps beyond the raw checkpoint tensors and the harness's own w13/scale
            concatenation, read after the whole M sweep (every serving bucket up to 8192 warmed), per layer; x48 is
            the bound if every layer owns its workspace.
Numerics first, at every M: the selected rows (at most 256) against an FP32 dequantized-weight reference (BF16
activations), layers 5 and 47, every pattern. Accuracy gate: err <= 1.10 x err(b12x-m32) + 0.005 in every cell.
Contract (fi-w4a4 only, reported): rel L2 difference to the b12x-m32 output <= 0.5 x err(b12x-m32). An FI error above
0.5 at M 2048 skips its timing (wrong scale convention, not a speed result).
Serving rule (fixed before the run; per FI variant and geometry, real routing, uniform only without a capture):
  GO-decode   accuracy gate PASS, geometric mean over M 5/10/20/40 of (best b12x / variant) >= 1.05, no M of that set
              below 0.97, kept memory x48 <= 2 GiB
  GO-prefill  the same over M 2048/3024/6048/8192
  else NO (reason). k99 lost 15% of the KV pool to FlashInfer scratch, hence the memory veto.
Usage: moe109.py b12x /out --ckpt C --num /num [--cap-dir /cap] | fi /out ... | run <variant> /out ... |
report /out (stdlib) | selftest (stdlib) | check --ckpt C [--fi] (CPU, in the image)
"""
import glob, json, math, os, statistics, subprocess, sys, time, traceback

E, H, TOPK = 512, 2560, 10
PARTS = {"tp1": 640, "tp2": 320}
MS = (1, 5, 10, 20, 40, 2048, 3024, 6048, 8192)
DECODE, PREFILL = (5, 10, 20, 40), (2048, 3024, 6048, 8192)
CAP = 8192
TIME_LAYER, NUM_LAYERS, NUM_M, NUM_ROWS = 5, (5, 47), 2048, 256
SYNTH = ("uniform",)
K81_MS = {3024: 431.0 / 48, 6048: (609.8 + 597.4) / 2 / 48}  # k81 routed MoE ms per layer per chunk
LAYERS = 48
VARIANTS = {"b12x-m32": ("b12x", {"B12X_DYNAMIC_TILE_MN": "32x128"}), "b12x-auto": ("b12x", {}),
            "b12x-wm": ("b12x", {"B12X_MOE_DECODE_BACKEND": "wm"}), "fi-w4a4": ("fi", {}), "fi-w4a16": ("fi", {})}
ANCHOR = "b12x-m32"
GO_X, CELL_MIN, MEM_GIB = 1.05, 0.97, 2.0
COLS = ("variant", "part", "layer", "pattern", "m", "ms", "tflops", "err_ref", "diff_b12x", "prep_s", "extra_mib",
        "backend", "src", "uniq", "mode")


def flop_per_token(i):
    return TOPK * 3 * 2 * H * i


# ---------------------------------------------------------------- GPU side (torch imported lazily)
def load_layer(ckpt, layer):
    import torch
    from safetensors import safe_open
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    pre = f"model.language_model.layers.{layer}.mlp.experts"
    files, out = {}, {}

    def get(k):
        f = idx[k]
        if f not in files:
            files[f] = safe_open(os.path.join(ckpt, f), "pt", device="cpu")
        return files[f].get_tensor(k)

    out["router"] = get(f"model.language_model.layers.{layer}.mlp.gate.weight").cuda()  # [E, H], checks the capture
    for name, p in (("g", "gate_proj"), ("u", "up_proj"), ("d", "down_proj")):
        out[name + "w"] = torch.stack([get(f"{pre}.{e}.{p}.weight") for e in range(E)]).cuda()
        out[name + "s"] = torch.stack([get(f"{pre}.{e}.{p}.weight_scale") for e in range(E)]).cuda()
        out[name + "g"] = torch.stack([get(f"{pre}.{e}.{p}.weight_scale_2").float().reshape(()) for e in range(E)]).cuda()
        out[name + "i"] = torch.stack([get(f"{pre}.{e}.{p}.input_scale").float().reshape(()) for e in range(E)]).cuda()
    return out


def shard(lw, i):
    """TP rank-0 shard of the intermediate dimension: gate/up rows [0, i), down columns [0, i)."""
    if i == lw["gw"].shape[1]:
        return lw
    s = dict(lw)
    for n in ("g", "u"):
        s[n + "w"], s[n + "s"] = lw[n + "w"][:, :i].contiguous(), lw[n + "s"][:, :i].contiguous()
    s["dw"], s["ds"] = lw["dw"][:, :, : i // 2].contiguous(), lw["ds"][:, :, : i // 16].contiguous()
    return s


def make_inputs(lw, pattern, seed, cap_dir, layer):
    """-> x [R, H] BF16, ids [R, TOPK] int32, w [R, TOPK] fp32 on GPU, R >= CAP; source label (k98)."""
    import torch
    rows = CAP
    if pattern == "real":
        files = sorted(glob.glob(os.path.join(cap_dir or "/nonexistent", f"moe{layer}.*.pt")))
        parts = [torch.load(f, weights_only=True) for f in files]
        x = torch.cat([p["x"] for p in parts]).to(torch.bfloat16)
        ids = torch.cat([p["ids"] for p in parts]).to(torch.int32)
        w = torch.cat([p["w"] for p in parts]).float()
        n = min(4096, x.shape[0])
        mine = torch.topk(x[:n].cuda().float() @ lw["router"].float().T, TOPK, dim=-1).indices.sort(-1).values
        theirs = ids[:n].cuda().long().sort(-1).values
        overlap = float((mine[:, :, None] == theirs[:, None, :]).any(-1).float().mean())
        src = (f"captured {x.shape[0]} rows, router overlap {overlap:.3f}, n_moe {parts[0]['n_moe']}"
               + ("" if overlap >= 0.9 else ", capture-mismatch"))
        while x.shape[0] < rows:
            x, ids, w = (torch.cat([t, t[: rows - t.shape[0]]]) for t in (x, ids, w))
            src += ", repeated"
        return x.cuda().contiguous(), ids.cuda().contiguous(), w.cuda().contiguous(), src
    g = torch.Generator().manual_seed(seed)
    sigma = float(lw["gi"].median()) * 6 * 448 / 5
    x = (torch.randn(rows, H, generator=g) * sigma).to(torch.bfloat16)
    ids = torch.topk(torch.randn(rows, E, generator=g), TOPK, dim=-1).indices
    w = torch.softmax(torch.randn(rows, TOPK, generator=g), dim=-1)
    return x.cuda(), ids.to(torch.int32).cuda().contiguous(), w.float().cuda().contiguous(), "synthetic"


def pick(m, n):
    """row indices for M: decode M = windows of 5 consecutive rows spaced n // 8 apart; prefill M = rows 0..M-1."""
    if m > 40:
        return list(range(m))
    step = n // 8
    return [s * step + j for s in range(-(-m // 5)) for j in range(5)][:m]


def uniq(ids):
    import torch
    return int(torch.unique(ids.reshape(-1)).numel())


def dequant(w_u8, sf, gs):
    """[R, K/2] packed E2M1 (low nibble first) x [R, K/16] E4M3 x scalar -> fp32 [R, K]."""
    import torch
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=w_u8.device)
    b = w_u8.view(torch.uint8)
    v = lut[torch.stack(((b & 0xF).long(), (b >> 4).long()), dim=-1).reshape(b.shape[0], -1)]
    return v * sf.float().repeat_interleave(16, dim=-1) * gs


def reference(lw, x, ids, w):
    import torch
    r0 = min(NUM_ROWS, x.shape[0])
    xf = x[:r0].float()
    out = torch.zeros(r0, H, device=x.device)
    sub = ids[:r0].long()
    for e in torch.unique(sub).tolist():
        r, k = (sub == e).nonzero(as_tuple=True)
        gt = xf[r] @ dequant(lw["gw"][e], lw["gs"][e], lw["gg"][e]).T
        up = xf[r] @ dequant(lw["uw"][e], lw["us"][e], lw["ug"][e]).T
        y = (torch.nn.functional.silu(gt) * up).to(torch.bfloat16).float() @ dequant(lw["dw"][e], lw["ds"][e], lw["dg"][e]).T
        out.index_add_(0, r, y * w[r, k, None])
    return out


def rel(a, ref):
    return float((a.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30))


class B12x:
    """k98's serving-contract b12x layer; one prepared weight set, one execution plan of capacity 8192 with a prepared
    variant per M of the sweep."""

    def __init__(self, lw):
        import torch
        from b12x._lib.intrinsics import swizzle_block_scale
        from b12x.moe import fused_moe
        self.fm, self.i = fused_moe, lw["dw"].shape[2] * 2
        w13 = torch.cat([lw["gw"], lw["uw"]], dim=1).contiguous()  # vLLM [gate; up] = W31, flipped by b12x at first use
        s13 = swizzle_block_scale(torch.cat([lw["gs"], lw["us"]], dim=1).contiguous())
        s2 = swizzle_block_scale(lw["ds"].contiguous())
        self.own = sum(t.numel() * t.element_size() for t in (w13, s13, s2))
        plan_w = fused_moe.plan_weights(
            source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat("modelopt_nvfp4"),
                                          w13_layout=fused_moe.W13Layout.W31),
            activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A4, nonlinearity="silu",
                                                io_dtype=torch.bfloat16),
            geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=self.i))
        self.experts = fused_moe.prepare_weights(plan=plan_w, weights=fused_moe.PackedWeights(
            w13=w13, w2=lw["dw"].contiguous(), w13_block_scales=s13, w2_block_scales=s2,
            w13_global_scales=lw["gg"].contiguous(), w2_global_scales=lw["dg"].contiguous(),
            input_scale=(1.0 / lw["gi"]).float().contiguous(), intermediate_scale=(1.0 / lw["di"]).float().contiguous()))
        self._plan(CAP)
        self.config = describe(self.plan)

    def _plan(self, cap):
        import torch
        from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall
        fm = self.fm
        plan = fm.plan_execution(
            experts=self.experts,
            capacity=fm.ExecutionCapacity(max_tokens=cap, top_k=TOPK, warmup_token_counts=MS, route_num_experts=0),
            routing=fm.RoutingSpec(apply_router_weight_on_input=False),
            invocation=FrozenMapping({"tuning_route_pattern": "cyclic_disjoint_topk"}))
        def make(n):  # one call per token count, sized to it (vLLM's _prepared_moe_call_factory(tokens=count))
            hid = torch.empty(n, H, dtype=torch.bfloat16, device="cuda").normal_(0, 0.125)
            out = torch.empty_like(hid)
            rr = torch.arange(n, dtype=torch.int32, device="cuda")[:, None] * TOPK + torch.arange(TOPK, dtype=torch.int32, device="cuda")
            ids, wts = rr.remainder(E).contiguous(), torch.full((n, TOPK), 1.0 / TOPK, device="cuda")

            def factory(state):
                scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=s.device) for s in state.scratch.scratch_specs())
                b = state.bind(scratch=scratch, a=hid, experts=self.experts, topk_weights=wts, topk_ids=ids, output=out,
                               input_scales_static=True)
                return PreparedCall(run=b.run, output=out, owners=(b, scratch))
            return factory
        counts = getattr(plan, "token_counts", None)
        req = (plan.request(name=f"k109.moe.c{cap}", prepare_calls={c: make(c) for c in counts}) if counts is not None
               else plan.request(name=f"k109.moe.c{cap}", prepare_call=make(cap)))
        with PreparationSession(device=torch.device("cuda"), autotune=False, compile_workers=2) as s:
            s.prepare((req,))
            s.freeze()
        nbytes = sum(spec.nbytes for spec in plan.scratch_specs())
        self.plan, self.scratch = plan, torch.empty(nbytes, dtype=torch.uint8, device="cuda")

    def backend(self, m):
        wm = os.environ.get("B12X_MOE_DECODE_BACKEND") == "wm" and m <= int(os.environ.get("B12X_MOE_WM_MAX_TOKENS", "32"))
        return f"cap{CAP}{'+wm-env' if wm else ''}"  # the env asked for wm at this M; b12x picks the kernel

    def fn(self, x, ids, w, out):
        # bind inside the call, as vLLM's B12xExperts.apply does per forward
        return lambda: self.fm.run(binding=self.fm.bind(self.plan, scratch=self.scratch, a=x, experts=self.experts,
                                                        topk_weights=w, topk_ids=ids, output=out, input_scales_static=True))

    def result(self, out):
        return out


def describe(plan):
    for obj in (plan, *getattr(plan, "variants", {}).values()):
        sel = getattr(obj, "selection", None)
        if sel is not None:
            return f"{getattr(sel, 'source', '?')} {getattr(sel, 'config', sel)}".replace("\t", " ").replace("\n", " ")
    return "?"


def fold_w13_globals(us, ug, gs, gg):
    """One FP32 global per expert for [up; gate]: equal weight_scale_2 pass through; otherwise the smaller side's E4M3
    block scales are rescaled to the larger global (lossy only for experts whose two globals differ)."""
    import torch
    g = torch.maximum(ug, gg)
    if torch.equal(ug, gg):
        return us, gs, g, 0
    rs = lambda s, own: (s.float() * (own / g)[:, None, None]).to(s.dtype)
    return rs(us, ug), rs(gs, gg), g, int((ug != gg).sum())


class FIUnified:
    """FlashInfer unified MoE API, SM12x CuTe DSL backends of #5863 (W4A4) and #5735 (W4A16), one MoELayer per layer."""

    def __init__(self, lw, name):
        import torch
        from flashinfer.fused_moe import (BackendOptions, ExecutionConfig, ExpertConfig, MoEConfig, MoEFinalizeConfig,
                                          MoELayer, MoEWeightPack, QuantConfig, QuantFormat, RoutingConfig,
                                          SM12xNvfp4Bf16Config, SM12xNvfp4Config, SwiGLU)
        self.w4a4 = name == "fi-w4a4"
        self.i = i = lw["dw"].shape[2] * 2
        cfg_t = SM12xNvfp4Config if self.w4a4 else SM12xNvfp4Bf16Config
        act = SwiGLU()
        us, gs, g1, self.folded = fold_w13_globals(lw["us"], lw["ug"], lw["gs"], lw["gg"])
        w1 = torch.cat([lw["uw"], lw["gw"]], dim=1).contiguous()  # [up; gate]: FlashInfer splits up, then gate
        s1 = torch.cat([us, gs], dim=1).contiguous()
        self.own = sum(t.numel() * t.element_size() for t in (w1, s1))
        dev = torch.device("cuda")
        view = cfg_t.prepare_weights(w1, s1, g1.float().contiguous(), lw["dw"].contiguous(), lw["ds"].contiguous(),
                                     lw["dg"].float().contiguous(), activation=act, num_local_experts=E, hidden_size=H,
                                     intermediate_size=i, device=dev)
        if self.w4a4:  # PR #5863: FP32 scalar quantization multipliers, 1 / input_scale (per layer: the max over experts)
            if not hasattr(view, "items"):
                raise TypeError(f"prepare_weights returned {type(view).__name__}, not a mapping: cannot add the input scales")
            view = dict(view.items())
            view["w1_input_global_scale"] = (1.0 / lw["gi"].max()).float().reshape(())
            view["w2_input_global_scale"] = (1.0 / lw["di"].max()).float().reshape(())
        config = MoEConfig(
            routing=RoutingConfig(num_experts=E, top_k=TOPK),
            quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4 if self.w4a4 else QuantFormat.BF16),
            experts=ExpertConfig(intermediate_size=i), activation=act, backend=BackendOptions((cfg_t(),)),
            finalize=MoEFinalizeConfig(do_finalize=True, use_fused_finalize=True),
            execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=CAP))
        self.layer = MoELayer(config, device=dev)
        self.runner = self.layer.runners[0]
        self.weights = MoEWeightPack()
        self.weights.prepare_for(self.runner.backend_key, view)
        self.config = f"FlashInfer {self.runner.backend_key} ({'W4A4' if self.w4a4 else 'W4A16'}), w13 globals folded on {self.folded} experts"
        self.last = None

    def backend(self, m):
        return self.runner.backend_key

    def fn(self, x, ids, w, out):
        from flashinfer.fused_moe import MoEActivationPack
        acts = MoEActivationPack(hidden_states_q=x, hidden_states_scale=None, topk_ids=ids, topk_weights=w)
        inputs = self.runner.pack_inputs(acts, self.weights)
        self.runner.forward(inputs, tactic=-1, do_preparation=True)

        def call():
            self.last = self.runner.forward(inputs, tactic=-1)
        return call

    def result(self, out):
        return self.last


def timed(fn, flush, iters=30, warm=3):
    """median ms of graph replays with a cold L2 (eager when the capture fails) -> (ms, mode)"""
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    run, mode = fn, "eager"
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay(); torch.cuda.synchronize()
        run, mode = g.replay, "graph"
    except Exception as e:  # noqa: BLE001 - some paths allocate or sync inside the call
        print(f"graph capture failed ({type(e).__name__}: {e}); timing eager", flush=True)
        torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); run(); b.record(); b.synchronize()
        t.append(a.elapsed_time(b))
    return statistics.median(t), mode


def run_variant(name, out_dir, ckpt, cap_dir, num_dir):
    import torch
    kind = VARIANTS[name][0]
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    patterns = (("real",) if glob.glob(os.path.join(cap_dir or "/nonexistent", "moe5.*.pt")) else ()) + SYNTH
    with open(os.path.join(out_dir, f"{name}.tsv"), "w") as f:
        for layer in NUM_LAYERS:
            full = load_layer(ckpt, layer)
            for part, i in PARTS.items():
                lw = shard(full, i)
                torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); t0 = time.time()
                impl = B12x(lw) if kind == "b12x" else FIUnified(lw, name)
                t_build = time.time() - t0
                refs = {}
                for pi, pat in enumerate(patterns):
                    X, IDS, W, src = make_inputs(lw, pat, 10900 + 10 * layer + pi, cap_dir, layer)
                    src = f"{src}; {impl.config}"
                    ms_list = MS if layer == TIME_LAYER else (NUM_M,)
                    for m in ms_list:
                        sel = torch.tensor(pick(m, X.shape[0]), device="cuda")
                        x, ids, w = X[sel].contiguous(), IDS[sel].contiguous(), W[sel].contiguous()
                        o = torch.empty(m, H, dtype=torch.bfloat16, device="cuda")
                        t1 = time.time()
                        fm = impl.fn(x, ids, w, o)
                        fm(); torch.cuda.synchronize()
                        prep_s = t_build + time.time() - t1 if (pi == 0 and m == ms_list[0]) else float("nan")
                        rk = (pat, tuple(pick(m, X.shape[0])[:NUM_ROWS]))
                        if rk not in refs:  # prefill M share rows 0..255: one reference per row set
                            refs[rk] = reference(lw, x, ids, w)
                        ref = refs[rk]
                        r0 = ref.shape[0]
                        res = impl.result(o)
                        err, diff = rel(res[:r0], ref), float("nan")
                        tag = f"{part}_L{layer}_{pat}"
                        if kind == "b12x":
                            torch.save(res[:r0].cpu(), os.path.join(num_dir, f"{name}_{tag}_m{m}.pt"))
                        else:
                            pa = os.path.join(num_dir, f"{ANCHOR}_{tag}_m{m}.pt")
                            if os.path.exists(pa):
                                diff = rel(res[:r0].cpu(), torch.load(pa, weights_only=True))
                        broken = kind == "fi" and m >= NUM_M and err > 0.5
                        ms, mode = (float("nan"), "-") if (layer != TIME_LAYER or broken) else timed(fm, flush)
                        torch.cuda.synchronize()
                        live = sum(t.numel() * t.element_size() for t in (X, IDS, W, x, ids, w, o, sel, *refs.values())) + impl.own
                        extra = (torch.cuda.memory_allocated() - base - live) / 2**20  # kept beyond raw tensors + inputs
                        row = (name, part, layer, pat, m, f"{ms:.4f}", f"{m * flop_per_token(i) / ms / 1e9:.1f}",
                               f"{err:.5f}", f"{diff:.5f}", f"{prep_s:.1f}", f"{extra:.0f}", impl.backend(m), src,
                               uniq(ids), mode)
                        f.write("\t".join(map(str, row)) + "\n"); f.flush()
                        print("\t".join(map(str, row)), flush=True)
                        del x, ids, w, o, sel
                    del X, IDS, W
                refs.clear()
                del impl, lw
                torch.cuda.empty_cache()
            del full
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- drivers and report (stdlib only)
def run_kind(kind, out_dir, ckpt, cap_dir, num_dir, budget_s):
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()
    for name, (k, env) in VARIANTS.items():
        if k != kind:
            continue
        left = budget_s - (time.time() - start)
        if left < 120:
            print(f"{name}: skipped, budget ({budget_s} s) spent", flush=True)
            continue
        t0 = time.time()
        args = [sys.executable, os.path.abspath(__file__), "run", name, out_dir, "--ckpt", ckpt, "--num", num_dir]
        with open(os.path.join(out_dir, f"{name}.log"), "w") as log:
            try:
                rc = subprocess.run(args + (["--cap-dir", cap_dir] if cap_dir else []), env={**os.environ, **env},
                                    stdout=log, stderr=subprocess.STDOUT, timeout=left).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        print(f"{name}: rc={rc} {time.time() - t0:.0f} s", flush=True)


def fnum(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


def read_rows(out_dir):
    rows = []
    for name in VARIANTS:
        p = os.path.join(out_dir, f"{name}.tsv")
        if os.path.exists(p):
            for line in open(p):
                v = line.rstrip("\n").split("\t")
                if len(v) == len(COLS):
                    r = dict(zip(COLS, v))
                    r["layer"], r["m"] = int(r["layer"]), int(r["m"])
                    for k in ("ms", "tflops", "err_ref", "diff_b12x", "prep_s", "extra_mib", "uniq"):
                        r[k] = fnum(r[k])
                    rows.append(r)
    return rows


def gmean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def report(out_dir, rows=None):
    rows = read_rows(out_dir) if rows is None else rows
    by = {(r["variant"], r["part"], r["layer"], r["pattern"], r["m"]): r for r in rows}
    names = [n for n in VARIANTS if any(r["variant"] == n for r in rows)]
    pats = list(dict.fromkeys(r["pattern"] for r in rows))
    fis = [n for n in names if VARIANTS[n][0] == "fi"]
    L, verdict = [], []
    srcs = sorted({f"{r['pattern']} ({r['variant']}): {r['src']}" for r in rows if r["m"] == NUM_M and r["layer"] == TIME_LAYER})
    L.append("routing/activations and configs:")
    L += [f"  {s}" for s in srcs] or ["  none"]
    gate, contract = {}, {}
    for part, i in PARTS.items():
        L.append("")
        L.append(f"== {part} (I {i}) numerics: rel L2 error vs FP32 reference on the selected rows (<= {NUM_ROWS}) at M "
                 f"{NUM_M} [fi-w4a4: rel L2 difference to the {ANCHOR} output]")
        L.append(f"   accuracy gate: err <= 1.10 x {ANCHOR} + 0.005 in every cell and at every M; contract check (fi-w4a4): "
                 f"difference to {ANCHOR} <= 0.5 x err({ANCHOR}) in every cell and at every M")
        L.append("variant".ljust(12) + "".join(f"L{l} {p}".rjust(16) for l in NUM_LAYERS for p in pats) + "  accuracy contract")
        for n in names:
            cells, ok, judged, cok, cjudged = [], True, 0, True, 0
            for l in NUM_LAYERS:
                for p in pats:
                    r = by.get((n, part, l, p, NUM_M))
                    if r is None or math.isnan(r["err_ref"]):
                        cells.append("n/a".rjust(16)); ok = False; continue
                    d = "" if math.isnan(r["diff_b12x"]) or n != "fi-w4a4" else f"[{r['diff_b12x']:.3f}]"
                    cells.append(f"{r['err_ref']:.4f}{d}".rjust(16))
            for (v, pt, l, p, m), r in by.items():
                a = by.get((ANCHOR, pt, l, p, m))
                if v != n or pt != part or math.isnan(r["err_ref"]) or a is None or math.isnan(a["err_ref"]):
                    continue
                judged += 1
                ok &= r["err_ref"] <= 1.10 * a["err_ref"] + 0.005
                if n == "fi-w4a4" and not math.isnan(r["diff_b12x"]):
                    cjudged += 1
                    cok &= r["diff_b12x"] <= 0.5 * a["err_ref"]
            gate[(n, part)] = ("PASS" if ok else "FAIL") if judged else "NOANCHOR"
            contract[(n, part)] = ("PASS" if cok else "FAIL") if cjudged else "N/A"
            L.append(n.ljust(12) + "".join(cells) + f"  {gate[(n, part)]:>8} {contract[(n, part)]:>8}")
        for p in pats:
            L.append(f"  layer {TIME_LAYER} {p}, error per M " + "  ".join(
                f"{n}: " + " ".join(f"{m}={by[(n, part, TIME_LAYER, p, m)]['err_ref']:.4f}" for m in MS if (n, part, TIME_LAYER, p, m) in by)
                for n in names))
    L.append("")
    L.append(f"== anchor: {ANCHOR} tp1 layer {TIME_LAYER} vs k81 serving (routed MoE ms per layer per chunk); tp2: N/A")
    anc = {}
    for p in pats:
        parts, d6 = [], None
        for m, ref in sorted(K81_MS.items()):
            r = by.get((ANCHOR, "tp1", TIME_LAYER, p, m))
            if r and not math.isnan(r["ms"]):
                d = 100 * (r["ms"] / ref - 1)
                parts.append(f"M {m}: {r['ms']:.2f} vs {ref:.2f} ({d:+.0f}%)")
                if m == 6048:
                    d6 = d
        anc[p] = d6
        L.append(f"  {p}: " + ("; ".join(parts) if parts else "no rows"))
    best = min((abs(v), p) for p, v in anc.items() if v is not None) if any(v is not None for v in anc.values()) else None
    anchor = "N/A" if best is None else ("OK" if best[0] <= 10 else "OFF")
    L.append(f"ANCHOR={anchor}" + (f" (closest: {best[1]} at M 6048, {anc[best[1]]:+.0f}%)" if best else ""))

    def bestb(part, p, m):
        bb = [by[(n, part, TIME_LAYER, p, m)]["ms"] for n in names if VARIANTS[n][0] == "b12x"
              and (n, part, TIME_LAYER, p, m) in by and not math.isnan(by[(n, part, TIME_LAYER, p, m)]["ms"])]
        return min(bb) if bb else None
    for part, i in PARTS.items():
        for p in pats:
            L.append("")
            L.append(f"== {part} (I {i}) speed, layer {TIME_LAYER}, {p} routing: us per call [backend, timing] and "
                     f"x = best b12x / variant; uniq = experts touched")
            L.append("M".rjust(6) + "  uniq" + "".join(n.rjust(32) for n in names))
            for m in MS:
                ctl = bestb(part, p, m)
                u = next((by[(n, part, TIME_LAYER, p, m)]["uniq"] for n in names if (n, part, TIME_LAYER, p, m) in by), float("nan"))
                cells = []
                for n in names:
                    r = by.get((n, part, TIME_LAYER, p, m))
                    if r is None or math.isnan(r["ms"]):
                        cells.append("n/a".rjust(32)); continue
                    x = f" x{ctl / r['ms']:.2f}" if ctl else ""
                    flag = "" if gate.get((n, part)) == "PASS" or n == ANCHOR else "*"
                    cells.append(f"{1000 * r['ms']:8.1f} [{r['backend'][:9]},{r['mode'][:1]}]{x}{flag}".rjust(32))
                L.append(f"{m:6d}{u:6.0f}" + "".join(cells))
    L.append("* = numerics gate FAIL, NOANCHOR or missing: no speed claim; timing g = CUDA graph replay, e = eager")
    L.append("")
    L.append("== preparation (s: weights + plans or FlashInfer layer + first call) and GPU memory kept beyond the raw "
             "checkpoint tensors and the harness's own copies after the whole M sweep (MiB per layer; x48 if per layer)")
    mem = {}
    for part in PARTS:
        for n in names:
            rs = [r for (v, pt, l, p, m), r in by.items() if v == n and pt == part and l == TIME_LAYER]
            if not rs:
                continue
            kept = max(r["extra_mib"] for r in rs)
            prep = next((r["prep_s"] for r in rs if not math.isnan(r["prep_s"])), float("nan"))
            mem[(n, part)] = kept * LAYERS / 1024
            L.append(f"  {part} {n}: prep {prep:.1f} s; kept {kept:.0f} MiB per layer, {mem[(n, part)]:.2f} GiB x48")
    L.append("")
    L.append(f"== serving rule (fixed before the run): GO-decode / GO-prefill = accuracy gate PASS, geometric mean of "
             f"(best b12x / variant) >= {GO_X:.2f} over M {list(DECODE)} / {list(PREFILL)}, no M of the set below "
             f"{CELL_MIN:.2f}, kept memory x48 <= {MEM_GIB:.0f} GiB")
    p = "real" if "real" in pats and not any("capture-mismatch" in r["src"] for r in rows if r["pattern"] == "real") else "uniform"
    for n in fis:
        for part in PARTS:
            out = []
            for lab, ms_set in (("decode", DECODE), ("prefill", PREFILL)):
                xs = []
                for m in ms_set:
                    r, c = by.get((n, part, TIME_LAYER, p, m)), bestb(part, p, m)
                    if r is None or c is None or math.isnan(r["ms"]):
                        xs = None; break
                    xs.append(c / r["ms"])
                if xs is None:
                    out.append(f"{lab}=NO(missing)"); continue
                gm, lo = gmean(xs), min(xs)
                why = [w for w, bad in (("gate", gate.get((n, part)) != "PASS"), (f"x{gm:.3f}<{GO_X}", gm < GO_X),
                                        (f"min x{lo:.2f}<{CELL_MIN}", lo < CELL_MIN),
                                        (f"mem {mem.get((n, part), float('nan')):.1f}GiB", not mem.get((n, part), 99) <= MEM_GIB)) if bad]
                out.append(f"{lab}={'GO' if not why else 'NO'}(x{gm:.3f}{'' if not why else ' ' + ','.join(why)})")
            verdict.append(f"{n}_{part}[{p}] " + " ".join(out))
    for v in verdict:
        L.append("  " + v)
    L.append(f"VERDICT anchor={anchor} " + "; ".join(verdict))
    return "\n".join(L)


def check(ckpt, fi):
    """CPU-only check inside the image: every API name the GPU path touches, and the checkpoint keys."""
    import torch  # noqa: F401
    if fi:
        import flashinfer
        from flashinfer.fused_moe import (BackendOptions, ExecutionConfig, ExpertConfig, MoEActivationPack,  # noqa: F401
                                          MoEConfig, MoEFinalizeConfig, MoELayer, MoEWeightPack, QuantConfig,
                                          QuantFormat, RoutingConfig, SM12xNvfp4Bf16Config, SM12xNvfp4Config, SwiGLU)
        import importlib.metadata as md
        for c in (SM12xNvfp4Config, SM12xNvfp4Bf16Config):
            assert hasattr(c, "prepare_weights"), c
        print("flashinfer from", flashinfer.__file__, "cutlass-dsl", md.version("nvidia-cutlass-dsl"))
    else:
        from b12x._lib.intrinsics import swizzle_block_scale  # noqa: F401
        from b12x.moe import fused_moe as fm
        from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall  # noqa: F401
        for n in ("plan_weights", "prepare_weights", "PackedWeights", "plan_execution", "bind", "run"):
            assert hasattr(fm, n), n
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    for layer in NUM_LAYERS:
        for e in (0, E - 1):
            for p in ("gate_proj", "up_proj", "down_proj"):
                for t in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                    f = idx[f"model.language_model.layers.{layer}.mlp.experts.{e}.{p}.{t}"]
                    assert os.path.exists(os.path.join(ckpt, f)), f
    print("check ok")


def selftest():
    assert pick(1, 16384) == [0] and pick(5, 16384) == [0, 1, 2, 3, 4]
    assert pick(10, 16384) == [0, 1, 2, 3, 4, 2048, 2049, 2050, 2051, 2052] and len(pick(40, 16384)) == 40
    assert pick(2048, 16384) == list(range(2048))
    rows = []
    for n in VARIANTS:
        for part, i in PARTS.items():
            for l in NUM_LAYERS:
                for p in ("real",) + SYNTH:
                    for m in (MS if l == TIME_LAYER else (NUM_M,)):
                        ms = K81_MS.get(m, 0.2 + m / 1000)
                        ms *= {"fi-w4a4": 0.90, "fi-w4a16": 1.02, "b12x-auto": 1.1}.get(n, 1.0)
                        err = 0.2 if n != "fi-w4a16" else 0.05
                        rows.append(dict(variant=n, part=part, layer=l, pattern=p, m=m, ms=ms, tflops=1.0,
                                         err_ref=err, diff_b12x=0.05 if n.startswith("fi") else float("nan"),
                                         prep_s=3.0, extra_mib=20.0, backend="dynamic", src="x", uniq=33.0, mode="graph"))
    out = report("/nonexistent", rows)
    assert "ANCHOR=OK" in out, out
    assert "fi-w4a4_tp1[real] decode=GO(x1.111) prefill=GO(x1.111)" in out, out
    assert "fi-w4a16_tp2[real] decode=NO(x0.980 x0.980<1.05)" in out, out
    for r in rows:
        if r["variant"] == "fi-w4a4" and r["part"] == "tp2" and r["m"] == 20:
            r["ms"] *= 1.2   # one decode M slower than 0.97 x best b12x
    o = report("/nonexistent", rows)
    assert "fi-w4a4_tp2[real] decode=NO(" in o and "min x0.93<0.97" in o and "prefill=GO" in o.split("fi-w4a4_tp2[real]")[1].split("\n")[0], o
    for r in rows:
        if r["variant"] == "fi-w4a4" and r["part"] == "tp1":
            r["extra_mib"] = 100.0   # 4.7 GiB x48: memory veto
    assert "mem 4.7GiB" in report("/nonexistent", rows)
    for r in rows:
        if r["variant"] == "fi-w4a4" and r["part"] == "tp1" and r["m"] == 5:
            r["err_ref"] = 0.5   # one bad M fails the accuracy gate
    assert "gate" in report("/nonexistent", rows).split("fi-w4a4_tp1[real]")[-1].split("\n")[0]
    for r in rows:
        if r["pattern"] == "real":
            r["src"] = "captured 16384 rows, router overlap 0.412, n_moe 49, capture-mismatch"
    assert "fi-w4a4_tp1[uniform]" in report("/nonexistent", rows)
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k: a[a.index(k) + 1] if k in a else None
    ck, cd, nd = opt("--ckpt"), opt("--cap-dir"), opt("--num") or "/num"
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["check"]:
        check(ck, "--fi" in a)
    elif a[:1] == ["report"]:
        print(report(a[1]))
    elif a[:1] == ["b12x"]:
        run_kind("b12x", a[1], ck, cd, nd, 2700)
    elif a[:1] == ["fi"]:
        run_kind("fi", a[1], ck, cd, nd, 2700)
    elif a[:1] == ["run"]:
        try:
            run_variant(a[1], a[2], ck, cd, nd)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    else:
        sys.exit(__doc__)
