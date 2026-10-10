#!/usr/bin/env python3
"""k76 helper (2026-10-09): cell plan, recipe/serve-log facts, and the report (manifest.json, capability.csv,
capability-t0.csv, k76.txt, comment.md) of the capability matrix.

  k76.py plan <1x|dp2|2x> [--lib]   benchy groups as "name<TAB>timeout s<TAB>args" (or the lib ITL args)
  k76.py plan-coding|plan-copy <setup>   "mode conc" lines for coding.py / stream counts for the copy bench
  k76.py recipe <recipe.yaml> [--short]
  k76.py serve <serve.log>          excerpt: KV pool, engine version, snapshot paths, drafter, AOT, b12x plans
  k76.py report <RES>
  k76.py --selftest
Results layout read by `report`: <RES>/<unit>/{pp,d0,d16k,d64k,d128k,t0}.json (llama-benchy --format json),
<unit>/itl.json (llm-inference-bench), <unit>/coding-<mode>-c<N>.json (coding.py), <unit>/copy.json
(bench_copy_streams.py), <unit>/recipe.json, <unit>/serve.log, <unit>/image-dgx0N.json, with
unit in 1x-dgx01, 1x-dgx02, dp2, 2x; <RES>/phases.txt, <RES>/dp2/router.txt.
"""
import csv, glob, io, json, os, re, statistics, sys

CAMPAIGN = "k76-capability-matrix"
COMMON = "--runs 3 --prompt-mode task --no-force-length --enable-prefix-caching --format json"
PP = "--pp 512 2048 8192 16384 32768 65536 131072 --tg 32 --depth 0 --concurrency 1 --warmup-runs 0"
D = "--pp 2048 --tg 512 --warmup-runs 0 --depth"
# name, timeout s (about 3x the estimate in ETA.txt), args
TG128 = ("tg128", 600, "--pp 2048 --tg 128 --depth 0 --concurrency 1")
T0 = ("t0", 900, "--pp 2048 --tg 512 --depth 0 --concurrency 1 8 --temperature 0")
PLAN = {
    "1x": [("pp", 1500, PP), ("d0", 1200, "--pp 2048 --tg 512 --depth 0 --concurrency 1 2 4 8"), TG128,
           ("d16k", 2100, D + " 16384 --concurrency 1 4 8"), ("d64k", 2400, D + " 65536 --concurrency 1 4"),
           ("d128k", 3900, D + " 131072 --concurrency 1 4"), T0],
    "dp2": [("d0", 1500, "--pp 2048 --tg 512 --depth 0 --concurrency 1 2 4 8 16"), TG128, T0],
    "2x": [("pp", 1200, PP), ("d0", 1500, "--pp 2048 --tg 512 --depth 0 --concurrency 1 2 4 8 16"), TG128,
           ("d16k", 3000, D + " 16384 --concurrency 1 4 8 16"), ("d64k", 1800, D + " 65536 --concurrency 1 4"),
           ("d128k", 2700, D + " 131072 --concurrency 1 4"), T0],
}
# coding corpus (k55, 36 prompts, max_tokens 768), both modes, 3 runs per cell; dp2 c1 = one 1x replica (the 1x rows).
# c1 = the single-user case, c8 = the loaded case; c4 dropped for runtime (2026-10-09 decision).
CODING_MODES = {"t0-nothink": "T=0, thinking off", "default": "server defaults, thinking on"}
CODING_CONC = {"1x": (1, 8), "dp2": (8,), "2x": (1, 8)}
COPY_STREAMS = {"1x": (1, 4, 8), "2x": (1, 4, 8)}     # no dp2: the copy bench reads /metrics, which the router lacks
CODING_H, COPY_H = "coding_probe.py, up to 768 tokens out", "copy-heavy benchmark, 1,500 tokens out"
LIB = {"1x": "--contexts 0,16k,64k --concurrency 1,4,8", "dp2": "--contexts 0 --concurrency 1,4,8,16",
       "2x": "--contexts 0,16k,64k --concurrency 1,4,8,16"}
NOT_PLANNED = [
    "1x: no c16 (max_num_seqs 8); 128K only c1/c4 (the 14 GiB pool holds ~7.5 requests at 128K, 1x results/"
    "tp1-v3c-20261005/kv-capacity.txt); 16K c2, 64K c2/c8 and 128K c2 trimmed for runtime.",
    "2x: 16K c2, 64K c2/c8/c16 and 128K c2/c8/c16 trimmed for runtime.",
    "dp2: no depth > 0 cells: the router keys a conversation on its first user message, and llama-benchy's "
    "context-load turn sends the fixed user message '.', so the cached context and the measured turn land on "
    "different replicas (the cell would measure prefill, not decode at depth). No pp sweep: one request through "
    "the router reaches one 1x replica, the same path as the 1x rows (also why the coding corpus skips c1 on dp2). "
    "ITL at depth 0 only. No copy-heavy cell: bench_copy_streams.py reads /metrics, which the router does not serve.",
    "coding-36: c1 and c8 only (c4 dropped for runtime). tg128: depth 0, c1 only. ITL (llm-inference-bench): depth 0/16K/64K, one 30 s window per cell.",
]


def plan_cells(args):
    """args string -> [(pp, tg, depth, conc)]"""
    a, cur, out = args.split(), None, {}
    for t in a:
        if t.startswith("--"):
            cur = t[2:]; out.setdefault(cur, [])
        elif cur:
            out[cur].append(t)
    return [(int(p), int(g), int(d), int(c)) for p in out["pp"] for g in out["tg"] for d in out["depth"]
            for c in out["concurrency"]]


# ---------------- recipe / serve facts ----------------

def gen_config(model, rev):
    p = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/{rev}/generation_config.json")
    try:
        return json.load(open(p)), p
    except OSError:
        return {}, p


def sampling_of(text, gc):
    m = re.search(r"--override-generation-config\s+'([^']+)'", text)
    if m:
        gc = {**gc, **json.loads(m.group(1).replace("{{", "{").replace("}}", "}"))}
    if re.search(r"--generation-config\s+vllm", text):
        gc = {}
    think = "thinking off" if re.search(r'enable_thinking"?\s*:\s*false', text) else "thinking on"
    if gc.get("temperature") is None:
        return f"server defaults, {think}"
    s = f"T={gc['temperature']}"
    if gc.get("top_p") is not None:
        s += f" top-p {gc['top_p']}"
    if gc.get("top_k") is not None:
        s += f" top-k {gc['top_k']}"
    return f"{s}, {think}"


def recipe_facts(path, text=None, gc=None):
    import yaml
    text = text if text is not None else open(path).read()
    y = yaml.safe_load(text)
    img = y["container"]; tag = img.rsplit(":", 1)[-1]
    rel = re.search(r"^# Release: (v\d+\.\d+\.\d+) \(old name ([^)]+)\)", text, re.M)
    relline = re.search(r"^# Release:.*$", text, re.M)
    alias = rel.group(2) if rel else tag
    dig = re.findall(r"^#.*Digest of (.*?): (sha256:[0-9a-f]{64})", text, re.M)
    tok = rel.group(2) if rel else tag.split("-")[0]
    pick = [d for lab, d in dig if tok in lab] or [d for _, d in dig[-1:]]
    th = re.search(r"-([0-9a-f]{8})-([0-9a-f]{8})-warm$", tag)
    gc0, gcp = gen_config(y["model"], y.get("model_revision", ""))
    gc = gc if gc is not None else gc0
    eff = re.search(r'"reasoning_effort"\s*:\s*"(\w+)"', text)
    git = {}
    if path:
        d = os.path.dirname(os.path.abspath(path))
        for k, cmd in (("registry_commit", "git -C '%s' rev-parse HEAD"), ("registry_commit_date", "git -C '%s' log -1 --format=%%ci")):
            git[k] = os.popen((cmd % d) + " 2>/dev/null").read().strip()
    return {"recipe_file": path, **git, "name": y.get("name"), "release_line": relline.group(0) if relline else "",
            "release": rel.group(1) if rel else "", "alias": alias, "image": img, "image_tag": tag,
            "recipe_digest_comment": pick[0] if pick else "", "digest_comments": [f"{a}: {b}" for a, b in dig],
            "b12x_commit_from_tag": th.group(1) if th else "", "vllm_commit_from_tag": th.group(2) if th else "",
            "model": y.get("model"), "model_revision": y.get("model_revision", ""),
            "max_num_seqs": (y.get("defaults") or {}).get("max_num_seqs"),
            "tensor_parallel": (y.get("defaults") or {}).get("tensor_parallel"),
            "sampling": sampling_of(text, gc), "generation_config": gcp,
            "reasoning_effort_default": eff.group(1) if eff else ""}


SERVE_PAT = re.compile(r"GPU KV cache size|Available KV cache memory|Initializing a V1 LLM engine|snapshots/[0-9a-f]{40}"
                       r"|mtp-refit|overlay:|Directly load AOT compilation|Dynamo bytecode transform|b12x ready|sampling param"
                       r"|non-default args|max_num_seqs")


def serve_facts(text):
    f = {}
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    f["kv_tokens"] = int(m.group(1).replace(",", "")) if m else None
    m = re.search(r"Initializing a V1 LLM engine \(([^)]*)\)", text)
    f["vllm_version"] = m.group(1) if m else ""
    m = re.search(r"\+g([0-9a-f]{7,})", f["vllm_version"])
    f["vllm_commit"] = m.group(1) if m else ""
    m = re.search(r"model='([^']+)'", text)
    f["served_model_path"] = m.group(1) if m else ""
    m = re.search(r"speculative_config=SpeculativeConfig\(method='([^']+)', model='([^']+)'", text)
    f["drafter"] = f"{m.group(1)} from {m.group(2)}" if m else ""
    f["snapshot_paths"] = sorted(set(re.findall(r"[^\s'\",]*snapshots/[0-9a-f]{40}[^\s'\",]*", text)))
    f["overlay"] = sorted(set(re.findall(r"overlay: \S+", text)))
    f["aot"] = "AOT loaded" if "Directly load AOT compilation" in text else ("compiled (cold)" if "Dynamo bytecode transform" in text else "")
    f["b12x_measured_plans"] = sum(int(x) for x in re.findall(r"b12x ready [a-z_.]+: \d+/\d+ ready, (\d+) measured", text))
    m = re.search(r"'max_num_seqs': (\d+)", text)
    f["max_num_seqs_log"] = int(m.group(1)) if m else None
    return f


# ---------------- measurements ----------------

def pct(xs, q):
    xs = sorted(xs); p = (len(xs) - 1) * q; i = int(p)
    return xs[i] if i + 1 >= len(xs) else xs[i] + (xs[i + 1] - xs[i]) * (p - i)


def chunks(vals, c, n):
    """per-run means of per-request values when the list is n runs x c requests, else the values themselves"""
    if c > 1 and len(vals) == n * c:
        return [statistics.mean(vals[i * c:(i + 1) * c]) for i in range(n)]
    return list(vals)


def benchy(path):
    """-> version, {(pp, tg, depth, conc): run dict} (measured phase only)"""
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return "", {}
    out = {}
    for b in d.get("benchmarks", []):
        if not b.get("is_context_prefill_phase"):
            out[(b["prompt_size"], b["response_size"], b["context_size"], b["concurrency"])] = b
    return d.get("version", ""), out


def vals(b, k):
    m = b.get(k) or {}
    return list(m.get("values") or [])


def measure(udir, t0=False):
    """unit dir -> {key: dict(kind, data, file)}; key = (metric, conc, prompt, depth); kind runs|pool|one"""
    M, ver = {}, {}
    def put(k, kind, data, f, wl="chat", **kw):
        if data:
            M[k + (None,) * (5 - len(k))] = dict(kind=kind, data=data, file=f, wl=wl, **kw)
    files = ["t0"] if t0 else ["pp", "d0", "tg128", "d16k", "d64k", "d128k"]
    for n in files:
        v, cells = benchy(os.path.join(udir, n + ".json"))
        if v:
            ver["benchy"] = v
        for (p, g, dep, c), b in cells.items():
            f = n + ".json"
            tot, req = vals(b, "tg_throughput"), vals(b, "tg_req_throughput")
            ttft = [x / 1000 for x in vals(b, "e2e_ttft")]
            nr = len(tot) if c > 1 else len(req)
            if g == 32 and dep == 0 and c == 1:          # pp sweep
                put(("pp", 1, p, None), "runs", vals(b, "pp_throughput"), f, "long-prompt")
                put(("ttft_s", 1, p, None), "runs", ttft, f, "long-prompt")
                continue
            if g == 128 and dep == 0:
                put(("tg128_total", c, p, None), "runs", tot, f)
                continue
            if g != 512:
                continue
            if dep == 0:
                put(("tg_total", c, p, None), "runs", tot, f)
                put(("tg_req", c, p, None), "runs", chunks(req, c, nr), f)
            if not t0:
                put(("decode_depth_total", c, p, dep), "runs", tot, f)
                put(("decode_depth_req", c, p, dep), "runs", chunks(req, c, nr), f)
            for q in (50, 90, 99):
                put((f"ttft_p{q}_s", c, p, dep), "pool", ttft, f)
    if not t0:
        try:
            d = json.load(open(os.path.join(udir, "itl.json")))
            ver["lib"] = d["metadata"]["version"]
            for r in d.get("results", []):
                if r.get("failure_reason") or not r.get("inter_token_latency_p50"):
                    continue
                for q in (50, 90, 99):
                    put((f"itl_p{q}_ms", r["concurrency"], None, r["context_tokens"]), "one",
                        [r[f"inter_token_latency_p{q}"] * 1000], "itl.json", "sustained-decode")
        except (OSError, ValueError, KeyError):
            pass
        for mode, smp in CODING_MODES.items():
            for f in sorted(glob.glob(os.path.join(udir, f"coding-{mode}-c*.json"))):
                d = load_json(f, {}); runs = [r for r in d.get("runs", []) if r.get("tps")]
                if not runs:
                    continue
                c, fn = d["conc"], os.path.basename(f)
                hits, nreq = sum(r["hit_max"] for r in runs), sum(r["n"] for r in runs)
                kw = dict(samp=smp, harness=CODING_H, hits=hits, nreq=nreq)
                put(("coding_probe_median", c, None, None, mode), "cmedian", [statistics.median(r["tps"]) for r in runs], fn, "coding-36", **kw)
                allv = [x for r in runs for x in r["tps"]]
                put(("coding_probe_min", c, None, None, mode), "cmin", allv, fn, "coding-36", **kw)
                put(("coding_probe_max", c, None, None, mode), "cmax", allv, fn, "coding-36", **kw)
                put(("coding_total", c, None, None, mode), "runs", [r["agg_tps"] for r in runs], fn, "coding-36", **kw)
        d = load_json(os.path.join(udir, "copy.json"), {})
        for n in sorted({r["n"] for r in d.get("rounds", [])}):
            w = [r["window"]["tok_s"] for r in d["rounds"] if r["n"] == n and (r.get("window") or {}).get("tok_s")]
            put(("copy_total", n, None, None, "copy"), "runs", w, "copy.json", "copy-heavy",
                samp="low reasoning effort", harness=COPY_H, what="window tok/s (all streams decoding), ")
    return M, ver


def num(v):
    return "" if v is None else (str(int(v)) if float(v).is_integer() and abs(v) >= 1000 else str(round(v, 3)))


def rows_for(setup, units, rel, alias, date, sampling, res, campaign, ver):
    """units: [(suffix, dir, M)]; 1x: two units (main pooled + per-Spark suffixed rows)."""
    out = []
    hb = f"llama-benchy {ver.get('benchy', '')}".strip()
    hl = f"llm-inference-bench {ver.get('lib', '')}".strip()
    def emit(k, entries, suffix, src):
        metric = k[0]; e0 = entries[0]; kind = e0["kind"]; ns = len(entries)
        data = [x for e in entries for x in e["data"]]
        harness = e0.get("harness") or (hl if kind == "one" else hb)
        n = len(e0["data"]); what = f"{n} runs" if ns == 1 else f"2 Sparks x {n} runs"
        def row(metric, value, sd, stat):
            _, c, p, dep = k[:4]
            out.append([setup, rel, alias, date, campaign, metric, num(c), num(p), num(dep), num(value), num(sd), stat,
                        e0.get("samp") or sampling, harness, src, e0["wl"]])
        if kind in ("cmedian", "cmin", "cmax"):
            hit = f", {sum(e['hits'] for e in entries)}/{sum(e['nreq'] for e in entries)} requests hit max_tokens"
            if kind == "cmedian":
                row(metric + suffix, statistics.mean(data), statistics.stdev(data) if len(data) > 1 else None,
                    f"median of 36 prompts, mean of {what}" + hit)
                row(metric + "_max" + suffix, max(data), None, f"max of {what} of the median" + hit)
            else:
                nr = len(entries[0]["data"]) // 36 or 1
                w = f"36 prompts x {nr} runs" if ns == 1 else f"36 prompts x 2 Sparks x {nr} runs"
                row(metric + suffix, (min if kind == "cmin" else max)(data), None, f"{kind[1:]} of {w}" + hit)
        elif kind == "runs":
            row(metric + suffix, statistics.mean(data), statistics.stdev(data) if len(data) > 1 else None,
                f"{e0.get('what', '')}mean ± sd of {what}, one boot" + ("" if ns == 1 else " each"))
            row(metric + "_max" + suffix, max(data), None, f"max of {what}")
        elif kind == "pool":
            q = int(metric.split("_p")[1].split("_")[0])
            row(metric + suffix, pct(data, q / 100), None,
                f"p{q} of {len(data)} requests ({'2 Sparks x ' if ns > 1 else ''}3 runs x c{k[1]}), one boot{' each' if ns > 1 else ''}")
        else:
            row(metric + suffix, statistics.mean(data), None,
                "one 30 s sustained decode window, one boot" if ns == 1 else "mean of 2 Sparks, one 30 s sustained decode window each")
    rd = os.path.basename(os.path.normpath(res))
    keys = sorted({k for _, _, M in units for k in M}, key=lambda k: (k[0], k[4] or "", k[1] or 0, k[2] or 0, k[3] if k[3] is not None else -1))
    for k in keys:
        have = [(sfx, d, M[k]) for sfx, d, M in units if k in M]
        if len(units) > 1 and len(have) == len(units):
            f = have[0][2]["file"]
            emit(k, [h[2] for h in have], "", f"2x:results/{rd}/1x-dgx0{{1,2}}/{f}")
        for sfx, d, e in have:
            emit(k, [e], sfx if len(units) > 1 else "", f"2x:results/{rd}/{os.path.basename(d)}/{e['file']}")
    return out


HEAD = "setup,release,alias,date,campaign,metric,conc,prompt_tokens,depth_tokens,value,sd,stat,sampling,harness,source,workload".split(",")


def load_json(p, default=None):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return default


def report(res):
    m = re.search(r"(\d{4})(\d{2})(\d{2})-\d{4}$", os.path.normpath(res))
    date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""
    U = {u: os.path.join(res, u) for u in ("1x-dgx01", "1x-dgx02", "dp2", "2x")}
    man = {"job": "k76", "date": date, "results": res, "setups": {}}
    for u, d in U.items():
        r = load_json(os.path.join(d, "recipe.json"), {})
        try:
            s = serve_facts(open(os.path.join(d, "serve.log"), errors="replace").read())
        except OSError:
            s = {}
        imgs = {}
        for f in sorted(glob.glob(os.path.join(d, "image-*.json"))):
            j = load_json(f, [])
            if j:
                imgs[os.path.basename(f)[6:-5]] = {"id": j[0].get("Id"), "repo_digests": j[0].get("RepoDigests")}
        man["setups"][u] = {"recipe": r, "serve": s, "images": imgs}
    rt = os.path.join(res, "dp2", "router.txt")
    man["setups"]["dp2"]["router"] = open(rt).read().strip() if os.path.exists(rt) else ""
    if os.path.isdir(U["dp2"]):
        man["setups"]["dp2"]["recipe"] = man["setups"]["1x-dgx01"]["recipe"]   # replicas = the 1x recipe
    ph = os.path.join(res, "phases.txt")
    man["phases"] = open(ph).read().splitlines() if os.path.exists(ph) else []
    main, t0rows, ver, missing = [], [], {}, []
    for setup, units in (("1x", [("_dgx01", "1x-dgx01"), ("_dgx02", "1x-dgx02")]), ("dp2", [("", "dp2")]), ("2x", [("", "2x")])):
        r = man["setups"][units[0][1]]["recipe"]
        if not r:
            missing.append(f"{setup}: no recipe facts (setup not reached)")
            continue
        rel, alias, samp = r.get("release", ""), r.get("alias", ""), r.get("sampling", "")
        for t0, rows, camp, smp in ((False, main, CAMPAIGN, samp), (True, t0rows, CAMPAIGN + "-t0", "T=0, " + samp.split(", ")[-1])):
            us = []
            for sfx, u in units:
                M, v = measure(U[u], t0); ver.update(v); us.append((sfx, U[u], M))
            rows += rows_for(setup, us, rel, alias, date, smp, res, camp, ver)
        kv = [man["setups"][u]["serve"].get("kv_tokens") for _, u in units]
        if setup == "dp2":
            kv = [man["setups"][u]["serve"].get("kv_tokens") for u in ("1x-dgx01", "1x-dgx02")]
        if all(kv):
            src = f"2x:results/{os.path.basename(os.path.normpath(res))}"
            if setup == "1x":
                main.append(["1x", rel, alias, date, CAMPAIGN, "kv_tokens", "", "", "", num(statistics.mean(kv)), "",
                             "vLLM serve log, " + ("same on both Sparks" if len(set(kv)) == 1 else "mean of 2 Sparks"), samp, "vLLM serve log", f"{src}/1x-dgx0{{1,2}}/serve-excerpt.txt", ""])
                for (sfx, u), k in zip(units, kv):
                    main.append(["1x", rel, alias, date, CAMPAIGN, "kv_tokens" + sfx, "", "", "", num(k), "", "vLLM serve log", samp, "vLLM serve log", f"{src}/{u}/serve-excerpt.txt", ""])
            elif setup == "dp2":
                main.append(["dp2", rel, alias, date, CAMPAIGN, "kv_tokens", "", "", "", num(sum(kv)), "", f"2 replicas x {kv[0]:,}" if len(set(kv)) == 1 else "sum of 2 replicas", samp, "vLLM serve log", f"{src}/1x-dgx0{{1,2}}/serve-excerpt.txt", ""])
            else:
                main.append(["2x", rel, alias, date, CAMPAIGN, "kv_tokens", "", "", "", num(kv[0]), "", "vLLM serve log, rank 0", samp, "vLLM serve log", f"{src}/2x/serve-excerpt.txt", ""])
        for _, u in units:   # planned benchy cells without data
            for name, _, args in PLAN[setup]:
                _, cells = benchy(os.path.join(U[u], name + ".json"))
                for cell in plan_cells(args):
                    if cell not in cells:
                        missing.append(f"{u} {name}: pp {cell[0]} tg {cell[1]} depth {cell[2]} c{cell[3]} not measured")
            if not os.path.exists(os.path.join(U[u], "itl.json")):
                missing.append(f"{u}: ITL (llm-inference-bench) not measured")
            for mode in CODING_MODES:
                for c in CODING_CONC[setup]:
                    if not (load_json(os.path.join(U[u], f"coding-{mode}-c{c}.json"), {}).get("runs")):
                        missing.append(f"{u}: coding {mode} c{c} not measured")
            got = {r["n"] for r in load_json(os.path.join(U[u], "copy.json"), {}).get("rounds", [])}
            missing += [f"{u}: copy-heavy {n} streams not measured" for n in COPY_STREAMS.get(setup, ()) if n not in got]
    man["harness"] = {k: v for k, v in ver.items()}
    json.dump(man, open(os.path.join(res, "manifest.json"), "w"), indent=2)
    for name, rows in (("capability.csv", main), ("capability-t0.csv", t0rows)):
        with open(os.path.join(res, name), "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n"); w.writerow(HEAD); w.writerows(rows)
    txt = summary(man, main, t0rows, missing)
    open(os.path.join(res, "k76.txt"), "w").write(txt)
    print(f"rows {len(main)} capability.csv, {len(t0rows)} capability-t0.csv, {len(missing)} cells not measured")
    return len(main)


def summary(man, main, t0rows, missing):
    L = [f"k76 capability matrix {man['date']}: shipped 1x (one per Spark), DP=2 (the two 1x behind tools/dp2/pa_router.py), 2x TP=2.",
         "Sampling: server defaults from the recipe (see the sampling column), thinking on; T=0 diagnostic rows in capability-t0.csv.",
         f"Harness: {', '.join({'benchy': 'llama-benchy', 'lib': 'llm-inference-bench'}.get(k, k) + ' ' + v for k, v in man.get('harness', {}).items())}. TTFT percentiles: llama-benchy per-request "
         "e2e TTFT pooled over 3 runs. ITL percentiles: llm-inference-bench, one 30 s sustained decode window per cell.", ""]
    for u, s in man["setups"].items():
        r, v = s.get("recipe", {}), s.get("serve", {})
        if not r:
            continue
        L.append(f"[{u}] {r.get('release') or '-'} (alias {r.get('alias')}), recipe {r.get('recipe_file')} @ {str(r.get('registry_commit'))[:8]}, "
                 f"image {r.get('image_tag')} ids {', '.join(h + ' ' + str(i.get('id'))[7:19] for h, i in s.get('images', {}).items())}, "
                 f"digest comment {r.get('recipe_digest_comment') or '-'}")
        if v:
            L.append(f"     model {r.get('model')}@{str(r.get('model_revision'))[:8]}, vLLM {v.get('vllm_version')}, b12x {r.get('b12x_commit_from_tag') or '-'} (tag), "
                     f"KV {v.get('kv_tokens')}, max_num_seqs {r.get('max_num_seqs')}, {v.get('aot')}, b12x plans measured {v.get('b12x_measured_plans')}, "
                     f"drafter {v.get('drafter') or '-'}")
        if s.get("router"):
            L.append(f"     router {s['router']}")
    L += ["", "Phases (name, start, end epoch s, min MemAvailable GiB dgx-01, dgx-02):"] + ["  " + p for p in man.get("phases", [])]
    L += ["", f"{'setup':5s} {'workload':16s} {'metric':24s} {'c':>3s} {'prompt':>7s} {'depth':>7s} {'value':>10s} {'sd':>8s}  stat | sampling"]
    for r in main + t0rows:
        if r[5].endswith(("_dgx01", "_dgx02")):
            continue
        L.append(f"{r[0]:5s} {r[15]:16s} {r[5] + (' T=0' if r[4].endswith('-t0') else ''):24s} {r[6]:>3s} {r[7]:>7s} {r[8]:>7s} {r[9]:>10s} {r[10]:>8s}  {r[11]} | {r[12]}")
    L += ["", "Not planned:"] + ["  " + x for x in NOT_PLANNED]
    L += ["", "Planned but not measured:"] + (["  " + x for x in missing] or ["  none"])
    return "\n".join(L) + "\n"


# ---------------- selftest ----------------

def selftest():
    import tempfile
    rec = """# Release: v3.1.0 (old name b1.6)
# Digest of this tag: sha256:%s
# Digest of the b1.6 tag: sha256:%s
recipe_version: "1"
name: qwen3.8-flash-next-2x-dgx-spark
model: org/M
model_revision: abc
container: ghcr.io/x/spark-vllm-b12x:b1.6-20261008-b7fbaf96-a7e649d8-warm
defaults:
  max_num_seqs: 16
command: |
  vllm serve x --default-chat-template-kwargs '{{"reasoning_effort":"medium"}}'
""" % ("1" * 64, "2" * 64)
    f = recipe_facts("", rec, {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
    assert f["release"] == "v3.1.0" and f["alias"] == "b1.6" and f["recipe_digest_comment"] == "sha256:" + "2" * 64, f
    assert f["sampling"] == "T=1.0 top-p 0.95 top-k 20, thinking on" and f["max_num_seqs"] == 16, f
    assert f["vllm_commit_from_tag"] == "a7e649d8" and f["reasoning_effort_default"] == "medium"
    f2 = recipe_facts("", rec.replace("# Release: v3.1.0 (old name b1.6)\n", ""), {})
    assert f2["release"] == "" and f2["alias"] == "b1.6-20261008-b7fbaf96-a7e649d8-warm" and f2["sampling"] == "server defaults, thinking on"
    s = serve_facts("x INFO Initializing a V1 LLM engine (v20260926.dev17+ga7e649d8f.d20261001) with config: model='/m', "
                    "speculative_config=SpeculativeConfig(method='mtp', model='/m/snapshots/" + "a" * 40 + "', x\n"
                    "GPU KV cache size: 993,754 tokens, Max\nDirectly load AOT compilation\n")
    assert s["kv_tokens"] == 993754 and s["vllm_commit"] == "a7e649d8f" and s["aot"] == "AOT loaded" and len(s["snapshot_paths"]) == 1, s
    assert plan_cells(PLAN["1x"][1][2]) == [(2048, 512, 0, c) for c in (1, 2, 4, 8)]
    assert abs(pct([1, 2, 3, 4], 0.5) - 2.5) < 1e-9 and pct([5], 0.99) == 5
    with tempfile.TemporaryDirectory() as td:
        res = os.path.join(td, "k76-capability-matrix-20261010-0100")
        def bm(p, g, d, c, tot, req, ttft):
            return {"prompt_size": p, "response_size": g, "context_size": d, "concurrency": c, "is_context_prefill_phase": False,
                    "tg_throughput": {"values": tot}, "tg_req_throughput": {"values": req}, "e2e_ttft": {"values": ttft},
                    "pp_throughput": {"values": [2000.0, 2100.0, 2200.0]}}
        for u, base in (("1x-dgx01", 50.0), ("1x-dgx02", 54.0), ("2x", 70.0)):
            os.makedirs(os.path.join(res, u))
            json.dump({"version": "0.4.1", "benchmarks": [
                bm(2048, 512, 0, 1, [base, base + 1, base + 2], [base, base + 1, base + 2], [100.0, 110.0, 120.0]),
                bm(2048, 512, 0, 2, [80.0, 82.0, 84.0], [40.0, 41.0, 42.0, 43.0, 44.0, 45.0], [200.0] * 6),
                dict(bm(2048, 512, 0, 2, [1.0], [1.0], [1.0]), is_context_prefill_phase=True)]}, open(os.path.join(res, u, "d0.json"), "w"))
            json.dump({"version": "0.4.1", "benchmarks": [bm(131072, 32, 0, 1, [1.0], [1.0], [60000.0, 61000.0, 62000.0])]},
                      open(os.path.join(res, u, "pp.json"), "w"))
            json.dump({"metadata": {"version": "0.7.6"}, "results": [{"concurrency": 1, "context_tokens": 0, "failure_reason": "",
                       "inter_token_latency_p50": 0.02, "inter_token_latency_p90": 0.03, "inter_token_latency_p99": 0.05}]},
                      open(os.path.join(res, u, "itl.json"), "w"))
            json.dump(recipe_facts("", rec, {"temperature": 1.0, "top_p": 0.95, "top_k": 20}), open(os.path.join(res, u, "recipe.json"), "w"))
            open(os.path.join(res, u, "serve.log"), "w").write("GPU KV cache size: 993,754 tokens\n")
            for mode, b in (("t0-nothink", 70.0), ("default", 60.0)):
                json.dump({"mode": mode, "conc": 4, "runs": [{"n": 36, "hit_max": 2, "agg_tps": 150.0 + i, "wall_s": 100,
                           "tps": [b + i + j / 10 for j in range(36)]} for i in range(3)]},
                          open(os.path.join(res, u, f"coding-{mode}-c4.json"), "w"))
            json.dump({"rounds": [{"n": 4, "window": {"tok_s": t}} for t in (200.0, 210.0, 220.0)]}, open(os.path.join(res, u, "copy.json"), "w"))
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            report(res)
        rows = list(csv.DictReader(open(os.path.join(res, "capability.csv"))))
        assert list(rows[0]) == HEAD and all(len(r) == 16 for r in csv.reader(open(os.path.join(res, "capability.csv"))))
        g = {(r["setup"], r["metric"], r["conc"], r["prompt_tokens"], r["depth_tokens"]): r for r in rows}
        r = g[("1x", "tg_total", "1", "2048", "")]
        assert r["value"] == "53.0" and r["stat"] == "mean ± sd of 2 Sparks x 3 runs, one boot each" and r["date"] == "2026-10-10", r
        assert g[("1x", "tg_total_max", "1", "2048", "")]["value"] == "56.0"
        assert g[("1x", "tg_total_dgx02", "1", "2048", "")]["value"] == "55.0"
        assert g[("1x", "tg_req", "2", "2048", "")]["value"] == "42.5"          # per-run means 40.5/42.5/44.5 on each Spark
        assert g[("2x", "decode_depth_total", "1", "2048", "0")]["value"] == "71.0"
        assert g[("2x", "pp", "1", "131072", "")]["value"] == "2100" and g[("2x", "ttft_s", "1", "131072", "")]["value"] == "61.0"
        assert g[("2x", "ttft_p50_s", "2", "2048", "0")]["value"] == "0.2"
        assert g[("2x", "itl_p99_ms", "1", "", "0")]["value"] == "50.0" and g[("2x", "itl_p99_ms", "1", "", "0")]["harness"] == "llm-inference-bench 0.7.6"
        assert g[("2x", "kv_tokens", "", "", "")]["value"] == "993754" and g[("2x", "tg_total", "1", "2048", "")]["harness"] == "llama-benchy 0.4.1"
        assert g[("1x", "kv_tokens", "", "", "")]["source"].endswith("1x-dgx0{1,2}/serve-excerpt.txt")
        assert ("dp2", "kv_tokens", "", "", "") not in g        # dp2 not reached: no rows
        assert g[("2x", "tg_total", "1", "2048", "")]["workload"] == "chat" and g[("2x", "pp", "1", "131072", "")]["workload"] == "long-prompt"
        assert g[("2x", "itl_p99_ms", "1", "", "0")]["workload"] == "sustained-decode" and g[("2x", "kv_tokens", "", "", "")]["workload"] == ""
        cm = {(r["setup"], r["metric"], r["sampling"]): r for r in rows if r["workload"] == "coding-36"}
        r = cm[("2x", "coding_probe_median", "T=0, thinking off")]      # run medians 71.75, 72.75, 73.75
        assert r["value"] == "72.75" and r["sd"] == "1.0" and r["conc"] == "4" and r["depth_tokens"] == "" and \
            r["stat"] == "median of 36 prompts, mean of 3 runs, 6/108 requests hit max_tokens" and r["harness"] == CODING_H, r
        assert cm[("2x", "coding_probe_median_max", "T=0, thinking off")]["value"] == "73.75"
        assert cm[("2x", "coding_probe_min", "server defaults, thinking on")]["value"] == "60.0"
        assert cm[("2x", "coding_probe_max", "server defaults, thinking on")]["stat"].startswith("max of 36 prompts x 3 runs")
        assert cm[("1x", "coding_probe_median", "server defaults, thinking on")]["stat"].startswith("median of 36 prompts, mean of 2 Sparks x 3 runs, 12/216")
        assert cm[("2x", "coding_total", "T=0, thinking off")]["value"] == "151.0" and cm[("1x", "coding_total_dgx02", "T=0, thinking off")]["value"] == "151.0"
        cp = {(r["setup"], r["metric"]): r for r in rows if r["workload"] == "copy-heavy"}
        assert cp[("2x", "copy_total")]["value"] == "210.0" and cp[("2x", "copy_total_max")]["value"] == "220.0" and \
            cp[("2x", "copy_total")]["sampling"] == "low reasoning effort" and cp[("2x", "copy_total")]["harness"] == COPY_H
        assert not [r for r in csv.DictReader(open(os.path.join(res, "capability-t0.csv"))) if r["workload"] in ("coding-36", "copy-heavy")]
        assert "Planned but not measured" in open(os.path.join(res, "k76.txt")).read()
    print("k76 selftest OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--selftest"]:
        selftest()
    elif a[:1] == ["plan"]:
        if "--lib" in a:
            print(LIB[a[1]])
        else:
            for n, t, args in PLAN[a[1]]:
                print(f"{n}\t{t}\t{COMMON} {args}")
    elif a[:1] == ["plan-coding"]:
        print("\n".join(f"{m} {c}" for m in CODING_MODES for c in CODING_CONC[a[1]]))
    elif a[:1] == ["plan-copy"]:
        print(" ".join(map(str, COPY_STREAMS.get(a[1], ()))))
    elif a[:1] == ["recipe"]:
        f = recipe_facts(a[1])
        print(" ".join(f"{k}={f[k]}" for k in ("release", "alias", "image_tag", "max_num_seqs", "sampling")) if "--short" in a else json.dumps(f, indent=2))
    elif a[:1] == ["serve"]:
        t = open(a[1], errors="replace").read()
        print(json.dumps(serve_facts(t), indent=2)); print("--- excerpt")
        print("\n".join(l[:400] for l in t.splitlines() if SERVE_PAT.search(l)))
    elif a[:1] == ["report"]:
        sys.exit(0 if report(a[1]) else 1)
    else:
        sys.exit(__doc__)
