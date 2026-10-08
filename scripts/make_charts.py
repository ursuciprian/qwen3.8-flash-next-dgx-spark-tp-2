#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Render the README charts from committed files under results/ into docs/img/*.svg.

    uv run scripts/make_charts.py            # or: pip install matplotlib && python3 scripts/make_charts.py

Every number drawn comes from a file listed in SRC. Transparent background and mid-tone colours, so the
same SVG reads on GitHub's light and dark themes.
"""
import json, math
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "docs" / "img"

T1 = RES / "tp1-v3d-20261005"   # single-Spark v3d: gate files and bench/ (one boot on dgx-01)
LIB = RES / "lib-bench-20261005"
HC = RES / "high-conc-k46b-20261005"   # max_num_seqs 32 runs (k46b): not the shipped cap, no quality gate at this cap

SRC = {
    # 2x Spark (TP=2), b1.4: coding grid (mean of two candidate boots).
    "tp2_verdict": RES / "b1.4-20261001/b14.json",
    # Counting on b1.4, 2026-10-05: one file per level, 5 rounds each, every round saved (max round is drawn).
    "tp2_count": sorted((LIB / "tp2-b1.4").glob("count-c*.json")),
    "tp2_benchy": [RES / "b1.4-20261001/benchy/cand1-task.csv", RES / "b1.4-20261001/benchy/cand2-task.csv"],
    # Copy-heavy on b1.4, 2026-10-04 showcase run.
    "tp2_copy": [RES / "showcase-20261004/A/copy-streams.json"],
    # llm-inference-bench decode, 30 s per cell, c1/c4/c8 at 0/16K/64K context.
    "tp2_lib": LIB / "tp2-b1.4/lib-decode.json",
    # 1x Spark (TP=1), v3d: coding grid, copy-heavy, counting (5 rounds, every round saved), llm-inference-bench.
    "tp1_benchy": T1 / "bench/task.csv",
    "tp1_copy": [T1 / "bench/copy-streams.json"],
    "tp1_count": sorted((T1 / "bench").glob("count-c*.json")),
    "tp1_lib": T1 / "bench/lib-decode.json",
    # Promoted 2x Spark builds, in order. The first verdict's baseline is the 2026-09-23 shipped build.
    "builds": [
        ("b1", "09-25", RES / "b1-20260925/verdict-oldb12x-on.json"),
        ("b1.1", "09-26", RES / "b1.1-20260926/verdict-b11-prefixdrop.json"),
        ("b1.2", "09-27", RES / "b1.2-20260927/b12-hcq.json"),
        ("b1.3", "09-29", RES / "b1.3-20260929/b13.json"),
        ("b1.4", "10-01", RES / "b1.4-20261001/b14.json"),
        # b1.6 (#115): its A/B is k71 (Thunderdome, llama-benchy 4 runs): arm boots' coding grid; no counting sweep
        ("b1.6", "10-08", [RES / f"k71-tp2-refit-pinned-plans-20261008-0921/screen/arm-p{i}/task.csv" for i in (1, 2)]),
    ],
    # Promoted 1x Spark builds, in order: each build's own llama-benchy coding grid (one boot, 3 runs).
    "tp1_builds": [
        ("v2", "10-02", RES / "tp1-v2-20261002/bench-task.csv"),
        ("v3a", "10-04", RES / "tp1-v3a-20261004/bench-task.csv"),
        ("v3b", "10-04", RES / "tp1-v3b-20261004/bench-task.csv"),
        ("v3c", "10-05", RES / "tp1-v3c-20261005/bench/task.csv"),
        ("v3d", "10-05", T1 / "bench/task.csv"),
        ("v3e", "10-08", RES / "tp1-v3e-hf-20261008/bench/task.csv"),
    ],
    "tp2_fidelity": RES / "b1.4-20261001/regate-b14-20261001/gate-seed7.txt",
    # High concurrency, max_num_seqs 32 (one boot per setup): counting, copy-heavy, coding at c16/c32.
    "hc": {"2× Spark (TP=2), b1.4": HC / "tp2-s32", "1× Spark (TP=1), v3d": HC / "v3d-s32"},
    "tp2_tc45": RES / "b1.4-20261001/gate/tc45-cand1.txt",
    "tp1_fidelity": T1 / "gate-summary.txt",
    "tp1_gate": T1 / "gate-summary.txt",
}

# Neutral ink that keeps >= 3:1 contrast on both #ffffff and #0d1117; series hues are mid-tone.
INK, MUTED, GRID = "#768390", "#8b949e", "#8b949e40"
C_COPY, C_COUNT, C_CODE, C_CODE16 = "#3987e5", "#1baf7a", "#e0662f", "#d4a017"
C_TP2, C_TP1 = "#3987e5", "#e0662f"
C_C1, C_C4, C_C8 = "#3987e5", "#1baf7a", "#e0662f"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": GRID, "svg.fonttype": "path",
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
    "legend.frameon": False, "svg.hashsalt": "make_charts", "axes.spines.top": False, "axes.spines.right": False,
})


# ---------- loaders ----------

def benchy(path):
    """llama-benchy markdown table -> {test: (total t/s, accept/draft)}."""
    rows = {}
    for line in Path(path).read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 10 or not re.search(r"\(c\d+\)", cells[1]):
            continue
        rows[cells[1]] = (float(cells[2].split("±")[0]), float(cells[9]))
    return rows


def grid(prefix, rows):
    """'tg512' or 'tg512 @ d16384' -> {c: t/s}."""
    pat = re.compile(re.escape(prefix) + r" \(c(\d+)\)$")
    return {int(m.group(1)): v[0] for k, v in rows.items() if (m := pat.match(k))}


def verdict_means(path, key, prefix=""):
    d = json.loads(Path(path).read_text())[key]
    return {int(k[len(prefix) + 1:]): v["cand_mean"] for k, v in d.items() if k.startswith(prefix + "c")}


def copy_best(paths):
    """Copy-heavy files -> {streams: max round window tok/s over all files}."""
    out = {}
    for p in paths:
        for r in json.loads(Path(p).read_text())["rounds"]:
            out[r["n"]] = max(out.get(r["n"], 0), r["window"]["tok_s"])
    return out


def count_best(paths):
    """Counting sweep files -> {c: max round agg tok/s}. Files that saved every round carry agg_tok_s_max."""
    out = {}
    for p in paths:
        for r in json.loads(Path(p).read_text())["rows"]:
            out[r["c"]] = max(out.get(r["c"], 0), r.get("agg_tok_s_max", r["agg_tok_s"]))
    return out


def lib_decode(path):
    """llm-inference-bench JSON -> {c: {context tokens: aggregate tok/s}} over the cells that ran."""
    out = {}
    for r in json.loads(Path(path).read_text())["results"]:
        if r.get("aggregate_tps") and not r.get("failure_reason"):
            out.setdefault(r["concurrency"], {})[r["context_tokens"]] = r["aggregate_tps"]
    return out


def fidelity(path):
    out = []
    for line in Path(path).read_text().splitlines():
        m = re.search(r"actual_tokens\s+(\d+)\s+\|\s+exact\s+(\d+)", line)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


def load():
    v = SRC["tp2_verdict"]
    tp2 = {
        "copy": copy_best(SRC["tp2_copy"]),
        "count": count_best(SRC["tp2_count"]),
        "code": verdict_means(v, "coding_grid_pct_diff", "d0_"),
        "code16": verdict_means(v, "coding_grid_pct_diff", "d16384_"),
    }
    t1 = benchy(SRC["tp1_benchy"])
    tp1 = {
        "copy": copy_best(SRC["tp1_copy"]),
        "count": count_best(SRC["tp1_count"]),
        "code": grid("tg512", t1),
        "code16": grid("tg512 @ d16384", t1),
    }
    t2 = [benchy(p) for p in SRC["tp2_benchy"]]
    prefill = {
        "2× Spark": ([r["pp2048 (c1)"][0] for r in t2], [r["ctx_pp @ d16384 (c1)"][0] for r in t2]),
        "1× Spark": ([t1["pp2048 (c1)"][0]], [t1["ctx_pp @ d16384 (c1)"][0]]),
    }
    builds = [("shipped", "09-23", None)] + SRC["builds"]
    hist = []
    for name, date, path in builds:
        if isinstance(path, list):  # llama-benchy grids of the arm boots, mean; counting not measured
            g = [benchy(x) for x in path]
            m = lambda k: sum(r[k][0] for r in g) / len(g)
            hist.append((name, date, m("tg512 (c1)"), m("tg512 (c8)"), math.nan, math.nan))
            continue
        p = path or SRC["builds"][0][2]
        col = "base_mean" if path is None else "cand_mean"
        d = json.loads(p.read_text())
        hist.append((name, date,
                     d["coding_grid_pct_diff"]["d0_c1"][col], d["coding_grid_pct_diff"]["d0_c8"][col],
                     d["counting_pct_diff"]["c1"][col], d["counting_pct_diff"]["c8"][col]))
    hist1 = []
    for name, date, path in SRC["tp1_builds"]:
        g = benchy(path)
        hist1.append((name, date, g["tg512 (c1)"][0], g["tg512 (c8)"][0],
                      g["tg512 @ d16384 (c1)"][0], g["tg512 @ d16384 (c8)"][0]))
    lib = {"2× Spark (TP=2), b1.4": lib_decode(SRC["tp2_lib"]), "1× Spark (TP=1), v3d": lib_decode(SRC["tp1_lib"])}
    for name, data in (("tp2", tp2), ("tp1", tp1), ("lib", lib)):
        for k, s in data.items():
            assert s, f"{name}.{k}: no data parsed"
    return tp2, tp1, prefill, hist, hist1, lib


# ---------- charts ----------

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def style_ax(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


SERIES = [
    ("copy", "Copy-heavy (high acceptance, ~4.9 tok/step)", C_COPY, "-"),
    ("count", "Counting (high acceptance, ~5.0 tok/step)", C_COUNT, "-"),
    ("code", "Coding, benchy tg512 (~2.7–3.5 tok/step)", C_CODE, "-"),
    ("code16", "Coding at 16k cached context", C_CODE16, "--"),
]


def chart_throughput(tp2, tp1):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, gridspec_kw={"width_ratios": [5, 4]})
    for ax, data, title, cmax in ((axes[0], tp2, "2× Spark (TP=2)", 16), (axes[1], tp1, "1× Spark (TP=1)", 8)):
        style_ax(ax)
        for key, label, color, ls in SERIES:
            pts = sorted((c, y) for c, y in data[key].items() if c <= cmax)
            xs, ys = zip(*pts)
            ax.plot(xs, ys, ls, color=color, lw=2.2, marker="o", ms=5, label=label)
            note = " (KV pool full)" if key == "code16" and ys[-1] < ys[-2] / 2 else ""
            dy = {"code": 5, "code16": -6}.get(key, 0)   # the two coding end points can sit close together
            ax.annotate(f"{ys[-1]:.0f}{note}", (xs[-1], ys[-1]), xytext=(6, dy), textcoords="offset points",
                        va="center", fontsize=10, color=INK, fontweight="bold")
        ax.set_xscale("log", base=2)
        ticks = [1, 2, 4, 8, 16][: 5 if cmax == 16 else 4]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_xlim(0.85, cmax * 1.45)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
        ax.set_xlabel("Concurrent requests")
    axes[0].set_ylabel("Aggregate decode tok/s")
    axes[0].set_ylim(0, None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), fontsize=10)
    fig.text(0.0, -0.22, "Copy-heavy: max of 3 rounds per task count, low thinking effort. "
             "Counting: T=0, thinking off, max of 5 rounds per level.\n"
             "Coding: llama-benchy task mode, T=1.0, thinking on (2×: mean of two boots; 1×: one boot), 3 runs.\n"
             "Builds: 2× b1.4; 1× v3d (one Spark, one boot). Raw files under results/.",
             fontsize=8.5, color=MUTED)
    save(fig, "throughput.svg")


def chart_prefill(prefill):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    style_ax(ax)
    groups = ["2,048-token prompt", "16k context fill"]
    w = 0.36
    for i, (name, color) in enumerate((("2× Spark", C_TP2), ("1× Spark", C_TP1))):
        for g, vals in enumerate(prefill[name]):
            x = g + (i - 0.5) * w
            m = statistics.mean(vals)
            ax.bar(x, m, w * 0.92, color=color, label=name if g == 0 else None)
            txt = f"{m:,.0f}" if len(vals) == 1 else f"{min(vals):,.0f}–{max(vals):,.0f}"
            ax.text(x, m + 40, txt, ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel("Prefill tok/s, 1 request")
    ax.set_ylim(0, 3500)
    ax.legend(loc="upper left", ncol=2, fontsize=10)
    fig.text(0.0, -0.06, "llama-benchy pp2048 and ctx_pp at depth 16,384. 2× Spark b1.4 (two boots, range shown); "
             "1× Spark v3d.", fontsize=8.5, color=MUTED)
    save(fig, "prefill.svg")


def chart_builds(hist, hist1):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    rows = (
        (hist, "2× Spark", ((4, "Counting", C_COUNT), (2, "Coding, benchy tg512", C_CODE)),
                           ((5, "Counting", C_COUNT), (3, "Coding, benchy tg512", C_CODE))),
        (hist1, "1× Spark", ((2, "Coding, benchy tg512", C_CODE), (4, "Coding at 16k cached context", C_CODE16)),
                            ((3, "Coding, benchy tg512", C_CODE), (5, "Coding at 16k cached context", C_CODE16))),
    )
    for r, (data, setup, s1, s8) in enumerate(rows):
        labels = [f"{n}\n{d}" for n, d, *_ in data]
        xs = range(len(data))
        for ax, title, series in ((axes[r][0], "1 request", s1), (axes[r][1], "8 concurrent requests", s8)):
            style_ax(ax)
            for idx, name, color in series:
                ys = [h[idx] for h in data]
                ax.plot(xs, ys, color=color, lw=2.2, marker="o", ms=5, label=name,
                        ls="--" if color == C_CODE16 else "-")
                last = max(i for i, y in enumerate(ys) if not math.isnan(y))
                for x, y in ((0, ys[0]), (last, ys[last])):
                    # the lower of the two series at this point is labelled below its marker, so labels never cross
                    below = any(h2[x] > y or (h2[x] == y and color == C_CODE16)
                                for h2 in ([h[i] for h in data] for i, _, _ in series if i != idx))
                    ax.annotate(f"{y:.0f}", (x, y), xytext=(0, -16 if below else 8), textcoords="offset points",
                                ha="center", fontsize=10, color=INK, fontweight="bold")
            ax.set_xticks(list(xs), labels, fontsize=9)
            ax.set_ylim(0, None)
            ax.set_title(f"{setup}, {title}", loc="left", fontsize=13, fontweight="bold", color=INK)
        axes[r][0].set_ylabel("Aggregate decode tok/s")
        axes[r][1].legend(loc="center right" if r == 0 else "center left", fontsize=9.5)
    fig.tight_layout(h_pad=2.2)
    fig.text(0.0, -0.05, "Promoted builds in order (2026). 2× Spark: each value is that build's A/B, mean of two boots; "
             "'shipped' is the baseline boots of the b1 A/B.\nb1.6: the arm boots of its k71 A/B "
             "(llama-benchy 4 runs, b1.4 control 72.9 / 181.1 on the same day); counting was not run.\n1× Spark: each build's llama-benchy coding grid "
             "(one boot, 3 runs, T=1.0); single cells vary by up to ~10% between runs. "
             "At 16k with 8 requests, v2-v3b\nran out of KV pool (6 GiB); v3c to v3e have 14 GiB. "
             "Every build passed the quality gate.",
             fontsize=8.5, color=MUTED)
    save(fig, "build-history.svg")


def chart_depth(lib):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    ctx = [0, 16384, 65536]
    for ax, (title, data) in zip(axes, lib.items()):
        style_ax(ax)
        for c, color in ((8, C_C8), (4, C_C4), (1, C_C1)):
            pts = [(i, data.get(c, {}).get(x)) for i, x in enumerate(ctx)]
            pts = [(i, y) for i, y in pts if y]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, lw=2.2, marker="o", ms=5, label=f"{c} request{'s' if c > 1 else ''}")
            for x, y in pts:
                ax.annotate(f"{y:.0f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center",
                            fontsize=9.5, color=INK, fontweight="bold")
        ax.set_xticks(range(len(ctx)), ["0", "16K", "64K"])
        ax.set_xlim(-0.3, len(ctx) - 0.7)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
        ax.set_xlabel("Context already in the prompt (tokens)")
    axes[0].set_ylabel("Aggregate decode tok/s")
    axes[0].set_ylim(0, None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=10)
    fig.text(0.0, -0.14, "llm-inference-bench, 30 s of sustained decode per cell, server default sampling, "
             "one boot per setup. Raw files: results/lib-bench-20261005/ and results/tp1-v3d-20261005/bench/.",
             fontsize=8.5, color=MUTED)
    save(fig, "depth.svg")


def benchy_tg_max(path):
    """llama-benchy JSON -> {concurrency: max run tg tok/s total} at depth 0."""
    out = {}
    for b in json.loads(Path(path).read_text())["benchmarks"]:
        if b["context_size"] == 0 and not b["is_context_prefill_phase"]:
            out[b["concurrency"]] = max(b["tg_throughput"]["values"])
    return out


def chart_concurrency():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (title, d) in zip(axes, SRC["hc"].items()):
        style_ax(ax)
        series = (("Copy-heavy, max of 3 rounds", C_COPY, copy_best([d / "copy-streams.json"])),
                  ("Counting, max of 5 rounds", C_COUNT, count_best(sorted(d.glob("count-c*.json")))),
                  ("Coding, benchy tg512, max of 3 runs", C_CODE, benchy_tg_max(d / "benchy.json")))
        for label, color, data in series:
            pts = sorted((c, y) for c, y in data.items() if c in (1, 8, 16, 32))
            assert pts, f"{title} {label}: no data"
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, lw=2.2, marker="o", ms=5, label=label)
            ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=10, color=INK, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 8, 16, 32], ["1", "8", "16", "32"])
        ax.set_xlim(0.85, 32 * 1.5)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
        ax.set_xlabel("Concurrent requests")
    axes[0].set_ylabel("Aggregate decode tok/s")
    axes[0].set_ylim(0, None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=10)
    fig.text(0.0, -0.17, "Measured with max_num_seqs 32 (the shipped recipes use 16 on 2× and 8 on 1×); the quality gate "
             "was not run at this cap.\nOne boot per setup; coding measured at c16 and c32 only. Raw files: "
             "results/high-conc-k46b-20261005/.", fontsize=8.5, color=MUTED)
    save(fig, "concurrency.svg")


def chart_quality():
    t2f, t1f = fidelity(SRC["tp2_fidelity"]), fidelity(SRC["tp1_fidelity"])
    v = json.loads(SRC["tp2_verdict"].read_text())
    gate1 = SRC["tp1_gate"].read_text()
    hard1 = re.search(r"Quality:\s+(\d+)/100", gate1).group(1)
    tc1 = re.search(r"Score:\s+([\d.]+) ±", gate1).group(1)
    tc2 = re.search(r"Score:\s+([\d.]+) ±", SRC["tp2_tc45"].read_text()).group(1)
    rows = [
        ("Hard multi-step tool use (88 scenarios)", f"{min(v['quality_gate']['hardmode_scores'])}/100",
         f"{hard1}/100", "≥ 88"),
        ("tool_choice=required (TC-45, 5 trials)", f"{float(tc2):.0f}/100", f"{float(tc1):.0f}/100", "-"),
    ]
    for (tok2, ex2), (tok1, ex1) in zip(t2f[:4], t1f[:4]):
        rows.append((f"Tool-call retrieval at {round(tok2, -3) / 1000:.0f}k tokens", f"{ex2}/20", f"{ex1}/20",
                     "20/20"))
    stragglers2 = "none" if not v["quality_gate"]["straggler_violations"] else "FAIL"
    stragglers1 = "none" if "preemptions +0" in gate1 else "check"
    rows.append(("Batch stragglers", f"{stragglers2} (c5–c16)", f"{stragglers1} (c8–c16)", "none"))

    fig, ax = plt.subplots(figsize=(10, 0.46 * (len(rows) + 1) + 0.4))
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99)
    cols = (0.0, 0.45, 0.65, 0.87)
    for x, h in zip(cols, ("Check", "2× Spark b1.4", "1× Spark v3d", "Threshold")):
        ax.text(x, len(rows), h, fontweight="bold", fontsize=11, color=INK, va="center")
    for i, (name, a, b, thr) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.axhline(y + 0.5, color=GRID, lw=0.8)
        ax.text(cols[0], y, name, fontsize=10.5, color=INK, va="center")
        for x, val in ((cols[1], a), (cols[2], b)):
            ax.text(x, y, "✓", fontsize=12, color=C_COUNT, va="center", fontweight="bold")
            ax.text(x + 0.03, y, val, fontsize=10.5, color=INK, va="center", fontweight="bold")
        ax.text(cols[3], y, thr, fontsize=9.5, color=MUTED, va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) + 0.5)
    fig.text(0.0, -0.02, "Retrieval depths are the logged prompt sizes (labels 8k/32k/64k/128k in the probe). "
             "2× Spark: re-gate boot, seed 7. A build that misses any check is not promoted.",
             fontsize=8.5, color=MUTED)
    save(fig, "quality-gate.svg")


if __name__ == "__main__":
    tp2, tp1, prefill, hist, hist1, lib = load()
    chart_throughput(tp2, tp1)
    chart_prefill(prefill)
    chart_builds(hist, hist1)
    chart_depth(lib)
    chart_quality()
    chart_concurrency()
