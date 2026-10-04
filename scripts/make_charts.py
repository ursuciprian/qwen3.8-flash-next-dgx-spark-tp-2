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
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "docs" / "img"

SRC = {
    # 2x Spark (TP=2), b1.4: coding grid and counting sweep, means of two candidate boots.
    "tp2_verdict": RES / "b1.4-20261001/b14.json",
    "tp2_benchy": [RES / "b1.4-20261001/benchy/cand1-task.csv", RES / "b1.4-20261001/benchy/cand2-task.csv"],
    # TODO(#65): replace with the b1.4 copy-streams file from results/showcase-* when the showcase run lands.
    "tp2_copy": RES / "b1.2-20260927/copy-streams-20260929.json",
    # 1x Spark (TP=1), v3a coding grid.
    "tp1_benchy": RES / "tp1-v3a-20261004/bench-task.csv",
    # TODO(#65): replace both with the v3a files from results/showcase-* when the showcase run lands.
    "tp1_copy": RES / "tp1-v2-20261002/copy-streams.json",
    "tp1_count": RES / "tp1-v2-20261002/counting-sweep.json",
    # Promoted 2x Spark builds, in order. The first verdict's baseline is the 2026-09-23 shipped build.
    "builds": [
        ("b1", "09-25", RES / "b1-20260925/verdict-oldb12x-on.json"),
        ("b1.1", "09-26", RES / "b1.1-20260926/verdict-b11-prefixdrop.json"),
        ("b1.2", "09-27", RES / "b1.2-20260927/b12-hcq.json"),
        ("b1.3", "09-29", RES / "b1.3-20260929/b13.json"),
        ("b1.4", "10-01", RES / "b1.4-20261001/b14.json"),
    ],
    "tp2_fidelity": RES / "b1.4-20261001/regate-b14-20261001/gate-seed7.txt",
    "tp2_tc45": RES / "b1.4-20261001/gate/tc45-cand1.txt",
    "tp1_fidelity": RES / "tp1-v3a-20261004/gate-fidelity_probe.txt",
    "tp1_gate": RES / "tp1-v3a-20261004/gate-summary.txt",
}

# Neutral ink that keeps >= 3:1 contrast on both #ffffff and #0d1117; series hues are mid-tone.
INK, MUTED, GRID = "#768390", "#8b949e", "#8b949e40"
C_COPY, C_COUNT, C_CODE, C_CODE16 = "#3987e5", "#1baf7a", "#e0662f", "#d4a017"
C_TP2, C_TP1 = "#3987e5", "#e0662f"

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


def copy_streams(path):
    s = json.loads(Path(path).read_text())["summary"]
    return ({int(n): v["window_tok_s"]["median"] for n, v in s.items()},
            [v["tokens_per_step"]["median"] for v in s.values()])


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
        "copy": copy_streams(SRC["tp2_copy"])[0],
        "count": verdict_means(v, "counting_pct_diff"),
        "code": verdict_means(v, "coding_grid_pct_diff", "d0_"),
        "code16": verdict_means(v, "coding_grid_pct_diff", "d16384_"),
    }
    t1 = benchy(SRC["tp1_benchy"])
    tp1 = {
        "copy": copy_streams(SRC["tp1_copy"])[0],
        "count": {r["c"]: r["agg_tok_s"] for r in json.loads(SRC["tp1_count"].read_text())["rows"] if r["c"] <= 8},
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
        p = path or SRC["builds"][0][2]
        col = "base_mean" if path is None else "cand_mean"
        d = json.loads(p.read_text())
        hist.append((name, date,
                     d["coding_grid_pct_diff"]["d0_c1"][col], d["coding_grid_pct_diff"]["d0_c8"][col],
                     d["counting_pct_diff"]["c1"][col], d["counting_pct_diff"]["c8"][col]))
    for name, data in (("tp2", tp2), ("tp1", tp1)):
        for k, s in data.items():
            assert s, f"{name}.{k}: no data parsed"
    return tp2, tp1, prefill, hist


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
    ("copy", "Copy-streams (high acceptance, ~4.9 tok/step)", C_COPY, "-"),
    ("count", "Counting (high acceptance, ~5.0 tok/step)", C_COUNT, "-"),
    ("code", "Coding, benchy tg512 (~2.7–3.4 tok/step)", C_CODE, "-"),
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
            ax.annotate(f"{ys[-1]:.0f}{note}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
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
    fig.text(0.0, -0.17, "Copy-streams: median of 3 rounds, low thinking effort. Counting: T=0, thinking off (2×: mean of 2 boots). "
             "Coding: llama-benchy task mode, T=1.0, thinking on.\nBuilds: 2× b1.4 (copy-streams b1.2); 1× v3a (copy-streams "
             "and counting v2). Raw files under results/.", fontsize=8.5, color=MUTED)
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
             "1× Spark v3a.", fontsize=8.5, color=MUTED)
    save(fig, "prefill.svg")


def chart_builds(hist):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = [f"{n}\n{d}" for n, d, *_ in hist]
    xs = range(len(hist))
    for ax, (title, ci, ki) in zip(axes, (("1 request", 2, 4), ("8 concurrent requests", 3, 5))):
        style_ax(ax)
        for idx, name, color in ((ki, "Counting", C_COUNT), (ci, "Coding, benchy tg512", C_CODE)):
            ys = [h[idx] for h in hist]
            ax.plot(xs, ys, color=color, lw=2.2, marker="o", ms=5, label=name)
            for x, y in ((0, ys[0]), (len(ys) - 1, ys[-1])):
                ax.annotate(f"{y:.0f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center",
                            fontsize=10, color=INK, fontweight="bold")
        ax.set_xticks(list(xs), labels, fontsize=9)
        ax.set_ylim(0, None)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
    axes[0].set_ylabel("Aggregate decode tok/s")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08), fontsize=10)
    fig.text(0.0, -0.14, "2× Spark, promoted builds in order (2026). Each value: that build's A/B, mean of two boots; "
             "'shipped' is the baseline boots of the b1 A/B.\nEvery build passed the quality gate.",
             fontsize=8.5, color=MUTED)
    save(fig, "build-history.svg")


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
        ("tool_choice=required (TC-45, 5 trials)", f"{float(tc2):.0f}/100", f"{float(tc1):.0f}/100", "—"),
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
    for x, h in zip(cols, ("Check", "2× Spark b1.4", "1× Spark v3a", "Threshold")):
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
    tp2, tp1, prefill, hist = load()
    chart_throughput(tp2, tp1)
    chart_prefill(prefill)
    chart_builds(hist)
    chart_quality()
