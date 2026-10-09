#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Render the README charts and the capability table from docs/data/capability.csv.

    uv run scripts/make_charts.py            # or: pip install matplotlib && python3 scripts/make_charts.py

The same script and the same CSV are in both recipe repos (1x and 2x), byte for byte, so the charts compare the
setups the same way in both. Every drawn number is a CSV row, and every row names the raw file it came from
(`source`, prefixed with the repo: `1x:` or `2x:`). Nothing is interpolated: a line is broken where a point was not
measured.

Rules: the shipped release of a setup is its highest version in the CSV. Its points are filled; points from any
older release are hollow, on a dotted line, and named in the footnote. For each x value the newest row wins, and an
older release is drawn only where the newest one has no point.

New measurements (k76-capability-matrix and later) append rows with the same columns; rerun this script to refresh
docs/img/ and the table between the `capability-table` markers in README.md.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "docs" / "data" / "capability.csv"
OUT = ROOT / "docs" / "img"
GH = {"1x": "https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark/tree/main/",
      "2x": "https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/tree/main/"}
QUEUED = "full grid queued (tp-2 #128)"

# White card on both GitHub themes: dark-theme readers see a light card with its own ink, so one file serves both.
CARD, INK, MUTED, GRID = "#ffffff", "#1f2328", "#424a53", "#d1d9e0"
# Colourblind-checked categorical slots (blue, orange, aqua: worst all-pairs CVD dE 9.2). Each setup also has its
# own marker, and bars carry direct labels, so identity never rests on colour alone.
SETUPS = {
    "1x": dict(name="1× Spark", color="#2a78d6", marker="o"),
    "2x": dict(name="2× Spark, TP=2", color="#eb6834", marker="s"),
    "dp2": dict(name="2× Spark, DP=2", color="#1baf7a", marker="D"),
}
PREFIX = {"1x": "1× ", "2x": "2× ", "dp2": "DP=2 on 1× "}   # versions are per repo: 1× v2.1.0 is not 2× v2.1.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": GRID, "svg.fonttype": "path",
    "figure.facecolor": CARD, "axes.facecolor": CARD, "savefig.facecolor": CARD, "legend.frameon": False,
    "svg.hashsalt": "make_charts", "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


# ---------- data ----------

def vkey(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)) if v else ()


def load():
    rows = list(csv.DictReader(CSV.open()))
    for r in rows:
        for k in ("conc", "prompt_tokens", "depth_tokens"):
            r[k] = int(r[k]) if r[k] else None
        try:
            r["value"] = float(r["value"])
        except ValueError:
            pass
    global SHIPPED
    SHIPPED = {s: max((r["release"] for r in rows if r["setup"] == s), key=vkey, default="") for s in SETUPS}
    return rows


def old(r):
    return r["release"] != SHIPPED[r["setup"]]


def label(r, prefix=True):
    """Release with its old build name, e.g. '1× v2.1.0 (old name v3e)'; the old name alone if not yet named."""
    name = f"{r['release']} (old name {r['alias']})" if r["release"] else r["alias"]
    return (PREFIX[r["setup"]] if prefix else "") + name


def has(rows, metric):
    return [s for s in SETUPS if any(r["setup"] == s and r["metric"] == metric for r in rows)]


def by_release(rows, setup, metric, x, keep=None, **eq):
    """[{x: row}] per (release, campaign), newest first; older ones keep only the x values nothing newer has."""
    rel = defaultdict(list)
    for r in rows:
        if r["setup"] == setup and r["metric"] == metric and all(r[k] == v for k, v in eq.items()) \
                and (keep is None or r[x] in keep):
            rel[(r["release"], r["alias"], r["campaign"])].append(r)
    out, seen = [], set()
    for key in sorted(rel, key=lambda k: (vkey(k[0]), max(r["date"] for r in rel[k])), reverse=True):
        pts = {r[x]: r for r in sorted(rel[key], key=lambda r: r["date"]) if r[x] not in seen}
        if pts:
            out.append(dict(sorted(pts.items())))
            seen |= set(pts)
    return out


def segments(pts, keep):
    """Split {x: row} into runs with no missing x of `keep` in between, so no line crosses an unmeasured point."""
    order = sorted(keep) if keep else sorted(pts)
    runs, cur = [], []
    for x in order:
        if x in pts:
            cur.append(x)
        elif cur:
            runs.append(cur)
            cur = []
    return runs + ([cur] if cur else [])


def line(ax, rows, setup, metric, x, notes, keep=None, fmt="{:.0f}", dy=0, label_end=True, **eq):
    s = SETUPS[setup]
    groups = by_release(rows, setup, metric, x, keep, **eq)
    for pts in groups:
        r0 = next(iter(pts.values()))
        o = old(r0)
        for run in segments(pts, keep or set(pts)):
            ax.plot(run, [pts[k]["value"] for k in run], ":" if o else "-", color=s["color"], lw=1.5 if o else 2.4,
                    marker=s["marker"], ms=6.5 if o else 7.5, mfc=CARD if o else s["color"], mew=1.8, zorder=2)
        notes.setdefault(setup, {}).setdefault((label(r0), r0["date"], r0["harness"], o), None)
    if label_end:
        for pts in groups:
            k = max(pts)
            v = pts[k]["value"]
            ax.annotate(fmt(v) if callable(fmt) else fmt.format(v), (k, v), xytext=(7, dy),
                        textcoords="offset points", va="center", fontsize=10, color=INK, fontweight="bold")


def legend(fig, setups, y=-0.02, hollow=False, patches=False):
    if patches:
        h = [plt.Rectangle((0, 0), 1, 1, color=SETUPS[s]["color"], label=SETUPS[s]["name"]) for s in setups]
    else:
        h = [Line2D([], [], color=SETUPS[s]["color"], marker=SETUPS[s]["marker"], lw=2.4, ms=7.5,
                    label=SETUPS[s]["name"]) for s in setups]
    if hollow:
        h.append(Line2D([], [], color=MUTED, marker="o", mfc=CARD, mew=1.8, ls=":", lw=1.5, ms=6.5,
                        label="hollow, dotted: older release (shipped one not measured here)"))
    fig.legend(handles=h, loc="upper center", ncol=len(h), bbox_to_anchor=(0.5, y), fontsize=10)


def footnote(fig, notes, extra="", y=-0.1):
    parts = [f"{rel}, {date}, {harness}" + (" (older release)" if o else "")
             for items in notes.values() for (rel, date, harness, o) in items]
    text = "\n".join(parts + ([extra] if extra else []) + ["Data: docs/data/capability.csv (source file per point)."])
    fig.text(0.01, y, text, fontsize=9.5, color=MUTED, va="top", ha="left", linespacing=1.45)


def style(ax, ylabel=None, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if ylabel:
        ax.set_ylabel(ylabel)


def title(fig, text, y=1.03):
    fig.suptitle(text, x=0.01, ha="left", fontsize=14, fontweight="bold", y=y)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        kw = dict(metadata={"Date": None}) if ext == "svg" else dict(dpi=160, metadata={"Software": None})
        fig.savefig(OUT / f"{name}.{ext}", format=ext, bbox_inches="tight", pad_inches=0.25, **kw)
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT), "svg+png")


def ktok(n):
    return f"{n // 1024}K" if n >= 1024 else str(n)


# ---------- charts ----------

def chart_concurrency(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    notes, keep = {}, {1, 2, 4, 8, 16}
    for ax, metric, ttl in ((axes[0], "tg_req", "Per request"), (axes[1], "tg_total", "All requests together")):
        style(ax, "Decode tok/s")
        for setup in has(rows, metric):
            line(ax, rows, setup, metric, "conc", notes, keep=keep, depth_tokens=None)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(keep), [str(k) for k in sorted(keep)])
        ax.set_xlim(0.8, 24)
        ax.set_ylim(0, None)
        ax.set_xlabel("Concurrent requests")
        ax.set_title(ttl)
    if "1x" in has(rows, "tg_total"):
        r8 = max((r for r in rows if r["setup"] == "1x" and r["metric"] == "tg_total" and r["conc"] == 8),
                 key=lambda r: r["date"])
        axes[1].annotate("1× stops at 8\n(max_num_seqs 8)", (8, r8["value"]), xytext=(10, -44),
                         textcoords="offset points", fontsize=9, color=MUTED)
    title(fig, "Decode speed by concurrent requests")
    legend(fig, has(rows, "tg_total"), hollow=True)
    footnote(fig, notes, "512 tokens out after a 2,048-token prompt, no context, server sampling (T=1.0), thinking on. "
             "Lines break where a point was not measured." + ("" if "dp2" in has(rows, "tg_total")
                                                              else f"\nDP=2: not on this grid yet, {QUEUED}."))
    save(fig, "decode-concurrency")


def chart_prefill(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    notes = {}
    for ax, metric, ttl, ylab in ((axes[0], "pp", "Prefill speed", "Prefill tok/s, 1 request"),
                                  (axes[1], "ttft_s", "Time to first token", "Seconds, 1 request, uncached")):
        style(ax, ylab)
        for setup in has(rows, metric):
            line(ax, rows, setup, metric, "prompt_tokens", notes, conc=1,
                 fmt="{:,.0f}" if metric == "pp" else "{:.0f} s",
                 dy={"1x": -8, "2x": 8}.get(setup, 0) if metric == "pp" else {"1x": 7, "2x": -7}.get(setup, 0))
        ax.set_xscale("log", base=2)
        ticks = sorted({r["prompt_tokens"] for r in rows if r["metric"] == metric})
        ax.set_xticks(ticks, [ktok(t) if t % 1024 == 0 else f"{t / 1000:.0f}K" for t in ticks])
        ax.set_xlim(ticks[0] / 1.35, ticks[-1] * 1.5)
        ax.set_xlabel("Prompt length (tokens)")
        ax.set_title(ttl)
        if metric == "pp":
            ax.set_ylim(0, None)
        else:
            ax.set_yscale("log")
            yt = [1, 2, 5, 10, 30, 60, 120]
            ax.set_yticks(yt, [f"{t:g}" for t in yt])
            ax.minorticks_off()
    title(fig, "Prompt processing by prompt length")
    legend(fig, has(rows, "pp"), hollow=True)
    st = sorted({r["stat"].split(",")[0] for r in rows if r["metric"] == "pp"})
    footnote(fig, notes, "One request, prompt not in the prefix cache. Samples per point: " + "; ".join(st) + ".\n"
             "The 2,048-token llama-benchy prefill is in the capability table (another harness, not drawn here)."
             + ("" if "dp2" in has(rows, "pp") else "\nDP=2: each request is prefilled by one 1× replica; not "
                f"measured separately, {QUEUED}."))
    save(fig, "prefill-ttft")


def chart_depth(rows):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    notes = {}
    depths = sorted({r["depth_tokens"] for r in rows if r["metric"] == "decode_depth_total"})
    for ax, c in zip(axes, (1, 4, 8)):
        style(ax, "Decode tok/s, all requests" if c == 1 else None)
        for setup in has(rows, "decode_depth_total"):
            line(ax, rows, setup, "decode_depth_total", "depth_tokens", notes, keep=set(depths), conc=c)
        ax.set_xticks(depths, [ktok(d) if d else "0" for d in depths])
        ax.set_xlim(-0.08 * depths[-1], depths[-1] * 1.25)
        ax.set_xlabel("Context in the prompt (tokens)")
        ax.set_title(f"{c} request{'s' if c > 1 else ''}")
    axes[0].set_ylim(0, None)
    title(fig, "Decode speed with context already in the prompt")
    legend(fig, has(rows, "decode_depth_total"), hollow=any(old(r) for r in rows if r["metric"] == "decode_depth_total"))
    gap = [d for d in (131072, 262144) if d not in depths]
    harness = {r["harness"] for r in rows if r["metric"] == "decode_depth_total"}
    footnote(fig, notes, "Sustained decode with the context cached, server default sampling. "
             + ("Compare points within this chart: its 30 s steady-state window reads 10-30% above the 512-token "
                "runs of the concurrency chart.\n" if "llm-inference-bench 0.7.6" in harness else "")
             + (f"Not measured yet at {' and '.join(ktok(d) for d in gap)} context: {QUEUED}." if gap else ""))
    save(fig, "decode-depth")


def chart_coding(rows):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    notes = {}
    style(ax, "Decode tok/s, 1 request")
    modes = ["T=0, thinking off", "server defaults, thinking on"]
    sets = has(rows, "coding_probe_median")
    w = 0.8 / len(sets)
    for i, setup in enumerate(sets):
        s = SETUPS[setup]
        for j, mode in enumerate(modes):
            def get(m):
                c = [r for r in rows if r["setup"] == setup and r["metric"] == m and r["sampling"] == mode]
                return max(c, key=lambda r: r["date"]) if c else None
            r, lo, hi = get("coding_probe_median"), get("coding_probe_min"), get("coding_probe_max")
            if not r:
                continue
            x = j + (i - (len(sets) - 1) / 2) * w
            o = old(r)
            ax.bar(x, r["value"], w * 0.86, color=CARD if o else s["color"], edgecolor=s["color"], linewidth=2,
                   hatch="//" if o else None, zorder=2)
            if lo and hi:
                ax.plot([x, x], [lo["value"], hi["value"]], color=INK, lw=1.3, zorder=3)
                for v in (lo["value"], hi["value"]):
                    ax.plot([x - w * 0.12, x + w * 0.12], [v, v], color=INK, lw=1.3, zorder=3)
            top = hi["value"] if hi else r["value"]
            ax.text(x, top + 2, f"{PREFIX[setup].strip()} {r['value']:.0f}", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color=INK, zorder=4)
            notes.setdefault(setup, {}).setdefault((label(r), r["date"], r["harness"], o), None)
    ax.set_xticks([0, 1], ["T=0, thinking off", "server defaults, thinking on"])
    ax.set_ylim(0, 135)
    title(fig, "Coding, one request at a time: 36 prompts in Python, C++, Rust and Go", y=1.0)
    legend(fig, sets, patches=True)
    hits = "; ".join(f"{PREFIX[r['setup']].strip()} {r['stat'].split(', ')[1]}" for r in rows
                     if r["metric"] == "coding_probe_median" and r["sampling"].startswith("server"))
    footnote(fig, notes, "Bar and number = median of the 36 prompts, line = min to max. Hatched = older release. "
             "Up to 768 tokens out; decode tok/s = (tokens - 1) / time after the first token.\n"
             f"Server defaults (thinking on) hit max_tokens inside reasoning on most prompts ({hits}).")
    save(fig, "coding")


def chart_agents(rows):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.subplots_adjust(left=0.3)
    notes = {}
    style(ax, axis="x")
    wl = ["agent8", "agent16", "long-4", "long-8", "long-12", "long-16"]
    names = {"agent8": "8 sessions × 6 turns", "agent16": "16 sessions × 4 turns", "long-4": "4 sessions × 2 turns",
             "long-8": "8 sessions × 2 turns", "long-12": "12 sessions × 2 turns", "long-16": "16 sessions × 2 turns"}
    start = {"agent": "~32K-token start", "long-": "~128K-token start"}
    sets = has(rows, "agent_wall_s")
    h = 0.8 / len(sets)
    for i, setup in enumerate(sets):
        s = SETUPS[setup]
        pts = {r["stat"]: r for r in rows if r["setup"] == setup and r["metric"] == "agent_wall_s"}
        for j, k in enumerate(wl):
            if k not in pts:
                continue
            r = pts[k]
            y = j + (i - (len(sets) - 1) / 2) * h
            ax.barh(y, r["value"], h * 0.88, color=s["color"], zorder=2)
            ax.text(r["value"] + 8, y, f"{r['value']:.0f} s  {'TP=2' if setup == '2x' else 'DP=2'}", va="center",
                    fontsize=9.5, color=INK)
            notes.setdefault(setup, {}).setdefault((label(r), r["date"], r["harness"], old(r)), None)
    ax.set_yticks(range(len(wl)), [f"{names[k]}, {start['agent' if k.startswith('agent') else 'long-']}"
                                   for k in wl], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Wall time for the whole workload, s (shorter is better)")
    ax.set_xlim(0, max(r["value"] for r in rows if r["metric"] == "agent_wall_s") * 1.18)
    title(fig, "Two Sparks, many agent sessions at once: TP=2 or DP=2?", y=1.0)
    legend(fig, sets, patches=True)
    pt = [r["value"] for r in rows if r["metric"] == "agent_prompt_tokens"]
    ot = [r["value"] for r in rows if r["metric"] == "agent_output_tokens"]
    footnote(fig, notes, "All sessions start together; every turn resends the conversation with tools on, "
             "temperature 0.6, thinking off.\n"
             f"Prompt work dominates: {min(pt) / 1e6:.1f}M to {max(pt) / 1e6:.1f}M prompt tokens against "
             f"{min(ot):,.0f} to {max(ot):,.0f} output tokens per workload, so this compares prefill capacity.")
    save(fig, "agents")


def chart_gains(rows):
    sets = [s for s in ("1x", "2x") if any(r["setup"] == s and r["metric"] == "release_gain_pct" for r in rows)]
    data = {s: sorted((r for r in rows if r["setup"] == s and r["metric"] == "release_gain_pct"),
                      key=lambda r: vkey(r["release"])) for s in sets}
    n = max(len(v) for v in data.values())
    fig, axes = plt.subplots(1, len(sets), figsize=(12.5, 0.62 * n + 1.6), squeeze=False,
                             gridspec_kw={"wspace": 0.5})
    cap = 60
    for ax, s in zip(axes[0], sets):
        style(ax, axis="x")
        col = SETUPS[s]["color"]
        for i, r in enumerate(data[s]):
            v = r["value"]
            unpaired = "not paired" in r["stat"]
            ax.barh(i, min(v, cap), 0.62, color=GRID if unpaired else col, hatch="//" if unpaired else None,
                    edgecolor=col if unpaired else "none", zorder=2)
            noise = f" (noise {float(r['sd']):.1f}%)" if r["sd"] else ""
            cell = r["stat"].split(" (")[0].split(":")[0]
            more = r["stat"][len(cell):].strip(" :") if not unpaired else ""
            if v > cap:
                for dx in (-3.5, -1.5):   # break mark: the bar is cut at the axis cap
                    ax.plot([cap + dx - 0.6, cap + dx + 0.6], [i + 0.4, i - 0.4], color=INK, lw=1.4, zorder=3)
            if v > 32:
                ax.text(1.2, i, f"+{v:.0f}%  {cell}" if v > cap else f"+{v:.1f}%  {cell}", va="center", fontsize=10,
                        color=INK if unpaired else CARD, fontweight="bold", zorder=4,
                        bbox=dict(facecolor=CARD, edgecolor="none", pad=1.5) if unpaired else None)
                tail = (f"{more} " if more else "") + noise.strip()
                if tail:
                    ax.text(min(v, cap) + 1.2, i, tail, va="center", fontsize=9, color=INK, zorder=3)
            else:
                ax.text(v + 1.2, i, f"+{v:.1f}%  {cell}{(' ' + more) if more else ''}{noise}", va="center",
                        fontsize=9.5, color=INK, zorder=3)
        ax.set_yticks(range(len(data[s])), [f"{label(r, prefix=False).replace(' (', chr(10) + '(')}"
                                            for r in data[s]], fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlim(0, cap + 4)
        ax.set_xlabel("Gain over the previous release, %")
        ax.set_title(SETUPS[s]["name"])
    title(fig, "Largest gain beyond noise in each release's own A/B", y=1.02)
    fig.text(0.01, -0.02, "Each bar: that release against the one before it, same Sparks, boots alternating; noise = "
             "the control's boot-to-boot spread (1× v1.2.0: run-to-run sd). Cells differ, so bars do not add up.\n"
             "A release is promoted only if no cell at 1-4 requests got slower beyond noise. 1× v1.0.0 and 2× v1.0.0 "
             "are measured against the builds before them (2×: the shipped recipe of 2026-09-23).\n"
             "cN = N concurrent requests; tg512 = 512-token decode; pp2048 = 2,048-token prefill; 16K = 16K tokens of "
             "context; probe fresh = T=0 decode on a new prompt;\ncounting = a task where MTP accepts nearly every "
             "draft (upper bound). Hatched grey (1× v1.3.0): the KV pool grew from 6 to 14 GiB, so 8 requests at 16K no longer "
             "queue for KV space (22 to 109 tok/s);\ntwo single-boot runs, not paired. "
             "Data: docs/data/capability.csv.", fontsize=9.5, color=MUTED, va="top", linespacing=1.45)
    save(fig, "release-gains")


# ---------- README table ----------

def table(rows):
    """Markdown capability table; every number carries a letter for its release, date and run."""
    keys = []

    def cell(setup, metrics, fmt="{:,.1f}", **eq):
        pts = [r for r in rows if r["setup"] == setup and r["metric"] in metrics
               and all(r[k] == v for k, v in eq.items())]
        if not pts:
            return None
        r = max(pts, key=lambda r: (not old(r), r["date"]))   # shipped release first, then the newest run
        k = (label(r), r["date"], r["campaign"], r["harness"], r["source"])
        if k[:4] not in [x[:4] for x in keys]:
            keys.append(k)
        i = [x[:4] for x in keys].index(k[:4])
        v = r["value"]
        return (fmt.format(v) if isinstance(v, float) else v) + f" <sup>{chr(97 + i)}</sup>"

    def join(*cells, sep=" / "):
        return sep.join(c or "–" for c in cells) if any(cells) else "not measured"

    out = ["| | 1× Spark | 2× Spark, TP=2 | 2× Spark, DP=2 |", "|---|---|---|---|"]

    def row(name, f):
        out.append(f"| {name} | " + " | ".join(f(s) for s in SETUPS) + " |")

    tg = lambda m, c: lambda s: join(cell(s, [m], conc=c, depth_tokens=None))  # noqa: E731
    row("Decode tok/s, 1 request", tg("tg_total", 1))
    for c in (4, 8, 16):
        def f(s, c=c):
            if s == "1x" and c > 8:
                return "over the cap (max_num_seqs 8)"
            return join(cell(s, ["tg_req"], conc=c, depth_tokens=None), cell(s, ["tg_total"], conc=c, depth_tokens=None))
        row(f"Decode tok/s, {c} requests: each / total", f)
    sizes = (2048, 16384, 65536, 131072)
    row("Prefill tok/s at 2K / 16K / 64K / 128K prompt, 1 request",
        lambda s: join(*(cell(s, ["pp", "pp_benchy"], "{:,.0f}", conc=1, prompt_tokens=p) for p in sizes)))
    row("Time to first token at 2K / 16K / 64K / 128K, uncached, s",
        lambda s: join(*(cell(s, ["ttft_s", "ttft_benchy_s"], "{:.1f}", conc=1, prompt_tokens=p) for p in sizes)))
    row("Decode, mean ms per token at 1 / 8 requests (MTP emits several tokens per step)",
        lambda s: join(cell(s, ["itl_p50_ms"], "{:.0f}", conc=1, depth_tokens=0),
                       cell(s, ["itl_p50_ms"], "{:.0f}", conc=8, depth_tokens=0)))
    row("Gap between streamed chunks p50 at 1 / 8 requests, ms",
        lambda s: join(cell(s, ["chunk_gap_p50_ms"], "{:.0f}", conc=1, depth_tokens=0),
                       cell(s, ["chunk_gap_p50_ms"], "{:.0f}", conc=8, depth_tokens=0)))
    row("Decode tok/s total at 0 → 64K context, 1 request / 4 requests",
        lambda s: join(*(join(cell(s, ["decode_depth_total"], conc=c, depth_tokens=0),
                              cell(s, ["decode_depth_total"], conc=c, depth_tokens=65536), sep=" → ")
                         for c in (1, 4))) if cell(s, ["decode_depth_total"], conc=1, depth_tokens=0)
        else "not measured")
    row("Max context per request", lambda s: "262,144 (recipe)")
    row("KV pool, tokens", lambda s: join(cell(s, ["kv_tokens"], "{:,.0f}")))
    row("Requests of 262,144 tokens the pool holds (vLLM's count)", lambda s: join(cell(s, ["kv_conc_262k"], "{:.2f}")))
    row("Requests that fit the KV pool at 16K / 64K / 128K",
        lambda s: join(*(cell(s, ["sessions_fit"], "{:.1f}", prompt_tokens=p) for p in (16384, 65536, 131072))))
    row("Quality gate: hardmode / TC-45 / retrieval to ~245K / stragglers",
        lambda s: join(cell(s, ["gate_hardmode"], "{:.0f}"), cell(s, ["gate_tc45"], "{:.0f}"),
                       cell(s, ["gate_retrieval"], "{}"), cell(s, ["gate_stragglers"], "{}")))
    out += ["", "Releases and runs behind the numbers:", ""]
    for i, (rel, date, camp, harness, src) in enumerate(keys):
        repo, path = src.split(":", 1)
        d = path if path.endswith(".md") else str(Path(path).parent) + "/"
        link = d if repo == REPO else (GH[repo].replace("/tree/", "/blob/") if d.endswith(".md") else GH[repo]) + d
        out.append(f"- <sup>{chr(97 + i)}</sup> {rel}, {date}, {camp}" + (f", {harness}" if harness else "")
                   + f" ([files]({link}))")
    return "\n".join(out)


REPO = "2x" if (ROOT / "tools" / "dp2").exists() else "1x"   # the DP=2 router ships only in the tp-2 repo


def write_table(rows):
    readme = ROOT / "README.md"
    text = readme.read_text()
    a, b = "<!-- capability-table:start (scripts/make_charts.py writes this block) -->", "<!-- capability-table:end -->"
    if a not in text:
        print("README.md has no capability-table markers; table not written")
        return
    new = re.sub(re.escape(a) + r".*?" + re.escape(b), lambda m: f"{a}\n{table(rows)}\n{b}", text, flags=re.S)
    if new != text:
        readme.write_text(new)
    print("wrote README.md capability table")


def selftest():
    """Line breaks at gaps; an older release only fills x values the newest lacks."""
    global SHIPPED
    SHIPPED = {"2x": "v3.1.0"}
    assert segments({1: 0, 4: 0, 8: 0}, {1, 2, 4, 8, 16}) == [[1], [4, 8]]
    mk = lambda rel, x, d: dict(setup="2x", metric="m", release=rel, alias=rel, campaign="c", conc=x, date=d, value=1.0)
    rows = [mk("v3.1.0", 1, "2026-10-08"), mk("v3.1.0", 4, "2026-10-08"), mk("v3.0.0", 1, "2026-10-01"),
            mk("v3.0.0", 16, "2026-10-01")]
    g = by_release(rows, "2x", "m", "conc")
    assert [sorted(p) for p in g] == [[1, 4], [16]] and old(g[1][16]) and not old(g[0][1])
    print("selftest ok")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
        raise SystemExit
    rows = load()
    chart_concurrency(rows)
    chart_prefill(rows)
    chart_depth(rows)
    chart_coding(rows)
    chart_agents(rows)
    chart_gains(rows)
    write_table(rows)
