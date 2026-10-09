#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Render the README hero card, charts, setup icons, quality block and capability table from docs/data/capability.csv.

    uv run scripts/make_charts.py            # or: pip install matplotlib && python3 scripts/make_charts.py

The same script and the same CSV are in both recipe repos (1x and 2x), byte for byte, so the charts compare the
setups the same way in both. Every drawn number is a CSV row, and every row gives the raw file it came from
(`source`, prefixed with the repo: `1x:` or `2x:`). Nothing is interpolated: no point is drawn that was not measured;
the README charts join measured points, the detail charts also break a line at an x that was not measured.

Rules: the shipped release of a setup is its highest version in the CSV. The hero card shows this repo's setup
(tools/dp2 present = the two-Spark repo) from its shipped release. In the two README charts each line is the newest
release with 2+ points; an older release adds hollow points on a dotted line only beyond that line's last x. In the
detail charts, points from any older release are hollow, on a dotted line, and listed in the footnote. Every chart
lists the release, date and harness of each line in its small print.

New measurements (k76-capability-matrix and later) append rows with the same columns; rerun this script to refresh
docs/img/ and the blocks between the `capability-table` and `quality` markers in README.md.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

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
    "1x": dict(name="One Spark", color="#2a78d6", marker="o"),
    "2x": dict(name="Two Sparks, TP=2", color="#eb6834", marker="s"),
    "dp2": dict(name="Two Sparks, DP=2", color="#1baf7a", marker="D"),
}
# Versions are per repo (one-Spark v2.1.0 is not two-Spark v2.1.0), so a release is always written with its setup.
PREFIX = {"1x": "one-Spark ", "2x": "two-Spark ", "dp2": "DP=2 on one-Spark "}

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
    """Release with its old build name, e.g. 'one-Spark v2.1.0 (old name v3e)'; the old name alone if not yet named."""
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


def save(fig, name, card=False):
    """card=True: the figure draws its own rounded card, so everything outside it stays transparent."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        kw = dict(metadata={"Date": None}) if ext == "svg" else dict(dpi=160, metadata={"Software": None})
        kw.update(facecolor="none") if card else kw.update(bbox_inches="tight", pad_inches=0.25)
        fig.savefig(OUT / f"{name}.{ext}", format=ext, **kw)
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT), "svg+png")


def ktok(n):
    """131072 -> '128K' (binary, as the context sizes are set); 245267 -> '245K' (a measured prompt)."""
    return str(n) if n < 1000 else f"{n // 1024}K" if n % 1024 == 0 else f"{n / 1000:.0f}K"


# ---------- charts ----------

def main_line(rows, setup, metric, x, keep=None, **eq):
    """(main, extra): the newest release with 2+ points of `metric` as {x: row}, plus points of older releases beyond
    its largest x (drawn hollow). Nothing is interpolated; a line only joins measured points."""
    groups = defaultdict(dict)
    for r in sorted(rows, key=lambda r: r["date"]):
        if r["setup"] == setup and r["metric"] == metric and r[x] is not None and (keep is None or r[x] in keep) \
                and all(r[k] == v for k, v in eq.items()):
            groups[r["release"]][r[x]] = r
    if not groups:
        return {}, {}
    rels = sorted(groups, key=vkey, reverse=True)
    main = next((groups[v] for v in rels if len(groups[v]) > 1), groups[rels[0]])
    extra = {}
    for v in rels:
        for k, r in groups[v].items():
            if k > max(main) and k not in extra and v != main[max(main)]["release"]:
                extra[k] = r
    return dict(sorted(main.items())), dict(sorted(extra.items()))


def twin(rows, r, metric):
    """The row of another metric from the same run and cell as r (e.g. per-request decode beside the total)."""
    return next((t for t in rows if t["metric"] == metric and all(t[k] == r[k] for k in
                 ("setup", "release", "date", "campaign", "conc", "prompt_tokens", "depth_tokens"))), None)


def provenance(points):
    """'v1.5.0, 2026-10-08, llama-benchy task mode' for each distinct run behind a list of rows."""
    seen = []
    for r in points:
        k = f"{r['release']}, {r['date']}, {r['harness']}"
        if k not in seen:
            seen.append(k)
    return "; ".join(seen)


def draw(ax, setup, main, extra, fmt, dy=0, left=False):
    """Solid line through the main release, hollow points for older ones; the setup's name and last value at the
    end of the line instead of a legend."""
    s = SETUPS[setup]
    xs = list(main)
    ax.plot(xs, [main[k]["value"] for k in xs], "-", color=s["color"], lw=3, marker=s["marker"], ms=9, zorder=3)
    if extra:
        ex = [xs[-1]] + list(extra)
        ax.plot(ex, [main[xs[-1]]["value"]] + [extra[k]["value"] for k in extra], ":", color=s["color"], lw=2,
                zorder=2)
        ax.plot(list(extra), [extra[k]["value"] for k in extra], s["marker"], color=s["color"], mfc=CARD, mew=2,
                ms=9, zorder=3)
    end = extra or main
    k = max(end)
    for text, off, kw in ((s["name"], 10, dict(color=s["color"], fontsize=14)), (fmt(end[k]), -10, dict(color=INK))):
        ax.annotate(text, (k, end[k]["value"]), xytext=(-14 if left else 12, dy + off), textcoords="offset points",
                    va="center", ha="right" if left else "left", fontweight="bold", **{"fontsize": 13.5, **kw}, zorder=4)
    return k, end[k]["value"]


def big(ax):
    """Larger type for the two README charts, so they stay readable when a phone scales them to ~400 px."""
    ax.tick_params(labelsize=13)
    ax.xaxis.label.set_size(13.5)
    ax.yaxis.label.set_size(13.5)


def small_print(fig, lines, y=-0.02):
    fig.text(0.01, y, "\n".join(lines), fontsize=10, color=MUTED, va="top", ha="left", linespacing=1.5)


def chart_users(rows):
    """How fast is it? Combined tok/s against the number of users; each line labelled where it ends."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    big(ax)
    style(ax, "Tokens per second, all users together")
    notes, ymax, ticks = [], 0, [1, 2, 4, 8, 16]
    for setup in has(rows, "tg_total"):
        main, extra = main_line(rows, setup, "tg_total", "conc", keep=ticks, depth_tokens=None)
        k, v = draw(ax, setup, main, extra, lambda r: f"{r['value']:.0f} tok/s" + (
            f", {e['value']:.0f} per user" if (e := twin(rows, r, "tg_req")) and r["conc"] > 1 else ""))
        ax.annotate(f"{main[1]['value']:.0f}", (1, main[1]["value"]), xytext=(-12, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=13.5, color=INK, fontweight="bold") if 1 in main else None
        ymax = max(ymax, v)
        notes.append(f"{SETUPS[setup]['name']}: {provenance(main.values())}"
                     + (f"\n    hollow point: older release {provenance(extra.values())}" if extra else ""))
    if "1x" in has(rows, "tg_total"):
        ax.axvline(8, color=GRID, lw=1.2, ls="--", zorder=1)
        ax.text(8.3, 6, "One Spark takes\nup to 8 at once", fontsize=11.5, color=MUTED, va="bottom")
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(-0.5, 18)
    ax.set_ylim(0, ymax * 1.2)
    ax.set_xlabel("People or agents using it at the same time")
    missing = [SETUPS[s]["name"] for s in SETUPS if s not in has(rows, "tg_total")]
    small_print(fig, ["Each user sends a 2,048-token prompt and gets 512 tokens back; default sampling, thinking on."]
                + notes + ([f"{', '.join(missing)}: not measured on this test yet."] if missing else [])
                + ["Data: docs/data/capability.csv (source file per point)."])
    save(fig, "speed-users")


def chart_first_token(rows):
    """How long until the first word? Seconds to the first token against prompt length, prompt not cached."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    big(ax)
    style(ax, "Seconds until the answer starts")
    notes, ymax = [], 0
    for setup in has(rows, "ttft_s"):
        main, extra = main_line(rows, setup, "ttft_s", "prompt_tokens", conc=1)
        k, v = draw(ax, setup, main, extra, lambda r: f"{r['value']:.0f} s at {ktok(r['prompt_tokens'])}",
                    dy=26 if setup == "1x" else 0, left=setup == "1x")
        if 131072 in main and k != 131072:
            r = main[131072]
            ax.annotate(f"{r['value']:.0f} s", (131072, r["value"]), xytext=(4, -16), textcoords="offset points",
                        ha="left", va="center", fontsize=13.5, color=INK, fontweight="bold")
        ymax = max(ymax, v)
        notes.append(f"{SETUPS[setup]['name']}: {provenance((main | extra).values())}".replace("; ", "\n    "))
    ticks = [8192, 32768, 65536, 131072, 262144]
    ax.set_xticks(ticks, [f"{t // 1024}K" for t in ticks])
    ax.set_xlim(0, 285000)
    ax.set_ylim(0, ymax * 1.2)
    ax.set_xlabel("Prompt length in tokens (a token is about 3/4 of a word)")
    small_print(fig, ["One request with a prompt the server has not seen before.",
                      "Later turns of a chat reuse the cached prompt and start sooner."] + notes
                + ["Data: docs/data/capability.csv (source file per point)."])
    save(fig, "first-token")


def recipe_value(key):
    m = re.search(rf"^\s*{key}:\s*(\d+)", RECIPE.read_text(), re.M) if RECIPE.exists() else None
    return int(m.group(1)) if m else None


def hero(rows):
    """The card at the top of the README: four measured numbers for this repo's setup, run and date in small print."""
    s = REPO
    ship = [r for r in rows if r["setup"] == s and r["release"] == SHIPPED[s]]

    def pick(metric, pool, **eq):
        c = [r for r in pool if r["metric"] == metric and all(r[k] == v for k, v in eq.items())]
        return max(c, key=lambda r: (r["conc"] or 0, r["date"])) if c else None

    pool = ship if pick("tg_total", ship) else [r for r in rows if r["setup"] == s]
    one = pick("tg_total", pool, conc=1, depth_tokens=None)
    many = pick("tg_total", [r for r in pool if r["depth_tokens"] is None])
    tc = pick("gate_tc45", [r for r in rows if r["setup"] == s])
    ctx = recipe_value("max_model_len")
    tiles = [(f"{one['value']:.0f}", "tok/s", "answer speed, one chat"),
             (f"{many['value']:.0f}", "tok/s", f"{many['conc']} chats at once, combined"),
             (f"{ctx // 1000}K", "tokens", "of context in one chat"),
             (f"{tc['value']:.0f}", "/100", "on a tool-calling test")]
    fig = plt.figure(figsize=(10, 4.6))
    fig.patch.set_alpha(0)
    fig.add_artist(FancyBboxPatch((0.006, 0.01), 0.988, 0.98, boxstyle="round,pad=0,rounding_size=0.035",
                                  transform=fig.transFigure, facecolor=CARD, edgecolor=GRID, lw=1.5,
                                  mutation_aspect=10 / 4.6))
    col, r = SETUPS[s]["color"], fig.canvas.get_renderer()
    for i, (num, unit, cap) in enumerate(tiles):
        x, y = (0.07, 0.54)[i % 2], (0.6, 0.27)[i // 2]
        fig.add_artist(plt.Rectangle((x - 0.03, y - 0.035), 0.009, 0.24, transform=fig.transFigure, color=col))
        t = fig.text(x, y + 0.04, num, fontsize=54, fontweight="bold", color=INK, va="baseline")
        w = t.get_window_extent(r).transformed(fig.transFigure.inverted()).width
        fig.text(x + w + 0.012, y + 0.04, unit, fontsize=22, color=INK, va="baseline")
        fig.text(x, y - 0.015, cap, fontsize=20, color=MUTED, va="top")
    where = {"1x": "One DGX Spark", "2x": "Two DGX Sparks, TP=2"}[s]
    fig.text(0.04, 0.9, f"Qwen3.8-Flash-Next  ·  {where}  ·  release {SHIPPED[s]}", fontsize=13, color=INK,
             fontweight="bold", va="center")
    fig.text(0.04, 0.075, f"Speed: release {provenance([one, many])}; each chat sends 2,048 tokens and gets 512 "
             f"back.\nTool calling: TC-45, release {tc['release']}, {tc['date']}.  Context: recipe max_model_len "
             f"{ctx:,}.  Sources: docs/BENCHMARKS.md", fontsize=10, color=MUTED, va="center", linespacing=1.5)
    save(fig, "hero", card=True)


def icons():
    """Small pictures for the setup guide: one Spark, two Sparks as one server (TP=2), two Sparks behind a router."""
    def box(ax, x, y, c):
        ax.add_patch(FancyBboxPatch((x, y), 1.5, 1.0, boxstyle="round,pad=0,rounding_size=0.14", facecolor=c,
                                    edgecolor="none"))
        for i in range(4):
            ax.plot([x + 0.3 + i * 0.3] * 2, [y + 0.28, y + 0.72], color=CARD, lw=2.2, solid_capstyle="round")
    for name, setup in (("setup-one", "1x"), ("setup-tp2", "2x"), ("setup-dp2", "dp2")):
        fig, ax = plt.subplots(figsize=(1.8, 1.2))
        fig.subplots_adjust(0, 0, 1, 1)
        fig.patch.set_alpha(0)
        ax.set_xlim(0, 4.2), ax.set_ylim(0, 2.8), ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.05, 0.05), 4.1, 2.7, boxstyle="round,pad=0,rounding_size=0.3", facecolor=CARD,
                                    edgecolor=GRID, lw=1))
        c = SETUPS[setup]["color"]
        if setup == "1x":
            box(ax, 1.35, 0.9, c)
        elif setup == "2x":   # two Sparks joined into one server by the cable
            box(ax, 0.35, 0.9, c), box(ax, 2.35, 0.9, c)
            ax.plot([1.85, 2.35], [1.4, 1.4], color=INK, lw=5)
        else:                 # a router on top sends each chat to one of two separate Sparks
            box(ax, 0.35, 0.45, c), box(ax, 2.35, 0.45, c)
            ax.add_patch(plt.Circle((2.1, 2.2), 0.28, color=INK))
            for xe in (1.1, 3.1):
                ax.annotate("", (xe, 1.5), (2.1, 2.2), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2,
                                                                      shrinkA=10, shrinkB=2))
        save(fig, name, card=True)


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
             "the control's boot-to-boot spread (one-Spark v1.2.0: run-to-run sd). Cells differ, so bars do not add up.\n"
             "A release is promoted only if no cell at 1-4 requests got slower beyond noise. one-Spark v1.0.0 and two-Spark v1.0.0 "
             "are measured against the builds before them (two Sparks: the shipped recipe of 2026-09-23).\n"
             "cN = N concurrent requests; tg512 = 512-token decode; pp2048 = 2,048-token prefill; 16K = 16K tokens of "
             "context; probe fresh = T=0 decode on a new prompt;\ncounting = a task where MTP accepts nearly every "
             "draft (upper bound). Hatched grey (one-Spark v1.3.0): the KV pool grew from 6 to 14 GiB, so 8 requests at 16K no longer "
             "queue for KV space (22 to 109 tok/s);\ntwo single-boot runs, not paired. "
             "Data: docs/data/capability.csv.", fontsize=9.5, color=MUTED, va="top", linespacing=1.45)
    save(fig, "release-gains")


# ---------- README table ----------

def table(rows):
    """Markdown capability table; every number has a letter for its release, date and run."""
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

    out = ["| | " + " | ".join(s["name"] for s in SETUPS.values()) + " |", "|---|---|---|---|"]

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
RECIPE = ROOT / "recipes" / "qwen3.8-flash-next" / f"qwen3.8-flash-next-{REPO}-dgx-spark.yaml"


def quality(rows):
    """Badges and one plain line per quality check for this repo's setup, from the gate rows of the CSV."""
    def get(metric):
        c = [r for r in rows if r["setup"] == REPO and r["metric"] == metric]
        return max(c, key=lambda r: (vkey(r["release"]), r["date"])) if c else None
    g = {m: get("gate_" + m) for m in ("tc45", "hardmode", "retrieval", "stragglers")}
    if not all(g.values()):
        return "Quality gate: not measured yet."
    badge = lambda label, value: (f"![{label}](https://img.shields.io/badge/" + "-".join(
        x.replace("-", "--").replace("_", "__").replace(" ", "%20").replace("/", "%2F") for x in (label, value))
        + "-2ea44f)")  # noqa: E731
    v = lambda m, f="{:.0f}": f.format(g[m]["value"]) if isinstance(g[m]["value"], float) else g[m]["value"]  # noqa: E731
    retr = v("retrieval").split(" (")[0]
    lines = [
        (badge("tool calling", v("tc45") + "/100"), "When a request requires a tool call, the reply contains one (TC-45, 5 trials)."),
        (badge("multi-step tools", v("hardmode") + "/100"), "Score on 88 hard multi-step tool-use scenarios; a release ships only at 88/100 or more."),
        (badge("long prompts", retr + " up to ~245K tokens"), "Finds 20 facts hidden in a long prompt and returns each through a tool call."
         + (f" ({v('retrieval').split(' (')[1].rstrip(')').replace('seeds', 'prompts')})" if " (" in v("retrieval")
            else "")),
        (badge("stalled requests", v("stragglers").split(",")[0]), "No request falls behind the others while "
         + re.sub(r"c(\d+)-c(\d+)", r"\1 to \2", v("stragglers").split(", ")[-1]) + " are sent at once."),
    ]
    rel = sorted({(g[m]["release"], g[m]["date"]) for m in g})
    return "\n".join(f"- {b} {t}" for b, t in lines) + "\n\nGate run: " + "; ".join(
        f"release {r}, {d}" for r, d in rel) + ". Every release passes this gate before it ships."


def write_blocks(rows):
    readme = ROOT / "README.md"
    text = new = readme.read_text()
    for name, body in (("capability-table", table), ("quality", quality)):
        a, b = f"<!-- {name}:start (scripts/make_charts.py writes this block) -->", f"<!-- {name}:end -->"
        if a not in text:
            print(f"README.md has no {name} markers; block not written")
            continue
        new = re.sub(re.escape(a) + r".*?" + re.escape(b), lambda m: f"{a}\n{body(rows)}\n{b}", new, flags=re.S)
        print(f"wrote README.md {name} block")
    if new != text:
        readme.write_text(new)


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
    hero(rows)
    chart_users(rows)
    chart_first_token(rows)
    icons()
    chart_depth(rows)
    chart_coding(rows)
    chart_agents(rows)
    chart_gains(rows)
    write_blocks(rows)
