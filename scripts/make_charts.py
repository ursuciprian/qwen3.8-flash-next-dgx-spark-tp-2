#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Render README charts and generated blocks from docs/data/capability.csv.
The hero.png banner is separately authored artwork and is never regenerated here.

    uv run scripts/make_charts.py            # or: pip install matplotlib && python3 scripts/make_charts.py
    uv run scripts/make_charts.py --selftest

The same script and the same CSV are in both recipe repos (1x and 2x), byte for byte, so the charts compare the
setups the same way in both. Every drawn number is a CSV row, and every row gives the raw file it came from
(`source`, prefixed with the repo: `1x:` or `2x:`). Nothing is interpolated: a line only joins measured points.

Output: docs/img/<name>-light.svg and <name>-dark.svg (the README shows them through <picture>, so GitHub picks the
one that matches the reader's theme) and docs/img/<name>.png (light, a fallback for pages that do not show SVG).
Type is Inter from docs/fonts/ (SIL OFL 1.1, see docs/fonts/LICENSE.txt), drawn as outlines, so the images look the
same on every machine; without the font files the script falls back to DejaVu Sans.

Rules:
- The shipped release of a setup is its highest version in the CSV. This repo's setup (tools/dp2 present = the
  two-Spark repo) is drawn in full colour, the other setups lighter.
- A chart line is the newest release (per harness) with 3+ points of that metric, else the one with the most
  points; points of older releases beyond its last x are drawn hollow and named in the caption.
- Each image has a caption block in the README with the release, date, harness and statistic of every line.
- CSV columns: setup, release, alias, date, campaign, metric, conc, prompt_tokens, depth_tokens, value, sd, stat,
  sampling, harness, source, workload. `sd` (optional) draws an error bar of +-1 sd; `stat` says what the value is
  (for example "mean of 2 boots x 4 runs, sd between boots"). Release-history rows are ordinary rows of older
  releases (campaign "release history" or the release's own A/B); nothing else is needed.

New measurements (k76-capability-matrix and later) append rows with the same columns; a rerun refreshes docs/img/
and every block between `<!-- name:start ... -->` and `<!-- name:end -->` markers in README.md.
"""
import csv
import re
import sys
from urllib.parse import quote
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402  (installed with matplotlib)
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "docs" / "data" / "capability.csv"
OUT = ROOT / "docs" / "img"
FONTS = ROOT / "docs" / "fonts"
GH = {"1x": "https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark/tree/main/",
      "2x": "https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/tree/main/"}
REPO = "2x" if (ROOT / "tools" / "dp2").exists() else "1x"   # the DP=2 router ships only in the tp-2 repo
RECIPE = ROOT / "recipes" / "qwen3.8-flash-next" / f"qwen3.8-flash-next-{REPO}-dgx-spark.yaml"
DATA_LINK = "[docs/data/capability.csv](docs/data/capability.csv)"

for f in sorted(FONTS.glob("Inter-*.ttf")):
    font_manager.fontManager.addfont(str(f))
FAMILY = "Inter" if any(FONTS.glob("Inter-*.ttf")) else "DejaVu Sans"

# One palette for both repos. Setup colours are the first three slots of a colour-blind-checked categorical set
# (worst all-pairs CVD dE 9.2 light / 9.4 dark); each mode has its own steps. Text never wears a series colour.
THEMES = {
    "light": dict(bg="#ffffff", card="#f6f8fa", tile="#ffffff", ink="#1f2328", muted="#59636e", faint="#818b98",
                  grid="#e8ebef", axis="#c8d1da", ring="#ffffff",
                  setup={"1x": "#2a78d6", "2x": "#eb6834", "dp2": "#1baf7a"}),
    "dark": dict(bg="#0d1117", card="#151b23", tile="#0d1117", ink="#e6edf3", muted="#9198a1", faint="#6e7681",
                 grid="#21272f", axis="#3d444d", ring="#0d1117",
                 setup={"1x": "#3987e5", "2x": "#d95926", "dp2": "#199e70"}),
}
T = THEMES["light"]
SETUPS = {"1x": dict(name="One Spark", short="1x", marker="o"),
          "2x": dict(name="Two Sparks, TP=2", short="TP=2", marker="s"),
          "dp2": dict(name="Two Sparks, DP=2", short="DP=2", marker="D")}
# Versions are per repo (one-Spark v2.1.0 is not two-Spark v2.1.0), so a release is always written with its setup.
PREFIX = {"1x": "one-Spark ", "2x": "two-Spark ", "dp2": "DP=2 on one-Spark "}
W = 7.2   # inches; set per image by render(): 7.2 for a full-width chart shown at 640 px, 5.0 for a half-width
          # chart shown at 420 px, so 11.5 pt type lands at 13-14 px on the page


def theme(mode):
    global T
    T = THEMES[mode]
    plt.rcParams.update({
        "font.family": FAMILY, "font.size": 12, "text.color": T["ink"], "axes.labelcolor": T["muted"],
        "xtick.color": T["muted"], "ytick.color": T["muted"], "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
        "axes.edgecolor": T["axis"], "svg.fonttype": "path", "svg.hashsalt": "make_charts",
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none", "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.labelsize": 11.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def color(setup):
    return T["setup"][setup]


def tint(c, f):
    """The colour c mixed with white (f = share of c): a lighter step of the same hue in both themes."""
    return matplotlib.colors.to_hex(tuple(a * f + (1 - f) for a in matplotlib.colors.to_rgb(c)))


def own(setup):
    return setup == REPO or (REPO == "2x" and setup == "dp2")


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
        r["sd"] = float(r["sd"]) if r.get("sd") else None
    global SHIPPED
    SHIPPED = {s: max((r["release"] for r in rows if r["setup"] == s), key=vkey, default="") for s in SETUPS}
    return rows


def old(r):
    return r["release"] != SHIPPED[r["setup"]]


def label(r, prefix=True):
    """Release with its old build name, e.g. 'one-Spark v2.1.0 (old name v3e)'."""
    name = (f"{r['release']} (old name {r['alias']})" if r["alias"] else r["release"]) if r["release"] else r["alias"]
    return (PREFIX[r["setup"]] if prefix else "") + name


def has(rows, metric):
    metrics = metric if isinstance(metric, (list, tuple)) else [metric]
    return [s for s in SETUPS if any(r["setup"] == s and r["metric"] in metrics for r in rows)]


def numeric(r):
    return isinstance(r["value"], float)


def main_line(rows, setup, metrics, x, keep=None, **eq):
    """(main, extra): {x: row} of the line to draw and {x: row} of older points beyond its last x.
    Rows are grouped by (release, harness); the line is the newest group with 3+ points, else the largest group.
    Within a group a later date wins at the same x. Nothing is interpolated."""
    metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
    groups = defaultdict(dict)
    for r in sorted(rows, key=lambda r: r["date"]):
        if r["setup"] == setup and r["metric"] in metrics and r[x] is not None and numeric(r) \
                and (keep is None or r[x] in keep) and all(r[k] == v for k, v in eq.items()):
            groups[(r["release"], r["harness"])][r[x]] = r
    if not groups:
        return {}, {}
    order = sorted(groups, key=lambda g: (vkey(g[0]), len(groups[g])), reverse=True)
    best = next((g for g in order if len(groups[g]) >= 3), max(order, key=lambda g: (len(groups[g]), vkey(g[0]))))
    main = groups[best]
    extra = {}
    for g in order:
        if g == best:
            continue
        for k, r in groups[g].items():
            if k > max(main) and k not in extra:
                extra[k] = r
    return dict(sorted(main.items())), dict(sorted(extra.items()))


def pick(rows, setup, metrics, **eq):
    """The row to quote for one cell: shipped release first, then the newest release, then the newest date."""
    metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
    c = [r for r in rows if r["setup"] == setup and r["metric"] in metrics and all(r[k] == v for k, v in eq.items())]
    return max(c, key=lambda r: (not old(r), vkey(r["release"]), r["date"])) if c else None


def twin(rows, r, metric):
    """The row of another metric from the same run and cell as r (e.g. per-request decode beside the total)."""
    return next((t for t in rows if t["metric"] == metric and all(t[k] == r[k] for k in
                 ("setup", "release", "date", "campaign", "conc", "prompt_tokens", "depth_tokens"))), None)


def runs(points, stat=True):
    """'v2.0.0, 2026-10-09, llama-benchy task mode, mean of 2 boots x 4 runs' for each distinct run in points."""
    seen = []
    for r in points:
        k = f"{r['release']}, {r['date']}, {r['harness']}" + (f", {r['stat']}" if stat and r["stat"] and numeric(r) else "")
        if k not in seen:
            seen.append(k)
    return "; ".join(seen)


def ktok(n):
    """131072 -> '128K' (binary, as the context sizes are set); 245267 -> '245K' (a measured prompt)."""
    return str(n) if n < 1000 else f"{n // 1024}K" if n % 1024 == 0 else f"{n / 1000:.0f}K"


def recipe_value(key, setup=None):
    """A number from this repo's recipe (or the one-Spark recipe, which both repos carry)."""
    path = RECIPE.parent / f"qwen3.8-flash-next-{setup or REPO}-dgx-spark.yaml"
    m = re.search(rf"^\s*{key}:\s*(\d+)", path.read_text(), re.M) if path.exists() else None
    return int(m.group(1)) if m else None


def coding_one(rows, setup, mode="T=0, thinking off"):
    """Median decode tok/s of the coding corpus, one chat at a time: the newest release that has it."""
    c = [r for r in rows if r["setup"] == setup and r["metric"] == "coding_probe_median" and r["conc"] == 1
         and r["sampling"] == mode]
    return max(c, key=lambda r: (vkey(r["release"]), r["date"])) if c else None


# ---------- drawing helpers ----------

def figure(h, title, subtitle, legend=None, panels=1, hspace=0.5, top_extra=0.0, foot=True):
    """A figure W x h inches with a question as title, a subtitle, an optional legend row and `panels` stacked axes.
    Margins are fixed in inches, so the header looks the same in every chart."""
    fig = plt.figure(figsize=(W, h))
    head = 0.85 + (0.4 if legend else 0) + top_extra
    fig.text(0.2 / W, 1 - 0.30 / h, title, fontsize=15, fontweight="semibold", color=T["ink"], va="baseline")
    fig.text(0.2 / W, 1 - 0.56 / h, subtitle, fontsize=11, color=T["muted"], va="baseline")
    if legend:
        x = 0.2 / W
        r = fig.canvas.get_renderer()
        for name, kw in legend:
            fig.add_artist(Line2D([x, x + 0.26 / W], [1 - 0.86 / h] * 2, transform=fig.transFigure, lw=2.4,
                                  **{k: v for k, v in kw.items() if k in ("color", "ls", "alpha")}))
            if kw.get("marker"):
                fig.add_artist(Line2D([x + 0.13 / W], [1 - 0.86 / h], transform=fig.transFigure, ls="",
                                      marker=kw["marker"], ms=7, color=kw["color"], mfc=kw.get("mfc", kw["color"]),
                                      mew=1.6, alpha=kw.get("alpha", 1)))
            t = fig.text(x + 0.34 / W, 1 - 0.86 / h, name, fontsize=11.5, color=T["ink"], va="center")
            x += 0.34 / W + t.get_window_extent(r).width / fig.dpi / W + 0.3 / W
    axes = fig.subplots(panels, 1, sharex=True, squeeze=False)[:, 0] if panels else []
    fig.subplots_adjust(left=0.7 / W, right=1 - 0.8 / W, top=1 - head / h, bottom=(0.9 if foot else 0.6) / h,
                        hspace=hspace)
    for ax in axes:
        ax.grid(axis="y", color=T["grid"], lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=6)
        ax.spines["bottom"].set_color(T["axis"])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    return fig, list(axes)


def panel_title(ax, text):
    ax.text(0, 1.04, text, transform=ax.transAxes, fontsize=12, fontweight="semibold", color=T["ink"], va="bottom")


def line(ax, setup, pts, extra=None, err=True):
    """A setup's line: full colour and weight for this repo's setup, lighter for the others; +-1 sd bars where the
    CSV has sd; hollow points (dotted link) for older releases beyond the line's end."""
    c, o = color(setup), own(setup)
    a, lw = (1, 2.4) if o else (0.55, 1.8)
    xs = list(pts)
    ys = [pts[k]["value"] for k in xs]
    ax.plot(xs, ys, "-", color=c, lw=lw, alpha=a, zorder=3 if o else 2)
    ax.plot(xs, ys, SETUPS[setup]["marker"], color=c, ms=7.5 if o else 6.5, mec=T["ring"], mew=1.6, alpha=a,
            zorder=4 if o else 3)
    if err:
        for k in xs:
            if pts[k]["sd"]:
                ax.errorbar([k], [pts[k]["value"]], yerr=[pts[k]["sd"]], fmt="none", ecolor=c, elinewidth=1.2,
                            capsize=3, alpha=a, zorder=2)
    if extra:
        k = xs[-1]
        for x, r in extra.items():
            ax.plot([k, x], [pts[k]["value"], r["value"]], ":", color=c, lw=1.6, alpha=a, zorder=1)
            ax.plot([x], [r["value"]], SETUPS[setup]["marker"], color=c, mfc=T["ring"], mew=1.8, ms=7, alpha=a,
                    zorder=3)


def labels(ax, items, side="right", gap=15):
    """Direct value labels at line ends: (x, y, text, bold[, below]). Labels on the same side are pushed apart
    vertically so none overlap; `below` puts a label under its point (another line passes just above). Labels
    never wear the series colour."""
    fig = ax.figure
    fig.canvas.draw()
    shift = 16 * fig.dpi / 72
    pts = sorted(((ax.transData.transform((x, y))[1] - (shift if rest and rest[0] else 0), x, y, t, b)
                  for x, y, t, b, *rest in items))
    placed = []
    for py, x, y, t, b in pts:
        py2 = max(py, placed[-1][0] + gap * fig.dpi / 72) if placed else py
        placed.append((py2, x, y, t, b, ax.transData.transform((x, y))[1]))
    for py2, x, y, t, b, py in placed:
        dy = (py2 - py) * 72 / fig.dpi
        ax.annotate(t, (x, y), xytext=(9 if side == "right" else -9, dy), textcoords="offset points",
                    ha="left" if side == "right" else "right", va="center", fontsize=11.5,
                    fontweight="semibold" if b else "normal", color=T["ink"] if b else T["muted"], zorder=6,
                    annotation_clip=False, bbox=dict(boxstyle="round,pad=0.15", facecolor=T["bg"], edgecolor="none"))


def conc_axis(ax, ticks, label="Requests at the same time"):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(ticks[0] / 1.7, ticks[-1] * 1.25)   # room for the start labels left of x=1
    ax.set_xlabel(label)


def footer(fig, h, text):
    fig.text(0.2 / W, 0.16 / h, text, fontsize=9.5, color=T["faint"], va="baseline")


IMAGES, CAPTIONS = {}, {}


DISPLAY = {}


def render(name, fn, rows, alt, w=7.2, px=640):
    """Draw fn(rows) once per theme at w inches wide and save docs/img/<name>-light.svg, -dark.svg and <name>.png
    (light); the README shows it px wide."""
    global W
    W, DISPLAY[name] = w, px
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in THEMES:
        theme(mode)
        fig = fn(rows)
        fig.savefig(OUT / f"{name}-{mode}.svg", format="svg", dpi=200, metadata={"Date": None})
        if mode == "light":
            fig.savefig(OUT / f"{name}.png", format="png", dpi=200, facecolor=T["bg"], metadata={"Software": None})
        plt.close(fig)
    IMAGES[name] = alt
    print("wrote", (OUT / name).relative_to(ROOT), "-light.svg -dark.svg .png")


def caption(name, lines, short):
    """`short`: the one line shown under the image; `lines`: runs, method and data, in a collapsed block."""
    CAPTIONS[name] = (short, "<br>".join(lines))


def picture(name):
    if name == "hero":
        return f'<img src="docs/img/hero.png" alt="{IMAGES[name]}" width="{DISPLAY[name]}">'
    return (f'<picture><source media="(prefers-color-scheme: dark)" srcset="docs/img/{name}-dark.svg">'
            f'<img src="docs/img/{name}-light.svg" alt="{IMAGES[name]}" width="{DISPLAY[name]}"></picture>')


def figure_block(*names):
    """One image, or two side by side (inline, so they wrap one under the other on a phone), then one short line
    and the sources in a collapsed block."""
    pics = '<p align="center">\n' + "\n".join(picture(n) for n in names) + "\n</p>"
    tag = (lambda i: ("Left: ", "Right: ")[i]) if len(names) > 1 else (lambda i: "")
    short = " ".join(tag(i) + CAPTIONS[n][0] for i, n in enumerate(names))
    long = "<br><br>".join(tag(i) + CAPTIONS[n][1] for i, n in enumerate(names))
    return (f"{pics}\n\n<sub>{short}</sub>\n\n<details>\n<summary><sub>Runs, method and raw data</sub></summary>\n\n"
            f"<sub>{long}</sub>\n\n</details>")


# ---------- hero ----------

def hero_tiles(rows):
    s = REPO
    ship = [r for r in rows if r["setup"] == s and r["release"] == SHIPPED[s]]
    pool = ship if any(r["metric"] == "tg_total" for r in ship) else [r for r in rows if r["setup"] == s]

    def top(metric, pool, **eq):
        c = [r for r in pool if r["metric"] == metric and r["depth_tokens"] is None and numeric(r)
             and all(r[k] == v for k, v in eq.items())]
        return max(c, key=lambda r: (r["conc"] or 0, r["date"])) if c else None

    one, many = top("tg_total", pool, conc=1), top("tg_total", pool)
    tc = pick(rows, s, "gate_tc45")
    code = coding_one(rows, s)
    ctx = recipe_value("max_model_len")
    rel = lambda r: f" · {r['release']}" if old(r) else ""  # noqa: E731
    tiles = []
    if code:
        tiles.append(("Coding, one chat", f"{code['value']:.0f}", "tok/s", "T=0, thinking off" + rel(code)))
    tiles += [("Chat, one at a time", f"{one['value']:.0f}", "tok/s", "default settings, thinking on" + rel(one)),
              (f"{many['conc']} chats at once", f"{many['value']:.0f}", "tok/s", "combined, default settings" + rel(many))]
    if tc:
        tiles.append(("Tool calls when required", f"{tc['value']:.0f}", "/100", "quality gate" + rel(tc)))
    return tiles[:4], dict(code=code, one=one, many=many, tc=tc, ctx=ctx)


def numbers(rows):
    """The headline numbers as README text (renders large on GitHub, in search snippets and on phones), with the run
    behind each one in small print. Newest shipped value per pick(); an older release is named."""
    _, d = hero_tiles(rows)
    code, one, many, tc, ctx = d["code"], d["one"], d["many"], d["tc"], d["ctx"]
    rel = lambda r: f" · release {r['release']}" if old(r) else ""  # noqa: E731
    cells = []
    if code:
        hi = next((r for r in rows if r["setup"] == REPO and r["metric"] == "coding_probe_max" and r["conc"] == 1
                   and r["sampling"] == code["sampling"] and r["release"] == code["release"]), None)
        cells.append((f"{code['value']:.0f} tok/s", "coding, one chat",
                      "median of 36 prompts, T=0, thinking off" + (f"; fastest prompt {hi['value']:.1f}" if hi else "")
                      + rel(code)))
    cells.append((f"{many['value']:.0f} tok/s", f"{many['conc']} chats at once, combined",
                  "512-token replies, default settings" + rel(many)))
    cells.append((f"{one['value']:.0f} tok/s", "one chat, default settings", "temperature 1.0, thinking on" + rel(one)))
    if tc:
        cells.append((f"{tc['value']:.0f}/100", "tool calls when required", "TC-45, 5 trials" + rel(tc)))
    td = "\n".join(f'    <td align="center"><h2>{n}</h2><b>{lab}</b><br><sub>{sub}</sub></td>' for n, lab, sub in cells)
    seqs = recipe_value("max_num_seqs")
    line = " · ".join(x for x in (f"<b>{ctx:,}-token context</b>" if ctx else "",
                                  f"<b>{seqs} requests at once</b>" if seqs else "",
                                  "<b>OpenAI-compatible API</b>", "<b>quality-gated releases</b>") if x)
    return (f'<table align="center">\n  <tr>\n{td}\n  </tr>\n</table>\n\n<p align="center">{line}</p>\n\n'
            + f"<sub>{CAPTIONS['hero'][1]}</sub>")


def hero_caption(rows):
    _, d = hero_tiles(rows)
    code, one, many, tc = d["code"], d["one"], d["many"], d["tc"]
    parts = ["tok/s = tokens per second; a token is about 3/4 of a word."]
    if code:
        dflt = coding_one(rows, REPO, "server defaults, thinking on")
        parts.append(f"Coding: median decode speed of 36 coding prompts (Python, C++, Rust, Go) sent one at a time, "
                     f"temperature 0, thinking off, up to 768 tokens out, release {code['release']}, {code['date']}"
                     + (f"; at the server defaults the same prompts give {dflt['value']:.0f} tok/s" if dflt else "")
                     + ".")
    parts.append(f"Chat: each chat sends a 2,048-token prompt and gets 512 tokens back at the server defaults "
                 f"(temperature 1.0, thinking on), release {one['release']}, {one['date']}, {one['stat']}"
                 + (f"; {many['conc']} chats: release {many['release']}, {many['date']}, {many['stat']}"
                    if (many['release'], many['date'], many['campaign']) != (one['release'], one['date'], one['campaign'])
                    else "") + ".")
    if tc:
        parts.append(f"Tool calls: TC-45, 5 trials, release {tc['release']}, {tc['date']}.")
    caption("hero", [" ".join(parts) + f" Method: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)."], "")


# ---------- charts ----------

def legend_setups(sets):
    return [(SETUPS[s]["name"], dict(color=color(s), marker=SETUPS[s]["marker"], alpha=1 if own(s) else 0.55))
            for s in sets]


def short_runs(lines):
    """'One Spark v2.1.0 · Two Sparks, TP=2 v2.0.0' from {setup: [rows]}."""
    return " · ".join(f"{SETUPS[s]['name']} {'/'.join(sorted({r['release'] for r in rs}, key=vkey, reverse=True))}"
                      for s, rs in lines.items())


def chart_conc(rows, metric, title, subtitle):
    """Decode tok/s against requests at once (c1..c16), one line per setup, values at both ends."""
    ticks = [1, 2, 4, 8, 16]
    sets = has(rows, metric)
    h = 3.9
    fig, (ax,) = figure(h, title, subtitle, legend=legend_setups(sets))
    notes, used, ymax, lab = [], {}, 0, []
    for s in sets:
        main, extra = main_line(rows, s, metric, "conc", keep=ticks, depth_tokens=None)
        if not main:
            continue
        line(ax, s, main, extra)
        k0, k1 = min(main), max(main)
        near = [r for r in rows if r["metric"] == metric and r["conc"] == k1 and r["setup"] != s and numeric(r)
                and r["depth_tokens"] is None and 0 < r["value"] - main[k1]["value"] < 0.5 * main[k1]["value"]]
        lab += [(k1, main[k1]["value"], f"{main[k1]['value']:.0f}", own(s), bool(near)),
                (k0, main[k0]["value"], f"{main[k0]['value']:.0f}", own(s))]
        lab += [(x, r["value"], f"{r['value']:.0f} ({r['release']})", False) for x, r in extra.items()]
        ymax = max([ymax] + [r["value"] + (r["sd"] or 0) for r in (main | extra).values()])
        used[s] = list((main | extra).values())
        notes.append(f"{SETUPS[s]['name']}: release {runs(main.values())}"
                     + (f"; hollow: older release {runs(extra.values())}" if extra else ""))
    ax.set_ylim(0, ymax * 1.18)
    conc_axis(ax, ticks)
    labels(ax, [it for it in lab if it[0] != 1])
    labels(ax, [it for it in lab if it[0] == 1], side="left")
    cap = recipe_value("max_num_seqs", "1x")
    footer(fig, h, "Error bars: ±1 sd." + (f" One Spark runs {cap} at once." if cap and "1x" in sets else ""))
    missing = [SETUPS[s]["name"] for s in SETUPS if s not in sets]
    lines = ["llama-benchy task mode: each chat sends a 2,048-token coding prompt and gets up to 512 tokens back, "
             "temperature 1.0, top-p 0.95, top-k 20, thinking on. All chats together = every token written per "
             "second, including time spent reading prompts; each chat = the speed one reply streams at once it has "
             "started."] + notes + ([f"{', '.join(missing)}: not measured on this test yet."] if missing else []) \
        + [f"Versions are numbered per setup. Data: {DATA_LINK}, with the source file of every point."]
    return fig, lines, short_runs(used)


def chart_throughput(rows):
    fig, lines, short = chart_conc(rows, "tg_total", "How fast with more people at once?",
                                   "Decode tok/s, all chats together, default settings")
    caption("throughput", lines, f"{short}; llama-benchy, 2,048 in, 512 out.")
    return fig


def chart_perchat(rows):
    fig, lines, short = chart_conc(rows, "tg_req", "How fast is each reply?",
                                   "Decode tok/s of one chat, default settings")
    caption("perchat", lines, f"{short}; llama-benchy, 2,048 in, 512 out.")
    return fig


def chart_prompt(rows, metrics, title, subtitle, fmt, ylabel_note):
    """One request, prompt not cached: a value per prompt length, one line per setup."""
    sets = has(rows, metrics)
    h = 3.9
    fig, (ax,) = figure(h, title, subtitle, legend=legend_setups(sets))
    notes, used, ymax, xs_all, ends, lab, far = [], {}, 0, set(), {}, [], []
    for s in sets:
        main, extra = main_line(rows, s, metrics, "prompt_tokens", conc=1)
        if not main:
            continue
        line(ax, s, main, extra)
        k1 = max(main)
        lab.append((k1, main[k1]["value"], fmt(main[k1]["value"]), own(s)))
        far += [(x, r["value"], f"{fmt(r['value'])} at {ktok(x)}", own(s)) for x, r in extra.items()]
        ymax = max([ymax] + [r["value"] + (r["sd"] or 0) for r in (main | extra).values()])
        xs_all |= set(main) | set(extra)
        ends[s] = max(set(main) | set(extra))
        used[s] = list((main | extra).values())
        notes.append(f"{SETUPS[s]['name']}: release {runs((main | extra).values(), stat=False)}")
    ticks = [2 ** i for i in range(11, 19) if min(xs_all) <= 2 ** i <= max(xs_all) * 1.1] or sorted(xs_all)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, [ktok(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(min(xs_all) / 1.3, max(xs_all) * 1.35)
    ax.set_xlabel("Prompt length, tokens")
    labels(ax, lab)
    labels(ax, far, side="left")   # older points beyond the line's end sit at the right edge: label to their left
    footer(fig, h, ylabel_note)
    xmax = max(ends.values())
    lines = ["One request with a prompt the server has not seen before. Each point is the mean of the samples of one "
             "run (1 to 4 per prompt length; the CSV lists each)."] + notes \
        + [f"{SETUPS[s]['name']}: not measured above {ktok(k)} yet." for s, k in ends.items() if k < xmax] \
        + [f"Data: {DATA_LINK}, with the source file of every point."]
    return fig, lines, short_runs(used)


def chart_latency(rows):
    fig, lines, short = chart_prompt(rows, ["ttft_s", "ttft_benchy_s"], "How long until the answer starts?",
                                     "Seconds to the first token, prompt not cached",
                                     lambda v: f"{v:.0f} s" if v >= 10 else f"{v:.1f} s",
                                     "A token is about 3/4 of a word.")
    caption("latency", lines, f"{short}; one request.")
    return fig


def chart_prefill(rows):
    fig, lines, short = chart_prompt(rows, ["pp", "pp_benchy"], "How fast does it read a long prompt?",
                                     "Prompt tokens read per second, prompt not cached", lambda v: f"{v:,.0f}",
                                     "Error bars: ±1 sd where the run has several samples.")
    caption("prefill", lines, f"{short}; one request.")
    return fig


def chart_depth(rows):
    """Does a long context slow decoding down? This repo's setup, sustained decode at 0..N context."""
    m = "decode_depth_total"
    concs = sorted({r["conc"] for r in rows if r["metric"] == m and r["setup"] == REPO})
    h = 3.9
    ramp = {c: tint(color(REPO), f) for c, f in zip(concs, np.linspace(0.45, 1, len(concs)))}
    fig, (ax,) = figure(h, "Does a long context slow it down?",
                        f"{SETUPS[REPO]['name']}: decode tok/s, all requests together",
                        legend=[(f"{c} request{'s' if c > 1 else ''}", dict(color=ramp[c], marker="o"))
                                for c in concs])
    notes, lab, ymax, xs = [], [], 0, set()
    for c in concs:
        main, extra = main_line(rows, REPO, m, "depth_tokens", conc=c)
        xs |= set(main)
        ax.plot(list(main), [r["value"] for r in main.values()], "-", color=ramp[c], lw=2.4, zorder=3)
        ax.plot(list(main), [r["value"] for r in main.values()], "o", color=ramp[c], ms=7.5, mec=T["ring"],
                mew=1.6, zorder=4)
        k = max(main)
        lab.append((k, main[k]["value"], f"{main[k]['value']:.0f}", True))
        ymax = max([ymax] + [r["value"] for r in main.values()])
        n = runs(main.values())
        if n not in notes:
            notes.append(n)
    xs = sorted(xs)
    ax.set_xticks(xs, [ktok(d) if d else "0" for d in xs])
    ax.set_xlim(-0.06 * xs[-1], xs[-1] * 1.12)
    ax.set_ylim(0, ymax * 1.2)
    ax.set_xlabel("Context already in the prompt, tokens")
    used = {REPO: [r for r in rows if r["metric"] == m and r["setup"] == REPO]}
    labels(ax, lab)
    footer(fig, h, "Each point: 30 s of decode, context cached.")
    r0 = next(r for r in rows if r["metric"] == m and r["setup"] == REPO)
    caption("depth", [f"{SETUPS[REPO]['name']}: release {'; '.join(notes)}"
                      + (f" (older release; shipped {SHIPPED[REPO]} not measured on this test yet)" if old(r0) else "")
                      + ". Server default sampling. This harness's 30 s steady-state window reads 10-30% above the "
                      "512-token runs of the concurrency chart, so compare points within this chart.",
                      f"Data: {DATA_LINK}, with the source file of every point."],
            f"{short_runs(used)}; 30 s sustained decode.")
    return fig


SETUP_ROWS = [  # (title, metric(s), filter, unit, better, decimals)
    ("Coding, one chat, T=0 (median of 36 prompts)", ["coding_probe_median"],
     dict(conc=1, sampling="T=0, thinking off"), "tok/s", "higher", 0),
    ("Chat, one at a time", ["tg_total"], dict(conc=1, depth_tokens=None), "tok/s", "higher", 0),
    ("8 chats at once, combined", ["tg_total"], dict(conc=8, depth_tokens=None), "tok/s", "higher", 0),
    ("First token on a 16K prompt", ["ttft_s", "ttft_benchy_s"], dict(conc=1, prompt_tokens=16384), "s",
     "lower", 1),
    ("262K-token chats the KV cache holds", ["kv_conc_262k"], {}, "", "higher", 1),
]


def setup_value(r):
    """Numbers like '2 × 3.79' (DP=2: two replicas) are drawn as their total and labelled as written."""
    if numeric(r):
        return r["value"], None
    m = re.match(r"(\d+)\s*×\s*([\d.]+)", r["value"])
    return (int(m[1]) * float(m[2]), r["value"]) if m else (None, r["value"])


def chart_setups(rows):
    """One Spark or two? The same numbers for each setup, side by side; a setup without a measurement is left out of
    that row and named in the caption."""
    panels = []
    for title, metrics, eq, unit, better, dec in SETUP_ROWS:
        sets = [s for s in SETUPS if any(pick(rows, s, m, **eq) for m in metrics)]
        # one harness for every setup in a row: the first metric that all measured setups have
        metric = next((m for m in metrics if all(pick(rows, s, m, **eq) for s in sets)), metrics[0])
        vals = {s: pick(rows, s, metric, **eq) or pick(rows, s, metrics, **eq) for s in sets}
        if vals:
            panels.append((title, metric, eq, unit, better, dec, vals))
    bar, gap, head = 0.27, 0.55, 0.7   # gap = space above a row's bars, its title included
    h = head + sum(gap + bar * len(p[-1]) for p in panels) + 0.55
    fig, _ = figure(h, "One Spark or two?", "Shipped releases where measured; this repo's setup in full colour",
                    panels=0)
    lw = 1.75   # inches for the setup names
    y = 1 - head / h
    notes, missing = defaultdict(list), defaultdict(list)
    for title, metric, eq, unit, better, dec, vals in panels:
        fig.text(0.2 / W, y - 0.24 / h, title + (f", {unit}" if unit else "") + ("  (lower is better)"
                 if better == "lower" else ""), fontsize=12, fontweight="semibold", color=T["ink"], va="top")
        hh = bar * len(vals)
        ax = fig.add_axes((0.2 / W + lw / W, y - (gap + hh) / h, 1 - (lw + 1.2) / W, hh / h))
        ax.axis("off")
        vmax = max((setup_value(r)[0] or 0) for r in vals.values()) or 1
        for j, (s, r) in enumerate(vals.items()):
            yy = len(vals) - 1 - j
            fig.text(0.2 / W, y - (gap + bar * j + bar / 2) / h, SETUPS[s]["name"], fontsize=11.5, va="center",
                     color=T["ink"] if own(s) else T["muted"], fontweight="semibold" if own(s) else "normal")
            v, text = setup_value(r)
            ax.barh(yy, v, 0.62, color=color(s), alpha=1 if own(s) else 0.5, lw=0)
            end = v
            if metric == "coding_probe_median":
                lo, hi = pick(rows, s, "coding_probe_min", **eq), pick(rows, s, "coding_probe_max", **eq)
                if lo and hi and lo["release"] == r["release"]:
                    ax.plot([lo["value"], hi["value"]], [yy, yy], color=T["ink"], lw=1.2, alpha=0.75, zorder=3)
                    for q in (lo["value"], hi["value"]):
                        ax.plot([q, q], [yy - 0.18, yy + 0.18], color=T["ink"], lw=1.2, alpha=0.75, zorder=3)
                    end = max(end, hi["value"])
            ax.text(end + vmax * 0.025, yy, (text or f"{v:,.{dec}f}") + (f"  ({r['release']})" if old(r) else ""),
                    va="center", fontsize=11.5, color=T["ink"] if own(s) else T["muted"],
                    fontweight="semibold" if own(s) else "normal")
            notes[title].append(f"{SETUPS[s]['short']} {r['release']}, {r['date']}, {r['harness'] or r['campaign']}")
        for s in SETUPS:
            if s not in vals:
                missing[title].append(SETUPS[s]["name"])
        ax.set_xlim(0, vmax * 1.35)
        ax.set_ylim(-0.5, len(vals) - 0.5)
        y -= (gap + hh) / h
    footer(fig, h, "Coding: whisker = slowest to fastest of the 36 prompts. In brackets: an older release than "
           "the shipped one.")
    caption("setups", [f"{t}: " + "; ".join(v) + (f"; not measured: {', '.join(missing[t])}" if missing[t] else "")
                       + "." for t, v in notes.items()]
            + ["Chat rows: llama-benchy task mode at the server defaults (2,048-token prompt, 512 out). 262K-token "
               "chats: vLLM's own count at boot (DP=2: two replicas, one pool each).",
               f"Data: {DATA_LINK}, with the source file of every point."],
            "Shipped release of each setup where measured; the bracket names an older release.")
    return fig


def chart_agents(rows):
    """Many agents at once on two Sparks: TP=2 or DP=2?"""
    wl = ["agent8", "agent16", "long-4", "long-8", "long-12", "long-16"]
    names = {"agent8": "8 sessions × 6 turns", "agent16": "16 sessions × 4 turns", "long-4": "4 sessions × 2 turns",
             "long-8": "8 sessions × 2 turns", "long-12": "12 sessions × 2 turns", "long-16": "16 sessions × 2 turns"}
    sets = has(rows, "agent_wall_s")
    h = 1.25 + 0.48 * len(wl) + 0.85
    fig, (ax,) = figure(h, "Many agents at once on two Sparks: TP=2 or DP=2?",
                        "Wall time for the whole workload, seconds, shorter is better",
                        legend=[(SETUPS[s]["name"], dict(color=color(s))) for s in sets])
    fig.subplots_adjust(left=2.25 / W)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=T["grid"], lw=1)
    ax.spines["bottom"].set_visible(False)
    hb = 0.8 / len(sets)
    notes = {}
    vmax = max(r["value"] for r in rows if r["metric"] == "agent_wall_s")
    for i, s in enumerate(sets):
        pts = {r["stat"]: r for r in rows if r["setup"] == s and r["metric"] == "agent_wall_s"}
        for j, k in enumerate(wl):
            if k not in pts:
                continue
            r = pts[k]
            y = j + (i - (len(sets) - 1) / 2) * hb
            ax.barh(y, r["value"], hb * 0.82, color=color(s), lw=0)
            ax.text(r["value"] + vmax * 0.012, y, f"{r['value']:.0f}", va="center", fontsize=11, color=T["ink"])
            notes.setdefault(s, f"{SETUPS[s]['name']}: {label(r)}, {r['date']}, {r['harness']}")
    ax.set_yticks(range(len(wl)), [f"{names[k]}\n{'~32K' if k.startswith('agent') else '~128K'}-token start"
                                   for k in wl], fontsize=11, color=T["ink"])
    ax.invert_yaxis()
    ax.set_xlim(0, vmax * 1.12)
    pt = [r["value"] for r in rows if r["metric"] == "agent_prompt_tokens"]
    ot = [r["value"] for r in rows if r["metric"] == "agent_output_tokens"]
    footer(fig, h, f"Prompt work dominates: {min(pt) / 1e6:.1f}M to {max(pt) / 1e6:.1f}M prompt tokens against "
           f"{min(ot):,.0f} to {max(ot):,.0f} output tokens per workload.")
    caption("agents", ["All sessions start together; every turn resends the conversation with tools on, temperature "
                       "0.6, thinking off."] + list(notes.values())
            + [f"Data: {DATA_LINK}, with the source file of every point."],
            "Agent replay, one boot per layout: " + "; ".join(v.split(": ", 1)[1].split(",")[0] + " ("
                                                              + SETUPS[k]["short"] + ")" for k, v in notes.items())
            + ".")
    return fig


def history_rows(rows, metric, **eq):
    """{release: row}: each release's own first run of `metric` (a later run as the control of the next release's
    A/B is skipped). Values like '89 and 91' (two boots) are kept as text and drawn as one point per number."""
    out = {}
    for r in sorted(rows, key=lambda r: r["date"], reverse=True):
        if r["setup"] == REPO and r["metric"] == metric and "control" not in r["campaign"] \
                and (numeric(r) or re.fullmatch(r"[\d. and]+", str(r["value"]))) \
                and all(r[k] == v for k, v in eq.items()):
            out[r["release"]] = r   # the earliest date wins
    return out


def nums(r):
    return [r["value"]] if numeric(r) else [float(x) for x in re.findall(r"\d+(?:\.\d+)?", r["value"])]


def chart_history(rows):
    """Has each release kept quality while getting faster? This repo's releases, oldest to newest."""
    speed = {c: history_rows(rows, "tg_total", conc=c, depth_tokens=None) for c in (1, 8)}
    hard = history_rows(rows, "gate_hardmode")
    gains = {r["release"]: r for r in rows if r["setup"] == REPO and r["metric"] == "release_gain_pct"}
    rels = sorted(set(speed[1]) | set(speed[8]) | set(hard) | set(gains), key=vkey)
    h = 5.0
    shade = {1: 0.5, 8: 1.0}
    mix = lambda f: tint(color(REPO), f)  # noqa: E731
    fig, (top, bot) = figure(h, "What did each release change?",
                             f"{SETUPS[REPO]['name']}, every shipped release, oldest to newest",
                             legend=[("8 chats, combined", dict(color=mix(1.0), marker="o")),
                                     ("one chat", dict(color=mix(0.5), marker="o"))], panels=2, hspace=0.45)
    x = {v: i for i, v in enumerate(rels)}
    lab, ymax = [], 0
    for c in (8, 1):
        pts = speed[c]
        xs = [x[v] for v in rels if v in pts]
        ys = [pts[v]["value"] for v in rels if v in pts]
        top.plot(xs, ys, "-", color=mix(shade[c]), lw=2.4, zorder=3)
        top.plot(xs, ys, "o", color=mix(shade[c]), ms=7.5, mec=T["ring"], mew=1.6, zorder=4)
        for v in pts:
            if pts[v]["sd"]:
                top.errorbar([x[v]], [pts[v]["value"]], yerr=[pts[v]["sd"]], fmt="none", ecolor=mix(shade[c]),
                             elinewidth=1.2, capsize=3, zorder=2)
        if xs:
            lab.append((xs[-1], ys[-1], f"{ys[-1]:.0f}", True))
            ymax = max([ymax] + [p["value"] + (p["sd"] or 0) for p in pts.values()])
    top.set_ylim(0, ymax * 1.2)
    xh = [v for v in rels if any(v in speed[c] and "xhigh" in speed[c][v]["sampling"] for c in speed)]
    if xh:   # measured at a different thinking effort: shade those releases and say so
        top.axvspan(x[xh[0]] - 0.4, x[xh[-1]] + 0.4, color=T["grid"], alpha=0.6, lw=0, zorder=0)
        top.text(x[xh[0]] - 0.3, ymax * 1.12, "measured at xhigh thinking effort", fontsize=10.5, color=T["muted"],
                 va="center")
    panel_title(top, "Decode tokens per second, default settings")
    labels(top, lab)
    bot.set_ylim(80, 101)
    bot.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    bot.axhline(88, color=T["muted"], lw=1.2, ls=(0, (4, 3)), zorder=1)
    bot.text(len(rels) - 0.45, 87.6, "pass mark 88", fontsize=10.5, color=T["muted"], va="top", ha="right")
    for v, r in hard.items():
        ns = nums(r)
        bot.plot([x[v]] * len(ns), ns, "D", color=color(REPO), ms=8, mec=T["ring"], mew=1.6, zorder=4)
        bot.annotate(" / ".join(f"{n:.0f}" for n in ns), (x[v], max(ns)), xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=11, color=T["ink"])
    panel_title(bot, "Hard multi-step tool use, score out of 100")
    bot.set_xticks(range(len(rels)), [v if v.startswith("v") else f"{v}\npre-1.0" for v in rels])
    bot.set_xlim(-0.5, len(rels) - 0.5)
    footer(fig, h, "Releases from v1.0.0 on passed the full quality gate."
           + (" Two marks: two gate boots." if any(len(nums(r)) > 1 for r in hard.values()) else "")
           + " Error bars: ±1 sd.")
    groups = []   # consecutive releases measured the same way share one line
    for v in rels:
        r = speed[1].get(v) or speed[8].get(v)
        if not r:
            continue
        how = f"{r['harness']}, " + re.sub(r"\s*\([^)]*\)", "", r["stat"])
        if groups and groups[-1][1] == how:
            groups[-1][0].append(f"{v} {r['date']}")
        else:
            groups.append(([f"{v} {r['date']}"], how))
    notes = ["Speed runs: " + "; ".join(", ".join(g) + f" ({how})" for g, how in groups) + ".",
             "Hard tool use: the promotion gate of each release (b0: the gate on the pinned checkpoint), "
             + ", ".join(f"{v} {r['date']}" for v, r in sorted(hard.items(), key=lambda i: vkey(i[0])))
             + "".join(f"; {v}: score not in the data" for v in rels if v not in hard) + "."]
    caption("history", ["Speed: llama-benchy task mode, 2,048-token prompt, 512 out, temperature 1.0, thinking on, "
                        "from each release's own promotion run, so day-to-day drift of the Sparks is in these numbers; "
                        "the paired A/B of every release is in [VERSIONS.md](VERSIONS.md)."]
            + ([f"Shaded ({xh[0]} to {xh[-1]}): measured at the chat template's xhigh thinking effort; later releases at "
                "the recipe default, medium. Longer thinking changes the replies, so speeds on the two sides of the "
                "shade are not a like-for-like comparison."] if xh else [])
            + notes + [f"Data: {DATA_LINK}, with the source file of every point."],
            "Each release's own promotion run and gate" + (f"; shaded releases ran at xhigh thinking effort"
                                                          if xh else "") + ".")
    return fig


# ---------- README tables and blocks ----------

def src_link(r):
    repo, path = r["source"].split(":", 1)
    d = path if path.endswith(".md") else str(Path(path).parent) + "/"
    if repo == REPO:
        return d
    return (GH[repo].replace("/tree/", "/blob/") if d.endswith(".md") else GH[repo]) + d


def fmt_cell(r, dec=1):
    if not r:
        return "–"
    v = f"{r['value']:,.{dec}f}" if numeric(r) else r["value"]
    return v + (f" ± {r['sd']:,.{dec}f}" if r["sd"] else "")


def matrix(rows):
    """This repo's setup: decode at each concurrency, and prefill and first token at each prompt length."""
    s = REPO

    def best(metrics, **eq):
        return pick(rows, s, metrics, **eq)

    out = []
    concs = sorted({r["conc"] for r in rows if r["setup"] == s and r["metric"] == "tg_total"
                    and r["depth_tokens"] is None and r["conc"]})
    if concs:
        out += ["| Requests at once | All together, tok/s | Each, tok/s | Release, run |", "|--:|--:|--:|---|"]
        for c in concs:
            t, e = best("tg_total", conc=c, depth_tokens=None), best("tg_req", conc=c, depth_tokens=None)
            out.append(f"| {c} | {fmt_cell(t)} | {fmt_cell(e)} | [{t['release']}, {t['date']}]({src_link(t)}) |")
        out += ["", "llama-benchy task mode: 2,048-token prompt, 512 tokens out, temperature 1.0, thinking on; "
                "mean ± sd as given per run in the CSV.", ""]
    sizes = sorted({r["prompt_tokens"] for r in rows if r["setup"] == s and r["conc"] == 1
                    and r["metric"] in ("ttft_s", "ttft_benchy_s", "pp", "pp_benchy") and r["prompt_tokens"]})
    if sizes:
        out += ["| Prompt, tokens | Prompt reading, tok/s | First token, s | Release, run |", "|--:|--:|--:|---|"]
        for p in sizes:
            pp = best(["pp", "pp_benchy"], conc=1, prompt_tokens=p)
            tt = best(["ttft_s", "ttft_benchy_s"], conc=1, prompt_tokens=p)
            ref = tt or pp
            out.append(f"| {ktok(p)} | {fmt_cell(pp, 0)} | {fmt_cell(tt, 1)} | "
                       f"[{ref['release']}, {ref['date']}, {ref['harness']}]({src_link(ref)}) |")
        out += ["", "One request, prompt not cached."]
    return "\n".join(out)


def table(rows):
    """Markdown capability table; every number has a letter for its release, date and run."""
    keys = []

    def cell(setup, metrics, fmt="{:,.1f}", **eq):
        pts = [r for r in rows if r["setup"] == setup and r["metric"] in metrics
               and all(r[k] == v for k, v in eq.items())]
        if not pts:
            return None
        r = max(pts, key=lambda r: (not old(r), vkey(r["release"]), r["date"]))
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

    cap1 = recipe_value("max_num_seqs", "1x")
    row("Decode tok/s, 1 request", lambda s: join(cell(s, ["tg_total"], conc=1, depth_tokens=None)))
    for c in (4, 8, 16):
        def f(s, c=c):
            if s == "1x" and cap1 and c > cap1:
                return f"over the cap (max_num_seqs {cap1})"
            return join(cell(s, ["tg_req"], conc=c, depth_tokens=None), cell(s, ["tg_total"], conc=c, depth_tokens=None))
        row(f"Decode tok/s, {c} requests: each / total", f)
    row("Coding, 36 prompts one at a time: median (max) decode tok/s, T=0 / server defaults",
        lambda s: join(*((lambda a, b: a and f"{a} {b or ''}".strip())(
            cell(s, ["coding_probe_median"], "{:.0f}", conc=1, sampling=m),
            cell(s, ["coding_probe_max"], "({:.0f})", conc=1, sampling=m))
            for m in ("T=0, thinking off", "server defaults, thinking on"))))
    row("Copy-heavy (MTP accepts nearly every draft) total tok/s at 1 / 4 / 8 requests, max of 3 rounds",
        lambda s: join(*(cell(s, ["copy_total_max"], "{:.0f}", conc=c) for c in (1, 4, 8))))
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
    row("Max context per request", lambda s: f"{recipe_value('max_model_len', '1x') or 262144:,} (recipe)")
    row("KV pool, tokens", lambda s: join(cell(s, ["kv_tokens"], "{:,.0f}")))
    row("Requests of 262,144 tokens the pool holds (vLLM's count)", lambda s: join(cell(s, ["kv_conc_262k"], "{:.2f}")))
    row("Requests that fit the KV pool at 16K / 64K / 128K",
        lambda s: join(*(cell(s, ["sessions_fit"], "{:.1f}", prompt_tokens=p) for p in (16384, 65536, 131072))))
    row("Quality gate: hardmode / TC-45 / retrieval to ~245K / stragglers",
        lambda s: join(cell(s, ["gate_hardmode"], "{:.0f}"), cell(s, ["gate_tc45"], "{:.0f}"),
                       cell(s, ["gate_retrieval"], "{}"), cell(s, ["gate_stragglers"], "{}")))
    out += ["", "Releases and runs behind the numbers:", ""]
    for i, (rel, date, camp, harness, src) in enumerate(keys):
        link = src_link(dict(source=src))
        out.append(f"- <sup>{chr(97 + i)}</sup> {rel}, {date}, {camp}" + (f", {harness}" if harness else "")
                   + f" ([files]({link}))")
    return "\n".join(out)


def quality(rows):
    """The gate of this repo's shipped release (or the newest gated one) as a table."""
    g = {m: pick(rows, REPO, "gate_" + m) for m in ("tc45", "hardmode", "retrieval", "stragglers")}
    if not all(g.values()):
        return "Quality gate: not measured yet."
    v = lambda m: f"{g[m]['value']:.0f}" if numeric(g[m]) else g[m]["value"]  # noqa: E731
    seqs = recipe_value("max_num_seqs")
    hi = re.search(r"c(\d+)-c(\d+)", v("stragglers"))
    retr = v("retrieval")
    out = ["| Check | Result | What it checks |", "|---|---|---|",
           f"| Tool calls (TC-45) | **{v('tc45')}/100** | A request that requires a tool call gets one; 5 trials |",
           f"| Hard tool use | **{v('hardmode')}/100** | 88 multi-step tool-use scenarios; pass mark 88 |",
           f"| Long-context retrieval | **{retr.split(' (')[0]}**"
           + (f" ({retr.split(' (', 1)[1]}" if " (" in retr else "")
           + " | 20 facts hidden in prompts of 8K to ~245K tokens, each returned through a tool call |",
           f"| Stalled requests | **{v('stragglers').split(',')[0]}** | No request falls behind the others when "
           + (f"{hi[1]} to {hi[2]} are sent at once" if hi else "many are sent at once")
           + (f" (it runs {seqs} at a time and queues the rest)" if hi and seqs and seqs < int(hi[2]) else "")
           + " |"]
    rel = sorted({(g[m]["release"], g[m]["date"]) for m in g})
    return "\n".join(out) + "\n\nGate run: " + "; ".join(f"release {r}, {d}" for r, d in rel) \
        + ". Every release passes this gate before it ships."


def badges(rows):
    def b(label, value, colour):
        enc = lambda x: quote(x.replace("-", "--").replace("_", "__"), safe="")  # noqa: E731
        return (f'  <img alt="{label}: {value}" '
                f'src="https://img.shields.io/badge/{enc(label)}-{enc(value)}-{colour}?style=flat-square">')
    gate = all(pick(rows, REPO, "gate_" + m) for m in ("tc45", "hardmode", "retrieval", "stragglers"))
    hw = {"1x": "1× DGX Spark", "2x": "2× DGX Spark"}[REPO]
    return '<p align="center">\n' + "\n".join([
        b("release", SHIPPED[REPO], "0969da"), b("hardware", hw, "555555"),
        b("quality gate", "passed" if gate else "not run", "2ea44f" if gate else "d97706"),
        b("license", "Apache-2.0", "555555")]) + "\n</p>"


# README block -> the images in it; a pair is shown side by side at 420 px each
FIGURES = {"speed": ("throughput", "latency"), "context": ("prefill", "depth"), "perchat": ("perchat",),
           "setups": ("setups",), "agents": ("agents",), "history": ("history",)}


def write_blocks(rows):
    readme = ROOT / "README.md"
    text = new = readme.read_text()
    blocks = {"numbers": lambda: numbers(rows), "badges": lambda: badges(rows), "matrix": lambda: matrix(rows),
              "capability-table": lambda: table(rows), "quality": lambda: quality(rows)}
    blocks["hero"] = lambda: '<h1 align="center">' + picture("hero") + "</h1>"
    for name, imgs in FIGURES.items():
        if all(n in IMAGES for n in imgs):
            blocks[name] = (lambda imgs=imgs: figure_block(*imgs))
    for name, body in blocks.items():
        a, b = f"<!-- {name}:start (scripts/make_charts.py writes this block) -->", f"<!-- {name}:end -->"
        if a not in text:
            print(f"README.md has no {name} markers; block not written")
            continue
        new = re.sub(re.escape(a) + r".*?" + re.escape(b), lambda m: f"{a}\n{body()}\n{b}", new, flags=re.S)
        print(f"wrote README.md {name} block")
    if new != text:
        readme.write_text(new)


def selftest():
    """The newest release with 3+ points draws the line; older points only appear beyond its end."""
    global SHIPPED
    SHIPPED = {"2x": "v3.1.0", "1x": "", "dp2": ""}
    mk = lambda rel, x, d: dict(setup="2x", metric="m", release=rel, alias="", campaign="c", conc=x, date=d,  # noqa: E731
                                value=1.0, sd=None, harness="h", stat="")
    rows = [mk("v3.1.0", 1, "2026-10-08"), mk("v3.1.0", 4, "2026-10-08"), mk("v3.1.0", 8, "2026-10-08"),
            mk("v3.0.0", 1, "2026-10-01"), mk("v3.0.0", 16, "2026-10-01")]
    main, extra = main_line(rows, "2x", "m", "conc")
    assert sorted(main) == [1, 4, 8] and sorted(extra) == [16] and old(extra[16]) and not old(main[1])
    main, extra = main_line(rows[:2] + rows[3:], "2x", "m", "conc")   # newest has only 2 points: the larger wins
    assert sorted(main) == [1, 4] and sorted(extra) == [16]
    assert setup_value(dict(value="2 × 3.79"))[0] == 7.58 and ktok(131072) == "128K" and ktok(245267) == "245K"
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        raise SystemExit
    rows = load()
    who = {"1x": "one DGX Spark", "2x": "two DGX Sparks"}[REPO]
    IMAGES["hero"] = f"Qwen3.8 Flash Next on {who}: Qwen emblem, gold NVIDIA hardware and violet token trails."
    DISPLAY["hero"] = 840
    hero_caption(rows)
    half = dict(w=5.0, px=420)
    render("throughput", chart_throughput, rows, "Line chart of decode tokens per second, all chats together, against 1 "
           "to 16 requests at the same time, for each setup. Values are labelled at the line ends.", **half)
    render("perchat", chart_perchat, rows, "Line chart of decode tokens per second of each chat against 1 to 16 "
           "requests at the same time, for each setup. Values are labelled at the line ends.", **half)
    render("latency", chart_latency, rows, "Line chart of seconds until the first token against prompt length, "
           "prompt not cached, for each setup. Values are labelled at the line ends.", **half)
    render("prefill", chart_prefill, rows, "Line chart of prompt tokens read per second against prompt length, prompt "
           "not cached, for each setup. Values are labelled at the line ends.", **half)
    render("depth", chart_depth, rows, "Line chart of decode tokens per second with 0 to 64K tokens of context "
           "already in the prompt, at 1, 4 and 8 requests. Values are labelled at the line ends.", **half)
    render("setups", chart_setups, rows, "Bar charts comparing one Spark, two Sparks at TP=2 and two Sparks at DP=2 on "
           "coding speed, chat speed, 8 chats combined, first token on a 16K prompt and long chats that fit the KV "
           "cache. Values are labelled on the bars.")
    if has(rows, "agent_wall_s"):
        render("agents", chart_agents, rows, "Bar chart of wall time for six agent workloads on two Sparks at TP=2 "
               "and at DP=2. Values are labelled on the bars.")
    render("history", chart_history, rows, "Decode speed at one chat and at 8 chats, and the hard tool-use score, for "
           "every release of this setup. Values are labelled on the chart.")
    write_blocks(rows)
