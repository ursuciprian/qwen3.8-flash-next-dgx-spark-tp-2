#!/usr/bin/env python3
"""benchy_plot.py — visualize llama-benchy --save-result CSVs (markdown-pipe tables).

Usage:
    python3 scripts/benchy_plot.py <dir-or-files...> --out benchy.html [--arms la,la-pmu16,...]
    python3 scripts/benchy_plot.py results/benchy --out results/benchy/index.html --md
    python3 scripts/benchy_plot.py --selftest

Input format (one file per "arm", arm name = filename stem):
    | model | test | t/s (total) | t/s (req) | peak t/s | peak t/s (req) | ttfr (ms) | est_ppt (ms) | e2e_ttft (ms) |
with optional extra columns whose header contains "accept" or "prefix" (added by
our fork; not present in the sample files this was written against — parsed
opportunistically, absent when the column is absent).

`test` cell formats seen in the wild:
    pp2048 (c1)                 -- prompt-processing, depth 0, concurrency 1
    tg128 (c1)                  -- token-generation,  depth 0, concurrency 1
    ctx_pp @ d16384 (c4)        -- prompt-processing at context depth 16384, c4
    ctx_tg @ d16384 (c4)        -- token-generation  at context depth 16384, c4
    pp2048 @ d65536 (c10)       -- same idea, "pp"/"tg" kind with explicit size
Values are "mean ± std"; blank cells (peak t/s, ttfr, ... on tg-less/pp-less
rows) parse to (None, None).

stdlib only. The emitted HTML pulls plotly.js from a CDN — no python plotly.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

TEST_RE = re.compile(
    r"^(?P<kind>[A-Za-z_]+?)(?P<size>\d+)?\s*(?:@\s*d(?P<depth>\d+))?\s*\(c(?P<conc>\d+)\)$"
)


def parse_mean_std(cell):
    """'2906.78 ± 16.40' -> (2906.78, 16.4); '' -> (None, None)."""
    cell = cell.strip()
    if not cell:
        return None, None
    parts = cell.split("\xb1") if "\xb1" in cell else cell.split("+/-")
    try:
        if len(parts) == 2:
            return float(parts[0].strip()), float(parts[1].strip())
        return float(parts[0].strip()), None
    except ValueError:
        return None, None


def split_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def is_separator_row(cells):
    return all(re.fullmatch(r":?-+:?", c) for c in cells if c)


def parse_csv_text(text, arm):
    """Yield (row_dict, raw_test_str_or_None_on_unparseable) for each data row."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return
    header = split_row(lines[0])
    data_lines = lines[1:]
    if data_lines and is_separator_row(split_row(data_lines[0])):
        data_lines = data_lines[1:]

    idx = {name: i for i, name in enumerate(header)}
    accept_col = next((h for h in header if "accept" in h.lower()), None)
    prefix_col = next((h for h in header if "prefix" in h.lower()), None)

    for line in data_lines:
        cells = split_row(line)
        if len(cells) != len(header):
            yield None, line  # unparseable: column count mismatch
            continue
        test = cells[idx["test"]].strip() if "test" in idx else ""
        m = TEST_RE.match(test)
        if not m:
            yield None, line
            continue
        kind = m.group("kind")
        depth = int(m.group("depth")) if m.group("depth") else 0
        conc = int(m.group("conc"))
        is_pp = kind in ("pp", "ctx_pp")
        is_tg = kind in ("tg", "ctx_tg")

        total_mean, _ = parse_mean_std(cells[idx["t/s (total)"]]) if "t/s (total)" in idx else (None, None)
        req_mean, _ = parse_mean_std(cells[idx["t/s (req)"]]) if "t/s (req)" in idx else (None, None)
        peak_mean, _ = parse_mean_std(cells[idx["peak t/s"]]) if "peak t/s" in idx else (None, None)
        peak_req_mean, _ = parse_mean_std(cells[idx["peak t/s (req)"]]) if "peak t/s (req)" in idx else (None, None)
        ttfr_mean, _ = parse_mean_std(cells[idx["ttfr (ms)"]]) if "ttfr (ms)" in idx else (None, None)
        e2e_mean, _ = parse_mean_std(cells[idx["e2e_ttft (ms)"]]) if "e2e_ttft (ms)" in idx else (None, None)

        accept_draft = cells[idx[accept_col]].strip() or None if accept_col else None
        prefix_hit = cells[idx[prefix_col]].strip() or None if prefix_col else None

        row = {
            "arm": arm,
            "depth": depth,
            "concurrency": conc,
            "pp": is_pp,
            "tg": is_tg,
            "prompt_tps": total_mean if is_pp else None,
            "gen_tps_total": total_mean if is_tg else None,
            "gen_tps_req": req_mean if is_tg else None,
            "gen_peak_tps": peak_mean if is_tg else None,
            "gen_peak_tps_req": peak_req_mean if is_tg else None,
            "ttft_ms": (ttfr_mean if ttfr_mean is not None else e2e_mean) if is_pp else None,
            "accept_draft": accept_draft,
            "prefix_hit": prefix_hit,
        }
        yield row, None


def gather_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.csv"))))
        else:
            files.append(p)
    return files


def load(paths, arms_filter=None):
    rows = []
    unparsed = []  # (file, line)
    for f in gather_files(paths):
        arm = os.path.splitext(os.path.basename(f))[0]
        if arms_filter and arm not in arms_filter:
            continue
        try:
            with open(f, "r") as fh:
                text = fh.read()
        except OSError as e:
            unparsed.append((f, f"<could not read: {e}>"))
            continue
        for row, bad_line in parse_csv_text(text, arm):
            if row is None:
                unparsed.append((f, bad_line))
            else:
                rows.append(row)
    return rows, unparsed


def build_summary_table(rows):
    """Merge pp+tg rows keyed by (arm, depth, concurrency) -> dict of metrics."""
    keyed = {}
    for r in rows:
        k = (r["arm"], r["depth"], r["concurrency"])
        d = keyed.setdefault(k, {"arm": r["arm"], "depth": r["depth"], "concurrency": r["concurrency"],
                                  "ttft_ms": None, "gen_tps_total": None, "gen_tps_req": None,
                                  "accept_draft": None, "prefix_hit": None})
        if r["ttft_ms"] is not None:
            d["ttft_ms"] = r["ttft_ms"]
        if r["gen_tps_total"] is not None:
            d["gen_tps_total"] = r["gen_tps_total"]
        if r["gen_tps_req"] is not None:
            d["gen_tps_req"] = r["gen_tps_req"]
        if r["accept_draft"]:
            d["accept_draft"] = r["accept_draft"]
        if r["prefix_hit"]:
            d["prefix_hit"] = r["prefix_hit"]
    return sorted(keyed.values(), key=lambda d: (d["arm"], d["depth"], d["concurrency"]))


def write_markdown(summary, out_path):
    arms = sorted({s["arm"] for s in summary})
    keys = sorted({(s["depth"], s["concurrency"]) for s in summary})
    by_arm = {}
    for s in summary:
        by_arm.setdefault(s["arm"], {})[(s["depth"], s["concurrency"])] = s

    lines = ["| depth | c | " + " | ".join(f"{a} ttft/gen" for a in arms) + " |",
             "|---|---|" + "---|" * len(arms)]
    for depth, c in keys:
        cells = []
        for a in arms:
            s = by_arm.get(a, {}).get((depth, c))
            if not s:
                cells.append("-")
                continue
            ttft = f"{s['ttft_ms']:.0f}ms" if s["ttft_ms"] is not None else "-"
            gen = f"{s['gen_tps_total']:.1f}t/s" if s["gen_tps_total"] is not None else "-"
            cells.append(f"{ttft} / {gen}")
        lines.append(f"| {depth} | c{c} | " + " | ".join(cells) + " |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>benchy results</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { --bg:#ffffff; --fg:#111111; --muted:#666666; --border:#dddddd; --accent:#2563eb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#111318; --fg:#e8e8e8; --muted:#9aa0a6; --border:#333844; --accent:#60a5fa; }
  }
  body { background:var(--bg); color:var(--fg); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         margin:0; padding:16px; }
  h1 { font-size:1.4rem; margin:0 0 4px; }
  h2 { font-size:1.05rem; margin:28px 0 8px; color:var(--muted); }
  .note { color:var(--muted); font-size:0.85rem; }
  .chart { width:100%; max-width:1100px; }
  table { border-collapse:collapse; width:100%; max-width:1100px; font-size:0.85rem; }
  th, td { border:1px solid var(--border); padding:4px 8px; text-align:right; }
  th:first-child, td:first-child { text-align:left; }
  th { cursor:pointer; user-select:none; color:var(--muted); }
  th:hover { color:var(--fg); }
  select { background:var(--bg); color:var(--fg); border:1px solid var(--border); padding:4px; margin-bottom:8px; }
</style>
</head>
<body>
<h1>benchy results</h1>
<p class="note" id="meta"></p>

<h2>1. TTFT vs depth (per arm)</h2>
<label class="note">concurrency: <select id="ttft-conc"></select></label>
<div id="ttft-chart" class="chart"></div>

<h2>2. Aggregate gen tok/s vs concurrency (per arm)</h2>
<label class="note">depth: <select id="gen-depth"></select></label>
<div id="gen-chart" class="chart"></div>

<h2>3. Per-stream gen tok/s vs depth at c1 (per arm)</h2>
<div id="stream-chart" class="chart"></div>

<h2>4. Accept/draft, prefix-hit</h2>
<div id="accept-note" class="note"></div>
<div id="accept-chart" class="chart"></div>

<h2>5. Summary table</h2>
<table id="summary">
  <thead><tr>
    <th data-k="arm">arm</th><th data-k="depth">depth</th><th data-k="concurrency">c</th>
    <th data-k="ttft_ms">TTFT (ms)</th><th data-k="gen_tps_total">gen agg (t/s)</th>
    <th data-k="gen_tps_req">gen/stream (t/s)</th><th data-k="accept_draft">accept/draft</th>
  </tr></thead>
  <tbody></tbody>
</table>

<script>
const ROWS = __ROWS_JSON__;
const SUMMARY = __SUMMARY_JSON__;

document.getElementById('meta').textContent =
  ROWS.length + ' rows, arms: ' + [...new Set(ROWS.map(r => r.arm))].join(', ');

const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const paper = isDark ? '#111318' : '#ffffff';
const font = isDark ? '#e8e8e8' : '#111111';
const grid = isDark ? '#333844' : '#dddddd';
function baseLayout(title) {
  return {
    title, paper_bgcolor: paper, plot_bgcolor: paper,
    font: { color: font }, xaxis: { gridcolor: grid }, yaxis: { gridcolor: grid },
    margin: { t: 40 },
  };
}
const arms = [...new Set(ROWS.map(r => r.arm))].sort();
const colors = ['#2563eb','#dc2626','#16a34a','#d97706','#7c3aed','#0891b2','#db2777','#65a30d'];
const colorOf = a => colors[arms.indexOf(a) % colors.length];

// --- 1. TTFT vs depth, toggled by concurrency ---
const concLevels = [...new Set(ROWS.filter(r => r.pp).map(r => r.concurrency))].sort((a,b)=>a-b);
const ttftSel = document.getElementById('ttft-conc');
concLevels.forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = 'c' + c; ttftSel.appendChild(o); });
function drawTTFT() {
  const c = Number(ttftSel.value);
  const traces = arms.map(a => {
    const pts = ROWS.filter(r => r.arm === a && r.pp && r.concurrency === c && r.ttft_ms != null)
      .sort((x,y) => x.depth - y.depth);
    return { x: pts.map(p=>p.depth), y: pts.map(p=>p.ttft_ms), name: a, mode: 'lines+markers', line: { color: colorOf(a) } };
  }).filter(t => t.x.length);
  Plotly.newPlot('ttft-chart', traces, { ...baseLayout(''), xaxis: { title: 'depth', gridcolor: grid }, yaxis: { title: 'TTFT (ms)', gridcolor: grid } }, {responsive:true});
}
ttftSel.addEventListener('change', drawTTFT);
if (concLevels.length) { ttftSel.value = concLevels[0]; drawTTFT(); }

// --- 2. gen tok/s aggregate vs concurrency, toggled by depth ---
const depthLevels = [...new Set(ROWS.filter(r => r.tg).map(r => r.depth))].sort((a,b)=>a-b);
const genSel = document.getElementById('gen-depth');
depthLevels.forEach(d => { const o = document.createElement('option'); o.value = d; o.textContent = 'd' + d; genSel.appendChild(o); });
function drawGen() {
  const d = Number(genSel.value);
  const traces = arms.map(a => {
    const pts = ROWS.filter(r => r.arm === a && r.tg && r.depth === d && r.gen_tps_total != null)
      .sort((x,y) => x.concurrency - y.concurrency);
    return { x: pts.map(p=>p.concurrency), y: pts.map(p=>p.gen_tps_total), name: a, mode: 'lines+markers', line: { color: colorOf(a) } };
  }).filter(t => t.x.length);
  Plotly.newPlot('gen-chart', traces, { ...baseLayout(''), xaxis: { title: 'concurrency', gridcolor: grid }, yaxis: { title: 'gen t/s (total)', gridcolor: grid } }, {responsive:true});
}
genSel.addEventListener('change', drawGen);
if (depthLevels.length) { genSel.value = depthLevels[0]; drawGen(); }

// --- 3. per-stream gen tok/s vs depth at c1 ---
{
  const traces = arms.map(a => {
    const pts = ROWS.filter(r => r.arm === a && r.tg && r.concurrency === 1 && r.gen_tps_req != null)
      .sort((x,y) => x.depth - y.depth);
    return { x: pts.map(p=>p.depth), y: pts.map(p=>p.gen_tps_req), name: a, mode: 'lines+markers', line: { color: colorOf(a) } };
  }).filter(t => t.x.length);
  Plotly.newPlot('stream-chart', traces, { ...baseLayout(''), xaxis: { title: 'depth', gridcolor: grid }, yaxis: { title: 'gen t/s (per stream)', gridcolor: grid } }, {responsive:true});
}

// --- 4. accept/draft ---
{
  const withAccept = ROWS.filter(r => r.accept_draft != null);
  if (!withAccept.length) {
    document.getElementById('accept-note').textContent = 'no accept/draft or prefix-hit columns in the loaded CSVs.';
  } else {
    const cells = withAccept.map(r => `${r.arm} d${r.depth} c${r.concurrency}`);
    const nums = withAccept.map(r => parseFloat(r.accept_draft));
    Plotly.newPlot('accept-chart', [{ x: cells, y: nums, type: 'bar', marker: { color: '#2563eb' } }],
      { ...baseLayout(''), yaxis: { title: 'accept/draft', gridcolor: grid } }, {responsive:true});
  }
}

// --- 5. summary table ---
const tbody = document.querySelector('#summary tbody');
let sortKey = 'arm', sortAsc = true;
function fmt(v) { return v == null ? '-' : (typeof v === 'number' ? v.toFixed(1) : v); }
function renderTable() {
  const rows = [...SUMMARY].sort((a,b) => {
    const av = a[sortKey], bv = b[sortKey];
    if (av == null) return 1;
    if (bv == null) return -1;
    const cmp = av < bv ? -1 : av > bv ? 1 : 0;
    return sortAsc ? cmp : -cmp;
  });
  tbody.innerHTML = rows.map(r => `<tr>
    <td>${r.arm}</td><td>${r.depth}</td><td>${r.concurrency}</td>
    <td>${fmt(r.ttft_ms)}</td><td>${fmt(r.gen_tps_total)}</td>
    <td>${fmt(r.gen_tps_req)}</td><td>${fmt(r.accept_draft)}</td>
  </tr>`).join('');
}
document.querySelectorAll('#summary th').forEach(th => {
  th.addEventListener('click', () => {
    const k = th.dataset.k;
    sortAsc = (sortKey === k) ? !sortAsc : true;
    sortKey = k;
    renderTable();
  });
});
renderTable();
</script>
</body>
</html>
"""


def build_html(rows, summary):
    html = HTML_TEMPLATE
    html = html.replace("__ROWS_JSON__", json.dumps(rows))
    html = html.replace("__SUMMARY_JSON__", json.dumps(summary))
    return html


SELFTEST_CSV = """| model              |                  test |     t/s (total) |        t/s (req) |      peak t/s |   peak t/s (req) |            ttfr (ms) |         est_ppt (ms) |        e2e_ttft (ms) |
|:-------------------|----------------------:|-----------------:|-----------------:|--------------:|-----------------:|---------------------:|---------------------:|---------------------:|
| qwen3.8-flash-next |           pp2048 (c1) | 2906.78 ± 16.40 |  2906.78 ± 16.40 |               |                  |        706.81 ± 3.98 |        704.93 ± 3.98 |        706.81 ± 3.98 |
| qwen3.8-flash-next |            tg128 (c1) |    45.97 ± 2.53 |     45.97 ± 2.53 |  46.50 ± 2.50 |     46.50 ± 2.50 |                      |                      |                      |
| qwen3.8-flash-next |  ctx_pp @ d16384 (c4) |  3026.06 ± 38.80 |  934.98 ± 149.69 |               |                  |   17982.95 ± 2860.80 |   17981.07 ± 2860.80 |   17982.95 ± 2860.80 |
| qwen3.8-flash-next |  ctx_tg @ d16384 (c4) |    42.39 ± 0.03 |    17.88 ± 6.64 |  73.50 ± 1.50 |     18.38 ± 6.65 |                      |                      |                      |
"""


def selftest():
    rows = list(r for r, bad in parse_csv_text(SELFTEST_CSV, "selftest") if r is not None)
    assert len(rows) == 4, rows
    pp1 = rows[0]
    assert pp1["pp"] and pp1["depth"] == 0 and pp1["concurrency"] == 1
    assert abs(pp1["prompt_tps"] - 2906.78) < 1e-6
    assert abs(pp1["ttft_ms"] - 706.81) < 1e-6
    tg1 = rows[1]
    assert tg1["tg"] and tg1["gen_tps_total"] is not None and abs(tg1["gen_tps_total"] - 45.97) < 1e-6
    assert tg1["ttft_ms"] is None
    ctxpp = rows[2]
    assert ctxpp["pp"] and ctxpp["depth"] == 16384 and ctxpp["concurrency"] == 4
    assert abs(ctxpp["prompt_tps"] - 3026.06) < 1e-6
    ctxtg = rows[3]
    assert ctxtg["tg"] and ctxtg["depth"] == 16384 and abs(ctxtg["gen_tps_req"] - 17.88) < 1e-6
    print("selftest OK:", len(rows), "rows parsed")


def main():
    ap = argparse.ArgumentParser(description="Visualize llama-benchy --save-result CSVs.")
    ap.add_argument("paths", nargs="*", help="CSV files or a directory of CSVs")
    ap.add_argument("--out", default="benchy.html", help="output HTML path")
    ap.add_argument("--arms", help="comma-separated arm (filename stem) allowlist")
    ap.add_argument("--md", nargs="?", const="", help="also write a markdown comparison table (default: <out>.md)")
    ap.add_argument("--selftest", action="store_true", help="run embedded parser self-check and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.paths:
        ap.error("need at least one CSV file or directory (or --selftest)")

    arms_filter = set(a.strip() for a in args.arms.split(",")) if args.arms else None
    rows, unparsed = load(args.paths, arms_filter)
    summary = build_summary_table(rows)

    with open(args.out, "w") as f:
        f.write(build_html(rows, summary))
    print(f"wrote {args.out} ({len(rows)} rows, {len(unparsed)} unparsed lines)")

    if args.md is not None:
        md_path = args.md or (os.path.splitext(args.out)[0] + ".md")
        write_markdown(summary, md_path)
        print(f"wrote {md_path}")

    for f, line in unparsed:
        print(f"  UNPARSED {f}: {line}", file=sys.stderr)


if __name__ == "__main__":
    main()
