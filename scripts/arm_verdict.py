#!/usr/bin/env python3
"""Typed ship/reject verdict for a DGX arm vs a baseline, via Jev (TypeSafe System One).

v2 (default, --baseline/--arm): candidate-ab driver layout, aggregates repeat boots
(e.g. shipped1/shipped2, off1/off2), applies the 2026-09-24 win rule (quality gate
non-negotiable; win = beyond-noise gain in >=1 coding/counting cell with no c1-c4
cell worse beyond noise; c5-c16 losses cap at ship_with_caveat; c1 judged on the
median of its 5x repeats since it is bimodal). Supersedes v1's fixed rules for new
verdicts -- v1 stays reachable via the positional `arm` argument for reproducing
old verdicts byte-for-byte.

v1 (legacy, positional `arm`): parses results/{arms,benchy}/<arm>* artifacts into a
numbers-only fact sheet vs the `la` baseline, asks Jev one Choice question.
"""
import argparse, glob, json, os, re, statistics, subprocess, sys, time

DGX_ROOT = "dgx-01:~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/"
LA_BAND = (86, 93)
RULES = (
    "Ship-gate rules: hardmode quality score >= 88 (la's own band is 86-93; a score of 86-87 should be rerun once before judging). Fidelity must be "
    "20/20 exact at every probed depth (a depth that misses once should be rerun; judge on 2-of-3 passing). Temp-0 decode throughput (c1/c8/c16 tok/s) "
    "must be within 3% of la at the same concurrency. Default-temperature benchy c1 (depth 0, concurrency 1) gen throughput must be >= +10% vs la, OR "
    "TTFT must be >= 20% better (lower) than la, to ship on speed. The quality gate (hardmode + fidelity) is non-negotiable: no speed win overrides it. "
    "Large swings between reruns are noise, not necessarily a regression -- judge the more favorable-to-reality reading when a rerun clearly "
    "stabilizes, but flag it."
)


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def pct_diff(a, b):
    return None if not b else round(100 * (a - b) / b, 1)


# ---------------------------------------------------------------------------
# v1 (legacy): la-baseline, single-boot arms under results/{arms,benchy}/<arm>*
# ---------------------------------------------------------------------------
def parse_hardmode(d):
    scores, fails = [], []
    for p in sorted(glob.glob(os.path.join(d, "hardmode*.log"))):
        text = _read(p)
        m = re.search(r"Quality:\s*(\d+)/100", text)
        if m:
            scores.append(int(m.group(1)))
        fails.append(len(re.findall(r"❌\s*FAIL", text)))
    return {"quality_scores": scores, "fail_counts": fails}


def parse_fidelity(d):
    out = {}
    for line in _read(os.path.join(d, "fidelity_probe.txt")).splitlines():
        m = re.search(r"depth\s+(\d+).*?exact\s+(\d+).*?wrong\s+(\d+)", line)
        if m:
            out[int(m.group(1))] = {"exact": int(m.group(2)), "wrong": int(m.group(3))}
    return out


def parse_decode(d):
    by_c = {}
    for p in sorted(glob.glob(os.path.join(d, "decode_c*.json"))):
        try:
            data = json.loads(_read(p))
        except json.JSONDecodeError:
            continue
        for row in data.get("rows", []):
            by_c.setdefault(row["c"], []).append(row.get("agg_tok_s"))
    return by_c


def parse_straggler(d):
    """v1: round wall + accept ratio only (no per-request straggler detection)."""
    rounds = {}
    text = _read(os.path.join(d, "straggler.log"))
    for m in re.finditer(r"c=(\d+) round wall ([\d.]+)s.*?accept ([\d.]+)/draft", text, re.S):
        rounds[int(m.group(1))] = {"wall_s": float(m.group(2)), "accept": float(m.group(3))}
    return rounds


def parse_benchy(root, arm):
    """Default-temp prose grid, depth 0/c1. Prefers *-prose16.md over *-prose.md; a `-t0` file is a different (temp-0) benchmark, never the baseline."""
    cands = [p for p in glob.glob(os.path.join(root, "results", "benchy", f"{arm}-prose*.md")) if "-t0" not in p]
    for p in sorted(cands, key=lambda p: (0 if "16" in os.path.basename(p) else 1, p)):
        for line in _read(p).splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 6 or cells[0] in ("depth", "---:"):
                continue
            try:
                depth, conc = int(cells[0]), int(cells[1])
            except ValueError:
                continue
            if depth == 0 and conc == 1:
                return {"gen_tok_s": float(cells[3].split("±")[0]), "ttft_ms": float(cells[5].split("±")[0])}
    return {}


def collect(root, arm):
    d = os.path.join(root, "results", "arms", arm)
    return {
        "hardmode": parse_hardmode(d), "fidelity": parse_fidelity(d),
        "decode_c_tok_s": parse_decode(d), "straggler": parse_straggler(d),
        "benchy_c1": parse_benchy(root, arm),
    }


def noise_suspects(facts, arm):
    out = []
    for c, vals in facts["decode_c_tok_s"].items():
        if len(vals) > 1 and min(vals) > 0 and (max(vals) - min(vals)) / min(vals) > 0.05:
            out.append(f"{arm} c{c} {' vs '.join(str(v) for v in vals)} across reruns")
    qs = facts["hardmode"]["quality_scores"]
    if len(qs) > 1:
        lo, hi = LA_BAND
        note = f" (within la band {lo}-{hi})" if all(lo <= s <= hi for s in qs) else ""
        out.append(f"{arm} hardmode {' then '.join(str(s) for s in qs)}{note}")
    for depth, v in facts["fidelity"].items():
        if v["exact"] < 20:
            out.append(f"{arm} depth {depth} missed {20 - v['exact']}/20 on fidelity probe")
    return out


def fact_sheet(arm, arm_facts, base_facts):
    c_compare = {}
    for c in (1, 8, 16):
        av, bv = arm_facts["decode_c_tok_s"].get(c), base_facts["decode_c_tok_s"].get(c)
        if av and bv:
            c_compare[c] = pct_diff(statistics.mean(av), statistics.mean(bv))
    return {
        "arm": arm,
        "arm_hardmode_scores": arm_facts["hardmode"]["quality_scores"],
        "arm_hardmode_fail_counts": arm_facts["hardmode"]["fail_counts"],
        "la_hardmode_scores": base_facts["hardmode"]["quality_scores"],
        "arm_fidelity_exact_by_depth": {d: v["exact"] for d, v in arm_facts["fidelity"].items()},
        "arm_fidelity_wrong_by_depth": {d: v["wrong"] for d, v in arm_facts["fidelity"].items()},
        "temp0_decode_pct_diff_vs_la_by_c": c_compare,
        "benchy_c1_gain_pct_vs_la": pct_diff(arm_facts["benchy_c1"].get("gen_tok_s", 0), base_facts["benchy_c1"].get("gen_tok_s", 0)),
        "benchy_c1_ttft_gain_pct_vs_la": -pct_diff(arm_facts["benchy_c1"].get("ttft_ms", 0), base_facts["benchy_c1"].get("ttft_ms", 0)),
        "straggler_max_round_wall_s": max((r["wall_s"] for r in arm_facts["straggler"].values()), default=None),
        "noise_suspects": noise_suspects(arm_facts, arm) + noise_suspects(base_facts, "la"),
    }


def ask_jev_v1(sheet):
    from typesafe_sdk import Choice, TypeSafeClient
    api_key = subprocess.run(
        ["security", "find-generic-password", "-s", "dev/typesafe-ai-api-key", "-w"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    with TypeSafeClient(api_key=api_key) as client:
        response = client.system_one(
            {"rules": RULES, "facts": sheet},
            {"verdict": Choice(
                instructions=(
                    "Given `rules` and `facts` (a fact sheet for arm `facts.arm` vs baseline `la`, "
                    "numbers only, plus any `noise_suspects`), what is the ship verdict for this arm?"
                ),
                criteria={
                    "ship": "Clears the quality gate and the speed bar with clean, non-noisy numbers.",
                    "ship_with_caveat": "Clears the quality gate and speed bar, but a number is borderline or a noise_suspect entry casts some doubt.",
                    "reject": "Fails the quality gate (hardmode < 88 after any rerun, or fidelity misses persist), or fails the speed bar with no offsetting caveat.",
                    "rerun": "Too noisy or too few data points to call; a rule-required rerun (hardmode 86-87, or a single fidelity miss) has not happened yet."},
            )},
        )
    answer = response.choices["verdict"]
    return answer.choice, answer.confidence


def reason_for_v1(verdict, sheet):
    bits = [
        f"hardmode={sheet['arm_hardmode_scores'] or 'missing'}", f"fidelity_wrong_total={sum(sheet['arm_fidelity_wrong_by_depth'].values())}",
        f"c1/8/16_diff_vs_la={sheet['temp0_decode_pct_diff_vs_la_by_c']}", f"benchy_c1_gain={sheet['benchy_c1_gain_pct_vs_la']}%",
        f"ttft_gain={sheet['benchy_c1_ttft_gain_pct_vs_la']}%",
    ]
    if sheet["noise_suspects"]:
        bits.append(f"{len(sheet['noise_suspects'])} noise suspect(s)")
    return f"Jev verdict={verdict} from: " + "; ".join(bits)


def fetch(arm, scratch):
    for rel in (
        f"results/benchy/{arm}-prose*.md", f"results/arms/{arm}/decode_c*.json", f"results/arms/{arm}/hardmode*.log",
        f"results/arms/{arm}/fidelity_probe.txt", f"results/arms/{arm}/straggler.log",
    ):
        dest = os.path.join(scratch, os.path.dirname(rel))
        os.makedirs(dest, exist_ok=True)
        subprocess.run(["rsync", "-az", f"{DGX_ROOT}{rel}", dest + "/"], check=False)


SELFTEST_SHEET = {
    "arm": "selftest-arm", "arm_hardmode_scores": [86, 90], "arm_hardmode_fail_counts": [7, 1], "la_hardmode_scores": [91, 88, 89],
    "arm_fidelity_exact_by_depth": {8000: 20, 32000: 20, 64000: 20, 128000: 19},
    "arm_fidelity_wrong_by_depth": {8000: 0, 32000: 0, 64000: 0, 128000: 1},
    "temp0_decode_pct_diff_vs_la_by_c": {1: 1.2, 8: -0.5, 16: -2.1},
    "benchy_c1_gain_pct_vs_la": 12.4, "benchy_c1_ttft_gain_pct_vs_la": 1.1, "straggler_max_round_wall_s": 13.7,
    "noise_suspects": ["selftest-arm c16 399 vs 641 across reruns", "selftest-arm hardmode 86 then 90 (within la band 86-93)"],
}


def selftest_parse_benchy():
    """Regression for the la-prose-t0.md baseline bug: la (45.97/706.81) vs x (56.48/755.90) must give +22.9% gen gain, -6.9% (worse) TTFT gain."""
    import tempfile
    head = "| depth | concurrency | prompt t/s (agg) | gen t/s (agg) | gen t/s (per stream) | e2e TTFT (ms) |\n"
    row = "| 0 | 1 | 1 ± 1 | {gen} ± 1 | {gen} ± 1 | {ttft} ± 1 |\n"
    with tempfile.TemporaryDirectory() as t:
        d = os.path.join(t, "results", "benchy")
        os.makedirs(d)
        for name, gen, ttft in (("la-prose.md", 45.97, 706.81), ("la-prose-t0.md", 999, 1), ("x-prose.md", 56.48, 755.90)):
            open(os.path.join(d, name), "w").write(head + row.format(gen=gen, ttft=ttft))
        base, arm = parse_benchy(t, "la"), parse_benchy(t, "x")
        gain, ttft_gain = pct_diff(arm["gen_tok_s"], base["gen_tok_s"]), -pct_diff(arm["ttft_ms"], base["ttft_ms"])
        assert abs(gain - 22.9) < 0.05 and abs(ttft_gain - (-6.9)) < 0.05, (gain, ttft_gain)
    print(f"selftest_parse_benchy OK: gain=+{gain}% ttft_gain={ttft_gain}%")


def run_v1(sheet, label):
    verdict, confidence = ask_jev_v1(sheet)
    result = {
        "verdict": verdict, "reason": reason_for_v1(verdict, sheet),
        "noise_suspects": sheet["noise_suspects"], "confidence": confidence,
    }
    print(json.dumps(result, indent=2))
    print(f"[{label}] {result['verdict']} (confidence={result['confidence']:.2f}): {result['reason']}")


# ---------------------------------------------------------------------------
# v2 (default): candidate-ab layout, repeat-boot aggregation, 2026-09-24 win rule
# ---------------------------------------------------------------------------
GRID_DEPTHS = (0, 16384)
GRID_CONCS = (1, 2, 4, 5, 8, 10, 16)
LOW_CONCS = (1, 2, 4)
NOISE_FLOOR_PCT = 3.0
QUALITY_BAND = (86, 93)
QUALITY_MIN = 88
FIDELITY_DEPTHS = (8000, 32000, 64000, 128000)
STRAGGLER_LO_C, STRAGGLER_HI_C = 5, 16
STRAGGLER_MULT = 1.5

RULES_V2 = (
    "Win rule (2026-09-24). Quality gate (non-negotiable, checked in code before any speed "
    "judgment): hardmode quality >= 88 (baseline band 86-93; a lone 86-87 score needs a rerun "
    "before it counts), fidelity 20/20 exact at every probed depth (8k/32k/64k/128k), no "
    "stragglers (no single request more than 1.5x the median round wall at concurrency 5-16). "
    "Any quality-gate failure is an automatic reject regardless of speed -- `quality_gate.pass` "
    "is already false in that case and no speed reasoning should override it. Speed (only "
    "reached when quality_gate.pass is true): WIN if the candidate is faster beyond noise in at "
    "least one cell of the coding grid (task mode, depths 0 and 16384, concurrency 1,2,4,5,8,10,16) "
    "or the counting sweep, AND no cell at concurrency 1-4 (coding or counting) is worse beyond "
    "noise -- a low-concurrency (c1-c4) regression beyond noise means this must NOT be a plain "
    "ship, at best ship_with_caveat, more likely reject if the regression is not tiny. Losses only "
    "at concurrency 5-16 beyond noise cap the verdict at ship_with_caveat at best; list them. Noise "
    "is the larger of the boot-to-boot spread between repeat boots of the same arm at that cell and "
    "3%. Concurrency 1 is bimodal: judge it on the median of the 5x c1 repeat runs, not a single "
    "sample. `facts.noisy_cells` lists cells that look flat (no win, no loss beyond noise) only "
    "because the noise threshold itself is wide (boot-to-boot spread well above the 3% floor) -- "
    "that is inconclusive, not evidence of no difference, and should push toward rerun rather than "
    "reject when it is the only reason nothing beat the noise. Never claim a gain the numbers don't show."
)


def parse_task_csv(path):
    """llama-benchy markdown table. Keys: (mode, depth, c) -> mean t/s (total).
    mode in {pp2048, tg512, ctx_pp, ctx_tg}; depth 0 for bare rows like `tg512 (c4)`,
    else the @ dNNNN depth, e.g. `tg512 @ d16384 (c4)`."""
    row_re = re.compile(r"^(pp2048|tg512|ctx_pp|ctx_tg)(?:\s*@\s*d(\d+))?\s*\(c(\d+)\)$")
    out = {}
    for line in _read(path).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        m = row_re.match(cells[1])
        if not m:
            continue
        mode, depth_s, c_s = m.groups()
        depth, c = int(depth_s) if depth_s else 0, int(c_s)
        val = cells[2]
        try:
            out[(mode, depth, c)] = float(val.split("±")[0].strip())
        except ValueError:
            continue
    return out


def parse_count_json(path):
    """bench_sweep output. Returns {c: agg_tok_s}."""
    try:
        data = json.loads(_read(path))
    except json.JSONDecodeError:
        return {}
    return {row["c"]: row["agg_tok_s"] for row in data.get("rows", []) if "agg_tok_s" in row}


def parse_count_c1_repeats(dirpath):
    """The 5x count-c1-*.json repeat runs used to judge the bimodal c1 cell by median."""
    vals = []
    for p in sorted(glob.glob(os.path.join(dirpath, "count-c1-*.json"))):
        for c, v in parse_count_json(p).items():
            if c == 1:
                vals.append(v)
    return vals


def parse_straggler_v2(path):
    """Per-c round wall + per-request wall times, for the c5-c16 straggler check."""
    out = {}
    text = _read(path)
    for m in re.finditer(r"c=(\d+) round wall ([\d.]+)s \| per request \(submit order\): (.*)", text):
        c, wall, reqs_text = int(m.group(1)), float(m.group(2)), m.group(3)
        reqs = [float(x) for x in re.findall(r"#\d+:([\d.]+)s", reqs_text)]
        out[c] = {"wall_s": wall, "reqs_s": reqs}
    return out


def straggler_violations(straggler_by_c, label):
    rounds = {c: d for c, d in straggler_by_c.items() if STRAGGLER_LO_C <= c <= STRAGGLER_HI_C}
    if not rounds:
        return []
    median_wall = statistics.median(d["wall_s"] for d in rounds.values())
    out = []
    for c, d in rounds.items():
        for i, req_s in enumerate(d["reqs_s"]):
            if req_s > STRAGGLER_MULT * median_wall:
                out.append(f"{label} c={c} req#{i} {req_s}s > {STRAGGLER_MULT}x median round wall {median_wall}s")
    return out


def collect_boot(d):
    return {
        "dir": d,
        "task": parse_task_csv(os.path.join(d, "task.csv")),
        "count": parse_count_json(os.path.join(d, "count.json")),
        "count_c1_repeats": parse_count_c1_repeats(d),
        "hardmode": parse_hardmode(d) if _read(os.path.join(d, "hardmode.log")) or glob.glob(os.path.join(d, "hardmode*.log")) else None,
        "fidelity": parse_fidelity(d),
        "straggler": parse_straggler_v2(os.path.join(d, "straggler.log")),
    }


def gate_dir_for(arms_root, label):
    """Prefer a dedicated gate dir under results/arms/<label>/ (hardmode.log,
    fidelity_probe.txt, straggler.log); fall back to the boot dir itself, since
    the driver sometimes writes gate outputs alongside task.csv (e.g. gdndef-ab/off1)."""
    d = os.path.join(arms_root, label)
    return d if os.path.isdir(d) else None


def find_arms_root(root):
    root = os.path.normpath(root)
    for cand in (os.path.join(root, "arms"), os.path.join(os.path.dirname(root), "arms")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(root, "arms")


def collect_repeats(root, name, arms_root):
    dirs = sorted(glob.glob(os.path.join(root, f"{name}[0-9]*")))
    boots = []
    for d in dirs:
        label = os.path.basename(d)
        gate_dir = gate_dir_for(arms_root, label) or d  # fall back to the boot dir itself
        boot = collect_boot(d)
        if gate_dir != d:
            boot["hardmode"] = parse_hardmode(gate_dir)
            boot["fidelity"] = parse_fidelity(gate_dir) or boot["fidelity"]
            boot["straggler"] = parse_straggler_v2(os.path.join(gate_dir, "straggler.log")) or boot["straggler"]
        elif boot["hardmode"] is None:
            boot["hardmode"] = {"quality_scores": [], "fail_counts": []}
        boots.append(boot)
    return boots


def is_remote(root):
    head = root.split("/", 1)[0]
    return ":" in head


def fetch_v2(root, baseline, arm, scratch):
    """rsync (read-only) a candidate-ab dir + its sibling results/arms gate dirs from
    dgx-01, in v1's DGX_ROOT style: `host:path`."""
    host, path = root.split(":", 1)
    local_root = os.path.join(scratch, "candidate-ab")
    local_arms = os.path.join(scratch, "arms")
    os.makedirs(local_root, exist_ok=True)
    os.makedirs(local_arms, exist_ok=True)
    remote_arms = path.rstrip("/").rsplit("/", 1)[0] + "/arms"
    for label_prefix in (baseline, arm):
        subprocess.run(["rsync", "-az", "-e", "ssh", f"{host}:{path.rstrip('/')}/{label_prefix}*", local_root + "/"], check=False)
        subprocess.run(["rsync", "-az", "-e", "ssh", f"{host}:{remote_arms}/{label_prefix}*", local_arms + "/"], check=False)
    return local_root


def noise_pct(vals):
    if len(vals) < 2 or min(vals) <= 0:
        return NOISE_FLOOR_PCT
    return max(100 * (max(vals) - min(vals)) / min(vals), NOISE_FLOOR_PCT)


def compare_cell(cand_vals, base_vals):
    if not cand_vals or not base_vals:
        return None
    cand_mean, base_mean = statistics.mean(cand_vals), statistics.mean(base_vals)
    diff = pct_diff(cand_mean, base_mean)
    if diff is None:
        return None
    threshold = max(noise_pct(cand_vals), noise_pct(base_vals))
    return {
        "cand_mean": round(cand_mean, 2), "base_mean": round(base_mean, 2),
        "pct_diff": diff, "noise_pct": round(threshold, 1),
        "beyond_noise": abs(diff) > threshold,
    }


def coding_grid_compare(cand_boots, base_boots):
    grid = {}
    for depth in GRID_DEPTHS:
        for c in GRID_CONCS:
            cand_vals = [b["task"][("tg512", depth, c)] for b in cand_boots if ("tg512", depth, c) in b["task"]]
            base_vals = [b["task"][("tg512", depth, c)] for b in base_boots if ("tg512", depth, c) in b["task"]]
            cell = compare_cell(cand_vals, base_vals)
            if cell:
                grid[f"d{depth}_c{c}"] = cell
    return grid


def counting_compare(cand_boots, base_boots):
    out = {}
    for c in GRID_CONCS:
        if c == 1:
            # bimodal: judge on the median of each boot's 5x c1 repeats
            cand_vals = [statistics.median(b["count_c1_repeats"]) for b in cand_boots if b["count_c1_repeats"]]
            base_vals = [statistics.median(b["count_c1_repeats"]) for b in base_boots if b["count_c1_repeats"]]
            if not cand_vals:
                cand_vals = [b["count"][1] for b in cand_boots if 1 in b["count"]]
            if not base_vals:
                base_vals = [b["count"][1] for b in base_boots if 1 in b["count"]]
        else:
            cand_vals = [b["count"][c] for b in cand_boots if c in b["count"]]
            base_vals = [b["count"][c] for b in base_boots if c in b["count"]]
        cell = compare_cell(cand_vals, base_vals)
        if cell:
            out[f"c{c}"] = cell
    return out


def quality_gate(boots, label):
    scores = [s for b in boots for s in (b["hardmode"] or {}).get("quality_scores", [])]
    reasons, ok = [], True
    if not scores:
        ok = False
        reasons.append(f"{label}: no hardmode score found")
    else:
        below = [s for s in scores if s < QUALITY_BAND[0]]
        passing = [s for s in scores if s >= QUALITY_MIN]
        if below:
            ok = False
            reasons.append(f"{label}: hardmode {below} below band {QUALITY_BAND}")
        elif passing:
            reasons.append(f"{label}: hardmode {passing} clears >= {QUALITY_MIN}")
        elif len(scores) < 2:
            ok = False
            reasons.append(f"{label}: hardmode {scores} in 86-87 band, needs a rerun")
        else:
            ok = False
            reasons.append(f"{label}: hardmode {scores} stayed in 86-87 band after rerun, below {QUALITY_MIN}")

    exact_by_depth = {}
    for b in boots:
        for d, v in b["fidelity"].items():
            exact_by_depth[d] = max(exact_by_depth.get(d, 0), v["exact"])
    missing = [d for d in FIDELITY_DEPTHS if d not in exact_by_depth]
    misses = [d for d in FIDELITY_DEPTHS if exact_by_depth.get(d, 0) < 20]
    if missing:
        ok = False
        reasons.append(f"{label}: fidelity missing depths {missing}")
    if misses:
        ok = False
        reasons.append(f"{label}: fidelity {[f'{d}:{exact_by_depth[d]}/20' for d in misses]}")

    stragglers = []
    for b in boots:
        stragglers += straggler_violations(b["straggler"], f"{label}:{os.path.basename(b['dir'])}")
    if stragglers:
        ok = False
        reasons.append(f"{label}: {len(stragglers)} straggler violation(s)")

    return {
        "pass": ok, "hardmode_scores": scores, "fidelity_exact_by_depth": exact_by_depth,
        "straggler_violations": stragglers, "reasons": reasons,
    }


def fact_sheet_v2(baseline, arm, base_boots, cand_boots):
    coding = coding_grid_compare(cand_boots, base_boots)
    counting = counting_compare(cand_boots, base_boots)
    all_cells = {**{f"coding_{k}": v for k, v in coding.items()}, **{f"counting_{k}": v for k, v in counting.items()}}
    wins = [k for k, v in all_cells.items() if v["beyond_noise"] and v["pct_diff"] > 0]
    losses = [k for k, v in all_cells.items() if v["beyond_noise"] and v["pct_diff"] < 0]
    low_c_losses = [k for k in losses if any(k.endswith(f"_c{c}") for c in LOW_CONCS)]
    mid_high_losses = [k for k in losses if k not in low_c_losses]
    # Cells where the noise threshold is well above the 3% floor (boot-to-boot spread
    # dominates) but the cell shows neither a win nor a loss: the diff may be real and
    # simply swamped by run-to-run variance, not evidence of "no difference". Jev should
    # prefer rerun over reject when these are the only reason a cell looks flat.
    noisy_cells = [k for k, v in all_cells.items() if v["noise_pct"] > 3 * NOISE_FLOOR_PCT and not v["beyond_noise"]]
    qgate = quality_gate(cand_boots, arm)
    return {
        "baseline": baseline, "arm": arm,
        "baseline_boots": len(base_boots), "arm_boots": len(cand_boots),
        "quality_gate": qgate,
        "coding_grid_pct_diff": coding, "counting_pct_diff": counting,
        "wins_beyond_noise": wins, "losses_beyond_noise_low_c": low_c_losses, "losses_beyond_noise_mid_high_c": mid_high_losses,
        "noisy_cells": noisy_cells,
    }


def ask_jev_v2(sheet):
    from typesafe_sdk import Choice, TypeSafeClient
    api_key = subprocess.run(
        ["security", "find-generic-password", "-s", "dev/typesafe-ai-api-key", "-w"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    with TypeSafeClient(api_key=api_key) as client:
        response = client.system_one(
            {"rules": RULES_V2, "facts": sheet},
            {"verdict": Choice(
                instructions=(
                    "Given `rules` and `facts` (candidate arm `facts.arm` vs baseline `facts.baseline`; "
                    "`facts.quality_gate.pass` is already computed and non-negotiable; per-cell speed "
                    "deltas are in `coding_grid_pct_diff`/`counting_pct_diff` with `beyond_noise` already "
                    "flagged, and pre-sorted into wins/losses lists split by concurrency), what is the "
                    "ship verdict for this arm?"
                ),
                criteria={
                    "ship": "quality_gate.pass is true, there is a win beyond noise, and there is no beyond-noise loss anywhere (low or mid/high concurrency).",
                    "ship_with_caveat": "quality_gate.pass is true and there is a win beyond noise, but there is also a beyond-noise loss at concurrency 5-16 (or the win/loss picture is borderline) -- list the losses as caveats.",
                    "reject": "quality_gate.pass is false, or there is a confidently-flat/negative picture with no beyond-noise win and no unresolved noisy_cells, or there is a beyond-noise low-concurrency (c1-c4) loss that is not trivial.",
                    "rerun": "Too few boots, or noisy_cells is non-empty and is the reason no cell shows a beyond-noise win -- the result may be real but is currently swamped by boot-to-boot spread; more repeat boots are needed before judging."},
            )},
        )
    answer = response.choices["verdict"]
    return answer.choice, answer.confidence


def cap_verdict(verdict, sheet):
    """Deterministic floor under Jev's call: the quality gate and the low-c/mid-high-c
    loss rules are enforced in code, not left to the model's compliance."""
    if not sheet["quality_gate"]["pass"]:
        return "reject", "quality gate failed (forced, no speed override)"
    if sheet["losses_beyond_noise_low_c"] and verdict == "ship":
        return "ship_with_caveat", f"capped from ship: low-concurrency loss(es) {sheet['losses_beyond_noise_low_c']}"
    if sheet["losses_beyond_noise_mid_high_c"] and verdict == "ship":
        return "ship_with_caveat", f"capped from ship: c5-c16 loss(es) {sheet['losses_beyond_noise_mid_high_c']}"
    return verdict, None


def reason_for_v2(sheet, verdict):
    qg = sheet["quality_gate"]
    bits = [
        f"quality_gate={'pass' if qg['pass'] else 'fail'}", f"hardmode={qg['hardmode_scores'] or 'missing'}",
        f"wins={len(sheet['wins_beyond_noise'])}", f"low_c_losses={len(sheet['losses_beyond_noise_low_c'])}",
        f"mid_high_losses={len(sheet['losses_beyond_noise_mid_high_c'])}", f"noisy_cells={len(sheet['noisy_cells'])}",
    ]
    return f"Jev verdict={verdict} from: " + "; ".join(bits)


def run_v2(sheet, offline=False, root=None, json_path=None):
    if not sheet["quality_gate"]["pass"]:
        verdict, confidence, cap_note = "reject", 1.0, "quality gate failed (forced, no Jev call)"
    elif offline:
        verdict, confidence, cap_note = "rerun", 0.0, "offline mode: speed verdict needs Jev, not computed"
    else:
        raw_verdict, confidence = ask_jev_v2(sheet)
        verdict, cap_note = cap_verdict(raw_verdict, sheet)
    # Merged fact sheet + verdict, one JSON object, matching the shape the Jev dashboard's
    # verdicts view (jev/public/app.js armCard) and jev/examples/verdicts/*.json expect.
    result = {
        **sheet,
        "verdict": verdict, "confidence": confidence, "cap_note": cap_note,
        "reason": reason_for_v2(sheet, verdict),
        "noise_suspects": [],  # v1 field kept for schema parity; v2 signals this via noisy_cells instead
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": root,
    }
    print(json.dumps(result, indent=2))
    print(f"[{sheet['arm']} vs {sheet['baseline']}] {verdict} (confidence={confidence:.2f})"
          + (f" -- {cap_note}" if cap_note else ""))
    if json_path:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(result, f, indent=1)
        print(f"wrote {json_path}")
    return result


# ---------------------------------------------------------------------------
# v2 selftest: clean win, c1 regression, quality fail, noisy case
# ---------------------------------------------------------------------------
def _synthetic_boot(scratch, name, task_rows, count_rows, count_c1, hardmode_score, fidelity_ok=True, straggler_ok=True):
    d = os.path.join(scratch, name)
    os.makedirs(d, exist_ok=True)
    head = "| model | test | t/s (total) | t/s (req) | peak t/s | peak t/s (req) | ttfr (ms) | est_ppt (ms) | e2e_ttft (ms) | accept/draft | prefix-hit |\n"
    lines = [head]
    for (mode, depth, c), val in task_rows.items():
        test = f"{mode} (c{c})" if depth == 0 else f"{mode} @ d{depth} (c{c})"
        lines.append(f"| qwen3.8-flash-next | {test} | {val} ± 1 | {val} ± 1 |  |  |  |  |  | 0.4 | 0 |\n")
    open(os.path.join(d, "task.csv"), "w").write("".join(lines))
    json.dump({"rows": [{"c": c, "agg_tok_s": v} for c, v in count_rows.items()]}, open(os.path.join(d, "count.json"), "w"))
    for i, v in enumerate(count_c1, 1):
        json.dump({"rows": [{"c": 1, "agg_tok_s": v}]}, open(os.path.join(d, f"count-c1-{i}.json"), "w"))
    if hardmode_score is not None:
        open(os.path.join(d, "hardmode.log"), "w").write(f"│    Quality:        {hardmode_score}/100                                                    │\n")
    fid_exact = 20 if fidelity_ok else 15
    open(os.path.join(d, "fidelity_probe.txt"), "w").write("\n".join(
        f"depth {dep} | actual_tokens 1000 | exact {fid_exact} near 0 wrong {20 - fid_exact} explore 0 no_call 0 | typo_rate 0.00 | ttft 1.0s | lat 1.0s"
        for dep in FIDELITY_DEPTHS))
    reqs = "#0:5.0s/320t #1:5.0s/320t" if straggler_ok else "#0:5.0s/320t #1:50.0s/320t"
    open(os.path.join(d, "straggler.log"), "w").write(f"c=8 round wall 5.0s | per request (submit order): {reqs}\n")
    return d


def _v2_scenario_clean_win(scratch):
    base = _synthetic_boot(scratch, "shipped1", {("tg512", 0, 1): 60, ("tg512", 16384, 1): 55, ("tg512", 0, 16): 200},
                            {8: 400, 16: 600}, [100, 101, 99, 100, 102], 90)
    cand = _synthetic_boot(scratch, "off1", {("tg512", 0, 1): 62, ("tg512", 16384, 1): 57, ("tg512", 0, 16): 260},
                            {8: 410, 16: 780}, [102, 103, 101, 103, 104], 90)
    base_boots, cand_boots = [collect_boot(base)], [collect_boot(cand)]
    sheet = fact_sheet_v2("shipped", "off", base_boots, cand_boots)
    assert sheet["quality_gate"]["pass"], sheet["quality_gate"]
    assert not sheet["losses_beyond_noise_low_c"], sheet["losses_beyond_noise_low_c"]
    assert "coding_d0_c16" in sheet["wins_beyond_noise"] or "counting_c16" in sheet["wins_beyond_noise"], sheet["wins_beyond_noise"]
    return sheet


def _v2_scenario_c1_regression(scratch):
    base = _synthetic_boot(scratch, "shipped1", {("tg512", 0, 1): 60, ("tg512", 0, 16): 200}, {1: 100, 16: 600},
                            [100, 101, 99, 100, 102], 90)
    cand = _synthetic_boot(scratch, "on1", {("tg512", 0, 1): 45, ("tg512", 0, 16): 260}, {1: 78, 16: 780},
                            [78, 79, 77, 78, 80], 90)
    base_boots, cand_boots = [collect_boot(base)], [collect_boot(cand)]
    sheet = fact_sheet_v2("shipped", "on", base_boots, cand_boots)
    assert sheet["quality_gate"]["pass"], sheet["quality_gate"]
    assert sheet["losses_beyond_noise_low_c"], "expected a beyond-noise c1 loss"
    verdict, _ = cap_verdict("ship", sheet)
    assert verdict != "ship", verdict
    return sheet


def _v2_scenario_quality_fail(scratch):
    base = _synthetic_boot(scratch, "shipped1", {("tg512", 0, 1): 60}, {1: 100}, [100] * 5, 90)
    cand = _synthetic_boot(scratch, "off1", {("tg512", 0, 1): 90}, {1: 150}, [150] * 5, 80)  # hardmode below band
    base_boots, cand_boots = [collect_boot(base)], [collect_boot(cand)]
    sheet = fact_sheet_v2("shipped", "off", base_boots, cand_boots)
    assert not sheet["quality_gate"]["pass"], sheet["quality_gate"]
    verdict, note = cap_verdict("ship", sheet)
    assert verdict == "reject", verdict
    return sheet


def _v2_scenario_noisy(scratch):
    # two candidate boots that disagree with each other more than they disagree with baseline -> high noise
    base = _synthetic_boot(scratch, "shipped1", {("tg512", 0, 1): 60}, {1: 100}, [100] * 5, 90)
    cand1 = _synthetic_boot(scratch, "off1", {("tg512", 0, 1): 63}, {1: 105}, [105] * 5, 90)
    cand2 = _synthetic_boot(scratch, "off2", {("tg512", 0, 1): 80}, {1: 130}, [130] * 5, 90)
    base_boots = [collect_boot(base)]
    cand_boots = [collect_boot(cand1), collect_boot(cand2)]
    sheet = fact_sheet_v2("shipped", "off", base_boots, cand_boots)
    cell = sheet["coding_grid_pct_diff"]["d0_c1"]
    assert cell["noise_pct"] > NOISE_FLOOR_PCT, cell  # boot-to-boot spread (63 vs 80) should dominate the 3% floor
    assert "coding_d0_c1" in sheet["noisy_cells"], sheet["noisy_cells"]
    return sheet


def selftest_v2(offline=False, json_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        scenarios = {
            "clean_win": _v2_scenario_clean_win(os.path.join(t, "clean_win")),
            "c1_regression": _v2_scenario_c1_regression(os.path.join(t, "c1_regression")),
            "quality_fail": _v2_scenario_quality_fail(os.path.join(t, "quality_fail")),
            "noisy": _v2_scenario_noisy(os.path.join(t, "noisy")),
        }
    print("selftest_v2 deterministic checks OK: clean_win/c1_regression/quality_fail/noisy")
    if json_path:
        # clean_win has the richest grid (multiple depths/concurrencies), best for exercising
        # the dashboard heatmap; --offline keeps this call API-free.
        run_v2(scenarios["clean_win"], offline=offline, root="selftest", json_path=json_path)
    if offline:
        return
    for name, sheet in scenarios.items():
        if json_path and name == "clean_win":
            continue  # already written above
        print(f"--- v2 selftest scenario: {name} ---")
        run_v2(sheet, offline=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("legacy_arm", nargs="?", metavar="arm", help="[legacy/v1] arm name, e.g. la-mtpprob")
    ap.add_argument("--root", default=".", help="local root with results/ (v1), or the candidate-ab-<date> dir (v2); "
                                                 "'host:path' (v1's DGX_ROOT style) fetches read-only via rsync first")
    ap.add_argument("--fetch", action="store_true", help="rsync this arm (+ baseline) from dgx-01 into --root")
    ap.add_argument("--baseline", default="shipped", help="[v2] baseline label prefix, e.g. shipped (repeats shipped1, shipped2, ...)")
    ap.add_argument("--arm", dest="arm2", metavar="ARM", help="[v2] candidate label prefix, e.g. off | on (repeats off1, off2, ...)")
    ap.add_argument("--selftest", action="store_true", help="run v1 + v2 selftests against embedded/synthetic fact sheets")
    ap.add_argument("--offline", action="store_true", help="--selftest: skip Jev API calls, check deterministic parts only")
    ap.add_argument("--json", metavar="PATH", help="[v2] write the merged fact sheet + verdict as one JSON object to PATH "
                                                     "(jev dashboard --verdicts <dir> format); with --selftest, writes the "
                                                     "clean_win scenario")
    args = ap.parse_args()

    if args.selftest:
        selftest_parse_benchy()
        selftest_v2(offline=args.offline, json_path=args.json)
        if not args.offline:
            run_v1(SELFTEST_SHEET, "selftest")
        return

    if args.arm2:
        root = args.root
        if is_remote(root):
            root = fetch_v2(root, args.baseline, args.arm2, "/tmp/arm_verdict_fetch")
        arms_root = find_arms_root(root)
        base_boots = collect_repeats(root, args.baseline, arms_root)
        cand_boots = collect_repeats(root, args.arm2, arms_root)
        if not base_boots or not cand_boots:
            sys.exit(f"no repeat boots found for baseline={args.baseline!r} or arm={args.arm2!r} under {root}")
        sheet = fact_sheet_v2(args.baseline, args.arm2, base_boots, cand_boots)
        run_v2(sheet, root=root, json_path=args.json)
        return

    if not args.legacy_arm:
        ap.error("arm is required unless --selftest or --arm is given")
    if args.fetch:
        fetch(args.legacy_arm, args.root)
        fetch("la", args.root)
    sheet = fact_sheet(args.legacy_arm, collect(args.root, args.legacy_arm), collect(args.root, "la"))
    run_v1(sheet, args.legacy_arm)


if __name__ == "__main__":
    sys.exit(main())
