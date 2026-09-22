#!/usr/bin/env python3
"""Typed ship/reject verdict for a DGX arm vs the `la` baseline, via Jev (TypeSafe System One).
Parses results/{arms,benchy}/<arm>* artifacts into a numbers-only fact sheet, asks Jev one
Choice question against our written quality-gate rules, prints JSON + one-line summary."""
import argparse, glob, json, os, re, statistics, subprocess, sys

DGX_ROOT = "dgx-01:~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/"
LA_BAND = (86, 93)
RULES = (
    "Ship-gate rules: hardmode quality score >= 88 (la's own band is 86-93; a score of "
    "86-87 should be rerun once before judging). Fidelity must be 20/20 exact at every "
    "probed depth (a depth that misses once should be rerun; judge on 2-of-3 passing). "
    "Temp-0 decode throughput (c1/c8/c16 tok/s) must be within 3% of la at the same "
    "concurrency. Default-temperature benchy c1 (depth 0, concurrency 1) gen throughput "
    "must be >= +10% vs la, OR TTFT must be >= 20% better (lower) than la, to ship on "
    "speed. The quality gate (hardmode + fidelity) is non-negotiable: no speed win "
    "overrides it. Large swings between reruns are noise, not necessarily a regression -- "
    "judge the more favorable-to-reality reading when a rerun clearly stabilizes, but flag it."
)
def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""
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
    rounds = {}
    text = _read(os.path.join(d, "straggler.log"))
    for m in re.finditer(r"c=(\d+) round wall ([\d.]+)s.*?accept ([\d.]+)/draft", text, re.S):
        rounds[int(m.group(1))] = {"wall_s": float(m.group(2)), "accept": float(m.group(3))}
    return rounds
def parse_benchy(root, arm):
    """First prose table row at depth 0, concurrency 1 (default-temperature c1)."""
    for p in sorted(glob.glob(os.path.join(root, "results", "benchy", f"{arm}-prose*.md"))):
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
def pct_diff(a, b):
    return None if not b else round(100 * (a - b) / b, 1)
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
        "benchy_c1_ttft_gain_pct_vs_la": pct_diff(base_facts["benchy_c1"].get("ttft_ms", 0), arm_facts["benchy_c1"].get("ttft_ms", 0)),
        "straggler_max_round_wall_s": max((r["wall_s"] for r in arm_facts["straggler"].values()), default=None),
        "noise_suspects": noise_suspects(arm_facts, arm) + noise_suspects(base_facts, "la"),
    }
def ask_jev(sheet):
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
                    "Given `rules` and `facts` (a fact sheet for arm `facts.arm` vs baseline "
                    "`la`, numbers only, plus any `noise_suspects`), what is the ship verdict "
                    "for this arm?"
                ),
                criteria={
                    "ship": "Clears the quality gate and the speed bar with clean, non-noisy numbers.",
                    "ship_with_caveat": "Clears the quality gate and speed bar, but a number is borderline or a noise_suspect entry casts some doubt.",
                    "reject": "Fails the quality gate (hardmode < 88 after any rerun, or fidelity misses persist), or fails the speed bar with no offsetting caveat.",
                    "rerun": "Too noisy or too few data points to call; a rule-required rerun (hardmode 86-87, or a single fidelity miss) has not happened yet.",
                },
            )},
        )
    answer = response.choices["verdict"]
    return answer.choice, answer.confidence
def reason_for(verdict, sheet):
    bits = [
        f"hardmode={sheet['arm_hardmode_scores'] or 'missing'}",
        f"fidelity_wrong_total={sum(sheet['arm_fidelity_wrong_by_depth'].values())}",
        f"c1/8/16_diff_vs_la={sheet['temp0_decode_pct_diff_vs_la_by_c']}",
        f"benchy_c1_gain={sheet['benchy_c1_gain_pct_vs_la']}%",
        f"ttft_gain={sheet['benchy_c1_ttft_gain_pct_vs_la']}%",
    ]
    if sheet["noise_suspects"]:
        bits.append(f"{len(sheet['noise_suspects'])} noise suspect(s)")
    return f"Jev verdict={verdict} from: " + "; ".join(bits)
def fetch(arm, scratch):
    for rel in (
        f"results/benchy/{arm}-prose*.md", f"results/arms/{arm}/decode_c*.json",
        f"results/arms/{arm}/hardmode*.log", f"results/arms/{arm}/fidelity_probe.txt",
        f"results/arms/{arm}/straggler.log",
    ):
        dest = os.path.join(scratch, os.path.dirname(rel))
        os.makedirs(dest, exist_ok=True)
        subprocess.run(["rsync", "-az", f"{DGX_ROOT}{rel}", dest + "/"], check=False)
SELFTEST_SHEET = {
    "arm": "selftest-arm", "arm_hardmode_scores": [86, 90], "arm_hardmode_fail_counts": [7, 1],
    "la_hardmode_scores": [91, 88, 89],
    "arm_fidelity_exact_by_depth": {8000: 20, 32000: 20, 64000: 20, 128000: 19},
    "arm_fidelity_wrong_by_depth": {8000: 0, 32000: 0, 64000: 0, 128000: 1},
    "temp0_decode_pct_diff_vs_la_by_c": {1: 1.2, 8: -0.5, 16: -2.1},
    "benchy_c1_gain_pct_vs_la": 12.4, "benchy_c1_ttft_gain_pct_vs_la": 1.1,
    "straggler_max_round_wall_s": 13.7,
    "noise_suspects": [
        "selftest-arm c16 399 vs 641 across reruns",
        "selftest-arm hardmode 86 then 90 (within la band 86-93)",
    ],
}
def run(sheet, label):
    verdict, confidence = ask_jev(sheet)
    result = {
        "verdict": verdict, "reason": reason_for(verdict, sheet),
        "noise_suspects": sheet["noise_suspects"], "confidence": confidence,
    }
    print(json.dumps(result, indent=2))
    print(f"[{label}] {result['verdict']} (confidence={result['confidence']:.2f}): {result['reason']}")
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", nargs="?", help="arm name, e.g. la-mtpprob")
    ap.add_argument("--root", default=".", help="local root with results/ (default: cwd)")
    ap.add_argument("--fetch", action="store_true", help="rsync this arm + la from dgx-01 into --root")
    ap.add_argument("--selftest", action="store_true", help="run against an embedded fake fact sheet")
    args = ap.parse_args()

    if args.selftest:
        return run(SELFTEST_SHEET, "selftest")
    if not args.arm:
        ap.error("arm is required unless --selftest")
    if args.fetch:
        fetch(args.arm, args.root)
        fetch("la", args.root)

    sheet = fact_sheet(args.arm, collect(args.root, args.arm), collect(args.root, "la"))
    run(sheet, args.arm)
if __name__ == "__main__":
    sys.exit(main())
