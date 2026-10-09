"""Jev ship/no-ship on k73 arm c41 (GDN-MSE @ 16c9bd54 + M-dispatch at TP=2) vs shipped 2x v3.1.0 (b1.6). Gate enforced in code first."""
import json, subprocess, sys, time
from typesafe_sdk import Choice, TypeSafeClient

RULES = (
    "Quality gate (non-negotiable, checked in code): hardmode >= 88, TC-45 100, fidelity 20/20 exact at "
    "8k/32k/64k/128k, stragglers c8-c16 with 0 preemptions. Speed: ship if the arm is faster beyond the cell's "
    "noise band in at least one cell and no c1..c16 cell is worse beyond noise; a beyond-noise loss "
    "anywhere caps the verdict at ship_with_caveat at best, a low-concurrency (c1-c4) loss that is not tiny means reject. "
    "A cell whose difference is inside its noise band is neither a win nor a loss. The arm changes the checkpoint "
    "(GDN projections requantized to weight-only NVFP4, served below 41 rows; MXFP8 of the base at 41+ rows), so "
    "acceptance per draft position must stay within 0.03 of the control."
)
cell = lambda d, n: {"pct": d, "noise_pct": n, "beyond_noise": abs(d) > n}
facts = {
    "arm": "GDN-MSE checkpoint @ 16c9bd54 (drafter D1 included) + M-dispatch cutoff 41 (NVFP4 GDN weights below 41 rows, per-rank MXFP8 copies of 7c4f1bc1 at 41+), vLLM d21d7ade, b12x 21e0b201, plans pinned to the control's selections for every shared key",
    "baseline": "shipped 2x v3.1.0 (old name b1.6), TP=2 over two Sparks",
    "design": "Thunderdome rules on the pair: boots ctl, c26, c41, c41, c26, ctl; T=0 probe cells + llama-benchy 4 runs; noise = max(1%, control boot-to-boot, CI half-width); 0 plans measured on every counted boot",
    "quality_gate": {"pass": True, "hardmode": [90], "tc45": [100],
                     "fidelity": "20/20 exact at 8k/32k/64k and 128k x3 seeds", "stragglers_preemptions": 0,
                     "min_mem_available_gib": 9.48},
    "acceptance_per_position_T0": {"pair": {"control": [0.853, 0.709, 0.589, 0.492], "arm": [0.852, 0.710, 0.587, 0.485]}},
    "cells_arm_vs_control": {
        "pair": {"fresh_c1": cell(18.95, 11.58), "fresh_c4": cell(3.22, 4.69), "fresh_c8": cell(8.60, 10.35),
                 "d16k_c4": cell(4.83, 5.47), "count_c8": cell(11.98, 17.74), "d16k_c8_wall": cell(4.60, 7.45),
                 "pp2048_c1": cell(0.82, 1.55), "tg512_c1": cell(8.38, 12.34), "tg512_c8": cell(8.41, 11.26)}},
    "decode_step_change_pct": {"fresh_c1": -6.72, "fresh_c4": -3.96, "fresh_c8": -3.14, "d16k_c4": -5.34, "count_c8": -10.15},
    "thunderdome_verdict": {"pair": "PROMOTE"},
    "coverage_note": "No c16 A/B cell was measured. At 9-16 requests a decode step has 45-80 rows and runs the MXFP8 GDN path, the same weights as the control, on newer vLLM/b12x; the gate's straggler probe at c16 was clean.",
    "kv_pool_tokens": {"control": [3755980, 3658739], "arm": [3525078, 3522120]},
    "history": "b1.5 (GDN-MSE at TP=2 without the dispatch) failed 128k fidelity (18/20, 20/20, 19/20). Cutoff 26 also PROMOTE (fresh c1 +11.7%, d16k c4 +7.9%). Compact records on top of c41: INCONCLUSIVE, not shipped.",
}
losses = [f"{s}:{k}" for s, c in facts["cells_arm_vs_control"].items() for k, v in c.items() if v["beyond_noise"] and v["pct"] < 0]
if not facts["quality_gate"]["pass"]:
    verdict, conf, note, probs = "reject", 1.0, "quality gate failed (forced, no Jev call)", {}
else:
    key = subprocess.run(["security", "find-generic-password", "-s", "dev/typesafe-ai-api-key", "-w"],
                         capture_output=True, text=True, check=True).stdout.strip()
    with TypeSafeClient(api_key=key) as client:
        r = client.system_one({"rules": RULES, "facts": facts}, {"verdict": Choice(
            instructions=("Given `rules` and `facts` (candidate `facts.arm` vs `facts.baseline` on two Sparks; "
                          "`facts.quality_gate.pass` is computed and non-negotiable; each cell carries pct, noise_pct "
                          "and beyond_noise), should this build replace the shipped default?"),
            criteria={"ship": "Gate passes, at least one beyond-noise win, no beyond-noise loss.",
                      "ship_with_caveat": "Gate passes and there is a win, but there is a beyond-noise loss at c5+ or the picture is borderline.",
                      "reject": "Gate fails, or no beyond-noise win, or a non-trivial beyond-noise loss at c1-c4.",
                      "rerun": "The evidence is too thin or too noisy to judge."})})
    a = r.choices["verdict"]; verdict, conf, note, probs = a.choice, a.confidence, None, dict(a.probabilities)
    if losses and verdict == "ship":
        verdict, note = "ship_with_caveat", f"capped from ship: losses {losses}"
out = {**facts, "rules": RULES, "losses_beyond_noise": losses, "verdict": verdict, "confidence": conf, "probabilities": probs,
       "cap_note": note, "judge": "Jev (TypeSafe System One), Choice", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(verdict, round(conf, 3), {k: round(v, 3) for k, v in probs.items()}, note)
