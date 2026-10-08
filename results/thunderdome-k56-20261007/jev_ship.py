"""Jev ship/no-ship on k56 (refit run1 drafter vs shipped v3d). Gate enforced in code first."""
import json, subprocess, sys, time
from typesafe_sdk import Choice, TypeSafeClient

RULES = (
    "Quality gate (non-negotiable, checked in code): hardmode >= 88, TC-45 100, fidelity 20/20 exact at "
    "8k/32k/64k/128k, stragglers c8-c16 with 0 preemptions. Speed: ship if the arm is faster beyond the cell's "
    "noise band in at least one cell and no cell is worse beyond noise on either Spark; a beyond-noise loss "
    "anywhere caps the verdict at ship_with_caveat at best, a low-concurrency (c1-c4) loss that is not tiny means reject. "
    "A cell whose difference is inside its noise band is neither a win nor a loss. The arm only changes the MTP "
    "drafter weights, so an acceptance rise per draft position is the intended effect, not a risk."
)
cell = lambda d, n: {"pct": d, "noise_pct": n, "beyond_noise": abs(d) > n}
facts = {
    "arm": "k56 refit run1 drafter (72 steps on 3.5M tokens of own v3d outputs, fp8 drafter KV)",
    "baseline": "shipped single-Spark v3d",
    "design": "Thunderdome: per Spark, 2 passes ABBA, control and arm boots, T=0 probe cells + llama-benchy",
    "quality_gate": {"pass": True, "hardmode": [91, 91], "tc45": [100, 100],
                      "fidelity": "20/20 at 8k/32k/64k/128k on both; 128k extra seeds: dgx-01 one seed 19/20 (119/120 overall), dgx-02 120/120",
                      "stragglers_preemptions": 0},
    "acceptance_per_position_T0": {
        "dgx-01": {"control": [0.819, 0.661, 0.535, 0.431], "arm": [0.856, 0.713, 0.593, 0.489]},
        "dgx-02": {"control": [0.818, 0.666, 0.539, 0.438], "arm": [0.854, 0.708, 0.589, 0.488]}},
    "cells_arm_vs_control": {
        "dgx-01": {"fresh_c1": cell(5.66, 4.73), "fresh_c4": cell(4.88, 3.05), "fresh_c8": cell(6.79, 3.56),
                   "d16k_c4": cell(5.47, 3.65), "count_c8": cell(1.19, 1.00), "d16k_c8_wall": cell(2.86, 1.00),
                   "pp2048_c1": cell(0.01, 1.00), "tg512_c1": cell(4.31, 7.55), "tg512_c8": cell(3.68, 4.85)},
        "dgx-02": {"fresh_c1": cell(2.84, 8.54), "fresh_c4": cell(7.90, 2.82), "fresh_c8": cell(8.08, 2.61),
                   "d16k_c4": cell(5.74, 3.23), "count_c8": cell(-0.11, 2.77), "d16k_c8_wall": cell(2.04, 1.00),
                   "pp2048_c1": cell(-0.89, 2.02), "tg512_c1": cell(1.60, 10.26), "tg512_c8": cell(1.20, 3.58)}},
    "thunderdome_verdict": {"dgx-01": "PROMOTE", "dgx-02": "PROMOTE"},
}
losses = [f"{s}:{k}" for s, c in facts["cells_arm_vs_control"].items() for k, v in c.items() if v["beyond_noise"] and v["pct"] < 0]
if not facts["quality_gate"]["pass"]:
    verdict, conf, note = "reject", 1.0, "quality gate failed (forced, no Jev call)"
else:
    key = subprocess.run(["security", "find-generic-password", "-s", "dev/typesafe-ai-api-key", "-w"],
                         capture_output=True, text=True, check=True).stdout.strip()
    with TypeSafeClient(api_key=key) as client:
        r = client.system_one({"rules": RULES, "facts": facts}, {"verdict": Choice(
            instructions=("Given `rules` and `facts` (candidate `facts.arm` vs `facts.baseline` on two Sparks; "
                          "`facts.quality_gate.pass` is computed and non-negotiable; each cell carries pct, noise_pct "
                          "and beyond_noise), should this drafter replace the shipped one?"),
            criteria={"ship": "Gate passes, at least one beyond-noise win, no beyond-noise loss on either Spark.",
                      "ship_with_caveat": "Gate passes and there is a win, but there is a beyond-noise loss at c5+ or the picture is borderline.",
                      "reject": "Gate fails, or no beyond-noise win, or a non-trivial beyond-noise loss at c1-c4.",
                      "rerun": "The evidence is too thin or too noisy to judge."})})
    a = r.choices["verdict"]; verdict, conf, note = a.choice, a.confidence, None
    if losses and verdict == "ship":
        verdict, note = "ship_with_caveat", f"capped from ship: losses {losses}"
out = {**facts, "rules": RULES, "losses_beyond_noise": losses, "verdict": verdict, "confidence": conf,
       "cap_note": note, "judge": "Jev (TypeSafe System One), Choice", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(verdict, round(conf, 3), note)
