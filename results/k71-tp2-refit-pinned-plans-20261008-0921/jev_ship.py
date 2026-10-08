"""Jev ship/no-ship on k71 (refit run1 drafter on the 2x vs shipped b1.4, plans pinned). Gate enforced in code first."""
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
    "arm": "refit run1 MTP drafter (24 retrained mtp.* tensors, as the shipped 1x v3e) on the 2x b1.4, b12x plans pinned to the control's selections",
    "baseline": "shipped 2x b1.4 (TP=2 over two Sparks)",
    "design": "Thunderdome rules on the pair: boots ctl, arm, arm, ctl; T=0 probe cells + llama-benchy 4 runs; noise = control boot-to-boot (1% floor)",
    "quality_gate": {"pass": True, "hardmode": [92], "tc45": [100],
                      "fidelity": "20/20 exact at 8k/32k/64k and 128k x3 seeds", "stragglers_preemptions": 0},
    "acceptance_per_position_T0": {"pair": {"control": [0.819, 0.670, 0.546, 0.445], "arm": [0.855, 0.712, 0.592, 0.492]}},
    "cells_arm_vs_control": {
        "pair": {"fresh_c1": cell(3.41, 16.45), "fresh_c4": cell(10.89, 6.48), "fresh_c8": cell(6.21, 3.52),
                 "d16k_c4": cell(3.46, 3.75), "count_c8": cell(0.65, 7.92), "d16k_c8_wall": cell(1.72, 1.14),
                 "pp2048_c1": cell(3.53, 4.09), "tg512_c1": cell(1.12, 15.03), "tg512_c8": cell(7.15, 4.79)}},
    "thunderdome_verdict": {"pair": "PROMOTE"},
    "history": ("k70, the same arm without pinned plans, was KILL at c1: its image seed lacked 33 b12x GEMM plans, which the "
                "arm measured at boot and 8 got a different tile/split than the control. k71 pins them; the shipped image "
                "seed carries all 616 plans so a fresh install measures none."),
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
