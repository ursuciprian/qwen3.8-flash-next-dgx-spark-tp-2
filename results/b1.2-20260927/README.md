# b1.2 A/B, 2026-09-27

b1.2 = b1.1 + vLLM a9aa81b23 (HC/hyper-connection mixers in online MXFP8, router gate stays BF16;
`VLLM_QWEN38_HC_MXFP8=hc`). b12x b7fbaf96, unchanged from b1.1.

- `r3ab-hcq-20260927/`: A/B driver output. `base1`/`base2` = b1.1 (baseline), `cand1`/`cand2` = hcq (candidate).
  Each boot dir has the coding task grid (`task.csv`), counting sweeps (`count*.json`, 5x `count-c1-*.json` c1
  repeats), TTFT/decode-step probes (`probe-*.json`/`.log`), and (candidate boots only) the gate artifacts
  (`hardmode.log`, `fidelity_probe.txt`, `fidelity-seed11/13.*`, `straggler.log`, `tc45.log`).
- `c1c4-profile-20260926/`: the decode-step breakdown (`scripts/step_breakdown.py`) that found the BF16 HC
  mixers at 13-19% of the c1/c4 step and motivated this arm; see its own `README.md`.
- `b12-hcq.json`: `scripts/arm_verdict.py` v2 fact sheet + Jev verdict, **ship** (confidence 0.97). Quality gate
  pass (hardmode 90/92, fidelity 20/20 at every depth, no straggler violations). Wins beyond noise: coding d0
  c1/c2/c4, coding 16k c1/c2/c4, counting c1/c2/c5/c8/c10; no cell worse beyond noise.
