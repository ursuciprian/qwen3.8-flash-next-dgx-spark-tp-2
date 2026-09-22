# Arm la-mtpprob: draft_sample_method=probabilistic -- verdict (2026-09-22, corrected)

**Verdict: SHIP-with-caveat.**

## Gate results
- Initial gate: hardmode 86/100, fidelity pass on rerun, straggler clean.
- Initial suspected regression: bench_sweep c16 = 399.5 (single sweep) vs la reference c16 = 645.
- This pass (rerun, more sweeps):
  - hardmode2 (tool-eval-bench, full scenarios, thinking on): **Quality 90/100**, Deployability 77/100 (alpha=0.7).
  - decode_c16 rerun x2 (bench_sweep --levels 16 --rounds 3): **641.3 tok/s** (run1), **635.0 tok/s** (run2).
  - decode_c12 x1: 545.3 tok/s agg, 46.6 tok/s/stream.

## Threshold check
Rule: SHIP-with-caveat only if hardmode2 >= 88 AND c16 >= 600; else REJECT.
- hardmode2 = 90 >= 88: PASS
- c16 run1 = 641.3 >= 600: PASS
- c16 run2 = 635.0 >= 600: PASS

Both thresholds met on rerun. The single sweep=399.5 that triggered the initial
"suspected regression" note appears to have been a one-off straggler/noise
sample (bimodal decode behavior is a known characteristic of this stack, see
results/arms/c1-decline/verdict.md) -- not reproduced across 2 reruns of 3
rounds each (641.3, 635.0), both close to the la reference (645).

## Caveat
Only 2 c16 reruns + 1 c12 in this pass; recommend one more spot-check before
treating draft_sample_method=probabilistic as fully validated at c16, given
the single low outlier seen earlier in this arms cycle. Not promoted to the
default la recipe -- la (use_local_argmax_reduction) remains the shipped
baseline; la-mtpprob is SHIP-with-caveat as an alternative worth another look,
not a replacement.

(Note: an earlier automated pass mis-parsed decode_c16 JSON output -- read
top-level "agg_tok_s" instead of rows[0].agg_tok_s -- and wrote an erroneous
REJECT verdict with the values missing. This file supersedes that with the
correct parsed values.)
