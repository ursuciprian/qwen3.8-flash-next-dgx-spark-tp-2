# Arm la-mtpprob: draft_sample_method=probabilistic -- verdict (2026-09-22, corrected)

> **Workload.** Unless a line says otherwise, every tok/s, c1/c8/c16 and
> ms/step figure in this file is the `tools/tony-bench/bench_sweep.py`
> counting diagnostic: "List the numbers from 1 to 300 separated by commas…",
> temperature 0, thinking off, non-streaming, 320 max tokens, fresh context,
> aggregate tok/s (c1 = per-stream). MTP accepts ~4 of 4 drafts on it, so it is
> a speculative-decoding ceiling, not coding or chat speed. Agent-coding and
> prose numbers: top-level `README.md`.
>
> **Status update 2026-09-22:** promoted to the default `la` recipe by PR #25
> on the default-temperature prose grid (llama-benchy 0.4.0 book, pp 2048,
> tg 128, temp 1.0: c1 46.0 -> 61.2, c4 78.9 -> 95.3 aggregate; dgx-01
> `results/benchy/la-prose.md` vs `la-mtpprob-prose.md`). The full
> c1-c16 x depth grids were measured afterwards
> (`results/benchy/la-mtpprob-{task16,prose16}.md` vs `la-{task16,prose16}.md`,
> summarized in the top-level README). The "Not promoted" text below is the
> state at the time of writing. hardmode here is tool-eval-bench `--hardmode`,
> thinking on.

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
