k94-2x-ship-check result: I checked the 2x v2.1.0 candidate (v2.0.0 with drafter D3 at checkpoint revision `03f4a057`, plus `--enable-prompt-tokens-details`) on the pair with a plain `sparkrun run` of the recipe. Before the boot I moved the D3 plan file and both AOT keys out of the runtime caches on both Sparks, so the image had to supply them, as on a new install.

- Image `ghcr.io/ursuciprian/spark-vllm-b12x:k94-20261013-21e0b201-d21d7ade-warm` (also tagged `2x-v2.1.0`): up in 348 s, WARM on both ranks; seed check PASS; plan records 682 / 682.
- Rank 0 serves `--revision 03f4a0570496bbe741189f4a2a7d611679e85406` with the new flag; on the second of two identical prompts the usage field reads prompt_tokens=14239 cached_tokens=11456.
- T=0 fresh-c8 acceptance per draft position: 0.851 0.701 0.574 0.471 (k92 D3 arm: 0.853 0.702 0.576 0.469).
- llama-benchy (T=1, 3 runs) against the k92 D3 arm means: pp2048 (c1) 2847.7 (+2.6% vs k92 D3 arm 2775.8, band 2.73%); tg512 (c1) 89.5 (+6.8% vs k92 D3 arm 83.8, band 11.94%); tg512 (c4) 148.2 (-8.4% vs k92 D3 arm 161.9, band 8.16%); tg512 (c8) 187.2 (-1.9% vs k92 D3 arm 190.9, band 5.15%); tg512 (c16) 234.3 (-2.7% vs k92 D3 arm 240.9, band 7.48%).

Verdict: FAIL. Reasons: BAD (tg512 (c4)); The quality gate is the k92 gate (same image, weights and plan selections; the new flag only adds a usage field).

Results: {RESULTS_URL}
