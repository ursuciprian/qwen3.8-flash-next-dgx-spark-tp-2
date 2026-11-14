# k120 decision rule (written 2026-11-14, before any boot)

Arm: 2x v2.2.0 image with b12x 35573bc4 (k107fix race fixes + CuTe DSL 4.7.0 + the HC down-projection BF16 path of
1x v2.4.0). Image ghcr.io/ursuciprian/spark-vllm-b12x:k120-20261114-35573bc4-d21d7ade-10d0acc8.
Control: 2x v2.2.0, ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm (digest
sha256:bba1374d), recipe ~/GEN-AI/backlog/k104s/recipe.yaml (code sha256 ef181a0e).
Same vLLM (10d0acc8), same CuTe DSL (4.7.0), same FlashInfer, same base image layers; only b12x differs.

Order: bake, then ctl arm arm ctl ctl arm arm ctl (4 boots per side), TP=2, every boot through `sparkrun run`.
Harness: tp2ab.sh + report2x.py (k107fix copies; c16 cell and prebake pong check added). Thresholds are not changed
after the first boot. A rerun needs a new id.

1. Gate before any boot: the arm image carries b12x 35573bc4, the four MoE fences, the GDN barrier, the BF16 path file
   and CuTe DSL 4.7.0; GPU tests (GDN decode, deferred checkpoints, three NVFP4 MoE suites, A16 dequant) show no test
   that passes on the control and fails on the arm. Else FAILED, no boot.
2. Quality gate: prebake boot answers the pong probe; identity under load (identc.py, 16 at once, 3 rounds per boot)
   must be OK; identity across boots (ident.py) PASS or UNSTABLE (UNSTABLE = control boots disagree with each other,
   not judged). MTP acceptance is part of rule 4.
3. Target cells (MAIN, the only cells that can make a WIN): cold TTFT 16K, 64K, 128K and llama-benchy pp2048 c1.
4. Nothing worse, every cell, c1..c16: cold TTFT 16K/64K/128K, pp2048 c1, tg512 c1/c4/c8/c16, paired fresh-c1. Worse =
   beyond the cell's noise band (TTFT: larger boot-to-boot spread of either side, floor 0.5%; benchy: max of the k92
   band and either side's spread; k92 bands pp2048 2.73%, tg512 c1 11.94%, c4 8.16%, c8 5.15%, c16 7.48%) or by the
   CI guard (whole boot-level Welch 95% CI beyond 1% in the bad direction). MTP per-position acceptance (fresh c8, T=0,
   pooled per side) must not move more than 0.03 at any position (numerics canary).
5. RULE2 (report2x.py): WIN = rule 4 holds and at least one MAIN cell is better beyond its noise. FLAT = rule 4 holds,
   no MAIN cell better. LOSS = rule 4 fails. EMPTY = a counted boot is missing cells, failed, booted cold, or fewer than
   2 boots per side.
6. VERDICT: SHIP-CANDIDATE = RULE2 WIN and IDENTC OK and IDENT PASS/UNSTABLE. FLAT = RULE2 FLAT with the same quality
   (the race fixes alone stay a ship candidate from k107fix). REJECT = LOSS or a quality failure. EMPTY = no verdict.
7. Expected size (k100 probe, 2026-10-15): about +4.7% on cold 2x prefill; 16K TTFT about -0.23 s of ~5.5 s. A WIN
   smaller than the noise of the earlier 2x runs (k101s, k107fix) is reported with its CI, not rounded up.

Nothing ships from this job. Results stay on dgx-01 until I publish them; no issue post from this job.
