# k120 decision rule (written 2026-11-14, before any boot; revised the same day after review, still before any boot)

Arm: 2x v2.2.0 image with b12x 35573bc4 (k107fix race fixes + CuTe DSL 4.7.0 + the HC down-projection BF16 path of
1x v2.4.0). Image ghcr.io/ursuciprian/spark-vllm-b12x:k120-20261114-35573bc4-d21d7ade-10d0acc8.
Control: 2x v2.2.0, ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm (digest
sha256:bba1374d), recipe ~/GEN-AI/backlog/k104s/recipe.yaml (code sha256 ef181a0e), untouched.
Same vLLM (10d0acc8), same CuTe DSL (4.7.0), same FlashInfer, same base image layers; only b12x differs (checked:
`pip list --format=freeze` of both images differs in the b12x line only, else the job does not start).
The arm recipe is the control recipe with its own name, image and runtime cache, and without the torch AOT seed line
(in the arm image /opt/b12x-seed holds the control's AOT artifact), so the arm compiles its own AOT in the bake boots.

Order: bake, then ctl arm arm ctl ctl arm arm ctl (4 boots per side), TP=2, every boot through `sparkrun run`.
Harness: tp2ab.sh + report2x.py + ident.py (k107fix copies; c16 cell, prebake pong, BF16 path marker, short-prompt
identity and the gate boot added). Thresholds are not changed after the first boot. A rerun needs a new id.

1. Gate before any boot: the arm image carries b12x 35573bc4, the four MoE fences, the GDN barrier, the BF16 path file
   and CuTe DSL 4.7.0, and differs from the control in the b12x distribution only; the test sources match 35573bc4;
   GPU tests (GDN decode, deferred checkpoints, three NVFP4 MoE suites, A16 dequant) show no test that passes on the
   control and fails, errors or does not run on the arm, and no suite that exits 0 on the control and crashes or times
   out on the arm. Else FAILED, no boot.
2. Path proof: the prebake boot runs the marker mod (mods/b12x-a16-marker, prebake only, never a measured boot). Both
   ranks must log a BF16-path call on a weight with N <= 512, K >= 8192 and >= 1264 rows during the cold 16K prompt.
   Else PATH=NOT-ENGAGED: no screen boot, VERDICT=EMPTY ("path not engaged"), never FLAT.
3. Quality (each one a quality failure when it fails):
   - pong: the prebake boot answers the pong probe;
   - IDENTC: identity under load (identc.py, 16 at once, 3 rounds per boot) OK;
   - IDENT: identity across boots (ident.py) PASS or UNSTABLE, judged on prompts under 1264 prompt tokens only. The
     BF16 path runs at 1264 rows or more and matches the A16 kernel up to the order of the FP32 additions, so longer
     prompts may legitimately differ; they are reported, not judged. UNSTABLE = no short prompt the control boots agree
     on (not judged);
   - MTP per-position acceptance (fresh c8, T=0, pooled per side) within 0.03 of the control at every position (rule 5);
   - GATE: the release quality gate of the v2.2.0 ship check (k104 rules, k61.gate) on one arm boot after the screen:
     stragglers c5-c16 0 preemptions, tool-use hardmode >= 88, TC-45 x5 = 100, fidelity 8k/32k/64k/128k + 128k seeds
     11/13 all 20/20. It runs only when RULE2 is WIN or FLAT and IDENT/IDENTC pass (otherwise the verdict does not
     depend on it).
4. Target cells (MAIN, the only cells that can make a WIN): cold TTFT 16K, 64K, 128K and llama-benchy pp2048 c1.
5. Nothing worse, every cell, c1..c16: cold TTFT 16K/64K/128K, pp2048 c1, tg512 c1/c4/c8/c16, paired fresh-c1. Worse =
   beyond the cell's noise band (TTFT: larger boot-to-boot spread of either side, floor 0.5%; benchy: max of the k92
   band and either side's spread; k92 bands pp2048 2.73%, tg512 c1 11.94%, c4 8.16%, c8 5.15%, c16 7.48%) or by the
   CI guard (whole boot-level Welch 95% CI beyond 1% in the bad direction). MTP per-position acceptance must not move
   more than 0.03 at any position (numerics canary).
6. RULE2 (report2x.py): WIN = rule 5 holds and at least one MAIN cell is better beyond its noise. FLAT = rule 5 holds,
   no MAIN cell better. LOSS = rule 5 fails. EMPTY = a counted boot is missing cells, failed, booted cold (a compile or
   plan measurement on either rank), shows the wrong image on either rank, or fewer than 2 boots per side.
7. VERDICT (report2x.py verdict, applied in this order):
   - REJECT = the prebake pong failed, or RULE2 LOSS, or a quality failure (IDENTC WORSE, IDENT FAIL, GATE FAIL);
     a LOSS or quality failure is REJECT even when other measurements are missing;
   - EMPTY = PATH not ENGAGED ("path not engaged"), or anything needed for a verdict is missing (RULE2 EMPTY/FAILED,
     IDENT/IDENTC EMPTY, GATE not run because its boot failed);
   - SHIP-CANDIDATE = RULE2 WIN with every quality check passing (pong, IDENTC OK, IDENT PASS/UNSTABLE, GATE PASS);
   - FLAT = RULE2 FLAT with every quality check passing (the race fixes alone stay a ship candidate from k107fix).
8. Expected size (k100 probe, 2026-10-15): about +4.7% on cold 2x prefill; 16K TTFT about -0.23 s of ~5.5 s. A WIN
   smaller than the noise of the earlier 2x runs (k101s, k107fix) is reported with its CI, not rounded up.

Nothing ships from this job. Results stay on dgx-01 until I publish them; no issue post from this job.
