# k94b recheck rule (written 2026-10-13 before the recheck boot)

k94 (one boot of the 2x v2.1.0 release image) had llama-benchy tg512 c4 at 148.2 tok/s, -8.4% against the k92 D3 arm
mean of 161.9 with a noise band of 8.16%, so its single-boot rule said FAIL. That band was calibrated for 4-boot means;
k92's own arm boot 1 had 146.8 (-9.3%), so the single-boot rule would fail the k92 arm itself. Every other k94 check
passed (warm boot, 0 plans measured, acceptance within 0.002 of k92, cached_tokens).

Recheck, decided before it ran, one attempt only:
- One more boot of the same branch recipe on the same runtime caches, the same llama-benchy arguments as k94
  (pp2048 + tg512, depth 0, c1/c4/c8/c16, 3 runs, T=1.0 top-p 0.95 top-k 20, prefix caching).
- PASS when, for every cell (pp2048 c1, tg512 c1/c4/c8/c16):
  (a) the mean of k94 and the recheck is not below the k92 D3 arm mean by more than that cell's k92 noise band, and
  (b) neither boot is below the lowest per-boot value of the 8 k92 boots (control and arm).
- FAIL: 2x v2.1.0 does not ship today; v2.0.0 stays the default.

k92 references: arm means pp2048 c1 2775.8, tg512 c1 83.8, c4 161.9, c8 190.9, c16 240.9; bands 2.73 / 11.94 / 8.16 /
5.15 / 7.48 %; lowest per-boot values of the 8 boots 2659.2 / 74.5 / 146.8 / 180.9 / 225.0.
