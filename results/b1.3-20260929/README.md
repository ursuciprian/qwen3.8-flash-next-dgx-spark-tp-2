# b1.3 (2026-09-29): b1.2 + GDN uniform-decode metadata skip

- Candidate: `spark-vllm-b12x:b13-b7fbaf96-7344a997` (vLLM `shipped-b1.3-20260929` = `7344a9976`, b12x `b7fbaf96`),
  recipe = the b1.2 recipe + `VLLM_GDN_UNIFORM_DECODE_META_SKIP=1`. Shipped as the warm image
  `ghcr.io/ursuciprian/spark-vllm-b12x:b1.3-20260929-b7fbaf96-7344a997-warm`
  (`sha256:32cb8bd8800e413726b4cfe3d9947f80d4eb01d92dbe12405e3c763db7306a02`).
- A/B `r4ab-b13-20260929/`: base1, cand1 (gate + TC-45), cand2 (hardmode + 128k seeds 11/13), base2, same day.
  Verdict `b13.json` (arm_verdict v2 + Jev: ship_with_caveat 1.00); paired temp-0 probes `paired-report.txt`;
  logits `logits-cmp.txt` (cross-build within self-noise).
- The same change was first measured as a mod on the b1.2 image on 2026-09-28 (`b13-gdnmeta-mod-20260928.json`,
  ship 0.99); a single validation boot of the baked build on 2026-09-29 sent it to this same-day A/B.
