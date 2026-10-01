# b1.4 (2026-10-01): b1.3 + 131k-id MTP draft vocab + vllm#923/#914 fixes

- Candidate: `spark-vllm-b12x:b14-b7fbaf96-a7e649d8` (vLLM `shipped-b1.4-20261001` = `a7e649d8`, b12x `b7fbaf96`),
  recipe = the b1.3 recipe + `VLLM_MTP_DRAFT_VOCAB` (the r7 v2 131,072-id list, `archive/mods/r7-dvocab/`).
  Shipped as the warm image `ghcr.io/ursuciprian/spark-vllm-b12x:b1.4-20261001-b7fbaf96-a7e649d8-warm`
  (`sha256:@@DIGEST@@`), which also carries the id list at `/opt/mtp-vocab`, plus the recipe flag
  `--default-chat-template-kwargs '{"reasoning_effort":"medium"}'`.
- A/B `r4ab-b14-20261001/`: base1, cand1 (gate + TC-45), cand2 (hardmode + 128k seeds 11/13), base2, same day.
  Paired temp-0 probes `paired-report.txt`; logits `logits-cmp.txt` (cross-build within self-noise).
- Gate: the A/B gate boot had 32k fidelity 19/20 (one `no_call`), which forced a reject. `post-b14-20261001/`: 20
  fresh-prefix cold-32k trials per build, b1.3 20/20, b1.4 20/20. `regate-b14-20261001/`: a fresh b1.4 boot, gate
  seed 7 20/20 at 8k/32k/64k/128k and 32k seeds 21/22 20/20 each (`summary.txt`).
- Verdict `b14.json`: the A/B fact sheet with the re-gate as `quality_gate` (the original gate is kept as
  `quality_gate_original`, the forced reject as `superseded_verdict`); Jev ship 0.78. `effort_default`: Jev
  medium_default 1.00 on the DevOps evidence below.
- `devops-b14-medium.md`: DevOps 14 x 3 on the b1.4 candidate at TP=2 with `reasoning_effort=medium`: mean checks 95.9%,
  29/42 clean, 0/42 runaway thinking, median 33 s per task (b1.2 at xhigh: 42.5%, 17/42, 23/42, 246 s).
