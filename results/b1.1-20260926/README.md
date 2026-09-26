# b1.1 A/B, 2026-09-26

b1.1 = b1 + vLLM 6d232f16f (prefix-drop flags, vllm#887 padding fix, compile-worker cap). Arms, 2 boots each:
`b1run*` (b1, baseline), `b11pdrun*` (b1.1 with `VLLM_PREFIX_DROP_EXACT=1`, promoted), `b11run*` (b1.1 with the flag off).

- `ab/`: llama-benchy task grid (`task.csv`), counting sweeps (`count*.json`), TTFT probes per boot.
- `counting-c4/`: counting c4 check. The 3-round sweep showed c4 -6..-8% for prefix-drop; two more boots per arm with
  10-round sweeps and a paired temperature-0 probe on the counting prompt (`scripts/depth_decode_probe.py --prompt`)
  found no difference (c4 step time +0.4%, CI -0.3..+1.0; tokens/step +0.3%). Sweep noise, not a regression.
- `gate/`: hardmode, fidelity (8k-128k seed 7; 128k seeds 11, 13), stragglers, TC-45 x5.
- `verdict-b11-prefixdrop.json`: `scripts/arm_verdict.py` v2 fact sheet + Jev verdict (ship, 0.96). Counting c2/c4/c5
  are judged on 4 boots per arm (the A/B boots plus the `counting-c4/` boots as `*run3/4`).
