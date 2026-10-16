k104s-2x-ship-check result: I built the 2x v2.2.0 release image and booted it the way a new user would, with a plain `sparkrun run` of the release recipe (2x v2.1.0 plus `VLLM_GDN_PREFILL_VIEWS=1` and a compile cache root of its own).

- Image `ghcr.io/ursuciprian/spark-vllm-b12x:k104s-20261016-21e0b201-d21d7ade-10d0acc8-warm`, digest `sha256:bba1374d1026bf3eba13d68102a0001d66e8b39c35c582937094359cc91ed2e2`. It is the image the A/B and the gate ran on, plus one layer with the compile cache of both ranks for the new GDN files. Every layer of the tested image is in it unchanged, and the vLLM and b12x code is byte-identical to the tested image. The second Spark pulled it from ghcr and got the same image.
- Before the boot I moved the plan file and the new compile cache root out of the runtime caches on both Sparks, so the image had to supply them. Up in 279 s, WARM on both ranks (compile cache loaded from the image, no compile), seed check PASS, plan records 682 / 682. Both ranks log the GDN views path.
- Pong and `/v1/models` answer; rank 0 serves `--revision 03f4a057`.
- T=0 fresh-c8 acceptance per draft position: 0.860 0.713 0.600 0.490 (A/B arm: 0.850 0.704 0.576 0.470).
- llama-benchy (T=1, 3 runs, boots 1) against the A/B arm means: pp2048 (c1) 2939.3 -> 2939.3 (+1.6% vs k101s arm 2892.4, band 4.2%); tg512 (c1) 83.3 -> 83.3 (-1.9% vs k101s arm 84.9, band 15.5%); tg512 (c8) 192.8 -> 192.8 (-1.3% vs k101s arm 195.2, band 11.7%).

Verdict: PASS. The quality gate is k104 (same image code, weights, plans and flag).

Results: {RESULTS_URL}
