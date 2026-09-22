# mod vllm-qwen38-bf16-gemv — A/B verdict (2026-09-21)

> **Workload.** Unless a line says otherwise, every tok/s, c1/c8/c16 and
> ms/step figure in this file is the `tools/tony-bench/bench_sweep.py`
> counting diagnostic: "List the numbers from 1 to 300 separated by commas…",
> temperature 0, thinking off, non-streaming, 320 max tokens, fresh context,
> aggregate tok/s (c1 = per-stream). MTP accepts ~4 of 4 drafts on it, so it is
> a speculative-decoding ceiling, not coding or chat speed. Agent-coding and
> prose numbers: top-level `README.md`.

**Verdict: REVERT / do not ship.** `vllm-qwen38-bf16-gemv` fails closed at boot
(EngineCore crash during b12x weights-preparation), so no c1/c8/logits/gate
comparison against B was possible. la was restored and reconfirmed healthy.

## A. Baseline (`eugr-agents-serve-local16-la.yaml`, image `local-20260918-a8333658`)

Logits capture: `results/arms/gemv/base_logits.json` (20/20 prompts captured OK).

c1 sweeps (`bench_sweep.py --levels 1 --rounds 3`), tok/s:

| run | tok/s |
|---|---|
| A1 | 99.5 |
| A2 | 89.7 |
| A3 | 98.8 |
| A4 | 97.5 |
| A5 | 85.2 |
| A6 | 93.6 |

**median = 95.55 tok/s, max = 99.5 tok/s** (bimodal fast/slow per
`results/arms/c1-decline/verdict.md`, consistent with that baseline
characterization — no new instability introduced by this test).

c8 sweep: agg 433.0 tok/s, per-stream 55.2 tok/s, w2w med 5.80s p90 6.03s
(`results/arms/gemv/A_c8.json`).

## B. la-gemv (`eugr-agents-serve-local16-la-gemv.yaml` = la + mods:
[b12x-startup-boundedwait, vllm-qwen38-bf16-gemv])

**BOOT FAILED.** Job `66796e61d38e68a7_c9b8459a498b`, both containers came up
(`Up 3 minutes`, i.e. process-level "up") but the vLLM EngineCore crashed
during b12x weight/kernel preparation before health ever returned 200 or 000
turned to 200:

Boot log `/tmp/la_gemv_boot.log`:

```
250:b12x failed gemm.bf16_gemv: 99/367 ready, candidates 2/3 prepared, rank 0 batch 1, 0 measured, 272 cached, 2 compilations, 0:04
251:(EngineCore pid=336) ERROR 09-20 21:43:10 [core.py:1484] EngineCore failed to start.
252:(EngineCore pid=336) ERROR 09-20 21:43:10 [core.py:1484] Traceback (most recent call last):
...
(EngineCore pid=336) ERROR 09-20 21:43:10 [core.py:1484]     self._run_b12x_preparation(stage="weights")
(EngineCore pid=336) ERROR 09-20 21:43:10 [core.py:1484]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/abstract.py", line 322, in _run_b12x_preparation
(EngineCore pid=336) ERROR 09-20 21:43:10 [core.py:1484]     raise RuntimeError(
(EngineCore pid=336) ERROR 09-20 21:43:10 [core.py:1484] RuntimeError: b12x preparation failed on rank 0: ValueError: candidate races require an activation-producing context
```

Decisive line: **`/tmp/la_gemv_boot.log:252`ff, root cause
`ValueError: candidate races require an activation-producing context`**
raised out of `vllm/v1/executor/abstract.py:322 _run_b12x_preparation`, while
b12x was compiling/racing candidates for `gemm.bf16_gemv` (`99/367 ready,
candidates 2/3 prepared, ... 2 compilations, 0:04`). The plan cache (272
cached entries, mounted from the persistent runtime-cache volume) loaded fine
-- the failure is specific to the new `gemm.bf16_gemv` target the mod adds,
not a cache-mount regression.

No c1/c8/logits/gate measurements were taken for B: the server never reached
`/health` 200.

## Restore (step D)

- `sparkrun stop 66796e61d38e68a7_c9b8459a498b` -- stopped on both hosts.
- Verified both nodes had zero sparkrun/vllm containers before rebooting la.
- `nohup sparkrun run recipes/eugr/eugr-agents-serve-local16-la.yaml` --
  job `66796e61d38e68a7_87b740b8a744`.
- Health 200 confirmed; boot log shows `272 cached` during b12x planning
  (`b12x selecting gemm.blockscaled_precision: 14/349 ready, 0 measured, 272
  cached, ...`), CUDA graph capture completed (23/23 PIECEWISE), multi-modal
  warmup completed, `Application` serving.
- Post-restore c1 sanity sweeps (3x): 97.3, 96.8, 87.2 tok/s -- consistent
  with the known bimodal baseline, confirms la is back to its normal
  performance envelope.

**Final serving state: la (baseline, no gemv mod) is up and healthy on both
nodes.**

## Recommendation

Do not promote `vllm-qwen38-bf16-gemv` as-is. The b12x candidate-racing
preparation path for `gemm.bf16_gemv` needs a fix (the "activation-producing
context" requirement suggests the plan/race harness expects an activation
tensor context that this mod's static-shape M=1..8 targets don't supply -- likely
a b12x-side or mod-side call-order issue in how candidates are raced for this
op). Re-test only after that is fixed; re-run the full validation recipe from
`results/kernel-pass/dense-gemm-fusion.md` section 6 from scratch (this mod
touches model code, so no partial re-use of these results once the fix
changes the patch).
