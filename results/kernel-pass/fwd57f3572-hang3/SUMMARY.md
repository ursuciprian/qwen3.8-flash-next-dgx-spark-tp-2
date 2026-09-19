# fwd57f3572 hang evidence (hang #3) — 2026-09-19

Boot: `eugr-agents-serve-local16-la-fwd57f3572.yaml` = OLD known-good image
(`spark-vllm-b12x:local-20260918-a8333658`, b12x @ a8333658, the exact
proven-working preparation-protocol base) + `mods/b12x-fwd-57f3572`
(forward-ports only the narrowly-scoped MoE fix 57f3572, verified
byte-identical pre-images, no touch to the distributed rank-coordination
code), `B12X_AUTOTUNE: "1"`, empty plan cache (contract bump 4->5 forces a
retune).

## Result: hangs identically to hang #2, at the same kind of barrier

`nvidia-smi`: 0% util / ~9.3-9.7W on both GPUs (both nodes), confirmed by
two independent monitors (mine and the coordinator's watchdog).

Last rank0 ticker lines before freeze:
```
b12x compiling norm.hyperconnection: 2/349 ready, candidates 0/1 prepared, 2988 measured, 0 cached, 177 compilations, 0:56
b12x preparing candidates gemm.bf16_vocab_projection: 3/349 ready, candidates 1/11 prepared, rank 0 batch 1, 3288 measured, 0 cached, 194 compilations, 1:00
b12x autotuning gemm.bf16_vocab_projection: 3/349 ready, candidates 11/11 prepared, rank 0 batch 1, round 2/3, 3288 measured, 0 cached, 194 compilations, 1:00
b12x priming gemm.bf16_vocab_projection: 45/349 ready, candidates 0/1 prepared, 3299 measured, 0 cached, 194 compilations, 1:00
(EngineCore pid=282) INFO ... shm_broadcast.py:801 No available shared memory broadcast block found in 60 seconds.
```
Then silence for 20+ min. `/proc/<pid>/wchan` for both ranks'
`Worker_TP*` process: **both** `poll_schedule_timeout.constprop.0` --
symmetric mutual block, identical signature to hang #2. rank1's own server
log again stops right after model loading, never printing a single
`b12x compiling/priming/autotuning` line.

## Revised conclusion (supersedes the hang #2 writeup's framing)

This is the same "rank 0 batch 1" multi-batch candidate-racing barrier seen
in hang #2 -- but this time it appears on the **OLD a8333658 image**, whose
distributed-preparation protocol was otherwise proven working (`la`,
`spec3`, `fusear` all booted fine on it earlier this session). That rules
out "b12x HEAD's rewritten protocol vs. an old vllm fork" as the root
cause, and points at something narrower: **the batched candidate-racing
code in `b12x/preparation/session.py` -- present in a8333658 already --
only exercises its multi-batch path (`rank N batch M`) when a retune has
enough candidates to need more than one batch round**, which never happens
under the *default* moe.decode candidate contract (a8333658's normal boots
never produced more than ~222 total candidates and never showed a "batch"
label at all). Both times we forced a large retune (837 candidates via
b12x HEAD's contract bump, or 349 candidates via 57f3572's contract bump
forward-ported onto the old image), the multi-batch path activated and
deadlocked identically, regardless of which b12x commit or vllm fork
combination carried it.

**Practical conclusion: any moe.decode-candidate-contract bump (57f3572
and later) that meaningfully grows the retune size beyond a8333658's
original ~222-candidate scale hits this deadlock on TP=2 with the pinned
vllm fork (local-inference-lab/vllm@dev/jovian-judgement, 8e1f1e58) --
independent of which specific downstream b12x commit introduces the bump.
This is not fixable by reverting/forward-porting individual b12x commits;
it needs either (a) local-inference-lab/vllm's dev/jovian-judgement branch
to ship coordinator-side support for whatever batching contract b12x
expects at that candidate scale, or (b) a b12x-side fix to the
`rank N batch M` handshake itself (out of scope for a fork-side mod). Not
pursued further this session per explicit instruction to stop chasing.**

## Files in this directory
- `rank0-boot-log-tail150.log` -- sparkrun launcher stdout tail.
- `rank0-sparkrun_serve.log`, `rank1-sparkrun_serve.log` -- per-rank vLLM
  server logs from inside each container.
- `rank0-wchan.txt`, `rank1-wchan.txt` -- `/proc/<pid>/wchan` for each
  rank's Worker_TP process at hang time.
