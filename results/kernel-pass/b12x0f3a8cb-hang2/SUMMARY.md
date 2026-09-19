# b12x0f3a8cb-rev06809d5 hang evidence (hang #2) — 2026-09-19

Boot: `eugr-agents-serve-local16-la-b12x0f3a8cb-rev06809d5.yaml` (same image
as hang #1, plus `mods/b12x-revert-06809d5` reverting 06809d5's Python
changes, empty plan cache, `B12X_AUTOTUNE: "1"`).

## Progress made, then hung at a later, different barrier

The revert DID fix the first barrier: preparation advanced past the
"waiting for ranks: 0/837 ready, 0 measured, 0 cached, 0 compilations"
freeze into real work -- `b12x compiling sequence.mtp_feedback: 47/349
ready, candidates 0/8 prepared, rank 0 batch 1, 3299 measured, 0 cached,
194 compilations, 1:03` -- counters climbing, real progress, unlike hang #1
where every counter stayed at zero forever.

That line then froze too: log mtime stopped advancing, and after a ~13 min
silence:
- `nvidia-smi`: 0% util / ~9.3-9.6W on both GPUs (both nodes) -- idle again.
- rank0 and rank1 `Worker_TP` process wchan: **both** `poll_schedule_timeout.constprop.0`
  this time (symmetric, unlike hang #1's asymmetric spin/wait_woken) --
  both sides now blocked in a poll-with-timeout wait, consistent with both
  waiting on each other/a shared exchange that never completes.
- rank1's server log again never printed anything past initial model
  loading (rope-parameter warnings) -- it never reached any
  `b12x compiling`/priming line at all, same pattern as hang #1.

## Interpretation

The label `rank 0 batch 1` marks a distributed-compile-batch boundary
(`sequence.mtp_feedback`, one of the 837 candidate groups), a different
code path/barrier than the cache-reconciliation one 06809d5 introduced --
reverting 06809d5 fixed *that* specific handshake but exposed (or the retune
simply reached) a second, structurally similar coordination point that
still assumes fork-side (vLLM) support the pinned 8e1f1e58 fork lacks. Same
underlying disease as hang #1 (b12x HEAD's massively expanded/restructured
preparation-session protocol at 837 candidates vs. 222 assumes a newer
coordinator contract than our pinned vllm fork implements), different
symptom location. **One commit revert is not sufficient** to make b12x HEAD
work against this vllm fork tip.

## Decision

Per the coordinator: abandon "make b12x HEAD work by reverting individual
preparation-protocol commits" (unbounded scope, second hang already proves
it's not a single fix) and flip strategy to the reverse direction --
keep the known-good OLD image (`spark-vllm-b12x:local-20260918-a8333658`,
which already works fine with the 8e1f1e58 fork) and forward-port only the
narrowly-scoped MoE-kernel commit(s) we actually want (57f3572, and
a2b5152/8783519 only if required and safe) as a mod onto that old, proven-
compatible preparation-protocol base.

## Files in this directory
- `rank0-boot-log-tail150.log` -- sparkrun launcher stdout tail (head node).
- `rank0-sparkrun_serve.log`, `rank1-sparkrun_serve.log` -- per-rank vLLM
  server logs from inside each container.
- `rank0-wchan-shm.txt`, `rank1-wchan-shm.txt` -- `/proc/<pid>/wchan` and
  `/dev/shm` listing on each node at hang time.
