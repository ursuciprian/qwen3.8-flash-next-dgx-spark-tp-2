# b12x TP=2 preparation deadlock — mechanism (2026-09-19)

> **Workload.** Unless a line says otherwise, every tok/s, c1/c8/c16 and
> ms/step figure in this file is the `tools/tony-bench/bench_sweep.py`
> counting diagnostic: "List the numbers from 1 to 300 separated by commas…",
> temperature 0, thinking off, non-streaming, 320 max tokens, fresh context,
> aggregate tok/s (c1 = per-stream). MTP accepts ~4 of 4 drafts on it, so it is
> a speculative-decoding ceiling, not coding or chat speed. Agent-coding and
> prose numbers: top-level `README.md`.

Traced against b12x @ a8333658 (`b12x/preparation/session.py`, 1657 lines — the
version installed in the serving image) and the vLLM fork files that drive it
(`vllm/model_executor/warmup/b12x_prepare.py`, `vllm/v1/worker/b12x_startup.py`).
This supersedes the framing in `results/kernel-pass/*/SUMMARY.md` ("needs
coordinator-side support, root cause unlocated") with a concrete mechanism.

## The two protocols, and which one is broken

There are **two separate cross-rank mechanisms**, easy to conflate:

1. **Local candidate racing** (`session.py` `_race`, line 1097 area,
   `local_candidates = indexed_candidates[rank_index::rank_count]` at
   line 1153-1154). Each rank strides the candidate list and benchmarks its
   own disjoint shard, entirely locally — no `torch.distributed` calls
   anywhere in `session.py` (confirmed: no `import torch.distributed` in the
   file; `self.session._synchronize()` at line 315 is only
   `torch.cuda.synchronize(ordinal)`, a local op). `self.race_batch`
   (default 32, line 226) only chunks *this rank's own* shard into
   sub-batches — this is what the ticker's `rank 0 batch 1` label reports
   (`_batch_index`, line 1192). This path is not distributed and is not
   where the hang is.

2. **The real cross-rank rendezvous**: `B12xPreparationCoordinator` in
   `vllm/v1/worker/b12x_startup.py`. Each rank's local generator
   (`PreparationJob.advance()`, `session.py` line 631) yields a
   `CollectiveRequirement` or `TuningRequirement` when it reaches a boundary
   that needs agreement across ranks (a shared/collective plan like
   `gemm.bf16_vocab_projection`, or reconciling which rank's local
   candidate shard "won" a `TuningRequirement`). The coordinator collects
   these via `_ready()`/`_ready_tuning()` (`b12x_startup.py` lines 277-297)
   and only calls `_exchange()` (line 150/186) once **this rank itself**
   has something ready — `_advance()` line 147: `if not (self._local_done
   or self._error or self._ready() or self._ready_tuning()): return
   self._outcome()`.

## The exchange primitive: a TCPStore-backed round rendezvous, not a collective

`_exchange()` (`b12x_startup.py` line 186-203) does **not** use
`torch.distributed.all_gather`/`broadcast`. It uses a `dist.PrefixStore`
(`self._control`, constructed at line 71 as
`dist.PrefixStore(f"stage-{sequence}", group.store)`) as a manual
key/value rendezvous:

```
prefix = f"round-{self._round}"
self._control.set(f"{prefix}/{self.global_rank}", pickle.dumps(payload))   # line 191
if self.global_rank == self.world_ranks[0]:                                # line 192
    gathered = [self._control.get(f"{prefix}/{rank}") for rank in self.world_ranks]  # line 193-196: BLOCKS
    decision = self._decision(gathered)
    self._control.set(f"{prefix}/decision", pickle.dumps(decision))        # line 202
return self._control.get(f"{prefix}/decision")                             # line 203: BLOCKS on every rank
```

Both `TCPStore`/`PrefixStore.get()` calls poll-block (this is exactly the
`poll_schedule_timeout.constprop.0` wchan seen on **both** ranks'
`Worker_TP` process in every hang: rank0 stuck in the `.get()` at line 193
waiting for a peer key that never arrives, and/or every rank stuck in the
`.get()` at line 203 waiting for a `decision` key that rank0 never writes
because it's itself stuck at line 193).

## Why the peer key never arrives: `self._round` is a per-rank-local counter, not an agreed sequence number

`self._round` (`b12x_startup.py` line 63, `self._round = 0`) increments once
per rank, **only** at the end of `_advance()` after that rank personally
completed an exchange (line 183: `self._round += 1`, only reached past the
early-return gate at line 147-148). It is never itself exchanged or agreed
on ahead of time — the protocol's correctness depends entirely on an
implicit assumption that both ranks reach "something ready" the same
number of times, in the same order, so that "round N" means the same
logical boundary on both sides.

That assumption holds for genuinely fixed/shared collective plans (both
ranks declare the same units in the same order from the same model graph,
so they hit `gemm.bf16_vocab_projection`'s collective boundary at the same
round on both sides under normal, small (~222-candidate) retunes). It
**breaks down at scale**: once a retune has enough candidates per request
to force b12x's local, non-distributed candidate-racing loop into multiple
internal `race_batch` sub-batches (session.py line 1192,
`self._batch_index += 1`), the two ranks' *wall-clock/step pacing* through
their independent local shards diverges — different shard sizes from the
`[rank_index::rank_count]` stride, different per-candidate compile/measure
costs — even though both eventually finish. If, while diverged, rank0
reaches a *different* readiness boundary (e.g. finishes obligation A's
`TuningRequirement`) while rank1 is still mid-shard on an earlier
obligation and has nothing ready, rank0 calls `_exchange()` and — as
`world_ranks[0]` — blocks in the `.get()` at line 193-196 waiting on
`round-N/{rank1}`. This is fine and recoverable *as long as rank1
eventually also becomes ready for the *same* round's boundary set*.

The actual break: `_decision()` (line 205-225) raises
`RuntimeError("preparation ranks reached incompatible tuning boundaries")`
(line 217) or `"...has an invalid participant set"` (line 447/483) when
ranks disagree about the *content* of a round — but there is no code path
that **detects a rank stuck forever with nothing ready for the round the
peer is already blocked on**, i.e. no timeout/heartbeat on the `.get()`
calls at lines 193-196 and 203, and no cross-check that both ranks are
converging on the same next boundary. `PrefixStore.get()` has no explicit
timeout override here, so it inherits the process group's default TCP
store timeout (often very large / effectively "hang", matching the
20+ minute silences observed). Once the two ranks' local generators diverge
far enough in *pacing* at the enlarged (349-837 candidate) scale that one
rank's `gemm.bf16_vocab_projection` priming keys never line up with what
the other rank has queued for the same `round-N`, both sides end up
parked in a `.get()` that nothing will ever satisfy — a genuine protocol
gap, present in a8333658 itself (confirmed: the same hang signature
reproduces on the *unmodified* a8333658 preparation protocol once
`mods/b12x-fwd-57f3572` pushes total candidates past whatever threshold
first exercises this path — see `results/kernel-pass/fwd57f3572-hang3/SUMMARY.md`).

## Evidence cross-check

- Hang #2 (`b12x0f3a8cb-hang2`) and hang #3 (`fwd57f3572-hang3`): both show
  the **same symmetric** `poll_schedule_timeout.constprop.0` wchan on both
  ranks — consistent with both being parked in `PrefixStore.get()`
  (line 193-196 for one rank, line 203 for the other, or both on line 203
  if rank0 itself never got a matching peer key and never got to write
  `decision`).
- rank1's own server log "never printed a single `b12x compiling/priming`
  line" in both hangs — consistent with rank1 being stuck for its *entire*
  preparation lifetime on an early boundary (round 0's `.get()` at line
  203, or a local generator stall before ever reaching a `_race`/`_advance`
  cycle that would print a ticker line at all) while rank0 alone climbs
  through hundreds of candidates and prints ticker output.
- The `shm_broadcast.py:801 "No available shared memory broadcast block
  found in 60 seconds"` warning from vLLM's `EngineCore` is a **red
  herring / downstream symptom**, not the root cause: it is vLLM's
  unrelated RPC heartbeat between the engine and its workers timing out
  because the worker process is blocked inside the b12x preparation call
  (which runs on the worker's main thread) and cannot service the engine's
  liveness poll — it fires *after* the real deadlock (the `PrefixStore.get()`
  block) has already occurred, not before it.

## Named fix candidates, evaluated

- **(a) force a single batch round via an existing constant/env** — not
  found. `race_batch=32` (`session.py` line 226) is a constructor default,
  not read from any `B12X_*` env var anywhere in `session.py`, and it only
  controls the *local* per-rank batching (mechanism #1 above), which is
  not the broken protocol — raising it would not fix the cross-rank
  `_round` divergence in mechanism #2. No env-only fix exists.
- **(b) make both ranks enter the batch loop with an agreed batch count/
  round number, exchanged once up front** — the closest correct fix
  conceptually, but the actual gap is deeper: it is not "batch count" that
  needs agreeing on, it is *round pacing* — the coordinator would need a
  bounded-wait on `_exchange()` with a fallback (e.g., a rank that reaches
  round N with a boundary the peer hasn't reached should be able to detect
  "peer is still behind" and keep pumping `_advance_local()` — read: the
  fix belongs at `b12x_startup.py` line 147's gate, or by giving the
  `_control.get()` calls a timeout that surfaces as a retryable "not yet"
  instead of blocking indefinitely. This requires understanding b12x's own
  invariants about round agreement well enough to know it's safe (e.g.
  whether `round-N/{rank}` keys ever get *retried* with new content, which
  would make a naive polling fix corrupt state). This is not a one-line
  patch and carries real risk of turning a hang into silent wrong-answer
  candidate selection if done carelessly.
- **(c) relax the vLLM-side EngineCore heartbeat/shm timeout** — ruled out:
  per the evidence above, that timeout is a downstream symptom of the real
  deadlock, not its cause. Relaxing it would not un-stick the
  `PrefixStore.get()` calls; the process would still hang, just without
  the (currently useful) 60s warning that helps date the freeze.

## Recommendation

None of the three candidates from the task brief is a safe, small,
high-confidence fix once traced to the actual mechanism: the bug is a
missing bounded-wait/retry (or an explicit round-barrier handshake) in
`B12xPreparationCoordinator._exchange()` (`b12x_startup.py` lines 147-203),
not a constant, env var, or unrelated timeout. A correct fix needs either
upstream b12x work on the `round-N` rendezvous itself, or a locally-authored
retry/backoff patch to `b12x_startup.py` that is validated against b12x's
actual invariants for `round-N/{rank}` key reuse — which requires either
b12x source comments/tests we don't have visibility into, or a live
instrumented repro (add round/readiness debug logging to
`_advance()`/`_exchange()`, capture one more hang with that logging, and
confirm which rank is parked where and on what round before writing a
patch). Recommended next step, if pursued further: a cheap instrumentation
mod (log `self._round`, `self._ready()`, `self._ready_tuning()`,
`self._active_requests` names on every `_advance()` call, rank-tagged) on
one more empty-cache retune boot, *before* attempting a behavioral patch —
this confirms the round-divergence hypothesis above with actual line
numbers/rounds instead of inferring it from wchan alone, and costs the same
one boot as a blind fix attempt but produces a debuggable trace even if it
still hangs.

## Update 2026-09-19 (trace1 boot): the actual root cause is one layer deeper

The `results/kernel-pass/prep-deadlock/trace1/` boot (recipe
`eugr-agents-serve-local16-la-fwd57f3572-trace.yaml`, mods
`[b12x-startup-trace, b12x-fwd-57f3572]`, empty plan cache, both nodes'
`b12x` cache saved aside first as `b12x.la-serving-20260919`) did NOT
reproduce a silent hang. The bounded-wait instrumentation (120s Store.wait
retries in `_exchange()`) turned what would have been another silent
`poll_schedule_timeout` freeze into an explicit, fast error:

Round 0 (`candidates=8/8`, the RoCE collective's local candidate shard)
exchanged cleanly on both ranks — `round-0/0`, `round-0/1`, and the
`decision` key were all written and read normally, confirming the
metadata-level round rendezvous (`b12x_startup.py`'s `_exchange()`) is
**not** where this particular failure lives; the earlier mechanism
analysis of that rendezvous still stands as a real gap (no bound on
`Store.get()`), just not the trigger here.

Full error, rank0's log (`trace1/rank0-sparkrun_serve.log:582`):

```
(EngineCore pid=262) ERROR 09-19 11:48:07 [core.py:1484] RuntimeError: b12x preparation failed on rank 1: RuntimeError: distributed.roce.0-1.collectives failed to prepare with configuration BackendConfig(backend='native') (fixed): RoCE collective on rank 1 timed out waiting for rank 0, HCA 0, at sequence 1; the runtime is poisoned (its epoch stopped at 0, later launches do nothing) and rank data is no longer trustworthy
```

Rank1's own log (`trace1/rank1-sparkrun_serve.log`) has no separate
traceback: `B12xPreparationCoordinator._record_error()`
(`b12x_startup.py:332-337`) stores only `{rank, type, message}`, no
traceback, and this message is the complete text — it is not truncated,
there is nothing more to recover from rank1's side.

**Root cause, traced into b12x's RoCE communicator**
(`b12x/comm/roce/roce_oneshot.py`): the collective is a one-shot GPU
kernel that stages its payload, rings a proxy thread, and then
**busy-spins on the GPU** reading a peer-written flag (`ld_relaxed_gpu_u32`
in `_oneshot_cute.py`) for up to `self.spin_limit` polls
(`roce_oneshot.py:237`, `self.spin_limit = _env_int("B12X_ROCE_SPIN_LIMIT",
default=DEFAULT_SPIN_LIMIT)`) before giving up. `DEFAULT_SPIN_LIMIT =
20_000_000` (`roce_oneshot.py:51`), documented in the source as "about
20 s" (each poll ~1 microsecond). Once the spin limit is hit,
`check_health()` (`roce_oneshot.py:648-666`) raises exactly the error
above and marks the runtime `poisoned` (permanent — "later launches do
nothing", `roce_oneshot.py:670`).

This lines up perfectly with everything upstream: `distributed.roce.0-1.
collectives` is a genuinely **shared** collective plan (both ranks must
call it), authorized via the (working) round-0 metadata exchange, and then
both ranks independently call into `_advance_local()` (`b12x_startup.py:
238-259`) to actually execute it. Steady-state serving calls this from a
tight decode loop where both TP ranks are essentially always in lockstep,
so ~20s of margin is generous. During a from-empty-cache retune, the two
ranks' **local, non-collective MoE candidate racing** (mechanism above,
`session.py:1153-1154`, `[rank_index::rank_count]` sharding) runs at
different paces per rank (different shard sizes, different per-candidate
compile/measure cost), so one rank can be busy elsewhere for much longer
than 20s before it reaches this specific shared collective — exactly the
round-pacing divergence the original mechanism write-up predicted, just
surfacing here as a **hard 20s GPU-kernel timeout inside b12x's RoCE
runtime**, not as a `Store.get()` block. It is env-tunable:
`B12X_ROCE_SPIN_LIMIT` (`roce_oneshot.py:237`), a plain
`_env_int(...)` read, no code change required.

**Practical effect on the earlier (silent) hangs**: they are almost
certainly the same divergence, just landing on a still-unbounded
`Store.get()` in `_exchange()` (metadata layer) instead of the RoCE
spin-timeout (execution layer) — which one triggers first depends on
which shared boundary the ranks happen to diverge around
(`gemm.bf16_vocab_projection`'s priming vs. the RoCE collective's own
execution). Both are the same root disease: **rank pacing divergence
during a from-empty-cache retune, colliding with fixed, short timeouts
tuned for steady-state lockstep serving, not for retune-time skew.**

## Fix shipped

1. `B12X_ROCE_SPIN_LIMIT` raised in the recipe env (no code mod) to give
   the RoCE collective enough margin to tolerate retune-time rank-pacing
   skew.
2. `mods/b12x-startup-boundedwait/` (trimmed from the diagnostic
   `b12x-startup-trace/`): keeps only the fail-fast behavior (120s bounded
   `Store.wait`/retry in `_exchange()`, no verbose per-advance logging) as
   a permanent belt-and-suspenders guard against the metadata-level
   rendezvous hang, independent of the RoCE fix.

## Update 2026-09-19 (final): fwd57f3572-fix rejected, la kept with the fix

Re-screen on the same warm `fwd57f3572-fix` boot confirmed the first
c1=87.9 reading was not noise: c1=88.1 tok/s on rerun (vs. la baseline
95.7-98.7 across its own runs) -- a real, repeatable ~10% c1 regression.
Per the standing bar (c1 >= 101.8 or c8 >= 432.6, nothing worse than -2%),
this arm does not clear it and was not promoted; `gate_arm.sh` was not run
for it.

### moe.decode / fused_moe selection verification
Dumped the b12x selection cache from the fwd57f3572-fix warm boot
(`results/kernel-pass/prep-deadlock/moe-decode-selection.txt`): of 282
cached selection records, 38 carry a `w4a16_route_mode` field (i.e. are
fused-MoE configs where the native W4A16 direct route added by
`mods/b12x-fwd-57f3572` is even a candidate), and **4 of those races were
won by `backend="w4a16", w4a16_route_mode="direct"`** (the new route) --
so the route is not dead on arrival, it does win some races. The other 34
w4a16-route-mode-eligible records chose `dynamic`/`cutedsl`/`triton`
instead. Exact per-decode-capacity (1/2/4/8) attribution was not
recoverable from the cache alone -- `session.py`'s cache key
(`_choice_key`, `session.py:770-786`) is an opaque sha256 digest with no
stored request name; doing that mapping would require recomputing the
digest inside a live process with the real contract per capacity, out of
scope here. Net: the new route is real and sometimes wins, but its net
effect on this recipe's c1/c8 was negative, not neutral -- consistent with
either a regression in the route itself at the capacities that matter for
c1 (batch size 1, most latency-sensitive), or with the MoE retune landing
on a slightly worse spot overall for those shapes than the original
~222-candidate default tuning that `la`'s cache already had. Not
investigated further (out of scope for this pass): the fix's job was
"make the retune survive," which it does; whether `fwd57f3572` itself is
a net win is a separate question the numbers say "no, not yet."

### Restore + occupancy arms
`la`'s original cache restored from `b12x.la-serving-20260919`. The
`fwd57f3572`-tuned cache (empty-cache retune result) preserved as
`b12x.fwd57f3572-tuned` on both nodes for any future revisit.
`eugr-agents-serve-local16-la.yaml` updated in place: `mods:
[b12x-startup-boundedwait]` + `B12X_ROCE_SPIN_LIMIT: "300000000"` --
startup-only robustness, so a cold node (empty cache, e.g. after a crash)
now fails fast/retries instead of hanging, without touching any
serving-time behavior. Confirmed: boot healthy, 0 measured/272+ cached
(pure cache hit, artifact recompilation only, no retune), c1 98.7 tok/s /
c8 441.8 tok/s on the clean re-run (>= 95 bar met; the very first
measurement after boot read 85.7/411.9, attributable to background
artifact-compilation still finishing -- see "0 measured, N compilations"
ticker lines during warm boots -- not a real regression).

Occupancy knobs: `_get_impl_mac()` (`b12x/moe/fused_moe/_impl.py:9802-9822`)
computes `mac_limit = min(get_max_active_clusters(1), sm_count)` and clamps
any override to `min(override, mac_limit)` -- on this GPU (GB10, TP=2)
`sm_count = get_max_active_clusters(1) = 48`, so **any override above 48 is
a mathematically provable no-op** (clamped straight back to the unset
default). "Higher" arms (64) were still run empirically to confirm this
rather than assumed. Results, la's own cache kept warm throughout (no
retune):

| arm | c1 tok/s | c8 tok/s | vs la (c1/c8) |
|---|---|---|---|
| la (post-fix, clean run) | 98.7 | 441.8 | baseline |
| B12X_MICRO_MAX_ACTIVE_CLUSTERS=32 | 97.1 | 429.9 | -1.6% / -2.7% |
| B12X_MICRO_MAX_ACTIVE_CLUSTERS=64 | 87.9 | 423.7 | -10.9% / -4.1% (no-op point, confirms clamp -- within run-to-run noise of baseline) |
| B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS=32 | 94.7 | 426.2 | -4.1% / -3.5% |
| B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS=64 | 86.6 | 439.0 | -12.2% / -0.6% (no-op point) |

None beats la by the required margin (>= 3% c1 or >= 5% c8, nothing worse
than -2%) -- all four points fall inside this hardware's observed
run-to-run noise band (roughly 85-99 tok/s c1, 412-442 tok/s c8 across
every la-family boot measured this session). No occupancy arm was gated
or promoted. Final serving state reverted to plain `la` (no occupancy
override), which is the unmodified, hardware-default value anyway.
