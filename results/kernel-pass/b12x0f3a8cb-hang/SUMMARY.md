# b12x0f3a8cb hang evidence — 2026-09-19

Boot: `eugr-agents-serve-local16-la-b12x0f3a8cb.yaml`, image
`spark-vllm-b12x:local-20260919-0f3a8cbf-b12x0f3a8cb` (b12x 0f3a8cb, vllm
8e1f1e58+local patch commit e624ae1, eugr Dockerfile a33f4b5).

## Confirmed: genuine hang, not slow tuning

- `nvidia-smi`: both GPUs 0% util, ~9.6W (idle power draw) on both dgx-01
  and 192.168.100.53, sampled well after the "waiting for ranks" ticker
  froze.
- rank0 (dgx-01) log: `b12x waiting for ranks: 0/837 ready, 0 measured,
  0 cached, 0 compilations, 3:42` -- printed on a 10s cadence up to 3:42,
  then never printed again (checked repeatedly over 10+ minutes of wall
  time).
- rank1 (192.168.100.53) log (`/tmp/sparkrun_serve.log` inside the
  container -- `docker logs` only shows the CUDA container preamble, 15
  lines, the app's stdout isn't captured there for this headless worker):
  ends at "Padding mamba page size..." / the OMP_NUM_THREADS warning, i.e.
  right after model-weight loading completes (~09:14:41). **Rank1 never
  prints any "waiting for ranks"/priming line at all** -- it appears to
  reach the b12x preparation barrier and then simply stop logging.
- `/proc/<pid>/wchan`:
  - rank0 `Worker_TP0` (pid 266): wchan = `0` -- i.e. not blocked in an
    interruptible kernel sleep; it is on-CPU / runnable. Matches the
    measured ~60% CPU utime/stime delta (real cycles burned) despite 0%
    GPU utilization -- **consistent with a host-side spin/poll loop**, not
    GPU compute.
  - rank1 `Worker_TP1` (pid 233): wchan = `wait_woken` -- genuinely
    blocked in the kernel waiting to be woken (a real sleep, matches its
    lower and now-flat CPU%).
  - **Interpretation**: rank1 is legitimately parked waiting for a signal
    that rank0 should send; rank0 is spinning on-CPU (not actually
    computing GPU work) instead of either completing its side of the
    handshake or blocking the same way. This is the classic shape of a
    broken rendezvous/barrier in distributed preparation, not a slow
    autotune -- the GPUs would show >0% util if real MoE candidate timing
    were in flight.
- `/dev/shm` listing on both nodes shows long-lived `psm_*` (rank0) and
  `sem.mp-*` (rank1) shared-memory segments from earlier boots today, all
  with old timestamps except the newest ~09:14 entries from this boot's
  startup -- no obviously stuck/zero-byte segment, no smoking gun there
  beyond confirming this boot did reach IPC setup before stalling.
- `shm_broadcast.py:801` "No available shared memory broadcast block found
  in 60 seconds" repeated 3x (09:15:46, 09:16:46, 09:17:46) in rank0's log
  before the ticker itself froze at 3:42 -- this is vLLM's own
  shared-memory mailbox between EngineCore and the API server timing out
  repeatedly, additional corroboration that a distributed handshake is not
  completing.

## Suspected root cause

`06809d5` ("Reconcile cached tuning selections before distributed
preparation") rewired exactly this rendezvous: ranks exchange completed
selection metadata and agree on "the first completed record in rank order
for each key" before compiling/priming locally. The 837-candidate count
(vs. 222 on the a8333658 image) confirms the forced retune from 57f3572's
contract bump did trigger. The asymmetric wchan state (rank0 spinning,
rank1 legitimately parked) points at rank0's side of that new
agreement-exchange code never receiving/processing rank1's completion
signal (or looping instead of blocking while waiting for it) -- i.e. a bug
in this specific commit's 2-node coordination path, not a resource/OOM
issue and not something caused by our recipe or environment.

## Files in this directory
- `rank0-boot-log-tail150.log` -- last 150 lines of the sparkrun launcher
  stdout on dgx-01 (head node's view, includes the frozen ticker).
- `rank0-docker-logs-tail150.log`, `rank1-docker-logs-tail150.log` --
  `docker logs` output (only container preamble; the app's real stdout
  isn't captured there for either rank under this recipe/entrypoint).
- `rank0-sparkrun_serve.log`, `rank1-sparkrun_serve.log` -- the actual
  per-rank vLLM server logs from inside each container
  (`/tmp/sparkrun_serve.log`), last 150 lines each; rank1's is only 80
  lines total (nothing more was ever written).
- `rank0-wchan-shm.txt`, `rank1-wchan-shm.txt` -- `/proc/<pid>/wchan` plus
  `/dev/shm` listing captured on each node.
