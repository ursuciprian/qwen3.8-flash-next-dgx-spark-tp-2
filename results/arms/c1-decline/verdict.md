# c1 "decline" investigation — verdict (2026-09-21, 00:19–00:50 EEST)

Serving: `eugr-agents-serve-local16-la.yaml`, image `spark-vllm-b12x:local-20260918-a8333658`, fresh boot, no docker pulls / CI builds on either node during the series.

## Data (bench_sweep c1, 3 rounds x 300 tokens, back-to-back, ~24 s per run)

| series | c1 tok/s per run |
|---|---|
| idle series 1 (instrumented) | 99.1 88.2 96.7 97.1 84.2 |
| clock probe | 85.4 95.8 93.7 |
| core-placement probe | 97.3 90.6 98.0 97.3 92.9 |
| pinned to X925 cores 5-9,15-19 (both nodes, all container threads) | 84.3 85.9 97.5 98.7 95.5 97.8 |
| RoCE counter probe (still pinned) | 94.3 94.8 98.0 91.6 95.2 96.2 |

25 runs: median 95.5, fast mode 94-99 (~75% of runs), slow mode 84-88 (~25%). No monotonic decline; slow runs appear anywhere in a series. Within a slow run w2w p90 is close to the median (3.80/3.87 s) — the whole 24 s window is slow, not single stragglers.

## Ruled out (measured)
- MTP acceptance: draft/accepted deltas per run identical (2860/2820 ±10) across fast and slow runs.
- Thermal / power throttle: `nvidia-smi -q -d PERFORMANCE` all "Not Active", GPU 46 C.
- SM clock: 2184 MHz on both nodes in 100% of 211 samples during runs (governor `performance`, P0; `clocks.max.sm` reports 3003 MHz — not reached, likely the SoC sustained clock).
- CPU big/little placement (GB10 = 10x X925 cap ~1000 @3.9 GHz on cpus 5-9,15-19; 10x A725 cap ~720 @2.8 GHz on 0-4,10-14): vLLM threads do wander onto A725 cores (node1 `Worker_TP` up to 12/83 samples on little cores), but pinning every container thread to the X925 cores did not remove the slow mode (84.3, 85.9 still appear).
- RoCE (`rocep1s0f1` + `roceP2p1s0f1`, 200 Gb/s): per-run deltas of local_ack_timeout_err, packet_seq_err, implied_nak_seq_err, out_of_sequence, out_of_buffer, retrans, CNPs all zero on both nodes; ~930k pkts per device per run.
- Host memory pressure: PSI memory 0, swap 0 used, MemAvailable 10-14 GB during series.
- Concurrent pulls/builds: none (checked both nodes before and during).

## Still open
Step time toggles between ~55 ms and ~64 ms for windows of >= 24 s with nothing host-side changing. Remaining suspects are GPU-internal (unified-memory page placement / TLB after allocations, L2 residency, CUDA-graph replay path) or the vLLM async-scheduling pipeline. Needs a multi-window torch.profiler capture (extend `mods/vllm-decode-profiler` to record N spaced windows) on a la-lprof boot, then kernel-time comparison between a slow and a fast window.

## Operational consequences
1. Report c1 as median of >= 5 sweeps and also the max (fast-mode) value; a single 3-sweep median can land 10% low by chance. For kernel A/B, compare medians AND fast-mode values.
2. Earlier "decline" readings (96 -> 87 -> 73) were the slow mode plus real contention from a concurrent 24 GB docker pull (I/O + memory); the pull rule stands: no pulls / CI builds while la serves or benchmarks run.
3. Pinning to big cores is harmless (kept on the current la instance via `taskset` in-container; resets on restart). Not promoted to the recipe: no measurable gain.
