# September 9 handover

Read [README.md](README.md) and [results/STATUS.json](results/STATUS.json) first. This is the canonical local registry's isolated `codex-optimization/` package. Original recipes/mods were not edited.

## Completed evidence

- Real TP2 GB10 head parity passed: random rows, cross-rank ties, padded vocabulary, high IDs, eager and CUDA-graph paths. Both ranks used NCCL IB.
- Local argmax is a small batch benefit, effectively flat at batch 1. Head-only medians and every timing sample are in `results/head-{eager,graph}.json`. It has no measured serving gain.
- PR53945 metadata tests: 110 passed, one unrelated newer SWA API test fails identically on baseline. Raw logs are local in `results/`; repository policy ignores `.log` files. `cache-tests.sh` explicitly deselects that test.
- Both nodes built `flashnext-codex:20260909-cache` successfully from the locked original image. All six original Fable overlays were copied, including corrected ModelOpt and group annotation. Full fine/coarse serving remains unproven.
- The old Codex strict group patch and its stale loading status were retired in the older experimental repository's `codex-optimization/` folder.

## Incomplete live test

Experiment ID: **`sparkrun_96d0ff70f6fc7953_70afa7ea7889`**.

The dry run printed a different suffix. Use the ID above, confirmed from the actual containers, not the dry-run ID.

Last observed containers:

- Head: `sparkrun_96d0ff70f6fc7953_70afa7ea7889_node_0`
- Worker: `sparkrun_96d0ff70f6fc7953_70afa7ea7889_node_1`

This is **TP2 cache-fine**, MTP3, EP, stock block drop, prefix-match-unit 64. No argmax patch is included. Launch began September 9 at 15:16 UTC. Image/model sync completed; both containers were running at the last successful check. No successful health or inference result was observed before the Mac lost its IPv4 address/default route. Do not label this a pass or failure of the patch based on connectivity loss.

Remote experiment root on both nodes:

```text
/home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/codex-optimization/sept9/qwen3.8-flash-next
```

Launch log on dgx-01: `/tmp/codex-sept9-cache-launch.log`. The serving process is started with `docker exec` by sparkrun, so empty `docker logs` alone does not mean a failed boot. Inspect sparkrun's process log or `sparkrun logs --help`. The worker was reachable from dgx-01 over `192.168.100.53` even when the Mac could not reach its management IP.

After network restoration, inspect those two exact containers and health. If healthy and idle, reuse the existing six-request harness:

```sh
cd /home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2
python3 codex-optimization/cache-reuse/replay.py \
  --corpus codex-optimization/cache-reuse/corpus.txt \
  --out codex-optimization/sept9/cache-fine-six-requests
```

That harness was written for the older package but is independent of its rejected group patch. It checks actual prompt token counts, request-counter isolation, first-pass prefix reuse, and an exact retrieval key. It stops on the first failed gate and retains evidence. Do not rerun into an existing output directory.

Then save the real serving logs, HTTP output and counters, and stop **only** this experiment:

```sh
sparkrun stop sparkrun_96d0ff70f6fc7953_70afa7ea7889
```

Update STATUS.json with the observed outcome and final container state. Never use `stop --all`, change memory guards, or hide startup failures. The guards checked before launch were inactive. No guard or host configuration was changed by this package.

## Optional next component experiment

`probe_fp8_head.py` is an **unrun** standalone GPU feasibility probe for draft-only FP8 weight traffic. It never touches the target checkpoint or serving process. Run only after the serving experiment stops, in the pinned image. It uses torch's existing scaled matmul rather than a new kernel, records head timing and synthetic ranking agreement separately, and does not assert exact parity. A positive result would justify implementing a separately owned quantized draft head and then collecting real hidden-state/acceptance evidence. No FP8 serving recipe exists yet.

The argmax image can be built with `ARM=argmax`, but the tiny measured batch-1 gain does not justify making it the next expensive boot. Judge cache reuse first.
