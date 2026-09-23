# `vllm-instanttensor-memory`

Port of **eugr/spark-vllm-docker `f247397d`** ("Instanttensor memory buffer
fix", 2026-09-18), upstream file `docker/patch_instanttensor_vllm_memory.py`,
test `tests/test_instanttensor_vllm_memory_patch.py`. Credit: eugr.

Not referenced by a shipped recipe yet.

## What it changes

One line in `instanttensor/_impl.py`, inside
`safe_open._determine_io_params` (the only `torch.cuda.mem_get_info()` in the
package):

```python
-        free_bytes, total_bytes = torch.cuda.mem_get_info()
+        from vllm.utils.mem_utils import MemorySnapshot
+        free_bytes = MemorySnapshot(device=self.device).free_memory
```

`free_bytes * max_free_mem_usage` (default 0.5) becomes
`self._device_memory_budget`, which sizes the weight-load staging buffer and
the I/O depth; under TP it is `all_reduce(MIN)`-ed across ranks first.
`total_bytes` had one binding and no reader, so it goes.

## Why it matters on GB10

`MemorySnapshot.measure()` in `vllm/utils/mem_utils.py` starts from
`torch.accelerator.get_memory_info()` -- the same cudaMemGetInfo figure -- and
then, when `current_platform.is_integrated_gpu(...)` and **not** CUDA-on-WSL,
overwrites it with `psutil.virtual_memory().available`. On UMA parts
cudaMemGetInfo does not count reclaimable host memory (page cache, buffers),
so it underreports what the device can actually allocate.

Our pair is Linux GB10, integrated, not WSL, so the psutil branch is taken:
this is a real behaviour change here, not a WSL-only fix. The load budget
becomes half of `MemAvailable` instead of half of cudaMemGetInfo free. It
touches weight-load staging only -- no effect on `gpu_memory_utilization`, the
KV budget, or anything after load.

## Applies to

`spark-vllm-b12x:local-20260918-a8333658`, `instanttensor/_impl.py`
SHA256 `4c5dc400...58cb7` -> `c6ebded8...77886`. InstantTensor is a separate
dist-package, not part of our vLLM fork (`local-inference-lab/vllm 8e1f1e58`,
`dev/jovian-judgement`), so the patch is cut against the installed file; the
`patches/vllm-instanttensor-memory.patch` name follows the registry's
convention, not the package's provenance.

## Fail-closed / idempotent

Exact SHA256 pre-image, `patch --forward --fuzz=0` dry-run then apply, SHA256
post-image, `ast.parse`, an AST check that the edit landed inside
`safe_open._determine_io_params` and that no other cudaMemGetInfo budget
survives, and a CPU-only import smoke of `MemorySnapshot` asserting the
`is_integrated_gpu` / `in_wsl` / `psutil.virtual_memory().available` branch is
still there (if upstream drops it the patch is pointless and the mod says so).
Re-running on a patched tree exits 0.

Tested CPU-only in the image on 2026-09-22: apply, re-apply (skip), and
tampered pre-image (refuse, rc=1). Not GPU-tested -- it needs a load pass to
show any effect.
