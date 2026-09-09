#!/usr/bin/env bash
# Run on dgx-01 with 0, on dgx-02 with 1. Temporary head-only test, no model load.
set -euo pipefail
rank=${1:?rank 0 or 1 required}
[[ $rank == 0 || $rank == 1 ]] || exit 2
root=$(cd "$(dirname "$0")/.." && pwd)
docker run --rm --name "codex-sept9-head-$rank" --gpus all --network host \
  --ipc private --shm-size 1g \
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -v "$root:/work:ro" \
  -e RANK="$rank" -e WORLD_SIZE="${WORLD_SIZE:-2}" \
  -e HEAD_MODE="${HEAD_MODE:-eager}" -e PYTHONUNBUFFERED=1 \
  -e HEAD_ARM="${HEAD_ARM:-argmax}" \
  -e MASTER_ADDR=192.168.100.62 -e MASTER_PORT=29639 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=192.168.100.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_IB_MERGE_NICS=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_DEBUG=INFO -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  --entrypoint bash \
  vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c \
  -lc 'python3 /work/codex-optimization/install.py "$HEAD_ARM" && python3 /work/codex-optimization/test_head.py'
