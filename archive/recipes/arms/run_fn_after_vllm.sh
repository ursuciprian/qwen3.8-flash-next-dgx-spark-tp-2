#!/usr/bin/env bash
set -x
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
until grep -q LADDER_DONE /home/nvidia/GEN-AI/qwen3.8-27b-dgx-spark/results/vllm-tp1/ladder-vllm-tp1.log 2>/dev/null; do sleep 120; done
sleep 60
for _ in $(seq 1 36); do n=$(docker ps -q --filter name=sparkrun | wc -l); m=$(ssh 192.168.100.53 'docker ps -q --filter name=sparkrun | wc -l'); [ "$n" = 0 ] && [ "$m" = 0 ] && break; sleep 5; done
docker rm -f vllm-router sgl-gateway >/dev/null 2>&1
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; ssh 192.168.100.53 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null'
mkdir -p results/arms
exec scripts/ab_ladder.sh fn-control fn-capfit fn-capfit-mtp2 fn-capfit-mtp4 fn-capfit-ep fn-control > results/arms/ladder-fn1.log 2>&1
