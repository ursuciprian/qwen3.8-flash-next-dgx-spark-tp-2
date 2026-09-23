#!/usr/bin/env bash
# Boot one arm, run the reuse probe, collect the diagnostic log lines, stop. No grid.
#   scripts/vllm_probe_only.sh <arm>
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEAD=${HEAD_IP:-192.168.100.62}; WORKER=${WORKER_IP:-192.168.100.53}; CLUSTER=${CLUSTER:-dgx-cluster-cx7}
CORPUS=${CORPUS:-$REPO/codex-optimization/campaigns/20260907T154810Z/a1-c1/corpus.txt}
arm="$1"; OUT="$REPO/results/arms/vllm-imp"; tag="dbg-$arm"; BASE="http://$HEAD:8000"
echo "=== $(date +%T) $tag start"
sparkrun stop --all >/dev/null 2>&1; sleep 5
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
ssh "$WORKER" 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null'
slog="$OUT/$tag-serve.log"
sparkrun run "$REPO/recipes/arms/$arm.yaml" --cluster "$CLUSTER" --tp 2 > "$slog" 2>&1 &
for _ in $(seq 1 120); do
  [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' "$BASE/health")" = 200 ] && break
  grep -aqE "launch_inference failed|Engine core initialization failed" "$slog" && { echo "!!! boot failed"; grep -aE "Error" "$slog" | tail -3 | cut -c1-200; sparkrun stop --all; exit 1; }
  sleep 15
done
M=$(curl -s -m 6 "$BASE/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
echo "--- $(date +%T) reuse probe"
python3 "$REPO/scripts/reuse_probe.py" "$BASE" "$M" "$CORPUS" "$OUT/$tag-reuse.json" 2>&1 | tee "$OUT/$tag-reuse.log"
sleep 3
grep -aoE "DBG_(MAMBA_CACHE|COORD_CACHE).*|CODEX_CACHE_(CONFIG|SPLIT).*|Flagged KV cache group.*" "$slog" > "$OUT/$tag-diag.log"
echo "diag lines: $(wc -l < "$OUT/$tag-diag.log")"
sparkrun stop --all >/dev/null 2>&1
echo "=== $(date +%T) $tag done"
