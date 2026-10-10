#!/usr/bin/env bash
# Boot one arm, run the cached-vs-fresh equivalence probe, stop.  scripts/vllm_equiv_only.sh <arm>
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEAD=${HEAD_IP:-<cx7-ip-a>}; WORKER=${WORKER_IP:-<cx7-ip-b>}; CLUSTER=${CLUSTER:-dgx-cluster-cx7}
CORPUS=${CORPUS:-$REPO/codex-optimization/campaigns/20260907T154810Z/a1-c1/corpus.txt}
arm="$1"; OUT="$REPO/results/arms/vllm-imp"; tag="equiv-$arm"; BASE="http://$HEAD:8000"
echo "=== $(date +%T) $tag start"
sparkrun stop --all >/dev/null 2>&1; sleep 5
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
ssh "$WORKER" 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null'
slog="$OUT/$tag-serve.log"
sparkrun run "$REPO/recipes/arms/$arm.yaml" --cluster "$CLUSTER" --tp 2 > "$slog" 2>&1 &
for _ in $(seq 1 100); do
  [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' "$BASE/health")" = 200 ] && break
  grep -aqE "launch_inference failed|Engine core initialization failed|Timed out after .* waiting for clients" "$slog" && { echo "!!! boot failed"; sparkrun stop --all; exit 1; }
  sleep 15
done
M=$(curl -s -m 6 "$BASE/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
echo "--- $(date +%T) equivalence probe"
python3 "$REPO/scripts/cache_equiv.py" "$BASE" "$M" "$CORPUS" "$OUT/$tag.json" 12 2>&1 | tee "$OUT/$tag.log" | grep -E "^EQUIV|match=False"
sparkrun stop --all >/dev/null 2>&1
echo "=== $(date +%T) $tag done"
