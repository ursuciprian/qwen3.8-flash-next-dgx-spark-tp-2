#!/usr/bin/env bash
# Overnight node watchdog for unattended ladders. Runs on the head node.
# Every 60 s: memory on both nodes, container/health state, ladder liveness. Only mutation:
# `sparkrun stop --all` when MemAvailable stays below 4 GiB on either node for two consecutive
# samples (memguard is off; earlyoom fires at ~2.4 GiB, too late for a clean stop). Everything
# else is logged for the morning. Log: results/arms/watchdog.log
export PATH="$HOME/.local/bin:$PATH"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
WORKER=${WORKER_IP:-192.168.100.53}; LOG=results/arms/watchdog.log; low=0; unhealthy_since=""
echo "$(date +%F_%T) watchdog start" >> "$LOG"
while true; do
  a=$(free -g | awk '/^Mem:/{print $7}'); b=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$WORKER" "free -g | awk '/^Mem:/{print \$7}'" 2>/dev/null || echo "?")
  psi=$(awk '/^some/{for(i=1;i<=NF;i++) if($i ~ /^avg60=/){split($i,x,"="); print x[2]}}' /proc/pressure/memory)
  c=$(docker ps -q | wc -l); h=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://192.168.100.62:8000/health 2>/dev/null)
  up=$(docker ps --format '{{.Status}}' | head -1)
  lad=$(pgrep -f "scripts/(ab_ladder|vllm_ladder|vllm_probe_only).sh" | wc -l); wait=$(pgrep -f "until grep" | wc -l)
  line="$(date +%T) availG head=$a worker=$b psi60=$psi containers=$c health=$h up='$up' ladder_procs=$lad waiters=$wait"
  echo "$line" >> "$LOG"
  # memory floor: two consecutive samples under 4 GiB on either node -> stop the run, keep the node
  if { [ "$a" != "?" ] && [ "$a" -lt 4 ]; } || { [ "$b" != "?" ] && [ "$b" -lt 4 ]; }; then low=$((low+1)); else low=0; fi
  if [ "$low" -ge 2 ]; then echo "$(date +%T) !!! MEMORY FLOOR (head=$a worker=$b) - sparkrun stop --all" >> "$LOG"; sparkrun stop --all >> "$LOG" 2>&1; low=0; fi
  # container up but never healthy for 40 min -> note it (the ladders time out themselves at 25-30 min)
  if [ "$c" -gt 0 ] && [ "$h" != 200 ]; then [ -n "$unhealthy_since" ] || unhealthy_since=$(date +%s)
    if [ $(( $(date +%s) - unhealthy_since )) -gt 2400 ]; then echo "$(date +%T) !!! container up >40 min without health 200" >> "$LOG"; unhealthy_since=$(date +%s); fi
  else unhealthy_since=""; fi
  # nothing driving the cluster and no LADDER_DONE anywhere -> flag
  if [ "$lad" = 0 ] && [ "$wait" = 0 ] && ! grep -qs LADDER_DONE results/arms/vllm-imp/chain.log.all 2>/dev/null; then echo "$(date +%T) !!! no ladder/probe/waiter process alive" >> "$LOG"; fi
  sleep 60
done
