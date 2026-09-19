#!/usr/bin/env bash
# Node-local memory guard for the DGX Sparks.
#
# Why this exists: on GB10 the host and GPU share one 128 GB pool. A server
# that over-allocates starves userspace, and the first thing to go is sshd —
# so any watchdog that runs over SSH from another machine is useless exactly
# when it is needed. This runs ON the node.
#
# earlyoom is configured -m 2, i.e. it acts at ~2.5 GB free. sshd stops being
# able to accept new connections well above that, so there is a wide band
# where the box is alive but unmanageable. This guard acts inside that band.
#
# Install:
#   sudo cp spark-memguard.sh /usr/local/bin/
#   sudo cp spark-memguard.service /etc/systemd/system/
#   sudo systemctl enable --now spark-memguard
set -u
THRESHOLD_GB=${MEMGUARD_THRESHOLD_GB:-15}
GRACE_GB=${MEMGUARD_GRACE_GB:-25}
INTERVAL=${MEMGUARD_INTERVAL:-5}
LOG=/var/log/spark-memguard.log

log(){ echo "$(date -Is) $*" >> "$LOG"; }
avail(){ free -g | awk '/^Mem:/{print $7}'; }

log "memguard started: stop below ${THRESHOLD_GB}GB, warn below ${GRACE_GB}GB"
warned=0
while :; do
  a=$(avail)
  if [ "${a:-99}" -lt "$THRESHOLD_GB" ]; then
    log "CRITICAL: ${a}GB available, stopping inference to protect the host"
    # sparkrun first so cluster state stays consistent, then force any stragglers
    su - nvidia -c '~/.local/bin/sparkrun stop --all' >/dev/null 2>&1 || true
    docker ps -q --filter name=sparkrun | xargs -r docker rm -f >/dev/null 2>&1 || true
    sleep 20
    log "after stop: $(avail)GB available"
    warned=0
  elif [ "${a:-99}" -lt "$GRACE_GB" ]; then
    [ "$warned" = 0 ] && { log "WARN: ${a}GB available, approaching threshold"; warned=1; }
  else
    warned=0
  fi
  sleep "$INTERVAL"
done
