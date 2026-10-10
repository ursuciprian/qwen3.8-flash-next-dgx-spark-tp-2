# Addresses of the Spark pair, for the scripts in this folder (source it: . "$(dirname "$0")/spark_env.sh").
# Tracked files carry no addresses. Each SPARK_* variable comes from the environment if set, else from the untracked
# file ${SPARK_PAIR_ENV:-$HOME/.config/spark-pair.env} (KEY=VALUE lines; template: scripts/spark-pair.env.example).
#   SPARK_HEAD_IP, SPARK_WORKER_IP   CX-7 addresses of the head (dgx-01) and the worker (dgx-02)
#   SPARK_CX7_PREFIX                 CX-7 subnet prefix including the trailing dot (run.sh refuses hosts outside it)
#   SPARK_HEAD_LAN, SPARK_WORKER_LAN LAN addresses of the two Sparks
# spark_need VAR... exits 2 with a clear message when one of them is unset.
_spark_env_file=${SPARK_PAIR_ENV:-$HOME/.config/spark-pair.env}
if [ -f "$_spark_env_file" ]; then
  while IFS='=' read -r _spark_k _spark_v; do
    [ -n "${!_spark_k:-}" ] || export "$_spark_k=$_spark_v"
  done < <(grep -E '^SPARK_[A-Z_]+=' "$_spark_env_file")
fi
# Older variable names still work: HEAD_IP / WORKER_IP fill SPARK_HEAD_IP / SPARK_WORKER_IP when those are unset, and
# SPARK_CX7_PREFIX defaults to SPARK_HEAD_IP up to its last dot.
[ -n "${SPARK_HEAD_IP:-}" ] || { [ -z "${HEAD_IP:-}" ] || export SPARK_HEAD_IP=$HEAD_IP; }
[ -n "${SPARK_WORKER_IP:-}" ] || { [ -z "${WORKER_IP:-}" ] || export SPARK_WORKER_IP=$WORKER_IP; }
[ -n "${SPARK_CX7_PREFIX:-}" ] || { [ -z "${SPARK_HEAD_IP:-}" ] || export SPARK_CX7_PREFIX=${SPARK_HEAD_IP%.*}.; }
spark_need() {
  local v
  for v in "$@"; do
    [ -n "${!v:-}" ] || { echo "error: $v is not set; export it or add $v=<value> to $_spark_env_file" \
      "(template: scripts/spark-pair.env.example)" >&2; exit 2; }
  done
}
