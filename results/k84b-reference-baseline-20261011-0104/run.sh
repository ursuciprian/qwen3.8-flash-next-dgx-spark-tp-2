#!/usr/bin/env bash
# k84b-reference-baseline (2026-10-10, tp-2 #155): the checkpoint publisher's reference vLLM image for DGX Spark with its
# own two-Spark preset, on the unmodified checkpoint, same cells and scripts as k86/k84. Measure only; no tuning by me.
#   image    ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-spark-20261007-d071fb3dde0ba979 (= the
#            karmic-kraken-beta-spark alias on 2026-10-10, the newest linux/arm64 Spark tag; digest pinned below),
#            launcher revision 475fcd25 of the image's source repo
#   preset   PRESET=qwen38-dgx-spark-x2 (profile qwen38-flash-next, hardware dgx-spark): TP2 over the RoCE link, one
#            container per Spark (NODE_RANK 0/1, MASTER_ADDR = dgx-01's RoCE address), 16 sequences, utilization 0.75,
#            VRAM prefix cache, MTP 3 probabilistic drafts from the profile. The publisher ships no single-Spark Qwen
#            preset ("one box cannot hold the 100 GiB of weights"), so there is no 1x run.
#   set by me, not tuning:
#     MODEL_REVISION=7c4f1bc1 + HF_HUB_OFFLINE=1   the preset's "original" checkpoint is unpinned (Hub main is now
#                                                   6c01ac37, 2026-10-06, not on disk); this job measures 7c4f1bc1
#     NCCL_IB_HCA, NCCL/GLOO_SOCKET_IFNAME, VLLM_HOST_IP   the per-deployment RoCE settings the hardware profile asks
#                                                   for (values from sparkrun's own detection on these Sparks)
#     --served-model-name qwen3.8-flash-next        API name the harness sends (the preset's name is kept as well)
#     container names sparkrun-k84b-r0/r1           so lib.sh stop_all, the memory guard and others_busy see them
#   setups   ref-2x-mtp3 (preset as is); ref-2x-mtp4 (+ --draft-tokens 4, the one-option change to my recipes'
#            4 probabilistic drafts; the profile already samples drafts probabilistically), only if mtp3 booted
#   cells    cred.sh std_cells as k84/k86 (idle 120 s; benchy tg512 c1/c8 at T=1.0 top-p 0.95 top-k 20 thinking on and
#            T=0 thinking off; coding-36 T=0 thinking off c1/c8, one pass), GPU power 1 Hz; cred.py --vs k86
# ETA ~2 h: checks 5 min, boot <=45 min (cold JIT cache), cells 25 min, second boot ~15 min, cells 25 min, restore 6.
# Output: $RES/<unit>/ benchy + coding json/log, serve logs per rank, print-config, image json; units.tsv, marks.tsv,
# power, attempts.txt, cells.csv, boots.csv, cred.txt; comment.md here + RESULT.
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; EXIT trap removes the containers and restores the shipped 2x (lib.sh).
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k84b; K=$HOME/GEN-AI/backlog/k84b
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k84b-reference-baseline-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"
IMG=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-spark-20261007-d071fb3dde0ba979
DIG=sha256:af14f1a90e141ad7a1b4951f0fd37d695101ebd5887fcddb499f2f427eff5319
PRESET=qwen38-dgx-spark-x2; REV=7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd; PASSES=1; K86=$(cat "$BL/k86/RESULT" 2>/dev/null)
HCA=rocep1s0f1,roceP2p1s0f1; IF=enp1s0f1np1

# docker run args for rank r on host h, extra launcher options after the image
dargs() { # rank host-ip -> env/volume args
  echo "--init --privileged --gpus all --network host --ipc host --shm-size 32g --ulimit memlock=-1 \
 -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $K/cache-r$1:/cache \
 -e PRESET=$PRESET -e NODE_RANK=$1 -e NNODES=2 -e MASTER_ADDR=$H1 -e PORT=8000 -e MODEL_REVISION=$REV -e HF_HUB_OFFLINE=1 \
 -e NCCL_IB_HCA=$HCA -e NCCL_SOCKET_IFNAME=$IF -e GLOO_SOCKET_IFNAME=$IF -e VLLM_HOST_IP=$2 -e NCCL_CROSS_NIC=1"; }
OPTS0="--served-model-name qwen3.8-flash-next"   # the launcher takes one name; the preset default is Qwen3.8-Flash-Next
printcfg() { # extra-opts -> resolved config of rank 0 (no GPU, no services)
  timeout -k 30 300 docker run --rm $(dargs 0 $H1) "$IMG" --print-config $OPTS0 "$@" 2>&1; }

checks() {
  local ok=0 h t
  for h in $H1 $H2; do
    t=$(x $h "docker image inspect --format '{{.Architecture}} {{index .RepoDigests 0}}' $IMG 2>/dev/null")
    echo "$t" | grep -q "arm64 .*$DIG" && echo "$(hn $h): image present, $t" || { echo "$(hn $h): image missing or digest differs ($t)"; ok=1; }
    t=$(x $h "ls ~/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/$REV/model-000*-of-00036.safetensors 2>/dev/null | wc -l; ls /dev/infiniband >/dev/null 2>&1 && echo ib; ip -br addr show $IF | grep -oE '192\.168\.100\.[0-9]+'")
    [ "$(echo $t)" = "36 ib $h" ] && echo "$(hn $h): 7c4f1bc1 36 shards, /dev/infiniband, $IF = $h" || { echo "$(hn $h): checkpoint/RDMA/interface not as expected ($t)"; ok=1; }; done
  if [ $ok = 0 ]; then
    for o in "" "--draft-tokens 4"; do
      f=$K/print-config${o:+-mtp4}.txt; printcfg $o > "$f"
      t=$(python3 - "$f" <<'EOF2'
import json, sys
a = json.load(open(sys.argv[1]))["argv"]; v = lambda k: a[a.index(k) + 1] if k in a else "-"
spec = json.loads(v("--speculative-config")) if "--speculative-config" in a else {}
print(v("--revision"), v("--tensor-parallel-size"), v("--nnodes"), spec.get("num_speculative_tokens", "-"), spec.get("draft_sample_method", "-"))
EOF2
)
      echo "print-config ${o:-(preset)}: revision TP nnodes drafts sampling = $t"
      case "$t" in "$REV 2 2 "*) ;; *) echo "  not as expected"; ok=1 ;; esac
      [ -n "$o" ] && { case "$t" in *" 4 probabilistic") ;; *) echo "  --draft-tokens 4 did not give 4 probabilistic drafts"; ok=1 ;; esac; }; done
  fi
  [ -s "$K86/boots.csv" ] && echo "compare with k86: $K86" || { echo "k86 results missing"; ok=1; }
  python3 "$CRED/cred.py" --selftest > /dev/null && python3 "$CODING" --selftest > /dev/null || { echo "cred.py/coding.py selftest failed"; ok=1; }
  mkdir -p "$(dirname "$RES")" && [ -w "$(dirname "$RES")" ] && echo "results dir writable" || { echo "results dir not writable"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  checks; rc=$?
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "plan: $IMG, PRESET=$PRESET at 7c4f1bc1: ref-2x-mtp3 (preset), then ref-2x-mtp4 (--draft-tokens 4) if the first booted; k86 cells, 1 coding pass"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
rmc() { docker rm -f sparkrun-k84b-r0 > /dev/null 2>&1; ssh -n -o ConnectTimeout=10 $H2 "docker rm -f sparkrun-k84b-r1" > /dev/null 2>&1; }
trap 'pw_stop; rmc; kill $G1 $G2 2>/dev/null; restore || { sleep 60; restore; } || FINAL="FAILED: restore -- $FINAL"; st "$FINAL"' EXIT
echo "$RES" > "$K/RESULT"; : > "$K/comment.md"
st "running: preflight"
checks > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
cp "$K/run.sh" "$K"/print-config*.txt "$RES/"; mkdir -p "$K/cache-r0"; x $H2 "mkdir -p $K/cache-r1"
attempt() { echo "$(TZ=Europe/Bucharest date '+%F %T') $*" >> "$RES/attempts.txt"; }

st "running: GPU check in the image (CUDA 13.4 user space on driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader))"
docker run --rm --gpus all --entrypoint /opt/venv/bin/python "$IMG" -c "import torch; print('cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" > "$RES/gpu-check.txt" 2>&1
attempt "GPU check: $(tail -1 "$RES/gpu-check.txt" | cut -c1-200)"

rboot() { # unit extra-opts... -> 0 when rank 0 answers /health and the pong
  local u=$1 s=$(date +%s) a; shift; mkdir -p "$RES/$u"; stop_all; rmc
  printcfg "$@" > "$RES/$u/print-config.txt"
  ssh -n $H2 "docker run -d --name sparkrun-k84b-r1 $(dargs 1 $H2) $IMG $OPTS0 $*" > "$RES/$u/run-r1.txt" 2>&1
  docker run -d --name sparkrun-k84b-r0 $(dargs 0 $H1) "$IMG" $OPTS0 "$@" > "$RES/$u/run-r0.txt" 2>&1
  while [ "$(health $H1)" != 200 ]; do a=$(( $(date +%s) - s ))
    if [ $a -gt 2700 ] || { [ $a -gt 120 ] && { [ -z "$(docker ps -q --filter name=sparkrun-k84b-r0)" ] || [ -z "$(x $H2 "docker ps -q --filter name=sparkrun-k84b-r1")" ]; }; }; then
      docker logs sparkrun-k84b-r0 > "$RES/$u/serve-r0.log" 2>&1; x $H2 "docker logs sparkrun-k84b-r1" > "$RES/$u/serve-r1.log" 2>&1
      log "$u: not up after $a s"; return 1; fi
    sleep 20; done
  echo "boot_s $(( $(date +%s) - s ))" > "$RES/$u/boot.txt"
  docker logs sparkrun-k84b-r0 > "$RES/$u/serve-r0.log" 2>&1; x $H2 "docker logs sparkrun-k84b-r1" > "$RES/$u/serve-r1.log" 2>&1
  docker image inspect "$IMG" > "$RES/$u/image-dgx01.json"; a=$(pong $H1); echo "pong $a" >> "$RES/$u/boot.txt"; echo "$a" | grep -qi pong; }
why() { grep -hE -m6 'Error|error:|OutOfMemory|out of memory|CUDA driver|not supported|unrecognized|Traceback' "$RES/$1"/serve-r*.log 2>/dev/null | cut -c1-300; }

pw_start; n=0
for v in "ref-2x-mtp3|" "ref-2x-mtp4|--draft-tokens 4"; do
  u=${v%%|*}; o=${v#*|}
  [ $u = ref-2x-mtp4 ] && [ $n = 0 ] && { attempt "$u: skipped (the preset itself did not boot)"; break; }
  st "running: boot $u"
  if rboot $u $o; then n=$((n + 1)); attempt "$u: booted ($(tr '\n' ' ' < "$RES/$u/boot.txt"))"
    unit $u $u 1 dgx01+dgx02 "$IMG $PRESET $o"; st "running: $u cells (~25 min)"; std_cells $u $H1 $PASSES nodefault
  else attempt "$u: did not boot; $(why $u | head -3 | tr '\n' ' ')"; fi
  rmc; done
stop_all; pw_stop

st "running: report"
python3 "$CRED/cred.py" report "$RES" ${K86:+--vs "$K86"} > "$RES/report.log" 2>&1 || log "report: no measured cells"
{ echo "Reference baseline (k84b-reference-baseline): the checkpoint publisher's reference vLLM image for DGX Spark, \`$IMG\` (\`$DIG\`, linux/arm64, the newest Spark tag on 2026-10-10), with its own two-Spark preset \`$PRESET\` (TP2 over RoCE, one container per Spark, MTP 3 probabilistic drafts, 16 sequences, utilization 0.75), on the unmodified \`local-inference-lab/Qwen3.8-Flash-Next-NVFP4\` @ \`7c4f1bc1\`."
  echo; echo "What I set: the checkpoint revision (\`MODEL_REVISION=7c4f1bc1\`, offline; the preset leaves it unpinned and the Hub main is now a newer revision), the per-deployment RoCE settings the hardware profile asks for (\`NCCL_IB_HCA\`, socket interfaces, host IP), the API name (\`qwen3.8-flash-next\`, which my harness sends; the launcher takes one name) and container names. No serving option was tuned. The \`mtp4\` setup adds only \`--draft-tokens 4\`, the draft count of my recipes. There is no single-Spark Qwen preset from the publisher (its preset notes say one box cannot hold the weights), so this is 2x only. The resolved launcher config of each setup is in \`print-config.txt\`."
  echo; echo "Boot attempts:"; echo '```'; cat "$RES/attempts.txt"; echo '```'
  echo; echo "Cells and metrics are the same as k86 and k84: llama-benchy pp2048/tg512 task mode, 3 runs, c1 and c8, at T=1.0 top-p 0.95 top-k 20 with thinking on (\`tgdef\`) and at T=0 with thinking off (\`tgt0\`); the 36-prompt coding probe at T=0 with thinking off, c1 and c8, one pass. GPU power is nvidia-smi power.draw at 1 Hz on both Sparks (GPU only, not wall power). The last table lists my shipped builds from k86 (3 boots each) next to these runs."
  echo; echo '```'; cat "$RES/cred.txt" 2>/dev/null || echo "no measured cells"; echo '```'
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
cp "$K/comment.md" "$RES/comment.md"
FINAL="DONE: k84b $n/2 reference setups measured ($RES/attempts.txt)"
exit 0
