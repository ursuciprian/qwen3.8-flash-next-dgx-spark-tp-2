#!/usr/bin/env bash
# k67 (2026-10-07, #113): MTP draft-step attention width/stride on the single-Spark build (v3d). Draft steps 1-3 run
# QSA through DraftSelectionReuse; the reuse rows are 2055 wide (anchor 2051 + 4-column chain tail) but the attention
# runs programs.sparse, compiled for 2051 columns and a 2051 row stride: the tail is masked and, with 2+ requests,
# row r is read 4r columns early (loses its newest columns, attends the previous row's tail as its own positions).
#   k67a  dgx-01  b12x fix/qsa-draft-reuse-stride b53c5263: rows at the 2051 stride, tail still not attended (c1 = v3d)
#   k67b  dgx-02  b12x exp/qsa-draft-reuse-full-width fea34e1d: 2055-wide sparse_draft program, tail attended (upstream)
#   image   k67/IMAGE-<arm> = shipped v3d image + the changed b12x files (k67/img-a, img-b), built on both Sparks when
#           missing; base files md5-checked = b12x 21e0b201; preflight runs the CPU stride test inside each image
#           (Triton interpreter, no GPU, no network)
#   gpu     after stop_all, gpu_stride_check.py in the served, k67a and k67b images on dgx-01: which keys the sparse
#           program attends for row 1 of a 2-row reuse step (served = shifted, k67a = own, k67b = own + chain); an arm
#           image that reads the wrong set stops the job before the screen
#   screen  thunderdome.sh k67a (dgx-01) k67b (dgx-02), full 4 boots per Spark, bake (own VLLM_CACHE_ROOT
#           /cache/runtime/vllm-k67a|b), hook = k58/hook.sh (count c1/c2 + per-position acceptance), TD_ACC_RISE_OK=1
#   abort   a watcher stops the screen when an arm container serves for 4 min without its "b12x qsa draft reuse" line
#           (fix not live) or with the other variant's line, or an arm boot is not healthy after 30 min while b12x
#           logs "0 cached" selection lines (cold plan compile), or after 50 min in any case
#   post    per arm: fix line in every arm boot, AOT loaded and no measured b12x plans in screen boots, no Traceback,
#           KV blocks = control, min MemAvailable >= 6 GiB; acceptance per position per probe cell (c1/c4/c8, d16k)
#   gate    thunderdome.sh gate <arm>.yaml for each arm that is PROMOTE with post PASS, judged by k58/gate_read.py
#   verdict K67A / K67B = PASS when screen PROMOTE, post PASS and gate PASS
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; comment.md + RESULT for the poster; restores the 2x on exit.
# Queue: after_k66.sh (waits for the k66 runner, then runs backlog/runner.sh with JOBS=k67).
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k67; K=$HOME/GEN-AI/backlog/k67; K58=$HOME/GEN-AI/k58; MEMFLOOR=6; ARMS="k67a k67b"
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/thunderdome-k67-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
CTL=$K58/v3d.yaml; BASE=$(sed -n 's/^container: //p' "$CTL"); P=/usr/local/lib/python3.12/dist-packages/b12x/attention/qsa
MD5_contract=4987e7a4b7df25bc849327ef636ba28b; MD5_draft=d7cbdc97e2bdd4a9a714a6113f5e1502   # b12x 21e0b201
img() { sed -n 's/^container: //p' "$K/$1.yaml"; }
sub() { [ $1 = k67a ] && echo a || echo b; }
fixline() { [ $1 = k67a ] && echo "b12x qsa draft reuse: stride fix, read width 2051" || echo "b12x qsa draft reuse: full width, read width 2055"; }
on() { if [ $1 = $H1 ]; then bash -c "$2"; else ssh -n -o ConnectTimeout=10 $H2 "$2"; fi; }

build() { # both arm images on both Sparks (thin layers over the shipped v3d image)
  local h a i f m ok=0
  for h in $H1 $H2; do
    [ $h = $H2 ] && { ssh -n -o ConnectTimeout=10 $H2 "mkdir -p GEN-AI/backlog/k67" && rsync -a --delete "$K/img-a" "$K/img-b" "$H2:GEN-AI/backlog/k67/" || { echo "rsync to dgx-02 failed"; ok=1; continue; }; }
    for a in $ARMS; do i=$(img $a)
      on $h "docker image inspect $i >/dev/null 2>&1" && { echo "$h: $i present"; continue; }
      for f in _contract.py=$MD5_contract _draft_selection.py=$MD5_draft; do
        m=$(on $h "docker run --rm --network none --entrypoint md5sum $BASE $P/${f%%=*}" | cut -c1-32)
        [ "$m" = "${f#*=}" ] || { echo "$h: $BASE ${f%%=*} is not b12x 21e0b201 ($m)"; ok=1; continue 2; }; done
      on $h "cd ~/GEN-AI/backlog/k67/img-$(sub $a) && nice -n 10 docker build -q -t $i ." >/dev/null || { echo "$h: docker build $i failed"; ok=1; continue; }
      echo "$h: built $i"; done; done
  return $ok; }

preflight() { # out dir
  local ok=0 a
  build || ok=1
  [ "$(md5sum < "$CTL" | cut -c1-32)" = 6d56fbaa10b2793bc1786a96931e8829 ] || { echo "k58/v3d.yaml is not the shipped 1x"; ok=1; }
  for a in $ARMS; do
    [ "$(diff "$CTL" "$K/$a.yaml" | grep -c '^>')" = 4 ] || { echo "$a.yaml differs from v3d beyond header/name/container/cache root"; ok=1; }
    grep -qx "  VLLM_CACHE_ROOT: \"/cache/runtime/vllm-$a\"" "$K/$a.yaml" || { echo "$a.yaml lacks its own VLLM_CACHE_ROOT"; ok=1; }
    grep -qx "recipe: $K/$a.yaml" "$K/$a/thunderdome.arm" || { echo "$a/thunderdome.arm recipe"; ok=1; }
    docker image inspect "$(img $a)" >/dev/null 2>&1 && { timeout -k 30 900 docker run --rm --network=none -e TRITON_INTERPRET=1 \
      -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/pl:/w -v "$G/k18/pylib:/pl:ro" -v "$K/tests-$(sub $a):/w:ro" -w /w --entrypoint python3 "$(img $a)" -m pytest -q \
      -p no:cacheprovider --noconftest tests/attention/test_qsa_draft_reuse_stride_cpu.py > "$1/cputest-$a.txt" 2>&1 \
      && echo "cputest $a: $(tail -1 "$1/cputest-$a.txt")" || { echo "cputest $a FAILED ($1/cputest-$a.txt)"; ok=1; }; }; done
  grep -q ACC_RISE_OK "$R/scripts/thunderdome_report.py" && python3 "$R/scripts/thunderdome_report.py" --selftest >/dev/null \
    || { echo "thunderdome_report.py lacks TD_ACC_RISE_OK (PR #105)"; ok=1; }
  for x in hook_stats kvcap gate_read; do python3 "$K58/$x.py" --selftest >/dev/null || { echo "$x selftest"; ok=1; }; done
  bash -n "$K58/hook.sh" || ok=1
  python3 -m py_compile "$K/gpu_stride_check.py" "$K/acc_by_cell.py" || ok=1
  env RES="$1/dry" GATE=0 CHAIN_NO_RESTORE=1 TD_ACC_RISE_OK=1 SHIPPED_TAG=$E bash "$TD" $(for a in $ARMS; do echo "$K/$a"; done) --dry-run || ok=1
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
  preflight "$T"; rc=$?; cat "$T"/cputest-*.txt 2>/dev/null | tail -4
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "estimate: gpu check ~10 min + screen ~2.7 h (bake ~15 min, 4 boots x ~33 min per Spark incl. hook, both Sparks in parallel) + gate ~35 min per PROMOTE arm + restore ~5 min"
  echo "dry-run exit=$rc"; exit $rc; fi
need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
{ echo "k67 $(TZ=Europe/Bucharest date '+%F %T %Z'): MTP draft-reuse attention width/stride arms vs the shipped v3d (#113)"
  echo "k67a (dgx-01) $(img k67a): stride fix, tail not attended; k67b (dgx-02) $(img k67b): full width, tail attended"
  echo "screen with TD_ACC_RISE_OK=1 (acceptance rises are reported, not a KILL reason; drops stay strict)"; } > "$RES/k67.txt"

st "running: preflight"
preflight "$RES" > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
grep -q "REFUSED" "$RES/preflight.txt" && { FINAL="FAILED: preflight dropped an arm ($RES/preflight.txt)"; exit 1; }
cp -r "$K"/k67a.yaml "$K"/k67b.yaml "$K"/k67a "$K"/k67b "$K"/run.sh "$K"/gpu_stride_check.py "$K"/acc_by_cell.py "$RES/"

st "running: gpu check"
stop_all
GPU=; BADGPU=
for x in served:$BASE k67a:$(img k67a) k67b:$(img k67b); do n=${x%%:*}
  child timeout -k 30 1200 docker run --rm --gpus all --ipc host --network none -e CUTE_DSL_ARCH=sm_121a -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$K:/k:ro" --entrypoint python3 "${x#*:}" /k/gpu_stride_check.py > "$RES/gpucheck-$n.txt" 2>&1; rc=$?
  log "gpu check $n: exit $rc"; GPU="$GPU $n=$rc"
  [ $n != served ] && [ $rc = 3 ] && BADGPU="$BADGPU $n"; done
{ echo; echo "== GPU stride check (gpu_stride_check.py, dgx-01; exit 0 = the variant's expected read set, 3 = another set, other = error)"
  for n in served k67a k67b; do echo "-- $n"; tail -4 "$RES/gpucheck-$n.txt"; done; } >> "$RES/k67.txt"
[ -z "$BADGPU" ] || { FINAL="FAILED: GPU stride check, wrong read set in$BADGPU ($RES/k67.txt)"; exit 1; }

st "running: screen"
env RES="$RES" GPU_LOCK_HELD=1 CHAIN_NO_RESTORE=1 GATE=0 TD_ACC_RISE_OK=1 SHIPPED_TAG=$E timeout -k 600 32400 bash "$TD" $(for a in $ARMS; do echo "$K/$a"; done) > "$RES/screen.nohup" 2>&1 < /dev/null & CH=$!
( declare -A up0 boot0 cid   # timers per arm container (docker ID: arm:1 and arm:2 boot back to back)
  while kill -0 $CH 2>/dev/null; do
    for a in $ARMS; do h=$([ $a = k67a ] && echo $H1 || echo $H2); i=$(img $a)
      c=$(on $h "docker ps --format '{{.ID}} {{.Image}}' | awk '\$2 == \"$i\" {print \$1; exit}'")
      [ -n "$c" ] || continue
      [ "$c" = "${cid[$a]:-}" ] || { cid[$a]=$c; boot0[$a]=$(date +%s); up0[$a]=; }
      L=$(on $h "docker exec $c grep -aoE 'b12x qsa draft reuse: [a-z ]+, read width [0-9]+' /tmp/sparkrun_serve.log 2>/dev/null | head -1" 2>/dev/null | grep -aoE '^b12x qsa draft reuse: [a-z ]+, read width [0-9]+$')   # docker exec errors (container stopping) are not lines
      if [ -n "$L" ] && [ "$L" != "$(fixline $a)" ]; then echo "$a: wrong variant line '$L'" > "$RES/abort-wrongpath"; fi
      if [ "$(health $h)" = 200 ]; then up0[$a]=${up0[$a]:-$(date +%s)}
        [ -z "$L" ] && [ $(( $(date +%s) - ${up0[$a]} )) -gt 240 ] && echo "$a: serving 4 min without '$(fixline $a)'" > "$RES/abort-wrongpath"
      else t=$(( $(date +%s) - ${boot0[$a]} ))
        [ $t -gt 1800 ] && on $h "docker exec $c grep -aqE 'b12x [a-z ]+ [a-z_.]+: [0-9]+/[0-9]+ ready.*, 0 cached, [1-9][0-9]* compilations' /tmp/sparkrun_serve.log" \
          && echo "$a: not healthy after $t s with b12x '0 cached, N compilations' (cold plan compile)" > "$RES/abort-cold"
        [ $t -gt 3000 ] && echo "$a: not healthy after $t s" > "$RES/abort-cold"; fi; done
    f=$(ls "$RES"/abort-wrongpath "$RES"/abort-cold 2>/dev/null | head -1)
    [ -n "$f" ] && { kill -TERM $CH; break; }; sleep 15; done ) & WD=$!
wait $CH; rc=$?; CH=; kill $WD 2>/dev/null
log "screen exit=$rc STATE: $(head -1 "$RES/STATE" 2>/dev/null)"
for f in "$RES"/abort-wrongpath "$RES"/abort-cold; do [ -e "$f" ] && { FINAL="FAILED: $(cat "$f"); screen stopped"; exit 1; }; done

blocks() { grep -h '^kvcap num_gpu_blocks=' "$1"/hook/kvcap.txt 2>/dev/null | head -1 | sed 's/.*=//'; }
V=; PROM=
for x in $ARMS; do
  o=$RES/$x; v=$(sed -n 's/^VERDICT=//p' "$o/verdict.txt" 2>/dev/null); v=${v:-ERROR}; p=; n=0
  for s in "$o"/arm-p?/serve.log "$o"/bake-arm/serve.log; do [ -e "$s" ] || continue; n=$((n + 1))
    grep "Traceback" "$s" | grep -vq "triton_bundler.py" && p="$p; Traceback in $s"; done
  for s in "$o"/ctl-p?/serve.log; do grep -q "b12x qsa draft reuse" "$s" 2>/dev/null && p="$p; control logged a fix line ($s)"; done
  for s in "$o"/arm-p?/serve.log; do [ -e "$s" ] || continue
    grep -qF "$(fixline $x)" "$s" || p="$p; no '$(fixline $x)' in $s"
    [ "$(grep -m1 -oE 'Directly load AOT compilation|Dynamo bytecode transform time' "$s")" = "Directly load AOT compilation" ] || p="$p; backbone not loaded from AOT in $s"
    grep -oE "b12x ready [a-z_.]+: [0-9]+/[0-9]+ ready, [0-9]+ measured" "$s" | grep -vq ", 0 measured" && p="$p; b12x measured plans in $s"; done
  [ $n -gt 0 ] || p="$p; no arm boot logs"
  cb=$(blocks "$o/ctl-p1"); for b in "$o"/arm-p?; do ab=$(blocks "$b"); [ -n "$ab" ] && [ "$ab" = "$cb" ] || p="$p; num_gpu_blocks $(basename "$b") ${ab:-?} vs control ${cb:-?}"; done
  m=$(cat "$o"/arm-p?/mem.log "$o"/bake-arm/mem.log 2>/dev/null | awk '$2 ~ /^[0-9.]+$/' | sort -k2 -n | head -1 | awk '{print $2}')
  awk -v m="${m:-0}" -v f=$MEMFLOOR 'BEGIN{exit !(m >= f)}' || p="$p; min MemAvailable ${m:-?} GiB < $MEMFLOOR"
  sp=$([ $x = k67a ] && echo dgx-01 || echo dgx-02)
  [ $v = PROMOTE ] && [ -z "$p" ] && PROM="$PROM $x"
  printf -v "WHY_$x" %s "$([ $v = PROMOTE ] || echo "; $sp $v ($(grep -m3 '^reason:' "$o/verdict.txt" 2>/dev/null | sed 's/^reason: //' | tr '\n' ','))")${p:+; post [${p#; }]}"
  { echo; echo "== $x ($sp) vs v3d: screen=$v post=[${p:-PASS}] min_MemAvailable=${m:-?}GiB"
    echo "-- b12x ready lines (arm boots incl. bake)"; grep -hoE "b12x ready [a-z_.]+: [0-9]+/[0-9]+ ready, [0-9]+ measured, [0-9]+ cached" "$o"/arm-p?/serve.log "$o"/bake-arm/serve.log 2>/dev/null | sort | uniq -c
    sed -n '/^== acceptance/,$p' "$o/verdict.txt" 2>/dev/null
    echo "-- acceptance per draft position per probe cell (pooled boots; positions 1-3 are the reuse steps, 0 is the anchor run)"
    echo "control (v3d):"; python3 "$K/acc_by_cell.py" "$o"/ctl-p? 2>&1 | sed 's/^/  /'
    echo "arm ($x):"; python3 "$K/acc_by_cell.py" "$o"/arm-p? 2>&1 | sed 's/^/  /'
    echo "-- single-request hook cells (arm vs control, same pass; count c1 vs c2 = same prompt at two concurrencies)"
    python3 "$K58/hook_stats.py" --compare "$o" 2>&1; } >> "$RES/k67.txt"
  log "$x ($sp): screen=$v post=[${p:-PASS}]"; V="$V $sp-$x=$v"
done
for l in dgx01 dgx02; do echo "guard $l: min MemAvailable over the screen $(awk '$2 ~ /^[0-9]+$/' "$RES/guard-$l.log" | sort -k2 -n | head -1 | awk '{printf "%.2f GiB", $2/1024}')"; done >> "$RES/k67.txt"

G=
for x in $PROM; do
  st "running: gate $x"
  child env RES="$RES" GPU_LOCK_HELD=1 CHAIN_NO_RESTORE=1 SHIPPED_TAG=$E timeout -k 600 14400 bash "$TD" gate "$K/$x.yaml" > "$RES/gate-$x.nohup" 2>&1 < /dev/null
  log "gate $x exit=$? STATE: $(head -1 "$RES/STATE" 2>/dev/null)"
  f=$(ls "$RES"/gate-*"$x"*/summary.txt 2>/dev/null | head -1)
  g=$([ -n "$f" ] && python3 "$K58/gate_read.py" "$f" || echo "FAIL (no summary)")
  { echo; echo "== gate $x: $g"; [ -n "$f" ] && cat "$f"; } >> "$RES/k67.txt"; log "gate $x: $g"; G="$G $x=${g%% *}"
  w=WHY_$x; case $g in PASS*) ;; *) printf -v "WHY_$x" %s "${!w}; gate $g" ;; esac; done
for x in $ARMS; do w=WHY_$x; w=${!w}; case " $PROM " in *" $x "*) ;; *) [ -n "$w" ] || w="; not gated" ;; esac
  printf -v "K_$x" %s "$([ -z "$w" ] && echo PASS || echo "FAIL (${w#; })")"; done
{ echo; echo "K67A=$K_k67a"; echo "K67B=$K_k67b"; } >> "$RES/k67.txt"
FINAL="DONE: k67$V gates=[${G# }] K67A=${K_k67a%% *} K67B=${K_k67b%% *} ($RES/k67.txt)"
{ echo "k67 result (#113): MTP draft-reuse attention arms vs the shipped v3d, full Thunderdome screen (4 boots, \`TD_ACC_RISE_OK=1\`), k67a on dgx-01 (stride fix: reuse rows at the 2051 stride the program reads, chain tail still not attended), k67b on dgx-02 (full width: 2055-column sparse_draft program, chain tail attended); the gate for each PROMOTE arm."
  echo; echo "**K67A=$K_k67a**"; echo; echo "**K67B=$K_k67b**"; echo
  echo '```'; sed -n '4,$p' "$RES/k67.txt" | grep -vE '^(  self|  cross)' | head -c 55000; echo '```'
  echo; echo "Raw data on dgx-01: \`$RES\`."; } > "$K/comment.md"
