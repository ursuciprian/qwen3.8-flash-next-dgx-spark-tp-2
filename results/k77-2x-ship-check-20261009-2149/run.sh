#!/usr/bin/env bash
# k77-2x-ship-check (2026-10-09, #123): ship check of the 2x v2.0.0 image (k73-2x-gdnmse-dispatch, arm c41).
#   image   base = the k73 warm image spark-vllm-b12x:k73-gdnmse-tp2-21e0b201-d21d7ade-warm (plan 6fdfaa4c, 682 records,
#           torch AOT of both ranks); CI build-b0-warm (ursuciprian/spark-vllm-b12x) on top with recipe_ref
#           feat/k73-2x-gdnmse-default pushes TAG (the release tag 2x-v2.0.0 is added after a pass, outside this job). STATE ci-open: the agent triggers CI from the Mac and writes
#           k77/ci-done ("run <id> ok|failed <digest>"), cap 2 h. The CI runner refuses while dgx-01 serves, so the pair
#           stops for the window.
#   check   first boot of TAG with the branch recipe (plain sparkrun run of the file) from a clean runtime cache: the
#           6fdfaa4c plan file and k73's 9 AOT keys are moved out of every runtime cache on both Sparks first (moved/,
#           put back at exit unless the boot recreated them). Pass = up, pong, scripts/check_seed.py (0 measured on the
#           rank-0 log, both ranks' plan file still 682 records), AOT loaded (no Dynamo compile) on rank 0, MXFP8 copies
#           on both ranks, fresh-c4 T=0 acceptance per position within 0.03 of k73's c41 boots, llama-benchy tg512 c1
#           and c8 not below k73's c41 mean by more than k73's noise band (12.3% / 11.3%), and tg512 c16 not below the
#           shipped 2x's last c16 cell (no c16 A/B in k73; v1.5.0 has none): b1.4 llama-benchy task mode d0 c16 242.82 t/s
#           (results/b1.4-20261001/b14.json, mean of 2 boots 241.34 / 244.31, sd 4.55 / 1.63) by more than
#           max(3%, that run's spread 1.9%) = 3% (coordinator rule 2026-10-09).
#   end     pass: lib.sh SHIPPED_TAG_DEFAULT -> the new tag (backup lib.sh.bak-<time>-pre-k77) and E = the new tag, so
#           lib.sh restore keeps the new default serving (no-op); fail: lib.sh restores v3.1.0 by name with a pong.
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k77; K=$HOME/GEN-AI/backlog/k77
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k77-2x-ship-check-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
BASE=spark-vllm-b12x:k73-gdnmse-tp2-21e0b201-d21d7ade-warm
TAG=ghcr.io/ursuciprian/spark-vllm-b12x:k73-20261009-21e0b201-d21d7ade-warm
BRANCH=feat/k73-2x-gdnmse-default
REC=$K/recipe/qwen3.8-flash-next-2x-dgx-spark.yaml
SEED=6fdfaa4cdd6c827f17390e8fa20bc97a36d7be3200a2d97ceb73f7b0577c050e; NREC=682
K73=$R/results/k73-tp2-gdnmse-dispatch-20261009-1621
KEYS=$(ls "$K73/img-warm/torch_aot_compile" 2>/dev/null | tr '\n' ' ')
SNAP=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
GSNAP=$HOME/.cache/huggingface/hub/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE/snapshots/16c9bd54788d12390838a65ce4a4ecda97fa5f1d
CORPUS=$R/results/corpus-code.txt; BENCHY_SRC=$G/llama-benchy-fork; MODEL=qwen3.8-flash-next

pos() { curl -s -m 10 "localhost:8000/metrics" | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens(_per_pos)?_total'; }
acc() { python3 - "$@" <<'EOF'
import re, sys
def rd(p):
    c = {"accepted": {}, "draft": {}}
    for l in open(p):
        m = re.match(r'vllm:spec_decode_num_(accepted|draft)_tokens_per_pos_total\{.*position="(\d+)"\} ([0-9.e+]+)', l)
        if m:
            k = int(m[2]); c[m[1]][k] = c[m[1]].get(k, 0) + float(m[3])
    return c["accepted"], c["draft"]
a, d = {}, {}
for b, e in zip(sys.argv[1::2], sys.argv[2::2]):   # pairs before/after, pooled
    (a0, d0), (a1, d1) = rd(b), rd(e)
    for k in d1:
        a[k] = a.get(k, 0) + a1[k] - a0.get(k, 0); d[k] = d.get(k, 0) + d1[k] - d0.get(k, 0)
print(" ".join(f"{a[k] / max(d[k], 1):.3f}" for k in sorted(d)))
EOF
}
tgs() { # task.csv (llama-benchy markdown) test-name -> mean t/s (total) over the given files
  python3 - "$@" <<'EOF'
import sys
t, v = sys.argv[1], []
for f in sys.argv[2:]:
    for l in open(f):
        c = [x.strip() for x in l.split("|")]
        if len(c) > 3 and c[2] == t:
            v.append(float(c[3].split()[0]))
print(f"{sum(v) / len(v):.1f}" if v else "nan")
EOF
}
# runtime-cache surgery on one host (runs locally or over ssh): move the k73 AOT keys and the plan file out / put back
CLEAN="RC=\$HOME/.cache/sparkrun/runtime-cache/vllm; M=\$HOME/GEN-AI/backlog/k77/moved; mkdir -p \$M; [ -z \"\$(ls -A \$M)\" ] || { echo 'moved/ not empty'; exit 1; }
for d in \$RC/*/; do n=\$(basename \$d); a=\$d/vllm/torch_compile_cache/torch_aot_compile; p=\$d/b12x/compile/preparation
  for k in $KEYS; do [ -d \$a/\$k ] && { mkdir -p \$M/\$n/aot && mv \$a/\$k \$M/\$n/aot/ || exit 1; }; done
  [ -e \$p/$SEED.json ] && { mkdir -p \$M/\$n/plan && mv \$p/$SEED.json \$M/\$n/plan/ || exit 1; }; rm -f \$p/$SEED.lock; done
echo \"\$(hostname): moved \$(find \$M -mindepth 3 -maxdepth 3 | wc -l) entries\""
PUTBACK="RC=\$HOME/.cache/sparkrun/runtime-cache/vllm; M=\$HOME/GEN-AI/backlog/k77/moved; [ -d \$M ] || exit 0
for n in \$(ls \$M); do a=\$RC/\$n/vllm/torch_compile_cache/torch_aot_compile; p=\$RC/\$n/b12x/compile/preparation
  for k in \$(ls \$M/\$n/aot 2>/dev/null); do if [ -d \$a/\$k ]; then rm -rf \$M/\$n/aot/\$k; else mkdir -p \$a && mv \$M/\$n/aot/\$k \$a/; fi; done
  for f in \$(ls \$M/\$n/plan 2>/dev/null); do if [ -s \$p/\$f ]; then rm -f \$M/\$n/plan/\$f; else mkdir -p \$p && mv \$M/\$n/plan/\$f \$p/; fi; done; done
find \$M -depth -type d -empty -delete; echo \"\$(hostname): put back, \$(find \$M -type f 2>/dev/null | wc -l) files left in moved/\""
clean_host() { bash -c "$CLEAN" && ssh -n -o ConnectTimeout=10 $H2 "$CLEAN"; }
putback() { [ -n "${CLEANED:-}" ] || return 0; bash -c "$PUTBACK"; ssh -n -o ConnectTimeout=10 $H2 "$PUTBACK"; }

base_ok() { local o; o=$(docker run --rm --network none --entrypoint bash "$BASE" -c "python3 -c 'import json; print(len(json.load(open(\"/opt/b12x-seed/preparation/$SEED.json\"))[\"records\"]))'; n=0; for k in $KEYS; do [ -d /opt/b12x-seed/torch_compile_cache/torch_aot_compile/\$k/rank_0_0 ] && [ -d /opt/b12x-seed/torch_compile_cache/torch_aot_compile/\$k/rank_1_0 ] && n=\$((n + 1)); done; echo \$n" | tr '\n' ' '); [ "$o" = "$NREC 9 " ]; }
hostchk() { local c="ls $GSNAP/model.safetensors.index.json && test -s $SNAP/model-00035-of-00036.safetensors && { [ ! -d $K/moved ] || [ -z \"\$(ls -A $K/moved)\" ]; }"
  if [ "$1" = local ]; then bash -c "$c"; else ssh -n -o ConnectTimeout=10 "$1" "$c"; fi; }
if [ "${1:-}" = --dry-run ]; then
  rc=0; ok() { if eval "$2" >/dev/null 2>&1; then echo "ok   $1"; else echo "FAIL $1"; rc=1; fi; }
  ok "syntax" "bash -n $0"
  ok "recipe container $TAG" "grep -qx 'container: $TAG' $REC"
  ok "recipe serves GDN-MSE @ 16c9bd54 with cutoff 41" "grep -q -- '--revision 16c9bd54788d12390838a65ce4a4ecda97fa5f1d' $REC && grep -q 'VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS: \"41\"' $REC && grep -q 'model-00035-of-00036.safetensors' $REC"
  ok "branch $BRANCH has the seed (GitHub)" "curl -sfI -m 20 https://raw.githubusercontent.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/$BRANCH/docker/b0-warm/preparation/$SEED.json"
  ok "base $BASE: seed $NREC records + 9 k73 AOT keys (both ranks)" base_ok
  ok "k73 AOT key list (9)" "[ \$(echo $KEYS | wc -w) = 9 ]"
  ok "dgx-01: GDN-MSE @ 16c9bd54, 7c4f1bc1 shard 35, moved/ empty" "hostchk local"
  ok "dgx-02: GDN-MSE @ 16c9bd54, 7c4f1bc1 shard 35, moved/ empty" "hostchk $H2"
  ok "k73 c41 fresh-c4 pos + task.csv" "[ -s $K73/screen/c41-p1/pos-fresh-c4.after ] && [ -s $K73/screen/c41-p2/pos-fresh-c4.after ] && [ -s $K73/screen/c41-p1/task.csv ] && [ -s $K73/screen/c41-p2/task.csv ]"
  ok "lib.sh SHIPPED_TAG_DEFAULT line" "grep -c '^SHIPPED_TAG_DEFAULT=' $BL/lib.sh | grep -qx 1"
  ok "tools" "command -v sparkrun && command -v uvx && [ -d $BENCHY_SRC ] && [ -s $CORPUS ] && python3 $R/scripts/check_seed.py --selftest && [ -s $R/scripts/depth_decode_probe.py ]"
  echo "k73 c41 fresh-c4 acc: $(acc $K73/screen/c41-p1/pos-fresh-c4.before $K73/screen/c41-p1/pos-fresh-c4.after $K73/screen/c41-p2/pos-fresh-c4.before $K73/screen/c41-p2/pos-fresh-c4.after)"
  echo "k73 c41 tg512: c1 $(tgs 'tg512 (c1)' $K73/screen/c41-p1/task.csv $K73/screen/c41-p2/task.csv) c8 $(tgs 'tg512 (c8)' $K73/screen/c41-p1/task.csv $K73/screen/c41-p2/task.csv) t/s"
  others_busy && echo "other GPU jobs: busy" || echo "other GPU jobs: clear"
  echo "plan: stop 2x -> CI window (agent, cap 2 h) -> clean runtime caches -> check boot (~5 min) -> fresh-c4 + benchy c1/c8/c16 (~15 min)"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is queued or running: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
trap 'putback >> "$RES/$J.log" 2>&1; kill $G1 $G2 2>/dev/null; restore || { sleep 60; restore; } || FINAL="FAILED: restore -- $FINAL"; st "$FINAL"' EXIT
echo "$RES" > "$K/RESULT"; : > "$K/comment.md"; cp "$0" "$REC" "$RES/"
st "running: stop 2x"; stop_all

# ---- CI window ----
rm -f "$K/ci-done"; echo "$BASE $TAG recipe_ref=$BRANCH" > "$K/ci-request"; st "ci-open"
s=$(date +%s); until [ -f "$K/ci-done" ] || [ $(( $(date +%s) - s )) -gt 7200 ]; do sleep 20; done
log "ci window closed: $(cat "$K/ci-done" 2>/dev/null || echo timeout)"; rm -f "$K/ci-request"; cp "$K/ci-done" "$RES/ci-done.txt" 2>/dev/null
grep -q ' ok ' "$K/ci-done" 2>/dev/null || { FINAL="FAILED: CI ($(cat "$K/ci-done" 2>/dev/null || echo timeout))"; exit 1; }
{ docker image inspect -f 'TAG id {{.Id}}' "$TAG"
  docker run --rm --network none --entrypoint bash "$TAG" -c "ls -l /opt/b12x-seed/preparation/$SEED.json /opt/mtp-vocab; ls /opt/b12x-seed/torch_compile_cache/torch_aot_compile | wc -l"
} > "$RES/image-check.txt" 2>&1
grep -q "$SEED.json" "$RES/image-check.txt" || { FINAL="FAILED: $TAG lacks the seed ($RES/image-check.txt)"; exit 1; }
st "running: dgx-02 pull"
ssh -n -o ConnectTimeout=10 $H2 "timeout 2400 docker pull -q $TAG" >> "$RES/$J.log" 2>&1 && log "dgx-02 pulled $TAG" || log "dgx-02 pull failed; sparkrun copies the image from the head"

# ---- first-boot check ----
st "running: check boot"; D=$RES/check; mkdir -p "$D"; cp "$REC" "$D/recipe.yaml"
CLEANED=1; clean_host >> "$RES/$J.log" 2>&1 || { FINAL="FAILED: runtime cache clean"; exit 1; }
( cd "$D" && timeout -k 60 900 sparkrun run recipe.yaml --no-follow ) >> "$D/sparkrun.log" 2>&1 < /dev/null 9>&-
s=$(date +%s); up=0
while :; do
  n=$(n0); [ "$(health localhost)" = 200 ] && { up=1; break; }
  if [ $(( $(date +%s) - s )) -gt 180 ] && { [ -z "$n" ] || docker exec "$n" grep -qE 'Worker failed with error|EngineCore failed to start|Engine core initialization failed' /tmp/sparkrun_serve.log 2>/dev/null; }; then break; fi
  [ $(( $(date +%s) - s )) -ge 3600 ] && break; sleep 10; done
echo "boot $(( $(date +%s) - s ))s up=$up" > "$D/boot.txt"; log "check boot: $(cat "$D/boot.txt")"
c=$(docker ps -a --format '{{.Names}}' | grep node_0 | head -1)
docker exec "$c" cat /tmp/sparkrun_serve.log > "$D/serve-node0.log" 2>&1 || docker logs "$c" > "$D/serve-node0.log" 2>&1
ssh -n $H2 'c=$(docker ps -a --format "{{.Names}}" | grep node_1 | head -1); docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1' > "$D/serve-node1.log" 2>&1
[ $up = 1 ] || { FINAL="FAILED: check boot ($(cat "$D/boot.txt"))"; exit 1; }
P=$(pong localhost)
RC0=$(docker inspect "$c" --format '{{range .Mounts}}{{if eq .Destination "/cache/runtime"}}{{.Source}}{{end}}{{end}}')
RC1=$(ssh -n $H2 'c=$(docker ps --format "{{.Names}}" | grep node_1 | head -1); docker inspect $c --format "{{range .Mounts}}{{if eq .Destination \"/cache/runtime\"}}{{.Source}}{{end}}{{end}}"')
scp -q "$H2:$RC1/b12x/compile/preparation/$SEED.json" "$D/plan-node1.json" || echo '{"records": {}}' > "$D/plan-node1.json"
python3 "$R/scripts/check_seed.py" "$D/serve-node0.log" "$RC0/b12x/compile/preparation/$SEED.json:$NREC" "$D/plan-node1.json:$NREC" > "$D/plans-check.txt" 2>&1 && SEEDOK=1 || SEEDOK=0
AOT=$(grep -c 'Directly load AOT compilation' "$D/serve-node0.log"); DYN=$(grep -c 'Dynamo bytecode transform time' "$D/serve-node0.log")
CP0=$(grep -c 'MXFP8 copy serves rows >= 41' "$D/serve-node0.log"); CP1=$(grep -c 'MXFP8 copy serves rows >= 41' "$D/serve-node1.log")
{ echo "image node 0 $(docker inspect --format '{{.Config.Image}}' "$c"); node 1 $(ssh -n $H2 "docker ps --format '{{.Image}}' --filter name=sparkrun")"
  echo "image id $(docker image inspect -f '{{.Id}}' "$TAG"), node 1 $(ssh -n $H2 "docker image inspect -f '{{.Id}}' $TAG")"
  echo "$(cat "$D/boot.txt"), pong=$P"
  echo "runtime caches: node 0 $RC0, node 1 $RC1"
  echo "rank 0: AOT loaded $AOT, Dynamo compiles $DYN; MXFP8 copies node 0 $CP0, node 1 $CP1"
  grep -m1 -E 'GPU KV cache size' "$D/serve-node0.log" | sed 's/^.*GPU KV/GPU KV/'
  echo "check_seed: $(tail -1 "$D/plans-check.txt")"
  echo "moved entries: $(grep -E 'moved [0-9]+ entries' "$RES/$J.log" | sed 's/^.*] //' | tr '\n' ' ')"
} > "$D/check.txt" 2>&1
log "check: seed=$SEEDOK aot=$AOT dyn=$DYN copies=$CP0/$CP1 pong=$P"

st "running: probe"
pos > "$D/pos-fresh-c4.before"
( cd "$R" && child timeout -k 30 400 python3 scripts/depth_decode_probe.py --base "http://$H1:8000" --label fresh-c4 --corpus "$CORPUS" \
    --concurrency 4 --repeats 2 --temperature 0 --depth 256 --new 2048 --offset 314187 --max-tokens 512 \
    --out "$D/probe-fresh-c4.json" > "$D/probe-fresh-c4.log" 2>&1 )
pos > "$D/pos-fresh-c4.after"
A=$(acc "$D/pos-fresh-c4.before" "$D/pos-fresh-c4.after")
KA=$(acc $K73/screen/c41-p1/pos-fresh-c4.before $K73/screen/c41-p1/pos-fresh-c4.after $K73/screen/c41-p2/pos-fresh-c4.before $K73/screen/c41-p2/pos-fresh-c4.after)
ACCOK=$(python3 -c "import sys; a,k=(list(map(float,s.split())) for s in sys.argv[1:3]); print(1 if len(a)>=4 and all(abs(x-y)<=0.03 for x,y in list(zip(a,k))[:4]) else 0)" "$A" "$KA" 2>/dev/null || echo 0)
{ echo "fresh-c4 T=0 acceptance per position: this boot $A"; echo "k73 c41 (2 boots pooled): $KA"; echo "within 0.03 at positions 0-3: $ACCOK"
  echo "probe this boot:"; tail -4 "$D/probe-fresh-c4.log"; echo "probe k73 c41-p1:"; tail -4 "$K73/screen/c41-p1/probe-fresh-c4.log"; } > "$D/drafter-check.txt" 2>&1
mkdir -p "$RES/bench"
( cd "$R" && child timeout -k 30 1800 uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$H1:8000/v1" --model $MODEL \
    --tokenizer "$SNAP" --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 --concurrency 1 8 16 --runs 4 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$H1:8000/metrics" \
    --save-result "$RES/bench/task.csv" > "$RES/bench/benchy.log" 2>&1 ); log "benchy exit=$?"
[ "$(health localhost)" = 200 ] || { FINAL="FAILED: server died during the check bench"; exit 1; }
T1=$(tgs 'tg512 (c1)' "$RES/bench/task.csv"); T8=$(tgs 'tg512 (c8)' "$RES/bench/task.csv"); T16=$(tgs 'tg512 (c16)' "$RES/bench/task.csv"); PP=$(tgs 'pp2048 (c1)' "$RES/bench/task.csv")
K1=$(tgs 'tg512 (c1)' $K73/screen/c41-p1/task.csv $K73/screen/c41-p2/task.csv); K8=$(tgs 'tg512 (c8)' $K73/screen/c41-p1/task.csv $K73/screen/c41-p2/task.csv)
TGOK=$(python3 -c "import sys; t1,t8,k1,k8=map(float,sys.argv[1:5]); print(1 if t1>=k1*(1-0.123) and t8>=k8*(1-0.113) else 0)" "$T1" "$T8" "$K1" "$K8" 2>/dev/null || echo 0)
C16REF=242.82; C16OK=$(python3 -c "import sys; t,r=map(float,sys.argv[1:3]); print(1 if t>=r*(1-0.03) else 0)" "$T16" "$C16REF" 2>/dev/null || echo 0)
echo "llama-benchy pp2048/tg512 d0 T=1 (mean of 4 runs): pp2048 c1 $PP, tg512 c1 $T1 (k73 c41 $K1), c8 $T8 (k73 c41 $K8), c16 $T16 (b1.4 c16 $C16REF, noise 3%) t/s; c1/c8 within k73 noise: $TGOK, c16 not worse beyond noise: $C16OK" > "$D/bench-check.txt"
log "$(cat "$D/bench-check.txt")"
if [ "$SEEDOK" = 1 ] && [ "$AOT" -ge 1 ] && [ "$DYN" = 0 ] && [ "$CP0" -ge 1 ] && [ "$CP1" -ge 1 ] && echo "$P" | grep -qi pong && [ "$ACCOK" = 1 ] && [ "$TGOK" = 1 ] && [ "$C16OK" = 1 ]; then
  NEWE=${TAG##*:}; OLDE=$SHIPPED_TAG_DEFAULT; B=$BL/lib.sh.bak-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)-pre-k77
  cp -p "$BL/lib.sh" "$B" && sed -i "s|^SHIPPED_TAG_DEFAULT=.*|SHIPPED_TAG_DEFAULT=$NEWE; E=\$SHIPPED_TAG_DEFAULT   # 2x v2.0.0 since 2026-10-09 (k77, #123); was $OLDE|" "$BL/lib.sh"
  if bash -n "$BL/lib.sh" && grep -qx "SHIPPED_TAG_DEFAULT=$NEWE; E=\$SHIPPED_TAG_DEFAULT   # 2x v2.0.0 since 2026-10-09 (k77, #123); was $OLDE" "$BL/lib.sh"; then
    E=$NEWE; log "lib.sh: SHIPPED_TAG_DEFAULT $OLDE -> $NEWE (backup $B)"
    FINAL="DONE: k77 PASS: 0 measured on both ranks, AOT warm, acc $A, tg512 c1 $T1 c8 $T8 c16 $T16, $TAG serving ($RES)"
  else cp -p "$B" "$BL/lib.sh"; FINAL="FAILED: k77 lib.sh update (restored from $B)"; fi
else FINAL="FAILED: k77 check (seed $SEEDOK, AOT $AOT/dyn $DYN, copies $CP0/$CP1, pong=$P, acc $ACCOK ($A), tg512 $TGOK ($T1/$T8), c16 $C16OK ($T16 vs $C16REF))"; fi
{ echo "k77-2x-ship-check: I booted the pushed v2.0.0 image \`$TAG\` with the branch recipe on the pair, from runtime caches without its plan file and AOT keys, the way a fresh install starts."
  echo; echo "- $(cat "$D/boot.txt"), pong: $P"; echo "- check_seed: $(tail -1 "$D/plans-check.txt") ($NREC records on both ranks); rank 0 loaded the AOT cache ($AOT loads, $DYN Dynamo compiles); MXFP8 copies on both ranks ($CP0 / $CP1)"
  echo "- fresh-c4 T=0 acceptance per position $A (k73 c41: $KA)"
  echo "- llama-benchy pp2048/tg512 at depth 0, T=1, mean of 4 runs: pp2048 c1 $PP, tg512 c1 $T1 (k73 c41 $K1), c8 $T8 (k73 c41 $K8), c16 $T16 (b1.4 c16 $C16REF, noise 3%) t/s"
  echo; echo "Result: ${FINAL%% (*}"; } > "$K/comment.md"
exit 0
