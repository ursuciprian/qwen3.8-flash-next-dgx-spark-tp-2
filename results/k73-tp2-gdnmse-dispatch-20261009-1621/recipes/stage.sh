#!/usr/bin/env bash
# k73 staging (2026-10-08, #123): GDN-MSE with the M-dispatch on the 2x (TP=2). Idempotent, CPU only, no lock, safe
# next to a running job (one small image layer per Spark, nice 19). run.sh calls it in preflight and in its dry run.
#   source   vLLM ursuciprian/vllm feat/k73-mxfp8-large-m-tp d21d7ade = exp/v3d 5dad364d + per-rank MXFP8 large-M
#            copies for TP>1 (one file: vllm/model_executor/kernels/linear/nvfp4/b12x.py)
#   image    P1 = the 1x v3e image (vLLM 5dad364d, b12x 21e0b201, mtp-vocab) + that file + the plan seed, both Sparks
#   seed     the shipped b1.6 control's runtime plan file (15b63901, 616 records, equal on both Sparks) re-keyed to the
#            arm's served path (k48/remap_seed.py): every plan key the arm shares with b1.6 starts on b1.6's selection,
#            so the bake measures only the GDN plans b1.6 does not have (W4A16 NVFP4 + MXFP8 large-M regimes)
#   recipes  arm/k73-<arm>.yaml = the registry 2x recipe (b1.6, md5 e39fcf1e) with model GDN-MSE @ 16c9bd54 (HF id
#            + --revision, no overlay: 16c9bd54 already carries the retrained drafter), the AOT seed copy line of the
#            1x recipe, and VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS (c26 / c41) + _CHECKPOINT (7c4f1bc1 under /cache);
#            c26x / c41x add VLLM_GDN_COMPACT_RECORDS=1 + VLLM_GDN_SHARED_PREFILL_STAGING=1. Container = P1 (run.sh
#            swaps in the warm image after the bake)
#   tests    CPU pytest of the large-M tests (incl. fake TP=2) inside P1 (--network none, no GPU, 4 GiB cap)
set -eu
K=$HOME/GEN-AI/backlog/k73; G=$HOME/GEN-AI; H2=<cx7-ip-b>
BASE=ghcr.io/ursuciprian/spark-vllm-b12x:tp1-v3e-hf-20261008-21e0b201-5dad364d-warm
P1=spark-vllm-b12x:k73-gdnmse-tp2-21e0b201-d21d7ade
REV=16c9bd54788d12390838a65ce4a4ecda97fa5f1d
MXREV=7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
ARMPATH=/cache/huggingface/hub/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE/snapshots/$REV
HUB=$HOME/.cache/huggingface/hub
HF=$HUB/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE/snapshots/$REV
MX=$HUB/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/$MXREV
RC=$HOME/.cache/sparkrun/runtime-cache/vllm
CTLF=$RC/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-2d9615ab/b12x/compile/preparation/15b639016c932b03d623d76be56a320a94946895d5522c30e3e1d8829572c729.json
REG=$HOME/.cache/sparkrun/registries/_url_e47b9e8b5e91/recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml
VF=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/nvfp4/b12x.py
BASE_BLOB=bc752acf1dab89f713f14d60c75043d9b56f776d   # 5dad364d:vllm/.../nvfp4/b12x.py
NEW_BLOB=13e1f86e70bd4566b54be1519bc8242e31d544ae    # d21d7ade:vllm/.../nvfp4/b12x.py
mkdir -p "$K/arm" "$K/img/preparation"

# ---- snapshots on both Sparks ----
SNAPSH='set -e; H='$HF'; M='$MX'
for i in $(seq -f %05g 1 36); do test -s $H/model-$i-of-00036.safetensors || { echo "missing $H/model-$i"; exit 1; }; done
for f in config.json model.safetensors.index.json tokenizer.json; do test -s $H/$f || { echo "missing $H/$f"; exit 1; }; done
test -s $M/model-00035-of-00036.safetensors && test -s $M/model.safetensors.index.json || { echo "missing MXFP8 shard 35 of 7c4f1bc1"; exit 1; }
echo "$(hostname): GDN-MSE @ 16c9bd54 complete, 7c4f1bc1 shard 35 present"'
bash -c "$SNAPSH"; ssh -n -o ConnectTimeout=10 $H2 "$SNAPSH"

# ---- source file ----
[ "$(git hash-object "$K/src/b12x.py")" = $NEW_BLOB ] || { echo "src/b12x.py is not d21d7ade's"; exit 1; }
b=$(docker run --rm --network none --entrypoint cat "$BASE" "$VF" | git hash-object --stdin)
[ "$b" = $BASE_BLOB ] || { echo "base image nvfp4/b12x.py is $b, not 5dad364d's"; exit 1; }
cp "$K/src/b12x.py" "$K/img/b12x.py"

# ---- plan seed: b1.6 control records re-keyed to the arm path ----
for h in localhost $H2; do ssh -n -o ConnectTimeout=10 $h "python3 -c 'import json,hashlib,sys; r=json.load(open(sys.argv[1]))[\"records\"]; print(len(r), hashlib.sha256(json.dumps({k: v[\"config\"] for k, v in r.items()}, sort_keys=True).encode()).hexdigest()[:12])' $CTLF"; done > "$K/ctl-plans.txt"
[ "$(sort -u "$K/ctl-plans.txt" | wc -l)" = 1 ] || { echo "control plan files differ between the Sparks: $(tr '\n' ' ' < "$K/ctl-plans.txt")"; exit 1; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
cp "$CTLF" "$T/ctl.json"
docker run --rm --network none --user "$(id -u):$(id -g)" --entrypoint python3 -v "$T:/t" -v "$G/k48:/k48:ro" "$BASE" \
  /k48/remap_seed.py /t/ctl.json "$ARMPATH" /t/seed > "$K/remap-seed.txt"
[ "$(ls "$T"/seed/*.json | wc -l)" = 1 ] || { echo "seed remap failed"; exit 1; }
SEED=$(basename "$(ls "$T"/seed/*.json)")
if ! cmp -s "$T/seed/$SEED" "$K/img/preparation/$SEED"; then rm -f "$K"/img/preparation/*.json; cp "$T/seed/$SEED" "$K/img/preparation/"; fi
echo "$SEED" > "$K/SEED"

# ---- P1 image on both Sparks (rebuilt when its inputs change) ----
cat > "$K/img/Dockerfile" <<EOF
# k73 (#123): 1x v3e image (vLLM exp/v3d 5dad364d, b12x 21e0b201) + vLLM d21d7ade's nvfp4/b12x.py (per-rank MXFP8
# large-M copies for TP>1) + the TP=2 plan seed for GDN-MSE @ $REV (b1.6 control records re-keyed).
FROM $BASE
COPY b12x.py $VF
COPY preparation/ /opt/b12x-seed/preparation/
RUN chmod a+r $VF && chmod -R a+rX /opt/b12x-seed
EOF
STAMP=$(cat "$K/img/Dockerfile" "$K/img/b12x.py" "$K/img/preparation/$SEED" | sha256sum | cut -c1-16)
# one build on dgx-01; dgx-02 gets the same image Id by docker save | load (sparkrun ships an image whose Id differs
# inside the 900 s boot timeout). The load streams ~28 GB, so it only runs with K73_SYNC=1 (run.sh preflight, pair idle).
imgid() { ssh -n -o ConnectTimeout=10 $1 "docker image inspect -f '{{.Id}}' $P1 2>/dev/null"; }
[ "$(docker image inspect -f '{{index .Config.Labels "k73.stamp"}}' $P1 2>/dev/null)" = "$STAMP" ] \
  || nice -n 19 docker build -q --label k73.stamp=$STAMP -t $P1 "$K/img" > /dev/null
if [ "$(imgid $H2)" != "$(imgid localhost)" ]; then
  if [ "${K73_SYNC:-0}" = 1 ]; then docker save $P1 | ssh $H2 "docker load -q" > /dev/null
    [ "$(imgid $H2)" = "$(imgid localhost)" ] || { echo "dgx-02: $P1 Id still differs after docker load"; exit 1; }
  else echo "dgx-02: $P1 Id differs from dgx-01 ($(imgid $H2 | cut -c8-19) vs $(imgid localhost | cut -c8-19)); the run's preflight loads it (docker save | load)"; fi
fi
for h in localhost $H2; do
  got=$(ssh -n -o ConnectTimeout=10 $h "docker run --rm --network none --entrypoint cat $P1 $VF | sha256sum | cut -c1-64; docker run --rm --network none --entrypoint ls $P1 /opt/b12x-seed/preparation/$SEED")
  [ "$(echo "$got" | head -1)" = "$(sha256sum < "$K/img/b12x.py" | cut -c1-64)" ] && echo "$got" | grep -q "$SEED" \
    || { echo "$h: $P1 lacks d21d7ade's file or the seed: $got"; exit 1; }
done
echo "$P1" > "$K/IMAGE"; echo "image $P1 (stamp $STAMP) on both Sparks, seed $(cat "$K/remap-seed.txt")"

# ---- CPU tests in P1 ----
timeout -k 30 900 nice -n 19 docker run --rm --network=none --memory 4g -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/pl -v "$G/k18/pylib:/pl:ro" -v "$K/vllm-tests:/w:ro" -w /w --entrypoint python3 "$P1" -m pytest -q \
  -p no:cacheprovider --noconftest test_b12x_nvfp4_large_m.py > "$K/cputest.txt" 2>&1 || { echo "cputest FAILED ($K/cputest.txt)"; exit 1; }
echo "cputest: $(tail -1 "$K/cputest.txt")"

# ---- arm recipes ----
[ "$(md5sum < "$REG" | cut -c1-32)" = e39fcf1e48fbcccf79d06b1a6d2db613 ] || { echo "registry 2x recipe is not b1.6 (md5)"; exit 1; }
cp "$REG" "$K/arm/b16-2x.yaml"
python3 - "$K/arm/b16-2x.yaml" "$K/arm" "$P1" "$REV" "$MXREV" <<'PY'
import sys
src, out, img, rev, mxrev = sys.argv[1:6]
base = open(src).read()
aot = ('  A=/opt/b12x-seed/torch_compile_cache; if [ -d "$A" ] && [ -n "$XDG_CACHE_HOME" ]; then mkdir -p '
       '"$XDG_CACHE_HOME/vllm/torch_compile_cache" && cp -rn "$A/." "$XDG_CACHE_HOME/vllm/torch_compile_cache/" || true; fi\n')
def one(s, a, b):
    assert s.count(a) == 1, a
    return s.replace(a, b)
for arm, cut, compact in (("c26", 26, 0), ("c41", 41, 0), ("c26x", 26, 1), ("c41x", 41, 1)):
    s = base
    s = one(s, "\nname: qwen3.8-flash-next-2x-dgx-spark\n", f"\nname: qwen3.8-flash-next-2x-dgx-spark-k73-{arm}\n")
    s = one(s, "\nmodel: local-inference-lab/Qwen3.8-Flash-Next-NVFP4\n", "\nmodel: ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE\n")
    s = one(s, f"\nmodel_revision: {mxrev}\n", f"\nmodel_revision: {rev}\n")
    s = one(s, "\ncontainer: ghcr.io/ursuciprian/spark-vllm-b12x:b1.6-20261008-b7fbaf96-a7e649d8-warm\n", f"\ncontainer: {img}\n")
    env = (f'  VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS: "{cut}"\n'
           f'  VLLM_B12X_NVFP4_MXFP8_CHECKPOINT: "/cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/{mxrev}"\n')
    if compact:
        env += '  VLLM_GDN_COMPACT_RECORDS: "1"\n  VLLM_GDN_SHARED_PREFILL_STAGING: "1"\n'
    s = one(s, "\nenv:\n", "\nenv:\n" + env)
    s = one(s, "  python3 /opt/mtp-refit/overlay.py build || exit 1\n", aot)
    s = one(s, "  vllm serve /cache/runtime/mtp-refit/Qwen3.8-Flash-Next-NVFP4-7c4f1bc1-mtp-16c9bd54 \\\n",
            f"  vllm serve ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE \\\n    --revision {rev} \\\n")
    s = one(s, "--served-model-name qwen3.8-flash-next local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \\\n",
            "--served-model-name qwen3.8-flash-next ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE \\\n")
    head = (f"# k73 arm {arm} (#123, 2026-10-08): shipped 2x b1.6 with the GDN-MSE checkpoint @ {rev[:8]} (retrained drafter\n"
            f"# included) and the M-dispatch: GDN in_proj_qkvz/out_proj NVFP4 below {cut} rows, MXFP8 (7c4f1bc1) at {cut}+ rows"
            + (", compact GDN records + shared prefill staging" if compact else "") + ".\n"
            "# Image: vLLM d21d7ade (exp/v3d 5dad364d + per-rank MXFP8 copies for TP>1), b12x 21e0b201.\n")
    open(f"{out}/k73-{arm}.yaml", "w").write(head + s)
PY
for a in c26 c41 c26x c41x; do
  f=$K/arm/k73-$a.yaml; n=$(diff "$K/arm/b16-2x.yaml" "$f" | grep -c '^[<>]' || true)
  want=$([ "${a%x}" = "$a" ] && echo 20 || echo 22)
  [ "$n" = "$want" ] && grep -qx "container: $P1" "$f" && grep -q -- "--revision $REV" "$f" && grep -qx '  tensor_parallel: 2' "$f" \
    && ! grep -q "^  python3 /opt/mtp-refit" "$f" && grep -q 'torch_compile_cache' "$f" \
    || { echo "arm recipe $a edits did not apply (diff lines $n, want $want)"; diff "$K/arm/b16-2x.yaml" "$f"; exit 1; }
done
echo "recipes: $K/arm/k73-{c26,c41,c26x,c41x}.yaml"
