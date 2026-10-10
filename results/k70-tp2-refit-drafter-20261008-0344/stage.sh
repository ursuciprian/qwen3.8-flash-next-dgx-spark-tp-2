#!/usr/bin/env bash
# k70 staging (2026-10-07, #115): the #97 refit run1 drafter on the shipped 2x b1.4. Idempotent, CPU only, no lock,
# safe next to a running job (hardlinks, one small seed file, a one-layer image). run.sh calls it in preflight/dry run.
#   snapshot  F470 = 7c4f1bc1 (what the 2x serves) with model-00034 swapped for the refit shard: every file is a
#             hardlink of the 7c4f1bc1 blob, except model-00034 = hardlink of f4..61's spliced shard. Valid because
#             7c4f1bc1's model-00034 IS f4..03's model-00034 (same inode, blob 4ecb4883 on both Sparks; checked here),
#             and f4..61's model-00034 is splice.py(f4..03, run1 refit) (SPLICE.json there, sha f628108e). Both Sparks.
#   seed      b1.4's TP=2 b12x plan seed 8ccf4799 (namespace 7c4f1bc1 under /cache) re-keyed to F470's host path
#             (sparkrun serves a local snapshot under its host path; r9-dense TP2 boot log) with k48/remap_seed.py
#   image     IMG = b1.4 warm + that seed (img/Dockerfile), both Sparks
#   recipe    arm/k70.yaml = the registry 2x recipe (md5 7b546de5 = repo main) with name, model: F470 (no
#             model_revision, `vllm serve {model}` without --revision), container IMG. No VLLM_CACHE_ROOT:
#             sparkrun keys the runtime cache dir by the model path (a new dir for f4..70; bake fills it)
set -eu
K=$HOME/GEN-AI/backlog/k70; G=$HOME/GEN-AI; H2=<cx7-ip-b>
B14=ghcr.io/ursuciprian/spark-vllm-b12x:b1.4-20261001-b7fbaf96-a7e649d8-warm
IMG=spark-vllm-b12x:k70-b14-refit-run1-c92ac62d
SN=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots
SRC=$SN/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd; F403=$SN/f400000000000000000000000000000000000003
F461=$SN/f400000000000000000000000000000000000061; F470=$SN/f400000000000000000000000000000000000070
SH=model-00034-of-00036.safetensors; SEED=8ccf4799746f12aa17e1516b1378f0474077caca591570caa40dd18dc68b7db2.json
REG=$HOME/.cache/sparkrun/registries/_url_e47b9e8b5e91/recipes/qwen3.8-flash-next/qwen3.8-flash-next-2x-dgx-spark.yaml
mkdir -p "$K/arm" "$K/img/seed"

# snapshot (per host; same script text over ssh)
SNAPSH='set -eu; S='$SRC'; F3='$F403'; F61='$F461'; F='$F470'; SH='$SH'
[ "$(stat -L -c %i $S/$SH)" = "$(stat -L -c %i $F3/$SH)" ] || { echo "7c4f1bc1 $SH is not f4..03 $SH"; exit 1; }
grep -q "\"sha256\": \"f628108e0eef191fa8e32007da5485509781dd19c2e3c4fec0f2dcb7e1a5337f\"" $F61/SPLICE.json || grep -q f628108e0eef191fa8e32007da5485509781dd19c2e3c4fec0f2dcb7e1a5337f $F61/SPLICE.json || { echo "f4..61 SPLICE.json is not run1"; exit 1; }
grep -q c92ac62d1caaf69414edfad72210d73f67abceeeb881135181e12c59b2ccd4c8 $F61/SPLICE.json || { echo "f4..61 is not refit run1"; exit 1; }
if [ ! -d $F ]; then rm -rf $F.tmp; mkdir $F.tmp
  for p in $S/*; do f=$(basename $p); if [ $f = $SH ]; then ln $F61/$SH $F.tmp/$f; else ln "$(readlink -f $p)" $F.tmp/$f; fi; done
  for p in $S/.[!.]*; do [ -e "$p" ] && ln "$(readlink -f $p)" $F.tmp/$(basename $p); done
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d[\"base\"]=\"7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd\"; d[\"note\"]=\"k70: 7c4f1bc1 hardlinks + model-00034 hardlinked from f4..61 (7c4f1bc1 model-00034 = f4..03 model-00034)\"; json.dump(d, open(sys.argv[2], \"w\"), indent=1)" $F61/SPLICE.json $F.tmp/SPLICE.json
  mv $F.tmp $F; fi
n=0; for p in $S/*; do f=$(basename $p); [ $f = $SH ] && w=$F61/$SH || w=$p; [ "$(stat -L -c %i $F/$f)" = "$(stat -L -c %i $w)" ] || { echo "$F/$f wrong inode"; exit 1; }; n=$((n+1)); done
[ $n -ge 40 ] && echo "$(hostname): $F ok ($n files, $SH = f4..61 inode $(stat -c %i $F/$SH))"'
bash -c "$SNAPSH"
ssh -n -o ConnectTimeout=10 $H2 "$SNAPSH"

# seed + image
if [ ! -s "$K/IMAGE" ] || [ "$(ls "$K"/img/seed/*.json 2>/dev/null | wc -l)" != 1 ]; then
  rm -f "$K"/img/seed/*.json
  docker run --rm --network none --entrypoint cat "$B14" /opt/b12x-seed/preparation/$SEED > "$K/seed-src.json"
  docker run --rm --network none --user "$(id -u):$(id -g)" --entrypoint python3 -v "$K:/k" -v "$G/k48:/k48:ro" "$B14" \
    /k48/remap_seed.py /k/seed-src.json "$F470" /k/img/seed > "$K/remap-seed.txt"
  [ "$(ls "$K"/img/seed/*.json | wc -l)" = 1 ] || { echo "seed remap failed"; exit 1; }
  { echo "# k70 (#115): shipped b1.4 warm image + its TP=2 b12x seed ($SEED) re-keyed to the refit snapshot"
    echo "# $F470 (k48/remap_seed.py). Nothing else changes: the refit MTP tensors live in the snapshot."
    echo "FROM $B14"; echo "COPY seed/*.json /opt/b12x-seed/preparation/"; } > "$K/img/Dockerfile"
  docker build -q -t "$IMG" "$K/img" > /dev/null
  ssh -n $H2 "mkdir -p $K"; rsync -a --delete "$K/img/" "$H2:$K/img/"
  ssh -n $H2 "docker build -q -t $IMG $K/img" > /dev/null
  echo "$IMG" > "$K/IMAGE"; fi
for h in localhost $H2; do ssh -n -o ConnectTimeout=10 $h "docker run --rm --network none --entrypoint ls $IMG /opt/b12x-seed/preparation/$(basename "$(ls "$K"/img/seed/*.json)")" > /dev/null \
  || { echo "$h: $IMG lacks the re-keyed seed"; exit 1; }; done
echo "seed: $(cat "$K/remap-seed.txt"); image $IMG on both Sparks"

# arm recipe
[ "$(md5sum < "$REG" | cut -c1-32)" = 7b546de5eaf09561ce8bb53c551a6836 ] || { echo "registry 2x recipe is not b1.4 (md5)"; exit 1; }
cp "$REG" "$K/arm/b14-2x.yaml"
sed -e "1i # k70 arm (#115, 2026-10-07): shipped 2x b1.4 with the #97 refit run1 MTP drafter (snapshot f4..70 = 7c4f1bc1 + refit model-00034), image $IMG (b1.4 + re-keyed seed); runtime cache keyed by the new model path" \
    -e "s|^name: .*|name: qwen3.8-flash-next-2x-dgx-spark-k70|" -e "s|^model: .*|model: $F470|" -e "/^model_revision: /d" \
    -e "s|^container: .*|container: $IMG|" \
    -e "s|^  vllm serve local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \\\\\$|  vllm serve {model} \\\\|" \
    -e "/^    --revision 7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd \\\\\$/d" "$K/arm/b14-2x.yaml" > "$K/arm/k70.yaml"
d=$(diff "$K/arm/b14-2x.yaml" "$K/arm/k70.yaml" | grep -c '^[<>]' || true)
[ "$d" = 11 ] && grep -qx "model: $F470" "$K/arm/k70.yaml" && grep -qx "container: $IMG" "$K/arm/k70.yaml" \
  && grep -qx '  vllm serve {model} \\' "$K/arm/k70.yaml" && ! grep -q -- '^    --revision' "$K/arm/k70.yaml" && ! grep -q '^model_revision' "$K/arm/k70.yaml" \
  && grep -q -- '--tensor-parallel-size {tensor_parallel}' "$K/arm/k70.yaml" && grep -qx '  tensor_parallel: 2' "$K/arm/k70.yaml" \
  || { echo "arm recipe edits did not apply (diff lines $d)"; diff "$K/arm/b14-2x.yaml" "$K/arm/k70.yaml"; exit 1; }
echo "recipe: $K/arm/k70.yaml"
