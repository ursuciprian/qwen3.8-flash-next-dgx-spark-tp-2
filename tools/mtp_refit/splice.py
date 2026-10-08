"""Overlay snapshot that serves the refit MTP tensors (stdlib only, runs on the hosts).

    python3 tools/mtp_refit/splice.py <src snapshot> <mtp_refit.safetensors> <out snapshot>

Every refit tensor must exist in the source with the same dtype and shape, so the shard headers and
the index stay byte-identical: each shard holding refit tensors is copied and those byte ranges are
overwritten; every other file is hardlinked from the source (k53/mkhybrid.py approach). Writes
<out>/SPLICE.json (refit file sha256, tensors per shard, new shard sha256) and checks the result.
Next, for a warm boot under the new path: k48/remap_seed.py <seed.json> <out snapshot> <seed dir>.
"""
import hashlib
import json
import os
import shutil
import struct
import sys


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def splice(src, refit, out):
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    wmap = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    rh, rbase = header(refit)
    by_shard = {}
    for name, t in rh.items():
        if name not in wmap:
            raise SystemExit(f"{name}: not in the source checkpoint (would load as nothing)")
        by_shard.setdefault(wmap[name], []).append(name)
    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for f in os.listdir(src):
        if os.path.isdir(os.path.join(src, f)):
            shutil.copytree(os.path.join(src, f), os.path.join(tmp, f), copy_function=os.link)
        elif f not in by_shard:
            os.link(os.path.join(src, f), os.path.join(tmp, f))
    report = {"refit": os.path.abspath(refit), "refit_sha256": sha256(refit), "shards": {}}
    with open(refit, "rb") as rf:
        for shard, names in sorted(by_shard.items()):
            sh, sbase = header(os.path.join(src, shard))
            dst = os.path.join(tmp, shard)
            shutil.copyfile(os.path.join(src, shard), dst)
            with open(dst, "r+b") as df:
                for n in names:
                    a, b = rh[n], sh[n]
                    if (a["dtype"], a["shape"]) != (b["dtype"], b["shape"]):
                        raise SystemExit(f"{n}: refit {a['dtype']}{a['shape']} vs source {b['dtype']}{b['shape']}")
                    rf.seek(rbase + a["data_offsets"][0])
                    data = rf.read(a["data_offsets"][1] - a["data_offsets"][0])
                    df.seek(sbase + b["data_offsets"][0])
                    df.write(data)
            assert header(dst) == (sh, sbase)
            with open(dst, "rb") as df:  # read back
                for n in names:
                    rf.seek(rbase + rh[n]["data_offsets"][0])
                    df.seek(sbase + sh[n]["data_offsets"][0])
                    size = rh[n]["data_offsets"][1] - rh[n]["data_offsets"][0]
                    assert df.read(size) == rf.read(size), n
            report["shards"][shard] = {"tensors": len(names), "sha256": sha256(dst)}
    json.dump(report, open(os.path.join(tmp, "SPLICE.json"), "w"), indent=1)
    os.rename(tmp, out)
    return report


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    print(json.dumps(splice(*sys.argv[1:4]), indent=1))
