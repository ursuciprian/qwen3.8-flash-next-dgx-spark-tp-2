#!/usr/bin/env python3
"""b1.6 (#115): serve local-inference-lab/Qwen3.8-Flash-Next-NVFP4 @ 7c4f1bc1 with the retrained MTP drafter.

  overlay.py build                        recipe command, inside the container (sparkrun mounts the HF cache at
                                          /cache/huggingface and the runtime cache at /cache/runtime)
  overlay.py patch <base> <refit> <out>   image build, once: the tensors of <refit> whose bytes differ from <base>

build makes DST = the 7c4f1bc1 snapshot with model-00034 replaced: every other file is a symlink into the HF cache,
model-00034 is a copy of 7c4f1bc1's with the 24 retrained mtp.* tensors of PATCH written over theirs (same names,
dtypes, shapes and offsets). The result is byte-identical to model-00034 of ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE
@ 16c9bd54 (sha256 checked), the shard k70/k71 measured. A finished DST is reused; a broken one is rebuilt.
DST is a fixed container path because the b12x plan seed and vLLM's compile cache key on the served model path.
Costs 4.5 GB in sparkrun's runtime cache on each node and ~20 s on the first boot.
"""
import hashlib, json, os, shutil, struct, sys

SRC = "/cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd"
DST = "/cache/runtime/mtp-refit/Qwen3.8-Flash-Next-NVFP4-7c4f1bc1-mtp-16c9bd54"
PATCH = "/opt/mtp-refit/mtp-refit-16c9bd54.safetensors"
SHARD = "model-00034-of-00036.safetensors"
SHA = "f628108e0eef191fa8e32007da5485509781dd19c2e3c4fec0f2dcb7e1a5337f"
MARK = ".mtp-refit.json"


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def build():
    if not os.path.isfile(os.path.join(SRC, SHARD)):
        sys.exit(f"overlay: {SRC} has no {SHARD}; sparkrun downloads it from model/model_revision")
    try:
        if json.load(open(os.path.join(DST, MARK)))["sha256"] == SHA and \
                os.path.getsize(os.path.join(DST, SHARD)) == os.path.getsize(os.path.join(SRC, SHARD)) and \
                all(os.path.exists(os.path.join(DST, f)) for f in os.listdir(SRC)):
            print(f"overlay: {DST} ready")
            return
    except (OSError, ValueError, KeyError):
        pass
    tmp = DST + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for f in os.listdir(SRC):
        if f != SHARD:
            os.symlink(os.path.realpath(os.path.join(SRC, f)), os.path.join(tmp, f))
    out = os.path.join(tmp, SHARD)
    shutil.copyfile(os.path.join(SRC, SHARD), out)
    base, b0 = header(out)
    refit, r0 = header(PATCH)
    with open(PATCH, "rb") as p, open(out, "r+b") as o:
        for name, t in refit.items():
            s = base[name]
            assert (s["dtype"], s["shape"]) == (t["dtype"], t["shape"]), name
            (a, b), (c, d) = s["data_offsets"], t["data_offsets"]
            assert b - a == d - c, name
            p.seek(r0 + c)
            o.seek(b0 + a)
            o.write(p.read(d - c))
    got = sha256(out)
    if got != SHA:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(f"overlay: spliced {SHARD} sha256 {got}, expected {SHA}")
    json.dump({"sha256": SHA, "src": SRC, "patch": PATCH, "tensors": len(refit)}, open(os.path.join(tmp, MARK), "w"))
    shutil.rmtree(DST, ignore_errors=True)
    os.replace(tmp, DST)
    print(f"overlay: built {DST} ({len(refit)} tensors replaced, sha256 ok)")


def patch(base_p, refit_p, out_p):
    base, b0 = header(base_p)
    refit, r0 = header(refit_p)
    assert base == refit, "shards differ in names, dtypes, shapes or offsets"
    names, blobs = [], []
    with open(base_p, "rb") as fb, open(refit_p, "rb") as fr:
        for name, t in sorted(refit.items()):
            a, b = t["data_offsets"]
            fb.seek(b0 + a)
            fr.seek(r0 + a)
            x, y = fb.read(b - a), fr.read(b - a)
            if x != y:
                names.append(name)
                blobs.append(y)
    hdr, off = {"__metadata__": {"refit_shard_sha256": SHA, "note": "retrained mtp.* tensors, #97 refit run 1"}}, 0
    for name, blob in zip(names, blobs):
        hdr[name] = {"dtype": refit[name]["dtype"], "shape": refit[name]["shape"], "data_offsets": [off, off + len(blob)]}
        off += len(blob)
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(out_p, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for blob in blobs:
            f.write(blob)
    print(f"{out_p}: {len(names)} tensors, {off} bytes: {' '.join(names)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build()
    elif sys.argv[1:2] == ["patch"] and len(sys.argv) == 5:
        patch(*sys.argv[2:5])
    else:
        sys.exit(__doc__)
