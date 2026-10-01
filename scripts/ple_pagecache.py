#!/usr/bin/env python3
"""Page-cache control for the Qwen3.8-Flash-Next PLE table (r8 TP=1 profiling).

    ple_pagecache.py resident <snapshot_dir>   # GiB of the table in the page cache
    ple_pagecache.py evict    <snapshot_dir>   # drop it (posix_fadvise DONTNEED, no root)
    ple_pagecache.py touch    <snapshot_dir>   # read it once (fills what fits)

Works on the PLE tensors' byte ranges only (both planes of every shard), found from
the safetensors headers like mods/vllm-tp1-ple-mmap/ple_table_check.py. Host
python3, stdlib only; vmtouch/fincore are not installed on the pair. Eviction only
drops clean, unmapped pages: run it with no server mapping the table.
"""

import ctypes
import json
import mmap
import os
import re
import struct
import sys

PAT = re.compile(r"ple\.ple_embedding\.ngram_embedding\.shard_\d+\.(weight|weight_scale)$")
PAGE = os.sysconf("SC_PAGE_SIZE")


def ranges(snap):
    """(path, start, end) of every PLE plane in the snapshot."""
    wmap = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    out = []
    for fn in sorted({f for k, f in wmap.items() if PAT.search(k)}):
        path = os.path.realpath(os.path.join(snap, fn))
        with open(path, "rb") as fh:
            size = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(size))
        for name, meta in header.items():
            if PAT.search(name):
                a, b = meta["data_offsets"]
                out.append((path, 8 + size + a, 8 + size + b))
    if not out:
        sys.exit(f"no PLE tensors under {snap}")
    return out


def resident(spans):
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    pages = total = 0
    for path, start, end in spans:
        lo = start // PAGE * PAGE
        length = end - lo
        fd = os.open(path, os.O_RDONLY)
        try:
            addr = libc.mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED, fd, lo)
            if addr in (None, ctypes.c_void_p(-1).value):
                raise OSError(ctypes.get_errno(), "mmap")
            vec = ctypes.create_string_buffer((length + PAGE - 1) // PAGE)
            ok = libc.mincore(addr, length, vec) == 0
            libc.munmap(addr, length)
            if not ok:
                raise OSError(ctypes.get_errno(), "mincore")
        finally:
            os.close(fd)
        pages += len(vec.raw) - vec.raw.count(0)
        total += len(vec.raw)
    return pages * PAGE, total * PAGE


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("resident", "evict", "touch"):
        sys.exit(__doc__)
    cmd, spans = sys.argv[1], ranges(sys.argv[2])
    if cmd in ("evict", "touch"):
        buf = memoryview(bytearray(8 << 20))
        for path, start, end in spans:
            fd = os.open(path, os.O_RDONLY)
            try:
                if cmd == "evict":
                    os.posix_fadvise(fd, start, end - start, os.POSIX_FADV_DONTNEED)
                else:
                    pos = start
                    while pos < end:
                        got = os.preadv(fd, [buf[: min(len(buf), end - pos)]], pos)
                        if got <= 0:
                            break
                        pos += got
            finally:
                os.close(fd)
    cached, total = resident(spans)
    print(f"{cmd}: PLE table resident {cached / 2**30:.2f} of {total / 2**30:.2f} GiB "
          f"({100 * cached / total:.1f}%) in {len(spans)} planes")


if __name__ == "__main__":
    main()
