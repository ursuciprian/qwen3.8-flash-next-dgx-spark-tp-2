#!/usr/bin/env python3
"""NVMe read latency after an idle gap (opus-gaps, r11, 2026-10-03).

    nvme_idle_latency.py <file> [--reads 20]

One 4 KiB O_DIRECT read at a random aligned offset of <file> after each idle
gap; prints the median and max latency per gap. An APST drive (Linux primary
timeout 100 ms) shows the low-power exit latency only for gaps above the
timeout. Run with no server up: it measures the drive, not the page cache.
"""

import argparse
import mmap
import os
import random
import statistics
import time

GAPS_MS = (5, 20, 50, 80, 95, 110, 150, 300, 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--reads", type=int, default=20)
    a = ap.parse_args()
    fd = os.open(a.file, os.O_RDONLY | os.O_DIRECT)
    blocks = os.fstat(fd).st_size // 4096
    buf, rng = mmap.mmap(-1, 4096), random.Random(0)
    print(f"{a.file}: {blocks} blocks; gap_ms median_ms max_ms")
    for gap in GAPS_MS:
        lat = []
        for _ in range(a.reads):
            time.sleep(gap / 1e3)
            t0 = time.perf_counter()
            os.preadv(fd, [buf], rng.randrange(blocks) * 4096)
            lat.append((time.perf_counter() - t0) * 1e3)
        print(f"{gap:6d} {statistics.median(lat):8.3f} {max(lat):8.3f}", flush=True)


if __name__ == "__main__":
    main()
