"""Read-only phase 2 metrics sidecar (#97 dashboard): derives progress from the files refit-p2/job.sh
writes and rewrites one Prometheus textfile every --interval seconds. Touches nothing the job reads.

    nohup nice -n 19 python3 -m tools.mtp_refit.p2_metrics --state ~/GEN-AI/refit-p2/STATE \
      --res <results/refit-p2-DAY> --data ~/.cache/huggingface/mtp-refit/p2-DAY \
      --out /var/lib/node_exporter/textfile_collector/mtp_refit_p2.prom &

Exits after the job's final state (DONE/FAILED) is written, leaving the last snapshot in place.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

from . import prom

SETS = {"main": "gen", "live-t0": "gen-live-t0", "live-t1": "gen-live-t1"}


class Tail:
    """Counts records and response tokens of gen/*.jsonl incrementally (complete lines only)."""

    def __init__(self):
        self.pos, self.docs, self.tokens = {}, 0, 0

    def update(self, d):
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            with open(f, "rb") as fh:
                fh.seek(self.pos.get(f, 0))
                for line in fh:
                    if not line.endswith(b"\n"):
                        break
                    self.pos[f] = self.pos.get(f, 0) + len(line)
                    self.docs += 1
                    self.tokens += len(json.loads(line)["output_token_ids"])
        return self


def lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return None


def snapshot(a, tails, total_prompts):
    state = open(a.state).read().strip()
    lab = {"run": a.run, "phase": "p2"}
    yield "mtp_refit_p2_state_info", {**lab, "state": state[:120]}, 1
    yield "mtp_refit_p2_done", lab, 1 if state.startswith(("DONE", "FAILED")) else 0
    yield "mtp_refit_p2_failed", lab, 1 if "FAILED" in state else 0
    m = re.search(r"\((\d+) response tokens\)", state)
    if m:
        a.budget = int(m.group(1))
    yield "mtp_refit_p2_budget_tokens", lab, a.budget
    yield "mtp_refit_p2_prompts_total", lab, total_prompts
    for s, sub in SETS.items():
        t = tails[s].update(os.path.join(a.data, sub))
        yield "mtp_refit_p2_generated_docs", {**lab, "set": s}, t.docs
        yield "mtp_refit_p2_response_tokens", {**lab, "set": s}, t.tokens
        yield "mtp_refit_p2_captured_docs", {**lab, "set": s}, lines(os.path.join(a.data, "capture", s, "manifest.jsonl"))
    shards = glob.glob(os.path.join(a.data, "capture", "shards", "*"))
    yield "mtp_refit_p2_capture_shards", lab, len(shards)
    yield "mtp_refit_p2_capture_bytes", lab, sum(os.path.getsize(f) for f in shards if os.path.isfile(f))
    # results, once the checks step has written them
    m = re.search(r"parity drafts=(PASS|FAIL) live: t0=(PASS|FAIL) t1=(PASS|FAIL)", state)
    if m:
        for check, v in zip(("drafts", "live_t0", "live_t1"), m.groups()):
            yield "mtp_refit_parity_pass", {**lab, "check": check}, 1 if v == "PASS" else 0
    base = os.path.join(a.res, "eval-baseline.json")
    if os.path.exists(base):
        yield from prom.acceptance(json.load(open(base)), {**lab, "drafter": "shipped"})
    yield "mtp_refit_p2_sidecar_timestamp_seconds", lab, time.time()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", help="default: basename of --res")
    ap.add_argument("--budget", type=int, default=2000000, help="until STATE names it")
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    a.run = a.run or os.path.basename(os.path.normpath(a.res))
    total = sum(lines(f) for f in glob.glob(os.path.join(a.data, "prompts", "*.jsonl")))
    tails = {s: Tail() for s in SETS}
    while True:
        samples = list(snapshot(a, tails, total))
        prom.write(a.out, samples)
        if a.once or any(n == "mtp_refit_p2_done" and v for n, _, v in samples):
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
