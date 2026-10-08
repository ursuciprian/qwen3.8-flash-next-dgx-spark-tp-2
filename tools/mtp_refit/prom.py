"""Prometheus textfile metrics for the refit tools (#97 dashboard), stdlib only.

A writer owns one .prom file and rewrites it whole (tmp + rename, so the collector never reads a
partial file). Point it at the node exporter textfile dir, e.g.
/var/lib/node_exporter/textfile_collector/mtp_refit_train.prom (Alloy already scrapes that dir on both
Sparks). Delete the file when the run's numbers are no longer wanted: the collector exports it forever.
"""
from __future__ import annotations

import json
import os


def _fmt(name, labels, value):
    lab = ",".join(f"{k}={json.dumps(str(v))}" for k, v in labels.items())  # same escapes as the text format
    return f"{name}{{{lab}}} {float(value)!r}\n"


def write(path: str, samples) -> None:
    """samples: iterable of (name, labels dict, value); None values are skipped."""
    body = "".join(_fmt(n, lab, v) for n, lab, v in samples if v is not None)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(body)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def acceptance(res: dict, labels: dict):
    """Samples from an eval_offline.evaluate result ({category: _summary}); position is 1-based."""
    for cat, s in res.items():
        for mode in ("t0", "t1"):
            for kind in ("per_draft", "per_position"):
                for k, x in enumerate(s[mode][kind]):
                    yield ("mtp_refit_acceptance",
                           {**labels, "category": cat, "mode": mode, "kind": kind, "position": k + 1}, x)
            yield "mtp_refit_tokens_per_step", {**labels, "category": cat, "mode": mode, "depth": "4"}, \
                s[mode]["tokens_per_step_d4"]
            yield "mtp_refit_tokens_per_step", {**labels, "category": cat, "mode": mode,
                                                "depth": str(len(s[mode]["per_position"]))}, s[mode]["tokens_per_step"]
