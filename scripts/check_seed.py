#!/usr/bin/env python3
"""Image check: fail when a first boot measured any b12x plan, i.e. the image's plan seed is incomplete.

  check_seed.py <serve.log> [<plan.json>:<n> ...]   the rank-0 serve log of a first boot (no plan file for the served
                                                    path in the runtime cache before it); exit 0 = every plan group
                                                    0 measured. TP=2 workers print no b12x progress lines, so for
                                                    them pass the plan file of their runtime cache and the seed's
                                                    record count: a measured plan adds a record
  check_seed.py --selftest

k70 (#115) shipped-candidate image lacked 33 a16 GEMM plans: a fresh install measured them on its first boot and picked
other tiles than the long-lived caches the A/B was run on. Used by the backlog check boots (hfship for 1x images, b16
for 2x images) before an image is promoted.
"""
import json, re, sys

LINE = re.compile(r"b12x \w+ ([a-z_]+\.[a-z_.]+): \d+/\d+ ready, [^\n]*?(\d+) measured(?:, (\d+) cached)?")


def check(text):
    groups = {}
    for g, m, c in LINE.findall(text):
        mm, cc = groups.get(g, (0, 0))
        groups[g] = (max(mm, int(m)), max(cc, int(c or 0)))
    bad = sorted(g for g, (m, _) in groups.items() if m)
    return groups, bad


def main(paths):
    rc = 0
    for p in paths:
        if p.rpartition(":")[0].endswith(".json"):
            f, _, n = p.rpartition(":")
            got = len(json.load(open(f))["records"])
            print(f"{f}: {got} records, seed {n}" + ("" if got == int(n) else " FAIL"))
            rc |= got != int(n)
            continue
        groups, bad = check(open(p, errors="replace").read())
        print(f"{p}: " + (", ".join(f"{g} {m} measured / {c} cached" for g, (m, c) in sorted(groups.items())) or "no b12x lines"))
        if not groups or bad:
            print(f"FAIL {p}: " + (f"measured plans in {' '.join(bad)}" if bad else "no b12x plan lines"))
            rc = 1
    print("seed check " + ("PASS" if rc == 0 else "FAIL"))
    return rc


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        ok = "b12x compiling gemm.blockscaled_precision: 405/436 ready, 0 measured, 403 cached, 0 compilations, 0:10\n" \
             "b12x priming attention.qsa: 11/381 ready, candidates 0/1 prepared, 0 measured, 10 cached, 0 compilations, 0:01\n"
        assert check(ok)[1] == [] and len(check(ok)[0]) == 2
        assert check(ok + "b12x planning gemm.blockscaled_precision: 0/436 ready, 33 measured, 403 cached, 0:30\n")[1] == ["gemm.blockscaled_precision"]
        assert check("nothing")[0] == {}
        k70 = "b12x measuring gemm.blockscaled_precision: 402/436 ready, candidates 8/8 prepared, rank 0 batch 1, round 2/3, 48 measured\n"
        assert check(ok + k70)[1] == ["gemm.blockscaled_precision"]
        print("selftest ok")
        sys.exit(0)
    sys.exit(main(sys.argv[1:]) if sys.argv[1:] else __doc__)
