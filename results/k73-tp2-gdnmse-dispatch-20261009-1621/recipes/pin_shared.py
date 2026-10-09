#!/usr/bin/env python3
"""k73 (#123): pin the arm's b12x plan file to the control's selections for every key both have.

  pin_shared.py <arm plan file> <control plan file>     rewrites the arm file in place, prints one summary line
  pin_shared.py --selftest

The arm file starts as the control's records re-keyed to the arm path (stage.sh), so after the bake this should
replace nothing: it is the check that the bake kept every shared selection, plus a fix if it did not. Keys only the
arm has (the GDN W4A16 and MXFP8 large-M plans b1.6 lacks) keep the bake's measured record. Records hold no paths.
"""
import json, os, sys


def pin(arm, ctl):
    shared = [k for k in arm["records"] if k in ctl["records"]]
    changed = sum(arm["records"][k]["config"] != ctl["records"][k]["config"] for k in shared)
    for k in shared:
        arm["records"][k] = ctl["records"][k]
    return len(shared), changed, len(arm["records"]) - len(shared)


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        r = lambda c: {"assignment": {}, "config": {"t": c}, "coverage": {}, "programs": []}
        a = {"records": {"x": r(1), "y": r(2), "z": r(9)}}
        assert pin(a, {"records": {"x": r(1), "y": r(3)}}) == (2, 1, 1) and a["records"]["y"]["config"] == {"t": 3}
        print("selftest ok"); sys.exit(0)
    arm_p, ctl_p = sys.argv[1:3]
    arm, ctl = json.load(open(arm_p)), json.load(open(ctl_p))
    shared, changed, own = pin(arm, ctl)
    tmp = arm_p + ".k73tmp"
    json.dump(arm, open(tmp, "w"), sort_keys=True, separators=(",", ":"))
    os.replace(tmp, arm_p)
    print(f"{os.path.basename(arm_p)}: {len(arm['records'])} records, {shared} shared with the control "
          f"({changed} configs replaced by the control's), {own} arm-only")
