#!/usr/bin/env python3
"""k71 (#115): give the k70 arm the control's b12x plan selections.

  pin_plans.py <arm plan file> <control plan file> <backup path>

k70's arm bake measured 33 gemm.blockscaled_precision plans that the image seed (b1.4 8ccf4799, 583 records) lacks; the
control's runtime cache (8ccf4799, 1167 records) already holds the same 33 keys, and 8 of them select a different
tile/split. This rewrites every arm record to the control's record for the same key (records hold no paths, keys are
request digests), so arm and control run identical plans and differ only in the drafter weights. Idempotent; the
first run saves the k70 arm file to <backup path>.
"""
import json, os, shutil, sys

arm_p, ctl_p, bak = sys.argv[1:4]
arm, ctl = json.load(open(arm_p)), json.load(open(ctl_p))
missing = [k for k in arm["records"] if k not in ctl["records"]]
assert not missing, f"{len(missing)} arm keys not in the control cache"
if not os.path.exists(bak):
    shutil.copy2(arm_p, bak)
changed = sum(arm["records"][k] != ctl["records"][k] for k in arm["records"])
arm["records"] = {k: ctl["records"][k] for k in arm["records"]}
tmp = arm_p + ".k71tmp"
json.dump(arm, open(tmp, "w"), sort_keys=True, separators=(",", ":"))
os.replace(tmp, arm_p)
print(f"{os.path.basename(arm_p)}: {len(arm['records'])} records, {changed} replaced by the control's selection")
