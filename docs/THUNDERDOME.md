# Thunderdome

One-hour paired screening for single-Spark (TP=1) arms. Two arms enter, one arm leaves. Design issue: #83.

A full single-Spark arm used to cost 5 to 8 hours: a cold boot and bake, a 4-pass ABBA spread across both
Sparks (`r8_tp1_driver.sh`), then the full gate (`gate_arm.sh`). Most arms fail, and that only showed at
the end. Thunderdome screens two arms at once, one per Spark, in about 75 minutes. Only arms that pass go
on to the gate, which is split across both Sparks.

- `scripts/thunderdome.sh`: the driver (screen, gate, bake, dry run).
- `scripts/thunderdome_report.py`: the verdict for one Spark (`--selftest` checks the rules).
- Uses `depth_decode_probe.py`, `paired_decode_ab.py`, `logits_equiv.py`, `straggler_probe.py`,
  `fidelity_probe.py`, llama-benchy and tool-eval-bench.

## What runs

Each Spark screens its own arm against a control on that Spark. The default control is the v3b recipe.
Each Spark gets four fresh warm boots:

| boot | recipe  | pass |
|------|---------|------|
| 1    | control | 1    |
| 2    | arm     | 1    |
| 3    | arm     | 2    |
| 4    | control | 2    |

Arm pass K is compared with control pass K on the same Spark with the same prompts, so node bias cancels.
The control, arm, arm, control order also cancels linear drift. Drift is real here: in the 2026-10-03 kv10
run, PLE page-cache residency on one Spark went 0 %, 31 %, 44 %, 46 % over four boots.

Per boot (estimates from the 2026-10-02/03 TP1 driver logs):

| step | settings | time |
|------|----------|------|
| boot | sparkrun solo, warm compile cache required | ~3 min |
| warm-up | one 4K-context request | ~25 s |
| fresh c1 / c4 / c8 | `depth_decode_probe.py`, T=0, 256-token context + 2K new, 512 out, 2 reps | ~45 / 85 / 120 s |
| d16k c4 | 16K context + 2K new, 512 out, 2 reps | ~150 s |
| d16k c8 | 16K context, 1 rep, cut at 480 s, judged on wall time | ~135-480 s |
| count c8 | counting prompt, thinking off, 320 out, 2 reps | ~20 s |
| logits | `logits_equiv.py capture` twice (20 prompts x 16 greedy tokens) | ~30 s |
| benchy | llama-benchy pp2048 / tg512, depth 0, c1 and c8, T=1, 2 runs | ~2 min |

That is about 18.5 minutes per boot and 75 minutes for four boots. Both Sparks run in parallel, so two
arms also take about 75 minutes. If acceptance has already moved by more than 0.03 after pass 1, the
Spark skips pass 2 and the arm is KILL.

Why d16k c8 is different: on the 6 GiB KV pool of v2/v3a/v3b, 8 requests at 16K do not fit, so the engine
preempts and recomputes. Some requests then stall for most of their life, and per-request tok/s becomes
meaningless (one control request read 853 tok/s with 145 stalls). One rep took 134 to 441 s when it
finished, and about 40 % of runs hit the 900 s request timeout. So the cell runs once per boot, is cut at
480 s, and is judged on its wall time. An unfinished run counts as the cut.

## Verdict

`thunderdome_report.py <node dir>` writes `<node dir>/verdict.txt`. Its last line is
`VERDICT=KILL|PROMOTE|INCONCLUSIVE`, and it has `reason:` and `better:` lines. The thresholds are fixed.

Noise per cell:

- Probe cells (fresh c1/c4/c8, d16k c4, count c8). The delta is the mean paired change in decode tok/s,
  stalls excluded (`paired_decode_ab.compare`). The noise is the largest of 1 %, the control's own
  pass-2-vs-pass-1 change (boot to boot), and the half-width of the delta's 95 % CI.
- Wall cell (d16k c8). The delta is the mean over passes of control wall / arm wall - 1. The noise is the
  larger of 1 % and the control's boot-to-boot wall change.
- Benchy cells (pp2048 c1, tg512 c1, tg512 c8). The delta is the arm mean vs the control mean. The noise
  is the largest of 1 %, the control spread between its two boots, and the mean reported sd, all as a
  percentage of the control mean (the `winrule.py` rule).

Acceptance per draft position is accepted / drafted from `/metrics`, pooled over the T=0 probe cells
that finished on both sides.

| verdict | when |
|---------|------|
| KILL | any cell worse than 2x its noise, acceptance moved by more than 0.03 at any position, or an arm boot failed |
| PROMOTE | every cell measured, none worse than 1x its noise, at least one better than 1x its noise |
| INCONCLUSIVE | anything else: a cell worse than 1x but not 2x its noise, missing cells, control boot failed, a cold boot, or nothing better than noise |

The logits captures are reported, not judged. For each boot the report gives the a/b self-noise and the
arm-vs-control drift: identical prompts, and mean and max |dlogprob| before the first divergence.

PROMOTE means "worth the gate", not "ship". Promotion still follows the gate and the usual review.

Checked on old data (the report run against existing ABBA directories): v3a vs v2 gives fresh c4 +8.1 %,
fresh c8 +6.2 %, d16k c4 +7.9 % and count c8 +4.3 %, all better than noise, which matches the 2026-10-03
win-rule result. kv10 vs v3a on dgx-02 gives count c8 -5.4 % against a noise band of 2.3 %, so KILL. kv10
was rejected on 2026-10-03.

## Warm images only

- Before anything is stopped, every recipe the run will boot is checked on every Spark that boots it.
  The run refuses (exit 2) if the recipe is not TP=1, the image is missing on a Spark, or the image has
  no baked compile seed (`/opt/b12x-seed/torch_compile_cache` in `docker history`). That covers the arm
  on both Sparks when the gate is on.
- During each boot, the serve log is watched. A warm boot loads the backbone with "Directly load AOT
  compilation". A cold one starts with a Dynamo transform of the backbone, which happens when an arm
  changes a compile factor and the seed key misses. That boot is stopped at once, the boot dir gets a
  `COLD` marker, and the Spark's verdict is INCONCLUSIVE.
- `thunderdome.sh bake <recipe>...` is the one-time cold boot. It boots each recipe on both Sparks at
  once, with up to 90 minutes per boot (typically 6 to 7), and then stops them. The torch AOT and b12x
  plan caches on both hosts then hold the arm's keys, so the screen's boots are warm. Bake once per image
  or compile-factor change, not once per screen.

## Gate split

For each PROMOTE arm (unless `GATE=0`), the arm boots on both Sparks and two jobs run at once:

- dgx-01: `straggler_probe.py` (5 6 7 8 12 16), tool-eval-bench hardmode, and TC-45 x5.
- dgx-02: `fidelity_probe.py` at 8k-128k, plus 128k with seeds 11 and 13.

The serial gate took about 47 minutes on v3b (gate_arm.sh plus TC-45 and the two 128k seeds). The split
takes about 25 to 28 minutes, including both boots. The output is `<node dir>/gate/summary.txt`: hardmode
quality, TC-45 score, fidelity lines, straggler tail, and min MemAvailable per Spark. The gate is
summarized, not judged. It drops the parts of `gate_arm.sh` that are not in the TP1 gate record
(categories c1, decode_probe, sweep). To gate an arm by itself, run `thunderdome.sh gate <recipe>`.

## Usage

Everything runs on dgx-01, and dgx-02 is driven over ssh. Recipe paths can be relative. Recipes boot
from their own directory, so `mods:` paths keep working.

```bash
R=~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2

# plan and image checks only: nothing is locked, stopped or booted
ARM_A=$R/arms/x.yaml ARM_B=$R/arms/y.yaml RES=/tmp/td bash $R/scripts/thunderdome.sh --dry-run

# two arms, default v3b control, gate for PROMOTE arms, 2x restored at the end
ARM_A=$R/arms/x.yaml ARM_B=$R/arms/y.yaml RES=$R/results/td-xy-20261005 \
  setsid nohup bash $R/scripts/thunderdome.sh > /tmp/td-xy.nohup 2>&1 < /dev/null &

# one arm (dgx-02 stays idle), arm-specific control (same image, knob off)
ARM_A=$K/arm.yaml CONTROL_A=$K/ctl.yaml RES=$K/td bash $R/scripts/thunderdome.sh

# one-time cold boot for images whose compile key changed
RES=$K/td bash $R/scripts/thunderdome.sh bake $K/arm.yaml $K/ctl.yaml

# gate only
RES=$K/td bash $R/scripts/thunderdome.sh gate $K/arm.yaml
```

| variable | meaning |
|----------|---------|
| `ARM_A`, `ARM_B` | arm recipes screened on dgx-01 / dgx-02 (one may be empty) |
| `CONTROL` | control recipe for both Sparks (default v3b); `CONTROL_A` / `CONTROL_B` per Spark |
| `RES` | results dir (required); `OUT_A` / `OUT_B` override the per-Spark dirs (default `$RES/dgx01`, `$RES/dgx02`) |
| `GATE=0` | skip the gate after a PROMOTE |
| `GPU_LOCK_HELD=1` | the caller holds `~/GEN-AI/gpu-lock`; without it the script takes the lock and waits |
| `NO_RESTORE=1` or `CHAIN_NO_RESTORE=1` | leave the pair idle on exit instead of booting the 2x recipe |
| `SHIPPED_TAG` | image tag of the 2x recipe for the restore check (default: the 2x image serving at start) |

Exit codes: 0 means done (verdicts written, including KILL), 1 means failed, and 2 means refused by the
preflight, before anything was touched. `$RES/STATE` holds the state, `$RES/thunderdome.log` the log,
and `$RES/verdicts.txt` one line with every arm's verdict. To stop a run: `kill -TERM <pid>` (the pid is
in the log). The script stops its probes and servers and restores as configured. Never `pkill -f`.

Files per Spark (`$RES/dgx01`, `$RES/dgx02`):

```
arms.txt                 arm and control names and recipe paths
ctl-p1 arm-p1 arm-p2 ctl-p2/
  recipe.yaml sparkrun.log serve.log aot.txt env.txt image.txt boot.txt mem.log
  probe-<cell>.json/.log  pos-<cell>.before/.after  time-<cell>.txt   (cells: warm + 6)
  logits-a.json logits-b.json  task.csv benchy.log
  FAILED or COLD          when that boot did not come up warm
acc-p1.txt               acceptance check after pass 1
verdict.txt              report; last line VERDICT=...
gate/                    split gate of a PROMOTE arm (dgx01/, dgx02/, summary.txt)
```

## Calling it from a k31 stage script

The k31 chain (`~/GEN-AI/k31/chain.sh`) runs `~/GEN-AI/k3N/*_job.sh` with
`GPU_LOCK_HELD=1 CHAIN_NO_RESTORE=1 RES=<stage dir>`. Thunderdome inherits both flags. It does not take
the lock, and it does not boot the 2x between stages, because the chain restores it once at the end.

A stage with two arms, or an arm and a variant, puts one on each Spark:

```bash
#!/bin/bash
# ~/GEN-AI/k3N/<x>_job.sh, run by the k31 chain
set -u
R=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2; K=$HOME/GEN-AI/k3N
st() { echo "$*" > "$RES/STATE"; }
# 1. build the image on both Sparks; write $K/arm-a.yaml and $K/arm-b.yaml (TP=1, container: <image>)
# 2. only if the image or a compile factor changed since the last bake:
RES=$RES/td bash "$R/scripts/thunderdome.sh" bake "$K/arm-a.yaml" "$K/arm-b.yaml" || { st "FAILED: bake"; exit 1; }
# 3. screen both arms in one run; PROMOTE arms are gated in the same run
ARM_A=$K/arm-a.yaml ARM_B=$K/arm-b.yaml RES=$RES/td bash "$R/scripts/thunderdome.sh"
case $? in 0) st "DONE: $(cat "$RES/td/verdicts.txt")" ;; 2) st "FAILED: thunderdome refused by its preflight (see the job output)" ;; *) st "FAILED: thunderdome" ;; esac
```

If the arm's image adds code whose off path should match v3b, also pass the same-image control
(`CONTROL_A=$K/ctl-a.yaml`, the knob off). That way the screen compares the knob and not the image.

A stage with only one arm can share the run with another stage's arm, so the other Spark is not idle:

1. A stage whose arm is ready (image on both Sparks, baked) publishes it as `~/GEN-AI/k3M/thunderdome-arm.yaml`,
   plus an optional `thunderdome-ctl.yaml`. Write the file last (`mv` into place).
2. The stage that runs first takes the next published arm that has no verdict yet:

   ```bash
   B=; for o in k33 k34 k35; do d=$HOME/GEN-AI/$o
     [ "$d" != "$K" ] && [ -s "$d/thunderdome-arm.yaml" ] && [ ! -s "$d/thunderdome/verdict.txt" ] && { B=$d; break; }; done
   [ -n "$B" ] && [ -s "$B/thunderdome-ctl.yaml" ] && export CONTROL_B=$B/thunderdome-ctl.yaml
   env ARM_A=$K/thunderdome-arm.yaml OUT_A=$K/thunderdome ${B:+ARM_B=$B/thunderdome-arm.yaml OUT_B=$B/thunderdome} \
     RES=$RES/td bash "$R/scripts/thunderdome.sh"
   ```

3. When the owning stage's turn comes, it checks `[ -s $K/thunderdome/verdict.txt ]` and skips its own
   screen. If the arm was PROMOTE, the gate already ran into `$K/thunderdome/gate/`.

Reading the result in a stage:

```bash
v=$(sed -n 's/^VERDICT=//p' "$RES/td/dgx01/verdict.txt")   # KILL | PROMOTE | INCONCLUSIVE
grep -E '^(reason|better):' "$RES/td/dgx01/verdict.txt"
cat "$RES/td/dgx01/gate/summary.txt" 2>/dev/null            # only after a PROMOTE
```

On INCONCLUSIVE, read the reasons. A cold boot means bake and rerun. A cell between 1x and 2x noise
means the full ABBA (`r8_tp1_driver.sh`) is needed for that arm.

## Limits

- At c1 there are only 4 request pairs per arm (2 reps x 2 passes). c1 effects below about 3 % read as
  flat. Use the full ABBA for small c1 changes.
- The d16k c8 wall cell has one run per boot. On the 6 GiB control its boot-to-boot noise is large, so it
  only decides large KV changes. kv10 vs v2 was +397 % in benchy tg512 at 16K c8.
- The acceptance rule is two-sided. A drafter change that raises T=0 acceptance by more than 0.03 at any
  position also reads KILL. Read the acceptance lines before dropping such an arm.
- llama-benchy runs at T=1 with 2 runs per boot. Its noise band is wider than the probes'.
- The gate is summarized, not judged.
