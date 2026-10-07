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
arms also take about 75 minutes. Worst case, when d16k c8 hits its 480 s cut on every boot, a boot takes
about 21 minutes and the screen about 85 minutes. `bake: yes` adds about 14 minutes and a `hook` adds its
own time. If acceptance has already moved by more than 0.03 after pass 1, the
Spark skips pass 2 and the arm is KILL.

Why d16k c8 is different: on the 6 GiB KV pool of v2/v3a/v3b, 8 requests at 16K do not fit, so the engine
preempts and recomputes. Some requests then stall for most of their life, and per-request tok/s becomes
meaningless (one control request read 853 tok/s with 145 stalls). One rep took 134 to 441 s when it
finished, and 19 of 56 runs (34 %) in the last 14 TP1 driver logs hit the 900 s request timeout. So the cell runs once per boot, is cut at
480 s, and is judged on its wall time. A run cut by the timeout counts as the cut. A run that errors out
counts as missing.

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
| KILL | any cell worse than 2x its noise; acceptance moved by more than 0.03 at any draft position both sides have; an arm boot failed; the arm server died during a boot; or an arm cell timed out or errored in a pass where the control's finished |
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

Everything runs on dgx-01, and dgx-02 is driven over ssh. Thunderdome never takes `~/GEN-AI/gpu-lock`. The
caller holds it and sets `GPU_LOCK_HELD=1`, and the script refuses to run without that. The k31 chain does
this for its stages. By hand, use `flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 ...`.

```bash
R=~/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2

# two arm dirs (first on dgx-01, second on dgx-02), as the k31 chain calls it
RES=$R/results/thunderdome-k32-k34-20261005 bash $R/scripts/thunderdome.sh ~/GEN-AI/k32 ~/GEN-AI/k34

# plan, spec and image checks only: nothing is stopped, booted or written
RES=/tmp/td bash $R/scripts/thunderdome.sh ~/GEN-AI/k32 ~/GEN-AI/k34 --dry-run

# by hand, with recipes instead of dirs (default v3b control)
flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 ARM_A=x.yaml ARM_B=y.yaml RES=/tmp/td-xy bash $R/scripts/thunderdome.sh

# one-time cold boot on both Sparks; gate only
RES=$K/td bash $R/scripts/thunderdome.sh bake $K/arm.yaml
RES=$K/td bash $R/scripts/thunderdome.sh gate $K/arm.yaml
```

### Arm dir

An arm dir holds `thunderdome.arm`, written as `key: value` lines with `#` comments. Thunderdome turns it
into an arm recipe and a control recipe in `$RES/recipes/`. The arm's results go to `$RES/<dir name>/`.

| key | meaning |
|-----|---------|
| `name` | label for recipes and logs (default: the dir name) |
| `image` | arm = the base recipe on this image ... |
| `env` | ... plus `KEY=VALUE` in its env block (repeatable; replaces a key the base already has) |
| `control_image` | control = the base recipe on this image, for a same-image control. Default: the base as is |
| `control_env` | extra `KEY=VALUE` for the control (repeatable) |
| `base` | base recipe (default: the v3b recipe) |
| `recipe`, `control_recipe` | full recipes instead, for arms that change more than image and env (paths relative to the dir) |
| `bake: yes` | before the screen, cold-boot control and arm once on their Spark (the compile key changed). Adds ~14 min. The gate allows a cold boot on the other Spark |
| `hook` | script run after each boot's probes as `bash <hook> <host> <boot dir>`, capped at 1 h. Its time adds to the screen |

Example, `~/GEN-AI/k32/thunderdome.arm`:

```
name: tp1-v3b-heads4
image: spark-vllm-b12x:r14h4-5bf24021-e070a14c
env: VLLM_LM_HEAD_NVFP4=mse
control_image: spark-vllm-b12x:r14h4-5bf24021-e070a14c
bake: yes
hook: heads_probe.sh
```

A dir without `thunderdome.arm` can instead publish a recipe pair: `thunderdome-arm.yaml` and, optionally,
`thunderdome-ctl.yaml` (the default control is v3b). The pair is used as is. k33's `prep.sh` does this.

The preflight checks each arm on its own: spec, TP=1, images on the Sparks that boot them, and the compile
seed (skipped with `bake: yes`). A failing arm is dropped, its `verdict.txt` says why
(`VERDICT=INCONCLUSIVE`), and the other arm still runs. The run is refused (exit 2) only when no arm is
left or a shared check fails. Even then, `$RES/STATE` says why.

| variable | meaning |
|----------|---------|
| `RES` | results dir (required) |
| `GPU_LOCK_HELD=1` | required: the caller holds `~/GEN-AI/gpu-lock` |
| `ARM_A`, `ARM_B` | recipe mode instead of dirs: arms for dgx-01 / dgx-02 |
| `CONTROL`, `CONTROL_A`, `CONTROL_B` | recipe mode controls (default v3b) |
| `OUT_A`, `OUT_B` | recipe mode per-Spark dirs (default `$RES/dgx01`, `$RES/dgx02`) |
| `GATE=0` | skip the gate after a PROMOTE |
| `NO_RESTORE=1` or `CHAIN_NO_RESTORE=1` | leave the pair idle on exit instead of booting the 2x recipe |
| `SHIPPED_TAG` | image tag of the 2x recipe for the restore check (default: the 2x image serving at start) |

Exit codes: 0 means done (verdicts written, including KILL), 1 means failed, and 2 means refused.
Once RES is known, `$RES/STATE` holds the state on every path (a usage error before that only prints), `$RES/thunderdome.log` the log, and `$RES/verdicts.txt` one
line with every arm's verdict. To stop a run: `kill -TERM <pid>` (the pid is in the log). The script stops
its probes and servers and restores as configured. Never `pkill -f`.

Files per arm (`$RES/<dir name>` or `$RES/dgx0N`):

```
arms.txt                 arm and control names and recipe paths
bake-ctl/ bake-arm/      the cold boots, with bake: yes
ctl-p1 arm-p1 arm-p2 ctl-p2/
  recipe.yaml sparkrun.log serve.log aot.txt env.txt image.txt boot.txt mem.log
  probe-<cell>.json/.log  pos-<cell>.before/.after  time-<cell>.txt   (cells: warm + 6)
  logits-a.json logits-b.json  task.csv benchy.log  hook.log
  FAILED or COLD          when that boot did not come up warm; DIED when the server was gone after the pass
acc-p1.txt               acceptance check after pass 1
verdict.txt              report; last line VERDICT=...
gate/                    split gate of a PROMOTE arm (dgx01/, dgx02/, summary.txt)
```

## With the k31 chain

The chain pairs stages two at a time: k32 heads with k34 drafter, then k33 PLE graph with k35 attention
block. An arm takes part when `~/GEN-AI/kNN/READY` exists. The optional `~/GEN-AI/kNN/prep.sh` runs first
as its own stage, for example to build the image. Then the chain calls
`RES=<dir> thunderdome.sh <kNN dir> [<kNN dir>]` with `GPU_LOCK_HELD=1 CHAIN_NO_RESTORE=1`, and reads
`$RES/STATE`. The 2x recipe is not booted between pairs, because the chain restores it once at the end.

A stage that wants to take part needs these in its dir:

- `thunderdome.arm`.
- A way to get its image onto both Sparks before the screen: built already, or built by `prep.sh`.
- `READY`.

Reading the result:

```bash
v=$(sed -n 's/^VERDICT=//p' "$RES/k32/verdict.txt")   # KILL | PROMOTE | INCONCLUSIVE
grep -E '^(reason|better):' "$RES/k32/verdict.txt"
cat "$RES/k32/gate/summary.txt" 2>/dev/null            # only after a PROMOTE
```

On INCONCLUSIVE, read the reasons. A cold boot means set `bake: yes` or bake, then rerun. A cell between
1x and 2x noise means the full ABBA (`r8_tp1_driver.sh`) is needed for that arm. Arm-specific quality
checks (for example the k32 held-out top-1) run through `hook`. Thunderdome does not judge them, so a
PROMOTE does not cover them: whoever reads the verdict (or a post-screen step in the chain) must run the
stage's own check on the boot dirs before the arm is promoted.

## Limits

- At c1 there are only 4 request pairs per arm (2 reps x 2 passes). c1 effects below about 3 % read as
  flat. Use the full ABBA for small c1 changes.
- The d16k c8 wall cell has one run per boot. On the 6 GiB control its boot-to-boot noise is large, so it
  only decides large KV changes. kv10 vs v2 was +397 % in benchy tg512 at 16K c8.
- The acceptance rule is two-sided. A drafter change that raises T=0 acceptance by more than 0.03 at any
  position also reads KILL. Read the acceptance lines before dropping such an arm.
- `TD_ACC_RISE_OK=1` (opt-in, set per run for an arm whose intended effect is a higher acceptance): a rise past
  0.03 is printed but is not a KILL reason, so pass 2 runs and the verdict rests on the speed cells and the gate.
  Drops past 0.03 still KILL. Unset, the rule stays two-sided.
- llama-benchy runs at T=1 with 2 runs per boot. Its noise band is wider than the probes'.
- The gate is summarized, not judged.
