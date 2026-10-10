#!/usr/bin/env bash
# k71 staging (2026-10-08, #115): the k70 arm (snapshot f4..70, image k70-b14-refit-run1-c92ac62d, arm/k70.yaml) with
# the control's b12x plan selections pinned into its runtime cache on both Sparks (pin_plans.py). CPU only, idempotent.
set -eu
K=$HOME/GEN-AI/backlog/k71; H2=<cx7-ip-b>
C=$HOME/.cache/sparkrun/runtime-cache/vllm
ARMF=$C/f400000000000000000000000000000000000070-d93a4e42/b12x/compile/preparation/17e6fd5eaf392f3eff3a6bfad5f0c72eb03a0f688348e99d5773e92da553c215.json
CTLF=$C/local-inference-lab__Qwen3.8-Flash-Next-NVFP4-2d9615ab/b12x/compile/preparation/8ccf4799746f12aa17e1516b1378f0474077caca591570caa40dd18dc68b7db2.json
bash "$HOME/GEN-AI/backlog/k70/stage.sh"
python3 "$K/pin_plans.py" "$ARMF" "$CTLF" "$K/plans-k70-dgx01.json" | sed 's/^/dgx-01: /'
ssh -n -o ConnectTimeout=10 $H2 "mkdir -p $K"; scp -q "$K/pin_plans.py" "$H2:$K/"
ssh -n -o ConnectTimeout=10 $H2 "python3 $K/pin_plans.py $ARMF $CTLF $K/plans-k70-dgx02.json" | sed 's/^/dgx-02: /'
