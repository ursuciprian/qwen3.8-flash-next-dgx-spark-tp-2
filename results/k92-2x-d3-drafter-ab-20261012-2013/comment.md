k92-2x-d3-drafter-ab result: I ran drafter D3 (checkpoint revision `03f4a057`) against the shipped 2x v2.0.0 (D1, `16c9bd54`) on the same image and the same plan selections (the control's TP=2 plan re-keyed to the D3 snapshot path, 0 plans measured on every counted boot). TP=2 on the pair, 4 boots per side in the order ctl:1 arm:1 arm:2 ctl:2 arm:3 ctl:3 ctl:4 arm:4. T=0 probes fresh c1/c4/c8 and d16k-c4, llama-benchy pp2048 + tg512 c1/c4/c8/c16.

Screen rule: `PASS` (PASS: at least one cell better beyond its noise band and no cell from c1 to c16 worse beyond it). Gate: PASS. hardmode 94 (ok); long-context retrieval 8000:20/20 32000:20/20 64000:20/20 128000:20/20 128000:20/20 128000:20/20 (ok); stragglers c8/c12/c16 {'8': 'preempt 0, max/median 1.00', '12': 'preempt 0, max/median 1.01', '16': 'preempt 0, max/median 1.01'} (none).

MTP acceptance per draft position (T=0 probe cells, pooled, D1 then D3):
```
  pos 0: control 0.840 arm 0.853 shift +0.013
  pos 1: control 0.686 arm 0.702 shift +0.016
  pos 2: control 0.555 arm 0.576 shift +0.021
  pos 3: control 0.444 arm 0.469 shift +0.025
```

Cells (D3 vs D1, noise band) and the boot-level Welch 95% CI of the benchy cells:
```
  probe  fresh-c1      -1.30%  noise  4.11%            n=8 CI [-5.3,+2.9] control boot-to-boot +0.14%, step -0.46%
  probe  fresh-c4      +4.64%  noise  5.26%            n=32 CI [+1.9,+7.4] control boot-to-boot +5.26%, step -0.28%
  probe  fresh-c8      +1.84%  noise  1.16%  better    n=64 CI [+0.7,+3.0] control boot-to-boot +0.44%, step -0.58%
  probe  d16k-c4       +3.68%  noise  2.20%  better    n=32 CI [+1.6,+5.8] control boot-to-boot -2.20%, step -0.68%
  benchy pp2048 (c1)   -0.77%  noise  2.73%            control 2797.3 arm 2775.8 t/s (boots 4/4)
  benchy tg512 (c1)    +0.89%  noise 11.94%            control 83.0 arm 83.8 t/s (boots 4/4)
  benchy tg512 (c4)    +3.13%  noise  8.16%            control 157.0 arm 161.9 t/s (boots 4/4)
  benchy tg512 (c8)    +2.62%  noise  5.15%            control 186.0 arm 190.9 t/s (boots 4/4)
  benchy tg512 (c16)   +3.07%  noise  7.48%            control 233.7 arm 240.9 t/s (boots 4/4)
  pp2048 (c1)   -0.77%  CI [-5.05,+3.52]  control 2775.2 2799.9 2769.6 2844.4 | arm 2793.4 2659.2 2820.9 2829.7
  tg512 (c1)    +0.89%  CI [-11.24,+13.02]  control 83.4 86.1 77.9 84.7 | arm 86.4 74.5 88.7 85.4
  tg512 (c4)    +3.13%  CI [-8.86,+15.12]  control 151.9 152.9 159.0 164.3 | arm 146.8 157.9 174.6 168.5
  tg512 (c8)    +2.62%  CI [-1.47,+6.71]  control 190.5 182.7 189.9 180.9 | arm 188.5 189.4 189.8 195.8
  tg512 (c16)   +3.07%  CI [-1.95,+8.08]  control 235.6 231.7 242.5 225.0 | arm 246.1 234.3 238.5 244.5
better: probe fresh-c8 +1.84% (noise 1.16%); probe d16k-c4 +3.68% (noise 2.20%)
worse: none
better: probe  fresh-c8      +1.84%  noise  1.16%  better    n=64 CI [+0.7,+3.0] control boot-to-boot +0.44%, step -0.58%; probe  d16k-c4       +3.68%  noise  2.20%  better    n=32 CI [+1.6,+5.8] control boot-to-boot -2.20%, step -0.68%
```

D3 meets my rule on the 2x and passed the gate. This is the evidence for a 2x v2.1.0 (drafter only); nothing ships from this job.

Results: {RESULTS_URL}
