k104 result: I ran the release quality gate of v2.1.0 on the 2x with GDN prefill views (`VLLM_GDN_PREFILL_VIEWS=1`, vLLM `10d0acc85f` files on the v2.1.0 image), one TP=2 boot. Its TP=2 A/B against v2.1.0 already passed (cold TTFT 2.2 to 2.5% lower).

Gate: `PASS`.

```
k104 2026-10-16 12:01:54 EEST: release quality gate on the 2x v2.1.0 + VLLM_GDN_PREFILL_VIEWS=1 (spark-vllm-b12x:k101-gv-10d0acc8-d21d7ade), one TP=2 boot
GATE=PASS
views lines rank 0 1 / rank 1 1; rank 0 plans measured 0; Traceback 0/0; boot 172s Directly load AOT compilation; pong pong
│    Quality:        90/100                                                    │
│    Score:  100 / 100                                                         │
│    Score:   100.0 ± 0.0 / 100                                                │
depth    8000 | actual_tokens   15709 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   0.82s | lat    2.8s
depth   32000 | actual_tokens   61730 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   1.93s | lat    3.3s
depth   64000 | actual_tokens  122614 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   3.83s | lat    5.3s
depth  128000 | actual_tokens  245267 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   7.74s | lat    9.3s
depth  128000 | actual_tokens  244166 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   7.06s | lat    8.3s
depth  128000 | actual_tokens  244893 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   7.52s | lat    9.1s
c=7 round wall 3.9s | per request (submit order): #0:3.87s/320t #1:3.87s/320t #2:3.87s/320t #3:3.87s/320t #4:3.93s/320t #5:3.93s/320t #6:3.93s/320t
      preemptions +0 | queue time total 0.00s over 7 req | prefill total 1.14s | decode total 26.1s | accept 4.00/draft
c=8 round wall 4.1s | per request (submit order): #0:4.03s/320t #1:4.09s/320t #2:4.03s/320t #3:4.03s/320t #4:4.09s/320t #5:4.03s/320t #6:4.09s/320t #7:4.09s/320t
      preemptions +0 | queue time total 0.01s over 8 req | prefill total 1.42s | decode total 31.1s | accept 4.00/draft
c=12 round wall 5.2s | per request (submit order): #0:5.08s/320t #1:5.08s/320t #2:5.08s/320t #3:5.08s/320t #4:5.16s/320t #5:5.16s/320t #6:5.16s/320t #7:5.16s/320t #8:5.23s/320t #9:5.23s/320t #10:5.23s/320t #11:5.23s/320t
      preemptions +0 | queue time total 0.41s over 12 req | prefill total 2.93s | decode total 59.0s | accept 4.00/draft
c=16 round wall 6.0s | per request (submit order): #0:5.76s/320t #1:5.76s/320t #2:5.85s/320t #3:5.76s/320t #4:5.85s/320t #5:5.76s/320t #6:5.85s/320t #7:5.85s/320t #8:5.92s/320t #9:5.92s/320t #10:5.92s/320t #11:5.92s/320t #12:5.99s/320t #13:5.99s/320t #14:5.99s/320t #15:5.99s/320t
      preemptions +0 | queue time total 0.61s over 16 req | prefill total 5.08s | decode total 89.0s | accept 4.00/draft
```

Before a release the image still needs the warm build and a fresh-user boot check. Results: {RESULTS_URL}
