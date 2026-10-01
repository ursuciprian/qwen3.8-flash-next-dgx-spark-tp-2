| Task | Mean checks passed | Clean runs | Runaway thinking | Median time (s) | Max time (s) | Median time to answer (s) | Median completion tokens | Failed checks (all runs) |
|---|---|---|---|---|---|---|---|---|
| 01-tf-s3-module | 91% | 1/3 | 0/3 | 24 | 31 | 16.95 | 2263 | lifecycle: noncurrent expiry var + abort MPU 7d; terraform fmt -check; terraform validate |
| 02-tf-fix-validate | 90% | 1/3 | 0/3 | 25 | 44 | 19.75 | 2359 | explains errors (>=4 listed) |
| 03-k8s-deployment-fix | 100% | 3/3 | 0/3 | 17 | 20 | 12.67 | 1571 | - |
| 04-k8s-oomkilled | 100% | 3/3 | 0/3 | 20 | 20 | 12.65 | 1556 | - |
| 05-gha-fix | 88% | 0/3 | 0/3 | 41 | 64 | 35.69 | 3407 | permissions block present |
| 06-gha-oidc-terraform | 100% | 3/3 | 0/3 | 37 | 50 | 28.53 | 3352 | - |
| 07-iam-least-privilege | 100% | 3/3 | 0/3 | 37 | 41 | 24.01 | 3092 | - |
| 08-bash-pg-backup | 92% | 1/3 | 0/3 | 62 | 75 | 50.32 | 5514 | --keep 3 retention after 5 runs; failed dump: non-zero exit, no partial file |
| 09-helm-values-fix | 100% | 3/3 | 0/3 | 13 | 18 | 9.24 | 1176 | - |
| 10-dockerfile-harden | 92% | 1/3 | 0/3 | 79 | 112 | 65.66 | 6603 | hadolint (no warnings/errors; DL3008/DL3018 ignored); no remote ADD, no curl install |
| 11-tf-plan-review | 90% | 1/3 | 0/3 | 37 | 38 | 13.83 | 2723 | safe path: revert identifier / snapshot / prevent_destroy; verdict: do not apply |
| 12-k8s-networkpolicy | 100% | 3/3 | 0/3 | 22 | 27 | 14.31 | 2117 | - |
| 13-prometheus-rules | 100% | 3/3 | 0/3 | 20 | 21 | 15.95 | 1778 | - |
| 14-incident-db-pool | 100% | 3/3 | 0/3 | 47 | 47 | 19.58 | 3494 | - |
| **All (42 runs)** | **95.9%** | **29/42** | **0/42 (0%)** | 34 | 112 | | | |
