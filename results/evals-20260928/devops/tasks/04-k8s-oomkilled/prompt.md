A Java service in production keeps restarting. Diagnose the root cause from the output below, explain it briefly, and give the fix as the corrected container spec (the `containers:` entry for `orders`, including `env` and `resources`) in one ```yaml fenced block. Keep the memory request equal to the limit.

```
$ kubectl -n orders get pods
NAME                      READY   STATUS             RESTARTS        AGE
orders-6d9f8b7c5d-x2kqp   0/1     CrashLoopBackOff   14 (2m1s ago)   47m

$ kubectl -n orders describe pod orders-6d9f8b7c5d-x2kqp
...
Containers:
  orders:
    Image:          registry.example.com/orders:3.2.1
    State:          Waiting
      Reason:       CrashLoopBackOff
    Last State:     Terminated
      Reason:       OOMKilled
      Exit Code:    137
      Started:      Sun, 28 Sep 2026 09:12:03 +0000
      Finished:     Sun, 28 Sep 2026 09:13:41 +0000
    Restart Count:  14
    Limits:
      cpu:     1
      memory:  512Mi
    Requests:
      cpu:     500m
      memory:  512Mi
    Environment:
      JAVA_OPTS:  -Xms1g -Xmx1g -XX:+UseG1GC
...
Events:
  Warning  BackOff  2m (x190 over 45m)  kubelet  Back-off restarting failed container orders

$ kubectl -n orders logs orders-6d9f8b7c5d-x2kqp --previous | tail -3
2026-09-28 09:13:38 INFO  OrderCacheWarmer - warmed 180000 of 400000 entries
2026-09-28 09:13:40 INFO  OrderCacheWarmer - warmed 210000 of 400000 entries
2026-09-28 09:13:41 INFO  OrderCacheWarmer - warmed 240000 of 400000 entries
```
