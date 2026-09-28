Production incident, 14:05 UTC: the `checkout` API is returning HTTP 500 for about 40% of requests. Using the evidence below, give (1) the root cause with the specific numbers that prove it, (2) a short timeline, (3) the immediate mitigation you would run now, and (4) the long-term fix.

```
# Deploy log
14:02:11 argocd  app=checkout  sync OK  revision=9f3c2e1  "scale for campaign: replicas 4 -> 12"

# checkout values (unchanged in 9f3c2e1 apart from replicas)
replicas: 12
env:
  SPRING_DATASOURCE_HIKARI_MAXIMUM_POOL_SIZE: "50"

# checkout pod logs (several pods)
14:03:40 WARN  HikariPool-1 - Pool stats (total=31, active=31, idle=0, waiting=187)
14:03:52 ERROR HikariPool-1 - Connection is not available, request timed out after 30000ms.
14:04:05 ERROR o.h.e.j.s.SqlExceptionHelper - FATAL: remaining connection slots are reserved for non-replication superuser connections

# PostgreSQL (RDS, db.r6g.2xlarge)
SHOW max_connections;      -> 500
SELECT count(*) FROM pg_stat_activity;  -> 497
# other clients: reporting-service 20 connections, migrations job 0, rdsadmin 3

# CPU on the database: 31%.  Read IOPS: normal.  Replica lag: 0.
```
