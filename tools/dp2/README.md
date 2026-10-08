# DP=2: one 1× copy per Spark behind a router

Two Sparks can serve this model as one TP=2 cluster (the 2× recipe) or as two independent 1× replicas behind
[`pa_router.py`](pa_router.py). Which one to pick: [README](../../README.md#two-sparks-one-cluster-or-one-copy-per-spark).
This is the setup measured in k72 ([results](../../results/dp2-gate-k72-20261008-1135/)).

## Run it

1. Start the shipped [1× recipe](https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark) on each Spark with plain sparkrun (checkpoint downloaded first, see its
   [quick start](https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark#quick-start)). With both this registry and the 1× one added, scope the name as
   `@qwen38-flashnext-1x/qwen3.8-flash-next-1x-dgx-spark`; with only this registry, the compatibility copy here boots
   the same recipe:

   ```sh
   sparkrun run qwen3.8-flash-next-1x-dgx-spark --hosts <spark-a> --solo
   sparkrun run qwen3.8-flash-next-1x-dgx-spark --hosts <spark-b> --solo
   ```

2. Wait until both answer `GET /health` with 200 (see [Verify](../../README.md#verify)).

3. Start the router on either Spark or any host that reaches both. Python 3 stdlib only, nothing to install:

   ```sh
   python3 tools/dp2/pa_router.py --backends http://<spark-a>:8000,http://<spark-b>:8000
   ```

   Clients use `http://<router-host>:8100/v1` and model `qwen3.8-flash-next`, as with a single server.

k72 ran the router on dgx-01 and pointed it at the CX-7 addresses of both Sparks; the LAN addresses work the same.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--backends` | required | Comma-separated replica base URLs |
| `--host` | `0.0.0.0` | Listen address |
| `--port` | `8100` | Listen port |
| `--selftest` | | Runs the routing and streaming checks against two local stub servers, then exits |

There are no environment variables.

## How it routes

- The key of a request is the sha1 of its first two non-system messages (for `/v1/completions`, the first 4,096
  characters of the prompt). The system prompt and tools are left out, because agents share them.
- A key the router has not seen goes to the replica with the fewest conversations so far. Every later request with
  that key goes to the same replica, so each turn of a conversation hits the replica that holds its prefix cache.
- Streaming responses pass through chunk by chunk. Each response carries `X-Router-Backend: <index>`.
- `GET /health`: 200 only if every replica answers its own `/health` with 200 within 5 s, else 503.
- `GET /router/stats`: requests and conversations per replica.
- Any other `GET` (for example `/v1/models`) goes to replica 0.

## Known limits

- No failover. The router does not check health before routing: if a replica is down, its conversations and every
  new conversation assigned to it get a 502 until it is back. `/health` turns 503, so a supervisor can watch it.
- No `/metrics`. Scrape `http://<spark>:8000/metrics` on each replica.
- The key to replica map lives in memory and never expires; a restart resets it and the first turn after a restart
  can land on the replica without the prefix cache. Memory grows by one entry per conversation.
- Balance is by conversation count, not by live load. One very long conversation and one short one count the same.
- A client that rewrites the start of its history (context compaction, a new first user message) becomes a new
  conversation and can move to the other replica, with a cold prefix cache there.
- No TLS, no API key, no request limits, and it listens on all interfaces by default. Keep it on a trusted network,
  or bind `--host 127.0.0.1` and put an authenticating proxy in front.
- One thread per connection, responses close the connection (HTTP/1.0), backend timeout 2 h per request.
