#!/usr/bin/env python3
"""Minimal prefix-affinity router for DP=2 (k60, 2026-10-07): one OpenAI-compatible endpoint in front of N replicas.

A request goes to replica sha1(key) mod N, where key = the first two non-system messages of the conversation (or the
first 4096 chars of a completion prompt). Agent conversations only append, so every turn of a session lands on the
replica that already holds its prefix cache. The system prompt and tools are left out of the key: agents share them,
and keying on them would send every session to one replica.
k63 change (2026-10-07): sticky least-sessions instead of hash mod N. k60 agent16 put 13 of 16 sessions on one Spark with
the static hash. Now the first request of a conversation goes to the replica with the fewest conversations so far and
every later turn (same key) sticks to it, so prefix-cache affinity is unchanged.
ponytail: the key->replica map never expires and ignores live load/health; add expiry + least-in-flight if it matters.

  pa_router.py --port 8100 --backends http://192.168.100.62:8000,http://192.168.100.53:8000
  GET /router/stats -> {"requests": [n0, n1], "sessions": [s0, s1]}; GET /health -> 200 when every replica is healthy.
  pa_router.py --selftest
"""
import argparse, hashlib, http.client, json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

NCHARS = 4096
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host", "proxy-connection", "upgrade", "te", "trailer"}


def key(body):
    msgs = body.get("messages")
    if isinstance(msgs, list):
        return json.dumps([m for m in msgs if m.get("role") != "system"][:2], sort_keys=True)[:NCHARS]
    return str(body.get("prompt", ""))[:NCHARS]


def pick(body, n):
    return int(hashlib.sha1(key(body).encode()).hexdigest(), 16) % n


class Router(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # close-delimited bodies: streaming passes through chunk by chunk
    backends, lock, reqs, sess, assign = [], threading.Lock(), [], [], {}

    def log_message(self, *a):
        pass

    def forward(self, i, body):
        u = urlsplit(self.backends[i])
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=7200)
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        try:
            conn.request(self.command, self.path, body=body, headers=hdrs)
            r = conn.getresponse()
        except OSError as ex:
            self.send_error(502, f"backend {i}: {ex}"); return
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("X-Router-Backend", str(i)); self.end_headers()
        try:
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = {}
        h = hashlib.sha1(key(parsed).encode()).hexdigest()
        with self.lock:
            i = self.assign.get(h)
            if i is None:
                i = min(range(len(self.backends)), key=lambda j: (len(self.sess[j]), j))
                self.assign[h] = i
            self.reqs[i] += 1
            self.sess[i].add(h)  # distinct conversations seen per replica
        self.forward(i, body)

    def do_GET(self):
        if self.path == "/router/stats":
            b = json.dumps({"backends": self.backends, "requests": self.reqs, "sessions": [len(x) for x in self.sess]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)
        elif self.path == "/health":
            ok = True
            for b in self.backends:
                u = urlsplit(b)
                try:
                    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=5); c.request("GET", "/health")
                    ok = ok and c.getresponse().status == 200; c.close()
                except OSError:
                    ok = False
            self.send_response(200 if ok else 503); self.end_headers()
        else:
            self.forward(0, None)


def serve(port, backends):
    Router.backends, Router.reqs, Router.sess, Router.assign = backends, [0] * len(backends), [set() for _ in backends], {}
    s = ThreadingHTTPServer(("0.0.0.0", port), Router); s.daemon_threads = True
    return s


def selftest():
    import time, urllib.request
    sysmsg = {"role": "system", "content": "shared agent prompt"}
    a = [sysmsg, {"role": "user", "content": "session A"}, {"role": "assistant", "content": "x"}]
    assert pick({"messages": a}, 2) == pick({"messages": a + [{"role": "tool", "content": "more"}] * 3}, 2)
    picks = {pick({"messages": [sysmsg, {"role": "user", "content": f"s{i}"}]}, 2) for i in range(20)}
    assert picks == {0, 1}, picks

    class Echo(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def do_POST(self):
            n = int(self.headers["Content-Length"]); self.rfile.read(n)
            self.send_response(200); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            for part in (b"data: 1\n\n", b"data: [DONE]\n\n"):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part)); self.wfile.flush(); time.sleep(0.05)
            self.wfile.write(b"0\r\n\r\n")
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()
    be = [ThreadingHTTPServer(("127.0.0.1", 0), Echo) for _ in range(2)]
    for s in be:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    rt = serve(0, [f"http://127.0.0.1:{s.server_port}" for s in be])
    threading.Thread(target=rt.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{rt.server_port}"
    body = json.dumps({"messages": [sysmsg, {"role": "user", "content": "s1"}], "stream": True}).encode()
    r = urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"}))
    assert r.read() == b"data: 1\n\ndata: [DONE]\n\n" and r.headers["X-Router-Backend"] in ("0", "1")
    assert urllib.request.urlopen(base + "/health").status == 200
    st = json.load(urllib.request.urlopen(base + "/router/stats"))
    assert sum(st["requests"]) == 1 and sum(st["sessions"]) == 1, st
    def post(c):
        b = json.dumps({"messages": [sysmsg, {"role": "user", "content": c}]}).encode()
        return urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", b, {"Content-Type": "application/json"})).headers["X-Router-Backend"]
    first = {c: post(c) for c in [f"k{i}" for i in range(16)]}
    assert all(post(c) == b for c, b in first.items())             # sticky: a conversation keeps its replica
    st = json.load(urllib.request.urlopen(base + "/router/stats"))
    assert sorted(st["sessions"]) == [8, 9], st                     # 17 conversations split 8/9, not by hash
    for s in be + [rt]:
        s.shutdown()
    print("router selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100); ap.add_argument("--backends", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    srv = serve(a.port, [b.rstrip("/") for b in a.backends.split(",") if b])
    print(f"routing :{a.port} -> {Router.backends}", flush=True)
    srv.serve_forever()
