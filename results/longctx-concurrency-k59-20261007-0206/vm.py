"""vm.py (backlog, 2026-10-07): vLLM /metrics and memory-guard helpers shared by the k59/k60 drivers."""
import re, statistics, threading, time, urllib.request

COUNTERS = ("vllm:num_preemptions_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
            "vllm:generation_tokens_total", "vllm:prompt_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total")
GAUGES = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "vllm:num_requests_running",
          "vllm:num_requests_waiting")
LINE = re.compile(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN)$")


def parse(text):
    """metric name -> value summed over label sets (counters and gauges we know; *_created lines skipped)."""
    out = {}
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m and m.group(1) in COUNTERS + GAUGES and m.group(3) != "NaN":
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def scrape(url, timeout=10):
    try:
        return parse(urllib.request.urlopen(url, timeout=timeout).read().decode())
    except Exception:
        return {}


def delta(a, b):
    """counter deltas; all None when either scrape failed (an empty snapshot would give lifetime totals)."""
    if not a or not b:
        return {k: None for k in COUNTERS}
    return {k: b.get(k, 0.0) - a.get(k, 0.0) for k in COUNTERS}


def div(a, b):
    return a / b if a is not None and b else None


class Poller(threading.Thread):
    """Max of each gauge per URL, sampled every `every` s, until stop()."""
    def __init__(self, urls, every=5.0):
        super().__init__(daemon=True)
        self.urls, self.every, self.ev = list(urls), every, threading.Event()
        self.max = {u: {} for u in self.urls}

    def run(self):
        while not self.ev.is_set():
            for u in self.urls:
                for k, v in scrape(u, timeout=4).items():
                    if k in GAUGES:
                        self.max[u][k] = max(self.max[u].get(k, 0.0), v)
            self.ev.wait(self.every)

    def stop(self):
        self.ev.set(); self.join(15)
        return self.max


def kv_max(gmax):
    return gmax.get("vllm:kv_cache_usage_perc", gmax.get("vllm:gpu_cache_usage_perc"))


def minmem(path, t0, t1):
    """min MemAvailable (GiB) in a guard log ("<epoch> <MiB>" lines) between t0 and t1."""
    m = None
    try:
        for line in open(path):
            p = line.split()
            if len(p) == 2 and p[0].isdigit() and p[1].isdigit() and t0 <= int(p[0]) <= t1:
                m = int(p[1]) if m is None else min(m, int(p[1]))
    except OSError:
        pass
    return None if m is None else round(m / 1024, 2)


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def selftest():
    t = ('vllm:num_preemptions_total{engine="0",model_name="m"} 3.0\n'
         'vllm:num_preemptions_total{engine="1",model_name="m"} 2.0\n'
         'vllm:prefix_cache_hits_total{engine="0"} 10\n# HELP x\nvllm:kv_cache_usage_perc{engine="0"} 0.25\n'
         'vllm:num_preemptions_created{engine="0"} 1.7e9\n')
    d = parse(t)
    assert d["vllm:num_preemptions_total"] == 5.0 and d["vllm:kv_cache_usage_perc"] == 0.25, d
    assert "vllm:num_preemptions_created" not in d
    assert delta({"vllm:num_preemptions_total": 1}, d)["vllm:num_preemptions_total"] == 4.0
    assert delta({}, d)["vllm:num_preemptions_total"] is None and div(None, 3) is None and div(3, 0) is None
    assert pct([3, 1, 2], 0.5) == 2 and mean([1, None, 3]) == 2
    print("vm selftest OK")


if __name__ == "__main__":
    selftest()
