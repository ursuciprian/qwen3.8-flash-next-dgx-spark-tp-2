# /// script
# requires-python = ">=3.10"
# dependencies = ["mlflow-tracing>=3.17"]  # optional: traces only when MLFLOW_TRACKING_URI is set
# ///
"""k55: single-request decode speed on real Python coding prompts (c1, sequential).

decode tok/s = (completion_tokens - 1) / (t_last_token - t_first_token)   (same as scripts/decode_probe.py)
TTFT         = t_first_token - t_request_sent
tokens/step  = 1 + accepted / drafts, from vllm:spec_decode_* /metrics deltas around each request
Settings:
  t0-nothink : temperature 0, chat_template_kwargs enable_thinking=false
  default    : only model, messages, max_tokens, stream (what a plain client gets: server sampling + reasoning defaults)
Usage: python3 coding_probe.py <host> <label> <outdir> [max_tokens] [langs: python,cpp,rust,go (default all)]
Prompts: 12 Python (k55) + 8 each C++, Rust, Go (k55b, 2026-10-06); summaries per language and all languages.
"""
import json, os, statistics, sys, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
try:
    from mlflow_trace import traced, note
except ImportError:  # helper not shipped next to this copy: run untraced
    def traced(_benchmark): return lambda fn: fn
    def note(**_kw): pass

HOST, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MAX_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 768
BASE = f"http://{HOST}:8000"
MODEL = "qwen3.8-flash-next"

PROMPTS = [
    "Write a Python function `merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]` that merges overlapping intervals. Include type hints, a docstring and a few doctest examples.",
    "Implement an LRU cache class in Python with `get(key)` and `put(key, value)` in O(1) time, without using functools or OrderedDict. Explain the data structures briefly.",
    "Fix the bug in this Python code and explain it:\n\n```python\ndef add_item(item, items=[]):\n    items.append(item)\n    return items\n\nprint(add_item(1))\nprint(add_item(2))\n```",
    "Write a Python script that reads a CSV file of transactions (columns: date, category, amount), groups the totals by month and category using only the standard library, and prints a sorted table.",
    "Refactor this Python function to be more readable and Pythonic, keeping the behaviour:\n\n```python\ndef f(l):\n    r = []\n    for i in range(len(l)):\n        if l[i] % 2 == 0:\n            if l[i] > 10:\n                r.append(l[i] * 2)\n            else:\n                r.append(l[i])\n    return r\n```",
    "Write an asyncio-based Python function that downloads a list of URLs concurrently with aiohttp, limits concurrency to 5 with a semaphore, retries each failed request up to 3 times with exponential backoff, and returns a dict of url -> status code.",
    "Write pytest unit tests for a function `slugify(text: str) -> str` that lowercases text, replaces runs of non-alphanumeric characters with a single hyphen, and strips leading/trailing hyphens. Cover edge cases.",
    "Implement a thread-safe token bucket rate limiter class in Python with methods `allow() -> bool` and `wait()`. Use threading and time.monotonic.",
    "This Python code raises `RuntimeError: dictionary changed size during iteration`. Fix it and explain why:\n\n```python\nscores = {'a': 1, 'b': 0, 'c': 3}\nfor k in scores:\n    if scores[k] == 0:\n        del scores[k]\n```",
    "Write a Python dataclass-based inventory system: `Product` (sku, name, price, quantity) and `Inventory` with add, remove, restock, total_value and a method that returns products below a reorder threshold. Add type hints.",
    "Write a Python function that parses a log file where each line looks like `2026-10-06 12:00:01 ERROR [auth] message text`, and returns the count of each level per component as a nested dict. Use a compiled regex and handle malformed lines.",
    "Implement Dijkstra's shortest path in Python for a graph given as a dict of adjacency lists with weights, returning both the distance and the path to a target node. Use heapq.",
]

CPP = [
    "Write a C++17 function `std::vector<std::string> split(std::string_view s, char delim)` that keeps empty fields, plus a small `main` that demonstrates it on three inputs.",
    "Implement a C++ class `RingBuffer<T>` with a fixed capacity given at construction, `push`, `pop` returning `std::optional<T>`, `size` and `full`. Follow the rule of zero and explain the design briefly.",
    "Fix the bug in this C++ code and explain it:\n\n```cpp\n#include <vector>\nint sum_until_negative(const std::vector<int>& v) {\n    int total = 0;\n    for (size_t i = v.size() - 1; i >= 0; --i) {\n        if (v[i] < 0) break;\n        total += v[i];\n    }\n    return total;\n}\n```",
    "Refactor this C++ code to modern C++ (smart pointers, range-for, no manual delete), keeping the behaviour:\n\n```cpp\nstruct Node { int value; Node* next; };\nint count_even(Node* head) {\n    int c = 0;\n    Node* p = head;\n    while (p != NULL) { if (p->value % 2 == 0) c++; p = p->next; }\n    return c;\n}\nvoid free_list(Node* head) { while (head) { Node* n = head->next; delete head; head = n; } }\n```",
    "Write GoogleTest unit tests for a C++ function `bool is_valid_ipv4(const std::string& s)` covering valid addresses, leading zeros, out-of-range octets, empty parts and extra dots.",
    "Write a C++17 program that counts word frequencies in a large text file using 4 `std::thread` workers, each processing a chunk of lines into a local `std::unordered_map`, then merges the maps and prints the top 10 words.",
    "Implement a thread-safe bounded blocking queue in C++ with `std::mutex` and two `std::condition_variable`s, offering `push`, `pop` and a `close()` that wakes all waiters.",
    "Implement topological sort (Kahn's algorithm) in C++ for a graph given as `std::vector<std::vector<int>>` adjacency lists, returning `std::optional<std::vector<int>>` that is empty when the graph has a cycle.",
]

RUST = [
    "Write a Rust function `fn parse_kv(input: &str) -> Result<HashMap<String, String>, ParseError>` that parses `key=value` pairs separated by semicolons, with a custom error enum that implements `std::fmt::Display` and `std::error::Error`.",
    "Implement a Rust struct `Matrix` backed by a `Vec<f64>` with `new(rows, cols)`, `get`, `set`, `transpose` and `impl Mul for &Matrix` that returns `Option<Matrix>` when the dimensions do not match.",
    "Fix the borrow checker error in this Rust code and explain it:\n\n```rust\nfn main() {\n    let mut names = vec![String::from(\"ana\"), String::from(\"bob\")];\n    let first = &names[0];\n    names.push(String::from(\"cid\"));\n    println!(\"{}\", first);\n}\n```",
    "Refactor this Rust code to idiomatic iterator chains, keeping the behaviour:\n\n```rust\nfn totals(orders: &Vec<(String, u32, f64)>) -> Vec<(String, f64)> {\n    let mut out = Vec::new();\n    let mut i = 0;\n    while i < orders.len() {\n        if orders[i].1 > 0 {\n            out.push((orders[i].0.clone(), orders[i].1 as f64 * orders[i].2));\n        }\n        i += 1;\n    }\n    out\n}\n```",
    "Write unit tests in Rust (`#[cfg(test)]` module) for a pair of functions `fn rle_encode(s: &str) -> String` and `fn rle_decode(s: &str) -> Result<String, String>` (run-length encoding like `aaab` -> `3a1b`), covering round trips, multi-digit counts, Unicode input, malformed input and the empty string.",
    "Write an async Rust program with tokio that fetches 20 URLs with reqwest, at most 4 at a time using a `tokio::sync::Semaphore`, with a 5 second timeout per request, and prints each URL with its status or error.",
    "Using only std threads and `std::sync::mpsc` channels in Rust, build a worker pool of 4 threads that receives jobs as boxed closures and shuts down cleanly when the pool is dropped.",
    "Implement binary search over a sorted slice in Rust as a generic function `fn lower_bound<T: Ord>(xs: &[T], target: &T) -> usize`, then use it to count occurrences of a value. Include doc comments with examples.",
]

GO = [
    "Write a Go function `func Chunk[T any](xs []T, size int) [][]T` that splits a slice into chunks of at most `size` elements, returning an error-free result and panicking on size <= 0. Include a doc comment and an Example test.",
    "Implement a Go struct `TTLCache` with `Set(key string, value any, ttl time.Duration)`, `Get(key string) (any, bool)` and a background goroutine that evicts expired entries until `Close()` is called. Make it safe for concurrent use.",
    "Fix the bug in this Go code and explain it:\n\n```go\nfunc main() {\n    var wg sync.WaitGroup\n    results := []int{}\n    for i := 0; i < 10; i++ {\n        wg.Add(1)\n        go func() {\n            defer wg.Done()\n            results = append(results, i*i)\n        }()\n    }\n    wg.Wait()\n    fmt.Println(len(results))\n}\n```",
    "Refactor this Go handler to separate parsing, validation and the response, with proper error handling and status codes:\n\n```go\nfunc handler(w http.ResponseWriter, r *http.Request) {\n    body, _ := io.ReadAll(r.Body)\n    var req map[string]interface{}\n    json.Unmarshal(body, &req)\n    name := req[\"name\"].(string)\n    age := int(req[\"age\"].(float64))\n    fmt.Fprintf(w, \"hello %s, next year you are %d\", name, age+1)\n}\n```",
    "Write table-driven Go tests for a function `func ParseDuration(s string) (time.Duration, error)` that accepts values like `90s`, `1h30m` and `2d` (days), covering valid inputs, unknown units, negative values and empty input.",
    "Write a Go pipeline with goroutines and channels: a generator that emits lines from a file, 3 worker goroutines that compute the SHA-256 of each line, and a collector that prints results in input order. Use context for cancellation.",
    "Implement a Go rate limiter that allows N events per second per client ID using a token bucket per client, safe for concurrent use with a `sync.Mutex`, plus a goroutine that removes idle clients every minute.",
    "Implement an LRU cache in Go using `container/list` and a map, with `Get` and `Put` in O(1), and write a short example in `main` that shows an eviction.",
]

LANGS = {"python": ("Python", PROMPTS), "cpp": ("C++", CPP), "rust": ("Rust", RUST), "go": ("Go", GO)}
LANG_SEL = (sys.argv[5] if len(sys.argv) > 5 else "python,cpp,rust,go").split(",")

SETTINGS = {
    "t0-nothink": {"temperature": 0, "chat_template_kwargs": {"enable_thinking": False}},
    "default": {},
}


def metrics():
    d = a = 0.0
    with urllib.request.urlopen(f"{BASE}/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:spec_decode_num_drafts_total"):
                d += float(line.split()[-1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
                a += float(line.split()[-1])
    return d, a


@traced("coding")
def run(prompt, extra):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOK,
            "stream": True, "stream_options": {"include_usage": True}, **extra}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d0, a0 = metrics()
    t0 = time.perf_counter(); t_first = t_last = None; completion = prompt_tokens = 0; finish = None; nreason = ncontent = 0
    text, reasoning = [], []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                break
            obj = json.loads(line[6:])
            if obj.get("usage"):
                completion = obj["usage"].get("completion_tokens", completion)
                prompt_tokens = obj["usage"].get("prompt_tokens", prompt_tokens)
            ch = obj.get("choices") or []
            if ch and ch[0].get("finish_reason"):
                finish = ch[0]["finish_reason"]
            dl = (ch[0].get("delta") or {}) if ch else {}
            rc = dl.get("reasoning_content") or dl.get("reasoning")
            text.append(dl.get("content") or ""); reasoning.append(rc or "")
            if dl.get("content") or rc:
                now = time.perf_counter()
                t_first = t_first or now
                t_last = now
                ncontent += bool(dl.get("content")); nreason += bool(rc)
    d1, a1 = metrics()
    dd, da = d1 - d0, a1 - a0
    ok = t_first and t_last > t_first and completion >= 2
    res = {"decode_tps": (completion - 1) / (t_last - t_first) if ok else None,
            "ttft_s": (t_first - t0) if t_first else None, "completion_tokens": completion, "finish": finish,
            "reasoning_chunks": nreason, "content_chunks": ncontent, "drafts": dd, "accepted": da,
            "tokens_per_step": 1 + da / dd if dd else None}
    note(outputs={**res, "content": "".join(text), "reasoning": "".join(reasoning)},
         prompt_tokens=prompt_tokens, completion_tokens=completion, ttft_s=res["ttft_s"])
    return res


def main():
    run("Write a Python hello world.", SETTINGS["t0-nothink"])  # warm-up, not counted
    rows, summary = [], {}
    for sname, extra in SETTINGS.items():
        for lang in LANG_SEL:
            for i, p in enumerate(LANGS[lang][1]):
                r = run(p, extra); r.update(setting=sname, lang=lang, prompt=i, build=LABEL); rows.append(r)
                print(json.dumps(r), flush=True)

    def summ(res):
        v = [r["decode_tps"] for r in res if r["decode_tps"]]
        D = sum(r["drafts"] for r in res); A = sum(r["accepted"] for r in res)
        return {"n": len(res), "median_tps": statistics.median(v), "max_tps": max(v), "min_tps": min(v),
                "tokens_per_step": 1 + A / D if D else None,
                "ttft_median_s": statistics.median(r["ttft_s"] for r in res),
                "completion_tokens_median": statistics.median(r["completion_tokens"] for r in res),
                "finish_length": sum(r["finish"] == "length" for r in res),
                "requests_with_reasoning": sum(r["reasoning_chunks"] > 0 for r in res)}

    groups = [(l, LANGS[l][0]) for l in LANG_SEL] + ([("all", "all-language")] if len(LANG_SEL) > 1 else [])
    for sname in SETTINGS:
        for key, _ in groups:
            summary[f"{sname}/{key}"] = summ([r for r in rows if r["setting"] == sname and key in ("all", r["lang"])])
    json.dump({"build": LABEL, "max_tokens": MAX_TOK, "langs": LANG_SEL, "rows": rows, "summary": summary},
              open(f"{OUT}/probe-{LABEL}.json", "w"), indent=1)
    sdesc = {"t0-nothink": "T=0 thinking off", "default": "server defaults (thinking on)"}
    with open(f"{OUT}/summary-{LABEL}.txt", "w") as f:
        for sname in SETTINGS:
            for key, name in groups:
                x = summary[f"{sname}/{key}"]
                line = (f"{LABEL} | {x['n']} {name} coding prompts, one request, {sdesc[sname]}, max_tokens {MAX_TOK}: "
                        f"decode median {x['median_tps']:.1f}, max {x['max_tps']:.1f}, min {x['min_tps']:.1f} tok/s; "
                        f"tokens/step {x['tokens_per_step']:.2f}; TTFT median {x['ttft_median_s']*1000:.0f} ms; "
                        f"completion tokens median {x['completion_tokens_median']}, hit max_tokens {x['finish_length']}/{x['n']}, "
                        f"with reasoning {x['requests_with_reasoning']}/{x['n']}")
                print(line); f.write(line + "\n")


if __name__ == "__main__":
    main()
