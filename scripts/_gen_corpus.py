#!/usr/bin/env python3
"""One-off draft-vocab corpus generator (prep task 2026-09-18).

Calls the live local vllm server for assistant-only output text, appending
one {"text": ...} per generation to a jsonl corpus. Stays polite: checks for
competing bench/probe jobs and /health before every batch, concurrency<=2,
stops on a wall-clock cap or a token-count target, whichever comes first.

Not part of the permanent repo tooling -- lives in scripts/ only for this
prep session, safe to delete afterward.
"""
import json, os, re, sys, time, subprocess, urllib.request, concurrent.futures

BASE = "http://localhost:8000"
URL = BASE + "/v1/chat/completions"
MODEL = "qwen3.8-flash-next"
OUT = os.path.expanduser("~/GEN-AI/build/draft-vocab/corpus/generated.jsonl")
TOKEN_TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000_000
TIME_CAP_S = int(sys.argv[2]) if len(sys.argv) > 2 else 5 * 3600
CONCURRENCY = 2
MAX_TOKENS = 1024
COMPETE_PAT = r"bench_|decode_probe|fidelity_probe|tool-eval|straggler"

# 40 prompts: 35 verbatim from tools/tony-bench/bench_categories.py (coding,
# reasoning, json, html, prose, narrative, format categories -- the ones
# that don't need external bench-docs/*.md files), plus 5 self-contained
# summary-style prompts standing in for that script's doc()-based category.
PROMPTS = [
    "Write a Python function merge_intervals(intervals) that merges overlapping [start, end] intervals and returns the merged list sorted by start. Return only one ```python code block.",
    "Write a Python function parse_duration(s) that converts strings like '1h30m15s', '45s', '2h', '3m', '1h1s' into total seconds (int). Units may be omitted but always appear in h, m, s order. Return only one ```python code block.",
    "Write a Python function top_k_frequent(words, k) returning the k most frequent words, most frequent first; ties broken alphabetically. Return only one ```python code block.",
    "This binary search has a bug that can make it loop forever. Return the corrected function only, as one ```python code block, same name and signature.\n\ndef binary_search(a, target):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] == target:\n            return mid\n        if a[mid] < target:\n            lo = mid\n        else:\n            hi = mid\n    return -1",
    "Write a Python function roman_to_int(s) converting a Roman numeral string to an integer. Return only one ```python code block.",
    "Ana is twice as old as Ben. In 6 years Ana will be 1.5 times Ben's age. How old is Ben now?\n\nGive your final answer on the last line exactly as: ANSWER: <answer>",
    "A fair coin is flipped 4 times. What is the probability of exactly 2 heads? Give a fraction in lowest terms.\n\nGive your final answer on the last line exactly as: ANSWER: <answer>",
    "Five runners finish a race. Dee finishes before Eve but after Cal. Bo finishes last. Al finishes before Cal. Who finishes third?\n\nGive your final answer on the last line exactly as: ANSWER: <answer>",
    "Two trains 300 km apart travel toward each other at 70 km/h and 80 km/h. A bird flying at 120 km/h goes back and forth between them until they meet. How many km does the bird fly?\n\nGive your final answer on the last line exactly as: ANSWER: <answer>",
    "What is the remainder when 7^100 is divided by 5?\n\nGive your final answer on the last line exactly as: ANSWER: <answer>",
    "Extract into a JSON object with keys invoice_id (integer), customer (string), issued (ISO date string), amount (number), due_days (integer):\n\n\"Invoice 4471 was issued to Marta Ruiz on 2026-03-14 for $1,250.50, due in 30 days.\"\n\nOutput only the JSON, no prose, no code fences.",
    "Convert this CSV to a JSON array of objects with numeric qty and price:\n\nname,qty,price\napple,3,0.5\npear,2,0.75\n\nOutput only the JSON, no prose, no code fences.",
    "Fix this invalid JSON and output only the corrected JSON:\n\n{'user': 'sam', 'tags': ['a', 'b',], 'active': True}\n\nOutput only the JSON, no prose, no code fences.",
    "Return a JSON object mapping each city to the arithmetic mean of its temperatures:\n\n{\"Oslo\": [2, 4, 6], \"Cairo\": [30, 32], \"Lima\": [18, 19, 20, 21]}\n\nOutput only the JSON, no prose, no code fences.",
    "Produce a JSON object with exactly these keys and values: id = integer 7, name = string \"Widget\", price = number 19.99, in_stock = boolean true, tags = array of strings [\"tools\", \"home\"].\n\nOutput only the JSON, no prose, no code fences.",
    "Write a complete HTML5 page: a <header> containing an <h1> that reads Spark Bench and a <nav> with three links (Home, Results, Method), then a <main> containing a <section id=\"summary\"> with one paragraph.\n\nOutput only the HTML.",
    "Write an HTML <table> with a <thead> row of Lane, Decode tok/s, TTFT s and a <tbody> with two rows: NVFP4, 64.0, 1.54 and EXL3, 61.5, 0.52.\n\nOutput only the HTML.",
    "Write an HTML <form> with: a required email input with id=\"email\", a <select id=\"lane\"> with the options NVFP4 and EXL3, a <textarea id=\"notes\">, and a submit <button>.\n\nOutput only the HTML.",
    "Write an accessible site navigation: a skip link <a href=\"#main\" class=\"skip\">Skip to content</a>, then <nav aria-label=\"Main\"> containing a <ul> of four <li> items each with an <a>.\n\nOutput only the HTML.",
    "Write an HTML card component with an inline <style> block: a <div class=\"card\"> containing an <img> with an alt attribute, an <h2>, a <p>, and an <a class=\"btn\">.\n\nOutput only the HTML.",
    "Explain what tensor parallelism is to a smart high-school student, in 120 to 180 words.",
    "Write a product description of 80 to 120 words for a compact AI workstation with 128 GB of unified memory. Do not use the words revolutionary, game-changing, or unleash.",
    "Write a professional email under 150 words politely declining a meeting request and proposing two specific alternative times next week.",
    "In 150 to 200 words, argue for or against benchmarking language models with synthetic prompts. Take one side clearly.",
    "In 100 to 150 words, explain in plain language why time to first token matters more than tokens per second for a chat assistant. Use exactly one concrete analogy.",
    "Write a 150 to 250 word short story in first person that includes the words lantern, ledger and thunder, contains at least two lines of dialogue, and ends with a twist.",
    "Write a 120 to 200 word fable with a talking fox. State the moral in the final sentence, starting with the word Moral:",
    "Write a 200 to 300 word scene: two engineers arguing at 2 a.m. about a benchmark result. Mostly dialogue; no narration longer than one sentence at a time.",
    "Write a 100 to 150 word micro-story told entirely in second person, present tense, set in a data center during a power outage.",
    "Write a 150 to 250 word story for children aged 6 to 8 about a robot learning to wait. Exactly three paragraphs.",
    "List exactly 7 prime numbers greater than 50, one per line, ascending. Output nothing else.",
    "Output a markdown table with the columns Lane | Context | KV pool and two rows: NVFP4 with 262,144 and 295,230; EXL3 with 1,048,576 and 1,396,551. Output nothing else.",
    "Reply with the sentence 'The quick brown fox jumps over the lazy dog' with the words in reverse order, all lowercase, no punctuation. Output only the result.",
    "In at most 20 words: why should a unified-memory machine drop its page cache before loading a large model?",
    "Write a haiku about a GPU fan: three lines, 5-7-5 syllables. Output only the three lines.",
    # stand-ins for the doc()-based summary category (self-contained, no bench-docs/ files needed)
    "Summarize in exactly 3 bullet points, each one sentence: a team migrated a Terraform-managed EKS cluster from self-managed node groups to Karpenter, cutting scale-up latency from 4 minutes to 40 seconds and monthly spot cost by 30%, but had to pin two node classes for a GPU workload that Karpenter kept bin-packing incorrectly.",
    "In one paragraph of at most 80 words, summarize: an incident postmortem found that a Redis eviction policy change (noeviction to allkeys-lru) during a traffic spike silently dropped session tokens, causing a wave of unexplained logouts; the fix was reverting the policy and adding a dedicated cache with its own eviction policy for sessions.",
    "In exactly two sentences, summarize what this says about quality: three quantization formats (NVFP4, FP8, EXL3) were compared on the same checkpoint using an 8k/16k/62k/123k/245k-depth needle-in-haystack fidelity probe; NVFP4 and FP8 held 100% exact recall at every depth while EXL3 dropped to 92% at the two longest depths, correlating with its lower per-weight bit budget on the attention projections.",
    "Summarize the following in 3 bullets labeled What, How, Credits: a small CLI tool named rtk wraps common dev commands (git, npm, docker) and rewrites them through a token-optimized proxy, cutting the tokens an AI coding agent spends parsing verbose command output by 60-90%; it was built by a solo maintainer and is installed as a Claude Code hook.",
    "Turn the following into a single post of at most 280 characters, output only the post: after switching the MTP draft head from full vocabulary to a frequency-ranked 65,149-token reduced vocabulary built from the model's own generation transcripts, single-stream decode throughput rose 8-21% on the same DGX Spark tensor-parallel-2 deployment, with no change in accepted output because every draft token is still verified by the full-vocabulary target model.",
]


def health_ok() -> bool:
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def competing_job() -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", COMPETE_PAT], capture_output=True, text=True)
        pids = [p for p in out.stdout.split() if p != str(os.getpid())]
        return bool(pids)
    except Exception:
        return False


def call(prompt: str, thinking: bool) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": thinking, "thinking": thinking},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    msg = resp["choices"][0]["message"]
    text = msg.get("content") or ""
    # keep reasoning content too when present -- it is still the model's own
    # generated tokens, and MTP has to draft through it same as final text
    reasoning = msg.get("reasoning_content") or ""
    return (reasoning + "\n" + text).strip() if reasoning else text


def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)  # rough chars/4 estimate for progress tracking only


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t0 = time.time()
    total_tok_est = 0
    n_calls = 0
    round_idx = 0
    with open(OUT, "a") as f:
        while total_tok_est < TOKEN_TARGET and (time.time() - t0) < TIME_CAP_S:
            if competing_job():
                print(f"[wait] competing job detected, sleeping 60s (t={time.time()-t0:.0f}s)", flush=True)
                time.sleep(60)
                continue
            if not health_ok():
                print("[stop] /health not OK, stopping", flush=True)
                break
            batch = [(PROMPTS[(round_idx * CONCURRENCY + i) % len(PROMPTS)],
                      ((round_idx * CONCURRENCY + i) % 2 == 0))
                     for i in range(CONCURRENCY)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
                futs = [ex.submit(call, p, th) for p, th in batch]
                for fut in futs:
                    try:
                        text = fut.result()
                    except Exception as e:
                        print(f"[err] {e}", flush=True)
                        continue
                    if not text:
                        continue
                    f.write(json.dumps({"text": text}) + "\n")
                    f.flush()
                    total_tok_est += approx_tokens(text)
                    n_calls += 1
            round_idx += 1
            if round_idx % 10 == 0:
                print(f"[progress] calls={n_calls} ~tokens={total_tok_est:,} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[done] calls={n_calls} ~tokens={total_tok_est:,} elapsed={time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
