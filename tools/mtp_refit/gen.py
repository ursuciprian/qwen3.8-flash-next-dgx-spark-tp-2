"""On-policy data for the MTP refit (#97): build the prompt mix, then let the served target answer it.

    python -m tools.mtp_refit.gen prompts --mix tools/mtp_refit/mix.yaml --out prompts
    python -m tools.mtp_refit.gen run --server http://localhost:8000 --prompts prompts --out gen \
        --concurrency 16 --budget-tokens 2000000

`prompts` needs `datasets` (dgx-01: ~/GEN-AI/evals/venv). It writes prompts/<category>.jsonl with
{id, category, split, messages, tools?}; split is "heldout" for a stable hash share of each category.
`run` is stdlib only. It posts each prompt to /v1/chat/completions with return_token_ids and writes
gen/<category>.jsonl with {id, category, split, prompt_token_ids, output_token_ids, finish_reason,
sampling}. Token ids come from the server, never from re-tokenized text. Reruns skip finished ids.
"""
from __future__ import annotations

import argparse
import array
import ast
import glob
import hashlib
import json
import os
import random
import re
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

NGRAM = 13
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run one command in the agent shell: bash, plus the special editor and search "
                       "commands listed in the system prompt.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}


def token_sha1(ids) -> str:
    """sha1 of a token list as little-endian int32, the digest the vLLM capture hook writes."""
    a = array.array("i", ids)
    if sys.byteorder != "little":
        a.byteswap()
    return hashlib.sha1(a.tobytes()).hexdigest()


# ---------------------------------------------------------------- adapters: dataset row -> prompt
def swe_agent(row, rng, max_chars):
    """SWE-agent trajectory -> chat with native bash tool calls, cut before a random assistant turn."""
    traj = row["trajectory"]
    msgs, n_call = [{"role": "system", "content": traj[0]["system_prompt"] or traj[0]["text"]}], 0
    cuts = []
    for turn in traj[1:]:
        text = turn["text"] or ""
        if turn["role"] == "ai":
            cuts.append(len(msgs))
            m = list(re.finditer(r"```[a-z]*\n(.*?)```", text, flags=re.S))
            if not m:
                msgs.append({"role": "assistant", "content": text})
                continue
            n_call += 1
            msgs.append({"role": "assistant", "content": text[: m[-1].start()].strip(),
                         "tool_calls": [{"id": f"call_{n_call}", "type": "function",
                                         "function": {"name": "bash",
                                                      "arguments": json.dumps({"command": m[-1].group(1).strip()})}}]})
        elif msgs[-1]["role"] == "assistant" and msgs[-1].get("tool_calls"):
            msgs.append({"role": "tool", "tool_call_id": msgs[-1]["tool_calls"][0]["id"], "content": text})
        else:
            msgs.append({"role": "user", "content": text})
    cuts = [c for c in cuts if len(json.dumps(msgs[:c])) <= max_chars]
    if not cuts:
        return None
    return {"messages": msgs[: rng.choice(cuts)], "tools": [BASH_TOOL]}


def _object_types(x):
    if isinstance(x, dict):
        return {k: ("object" if k == "type" and v == "dict" else _object_types(v)) for k, v in x.items()}
    return [_object_types(v) for v in x] if isinstance(x, list) else x


def toolace(row, rng, max_chars):
    """ToolACE: function list from the system text as native tools, first user turn as the prompt."""
    marker = "Here is a list of functions in JSON format that you can invoke:"
    sys_text, convs = row["system"], row["conversations"]
    if marker not in sys_text or not convs or convs[0]["from"] != "user":
        return None
    funcs, _ = json.JSONDecoder().raw_decode(sys_text.split(marker, 1)[1].strip())
    tools = [{"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
                                               "parameters": _object_types(f.get("parameters", {}))}}
             for f in funcs]
    p = {"messages": [{"role": "user", "content": convs[0]["value"]}], "tools": tools}
    return p if len(json.dumps(p)) <= max_chars else None


def field(row, rng, max_chars, field):
    text = row[field]
    return {"messages": [{"role": "user", "content": text}]} if text and len(text) <= max_chars else None


ADAPTERS = {"swe_agent": swe_agent, "toolace": toolace, "field": field}


# ---------------------------------------------------------------- exclusion
def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def shingles(text):
    w = _words(text)
    return {" ".join(w[i : i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def _strings(path):
    if path.endswith(".py"):
        try:
            return [n.value for n in ast.walk(ast.parse(open(path).read())) if isinstance(n, ast.Constant)
                    and isinstance(n.value, str)]
        except SyntaxError:
            pass
    if path.endswith((".json", ".jsonl")):
        out = []
        for line in open(path, errors="replace") if path.endswith(".jsonl") else [open(path).read()]:
            try:
                stack = [json.loads(line)]
            except ValueError:
                continue
            while stack:
                x = stack.pop()
                if isinstance(x, str):
                    out.append(x)
                elif isinstance(x, dict | list):
                    stack.extend(x.values() if isinstance(x, dict) else x)
        return out
    return [open(path, errors="replace").read()]


def exclusion_set(patterns):
    grams = set()
    for pat in patterns:
        hits = glob.glob(os.path.expanduser(pat))
        if not hits:
            raise SystemExit(f"exclude pattern matches nothing: {pat}")
        for h in hits:
            files = [os.path.join(r, f) for r, _, fs in os.walk(h) for f in fs] if os.path.isdir(h) else [h]
            for f in files:
                if not f.endswith((".pyc", ".so", ".png", ".gz")):
                    for s in _strings(f):
                        grams |= shingles(s)
    return grams


def prompt_text(p):
    return "\n".join(m.get("content") or "" for m in p["messages"])


def is_heldout(pid, fraction):
    return int(hashlib.sha1(pid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


# ---------------------------------------------------------------- commands
def _write_atomic(path, text):
    with open(path + ".tmp", "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def validate_prompts(out, mix):
    """Errors in a prompts dir against the mix: every category written whole (MANIFEST.json count and sha256,
    the wanted count, every line valid JSON with the fields gen run needs, ids unique). [] = complete."""
    errs, ids = [], set()
    try:
        man = json.load(open(os.path.join(out, "MANIFEST.json")))["categories"]
    except (OSError, ValueError, KeyError) as e:
        return [f"MANIFEST.json: {e}"]
    for cat, spec in mix["categories"].items():
        path, m = os.path.join(out, f"{cat}.jsonl"), man.get(cat)
        want = round(mix["total"] * spec["share"])
        if m is None or not os.path.exists(path):
            errs.append(f"{cat}: missing")
            continue
        if _sha256(path) != m["sha256"]:
            errs.append(f"{cat}: sha256 differs from MANIFEST.json")
        n = 0
        for i, line in enumerate(open(path)):
            try:
                p = json.loads(line)
                assert p["category"] == cat and p["split"] in ("train", "heldout") and p["messages"], "fields"
            except (ValueError, KeyError, AssertionError) as e:
                errs.append(f"{cat}:{i + 1}: {e}")
                break
            if p["id"] in ids:
                errs.append(f"{cat}:{i + 1}: duplicate id {p['id']}")
            ids.add(p["id"])
            n += 1
        if n != m["count"] or n != want:
            errs.append(f"{cat}: {n} lines, MANIFEST {m['count']}, mix wants {want}")
    return errs


def cmd_prompts(a):
    """Atomic per-category files plus MANIFEST.json, resumable: a rerun keeps categories already written
    whole. Exits through os._exit after validating, so the streaming readers' threads never reach
    interpreter shutdown (datasets/pyarrow abort there: PyGILState_Release while finalizing)."""
    import yaml

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    import datasets.config
    from datasets import load_dataset

    datasets.config.STREAMING_READ_MAX_RETRIES = 60  # flaky HF CDN reads (k: "peer closed connection")
    datasets.config.STREAMING_READ_RETRY_INTERVAL = 10
    mix = yaml.safe_load(open(a.mix))
    excl = exclusion_set(mix.get("exclude", []))
    max_chars = int(mix.get("max_prompt_tokens", 24000) * 3.5)
    os.makedirs(a.out, exist_ok=True)
    man_path = os.path.join(a.out, "MANIFEST.json")
    try:
        man = json.load(open(man_path))
    except (OSError, ValueError):
        man = {"categories": {}}
    rng = random.Random(mix["seed"])
    for cat, spec in mix["categories"].items():
        path = os.path.join(a.out, f"{cat}.jsonl")
        m = man["categories"].get(cat)
        if m and os.path.exists(path) and _sha256(path) == m["sha256"]:
            print(f"{cat}: kept ({m['count']} prompts)", flush=True)
            continue
        want = round(mix["total"] * spec["share"])
        per_source = -(-want // len(spec["sources"]))
        out, dropped = [], 0
        for src in spec["sources"]:
            ds = load_dataset(src["dataset"], src.get("config"), split=src["split"], streaming=True,
                              revision=src.get("revision"))
            ds = ds.shuffle(seed=mix["seed"], buffer_size=10_000)
            extra = {"field": src["field"]} if "field" in src else {}
            got = 0
            for i, row in enumerate(ds):
                if got >= per_source:
                    break
                p = ADAPTERS[src["adapter"]](row, rng, max_chars, **extra)
                if p is None:
                    continue
                if shingles(prompt_text(p)) & excl:
                    dropped += 1
                    continue
                pid = f"{src['dataset']}:{src.get('config') or ''}:{i}"
                p.update(id=pid, category=cat, split="heldout" if is_heldout(pid, mix["heldout_fraction"]) else "train")
                out.append(p)
                got += 1
        _write_atomic(path, "".join(json.dumps(p) + "\n" for p in out[:want]))
        man["categories"][cat] = {"count": len(out[:want]), "sha256": _sha256(path),
                                  "sources": [f"{x['dataset']}@{x.get('revision')}" for x in spec["sources"]]}
        _write_atomic(man_path, json.dumps(man, indent=1))
        print(f"{cat}: {len(out[:want])} prompts ({dropped} excluded by {NGRAM}-gram overlap)", flush=True)
    errs = validate_prompts(a.out, mix)
    print("\n".join(errs) or "prompts: complete", flush=True)
    sys.stderr.flush()
    os._exit(1 if errs else 0)


def cmd_validate(a):
    import yaml

    errs = validate_prompts(a.out, yaml.safe_load(open(a.mix)))
    print("\n".join(errs) or "prompts: complete")
    raise SystemExit(1 if errs else 0)


def _post(url, body, timeout):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _metrics(server, path):
    with urllib.request.urlopen(f"{server}/metrics", timeout=30) as r, open(path, "w") as f:
        f.write("".join(l for l in r.read().decode().splitlines(True) if l.startswith("vllm:")))


def cmd_run(a):
    prompts = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.prompts, "*.jsonl"))) for l in open(f)]
    if a.split:
        prompts = [p for p in prompts if p["split"] == a.split]
    if a.ids:
        keep = {l.strip() for l in open(a.ids) if l.strip()}
        prompts = [p for p in prompts if p["id"] in keep]
    os.makedirs(a.out, exist_ok=True)
    done = set()
    for f in glob.glob(os.path.join(a.out, "*.jsonl")):
        done |= {json.loads(l)["id"] for l in open(f)}
    todo = [p for p in prompts if p["id"] not in done]
    random.Random(0).shuffle(todo)  # categories interleaved, so a budget stop keeps the mix
    sampling = {"max_tokens": a.max_tokens}
    for k in ("temperature", "top_p", "top_k"):
        if getattr(a, k) is not None:
            sampling[k] = getattr(a, k)
    lock, total, stop = threading.Lock(), [0], threading.Event()
    files = {}

    def one(p):
        if stop.is_set():
            return
        body = {"model": a.model, "messages": p["messages"], "return_token_ids": True,
                "chat_template_kwargs": {"reasoning_effort": a.reasoning_effort}, **sampling}
        if p.get("tools"):
            body["tools"] = p["tools"]
        try:
            r = _post(f"{a.server}/v1/chat/completions", body, a.timeout)
        except Exception as e:  # noqa: BLE001 - one bad prompt must not stop the run
            print(f"{p['id']}: {e}")
            return
        c = r["choices"][0]
        rec = {"id": p["id"], "category": p["category"], "split": p["split"],
               "prompt_token_ids": r["prompt_token_ids"], "output_token_ids": c["token_ids"],
               "finish_reason": c["finish_reason"], "sampling": sampling}
        with lock:
            if p["category"] not in files:
                files[p["category"]] = open(os.path.join(a.out, f"{p['category']}.jsonl"), "a")
            files[p["category"]].write(json.dumps(rec) + "\n")
            files[p["category"]].flush()
            total[0] += len(c["token_ids"])
            if a.budget_tokens and total[0] >= a.budget_tokens:
                stop.set()

    if a.metrics:
        _metrics(a.server, os.path.join(a.out, "metrics-before.txt"))
    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, todo))
    if a.metrics:
        _metrics(a.server, os.path.join(a.out, "metrics-after.txt"))
    for f in files.values():
        f.close()
    print(f"{len(todo)} prompts sent, {total[0]} response tokens{' (budget reached)' if stop.is_set() else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prompts", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--mix", default=os.path.join(os.path.dirname(__file__), "mix.yaml"))
        p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--server", default="http://localhost:8000")
    r.add_argument("--model", default="qwen3.8-flash-next")
    r.add_argument("--prompts", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--split", choices=["train", "heldout"])
    r.add_argument("--ids", help="file with one prompt id per line (offline-vs-live check)")
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--budget-tokens", type=int, default=0, help="stop after this many response tokens (0 = all)")
    r.add_argument("--max-tokens", type=int, default=16384)
    r.add_argument("--temperature", type=float, help="default: the server's sampling defaults")
    r.add_argument("--top-p", type=float)
    r.add_argument("--top-k", type=int, help="-1 = off (plain T sampling, as eval_offline's T=1 formula)")
    r.add_argument("--reasoning-effort", default="medium")
    r.add_argument("--timeout", type=float, default=1800)
    r.add_argument("--metrics", action="store_true", help="save vllm:* /metrics before and after the run")
    a = ap.parse_args()
    {"prompts": cmd_prompts, "validate": cmd_validate, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
