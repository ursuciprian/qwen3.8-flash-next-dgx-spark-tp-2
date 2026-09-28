#!/usr/bin/env python3
"""Run the DevOps task set against an OpenAI-compatible endpoint, one request at a time.

  run_devops.py <build> [--repeats 3] [--base http://localhost:8000/v1]

For every tasks/<id>/prompt.md and repeat r it writes responses/<build>/<id>.r<r>.json:
the answer content, the reasoning text, token usage and wall-clock timings
(time to first reasoning token, time to first answer token = end of thinking, total).
Sampling is the model card's thinking-mode recommendation. Stdlib only.
"""
import argparse, json, pathlib, time, urllib.request

SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,
                repetition_penalty=1.0, max_tokens=16384)  # thinking cap (Jev evals-20260928-devops-config: xhigh_16k)
HERE = pathlib.Path(__file__).resolve().parent


def run_one(base, prompt, seed):
    body = dict(model="qwen3.8-flash-next", messages=[{"role": "user", "content": prompt}],
                stream=True, stream_options={"include_usage": True}, seed=seed, **SAMPLING)
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_reason = t_content = None
    content, reasoning, usage, finish = [], [], None, None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                rc = d.get("reasoning") or d.get("reasoning_content")
                if rc:
                    t_reason = t_reason or time.time()
                    reasoning.append(rc)
                if d.get("content"):
                    t_content = t_content or time.time()
                    content.append(d["content"])
                finish = ch.get("finish_reason") or finish
    t1 = time.time()
    rel = lambda t: round(t - t0, 2) if t else None
    return dict(content="".join(content), reasoning="".join(reasoning), usage=usage, finish_reason=finish,
                seed=seed, t_first_reasoning_s=rel(t_reason), t_first_answer_s=rel(t_content),
                total_s=round(t1 - t0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--base", default="http://localhost:8000/v1")
    a = ap.parse_args()
    out = HERE / "responses" / a.build
    out.mkdir(parents=True, exist_ok=True)
    tasks = sorted(p.parent for p in (HERE / "tasks").glob("*/prompt.md"))
    for r in range(a.repeats):
        for t in tasks:
            f = out / f"{t.name}.r{r}.json"
            if f.exists():
                continue
            res = run_one(a.base, (t / "prompt.md").read_text(), seed=1000 + r)
            res.update(task=t.name, repeat=r, build=a.build, sampling=SAMPLING)
            f.write_text(json.dumps(res, indent=1))
            u = res["usage"] or {}
            print(f"{t.name} r{r} total={res['total_s']}s think_end={res['t_first_answer_s']}s "
                  f"completion={u.get('completion_tokens')} finish={res['finish_reason']}", flush=True)


if __name__ == "__main__":
    main()
