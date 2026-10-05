#!/usr/bin/env python3
"""Flatten hotel-ab lib-hotel.json files into runs.jsonl, with reasoning/content token counts
from the serving model's /tokenize (same tokenizer for v3b, v3c and the 2x recipe)."""
import glob, json, os, urllib.request

D = os.path.expanduser("~/GEN-AI/hotel-ab")


def ntok(text):
    if not text:
        return 0
    req = urllib.request.Request("http://localhost:8000/tokenize", method="POST",
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": "qwen3.8-flash-next", "prompt": text,
                                                  "add_special_tokens": False}).encode())
    return json.load(urllib.request.urlopen(req, timeout=60))["count"]


with open(f"{D}/runs.jsonl", "w") as out:
    for p in sorted(glob.glob(f"{D}/v3*-dgx0*-h*/lib-hotel.json")):
        cell = p.split("/")[-2]
        arm, node, half = cell.split("-")
        for r in json.load(open(p))["runs"]:
            if r.get("phase") != "profile":
                continue
            label = "correct" if r["correct"] else ("wrong" if r["parsed_answer"] not in ("", None) else "unparsed")
            if not r["ok"] or r.get("error"):
                label = "error"
            content = r.get("content_text") or ""
            out.write(json.dumps({
                "arm": arm, "node": node, "half": half, "run": r["run_index"], "label": label,
                "parsed": r["parsed_answer"], "score_detail": r["score_detail"],
                "completion_tokens": r["completion_tokens"], "reasoning_tokens": ntok(r.get("reasoning_text") or ""),
                "content_tokens": ntok(content), "content_chars": len(content),
                "finish_reason": r["finish_reason"], "hit_max_tokens": r["hit_max_tokens"],
                "elapsed": round(r["elapsed"], 1), "error": r.get("error", ""),
                "final_answer": r["final_answer"], "content_tail": content[-600:],
            }) + "\n")
print(open(f"{D}/runs.jsonl").read().count("\n"), "runs")
