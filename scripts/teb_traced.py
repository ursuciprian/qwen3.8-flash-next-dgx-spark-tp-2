#!/usr/bin/env python3
"""tool-eval-bench (hardmode, TC-45) with every chat request traced to MLflow.

Takes the same arguments as tool-eval-bench:

  python3 scripts/teb_traced.py run --hardmode --temperature 0.0 --backend vllm ... \
      --base-url http://localhost:8000 --model qwen3.8-flash-next

With tracing off (see mlflow_trace.py) this is plain tool-eval-bench. When the
tool lives in its own venv (uv tool, pipx), the script re-runs itself under that
venv's python, so mlflow must be installed there: uv tool install tool-eval-bench --with mlflow-tracing
"""
import os
import shutil
import sys

try:
    import tool_eval_bench  # noqa: F401
except ImportError:
    exe = shutil.which("tool-eval-bench")
    py = os.path.join(os.path.dirname(exe), "python") if exe else ""  # uv tool / pipx venv layout
    if not os.path.exists(py) or os.environ.get("TEB_TRACED_REEXEC"):
        sys.exit("tool-eval-bench not found")
    os.environ["TEB_TRACED_REEXEC"] = "1"
    os.execv(py, [py, os.path.abspath(__file__), *sys.argv[1:]])

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlflow_trace  # noqa: E402
from tool_eval_bench.cli.bench import main  # noqa: E402


def benchmark_name(argv):
    name = "teb-hardmode" if any(a.startswith("--hardmode") for a in argv) else "teb"
    scen = []
    for i, a in enumerate(argv):
        if a.startswith("--scenarios="):
            scen += a.split("=", 1)[1].split(",")
        elif a == "--scenarios":
            for b in argv[i + 1:]:
                if b.startswith("-"):
                    break
                scen += b.split(",")
    return name + "".join("-" + s for s in scen if s)


if mlflow_trace.mlflow:
    # ponytail: patches the adapter method of tool-eval-bench 2.8; if it moves, the run goes untraced
    try:
        from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
        _orig = OpenAICompatibleAdapter.chat_completion

        @mlflow_trace.traced(benchmark_name(sys.argv))
        async def chat_completion(self, **kw):
            mlflow_trace.note(inputs={k: kw.get(k) for k in ("model", "messages", "tools", "temperature",
                                                              "max_tokens", "extra_params")})
            r = await _orig(self, **kw)
            mlflow_trace.note(
                outputs={"content": r.content, "reasoning": r.reasoning, "finish_reason": r.finish_reason,
                         "tool_calls": [{"name": t.name, "arguments": t.arguments_str} for t in r.tool_calls]},
                prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
                ttft_s=r.ttft_ms / 1000 if r.ttft_ms is not None else None, elapsed_s=r.elapsed_ms / 1000)
            return r

        OpenAICompatibleAdapter.chat_completion = chat_completion
    except (ImportError, AttributeError) as ex:
        print(f"mlflow tracing off for tool-eval-bench: {ex}", file=sys.stderr)

if __name__ == "__main__":
    sys.exit(main())
