"""Optional MLflow tracing for the benchmark and quality-gate clients.

Off unless MLFLOW_TRACKING_URI is set and `import mlflow` works. Off means the
clients run exactly as before: `traced` returns the function unchanged and
`note` does nothing. Tracing never fails a benchmark: a missing package or an
unreachable server prints one line to stderr and the run continues untraced.
The light client is enough: pip install "mlflow-tracing>=3.17" (or full mlflow).

  MLFLOW_TRACKING_URI     http://192.168.68.59:5050 on the Sparks, http://localhost:5050 on the Mac
  MLFLOW_EXPERIMENT_NAME  default my-experiment
  TRACE_RELEASE           release tag, default RELEASE below
  TRACE_HOST              host tag, default the hostname

Each traced call is one trace with one LLM span. The span holds the request
inputs, the model output, prompt/completion tokens (also as mlflow.chat.tokenUsage)
and ttft_s where the client streams. Trace tags: benchmark, release, host, run.
"""
import functools
import inspect
import os
import socket
import sys
import time

RELEASE = "2x v2.0.0"  # ponytail: bump with each release, or set TRACE_RELEASE in the job

mlflow = None
if os.environ.get("MLFLOW_TRACKING_URI"):
    # fail fast instead of minutes of retries when the tracking server is down
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    try:
        import mlflow
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "my-experiment"))
    except Exception as ex:
        print(f"mlflow tracing off: {type(ex).__name__}: {str(ex)[:200]}", file=sys.stderr)
        mlflow = None

TAGS = {"release": os.environ.get("TRACE_RELEASE", RELEASE),
        "host": os.environ.get("TRACE_HOST") or socket.gethostname(),
        "run": time.strftime("%Y%m%dT%H%M%S")}


def traced(benchmark):
    """Decorator: every call becomes a trace tagged with `benchmark`. Identity when tracing is off."""
    def deco(fn):
        if mlflow is None:
            return fn

        def tag():
            mlflow.update_current_trace(tags={**TAGS, "benchmark": benchmark})

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def inner(*a, **kw):
                tag()
                return await fn(*a, **kw)
        else:
            @functools.wraps(fn)
            def inner(*a, **kw):
                tag()
                return fn(*a, **kw)
        return mlflow.trace(inner, name=f"{benchmark}.{fn.__name__}", span_type="LLM",
                            attributes={"benchmark": benchmark})
    return deco


def note(inputs=None, outputs=None, prompt_tokens=None, completion_tokens=None, **attrs):
    """Set inputs, outputs, token counts and other attributes (ttft_s, ...) on the current span."""
    span = mlflow.get_current_active_span() if mlflow else None
    if span is None:
        return
    if inputs is not None:
        span.set_inputs(inputs)
    if outputs is not None:
        span.set_outputs(outputs)
    if prompt_tokens is not None or completion_tokens is not None:
        p, c = prompt_tokens or 0, completion_tokens or 0
        attrs.update(prompt_tokens=p, completion_tokens=c)
        attrs["mlflow.chat.tokenUsage"] = {"input_tokens": p, "output_tokens": c, "total_tokens": p + c}
    span.set_attributes({k: v for k, v in attrs.items() if v is not None})
