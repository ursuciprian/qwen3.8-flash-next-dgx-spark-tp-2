#!/usr/bin/env python3
"""Anchor-based text patcher for the vllm-spec-trace mod. Fails closed: each
anchor must match exactly once in its target file, or nothing is written."""
import sys

TRACE_HELPER = '''
def _spec_trace(stage, num_reqs, req_ids, per_req):
    """Debug-only tracer, VLLM_SPEC_TRACE=1. Never raises into the serving path."""
    import os
    if os.environ.get("VLLM_SPEC_TRACE") != "1":
        return
    try:
        import json, itertools
        global _SPEC_TRACE_STEP
        try:
            _SPEC_TRACE_STEP += 1
        except NameError:
            _SPEC_TRACE_STEP = 0
        rows = []
        for i in range(num_reqs):
            rid = req_ids[i] if req_ids is not None and i < len(req_ids) else i
            row = {"req": rid}
            row.update({k: (v[i] if v is not None and i < len(v) else None)
                        for k, v in per_req.items()})
            rows.append(row)
        with open("/tmp/spec_trace.log", "a") as f:
            f.write(json.dumps({"step": _SPEC_TRACE_STEP, "stage": stage,
                                 "num_reqs": num_reqs, "reqs": rows}) + "\\n")
    except Exception as e:  # pragma: no cover - tracing must never break serving
        try:
            with open("/tmp/spec_trace.log", "a") as f:
                f.write(f'{{"stage": "{stage}", "trace_error": "{e}"}}\\n')
        except Exception:
            pass
'''

SPEC_ANCHOR_EARLY = """        if num_speculative_tokens == 1:
            # Early exit.
            return self.draft_tokens[:num_reqs, :1]"""
SPEC_REPLACE_EARLY = """        if num_speculative_tokens == 1:
            # Early exit.
            _spec_trace("propose", num_reqs, list(input_batch.req_ids[:num_reqs]),
                         {"drafts_proposed": [1] * num_reqs,
                          "in_spec_batch": [True] * num_reqs,
                          "skip_reason": [None] * num_reqs})
            return self.draft_tokens[:num_reqs, :1]"""

SPEC_ANCHOR_FINAL = """        return self.draft_tokens[:num_reqs, :num_speculative_tokens]

    @torch.inference_mode()
    def _run_model("""
SPEC_REPLACE_FINAL = """        _spec_trace("propose", num_reqs, list(input_batch.req_ids[:num_reqs]),
                     {"drafts_proposed": [num_speculative_tokens] * num_reqs,
                      "in_spec_batch": [True] * num_reqs,
                      "skip_reason": [None] * num_reqs})
        return self.draft_tokens[:num_reqs, :num_speculative_tokens]

    @torch.inference_mode()
    def _run_model("""

REJ_ANCHOR = """        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )"""
REJ_REPLACE = """        _spec_trace("verify", input_batch.num_reqs, list(input_batch.req_ids),
                     {"num_sampled": num_sampled.tolist(),
                      "num_rejected": num_rejected.tolist()})
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )"""


def apply_one(path, anchor, replacement, label):
    src = open(path).read()
    n = src.count(anchor)
    if n != 1:
        sys.exit(f"FATAL: vllm-spec-trace anchor '{label}' matched {n} times "
                  f"(expected 1) in {path}")
    return src.replace(anchor, replacement, 1)


def main():
    spec_path, rej_path = sys.argv[1], sys.argv[2]

    spec_src = apply_one(spec_path, SPEC_ANCHOR_EARLY, SPEC_REPLACE_EARLY, "propose-early-exit")
    n = spec_src.count(SPEC_ANCHOR_FINAL)
    if n != 1:
        sys.exit(f"FATAL: vllm-spec-trace anchor 'propose-final-return' matched {n} "
                  f"times (expected 1) in {spec_path}")
    spec_src = spec_src.replace(SPEC_ANCHOR_FINAL, SPEC_REPLACE_FINAL, 1)
    spec_src = TRACE_HELPER + "\n" + spec_src
    open(spec_path, "w").write(spec_src)

    rej_src = apply_one(rej_path, REJ_ANCHOR, REJ_REPLACE, "rejection-sampler-return")
    rej_src = TRACE_HELPER + "\n" + rej_src
    open(rej_path, "w").write(rej_src)


if __name__ == "__main__":
    main()
