

# ---- local step profiler (mods/vllm-decode-profiler) ----
# Rank-local torch.profiler wrapper around GPUModelRunner.execute_model.
# Purely additive: appended after the class body, never edits vLLM's own
# logic, no cross-rank RPC (each worker process profiles only itself).
#
# Armed via VLLM_LOCAL_PROF_TRIGGER_DIR: drop any file into that directory
# and profiling starts on the next real decode step; the file's own content
# (stripped) becomes the trace label, so one boot can be re-armed for several
# labels across rounds. Each round MUST use a new filename (e.g.
# go-c1-<epoch>) -- reusing one fixed path was tried first and is unreliable:
# this worker process is long-lived, and once it has stat()'d a path as
# "missing" a few hundred times, recreating that exact path is not always
# observed again (a stale negative-lookup effect on the bind-mounted cache
# dir; a fresh process always sees it fine, but the same long-lived PID does
# not). Listing the directory and picking any entry sidesteps that: a
# directory read is a different, non-cached-per-name syscall than repeatedly
# stat()'ing one fixed inode. VLLM_LOCAL_PROF_START/VLLM_LOCAL_PROF_LABEL
# (a fixed step index / fixed label) remain as a fallback that never touches
# the filesystem after boot.
#
# Runs for VLLM_LOCAL_PROF_STEPS steps (default 100), then exports a chrome
# trace to /cache/runtime/prof/<label>/rank<tp_rank>.json. Every profiler
# action is try/except-guarded so a failure here can never break serving.
import os as _lp_os
import logging as _lp_logging

_lp_logger = _lp_logging.getLogger("vllm.local_profiler")


class _LocalStepProfiler:
    def __init__(self):
        self.step = 0
        self.prof = None
        self.steps_left = 0
        self.label = None

    def maybe_tick(self):
        trigger_dir = _lp_os.environ.get("VLLM_LOCAL_PROF_TRIGGER_DIR")
        start_step_raw = _lp_os.environ.get("VLLM_LOCAL_PROF_START")
        if not trigger_dir and not start_step_raw:
            return
        self.step += 1
        try:
            if self.prof is not None:
                self.prof.step()
                self.steps_left -= 1
                if self.steps_left <= 0:
                    self._finish()
                return
            if trigger_dir:
                try:
                    names = [n for n in _lp_os.listdir(trigger_dir) if not n.startswith(".")]
                except Exception:
                    names = []
                if names:
                    path = _lp_os.path.join(trigger_dir, sorted(names)[0])
                    trigger_label = None
                    try:
                        trigger_label = open(path).read().strip() or None
                    except Exception:
                        pass
                    try:
                        _lp_os.remove(path)
                    except Exception:
                        pass
                    self._start(trigger_label)
                    return
            start_step = int(start_step_raw or 0)
            if start_step and self.step >= start_step:
                self._start(None)
        except Exception:
            _lp_logger.exception("local step profiler tick failed, disabling")
            self.prof = None
            self.steps_left = 0

    def _start(self, trigger_label):
        import torch as _t

        self.steps_left = int(_lp_os.environ.get("VLLM_LOCAL_PROF_STEPS", "100") or 100)
        self.label = trigger_label or _lp_os.environ.get("VLLM_LOCAL_PROF_LABEL", "run")
        self.prof = _t.profiler.profile(
            activities=[_t.profiler.ProfilerActivity.CPU, _t.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        )
        self.prof.__enter__()
        _lp_logger.info(
            "local step profiler: started label=%s steps=%d", self.label, self.steps_left
        )

    def _finish(self):
        try:
            self.prof.__exit__(None, None, None)
            try:
                from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

                rank = get_tensor_model_parallel_rank()
            except Exception:
                rank = 0
            out_dir = f"/cache/runtime/prof/{self.label}"
            _lp_os.makedirs(out_dir, exist_ok=True)
            out_path = f"{out_dir}/rank{rank}.json"
            self.prof.export_chrome_trace(out_path)
            _lp_logger.info("local step profiler: wrote trace %s", out_path)
        except Exception:
            _lp_logger.exception("local step profiler: failed to export trace")
        finally:
            self.prof = None


_local_step_profiler = _LocalStepProfiler()
_lp_orig_execute_model = GPUModelRunner.execute_model


def _lp_profiled_execute_model(self, *args, **kwargs):
    try:
        if not kwargs.get("dummy_run", False) and not kwargs.get("is_profile", False):
            _local_step_profiler.maybe_tick()
    except Exception:
        _lp_logger.exception("local step profiler wrapper failed")
    return _lp_orig_execute_model(self, *args, **kwargs)


GPUModelRunner.execute_model = _lp_profiled_execute_model
# ---- end local step profiler (mods/vllm-decode-profiler) ----
