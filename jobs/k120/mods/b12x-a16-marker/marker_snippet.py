


# ---- k120 path marker (mods/b12x-a16-marker) ----
# Appended to b12x/gemm/blockscaled/_a16_dequant.py inside the prebake container only (never on a measured boot).
# _preparation.py calls _a16_dequant.engaged(...) at plan time and _a16_dequant.run(...) per serving call through the
# module attribute, so rebinding both names here logs each plan decision and the first BF16-path call per weight shape
# on this rank, then runs the original code unchanged.
import sys as _k120_sys

_k120_seen = set()
_k120_engaged, _k120_run = engaged, run


def _k120_log(msg):
    try:
        print(f"k120 marker: {msg}", file=_k120_sys.stderr, flush=True)
    except Exception:
        pass


def engaged(query, sm_count):
    result = _k120_engaged(query, sm_count)
    try:
        key = ("plan", query.recipe, query.in_features, query.out_features, query.expected_m, int(sm_count), result)
        if key not in _k120_seen:
            _k120_seen.add(key)
            _k120_log(f"plan recipe={query.recipe} k={query.in_features} n={query.out_features} "
                      f"expected_m={query.expected_m} sm={int(sm_count)} engaged={result}")
    except Exception:
        pass
    return result


def run(source, values, scale_bytes, out):
    try:
        rows = source.reshape(-1, source.shape[-1]).shape[0]
        key = ("run", tuple(values.shape))
        if key not in _k120_seen:
            _k120_seen.add(key)
            _k120_log(f"BF16 path ran n={values.shape[0]} k={values.shape[1]} rows={rows}")
    except Exception:
        pass
    return _k120_run(source, values, scale_bytes, out)
# ---- end k120 path marker (mods/b12x-a16-marker) ----
