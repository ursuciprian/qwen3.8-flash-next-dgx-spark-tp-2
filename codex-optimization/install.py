"""Install an isolated candidate into the pinned image; never run on the host."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
DEST = Path('/usr/local/lib/python3.12/dist-packages/vllm')
BASE = ROOT.parent / 'mods/vllm-flashnext-nightly-8a728663'
MAP = {
    'ops_ple.py': 'models/qwen4_exp/nvidia/ops/ple.py',
    'ops_qsa.py': 'models/qwen4_exp/nvidia/ops/qsa.py',
    'qsa.py': 'models/qwen4_exp/nvidia/qsa.py',
    'platforms_interface.py': 'platforms/interface.py',
    'modelopt.py': 'model_executor/layers/quantization/modelopt.py',
    'kv_cache_utils.py': 'v1/core/kv_cache_utils.py',
}


def install(arm):
    if arm not in ('baseline', 'argmax', 'cache', 'fp8-refine8'):
        raise ValueError(f'Unknown arm: {arm}')
    lock = json.loads((ROOT / 'source-lock.json').read_text())
    # Check the whole dependency set before replacing anything.
    for name, digest in lock['files'].items():
        actual = hashlib.sha256((DEST / name).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f'Wrong image or already modified source: {name}')
    copies = [(BASE / name, DEST / target) for name, target in MAP.items()]
    if arm == 'argmax':
        copies.append((ROOT / 'local-argmax/mtp.py', DEST / 'models/qwen4_exp/nvidia/mtp.py'))
    if arm == 'fp8-refine8':
        copies.append((ROOT / 'fp8-refine8/mtp.py', DEST / 'models/qwen4_exp/nvidia/mtp.py'))
    if arm == 'cache':
        for path in sorted((ROOT / 'cache-backport/vllm').rglob('*.py')):
            copies.append((path, DEST / path.relative_to(ROOT / 'cache-backport/vllm')))
    for source, target in copies:
        if not source.is_file() or not target.is_file():
            raise FileNotFoundError(f'{source} -> {target}')
    for source, target in copies:
        shutil.copyfile(source, target)
        print(f'{arm}: {target.relative_to(DEST)}')


if __name__ == '__main__':
    install(sys.argv[1])
