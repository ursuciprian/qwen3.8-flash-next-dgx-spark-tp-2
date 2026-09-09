"""Run on the Mac: python3 codex-optimization/test_package.py."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import yaml
import install

ROOT = Path(__file__).resolve().parent


def main():
    lock = json.loads((ROOT / 'source-lock.json').read_text())
    for name, digest in lock['files'].items():
        assert hashlib.sha256((ROOT / 'source' / name).read_bytes()).hexdigest() == digest, name
    for path in ROOT.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / 'vllm'
        shutil.copytree(ROOT / 'source', target)
        for patch in ('local-argmax/mtp.patch', 'cache-backport/pr53945.patch'):
            subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(ROOT / patch)],
                           cwd=temp, check=True, stdout=subprocess.PIPE)
        assert (target / 'models/qwen4_exp/nvidia/mtp.py').read_bytes() == (ROOT / 'local-argmax/mtp.py').read_bytes()
        for source in (ROOT / 'cache-backport/vllm').rglob('*.py'):
            assert (target / source.relative_to(ROOT / 'cache-backport/vllm')).read_bytes() == source.read_bytes()
    with tempfile.TemporaryDirectory() as temp:
        install.DEST = Path(temp)
        for arm in ('baseline', 'argmax', 'cache'):
            for name in set(lock['files']) | set(install.MAP.values()):
                target = install.DEST / name
                target.parent.mkdir(parents=True, exist_ok=True)
                source = ROOT / 'source' / name
                target.write_bytes(source.read_bytes() if source.exists() else b'baseline placeholder')
            with contextlib.redirect_stdout(io.StringIO()):
                install.install(arm)
            if arm == 'argmax':
                assert (install.DEST / 'models/qwen4_exp/nvidia/mtp.py').read_bytes() == (ROOT / 'local-argmax/mtp.py').read_bytes()
            if arm == 'cache':
                for source in (ROOT / 'cache-backport/vllm').rglob('*.py'):
                    assert (install.DEST / source.relative_to(ROOT / 'cache-backport/vllm')).read_bytes() == source.read_bytes()
        victim = install.DEST / 'config/speculative.py'
        victim.write_text('wrong source')
        try:
            install.install('baseline')
        except RuntimeError as error:
            assert 'Wrong image' in str(error)
        else:
            raise AssertionError('Installer accepted the wrong source')
    for path in (ROOT / 'recipes').glob('*.yaml'):
        recipe = yaml.safe_load(path.read_text())
        command = recipe['command'].format(model=recipe['model'], **recipe['defaults'])
        subprocess.run(['bash', '-n'], input=command, text=True, check=True)
        args = shlex.split(command.replace('\\\n', ' '))
        spec = json.loads(args[args.index('--speculative-config') + 1])
        assert recipe['defaults']['tensor_parallel'] == 2
        assert '--enable-expert-parallel' in args
        assert recipe.get('mods') is None  # Six baseline overlays are baked into image.
        assert spec.get('use_local_argmax_reduction', False) == (path.stem == 'argmax')
        assert spec.get('disable_eagle_block_drop', False) == (path.stem in ('baseline', 'argmax'))
        assert ('--enable-mamba-fine-grained-prefix-cache' in args) == (path.stem == 'cache-fine')
    print('PASS: source hashes, exact patch reproduction, Python syntax, installer variants/rejection, four TP2 recipes')


if __name__ == '__main__':
    main()
