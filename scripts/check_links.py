#!/usr/bin/env python3
"""Broken-reference check for the repo's markdown.

  scripts/check_links.py [FILE.md ...]   # default: README.md, docs/, results/README.md, recipes/, mods/

Checks relative markdown links `[x](path)` and backticked repo paths
(`recipes/...`, `mods/...`, `archive/...`, `docs/...`, `scripts/...`,
`results/...`, `docker/...`, `patches/...`) against the working tree.
Exit 1 on any miss.
"""
import re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\(([^)\s#]+)(?:#[^)]*)?\)")
TICK = re.compile(r"`((?:recipes|mods|archive|docs|scripts|results|docker|patches)/[^`\s*<>{}]+)`")


def targets(md: Path):
    for n, line in enumerate(md.read_text().splitlines(), 1):
        for m in LINK.finditer(line):
            t = m.group(1)
            if not re.match(r"[a-z]+:", t):
                yield n, t, (md.parent / t)
        for m in TICK.finditer(line):
            t = m.group(1).rstrip(".,;:")
            # results/arms/ is gitignored raw data that lives on dgx-01, not here
            if t.startswith("results/arms/"):
                continue
            yield n, t, REPO / t


def main() -> int:
    files = [Path(a).resolve() for a in sys.argv[1:]] or [
        REPO / "README.md", REPO / "results/README.md", REPO / "mods/README.md",
        *sorted((REPO / "docs").glob("*.md")), *sorted((REPO / "recipes").glob("*.md"))]
    bad = 0
    for md in files:
        for n, t, p in targets(md):
            if not p.exists():
                bad += 1
                print(f"{md.relative_to(REPO)}:{n}: {t}")
    print(f"{len(files)} files, {bad} broken")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
