#!/usr/bin/env python3
"""Regenerate python/rgsaddle/_core.pyi with the pinned nanobind stubgen.

The typing job runs ``python -m nanobind.stubgen -m rgsaddle._core`` on
the installed package and diffs the result against the committed stub.
The stub depends on two copies of nanobind: the one compiled into the
extension, which renders each signature, and the stubgen module, which
writes the file. ``build-system.requires`` in pyproject.toml pins the
first. This script refuses any other stubgen version.

Install the package against the pinned nanobind first:

    python -m pip install 'nanobind==<pin>' 'scikit-build-core>=1.1'
    python -m pip install --no-build-isolation .
    python scripts/regen_stub.py          # rewrite the stub
    python scripts/regen_stub.py --check  # diff only, as CI does
"""

from __future__ import annotations

import difflib
import importlib.metadata
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = "rgsaddle._core"
STUB = ROOT / "python" / "rgsaddle" / "_core.pyi"


def pinned_nanobind() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'"nanobind==([0-9][0-9A-Za-z.]*)"', text)
    if match is None:
        raise SystemExit("pyproject.toml does not pin nanobind with ==")
    return match.group(1)


def generate(out: Path) -> str:
    subprocess.run(
        [sys.executable, "-m", "nanobind.stubgen", "-q", "-m", MODULE, "-o", str(out)],
        check=True,
    )
    return out.read_text(encoding="utf-8")


def main() -> int:
    args = sys.argv[1:]
    if args not in ([], ["--check"]):
        raise SystemExit("usage: regen_stub.py [--check]")
    pin = pinned_nanobind()
    try:
        found = importlib.metadata.version("nanobind")
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(f"nanobind is not installed; pip install 'nanobind=={pin}'") from None
    if found != pin:
        raise SystemExit(f"nanobind {found} is installed; the stub is pinned to {pin}")
    with tempfile.TemporaryDirectory() as scratch:
        fresh = generate(Path(scratch) / "_core.pyi")
    committed = STUB.read_text(encoding="utf-8")
    if fresh == committed:
        print(f"{STUB.relative_to(ROOT)} matches nanobind {pin} stubgen")
        return 0
    if args == ["--check"]:
        rel = str(STUB.relative_to(ROOT))
        sys.stdout.writelines(
            difflib.unified_diff(
                committed.splitlines(keepends=True),
                fresh.splitlines(keepends=True),
                fromfile=rel,
                tofile=f"stubgen {MODULE}",
            )
        )
        print(f"{rel} is stale; run python scripts/regen_stub.py", file=sys.stderr)
        return 1
    STUB.write_text(fresh, encoding="utf-8")
    print(f"wrote {STUB.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
