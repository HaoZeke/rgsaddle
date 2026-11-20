#!/usr/bin/env python3
"""Download and verify the open PW91 USPP UPF files for the Ta/H example.

Run from the repository root:

    uv run --no-project python validation/lineb/tantalum/open_qe/fetch_pseudopotentials.py

Files land in validation/lineb/tantalum/open_qe/pseudo/ and are gitignored.
"""

from __future__ import annotations

import hashlib
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PSEUDO = ROOT / "pseudo"
PROV = ROOT / "PROVENANCE.toml"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    with PROV.open("rb") as fh:
        prov = tomllib.load(fh)
    PSEUDO.mkdir(parents=True, exist_ok=True)
    for key in ("Ta", "H"):
        meta = prov["pseudopotentials"][key]
        dest = PSEUDO / meta["filename"]
        expected = meta["sha256"]
        if dest.is_file() and sha256_file(dest) == expected:
            print(f"OK (cached) {dest.name}")
            continue
        print(f"fetch {meta['url']}")
        with urllib.request.urlopen(meta["url"], timeout=90) as response:
            data = response.read()
        got = hashlib.sha256(data).hexdigest()
        if got != expected:
            raise SystemExit(f"SHA-256 mismatch for {dest.name}: {got} != {expected}")
        dest.write_bytes(data)
        print(f"OK {dest.name}  {got}")
    print(f"pseudopotentials ready under {PSEUDO}")


if __name__ == "__main__":
    main()
