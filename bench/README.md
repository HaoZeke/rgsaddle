# Saddle-search coverage

`harness.py` builds the five systems and times the matched saddle searches.
Every path argument is required. There is no default directory.

```sh
python3 bench/harness.py \
  --phase setup \
  --out DIR \
  --pythonpath DIR \
  --libdir DIR \
  --rgsaddle-lib FILE \
  --cpu N \
  --threads N \
  --rounds 12 \
  --sha rgpot=HEX \
  --sha eon=HEX \
  --sha rgsaddle=HEX
```

`--pythonpath` may be repeated. `setup` writes `systems.json`. `measure` also
needs `--reference` pointing at that file, and writes `results.json`,
`results.md`, and `versions.json`.

`results.json` stores the host CPU model and the same mapping `versions.json`
writes. `results.md` is rewritten from that record, including the `## Build`
section, so a later `measure` does not drop the build table.

## libeonclib

`pyeonclient` loads `libeonclib` when its extension is imported. The dynamic
loader does not search the Python package path, so the import fails unless
that library is already loaded or its directory is on `LD_LIBRARY_PATH`.

Before the import, the harness looks for `libeonclib.so` (or a versioned
soname) in this order:

1. `--libdir`
2. the `pyeonclient` directory on `--pythonpath`, its parent, and `parent/lib`
3. directories already listed in `LD_LIBRARY_PATH`

It preloads the file it finds and puts that directory on `LD_LIBRARY_PATH`.
If none of those places has the library, the process exits with a message
that names `libeonclib` and `LD_LIBRARY_PATH` instead of a bare loader error.

Put the library next to the `pyeonclient` package, pass its directory as
`--libdir`, or export `LD_LIBRARY_PATH` to that directory before starting
the harness.
