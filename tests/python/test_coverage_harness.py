"""Saddle-coverage harness checks that do not need a potential."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "saddle_coverage", ROOT / "bench" / "harness.py"
)
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def test_write_table_keeps_the_build_section(tmp_path):
    payload = {
        "summary": [],
        "systems": {},
        "comparisons": [],
        "runs": [],
        "shas": {
            "rgsaddle": "abc12345",
            "rgpot": "def12345",
            "eon": "ghi12345",
        },
        "versions": {
            "ase": "3.29.0",
            "pyeonclient": "0.4.2",
            "rgsaddle_abi": "1.42",
            "numpy": "2.5.5",
        },
        "cpu_model": "Example CPU",
        "threads": 1,
        "rounds": 12,
    }
    path = tmp_path / "results.md"
    harness.write_table(str(path), payload)
    text = path.read_text(encoding="utf-8")
    assert "## Build" in text
    assert "Example CPU" in text
    assert "3.29.0" in text
    assert "0.4.2" in text
    assert "abc12345" in text
    harness.write_table(str(path), payload)
    rewritten = path.read_text(encoding="utf-8")
    assert rewritten.count("## Build") == 1
    assert "Example CPU" in rewritten


def test_coverage_record_stores_the_cpu_and_versions():
    class Args:
        shas = {"rgpot": "a" * 8, "eon": "b" * 8, "rgsaddle": "c" * 8}
        rounds = 12
        threads = 1

    recorded = {"ase": "3.29.0", "numpy": "2.5.5", "pyeonclient": "0.4.2"}
    payload = harness.coverage_record(Args, recorded, {"runs": []})
    assert payload["versions"]["ase"] == "3.29.0"
    assert payload["versions"]["pyeonclient"] == "0.4.2"
    assert payload["versions"]["eon_project"] == harness.EON_PROJECT_VERSION
    assert payload["cpu_model"]
    assert not str(payload["cpu_model"]).startswith("/")
