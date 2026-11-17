"""Saddle-index certification and escape through the extension and ASE.

V = (x^4 - 2 x^2) / 4 + (y^4 - y^2) / 4 + z^2 / 2 has an index-2
saddle at the origin, curvatures -1 and -1/2, and index-1 saddles at
(0, +-sqrt(1/2), 0).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import rgsaddle
from rgsaddle import _core

Float64 = NDArray[np.float64]


def wells(x: Float64) -> tuple[Float64, Float64]:
    rows = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    a, b, c = rows[:, 0], rows[:, 1], rows[:, 2]
    energy = (a**4 - 2.0 * a * a) / 4.0 + (b**4 - b * b) / 4.0 + c * c / 2.0
    gradient = np.stack([a**3 - a, b**3 - 0.5 * b, c], axis=1)
    return energy, gradient


def test_the_point_check_counts_both_negative_modes() -> None:
    top = rgsaddle.saddle_index(wells, np.zeros(3))
    assert isinstance(top, rgsaddle.SaddleIndex)
    assert top.certified and top.converged
    assert top.index == 2
    assert np.allclose(top.curvatures, [-1.0, -0.5, 1.0], atol=1e-2)
    assert top.evaluations > 0
    assert top.total_evaluations == top.evaluations
    side = rgsaddle.saddle_index(wells, np.array([0.0, np.sqrt(0.5), 0.0]))
    assert side.index == 1
    held = rgsaddle.saddle_index(wells, np.zeros(3), fixed=np.array([1], dtype=np.uint8))
    assert held.index == 0
    assert held.curvatures.shape == (0,)


def test_the_point_check_refuses_bad_arguments() -> None:
    with pytest.raises(ValueError):
        rgsaddle.saddle_index(wells, np.zeros(3), fixed=np.array([2], dtype=np.uint8))
    with pytest.raises(ValueError):
        rgsaddle.saddle_index(wells, np.zeros(3), lowest=0)
    with pytest.raises(ValueError):
        rgsaddle.saddle_index(wells, np.zeros(3), gradient=np.zeros(6))
    with pytest.raises(ValueError):
        rgsaddle.saddle_index(wells, np.zeros(4))


def _converge(search: _core.MinMode) -> list[_core.SessionReport]:
    reports = []
    for _ in range(5000):
        report = search.step(wells)
        reports.append(report)
        if report.status == 1:
            break
    return reports


def test_the_min_mode_session_certifies_the_index_two_top() -> None:
    search = _core.MinMode(np.array([0.2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), force_tol=1e-6, max_move=0.1)
    reports = _converge(search)
    assert reports[-1].status == 1
    assert reports[-1].saddle_index == 2
    assert all(report.saddle_index == -1 for report in reports[:-1])
    certificate = search.saddle_index()
    assert certificate.certified and certificate.index == 2
    assert certificate.escapes == 0
    search.set_saddle_check(False)
    assert not search.saddle_index().certified


def test_the_min_mode_escape_reaches_the_index_one_saddle() -> None:
    search = _core.MinMode(np.array([0.2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), force_tol=1e-6, max_move=0.1)
    search.set_saddle_check(escape_attempts=2)
    reports = _converge(search)
    assert reports[-1].status == 1
    assert reports[-1].saddle_index == 1
    certificate = search.saddle_index()
    assert certificate.index == 1 and certificate.escapes == 1
    assert certificate.total_evaluations > certificate.evaluations
    x = search.position()
    assert abs(x[0]) < 1e-4
    assert abs(abs(x[1]) - np.sqrt(0.5)) < 1e-4
    with pytest.raises(ValueError):
        search.set_saddle_check(escape_attempts=1, escape_push=0.0)
