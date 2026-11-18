"""ASE GrowingString driver: endpoints survive the run, interior rows come from the string."""

from __future__ import annotations

import os

import numpy as np
import pytest

if os.environ.get("RGSADDLE_REQUIRE_ASE") != "1":
    pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.calculator import Calculator

from rgsaddle.ase import GrowingString

MB_A = np.array([-200.0, -100.0, -170.0, 15.0])
MB_XX = np.array([-1.0, -1.0, -6.5, 0.7])
MB_XY = np.array([0.0, 0.0, 11.0, 0.6])
MB_YY = np.array([-10.0, -10.0, -6.5, 0.7])
MB_X0 = np.array([1.0, 0.0, -0.5, -1.0])
MB_Y0 = np.array([0.0, 0.5, 1.5, 1.0])

MIN_A = np.array([-0.558223634633, 1.441725841805, 0.0])
MIN_B = np.array([0.623499404931, 0.028037758529, 0.0])


class MuellerBrownCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        x = float(atoms.positions[0, 0]) - MB_X0
        y = float(atoms.positions[0, 1]) - MB_Y0
        terms = MB_A * np.exp(MB_XX * x * x + MB_XY * x * y + MB_YY * y * y)
        gx = float(np.sum(terms * (2.0 * MB_XX * x + MB_XY * y)))
        gy = float(np.sum(terms * (MB_XY * x + 2.0 * MB_YY * y)))
        forces = np.zeros_like(atoms.positions)
        forces[0, 0] = -gx
        forces[0, 1] = -gy
        self.results = {"energy": float(np.sum(terms)), "forces": forces}


def _images(count: int) -> list[Atoms]:
    images = []
    for idx in range(count):
        weight = idx / (count - 1)
        atoms = Atoms("H", positions=[(1.0 - weight) * MIN_A + weight * MIN_B])
        atoms.calc = MuellerBrownCalculator()
        images.append(atoms)
    return images


def _growing_string(images: list[Atoms]) -> GrowingString:
    return GrowingString(
        images,
        frontier_tol=15.0,
        max_steps_per_node=10,
        climb=True,
        ci_after=0.0,
        ci_after_rel=0.5,
        method="lbfgs",
        max_move=0.05,
    )


def test_growing_string_run_leaves_both_endpoints_untouched() -> None:
    images = _images(11)
    reactant = images[0].get_positions().copy()
    product = images[-1].get_positions().copy()
    string = _growing_string(images)

    assert string.run(fmax=0.001, steps=800)

    assert np.array_equal(images[0].get_positions(), reactant)
    assert np.array_equal(images[-1].get_positions(), product)


def test_growing_string_run_writes_joined_rows_to_interior_images() -> None:
    images = _images(11)
    product = images[-1].get_positions().copy()
    string = _growing_string(images)

    assert string.run(fmax=0.001, steps=800)

    joined = np.asarray(string._session.positions(), dtype=np.float64).reshape(11, 1, 3)
    for idx in range(1, 10):
        assert np.array_equal(images[idx].get_positions(), joined[idx])
    assert np.array_equal(joined[-1], product)
    assert string.ci is not None
    assert 1 <= string.ci <= 9


def test_growing_string_unfinished_run_restores_endpoints() -> None:
    images = _images(11)
    reactant = images[0].get_positions().copy()
    product = images[-1].get_positions().copy()
    string = _growing_string(images)

    assert not string.run(fmax=1.0e-12, steps=25)

    assert string._session is not None
    assert int(string._session.n_images()) > 4
    assert np.array_equal(images[0].get_positions(), reactant)
    assert np.array_equal(images[-1].get_positions(), product)
