"""Saddle index of ASE atoms and of an ASE band's climbing image."""

from __future__ import annotations

import os

import numpy as np
import pytest

# RGSADDLE_REQUIRE_ASE=1 turns the skip into a collection error.
if os.environ.get("RGSADDLE_REQUIRE_ASE") != "1":
    pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from ase.optimize import FIRE

from rgsaddle.ase import NEB, saddle_index


class Wells(Calculator):
    """Atom 0 on V = (x^4 - 2 x^2) / 4 + (y^4 - y^2) / 4 + z^2 / 2, atom 1 on a spring.

    The origin is an index-2 saddle of V and (0, +-sqrt(1/2), 0) are index-1 saddles.
    """

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        r = atoms.positions
        x, y, z = r[0]
        energy = (x**4 - 2.0 * x * x) / 4.0 + (y**4 - y * y) / 4.0 + z * z / 2.0
        forces = np.zeros_like(r)
        forces[0] = [x - x**3, 0.5 * y - y**3, -z]
        forces[1] = -r[1]
        self.results = {"energy": energy + 0.5 * float(np.dot(r[1], r[1])), "forces": forces}


def _atoms(position: list[float], fix: list[int] | None = None) -> Atoms:
    atoms = Atoms("H2", positions=[position, [0.4, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=False)
    if fix is not None:
        atoms.set_constraint(FixAtoms(indices=fix))
    atoms.calc = Wells()
    return atoms


def test_atoms_report_their_index_and_stay_put() -> None:
    atoms = _atoms([0.0, 0.0, 0.0])
    before = atoms.get_positions().copy()
    assert saddle_index(atoms).index == 2
    assert np.array_equal(atoms.get_positions(), before)
    held = saddle_index(_atoms([0.0, 0.0, 0.0], fix=[1]))
    assert held.converged
    assert held.index == 2
    assert saddle_index(_atoms([0.0, float(np.sqrt(0.5)), 0.0], fix=[1])).index == 1


def test_the_band_certifies_its_climbing_image() -> None:
    images = []
    for i in range(7):
        t = i / 6
        atoms = Atoms(
            "H2",
            positions=[[-1.0 + 2.0 * t, np.sqrt(0.5) + 0.2 * np.sin(np.pi * t), 0.0], [0.4, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=False,
        )
        atoms.set_constraint(FixAtoms(indices=[1]))
        atoms.calc = Wells()
        images.append(atoms)
    neb = NEB(images, k=1.0, climb=True)
    assert FIRE(neb, logfile=None).run(fmax=1e-4, steps=3000)
    assert neb.ci is not None
    certificate = neb.saddle_index()
    assert certificate.index == 1
    assert certificate.curvatures[0] < -0.5
    with pytest.raises(ValueError):
        NEB(images, k=1.0).saddle_index()
