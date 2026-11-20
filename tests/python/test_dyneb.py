"""Dynamic NEB mask and the ASE DyNEB driver."""

from __future__ import annotations

import os

import numpy as np
import pytest

if os.environ.get("RGSADDLE_REQUIRE_ASE") != "1":
    pytest.importorskip("ase")

from ase import Atoms
from ase.calculators.calculator import Calculator

from rgsaddle._core import dyneb_mask
from rgsaddle.ase import DyNEB, NEB


def test_dyneb_mask_zeros_a_side_image_and_keeps_the_highest() -> None:
    positions = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 0.0, 0.0],
        dtype=np.float64,
    )
    energies = np.array([0.0, 1.0, 5.0, 4.0], dtype=np.float64)
    forces = np.array(
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0],
        dtype=np.float64,
    )
    masked, frozen = dyneb_mask(positions, energies, forces, 1.0, 1.0)
    masked = np.asarray(masked)
    frozen = np.asarray(frozen)
    assert np.all(masked[3:6] == 0.0)
    assert masked[7] == 2.0
    assert frozen[0] == 1 and frozen[1] == 1 and frozen[2] == 0


class _Bowl(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        r = atoms.positions
        self.results = {
            "energy": float(np.sum(r * r)),
            "forces": -2.0 * r,
        }


def _images() -> list[Atoms]:
    images = []
    for x, y in ((-1.0, 0.0), (-0.3, 0.5), (0.3, 0.2), (1.0, 0.0)):
        atoms = Atoms("H", positions=[[x, y, 0.0]])
        atoms.calc = _Bowl()
        images.append(atoms)
    return images


def test_dyneb_leaves_a_quiet_image_and_neb_refuses_the_keywords() -> None:
    with pytest.raises(ValueError, match="unsupported NEB argument"):
        NEB(_images(), dynamic_relaxation=True)
    neb = NEB(_images(), k=0.1, climb=False)
    plain = np.array(neb.get_forces(), copy=True)
    dynamic = DyNEB(_images(), k=0.1, climb=False, fmax=1.0e6, scale_fmax=0.0)
    masked = dynamic.get_forces()
    assert masked.shape == plain.shape
    assert not np.allclose(plain, masked)
    before = [np.array(image.positions, copy=True) for image in dynamic.images]
    moved = np.array(dynamic.get_positions(), copy=True)
    moved[:, 0] += 0.2
    dynamic.set_positions(moved)
    for image, start in zip(dynamic.images, before, strict=True):
        assert np.allclose(image.positions, start)
