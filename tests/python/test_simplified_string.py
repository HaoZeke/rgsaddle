"""Simplified string method in Python: Band session, project_band, and ASE SimplifiedString."""

from __future__ import annotations

import os

import numpy as np
import pytest

import rgsaddle
from rgsaddle import _core

MB_A = np.array([-200.0, -100.0, -170.0, 15.0])
MB_XX = np.array([-1.0, -1.0, -6.5, 0.7])
MB_XY = np.array([0.0, 0.0, 11.0, 0.6])
MB_YY = np.array([-10.0, -10.0, -6.5, 0.7])
MB_X0 = np.array([1.0, 0.0, -0.5, -1.0])
MB_Y0 = np.array([0.0, 0.5, 1.5, 1.0])


def _muller_brown_surface(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = positions[:, 0:1] - MB_X0[None, :]
    y = positions[:, 1:2] - MB_Y0[None, :]
    terms = MB_A[None, :] * np.exp(MB_XX * x * x + MB_XY * x * y + MB_YY * y * y)
    energies = np.sum(terms, axis=1)
    gx = np.sum(terms * (2.0 * MB_XX * x + MB_XY * y), axis=1)
    gy = np.sum(terms * (MB_XY * x + 2.0 * MB_YY * y), axis=1)
    gradients = np.column_stack([gx, gy, np.zeros_like(gx)])
    return np.ascontiguousarray(energies, dtype=np.float64), np.ascontiguousarray(
        gradients, dtype=np.float64
    )


def test_band_simplified_string_mueller_brown_matches_cineb() -> None:
    min_a = np.array([-0.558223634633, 1.441725841805, 0.0])
    min_b = np.array([0.623499404931, 0.028037758529, 0.0])
    weights = np.linspace(0.0, 1.0, 21)[:, None]
    initial = np.ascontiguousarray((1.0 - weights) * min_a + weights * min_b, dtype=np.float64)

    ci_string = rgsaddle.Band(
        initial,
        climb=True,
        ci_after=0.0,
        ci_after_rel=0.5,
        max_move=0.05,
        force_tol=0.001,
        string=True,
        reparametrization="not-a-knot",
    )
    rep_str = ci_string.run(_muller_brown_surface, max_steps=800)
    assert rep_str.status == 1
    assert rep_str.ci_index >= 1
    assert rep_str.saddle_index == 1
    cert = ci_string.saddle_index()
    assert cert.certified and cert.index == 1

    ci_neb = rgsaddle.Band(
        initial,
        k=200.0,
        climb=True,
        ci_after=0.0,
        ci_after_rel=0.5,
        method="lbfgs",
        max_move=0.05,
        force_tol=0.001,
        string=False,
    )
    rep_neb = ci_neb.run(_muller_brown_surface, max_steps=800)
    assert rep_neb.status == 1

    saddle_str = ci_string.positions()[rep_str.ci_index]
    saddle_neb = ci_neb.positions()[rep_neb.ci_index]
    assert float(np.linalg.norm(saddle_str - saddle_neb)) < 0.001

    energies, gradients, projected = ci_string.evaluation()
    assert energies.shape == (21,)
    assert gradients.shape == (21, 3)
    assert projected.shape == (19, 3)


def test_project_band_string_uses_full_force_and_reflects_climber() -> None:
    eps = float(np.finfo(np.float64).eps)
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.3, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    energies = np.array([0.0, 1.0, 3.0, 0.5], dtype=np.float64)
    gradients = np.array(
        [[0.0, 0.0, 0.0], [2.0, -1.5, 0.25], [1.0, 0.5, -0.5], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    flat, ci = _core.project_band(
        positions,
        energies,
        gradients,
        k=50.0,
        climb=True,
        projection="string",
        remove_translation=False,
    )
    forces = flat.reshape((4, 3))
    assert ci == 2
    assert float(np.max(np.abs(forces[0]))) == 0.0
    assert float(np.max(np.abs(forces[3]))) == 0.0
    # Non-climbing image 1 keeps the full potential force -gradient, unaffected by k=50.
    assert float(np.max(np.abs(forces[1] + gradients[1]))) <= 4.0 * eps
    # Climbing image 2 has the same Euclidean norm as its true force (Householder reflection).
    assert abs(float(np.linalg.norm(forces[2]) - np.linalg.norm(gradients[2]))) <= 16.0 * eps


def test_band_string_validation_and_toggle() -> None:
    band_pos = np.array([[-1.0, 0.0, 0.0], [0.0, 0.2, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        rgsaddle.Band(band_pos, method="rtr", string=True)
    with pytest.raises(ValueError):
        rgsaddle.Band(band_pos, string=True, reparametrization="quintic")
    session = rgsaddle.Band(band_pos, climb=False, string=True, reparametrization="linear")
    session.set_string(False)
    session.set_string(True, reparametrization="spline")
    session.restart()
    session.reset()


def test_ase_simplified_string_driver_and_fixed_atoms() -> None:
    if os.environ.get("RGSADDLE_REQUIRE_ASE") != "1":
        pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.constraints import FixAtoms
    from rgsaddle.ase import SimplifiedString

    class CoupledWell(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
            Calculator.calculate(self, atoms, properties, system_changes)
            r = atoms.positions
            x, y, z = float(r[0, 0]), float(r[0, 1]), float(r[0, 2])
            bond = r[1] - np.array([2.0, 0.5, -0.25])
            energy = (x * x - 1.0) ** 2 + 2.0 * y * y + 2.0 * z * z + float(np.dot(bond, bond))
            forces = np.zeros_like(r)
            forces[0] = [-4.0 * x * (x * x - 1.0), -4.0 * y, -4.0 * z]
            forces[1] = -2.0 * bond
            self.results = {"energy": float(energy), "forces": forces}

    images = []
    for idx in range(7):
        t = idx / 6.0
        pos = [[-1.0 + 2.0 * t, 0.3 * np.sin(np.pi * t), 0.0], [2.0, 0.5, -0.25]]
        atoms = Atoms("H2", positions=pos)
        atoms.set_constraint(FixAtoms(indices=[1]))
        atoms.calc = CoupledWell()
        images.append(atoms)

    frozen_before = [img.positions[1].copy() for img in images]
    steps_seen: list[int] = []
    string = SimplifiedString(
        images,
        climb=True,
        ci_after=0.5,
        reparametrization="not-a-knot",
        max_move=0.1,
    )
    string.attach(lambda: steps_seen.append(1))
    converged = string.run(fmax=0.001, steps=400)
    assert converged
    assert len(steps_seen) > 0
    assert string.ci is not None
    saddle = images[string.ci].positions[0]
    assert float(np.linalg.norm(saddle)) < 0.01
    for img, ref in zip(images, frozen_before):
        assert np.array_equal(img.positions[1], ref)
    cert = string.saddle_index()
    assert cert.index == 1
