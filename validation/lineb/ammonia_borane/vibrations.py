"""Calculate internal harmonic modes at a gas-phase stationary point."""

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read
from ase.units import invcm, kB
from ase.vibrations import Vibrations

from stationary import BoraneSurface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    atoms = read(args.directory / "stationary.extxyz")
    energy = atoms.get_potential_energy()
    atoms.calc = BoraneSurface()
    vibrations = Vibrations(atoms, name=str(args.directory / "hessian"), delta=0.005)
    vibrations.run()
    hessian = vibrations.get_vibrations().get_hessian_2d()
    masses = atoms.get_masses()
    sqrt_mass = np.repeat(np.sqrt(masses), 3)
    weighted = hessian / np.outer(sqrt_mass, sqrt_mass)
    centered = atoms.positions - atoms.get_center_of_mass()
    translations = [np.tile(axis, len(atoms)) for axis in np.eye(3)]
    rotations = [np.cross(axis, centered).ravel() for axis in np.eye(3)]
    rigid = np.column_stack(translations + rotations) * sqrt_mass[:, None]
    basis, singular, _ = np.linalg.svd(rigid, full_matrices=True)
    internal = basis[:, np.count_nonzero(singular > singular[0] * 1e-10):]
    curvature = np.linalg.eigvalsh(internal.T @ weighted @ internal)
    from ase.units import _e, _amu, _hbar

    quanta = np.sign(curvature) * np.sqrt(np.abs(curvature)) * _hbar * 1e10 / np.sqrt(_e * _amu)
    unstable = quanta[quanta < 0]
    result = {
        "electronic_energy_eV": energy,
        "zero_point_energy_eV": float(quanta[quanta > 0].sum() / 2),
        "signed_wavenumbers_cm-1": (quanta / invcm).tolist(),
        "unstable_modes": len(unstable),
        "crossover_K": float(-unstable[0] / (2 * np.pi * kB)) if len(unstable) == 1 else None,
    }
    (args.directory / "vibrations.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
