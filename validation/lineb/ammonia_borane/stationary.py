"""Optimize ammonia borane on the B3LYP(VWN5)/def2-SV(P) + gCP surface."""

import argparse
from pathlib import Path

from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.optimize import BFGS
from ase.units import Bohr, Hartree
from dftd3.interface import GeometricCounterpoise
from pyscf import dft, gto


class BoraneSurface(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        molecule = gto.M(atom=list(zip(atoms.symbols, atoms.positions)), unit="Angstrom",
                         basis="def2-SV(P)", charge=0, spin=0, verbose=0)
        model = dft.RKS(molecule)
        model.xc = "B3LYP5"
        model.grids.level = 4
        model.conv_tol = 1e-10
        energy = model.kernel()
        if not model.converged:
            raise RuntimeError("SCF did not converge")
        derivative = model.nuc_grad_method()
        derivative.grid_response = True
        correction = GeometricCounterpoise(atoms.numbers, atoms.positions / Bohr,
                                            method="dft", basis="sv(p)").get_counterpoise(grad=True)
        self.results = {
            "energy": float(energy + correction["energy"]) * Hartree,
            "forces": -(derivative.kernel() + correction["gradient"]) * Hartree / Bohr,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--saddle", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    atoms = read(args.geometry)
    atoms.calc = BoraneSurface()
    files = dict(trajectory=str(args.output / "steps.traj"), logfile=str(args.output / "steps.log"))
    if args.saddle:
        from sella import Sella

        optimizer = Sella(atoms, order=1, internal=True, **files)
    else:
        optimizer = BFGS(atoms, **files)
    if not optimizer.run(fmax=0.002, steps=250):
        raise RuntimeError("Stationary point did not converge")
    write(args.output / "stationary.extxyz", atoms)


if __name__ == "__main__":
    main()
