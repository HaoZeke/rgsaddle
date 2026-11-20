"""Independent finite-bead saddle and harmonic flux for H2 on frozen Cu(110)."""

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
from ase.io import read
from scipy.linalg import eigh
from scipy.optimize import root
from scipy.sparse import block_diag, diags, eye, kron
from scipy.sparse.linalg import spsolve

HBAR = 0.06465415
KB = 8.617333262145e-5
MASS = 1.00793


class Slab:
    def __init__(self, library, geometry):
        atoms = read(geometry, format="eon")
        self.positions = np.ascontiguousarray(atoms.positions)
        self.numbers = np.asarray(atoms.numbers, dtype=np.int32)
        self.cell = np.ascontiguousarray(atoms.cell.array)
        if list(self.numbers) != [29] * 216 + [1, 1]:
            raise ValueError("expected 216 copper atoms followed by two hydrogen atoms")
        self.library = ctypes.CDLL(str(library))
        if hasattr(self.library, "rgpot_cuh2_ensure_params"):
            self.library.rgpot_cuh2_ensure_params()
        fp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.force = self.library.rgpot_cuh2_force
        self.force.argtypes = [ctypes.c_int, fp, np.ctypeslib.ndpointer(dtype=np.int32),
                               fp, fp, ctypes.POINTER(ctypes.c_double)]
        self.calls = 0

    def energy_gradient(self, q):
        self.positions[-2:] = q.reshape(2, 3) / np.sqrt(MASS)
        forces = np.zeros_like(self.positions)
        energy = ctypes.c_double()
        self.calls += 1
        if self.force(len(self.numbers), self.positions, self.numbers, self.cell,
                      forces, ctypes.byref(energy)):
            raise RuntimeError("CuH2 surface evaluation failed")
        return energy.value, -forces[-2:].ravel() / np.sqrt(MASS)

    def hessian(self, q):
        delta = 1e-4
        h = np.column_stack([(self.energy_gradient(q + dx)[1] - self.energy_gradient(q - dx)[1])
                             / (2 * delta) for dx in np.eye(6) * delta])
        return (h + h.T) / 2


def solve_ring(slab, seed, temperature, tolerance):
    """Newton on the reflection-symmetric half; verify the unfolded full ring."""
    half = len(seed)
    beads = 2 * (half - 1)
    tau = HBAR / (KB * temperature)
    kappa = (beads / tau) ** 2
    half_spring = diags([-np.ones(half-1), 2*np.ones(half), -np.ones(half-1)],
                        [-1, 0, 1], format="lil")
    half_spring[0, 1] = half_spring[-1, -2] = -2
    spring = kappa * kron(half_spring, eye(6), format="csr")

    def residual(q):
        grad = np.array([slab.energy_gradient(x)[1] for x in q])
        return grad + (spring @ q.ravel()).reshape(q.shape)

    q = seed.copy()
    for iteration in range(100):
        grad = residual(q)
        norm = np.linalg.norm(grad)
        print(temperature, iteration, abs(grad).max(), flush=True)
        if abs(grad).max() * np.sqrt(MASS) < tolerance:
            break
        matrix = spring + block_diag([slab.hessian(x) for x in q], format="csr")
        step = spsolve(matrix, -grad.ravel()).reshape(q.shape)
        scale = min(1.0, 0.3 / max(np.linalg.norm(step, axis=1)))
        while scale > 1e-7:
            trial = q + scale * step
            if np.linalg.norm(residual(trial)) < norm:
                q = trial
                break
            scale /= 2
        else:
            raise RuntimeError("ring Newton step failed to reduce the residual")
    else:
        raise RuntimeError("ring saddle did not converge")

    ring = np.concatenate((q, q[-2:0:-1]))
    if np.linalg.norm(q[0] - q[-1]) < 1e-3:
        raise RuntimeError("ring collapsed to a classical stationary point")
    energies, gradients = zip(*(slab.energy_gradient(x) for x in ring))
    full_grad = np.array(gradients) + kappa * (2*ring - np.roll(ring, 1, 0) - np.roll(ring, -1, 0))
    residual_max = np.abs(full_grad).max() * np.sqrt(MASS)
    if residual_max >= tolerance:
        raise RuntimeError("unfolded ring fails the Cartesian residual tolerance")
    blocks = np.array([slab.hessian(x) for x in q])
    hessian = block_diag(np.concatenate((blocks, blocks[-2:0:-1]))).toarray()
    laplace = 2*np.eye(beads) - np.roll(np.eye(beads), 1, 0) - np.roll(np.eye(beads), -1, 0)
    hessian += kappa * np.kron(laplace, np.eye(6))
    eigen, modes = eigh(hessian)
    tangent = (np.roll(ring, -1, 0) - np.roll(ring, 1, 0)).ravel()
    overlap = np.abs(modes.T @ tangent) / np.linalg.norm(tangent)
    gauge = int(np.argmax(overlap))
    kept = np.delete(eigen, gauge)
    if overlap[gauge] < 0.9 or np.count_nonzero(kept < 0) != 1:
        raise RuntimeError("stationary ring lacks one unstable mode and a cyclic tangent")
    b = 1 / (beads * KB * temperature)
    length2 = np.square(ring - np.roll(ring, 1, 0)).sum()
    action = length2 / (2*b*HBAR**2) + b*sum(energies)
    log_flux = (0.5 * np.log(length2/(2*np.pi*b*HBAR**2)) - (1+len(kept))*np.log(b*HBAR)
                - 0.5*np.log(np.abs(kept)).sum() - action)
    return q, dict(temperature_K=temperature, beads=beads, residual_eV_A=residual_max,
                   cyclic_overlap=overlap[gauge], cyclic_eigenvalue=eigen[gauge],
                   unstable_eigenvalue=kept.min(), log_k_times_Zr=log_flux,
                   action_over_hbar=action, surface_calls=slab.calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("geometry", type=Path, help="CON file defining the frozen substrate")
    parser.add_argument("mep", type=Path, help="mep.csv from the Cu/H2 example")
    parser.add_argument("output", type=Path)
    parser.add_argument("temperatures", type=float, nargs="+", help="descending temperatures in kelvin")
    parser.add_argument("--beads", type=int, default=150)
    parser.add_argument("--amplitude", type=float, default=0.20, help="initial normal-mode amplitude in mass-weighted coordinates")
    args = parser.parse_args()
    if args.beads < 4 or args.beads % 2 or min(args.temperatures) <= 0:
        parser.error("require an even bead count ≥ 4 and positive temperatures")
    slab = Slab(args.library, args.geometry)
    data = np.loadtxt(args.mep, skiprows=1, delimiter=",")
    guess = data[np.argmax(data[:,1]), 2:] * np.sqrt(MASS)
    saddle = root(lambda q: slab.energy_gradient(q)[1], guess, jac=slab.hessian, tol=1e-10)
    if np.max(np.abs(slab.energy_gradient(saddle.x)[1])) > 1e-7:
        raise RuntimeError("classical saddle did not converge")
    values, vectors = eigh(slab.hessian(saddle.x))
    if np.count_nonzero(values < 0) != 1:
        raise RuntimeError("classical seed has the wrong Hessian index")
    crossover = HBAR * np.sqrt(-values[0]) / (2*np.pi*KB)
    print("classical crossover (K)", crossover, flush=True)
    phase = np.linspace(0, np.pi, 1 + args.beads//2)
    q = saddle.x + args.amplitude * np.outer(np.cos(phase), vectors[:,0])
    args.output.mkdir(parents=True, exist_ok=True)
    for temperature in args.temperatures:
        q, result = solve_ring(slab, q, temperature, 1e-8)
        prefix = args.output / f"{temperature:g}K"
        np.save(str(prefix) + ".npy", q)
        Path(str(prefix) + ".json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
