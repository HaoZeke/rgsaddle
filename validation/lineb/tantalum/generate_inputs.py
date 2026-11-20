#!/usr/bin/env python3
"""Reconstruct VASP inputs for H diffusion in BCC Ta (§III B of JCP 148, 102334).

Run from the repository root:

    uv run --no-project python validation/lineb/tantalum/generate_inputs.py

Writes under validation/lineb/tantalum/reconstructed/. All geometry and tags
are rebuilt from the published text; they are not author-archived originals.
See README.md for PW91 vs PBE, POTCAR licensing, and quantitative targets.

Self-checks (no VASP required): Td–Td spacing, five mobile Ta neighbours,
and the D(T) = 9.13e-17 k(T) prefactor.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reconstructed"
TARGETS = ROOT / "paper_targets.toml"

# Conventional BCC lattice constant after volume relaxation (§III B).
A0 = 3.32  # Å
NX, NY, NZ = 2, 2, 2
ENCUT = 250  # eV
KMESH = (3, 3, 3)

# Adjacent Td sites in conventional fractional coordinates.
# Distance = a0 * sqrt(2) / 4 = 1.174… Å ≈ 1.17 Å (paper).
TD_A_CONV = (0.50, 0.25, 0.00)
TD_B_CONV = (0.25, 0.50, 0.00)


def load_targets() -> dict:
    with TARGETS.open("rb") as fh:
        return tomllib.load(fh)


def bcc_ta_fractional(nx: int, ny: int, nz: int) -> list[tuple[float, float, float]]:
    """2 atoms/cell conventional BCC, mapped into an nx×ny×nz supercell."""
    atoms: list[tuple[float, float, float]] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for basis in ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)):
                    atoms.append(
                        (
                            (i + basis[0]) / nx,
                            (j + basis[1]) / ny,
                            (k + basis[2]) / nz,
                        )
                    )
    return atoms


def conv_to_super(
    frac_conv: tuple[float, float, float],
    nx: int,
    ny: int,
    nz: int,
) -> tuple[float, float, float]:
    return (frac_conv[0] / nx, frac_conv[1] / ny, frac_conv[2] / nz)


def cart(
    frac: tuple[float, float, float],
    cell: float,
) -> tuple[float, float, float]:
    return (frac[0] * cell, frac[1] * cell, frac[2] * cell)


def mic_delta(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    out = []
    for x, y in zip(a, b, strict=True):
        d = x - y
        d -= round(d)
        out.append(d)
    return (out[0], out[1], out[2])


def dist_frac(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    cell: float,
) -> float:
    dx, dy, dz = mic_delta(a, b)
    return cell * math.sqrt(dx * dx + dy * dy + dz * dz)


def five_ta_around_jump(
    ta: list[tuple[float, float, float]],
    td_a: tuple[float, float, float],
    td_b: tuple[float, float, float],
    cell: float,
) -> list[int]:
    """Indices of the five Ta surrounding the two Td sites (§III B)."""

    def nearest4(site: tuple[float, float, float]) -> list[int]:
        ranked = sorted(
            range(len(ta)),
            key=lambda i: dist_frac(ta[i], site, cell),
        )
        return ranked[:4]

    union = sorted(set(nearest4(td_a)) | set(nearest4(td_b)))
    if len(union) != 5:
        raise RuntimeError(f"expected 5 surrounding Ta, found {len(union)}: {union}")
    return union


def write_poscar(
    path: Path,
    *,
    comment: str,
    cell: float,
    ta: list[tuple[float, float, float]],
    h_frac: tuple[float, float, float],
    free_ta: set[int] | None,
) -> None:
    selective = free_ta is not None
    lines = [
        comment,
        "1.0",
        f"  {cell:.8f}  0.00000000  0.00000000",
        f"  0.00000000  {cell:.8f}  0.00000000",
        f"  0.00000000  0.00000000  {cell:.8f}",
        "Ta H",
        f"  {len(ta)}  1",
    ]
    if selective:
        lines.append("Selective dynamics")
    lines.append("Direct")
    for i, (x, y, z) in enumerate(ta):
        if selective:
            flag = "T T T" if i in free_ta else "F F F"
            lines.append(f"  {x:.8f}  {y:.8f}  {z:.8f}  {flag}")
        else:
            lines.append(f"  {x:.8f}  {y:.8f}  {z:.8f}")
    hx, hy, hz = h_frac
    if selective:
        lines.append(f"  {hx:.8f}  {hy:.8f}  {hz:.8f}  T T T")
    else:
        lines.append(f"  {hx:.8f}  {hy:.8f}  {hz:.8f}")
    path.write_text("\n".join(lines) + "\n")


def write_kpoints(path: Path) -> None:
    nx, ny, nz = KMESH
    path.write_text(
        "\n".join(
            [
                "Automatic mesh",
                "0",
                "Monkhorst-Pack",
                f"  {nx}  {ny}  {nz}",
                "  0  0  0",
                "",
            ]
        )
    )


def write_incar(path: Path, *, kind: str) -> None:
    """Reconstructed INCAR. Paper-fixed: PW91 (GGA=91), ENCUT=250, PAW implied by POTCAR.

    Smearing and convergence tags are not stated in §III B; they are marked
    RECONSTRUCTED defaults for a metallic Ta host.
    """
    header = [
        f"SYSTEM = H in Ta BCC — {kind} (RECONSTRUCTED from JCP 148 102334 §III B)",
        "GGA    = 91        # PW91 as published; NOT PE (PBE)",
        f"ENCUT  = {ENCUT}",
        "PREC   = Accurate",
        "LREAL  = Auto",
        "ISMEAR = 1         # RECONSTRUCTED (metals); not stated in paper",
        "SIGMA  = 0.2       # RECONSTRUCTED",
        "EDIFF  = 1E-6      # RECONSTRUCTED",
        "EDIFFG = -0.01     # RECONSTRUCTED force threshold (eV/Å)",
        "NCORE  = 4         # RECONSTRUCTED site default; override locally",
    ]
    if kind == "volume_relax_host":
        body = header + [
            "NSW    = 80",
            "IBRION = 2",
            "POTIM  = 0.3",
            "ISIF   = 3         # volume + ions for empty host a0 check",
        ]
    elif kind == "neb_mep":
        body = header + [
            "NSW    = 500",
            "IBRION = 3         # RECONSTRUCTED; use VTST / rgsaddle externally if preferred",
            "POTIM  = 0",
            "ISIF   = 2",
            "ICHAIN = 0",
            "IMAGES = 5         # RECONSTRUCTED image count; paper uses NEB but omits N",
            "SPRING = -5",
            "LCLIMB = .TRUE.",
        ]
    elif kind == "frozen5":
        body = header + [
            "NSW    = 200",
            "IBRION = 2",
            "POTIM  = 0.3",
            "ISIF   = 2         # ions only; selective dynamics in POSCAR",
        ]
    else:
        body = header + [
            "NSW    = 200",
            "IBRION = 2",
            "POTIM  = 0.3",
            "ISIF   = 2",
        ]
    path.write_text("\n".join(body) + "\n")


def diffusion_prefactor(l_A: float, n: int, d: int) -> float:
    """D [cm^2/s] = prefactor * k [1/s], with l in Å."""
    l_cm = l_A * 1.0e-8
    return n * (l_cm**2) / (2.0 * d)


def self_check(targets: dict) -> None:
    cell = A0 * NX
    ta = bcc_ta_fractional(NX, NY, NZ)
    assert len(ta) == 16
    td_a = conv_to_super(TD_A_CONV, NX, NY, NZ)
    td_b = conv_to_super(TD_B_CONV, NX, NY, NZ)
    l = dist_frac(td_a, td_b, cell)
    paper_l = float(targets["sites"]["l_Td_Td_A"])
    if abs(l - paper_l) > 0.01:
        raise SystemExit(f"Td–Td distance {l:.4f} Å != paper {paper_l} Å")
    free = five_ta_around_jump(ta, td_a, td_b, cell)
    assert len(free) == 5
    pref = diffusion_prefactor(
        float(targets["diffusion_normalization"]["l_A"]),
        int(targets["diffusion_normalization"]["n"]),
        int(targets["diffusion_normalization"]["d"]),
    )
    paper_pref = float(targets["diffusion_normalization"]["prefactor_cm2"])
    if abs(pref - paper_pref) / paper_pref > 0.005:
        raise SystemExit(f"D prefactor {pref:.6e} != paper {paper_pref:.6e}")
    xc = targets["functional"]["paper_xc"]
    if xc != "PW91":
        raise SystemExit(f"paper_targets.toml paper_xc must be PW91, got {xc}")
    print("self-check OK:")
    print(f"  a0={A0} Å, supercell={cell} Å, N_Ta={len(ta)}")
    print(f"  Td–Td l={l:.4f} Å (paper {paper_l} Å)")
    print(f"  free Ta indices={free}")
    print(f"  D/k prefactor={pref:.6e} cm^2 (paper {paper_pref:.6e})")
    print(f"  XC={xc} (not {targets['functional']['not_paper_xc']})")


def main() -> None:
    targets = load_targets()
    self_check(targets)
    OUT.mkdir(parents=True, exist_ok=True)

    cell = A0 * NX
    ta = bcc_ta_fractional(NX, NY, NZ)
    td_a = conv_to_super(TD_A_CONV, NX, NY, NZ)
    td_b = conv_to_super(TD_B_CONV, NX, NY, NZ)
    free = set(five_ta_around_jump(ta, td_a, td_b, cell))

    write_poscar(
        OUT / "POSCAR.Td_initial",
        comment="RECONSTRUCTED: H at Td A, 16 Ta BCC 2x2x2, a0=3.32 A (JCP 148 102334 III B)",
        cell=cell,
        ta=ta,
        h_frac=td_a,
        free_ta=None,
    )
    write_poscar(
        OUT / "POSCAR.Td_final",
        comment="RECONSTRUCTED: H at Td B (adjacent), 16 Ta BCC 2x2x2",
        cell=cell,
        ta=ta,
        h_frac=td_b,
        free_ta=None,
    )
    write_poscar(
        OUT / "POSCAR.Td_initial_frozen5",
        comment="RECONSTRUCTED: selective dynamics — H + 5 Ta around jump free",
        cell=cell,
        ta=ta,
        h_frac=td_a,
        free_ta=free,
    )
    write_poscar(
        OUT / "POSCAR.Td_final_frozen5",
        comment="RECONSTRUCTED: selective dynamics — H + 5 Ta around jump free",
        cell=cell,
        ta=ta,
        h_frac=td_b,
        free_ta=free,
    )

    write_kpoints(OUT / "KPOINTS")
    write_incar(OUT / "INCAR.volume_relax_host", kind="volume_relax_host")
    write_incar(OUT / "INCAR.neb_mep", kind="neb_mep")
    write_incar(OUT / "INCAR.frozen5", kind="frozen5")

    sites = OUT / "sites.json"
    import json

    sites.write_text(
        json.dumps(
            {
                "status": "reconstructed",
                "doi": "10.1063/1.5007180",
                "a0_A": A0,
                "supercell_A": cell,
                "Td_A_frac_supercell": td_a,
                "Td_B_frac_supercell": td_b,
                "Td_A_cart_A": cart(td_a, cell),
                "Td_B_cart_A": cart(td_b, cell),
                "mobile_ta_indices_0based": sorted(free),
                "kpoints": list(KMESH),
                "encut_eV": ENCUT,
                "xc": "PW91",
                "targets": {
                    "Ea_eV": targets["barrier"]["Ea_eV"],
                    "Ea_frozen5_delta_eV": targets["barrier"]["Ea_frozen5_delta_eV"],
                    "Tc_K": targets["barrier"]["Tc_K"],
                    "D_over_k_cm2": targets["diffusion_normalization"]["prefactor_cm2"],
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote reconstructed inputs under {OUT}")


if __name__ == "__main__":
    main()
