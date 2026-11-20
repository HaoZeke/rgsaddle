#!/usr/bin/env python3
"""Write minimal Quantum ESPRESSO pw.x inputs for the Ta/H PW91 reconstruction.

Geometry matches validation/lineb/tantalum/generate_inputs.py (paper §III B cell).
Pseudopotentials: open PW91 Vanderbilt UPF (see PROVENANCE.toml).

Run from the repository root:

    uv run --no-project python validation/lineb/tantalum/open_qe/generate_qe_inputs.py

The generated inputs use relative paths to the downloaded pseudopotentials.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
OUT = ROOT / "inputs"
PSEUDO = ROOT / "pseudo"

# Import sibling generator without installing a package.
_spec = importlib.util.spec_from_file_location(
    "ta_generate_inputs",
    PARENT / "generate_inputs.py",
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["ta_generate_inputs"] = _mod
_spec.loader.exec_module(_mod)

A0 = _mod.A0
NX = NY = NZ = 2
KMESH = _mod.KMESH
TD_A_CONV = _mod.TD_A_CONV
TD_B_CONV = _mod.TD_B_CONV

# Paper ENCUT 250 eV → Ry; USPP density factor 8.
ECUTWFC_PAPER_RY = 18.4
ECUTRHO_PAPER_RY = 147.2
# Soft recommendation from H.pw91-van_ak info (20–40 Ry, ~25 typical).
ECUTWFC_REC_RY = 25.0
ECUTRHO_REC_RY = 200.0


def _geom():
    cell = A0 * NX
    ta = _mod.bcc_ta_fractional(NX, NY, NZ)
    td_a = _mod.conv_to_super(TD_A_CONV, NX, NY, NZ)
    td_b = _mod.conv_to_super(TD_B_CONV, NX, NY, NZ)
    free = set(_mod.five_ta_around_jump(ta, td_a, td_b, cell))
    return cell, ta, td_a, td_b, free


def _control(prefix: str, calculation: str) -> str:
    # Inputs live at inputs/<case>/<profile>/; UPF files at open_qe/pseudo/.
    return "\n".join(
        [
            "&CONTROL",
            f"  calculation = '{calculation}'",
            f"  prefix      = '{prefix}'",
            "  outdir      = './out'",
            "  pseudo_dir  = '../../../pseudo'",
            "  tprnfor     = .true.",
            "  tstress     = .true.",
            "/",
        ]
    )


def _system(nat: int, ntyp: int, ecutwfc: float, ecutrho: float) -> str:
    return "\n".join(
        [
            "&SYSTEM",
            "  ibrav       = 0",
            f"  nat         = {nat}",
            f"  ntyp        = {ntyp}",
            f"  ecutwfc     = {ecutwfc}",
            f"  ecutrho     = {ecutrho}",
            "  occupations = 'smearing'",
            "  smearing    = 'mv'",
            "  degauss     = 0.02",
            "  input_dft   = 'PW91'",
            "/",
        ]
    )


def _electrons() -> str:
    return "\n".join(
        [
            "&ELECTRONS",
            "  conv_thr    = 1.0d-8",
            "  mixing_beta = 0.3",
            "/",
        ]
    )


def _ions() -> str:
    return "\n".join(
        [
            "&IONS",
            "  ion_dynamics = 'bfgs'",
            "/",
        ]
    )


def _species() -> str:
    return "\n".join(
        [
            "ATOMIC_SPECIES",
            "  Ta  180.94788  Ta.pw91-nsp-van.UPF",
            "  H     1.00794  H.pw91-van_ak.UPF",
        ]
    )


def _cell(cell_A: float) -> str:
    return "\n".join(
        [
            "CELL_PARAMETERS angstrom",
            f"  {cell_A:.8f}  0.00000000  0.00000000",
            f"  0.00000000  {cell_A:.8f}  0.00000000",
            f"  0.00000000  0.00000000  {cell_A:.8f}",
        ]
    )


def _positions(
    ta: list[tuple[float, float, float]],
    h_frac: tuple[float, float, float],
    free_ta: set[int] | None,
) -> str:
    lines = ["ATOMIC_POSITIONS crystal"]
    for i, (x, y, z) in enumerate(ta):
        if free_ta is None:
            lines.append(f"  Ta  {x:.8f}  {y:.8f}  {z:.8f}")
        else:
            m = "1 1 1" if i in free_ta else "0 0 0"
            lines.append(f"  Ta  {x:.8f}  {y:.8f}  {z:.8f}  {m}")
    hx, hy, hz = h_frac
    if free_ta is None:
        lines.append(f"  H   {hx:.8f}  {hy:.8f}  {hz:.8f}")
    else:
        lines.append(f"  H   {hx:.8f}  {hy:.8f}  {hz:.8f}  1 1 1")
    return "\n".join(lines)


def _kpoints() -> str:
    nx, ny, nz = KMESH
    return "\n".join(
        [
            "K_POINTS automatic",
            f"  {nx} {ny} {nz}  0 0 0",
        ]
    )


def write_pw(
    path: Path,
    *,
    prefix: str,
    calculation: str,
    ta,
    h_frac,
    cell_A: float,
    free_ta: set[int] | None,
    ecutwfc: float,
    ecutrho: float,
    with_ions: bool,
) -> None:
    parts = [
        f"! RECONSTRUCTED open QE backend — Ásgeirsson JCP 148 102334 §III B",
        f"! PW91 USPP (not VASP PAW). prefix={prefix}  ecutwfc={ecutwfc} Ry",
        _control(prefix, calculation),
        _system(len(ta) + 1, 2, ecutwfc, ecutrho),
        _electrons(),
    ]
    if with_ions:
        parts.append(_ions())
    parts.extend(
        [
            _species(),
            _positions(ta, h_frac, free_ta),
            _cell(cell_A),
            _kpoints(),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    cell, ta, td_a, td_b, free = _geom()
    assert len(ta) == 16

    cases = [
        ("Td_initial", "scf", td_a, None, False),
        ("midpoint", "scf", tuple((a+b)/2 for a,b in zip(td_a,td_b)), None, False),
        ("midpoint_frozen5", "relax", tuple((a+b)/2 for a,b in zip(td_a,td_b)), free, True),
        ("Td_final", "scf", td_b, None, False),
        ("Td_initial_frozen5", "relax", td_a, free, True),
        ("Td_final_frozen5", "relax", td_b, free, True),
    ]

    for name, calc, hsite, free_ta, ions in cases:
        for label, ecutwfc, ecutrho in (
            ("paper_matched", ECUTWFC_PAPER_RY, ECUTRHO_PAPER_RY),
            ("pseudo_recommended", ECUTWFC_REC_RY, ECUTRHO_REC_RY),
        ):
            case = OUT / name / label
            infile = "pw.scf.in" if calc == "scf" else "pw.relax.in"
            write_pw(
                case / infile,
                prefix=f"ta_h_{name}_{label}",
                calculation=calc,
                ta=ta,
                h_frac=hsite,
                cell_A=cell,
                free_ta=free_ta,
                ecutwfc=ecutwfc,
                ecutrho=ecutrho,
                with_ions=ions,
            )
    print(f"wrote QE inputs under {OUT}")
    print(f"  profiles: paper_matched ({ECUTWFC_PAPER_RY} Ry), "
          f"pseudo_recommended ({ECUTWFC_REC_RY} Ry)")
    print(f"  free Ta indices: {sorted(free)}")


if __name__ == "__main__":
    main()
