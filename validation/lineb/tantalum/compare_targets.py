#!/usr/bin/env python3
"""Print §III B quantitative targets and verify the D/k normalisation.

Run: uv run --no-project python validation/lineb/tantalum/compare_targets.py

The geometric check uses the published hop length and multiplicity.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def prefactor(l_A: float, n: int, d: int) -> float:
    l_cm = l_A * 1.0e-8
    return n * (l_cm**2) / (2.0 * d)


def main() -> None:
    with (ROOT / "paper_targets.toml").open("rb") as fh:
        t = tomllib.load(fh)

    func = t["functional"]
    cell = t["cell"]
    sites = t["sites"]
    barrier = t["barrier"]
    diff = t["diffusion_normalization"]

    calc = prefactor(diff["l_A"], diff["n"], diff["d"])
    paper = diff["prefactor_cm2"]
    rel = abs(calc - paper) / paper

    print("Ásgeirsson et al., JCP 148, 102334 (2018) §III B — target sheet")
    print(f"  DOI: {t['provenance']['doi']}")
    print(f"  input_status: {t['provenance']['input_status']}")
    print()
    print("XC")
    print(f"  paper_xc          = {func['paper_xc']}  (GGA tag {func['paper_gga_tag']})")
    print(f"  not_paper_xc      = {func['not_paper_xc']}  ← do not use for a paper match")
    print(f"  ENCUT             = {func['encut_eV']} eV ({func['encut_Ry']} Ry)")
    print(f"  PAW               = {func['paw']}")
    print()
    print("Cell / BZ")
    print(f"  N_Ta + N_H        = {cell['n_ta']} + {cell['n_h']}")
    print(f"  a0 (relaxed)      = {cell['conventional_a0_A']} Å  (expt {cell['experimental_a0_A']} Å)")
    print(f"  supercell         = {cell['supercell_repeat']}")
    print(f"  k-mesh            = {cell['kpoints_monkhorst_pack']} Monkhorst–Pack")
    print()
    print("Jump")
    print(f"  sites             = {sites['jump']}")
    print(f"  l                 = {sites['l_Td_Td_A']} Å")
    print(f"  n_paths, d        = {sites['n_equivalent_paths']}, {sites['lattice_dimension']}")
    print()
    print("Barrier / Tc")
    print(f"  Ea                = {barrier['Ea_eV']} eV")
    print(f"  Ea (5 mobile Ta)  = Ea + {barrier['Ea_frozen5_delta_eV']} eV")
    print(f"  Tc                = {barrier['Tc_K']} K")
    print()
    print("Diffusion normalisation  D = prefactor · k")
    print(f"  paper prefactor      = {paper:.6e} cm^2")
    print(f"  recomputed        = {calc:.6e} cm^2")
    print(f"  relative error    = {rel:.3e}")
    if rel > 0.005:
        raise SystemExit("prefactor mismatch with paper 9.13e-17")
    # geometric cross-check: a0 * sqrt(2)/4
    l_geom = cell["conventional_a0_A"] * math.sqrt(2.0) / 4.0
    print(f"  a0·√2/4           = {l_geom:.4f} Å  (paper l={sites['l_Td_Td_A']} Å)")
    print()


if __name__ == "__main__":
    main()
