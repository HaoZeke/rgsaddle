# Hydrogen diffusion in tantalum

Section III B follows a hydrogen atom between adjacent tetrahedral interstitial sites
in a periodic cell with 16 tantalum atoms. The reconstructed endpoints use a
2 × 2 × 2 bcc supercell with lattice constant 3.32 Å. The jump length is 1.17 Å.
The electronic structure calculation uses VASP, PW91, PAW, a 250 eV cutoff and
a 3 × 3 × 3 Monkhorst–Pack mesh. [Paper](https://doi.org/10.1063/1.5007180)

Run `python generate_inputs.py` in this directory to generate the endpoint
structures, selective-dynamics masks, INCAR templates and k-point mesh under
`reconstructed/`. Supply the Ta and H PAW datasets in POSCAR species order
from a licensed installation. The paper leaves the PAW dataset versions
unspecified. Record the actual TITEL values with the calculation.

The fully relaxed reference barrier is 0.20 eV. Allowing only the H atom and
the five neighbouring Ta atoms to move raises it by 0.004 eV. The harmonic
crossover temperature is 180 K. For the rate comparison, convert each
reactant-normalized single-hop rate to diffusivity using equation (29),
`D = 9.13e-17 * k`, with `k` in inverse seconds and `D` in cm²/s.
`python compare_targets.py` checks this conversion against the hop geometry.
The digitized HQTST diffusivities are in [the Figure 5 table](../figure5_hqtst.csv).

The Quantum ESPRESSO reconstruction uses PW91 with the Vanderbilt ultrasoft
datasets for Ta and H distributed by Quantum ESPRESSO. This changes the
ionic potentials relative to the paper's PAW calculation. Under `open_qe/`,
run `python fetch_pseudopotentials.py` and `python generate_qe_inputs.py`.
From each generated case directory run `pw.x -in pw.scf.in` for a fixed
geometry or `pw.x -in pw.relax.in` for an ionic relaxation. The endpoint
and midpoint cases use the same reciprocal mesh and cutoff. A midpoint
energy gives a trial barrier. Establishing a saddle also requires a
stationary geometry and one unstable vibrational mode along the hop.
