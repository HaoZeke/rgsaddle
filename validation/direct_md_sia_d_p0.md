# Free dumbbell diffusivity, Mishin Cu

One <100> dumbbell in the perfect 13×13×13 cell (8789 atoms) on the Mishin EAM1 of `tests/mishin_cu.rs`, at the zero-total-pressure lattice constant of `direct_md_lattice.csv`. The dynamics are the V_iv dynamics: Maxwellian start, 2 ps of Langevin at gamma = 1/ps, then gamma and dt from the columns `gamma_per_ps` and `dt_fs`. The dumbbell is the one doubly occupied Wigner-Seitz cell of the ideal lattice, found every 0.1 ps after the centre-of-mass drift of the cell is removed. A sample with no single doubly occupied cell is skipped and counted in `unresolved_samples`. Each row is the dumbbell-site displacement over one production block of `tau_ps`.

D for one trajectory is the mean of |Δr|² over its blocks divided by 6τ. The quoted D is the mean over trajectories and σ is the standard error across trajectories. D(2τ) joins consecutive blocks; it matches D(τ) when τ is past the correlated part of the walk. Site changes per ps count samples at which the doubly occupied cell moved, so back-and-forth jumps inside 0.1 ps are not counted.

| T (K) | a (Å) | seeds | blocks | τ (ps) | D (Å²/ps) | σ | σ/D | D(2τ) | σ(2τ) | site changes/ps | unresolved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 3.631851 | 8 | 320 | 40.0 | 0.2717 | 0.0157 | 5.8% | 0.2641 | 0.0252 | 0.340 | 0.00% |
| 500 | 3.644035 | 8 | 320 | 20.0 | 1.0590 | 0.0449 | 4.2% | 1.0597 | 0.0660 | 1.171 | 0.00% |
| 700 | 3.657362 | 8 | 320 | 10.0 | 2.0246 | 0.1230 | 6.1% | 2.0078 | 0.1688 | 1.931 | 0.00% |

## Cross-check

Table 1 of Mwungeli, Liu, and Demkowicz lists D_i from their Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022), which took D_i from the mean-square displacement of SIAs in MD. Table 1 divides each measured rate coefficient by that D_i: V_iv = k a² / (α D_i) with α = 2/3. The defect-pair column is the free-dumbbell MD diffusivity quoted by the defect-pair rate work. z is the difference over the two errors in quadrature.

| T (K) | this run | Table 1 [12] | this/Table 1 | z | defect-pair MD | this/defect-pair | z |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 0.272 ± 0.016 | 0.143 ± 0.006 | 1.900 | 7.65 | 0.278 ± 0.033 | 0.977 | -0.17 |
| 500 | 1.059 ± 0.045 | 0.526 ± 0.006 | 2.013 | 11.76 | 0.860 ± 0.071 | 1.231 | 2.37 |
| 700 | 2.025 ± 0.123 | 0.910 ± 0.012 | 2.225 | 9.02 | 1.886 ± 0.188 | 1.073 | 0.62 |

Raw per-block displacements are the CSV this note was written from.
