# Free dumbbell diffusivity, Mishin Cu

One <100> dumbbell in the perfect 13×13×13 cell (8789 atoms) on the Mishin EAM1 of `tests/mishin_cu.rs`, at the lattice constant of the V_iv runs. The dynamics are the V_iv dynamics: Maxwellian start, 2 ps of Langevin at gamma = 1/ps, then gamma and dt from the columns `gamma_per_ps` and `dt_fs`. The dumbbell is the one doubly occupied Wigner-Seitz cell of the ideal lattice, found every 0.1 ps after the centre-of-mass drift of the cell is removed. A sample with no single doubly occupied cell is skipped and counted in `unresolved_samples`. Each row is the dumbbell-site displacement over one production block of `tau_ps`.

D for one trajectory is the mean of |Δr|² over its blocks divided by 6τ. The quoted D is the mean over trajectories and σ is the standard error across trajectories. D(2τ) joins consecutive blocks; it matches D(τ) when τ is past the correlated part of the walk. Site changes per ps count samples at which the doubly occupied cell moved, so back-and-forth jumps inside 0.1 ps are not counted.

| T (K) | a (Å) | seeds | blocks | τ (ps) | D (Å²/ps) | σ | σ/D | D(2τ) | σ(2τ) | site changes/ps | unresolved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 3.628311 | 8 | 320 | 40.0 | 0.2591 | 0.0148 | 5.7% | 0.2332 | 0.0163 | 0.321 | 0.00% |
| 500 | 3.638666 | 8 | 320 | 20.0 | 1.1014 | 0.0422 | 3.8% | 1.1366 | 0.1239 | 1.135 | 0.00% |
| 700 | 3.649788 | 8 | 320 | 10.0 | 1.8306 | 0.0783 | 4.3% | 1.7664 | 0.0908 | 1.853 | 0.00% |

## Cross-check

Table 1 of Mwungeli, Liu, and Demkowicz lists D_i from their Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022), which took D_i from the mean-square displacement of SIAs in MD. Table 1 divides each measured rate coefficient by that D_i: V_iv = k a² / (α D_i) with α = 2/3. The defect-pair column is the free-dumbbell MD diffusivity quoted by the defect-pair rate work. z is the difference over the two errors in quadrature.

| T (K) | this run | Table 1 [12] | this/Table 1 | z | defect-pair MD | this/defect-pair | z |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 0.259 ± 0.015 | 0.143 ± 0.006 | 1.812 | 7.28 | 0.278 ± 0.033 | 0.932 | -0.52 |
| 500 | 1.101 ± 0.042 | 0.526 ± 0.006 | 2.094 | 13.51 | 0.860 ± 0.071 | 1.281 | 2.92 |
| 700 | 1.831 ± 0.078 | 0.910 ± 0.012 | 2.012 | 11.62 | 1.886 ± 0.188 | 0.971 | -0.27 |

Raw per-block displacements are the CSV this note was written from.
