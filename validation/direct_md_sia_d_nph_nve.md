# Free dumbbell diffusivity, Mishin Cu, NPH then NVE

One <100> dumbbell in the perfect 13×13×13 cell (8789 atoms) on the Mishin EAM1 of `tests/mishin_cu.rs`, started at the lattice constant of the V_iv runs. The dynamics follow the protocol Mwungeli, Liu, and Demkowicz describe: Maxwellian start, 30 ps of isotropic zero-pressure NPH (Andersen piston, period 2.0 ps), then NVE in the mean box of the second half of that stage. Both stages rescale velocities to the target temperature every 0.10 ps whenever it is off by more than 0.0 K. The timestep is the column `dt_fs`; the lattice column is the production lattice constant after the NPH stage. The dumbbell is the one doubly occupied Wigner-Seitz cell of the ideal lattice, found every 0.1 ps after the centre-of-mass drift of the cell is removed. A sample with no single doubly occupied cell is skipped and counted in `unresolved_samples`. Each row is the dumbbell-site displacement over one production block of `tau_ps`.

D for one trajectory is the mean of |Δr|² over its blocks divided by 6τ. The quoted D is the mean over trajectories and σ is the standard error across trajectories. D(2τ) joins consecutive blocks; it matches D(τ) when τ is past the correlated part of the walk. Site changes per ps count samples at which the doubly occupied cell moved, so back-and-forth jumps inside 0.1 ps are not counted.

| T (K) | a (Å) | seeds | blocks | τ (ps) | D (Å²/ps) | σ | σ/D | D(2τ) | σ(2τ) | site changes/ps | unresolved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 3.632694 | 8 | 320 | 40.0 | 0.3137 | 0.0160 | 5.1% | 0.3050 | 0.0218 | 0.384 | 0.00% |
| 500 | 3.645172 | 8 | 320 | 20.0 | 1.1448 | 0.0638 | 5.6% | 1.1332 | 0.1008 | 1.204 | 0.00% |
| 700 | 3.658751 | 8 | 320 | 10.0 | 2.0717 | 0.1631 | 7.9% | 2.1491 | 0.2133 | 2.043 | 0.00% |

## Cross-check

Table 1 of Mwungeli, Liu, and Demkowicz lists D_i from their Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022), which took D_i from the mean-square displacement of SIAs in MD. Table 1 divides each measured rate coefficient by that D_i: V_iv = k a² / (α D_i) with α = 2/3. The defect-pair column is the free-dumbbell MD diffusivity quoted by the defect-pair rate work. z is the difference over the two errors in quadrature.

| T (K) | this run | Table 1 [12] | this/Table 1 | z | defect-pair MD | this/defect-pair | z |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 0.314 ± 0.016 | 0.143 ± 0.006 | 2.194 | 10.00 | 0.278 ± 0.033 | 1.128 | 0.97 |
| 500 | 1.145 ± 0.064 | 0.526 ± 0.006 | 2.176 | 9.66 | 0.860 ± 0.071 | 1.331 | 2.98 |
| 700 | 2.072 ± 0.163 | 0.910 ± 0.012 | 2.277 | 7.10 | 1.886 ± 0.188 | 1.098 | 0.75 |

Raw per-block displacements are the CSV this note was written from.
