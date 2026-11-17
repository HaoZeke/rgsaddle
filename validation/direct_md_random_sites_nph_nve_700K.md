# Random-vacancy survival, Mishin Cu

The vacancy is the lattice site nearest the cell center. The <100> dumbbell host is drawn uniformly from the other sites of the 13³ cell, and the dumbbell axis uniformly from the three orientations. Under periodic boundaries the separation is uniform over the cell, which is the ensemble in section 2 of Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025). The fit is ln S(t) = -(k/V_model) t through the origin, sampled every 25 ps, with V_model the cell volume and S the Kaplan-Meier survival, in which an uncaptured trajectory is censored at the end of its window; with equal windows S = n(t)/N. Samples after S reaches zero are left out. V_iv/Ω = 4k / (α a D_i) with α = 2/3 and Ω = a³/4.

Table 1 of that paper divides its k by the D_i in its own Table 1 column, which it quotes from Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022): SIA mean-square displacements in MD. The paper's own trajectories are NPH for 30 ps, then NVE with velocity rescaling, at its lattice constant. Our k comes from the CSV's own NPH-then-NVE dynamics: 30 ps of isotropic zero-pressure NPH (Andersen piston, period 2.0 ps), then NVE in the mean box of the second half of that stage, with velocities rescaled to the target temperature every 0.10 ps whenever it is off by more than 0.0 K. The capture clock starts at the Maxwellian start, NPH stage included, and the fit uses the mean production lattice constant 3.658694 Å of the trajectories that reached NVE. The self-consistent volume divides k by the free-dumbbell diffusivity measured with those same dynamics, potential, and cell, in `direct_md_sia_d_nph_nve.md`. Both conversions are listed. σ_k is the standard deviation over 400 bootstrap resamples of the trajectories. The bootstrap σ column carries only that; the total σ columns add the relative error of the D_i used in quadrature.

A trajectory is captured when every Wigner-Seitz site of the ideal lattice holds one atom at 4 consecutive samples, 1.00 ps; the capture time is the first of them. Sites move with the cell and the centroid drift is removed. The last two outcome columns count sites holding other than one atom at the start and at the end.

At 700 K, Table 1 D_i = 0.910 ± 0.012 Å²/ps; measured D_i = 2.072 ± 0.163 Å²/ps.

| t_max (ps) | captured | N | k (Å³/ps) | σ_k | V_iv/Ω, Table 1 D_i | bootstrap σ | total σ | V_iv/Ω, measured D_i | total σ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 39 | 96 | 197.14 | 32.87 | 355.28 | 59.24 | 59.43 | 156.05 | 28.77 |
| 600 | 63 | 96 | 193.62 | 25.84 | 348.93 | 46.57 | 46.80 | 153.26 | 23.75 |
| 900 | 78 | 96 | 196.97 | 23.79 | 354.97 | 42.88 | 43.14 | 155.92 | 22.48 |
| 1200 | 88 | 96 | 196.66 | 22.90 | 354.41 | 41.26 | 41.52 | 155.67 | 21.88 |
| 1500 | 90 | 96 | 198.97 | 24.52 | 358.56 | 44.19 | 44.44 | 157.50 | 23.03 |

At the longest window, 1500 ps, k = 198.97 ± 24.52 Å³/ps. With the Table 1 D_i, V_iv = 358.56 ± 44.19 Ω (bootstrap) or ± 44.44 Ω (with D_i). With the measured D_i, V_iv = 157.50 ± 23.03 Ω.
The constant-hazard maximum likelihood over the same 1500 ps, with every uncaptured trajectory censored at the end of its window, is k = 197.00 Å³/ps from 90 captures in 96 trajectories, 95 % interval 160.23 to 242.22 from ln k ± 1.96/√captures. V_iv is 355.03 Ω (95 % 288.76 to 436.50) with the Table 1 D_i and 155.94 Ω (95 % 126.84 to 191.73) with the measured D_i; these intervals leave out the D_i error.
Table 1 at this temperature, 8788 atoms, is 203 ± 10 Ω (k-method) and 210 ± 21 Ω (mean-time method). The k-method value with the Table 1 D_i and the paper's lattice constant 3.655 Å implies k = 112.53 ± 5.74 Å³/ps. Our longest-window k over that is 1.768 ± 0.236. The Table 1-D_i volume minus the k-method value is 155.56 Ω; the measured-D_i volume minus it is -45.50 Ω.

Table 1, 8788 atoms, k-method then mean-time method: 300 K 254 ± 21 and 286 ± 28; 500 K 185 ± 23 and 161 ± 15; 700 K 203 ± 10 and 210 ± 21. All use the Ref. [12] D_i of Table 1: 0.143, 0.526, and 0.910 Å²/ps.
