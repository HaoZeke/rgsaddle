# Random-vacancy survival, Mishin Cu

The vacancy is the lattice site nearest the cell center. The <100> dumbbell host is drawn uniformly from the other sites of the 13³ cell, and the dumbbell axis uniformly from the three orientations. Under periodic boundaries the separation is uniform over the cell, which is the ensemble in section 2 of Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025). The fit is ln S(t) = -(k/V_model) t through the origin, sampled every 25 ps, with V_model the cell volume and S the Kaplan-Meier survival, in which an uncaptured trajectory is censored at the end of its window; with equal windows S = n(t)/N. Samples after S reaches zero are left out. V_iv/Ω = 4k / (α a D_i) with α = 2/3 and Ω = a³/4.

Table 1 of that paper divides its k by the D_i in its own Table 1 column, which it quotes from Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022): SIA mean-square displacements in MD. The paper's own trajectories are NPH for 30 ps, then NVE with velocity rescaling, at its lattice constant. Our k comes from the CSV's own NPH-then-NVE dynamics: 30 ps of isotropic zero-pressure NPH (Andersen piston, period 2.0 ps), then NVE in the mean box of the second half of that stage, with velocities rescaled to the target temperature every 0.10 ps whenever it is off by more than 0.0 K. The capture clock starts at the Maxwellian start, NPH stage included, and the fit uses the mean production lattice constant 3.645023 Å of the trajectories that reached NVE. The self-consistent volume divides k by the free-dumbbell diffusivity measured with those same dynamics, potential, and cell, in `direct_md_sia_d_nph_nve.md`. Both conversions are listed. σ_k is the standard deviation over 400 bootstrap resamples of the trajectories. The bootstrap σ column carries only that; the total σ columns add the relative error of the D_i used in quadrature.

A trajectory is captured when every Wigner-Seitz site of the ideal lattice holds one atom at 4 consecutive samples, 1.00 ps; the capture time is the first of them. Sites move with the cell and the centroid drift is removed. The last two outcome columns count sites holding other than one atom at the start and at the end.

At 500 K, Table 1 D_i = 0.526 ± 0.006 Å²/ps; measured D_i = 1.145 ± 0.064 Å²/ps.

| t_max (ps) | captured | N | k (Å³/ps) | σ_k | V_iv/Ω, Table 1 D_i | bootstrap σ | total σ | V_iv/Ω, measured D_i | total σ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 33 | 96 | 151.23 | 26.53 | 473.26 | 83.03 | 83.20 | 217.45 | 40.02 |
| 600 | 55 | 96 | 149.85 | 20.17 | 468.95 | 63.12 | 63.35 | 215.47 | 31.39 |
| 900 | 65 | 96 | 142.29 | 17.35 | 445.28 | 54.29 | 54.53 | 204.59 | 27.42 |
| 1200 | 75 | 96 | 136.80 | 15.39 | 428.11 | 48.16 | 48.40 | 196.70 | 24.69 |
| 1500 | 77 | 96 | 130.67 | 14.47 | 408.92 | 45.28 | 45.52 | 187.88 | 23.29 |

At the longest window, 1500 ps, k = 130.67 ± 14.47 Å³/ps. With the Table 1 D_i, V_iv = 408.92 ± 45.28 Ω (bootstrap) or ± 45.52 Ω (with D_i). With the measured D_i, V_iv = 187.88 ± 23.29 Ω.
The constant-hazard maximum likelihood over the same 1500 ps, with every uncaptured trajectory censored at the end of its window, is k = 129.77 Å³/ps from 77 captures in 96 trajectories, 95 % interval 103.80 to 162.25 from ln k ± 1.96/√captures. V_iv is 406.12 Ω (95 % 324.82 to 507.76) with the Table 1 D_i and 186.60 Ω (95 % 149.25 to 233.30) with the measured D_i; these intervals leave out the D_i error.
Table 1 at this temperature, 8788 atoms, is 185 ± 23 Ω (k-method) and 161 ± 15 Ω (mean-time method). The k-method value with the Table 1 D_i and the paper's lattice constant 3.643 Å implies k = 59.08 ± 7.38 Å³/ps. Our longest-window k over that is 2.212 ± 0.369. The Table 1-D_i volume minus the k-method value is 223.92 Ω; the measured-D_i volume minus it is 2.88 Ω.

Table 1, 8788 atoms, k-method then mean-time method: 300 K 254 ± 21 and 286 ± 28; 500 K 185 ± 23 and 161 ± 15; 700 K 203 ± 10 and 210 ± 21. All use the Ref. [12] D_i of Table 1: 0.143, 0.526, and 0.910 Å²/ps.
