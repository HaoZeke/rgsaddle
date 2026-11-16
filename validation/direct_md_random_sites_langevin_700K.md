# Random-vacancy survival, Mishin Cu

The vacancy is the lattice site nearest the cell center. The <100> dumbbell host is drawn uniformly from the other sites of the 13³ cell, and the dumbbell axis uniformly from the three orientations. Under periodic boundaries the separation is uniform over the cell, which is the ensemble in section 2 of Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025). The fit is ln S(t) = -(k/V_model) t through the origin, sampled every 25 ps, with V_model the cell volume and S the Kaplan-Meier survival, in which an uncaptured trajectory is censored at the end of its window; with equal windows S = n(t)/N. Samples after S reaches zero are left out. V_iv/Ω = 4k / (α a D_i) with α = 2/3 and Ω = a³/4.

Table 1 of that paper divides its k by the D_i in its own Table 1 column, which it quotes from Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022): SIA mean-square displacements in MD. The paper's own trajectories are NPH for 30 ps, then NVE with velocity rescaling, at its lattice constant. Our k comes from the Langevin dynamics in the CSV, so the self-consistent volume divides it by the free-dumbbell diffusivity measured with those same dynamics, potential, and cell, in `direct_md_sia_d_p0.md`. Both conversions are listed. σ_k is the standard deviation over 400 bootstrap resamples of the trajectories. The bootstrap σ column carries only that; the total σ columns add the relative error of the D_i used in quadrature.

A trajectory is captured when every Wigner-Seitz site of the ideal lattice holds one atom at 4 consecutive samples, 1.00 ps; the capture time is the first of them. Sites move with the cell and the centroid drift is removed. The last two outcome columns count sites holding other than one atom at the start and at the end.

At 700 K, Table 1 D_i = 0.910 ± 0.012 Å²/ps; measured D_i = 2.025 ± 0.123 Å²/ps.

| t_max (ps) | captured | N | k (Å³/ps) | σ_k | V_iv/Ω, Table 1 D_i | bootstrap σ | total σ | V_iv/Ω, measured D_i | total σ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 22 | 48 | 229.68 | 52.95 | 414.07 | 95.46 | 95.62 | 186.11 | 44.37 |
| 600 | 33 | 48 | 217.07 | 39.72 | 391.32 | 71.60 | 71.79 | 175.89 | 33.91 |
| 900 | 42 | 48 | 236.04 | 40.41 | 425.53 | 72.85 | 73.06 | 191.27 | 34.74 |
| 1200 | 47 | 48 | 291.83 | 36.15 | 526.11 | 65.17 | 65.54 | 236.48 | 32.63 |
| 1500 | 48 | 48 | 298.58 | 36.35 | 538.27 | 65.53 | 65.91 | 241.94 | 32.92 |

At the longest window, 1500 ps, k = 298.58 ± 36.35 Å³/ps. With the Table 1 D_i, V_iv = 538.27 ± 65.53 Ω (bootstrap) or ± 65.91 Ω (with D_i). With the measured D_i, V_iv = 241.94 ± 32.92 Ω.
The constant-hazard maximum likelihood over the same 1500 ps, with every uncaptured trajectory censored at the end of its window, is k = 258.39 Å³/ps from 48 captures in 48 trajectories, 95 % interval 194.72 to 342.87 from ln k ± 1.96/√captures. V_iv is 465.82 Ω (95 % 351.04 to 618.13) with the Table 1 D_i and 209.38 Ω (95 % 157.78 to 277.84) with the measured D_i; these intervals leave out the D_i error.
Table 1 at this temperature, 8788 atoms, is 203 ± 10 Ω (k-method) and 210 ± 21 Ω (mean-time method). The k-method value with the Table 1 D_i and the paper's lattice constant 3.655 Å implies k = 112.53 ± 5.74 Å³/ps. Our longest-window k over that is 2.653 ± 0.350. The Table 1-D_i volume minus the k-method value is 335.27 Ω; the measured-D_i volume minus it is 38.94 Ω.

Table 1, 8788 atoms, k-method then mean-time method: 300 K 254 ± 21 and 286 ± 28; 500 K 185 ± 23 and 161 ± 15; 700 K 203 ± 10 and 210 ± 21. All use the Ref. [12] D_i of Table 1: 0.143, 0.526, and 0.910 Å²/ps.
