# Random-vacancy survival, Mishin Cu

The vacancy is the lattice site nearest the cell center. The <100> dumbbell host is drawn uniformly from the other sites of the 13³ cell, and the dumbbell axis uniformly from the three orientations. Under periodic boundaries the separation is uniform over the cell, which is the ensemble in section 2 of Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025). The fit is ln S(t) = -(k/V_model) t through the origin, sampled every 25 ps, with V_model the cell volume and S the Kaplan-Meier survival, in which an uncaptured trajectory is censored at the end of its window; with equal windows S = n(t)/N. Samples after S reaches zero are left out. V_iv/Ω = 4k / (α a D_i) with α = 2/3 and Ω = a³/4.

Table 1 of that paper divides its k by the D_i in its own Table 1 column, which it quotes from Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022): SIA mean-square displacements in MD. The paper's own trajectories are NPH for 30 ps, then NVE with velocity rescaling, at its lattice constant. Our k comes from the Langevin dynamics in the CSV, so the self-consistent volume divides it by the free-dumbbell diffusivity measured with those same dynamics, potential, and cell, in `direct_md_sia_d_p0.md`. Both conversions are listed. σ_k is the standard deviation over 400 bootstrap resamples of the trajectories. The bootstrap σ column carries only that; the total σ columns add the relative error of the D_i used in quadrature.

A trajectory is captured when every Wigner-Seitz site of the ideal lattice holds one atom at 4 consecutive samples, 1.00 ps; the capture time is the first of them. Sites move with the cell and the centroid drift is removed. The last two outcome columns count sites holding other than one atom at the start and at the end.

At 300 K, Table 1 D_i = 0.143 ± 0.006 Å²/ps; measured D_i = 0.272 ± 0.016 Å²/ps.

| t_max (ps) | captured | N | k (Å³/ps) | σ_k | V_iv/Ω, Table 1 D_i | bootstrap σ | total σ | V_iv/Ω, measured D_i | total σ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 14 | 96 | 69.32 | 18.97 | 800.82 | 219.14 | 221.70 | 421.47 | 117.89 |
| 600 | 22 | 96 | 50.45 | 11.47 | 582.87 | 132.54 | 134.78 | 306.76 | 71.98 |
| 900 | 24 | 96 | 42.33 | 8.86 | 489.00 | 102.31 | 104.35 | 257.36 | 55.87 |
| 1200 | 36 | 96 | 41.16 | 7.30 | 475.52 | 84.36 | 86.69 | 250.27 | 46.70 |
| 1500 | 47 | 96 | 43.01 | 6.70 | 496.87 | 77.38 | 80.14 | 261.51 | 43.45 |

At the longest window, 1500 ps, k = 43.01 ± 6.70 Å³/ps. With the Table 1 D_i, V_iv = 496.87 ± 77.38 Ω (bootstrap) or ± 80.14 Ω (with D_i). With the measured D_i, V_iv = 261.51 ± 43.45 Ω.
The constant-hazard maximum likelihood over the same 1500 ps, with every uncaptured trajectory censored at the end of its window, is k = 46.41 Å³/ps from 47 captures in 96 trajectories, 95 % interval 34.87 to 61.77 from ln k ± 1.96/√captures. V_iv is 536.16 Ω (95 % 402.84 to 713.61) with the Table 1 D_i and 282.18 Ω (95 % 212.02 to 375.57) with the measured D_i; these intervals leave out the D_i error.
Table 1 at this temperature, 8788 atoms, is 254 ± 21 Ω (k-method) and 286 ± 28 Ω (mean-time method). The k-method value with the Table 1 D_i and the paper's lattice constant 3.631 Å implies k = 21.98 ± 2.04 Å³/ps. Our longest-window k over that is 1.957 ± 0.355. The Table 1-D_i volume minus the k-method value is 242.87 Ω; the measured-D_i volume minus it is 7.51 Ω.

Table 1, 8788 atoms, k-method then mean-time method: 300 K 254 ± 21 and 286 ± 28; 500 K 185 ± 23 and 161 ± 15; 700 K 203 ± 10 and 210 ± 21. All use the Ref. [12] D_i of Table 1: 0.143, 0.526, and 0.910 Å²/ps.
