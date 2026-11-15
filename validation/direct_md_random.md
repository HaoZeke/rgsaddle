# Random-vacancy survival, Mishin Cu

The vacancy is the lattice site nearest the cell center. The <100> dumbbell host is drawn uniformly from the other sites of the 13³ cell, and the dumbbell axis uniformly from the three orientations. Under periodic boundaries the separation is uniform over the cell, which is the ensemble in section 2 of Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025). The fit is ln S(t) = -(k/V_model) t through the origin, sampled every 25 ps, with V_model the cell volume and S the Kaplan-Meier survival, in which an uncaptured trajectory is censored at the end of its window; with equal windows S = n(t)/N. Samples after S reaches zero are left out. V_iv/Ω = 4k / (α a D_i) with α = 2/3 and Ω = a³/4.

Table 1 of that paper divides its k by the D_i in its own Table 1 column, which it quotes from Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022): SIA mean-square displacements in MD. The paper's own trajectories are NPH for 30 ps, then NVE with velocity rescaling, at its lattice constant. Our k comes from the Langevin dynamics in the CSV, so the self-consistent volume divides it by the free-dumbbell diffusivity measured with those same dynamics, potential, and cell, in `direct_md_sia_d_long.md`. Both conversions are listed. σ_k is the standard deviation over 400 bootstrap resamples of the trajectories. The bootstrap σ column carries only that; the total σ columns add the relative error of the D_i used in quadrature.

At 500 K, Table 1 D_i = 0.526 ± 0.006 Å²/ps; measured D_i = 1.075 ± 0.040 Å²/ps.

| t_max (ps) | captured | N | k (Å³/ps) | σ_k | V_iv/Ω, Table 1 D_i | bootstrap σ | total σ | V_iv/Ω, measured D_i | total σ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 30 | 192 | 44.03 | 9.45 | 138.03 | 29.61 | 29.65 | 67.53 | 14.70 |
| 600 | 60 | 192 | 58.31 | 8.36 | 182.79 | 26.22 | 26.30 | 89.43 | 13.25 |
| 900 | 99 | 192 | 71.90 | 7.96 | 225.41 | 24.95 | 25.08 | 110.29 | 12.87 |
| 1200 | 127 | 192 | 84.97 | 8.12 | 266.38 | 25.47 | 25.65 | 130.33 | 13.36 |
| 1500 | 149 | 192 | 92.33 | 8.07 | 289.44 | 25.29 | 25.51 | 141.62 | 13.43 |

At the longest window, 1500 ps, k = 92.33 ± 8.07 Å³/ps. With the Table 1 D_i, V_iv = 289.44 ± 25.29 Ω (bootstrap) or ± 25.51 Ω (with D_i). With the measured D_i, V_iv = 141.62 ± 13.43 Ω.
The constant-hazard maximum likelihood over the same 1500 ps, with every uncaptured trajectory censored at the end of its window, is k = 91.58 Å³/ps from 149 captures in 192 trajectories, 95 % interval 78.00 to 107.54 from ln k ± 1.96/√captures. V_iv is 287.11 Ω (95 % 244.52 to 337.12) with the Table 1 D_i and 140.48 Ω (95 % 119.64 to 164.94) with the measured D_i; these intervals leave out the D_i error.
Table 1 at this temperature, 8788 atoms, is 185 ± 23 Ω (k-method) and 161 ± 15 Ω (mean-time method). The k-method value with the Table 1 D_i and the paper's lattice constant 3.643 Å implies k = 59.08 ± 7.38 Å³/ps. Our longest-window k over that is 1.563 ± 0.238. The Table 1-D_i volume minus the k-method value is 104.44 Ω; the measured-D_i volume minus it is -43.38 Ω.

Table 1, 8788 atoms, k-method then mean-time method: 300 K 254 ± 21 and 286 ± 28; 500 K 185 ± 23 and 161 ± 15; 700 K 203 ± 10 and 210 ± 21. All use the Ref. [12] D_i of Table 1: 0.143, 0.526, and 0.910 Å²/ps.
