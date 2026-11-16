# Random-vacancy survival, Mishin Cu

The vacancy is the lattice site nearest the cell center. The <100> dumbbell host is drawn uniformly from the other sites of the 13³ cell, and the dumbbell axis uniformly from the three orientations. Under periodic boundaries the separation is uniform over the cell, which is the ensemble in section 2 of Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025). The fit is ln S(t) = -(k/V_model) t through the origin, sampled every 25 ps, with V_model the cell volume and S the Kaplan-Meier survival, in which an uncaptured trajectory is censored at the end of its window; with equal windows S = n(t)/N. Samples after S reaches zero are left out. V_iv/Ω = 4k / (α a D_i) with α = 2/3 and Ω = a³/4.

Table 1 of that paper divides its k by the D_i in its own Table 1 column, which it quotes from Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022): SIA mean-square displacements in MD. The paper's own trajectories are NPH for 30 ps, then NVE with velocity rescaling, at its lattice constant. Our k comes from the Langevin dynamics in the CSV, so the self-consistent volume divides it by the free-dumbbell diffusivity measured with those same dynamics, potential, and cell, in `direct_md_sia_d_p0.md`. Both conversions are listed. σ_k is the standard deviation over 400 bootstrap resamples of the trajectories. The bootstrap σ column carries only that; the total σ columns add the relative error of the D_i used in quadrature.

A trajectory is captured when every Wigner-Seitz site of the ideal lattice holds one atom at 4 consecutive samples, 1.00 ps; the capture time is the first of them. Sites move with the cell and the centroid drift is removed. The last two outcome columns count sites holding other than one atom at the start and at the end.

At 500 K, Table 1 D_i = 0.526 ± 0.006 Å²/ps; measured D_i = 1.059 ± 0.045 Å²/ps.

| t_max (ps) | captured | N | k (Å³/ps) | σ_k | V_iv/Ω, Table 1 D_i | bootstrap σ | total σ | V_iv/Ω, measured D_i | total σ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 25 | 96 | 115.63 | 25.08 | 361.96 | 78.51 | 78.62 | 179.78 | 39.73 |
| 600 | 47 | 96 | 113.19 | 18.83 | 354.33 | 58.95 | 59.09 | 175.99 | 30.22 |
| 900 | 65 | 96 | 126.47 | 17.26 | 395.90 | 54.02 | 54.21 | 196.64 | 28.10 |
| 1200 | 76 | 96 | 128.50 | 16.45 | 402.25 | 51.50 | 51.70 | 199.79 | 26.95 |
| 1500 | 82 | 96 | 133.29 | 16.42 | 417.23 | 51.41 | 51.63 | 207.23 | 27.00 |

At the longest window, 1500 ps, k = 133.29 ± 16.42 Å³/ps. With the Table 1 D_i, V_iv = 417.23 ± 51.41 Ω (bootstrap) or ± 51.63 Ω (with D_i). With the measured D_i, V_iv = 207.23 ± 27.00 Ω.
The constant-hazard maximum likelihood over the same 1500 ps, with every uncaptured trajectory censored at the end of its window, is k = 129.74 Å³/ps from 82 captures in 96 trajectories, 95 % interval 104.49 to 161.10 from ln k ± 1.96/√captures. V_iv is 406.13 Ω (95 % 327.09 to 504.28) with the Table 1 D_i and 201.72 Ω (95 % 162.46 to 250.46) with the measured D_i; these intervals leave out the D_i error.
Table 1 at this temperature, 8788 atoms, is 185 ± 23 Ω (k-method) and 161 ± 15 Ω (mean-time method). The k-method value with the Table 1 D_i and the paper's lattice constant 3.643 Å implies k = 59.08 ± 7.38 Å³/ps. Our longest-window k over that is 2.256 ± 0.396. The Table 1-D_i volume minus the k-method value is 232.23 Ω; the measured-D_i volume minus it is 22.23 Ω.

Table 1, 8788 atoms, k-method then mean-time method: 300 K 254 ± 21 and 286 ± 28; 500 K 185 ± 23 and 161 ± 15; 700 K 203 ± 10 and 210 ± 21. All use the Ref. [12] D_i of Table 1: 0.143, 0.526, and 0.910 Å²/ps.
