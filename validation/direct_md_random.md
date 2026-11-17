# Random-vacancy survival, Mishin Cu

The vacancy is the lattice site nearest the cell center. The <100> dumbbell host is drawn uniformly from the other sites of the 13³ cell, and the dumbbell axis uniformly from the three orientations. Under periodic boundaries the separation is uniform over the cell, which is the ensemble in section 2 of Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025). The fit is ln S(t) = -(k/V_model) t through the origin, sampled every 25 ps, with V_model the cell volume and S the Kaplan-Meier survival, in which an uncaptured trajectory is censored at the end of its window; with equal windows S = n(t)/N. Samples after S reaches zero are left out. V_iv/Ω = 4 α k / (a D_i) with α = 2/3 and Ω = a³/4, from K_iv = V_iv D_i / (α a²) = V_iv ν in section 1 of that paper, where ν is the rate of one of the eight translation jumps of the dumbbell. That ν is the defect's own jump rate, so the D_i the relation needs is (2/3)a²ν, the diffusivity of the defect when its successive jumps are uncorrelated; `direct_md_dynamics.md` tests that, and also converts k with ν counted from site changes.

Table 1 of that paper converts its k with the D_i in its own Table 1 column, which it quotes from Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022). Ref. [12] took the mean-square displacement of every atom, its Eq. (5), and its erratum, Phys. Rev. Materials 9, 069901(E) (2025), multiplies that per-atom diffusivity by the 4951 atoms of its cell. The product is the tracer sum N D* = Σ_i ⟨|Δr_i|²⟩ / 6t, which is the defect diffusivity times the tracer correlation factor (`direct_md_dynamics.md`). The paper's own trajectories are NPH for 30 ps, then NVE with velocity rescaling, at its lattice constant. Our k comes from the Langevin dynamics in the CSV. The N D* column, the convention of Table 1, is empty: no tracer campaign ran with these dynamics at this lattice constant. The defect-D column uses the Wigner-Seitz defect diffusivity of those dynamics (`direct_md_sia_d_long.md`). σ_k is the standard deviation over 400 bootstrap resamples of the trajectories. The bootstrap σ column carries only that; the total σ columns add the relative error of the D_i used in quadrature.

At 500 K, Table 1 D_i = 0.526 ± 0.006 Å²/ps; measured defect D = 1.075 ± 0.040 Å²/ps.

| t_max (ps) | captured | N | k (Å³/ps) | σ_k | V_iv/Ω, Table 1 D_i | bootstrap σ | total σ | V_iv/Ω, measured N D* | total σ | V_iv/Ω, measured defect D | total σ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 30 | 192 | 44.03 | 9.45 | 61.34 | 13.16 | 13.18 | - | - | 30.01 | 6.53 |
| 600 | 60 | 192 | 58.31 | 8.36 | 81.24 | 11.65 | 11.69 | - | - | 39.75 | 5.89 |
| 900 | 99 | 192 | 71.90 | 7.96 | 100.18 | 11.09 | 11.15 | - | - | 49.02 | 5.72 |
| 1200 | 127 | 192 | 84.97 | 8.12 | 118.39 | 11.32 | 11.40 | - | - | 57.93 | 5.94 |
| 1500 | 149 | 192 | 92.33 | 8.07 | 128.64 | 11.24 | 11.34 | - | - | 62.94 | 5.97 |

At the longest window, 1500 ps, k = 92.33 ± 8.07 Å³/ps. With the Table 1 D_i, V_iv = 128.64 ± 11.24 Ω (bootstrap) or ± 11.34 Ω (with D_i). With the measured defect D, V_iv = 62.94 ± 5.97 Ω.
The constant-hazard maximum likelihood over the same 1500 ps, with every uncaptured trajectory censored at the end of its window, is k = 91.58 Å³/ps from 149 captures in 192 trajectories, 95 % interval 78.00 to 107.54 from ln k ± 1.96/√captures. V_iv is 127.60 Ω (95 % 108.67 to 149.83) with the Table 1 D_i and 62.43 Ω (95 % 53.17 to 73.31) with the measured defect D; these intervals leave out the D_i error.
Table 1 at this temperature, 8788 atoms, is 185 ± 23 Ω (k-method) and 161 ± 15 Ω (mean-time method). The k-method value with the Table 1 D_i and the paper's lattice constant 3.643 Å implies k = 132.94 ± 16.60 Å³/ps. Our longest-window k over that is 0.695 ± 0.106. The k-method value subtracted from our volume is -56.36 Ω with the Table 1 D_i.

Table 1, 8788 atoms, k-method then mean-time method: 300 K 254 ± 21 and 286 ± 28; 500 K 185 ± 23 and 161 ± 15; 700 K 203 ± 10 and 210 ± 21. All use the Ref. [12] D_i of Table 1: 0.143, 0.526, and 0.910 Å²/ps.
