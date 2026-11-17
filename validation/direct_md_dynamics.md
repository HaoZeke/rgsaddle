# Thermostat, lattice, and detector tests, Mishin Cu

Our free-dumbbell defect D is about twice the Table 1 D_i of Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025), which they quote from Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022). The first section shows from both papers and from our own trajectories that the two are different quantities. The rest repeats the D measurement under the paper's protocol (NPH for 30 ps, then NVE with velocity rescaling, at zero pressure), under Langevin with ten times weaker friction, and under Langevin at the zero-total-pressure lattice; and the whole-cell k with a Wigner-Seitz capture test under Langevin and NPH-then-NVE. Every campaign uses the same potential, 13³ cell, timestep, seeds, and blocks, except the shorter-timestep repeat of the Timestep section, which has half the blocks. Each number below is read from the CSV named in its row.

## What the Table 1 D_i is

Page numbers are those of the accepted manuscripts on OSTI: Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025), OSTI 2513909; B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022), their Ref. [12], OSTI 1870695.

- Section 1 (p. 2): "The recombination rate coefficient K_iv may be written as K_iv = V_iv D_i/(αa²), where V_iv is the vacancy-SIA recombination volume, D_i is SIA diffusivity, a is the cubic lattice parameter, and α is a numerical factor that depends on the SIA migration mechanism (α = 2/3 for [100] split dumbbell SIAs in Cu [10]). The quotient D_i/(αa²) is just the SIA migration frequency, ν."
- Section 2 (p. 5): "we calculate the corresponding recombination volume as V_iv = K_iv αa²/D_i. The values of SIA diffusivity D_i for the temperatures of interest are taken from Ref. [12], which obtained D_i directly from mean squared displacements of SIAs in MD simulations". With Ω = a³/4 this is V_iv/Ω = 4αK_iv/(aD_i); α multiplies K_iv.
- Section 5 (p. 10), after Eq. (4), K_iv = (1/3)Ω Σ_j ν_j over the Johnson jump paths that lead to recombination: "If we further assume that all the migration rates ν_j equal ν—the rate of SIA migration in the interior of a perfect crystal, far from any other defects—then, from the relation K_iv = V_iv D_i/(αa²) = V_iv ν (see section 1), we get V_iv = (1/3)Ωs". ν is the rate of one jump path of the defect, and D_i = (2/3)a²ν is the diffusivity of the defect for eight uncorrelated jumps of length a/√2; section 6 (p. 15) writes the same D_i = (2/3)a²ν₀^i e^(−ΔE_i/k_BT) for TST.
- Ref. [12], Eq. (5) (p. 4): the "generalized mean square displacement (MSD) per atom matrix" G_MSD = (1/N) Σ_i [Δx_i², Δx_iΔy_i, …], with "displacements of individual atoms from their initial positions", "2Dt = G_MSD in the limit of a long time", and "linear least-squares-fits to the corresponding elements of G_MSD over time" after the first 1 ps. The cell has 4951 atoms with the SIA (p. 3); the volume follows thermal expansion and the runs are 2 ns of NVE (p. 4).
- Erratum, Phys. Rev. Materials 9, 069901(E) (2025), doi:10.1103/8hb2-86lm, replaces the p. 4 sentence that this D "is the SIA diffusivity" with "The diffusivity thereby obtained is the self-diffusivity of Cu, not the SIA diffusivity, because G_MSD is computed on a per-atom basis", and states that the diffusivities of Figs. 4 and 5(b) "may be converted to SIA diffusivities by multiplying them by the number of atoms in the model, i.e., 4951." Fig. 4 of Ref. [12] (p. 9) plots those diffusivities at 700 K in µm²/s; at zero strain the components read 1.84 µm²/s = 1.84×10⁻⁴ Å²/ps, and 4951 times that is 0.911 Å²/ps, the 700 K D_i of Table 1 (0.910 ± 0.012).
- Potential: section 3 (p. 6), "All atomistic simulations are carried out using the embedded atom method (EAM) potential for Cu by Mishin et al. [32]"; Ref. [12] (p. 3), "an embedded atom method (EAM) [24] potential for Cu developed by Mishin et al. [25]". Both references are Y. Mishin, M. J. Mehl, D. A. Papaconstantopoulos, A. F. Voter, and J. D. Kress, Phys. Rev. B 63, 224106 (2001), the EAM1 of `tests/mishin_cu.rs` that every campaign here uses.

Table 1 D_i is therefore 4951 D*, with D* the per-atom diffusivity of Eq. (5): N D* = Σ_i ⟨Δx_i²⟩/2t, a sum of tracer displacements. One Johnson jump moves one atom by the defect's own jump vector, so N D* would equal the defect diffusivity only if successive jumps moved uncorrelated atoms. They do not: the atom that has just moved is one of the two atoms of the new dumbbell, so it moves again at the next jump with probability 1/2, and that move points back along its last one on average. N D* = f D_defect defines the tracer correlation factor f of the mechanism, while the ν of section 1 is the defect's own jump rate, with (2/3)a²ν = D_defect when successive defect jumps are uncorrelated. Both are measured below on the same free-dumbbell trajectories, with the dynamics of the whole-cell k ensembles (`direct_md_sia_tracer_*.md`). N D* is the Ref. [12] fit of drift-removed positions; defect D is the Wigner-Seitz block value of the same trajectories; f is N D* over defect D, the ratio of the two means. Ratios carry both errors in quadrature, and z is the difference over the two errors in quadrature.

| T (K) | dynamics | CSV | seeds | N D*, Ref. [12] fit (Å²/ps) | Table 1 D_i [12] | N D* / Table 1 | z | defect D (Å²/ps) | defect D / Table 1 | f |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | Langevin, γ = 0.1/ps, zero total pressure | `direct_md_sia_tracer_p0_gamma0.1_dt10fs.csv` | 8 | 0.1392 ± 0.0042 | 0.143 ± 0.006 | 0.973 ± 0.050 | -0.52 | 0.2717 ± 0.0157 | 1.900 ± 0.136 | 0.5122 ± 0.0334 |
| 300 | NPH then NVE, rescaled | `direct_md_sia_tracer_nph_nve_dt10fs.csv` | 8 | 0.1511 ± 0.0050 | 0.143 ± 0.006 | 1.056 ± 0.056 | 1.04 | 0.3198 ± 0.0143 | 2.236 ± 0.137 | 0.4725 ± 0.0262 |
| 500 | Langevin, γ = 0.1/ps, zero total pressure | `direct_md_sia_tracer_p0_gamma0.1_dt10fs.csv` | 8 | 0.5342 ± 0.0097 | 0.526 ± 0.006 | 1.016 ± 0.022 | 0.72 | 1.0590 ± 0.0449 | 2.013 ± 0.088 | 0.5044 ± 0.0233 |
| 500 | NPH then NVE, rescaled | `direct_md_sia_tracer_nph_nve_dt10fs.csv` | 8 | 0.5705 ± 0.0127 | 0.526 ± 0.006 | 1.085 ± 0.027 | 3.17 | 1.2415 ± 0.0698 | 2.360 ± 0.135 | 0.4595 ± 0.0278 |
| 700 | Langevin, γ = 0.1/ps, zero total pressure | `direct_md_sia_tracer_p0_gamma0.1_dt10fs.csv` | 8 | 0.9412 ± 0.0195 | 0.910 ± 0.012 | 1.034 ± 0.025 | 1.36 | 2.0246 ± 0.1230 | 2.225 ± 0.138 | 0.4649 ± 0.0298 |
| 700 | NPH then NVE, rescaled | `direct_md_sia_tracer_nph_nve_dt10fs.csv` | 8 | 0.9382 ± 0.0245 | 0.910 ± 0.012 | 1.031 ± 0.030 | 1.03 | 2.1380 ± 0.1123 | 2.349 ± 0.127 | 0.4388 ± 0.0257 |

Across these rows N D* / Table 1 runs 0.973 to 1.085, defect D / Table 1 runs 1.900 to 2.360, and f runs 0.439 to 0.512.

Per trajectory, the Wigner-Seitz block D of `direct_md_sia_tracer_p0_gamma0.1_dt10fs.csv` and the D of `direct_md_sia_d_p0.csv` differ by at most 8.8e-9 Å²/ps over 24 trajectories: these are the same trajectories.

Per trajectory, the Wigner-Seitz block D of `direct_md_sia_tracer_nph_nve_dt10fs.csv` and the D of `direct_md_sia_d_nph_nve.csv` differ by at most 1.3e0 Å²/ps over 24 trajectories: these are other trajectories with the same seeds. The NPH stage of `direct_md_sia_d_nph_nve.csv` started at the earlier lattice constants of the V_iv runs, and the tracer campaign starts it at the zero-total-pressure lattice of `direct_md_lattice.csv`.

The same correlation on an ideal lattice: a <100> dumbbell that moves only by Johnson jumps, the moving atom ending on the near side of the new dumbbell (`johnson_walk`). f(n) is Σ_i |Δs_i|² over n λ² after n jumps, mean ± standard error over the walks, seed 0x5EED. The lattice value is not an input to any number above.

| jumps n | walks | f(n) |
|---:|---:|---:|
| 10 | 40000 | 0.5175 ± 0.0010 |
| 100 | 10000 | 0.4480 ± 0.0008 |
| 1000 | 2000 | 0.4413 ± 0.0006 |
| 10000 | 400 | 0.4405 ± 0.0004 |
| 100000 | 100 | 0.4397 ± 0.0003 |

## Timestep

Every other campaign in this note integrates at 10 fs. `direct_md_sia_tracer_p0_gamma0.1_dt2fs.csv` repeats `direct_md_sia_tracer_p0_gamma0.1_dt10fs.csv` at 2 fs with the same seeds and τ and half the blocks. Neither Mwungeli, Liu, and Demkowicz nor Ref. [12] states an MD timestep.

| T (K) | quantity | 2 fs | 10 fs | ratio | z |
|---:|:---|---:|---:|---:|---:|
| 500 | site changes/ps | 1.1687 ± 0.0197 (8 seeds) | 1.1712 ± 0.0104 (8 seeds) | 0.998 ± 0.019 | -0.11 |
| 500 | D, jump sum | 1.2957 ± 0.0220 (8 seeds) | 1.2969 ± 0.0114 (8 seeds) | 0.999 ± 0.019 | -0.05 |
| 500 | defect D | 1.2999 ± 0.0670 (8 seeds) | 1.0590 ± 0.0449 (8 seeds) | 1.227 ± 0.082 | 2.98 |
| 500 | N D*, Ref. [12] fit | 0.5182 ± 0.0134 (8 seeds) | 0.5342 ± 0.0097 (8 seeds) | 0.970 ± 0.031 | -0.97 |
| 500 | f | 0.3986 ± 0.0230 (8 seeds) | 0.5044 ± 0.0233 (8 seeds) | 0.790 ± 0.058 | -3.23 |

At 500 K, defect D over the jump-sum D, which is 1 when successive defect jumps are uncorrelated, is 1.003 ± 0.054 at 2 fs and 0.817 ± 0.035 at 10 fs. N D* over the Table 1 D_i is 0.985 ± 0.028 (z = -0.53) at 2 fs and 1.016 ± 0.022 (z = 0.72) at 10 fs.

## Free dumbbell diffusivity

D for one trajectory is the block mean of |Δr|² over 6τ of the doubly occupied Wigner-Seitz cell; σ is the standard error across trajectories. Ratios carry both errors in quadrature, and z is the difference over the two errors in quadrature. The reference for the last two columns is the first campaign.

| T (K) | dynamics | CSV | a (Å) | seeds | blocks | τ (ps) | D (Å²/ps) | σ | σ/D | D / Table 1 | D / Langevin, γ = 0.1/ps | z |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | Langevin, γ = 0.1/ps | `direct_md_sia_d_long.csv` | 3.628311 | 8 | 320 | 40.0 | 0.2770 | 0.0182 | 6.6% | 1.937 ± 0.151 | - | - |
| 300 | Langevin, γ = 0.01/ps | `direct_md_sia_d_gamma0.01.csv` | 3.628311 | 8 | 320 | 40.0 | 0.2591 | 0.0148 | 5.7% | 1.812 ± 0.128 | 0.935 ± 0.081 | -0.76 |
| 300 | NPH then NVE, rescaled | `direct_md_sia_d_nph_nve.csv` | 3.632699 | 8 | 320 | 40.0 | 0.3137 | 0.0160 | 5.1% | 2.194 ± 0.145 | 1.132 ± 0.094 | 1.51 |
| 300 | Langevin, γ = 0.1/ps, zero total pressure | `direct_md_sia_d_p0.csv` | 3.631851 | 8 | 320 | 40.0 | 0.2717 | 0.0157 | 5.8% | 1.900 ± 0.136 | 0.981 ± 0.086 | -0.22 |
| 500 | Langevin, γ = 0.1/ps | `direct_md_sia_d_long.csv` | 3.638666 | 8 | 320 | 20.0 | 1.0751 | 0.0397 | 3.7% | 2.044 ± 0.079 | - | - |
| 500 | Langevin, γ = 0.01/ps | `direct_md_sia_d_gamma0.01.csv` | 3.638666 | 8 | 320 | 20.0 | 1.1014 | 0.0422 | 3.8% | 2.094 ± 0.084 | 1.025 ± 0.054 | 0.46 |
| 500 | NPH then NVE, rescaled | `direct_md_sia_d_nph_nve.csv` | 3.645155 | 8 | 320 | 20.0 | 1.1448 | 0.0638 | 5.6% | 2.176 ± 0.124 | 1.065 ± 0.071 | 0.93 |
| 500 | Langevin, γ = 0.1/ps, zero total pressure | `direct_md_sia_d_p0.csv` | 3.644035 | 8 | 320 | 20.0 | 1.0590 | 0.0449 | 4.2% | 2.013 ± 0.088 | 0.985 ± 0.055 | -0.27 |
| 700 | Langevin, γ = 0.1/ps | `direct_md_sia_d_long.csv` | 3.649788 | 8 | 320 | 10.0 | 1.9114 | 0.0822 | 4.3% | 2.100 ± 0.094 | - | - |
| 700 | Langevin, γ = 0.01/ps | `direct_md_sia_d_gamma0.01.csv` | 3.649788 | 8 | 320 | 10.0 | 1.8306 | 0.0783 | 4.3% | 2.012 ± 0.090 | 0.958 ± 0.058 | -0.71 |
| 700 | NPH then NVE, rescaled | `direct_md_sia_d_nph_nve.csv` | 3.658733 | 8 | 320 | 10.0 | 2.0717 | 0.1631 | 7.9% | 2.277 ± 0.182 | 1.084 ± 0.097 | 0.88 |
| 700 | Langevin, γ = 0.1/ps, zero total pressure | `direct_md_sia_d_p0.csv` | 3.657362 | 8 | 320 | 10.0 | 2.0246 | 0.1230 | 6.1% | 2.225 ± 0.138 | 1.059 ± 0.079 | 0.76 |

Langevin, γ = 0.01/ps over Langevin, γ = 0.1/ps: D ratio 0.935 to 1.025 across temperatures.
NPH then NVE, rescaled over Langevin, γ = 0.1/ps: D ratio 1.065 to 1.132 across temperatures.
Langevin, γ = 0.1/ps, zero total pressure over Langevin, γ = 0.1/ps: D ratio 0.981 to 1.059 across temperatures.
Every campaign over Table 1: D ratio 1.812 to 2.277.

Table 1 D_i (Å²/ps): 300 K 0.143 ± 0.006; 500 K 0.526 ± 0.006; 700 K 0.910 ± 0.012.

## Lattice constant

The first two campaigns ran at the lattice constants `npt_lattice` in `tests/mishin_cu.rs` produced while its Berendsen loop balanced only the configurational stress from `evaluate`; that lattice carries the kinetic pressure N k_B T / V in the last column. `npt_lattice` now subtracts N k_B T / V from the tension, so it reaches zero total pressure; `direct_md_lattice` in that file writes the result to `direct_md_lattice.csv` (σ from eight blocks of the averaged half), and the driver reads its lattice constants from there. The NPH column is the mean and standard deviation over trajectories of the production lattice constant of `direct_md_sia_d_nph_nve.csv`, whose barostat uses the full virial pressure (2K - dE/dλ) / 3V. Table 1 lists the paper's zero-pressure lattice constants.

| T (K) | configurational-stress lattice (Å) | zero total pressure (Å) | σ | after NPH (Å) | sd | trajectories | Table 1 (Å) | N k_B T / V at the configurational-stress lattice (GPa) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 3.628311 | 3.631851 | 0.000045 | 3.632699 | 0.000012 | 8 | 3.631 | 0.347 |
| 500 | 3.638666 | 3.644035 | 0.000067 | 3.645155 | 0.000024 | 8 | 3.643 | 0.573 |
| 700 | 3.649788 | 3.657362 | 0.000095 | 3.658733 | 0.000017 | 8 | 3.655 | 0.795 |

## Whole-cell k

Random-vacancy ensembles as in `direct_md_random.md`: the through-origin fit of ln(n/N) every 25 ps up to the longest window, σ_k from 400 bootstrap resamples. The coordination detector calls a trajectory captured at the first sample at which every atom has 12 neighbours inside 3.1 Å; the site detector when every Wigner-Seitz site of the ideal lattice holds one atom for the confirmation window, dated to the first sample of that window. An ensemble with fewer trajectories than the others is a partial run of the same seeds. The Table 1 row backs k out of the k-method volume with V_iv/Ω = 4αk/(aD_i), the Table 1 D_i, and the paper's lattice constant.

| T (K) | dynamics | detector | CSV | N | captured | t_max (ps) | a (Å) | k (Å³/ps) | σ_k | k / Table 1 k |
|---:|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 300 | Langevin, γ = 0.1/ps | site_occupancy | `direct_md_random_sites_langevin_300K.csv` | 96 | 47 | 1500 | 3.631851 | 43.01 | 6.70 | 0.870 ± 0.158 |
| 300 | NPH then NVE, rescaled | site_occupancy | `direct_md_random_sites_nph_nve_300K.csv` | 96 | 41 | 1500 | 3.632566 | 36.88 | 6.07 | 0.746 ± 0.141 |
| 300 | Table 1, k-method, D_i 0.143 ± 0.006 | - | - | 100 | - | - | 3.631 | 49.46 | 4.59 | - |
| 500 | Langevin, γ = 0.1/ps | coordination | `direct_md_random.csv` | 192 | 149 | 1500 | 3.638666 | 92.33 | 8.07 | 0.695 ± 0.106 |
| 500 | Langevin, γ = 0.1/ps | site_occupancy | `direct_md_random_sites_langevin_500K.csv` | 96 | 82 | 1500 | 3.644035 | 133.29 | 16.42 | 1.003 ± 0.176 |
| 500 | NPH then NVE, rescaled | site_occupancy | `direct_md_random_sites_nph_nve_500K.csv` | 96 | 77 | 1500 | 3.645023 | 130.67 | 14.47 | 0.983 ± 0.164 |
| 500 | Table 1, k-method, D_i 0.526 ± 0.006 | - | - | 100 | - | - | 3.643 | 132.94 | 16.60 | - |
| 700 | Langevin, γ = 0.1/ps | site_occupancy | `direct_md_random_sites_langevin_700K.csv` | 96 | 92 | 1500 | 3.657362 | 232.40 | 30.18 | 0.918 ± 0.128 |
| 700 | NPH then NVE, rescaled | site_occupancy | `direct_md_random_sites_nph_nve_700K.csv` | 96 | 90 | 1500 | 3.658694 | 198.97 | 24.52 | 0.786 ± 0.105 |
| 700 | Table 1, k-method, D_i 0.910 ± 0.012 | - | - | 100 | - | - | 3.655 | 253.20 | 12.91 | - |

At 300 K with the site detector, NPH then NVE, rescaled over Langevin, γ = 0.1/ps: k ratio 0.858 ± 0.194 (z = -0.68).
At 500 K with the site detector, NPH then NVE, rescaled over Langevin, γ = 0.1/ps: k ratio 0.980 ± 0.162 (z = -0.12).
At 700 K with the site detector, NPH then NVE, rescaled over Langevin, γ = 0.1/ps: k ratio 0.856 ± 0.153 (z = -0.86).

## V_iv by D_i convention

V_iv/Ω = 4αk/(aD_i) with α = 2/3 and the k of the table above, for four D_i: the Table 1 value, which is 4951 D* from Ref. [12]; N D* measured on our trajectories with the dynamics of the ensemble; the defect D of those dynamics; and their jump-sum D, (2/3)a²ν with ν the counted site changes over the eight jump paths, so that this column is k/ν. The defect D is the D_i of ν = D_i/(αa²) when successive defect jumps are uncorrelated, and then the defect-D and jump-sum columns agree; the Timestep section tests that. σ adds the relative errors of k and D_i in quadrature. The previous column is the conversion this driver used before, k a²/(α D_i Ω), with the defect D: α was inverted, which multiplies V by 1/α² = 9/4. The last column is Table 1, k-method, 8788 atoms.

| T (K) | dynamics | detector | Table 1 D_i | N D*, ours | defect D, ours | jump-sum D, ours | previous: k a²/(αD), defect D | Table 1 |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|
| 300 | Langevin, γ = 0.1/ps | site_occupancy | 220.8 ± 35.6 | 226.9 ± 36.0 | 116.2 ± 19.3 | 84.5 ± 13.3 | 261.5 ± 43.4 | 254 ± 21 |
| 300 | NPH then NVE, rescaled | site_occupancy | 189.3 ± 32.2 | 179.2 ± 30.1 | 86.3 ± 14.9 | 67.3 ± 11.1 | 194.2 ± 33.5 | 254 ± 21 |
| 500 | Langevin, γ = 0.1/ps | coordination | 128.6 ± 11.3 | - | 62.9 ± 6.0 | - | 141.6 ± 13.4 | 185 ± 23 |
| 500 | Langevin, γ = 0.1/ps | site_occupancy | 185.4 ± 22.9 | 182.6 ± 22.7 | 92.1 ± 12.0 | 75.2 ± 9.3 | 207.2 ± 27.0 | 185 ± 23 |
| 500 | NPH then NVE, rescaled | site_occupancy | 181.7 ± 20.2 | 167.6 ± 18.9 | 83.5 ± 10.3 | 70.0 ± 7.8 | 187.9 ± 23.3 | 185 ± 23 |
| 700 | Langevin, γ = 0.1/ps | site_occupancy | 186.2 ± 24.3 | 180.0 ± 23.7 | 83.7 ± 12.0 | 78.4 ± 10.2 | 188.3 ± 27.0 | 203 ± 10 |
| 700 | NPH then NVE, rescaled | site_occupancy | 159.4 ± 19.8 | 154.6 ± 19.5 | 70.0 ± 10.2 | 64.5 ± 8.0 | 157.5 ± 23.0 | 203 ± 10 |

The previous column over the N D* column is (9/4) N D*/D, with D the defect D of the row. On the "Langevin, γ = 0.1/ps" rows with an N D*, the tracer campaign reran the trajectories of that D, so N D*/D is f, here 0.465 to 0.512, and the ratio is 1.046 to 1.152. The "NPH then NVE, rescaled" rows take N D* and D from different trajectories with the same seeds (see the first section).

Table 1, 8788 atoms: 300 K 254 ± 21 Ω (k-method), 286 ± 28 Ω (mean-time); 500 K 185 ± 23 Ω (k-method), 161 ± 15 Ω (mean-time); 700 K 203 ± 10 Ω (k-method), 210 ± 21 Ω (mean-time).
