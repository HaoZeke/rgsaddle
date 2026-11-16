# Thermostat, lattice, and detector tests, Mishin Cu

Our free-dumbbell D under Langevin at γ = 0.1/ps is larger than the Table 1 D_i of Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025), which they quote from Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022), and our first whole-cell k at 500 K is larger than the k implied by their Table 1. Their trajectories are NPH for 30 ps, then NVE with velocity rescaling, at zero pressure. This note repeats the D measurement under that protocol, under Langevin with ten times weaker friction, and under Langevin at the zero-total-pressure lattice; and the whole-cell k with a Wigner-Seitz capture test under Langevin and NPH-then-NVE. Every campaign uses the same potential, 13³ cell, timestep, seeds, and blocks. Each number below is read from the CSV named in its row.

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

Random-vacancy ensembles as in `direct_md_random.md`: the through-origin fit of ln(n/N) every 25 ps up to the longest window, σ_k from 400 bootstrap resamples. V_iv/Ω = 4k / (α a D_i) with α = 2/3; the total σ adds the D_i error in quadrature. Own D is the free-dumbbell D measured with the same dynamics at the same lattice constant. The coordination detector calls a trajectory captured at the first sample at which every atom has 12 neighbours inside 3.1 Å; the site detector when every Wigner-Seitz site of the ideal lattice holds one atom for the confirmation window, dated to the first sample of that window. An ensemble with fewer trajectories than the others is a partial run of the same seeds. The Table 1 row backs k out of the k-method volume with the Table 1 D_i and the paper's lattice constant.

| T (K) | dynamics | detector | CSV | N | captured | t_max (ps) | a (Å) | k (Å³/ps) | σ_k | own D (Å²/ps) | V_iv/Ω, own D | σ | V_iv/Ω, Table 1 D | σ | k / Table 1 k |
|---:|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | Table 1, k-method | - | - | 100 | - | - | 3.631 | 21.98 | 2.04 | 0.143 ± 0.006 | 254 | 21 | 254 | 21 | - |
| 500 | Langevin, γ = 0.1/ps | coordination | `direct_md_random.csv` | 192 | 149 | 1500 | 3.638666 | 92.33 | 8.07 | 1.075 ± 0.040 | 141.62 | 13.43 | 289.44 | 25.51 | 1.563 ± 0.238 |
| 500 | Table 1, k-method | - | - | 100 | - | - | 3.643 | 59.08 | 7.38 | 0.526 ± 0.006 | 185 | 23 | 185 | 23 | - |
| 700 | Table 1, k-method | - | - | 100 | - | - | 3.655 | 112.53 | 5.74 | 0.910 ± 0.012 | 203 | 10 | 203 | 10 | - |


Table 1, 8788 atoms: 300 K 254 ± 21 Ω (k-method), 286 ± 28 Ω (mean-time); 500 K 185 ± 23 Ω (k-method), 161 ± 15 Ω (mean-time); 700 K 203 ± 10 Ω (k-method), 210 ± 21 Ω (mean-time).
