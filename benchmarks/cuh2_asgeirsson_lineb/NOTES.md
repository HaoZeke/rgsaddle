# H₂ / Cu(110) LI-NEB + instanton harness (Ásgeirsson 2018 method)

## Verdict

This directory is the **reproducible, runnable source** for the H₂ associative
desorption benchmark that `docs/orgmode/howto/lineb.org` summarizes: six
hydrogen degrees of freedom over a **fixed** 216-atom Cu(110) slab, the
canonical **rgpot CuH2** EAM (Jonsson-group lineage via eOn /
`fortran_cuh2_src`), a classical MEP, a fixed-energy LI-NEB ladder, and a
head-to-head of **direct** ring-polymer HQ-TST (`hqtst_rate`) against the
paper’s interpolated-Hessian oracle (`hqtst_rate_interpolated_hessian`), with
**every** surface evaluation counted.

It is **not** a claim that Ásgeirsson, Arnaldsson & Jónsson, *J. Chem. Phys.*
**148**, 102334 (2018) published a full H₂/Cu(110) figure set. That paper’s
worked electronic-structure examples are Müller–Brown, H₃BNH₃, and H in Ta.
Section II.E (and the force-call discussion reused in
`src/lineb/check.rs`) cites **H₂ desorbing from a 216-atom Cu(110) slab on an
empirical potential**, with **~900** energy/force evaluations per temperature
to compute the reactive flux — the cost target this harness accounts against.
The same EAM system is treated in detail in Arnaldsson’s 2007 thesis
(HQTST / instanton on Mills–Jónsson EAM), which supplies geometric and rate
comparisons used as published reference points below.

## Primary sources

| source | role |
|---|---|
| Ásgeirsson, Arnaldsson, Jónsson, JCP **148**, 102334 (2018), doi:10.1063/1.5007180 | LI-NEB method, fixed-energy ladder, instanton temperature from half-period, harmonic rate; ~900 evals/T for H₂/Cu(110) EAM flux (Sec. II.E) |
| Arnaldsson, PhD thesis, University of Washington (2007), hdl:20.500.11815/717 | Associative desorption of H₂ from Cu(110) on the Mills–Jónsson EAM; geometries, crossover ~264.6 K, HQTST vs RAW-QTST |
| Mills, Jónsson, Schenter, *Surf. Sci.* / related EAM papers | Empirical Cu–H EAM used for H₂/Cu(110) |
| OmniPotentRPC/rgpot `CppCore/rgpot/CuH2/` + Fortran `rgpot_cuh2_*` | Canonical CuH2 potential + `tmp.con` fixture served as `CuH2` |
| TheochemUI/fortran_cuh2_src | Upstream fortcuh2 extraction from eOn |
| TheochemUI/eOn `client/unit_tests/data/systems/cuh2_neb_test/` | 218-atom CON endpoints (216 Cu + 2 H) used for NEB unit tests |

## What this harness computes

1. **Absolute energies** of the adsorbed reactant, desorbed product, and
   climbing-image MEP saddle on the **same** CuH2 energy zero (raw total eV).
2. A **15-image** climbing MEP in the **6 H coordinates** with Cu frozen.
3. A **fixed-energy ladder**: `Continuation::new(0.05, 10, 9)` — nine energies
   starting `0.05 eV` below the MEP maximum, ten LI-NEB images, tolerance
   `6.44e-4` (paper’s `1e-2`/Å for H in eV·Å·amu), otherwise `LinebConfig`
   defaults, with `ℏ = 0.06466` and `k_B = 8.617333262145e-5 eV/K`.
4. At each ladder level, **direct** `hqtst_rate(..., beads=32)` vs
   **oracle** `hqtst_rate_interpolated_hessian`, comparing
   `log_rate_over_reactant` (log flux / `ln(k Z_r)`).
5. **Call accounting** for every phase: MEP band evaluations, ladder seed
   evaluations (total surface calls minus returned `LinebPath.calls`),
   per-path `optimizer` / `measure` / `dynamics`, and rate `evaluations`.

## Potential and setup discrepancies

On the eOn endpoint substrate, the climbing band converged to a barrier
of 0.52874149 eV above the adsorbed state. At the highest image, central
energy differences at 0.0001 Å agreed with the six analytic force
components to a Euclidean error of 6.6 × 10⁻⁹ eV/Å. The corresponding
fixed-energy ladder stopped above its force tolerance on this fixture.
The 0.469 eV result below uses different frozen copper coordinates.

`direct_ring.py` takes a CuH2 shared library, a CON substrate, the
calculated `mep.csv`, an output directory, and temperatures in kelvin.
It solves the discrete imaginary-time stationarity equations at 150 beads,
starting from the negative mode of the classical saddle. The full Cartesian
gradient must be below 10⁻⁸ eV/Å componentwise. It identifies the cyclic
mode by overlap with the bead tangent and requires one remaining negative
eigenvalue before writing a logarithmic flux. The absolute energy zero of
the substrate enters this flux and must match the LI-NEB comparison.

### Potential lineage

- **rgpot `CuH2Pot`** is the Fortran 2018 port of the eOn Cu–H EAM
  (`fortran_cuh2_src` / eOn `PotType::CUH2`). It is the potential this
  harness calls (via `rgpot_cuh2_force` in a built `librgpot` / potserv, not
  a toy surrogate).
- Absolute energies are **large negative totals** (~−700 eV for the slab);
  only **differences** are chemically meaningful. The harness always reports
  both absolute totals and differences on one energy zero.
- The Ásgeirsson 2018 “~900 evaluations” figure and Arnaldsson’s rate curves
  use the **Mills–Jónsson EAM**. rgpot CuH2 is that family as carried by
  eOn/rgpot; residual parameter or neighbour-table differences versus a 2007
  binary cannot be ruled out without bit-identical reference forces.

### Fixtures (see `fixtures/provenance.json`)

| file | geometry | notes |
|---|---|---|
| `eon_adsorbed_min2.con` | HH ≈ 3.157 Å, height above top Cu ≈ 0.78 Å, d(H–Cu) ≈ 2.03 Å | Reactant for associative desorption. Thesis Fig. 3.12 quotes HH = 3.05 Å, height 2.06 Å above nearest Cu. |
| `eon_desorbed_min1.con` | HH ≈ 0.740 Å, height ≈ 4.76 Å | Product (molecular H₂). Same Cu lattice as min2 (Cu RMS = 0). |
| `rgpot_tmp.con` | HH ≈ 1.343 Å, d(H–Cu) ≈ 2.17 Å | **Near the classical TS**, not a basin. All CON free-flags are 0; the harness freezes Cu in software. Cu lattice RMS vs eOn ≈ 0.009 Å. |
| `fortran_cuh2_tmp.con` | matches eOn desorbed to printed precision | fortcuh2 demo; energy −697.31345026617748 eV in upstream README. |

eOn CON column 4 is **`isFixed`** (1 = fixed, 0 = free): all 216 Cu fixed, both H free → **6 DOF**, matching the intended setup. rgpot’s `tmp.con` does not encode that mask; freezing is the `PointSurface`’s job.

### Energetics

| quantity | published / prior | this setup |
|---|---|---|
| Thesis adsorbed vs gas-phase H₂ | 0.157 eV | measured at run time on CuH2 |
| Thesis TS vs gas-phase H₂ | 0.702 eV | measured at run time |
| Thesis classical barrier (adsorbed → TS) | 0.545 eV | — |
| rgsaddle howto measured barrier (rgpot CuH2, 15-image MEP) | **0.469 eV** above reactant | harness must reproduce this order when Cu lattice + H path match that measurement |
| Crossover temperature (thesis EAM) | 264.6 K | ladder here targets ~235–63 K (howto) |

A barrier of 0.469 eV on rgpot CuH2 versus 0.545 eV in the thesis is a **real potential/setup discrepancy**, not a rounding error. Possible contributors: Cu relaxation in the fixture, H endpoint placement (eOn vs thesis), and EAM parameter spelling in the Fortran port. The harness prints absolute endpoint and saddle energies so the difference is auditable.

### Method vs paper oracle

- **Direct rate**: Hessian–vector products at distinct ring-polymer replicas
  (`hqtst_rate`). Cost `(P/2+1)(1+2×6) = 221` force calls at `P = 32` with
  finite-difference HVPs — matching the howto.
- **Oracle**: finite-difference Hessians at LI-NEB images, cubic
  interpolation, dense spectrum (`lineb-oracle`). This is the paper’s
  rate construction, compiled only behind that feature.
- Log comparison uses `log_rate_over_reactant`; the exponential form
  overflows on a ~−700 eV energy zero.

### Call-budget reference (howto, 6 serial identical runs)

| phase | mean force calls |
|---|---|
| LI-NEB optimizer / energy | 466.7 (σ = 98.0; range 310–580) |
| quadrature + time table | 925.0 / energy |
| constrained dynamics | 259.8 / energy |
| seeds (all 9 levels) | 977 |
| rate P = 32 | 221 / energy |
| paper Sec. II.E reactive flux | ~900 / temperature |

The harness prints the same breakdown and the total surface-call counter so
nothing is left outside the ledger.

## How to run

Fixtures-only (no potential library; still needs feature `cuh2-lineb` to
compile the example):

```bash
cargo run --features cuh2-lineb --example cuh2_asgeirsson_lineb -- fixtures
```

Full MEP + ladder + log-flux compare (needs feature `cuh2-lineb` and a library
that exports `rgpot_cuh2_ensure_params` / `rgpot_cuh2_force`, e.g. librgpot
built with Fortran pots):

```bash
export RGPOT_CUH2_LIB=/path/to/librgpot_cuh2.so
cargo run --release --features cuh2-lineb --example cuh2_asgeirsson_lineb -- \
  --lib "$RGPOT_CUH2_LIB" --all --out target/cuh2_asgeirsson
```

Stages: `--smoke` (absolute endpoint energies), `--mep`, `--ladder`,
`--rates`, or `--all`. Outputs land under `--out` as CSV/JSON.

## Layout

```
benchmarks/cuh2_asgeirsson_lineb/
  NOTES.md                 this file
  published_refs.csv       thesis / paper numbers used as oracles
  fixtures/                CON + provenance.json
examples/cuh2_asgeirsson_lineb.rs
```
