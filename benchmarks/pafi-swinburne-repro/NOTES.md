# PAFI Swinburne reproduction

## Verdict

**Partially reproduced.** The fast test exactly replays the public PAFI
repository's pinned Fe-SIA zero-temperature gradient fixture and its cubic
postprocessing. The ignored test is a statistical regression against the
three numbers printed in the public repository's W-vacancy example and sends
the EAM evaluations through `rgpot` and `lammpc`. A four-trajectory moderate
smoke passes the 1000 and 2000 K rows but fails the deterministic 0 K row.
The full BenchBot result is not claimed until that campaign has run.

The two applications in the PRL itself were **not reproduced**. The article
does not publish their atomic paths, cells, or numerical free-energy tables,
and no supplementary-information file is linked by the PRL or HAL records or
present in the public PAFI repository. Figure ordinates alone are not a
regression oracle. The W application also contains about 120,000 atoms and is
outside the requested moderate scale.

## Primary-source record

The method source is T. D. Swinburne and M.-C. Marinica, “Unsupervised
Calculation of Free Energy Barriers in Large Crystalline Systems,” *Physical
Review Letters* **120**, 135503 (2018),
<https://doi.org/10.1103/PhysRevLett.120.135503>. Page numbers below are the
printed article pages. The implementation source is
`tomswinburne/pafi` at commit
`1661a1d61b2ccbbd33ec4c9656d60947b8b74cc1`.

Exact numerical statements available in the article are:

- p. 135503-3: “typically 100–1000 independent trajectories,” “initially 10
  values of ξ,” “1000–4000 integration steps” for thermalization,
  “2000–5000 steps” for production, and an ensemble size “100–500” for
  acceptable error.
- Fig. 1, p. 135503-4: the displayed Fe profiles are labelled 0, 100, 300,
  500, 700, and 900 K. The text says 1000 independent simulations at 10
  temperatures up to 900 K, agreement with the harmonic approximation below
  200 K, and reversal of the harmonic stability prediction above 450 K.
  The caption says the ensemble error is not visible for ensemble size above
  300.
- Fig. 2 and pp. 135503-4–5: the W double-kink NEB formation energy is
  exactly 1.67 eV for a system of about 120,000 atoms. The displayed profiles
  are labelled 0, 200, 400, 600, 800, and 1000 K. The text reports a velocity
  boost “over ×10³” at 800 K and anharmonic corrections increasing from
  around 200 K.

The public repository, not the PRL, supplies a runnable 249-atom W-vacancy
example. `doc/test_output.png` prints, exactly:

| source case | temperature (K) | source ΔF (eV) | this work ΔF (eV) | status |
|---|---:|---:|---:|---|
| public W vacancy | 0 | 1.97 ± 0.00 | 1.92446638 ± 0.00000000 | **fail** in moderate smoke |
| public W vacancy | 1000 | 2.07 ± 0.04 | 2.07190421 ± 0.01701695 | pass in moderate smoke |
| public W vacancy | 2000 | 1.82 ± 0.09 | 1.81574906 ± 0.03376184 | pass in moderate smoke |
| PRL W double kink, Fig. 2 | 0 | 1.67 (NEB) | not run | different, unavailable 120,000-atom case |
| PRL Fe C15, Fig. 1 | 0–900 | no numerical table | not run | path and configuration unavailable |

The public `tests/test_pafi.py` pins the nine Fe-SIA gradients
`[0, 0.614145, 0.846192, 0.592260, 0.089912, -0.426347, -0.853405,
-0.792713, 0]`. Default CI replays those exact values and obtains the public
cubic-remesh barrier `0.27622876229655097 eV` and end difference
`0.003824869324197502 eV`. This checks the path interpolation and integration;
it deliberately does not claim an independent EAM calculation.

## Protocol and implementation

| item | PRL protocol | this reproduction |
|---|---|---|
| path | NEB, then a spline for every atomic coordinate (p. 135503-2) | rgsaddle NEB, initialized by the nine public W frames, then a not-a-knot cubic spline |
| hyperplanes | initially 10 equally spaced ξ values; add planes where the mean force is large | 9 equally spaced planes, the public example's default; this is a known difference from the unavailable PRL applications |
| thermostat | projected overdamped Brownian dynamics, Eq. (9) | `PafiIntegrator::EulerMaruyama`, projected Gaussian Brownian increments |
| thermalization | 1000–4000 steps | 2000 steps |
| production | 2000–5000 steps | 5000 steps |
| ensemble | typically 100–1000 trajectories; 100–500 acceptable | 100 independent trajectories |
| temperatures | application-dependent sets shown above | 0, 1000, and 2000 K, because those are the public W-example oracle |
| potential path | EAM | `rgpot` profile session backed by the canonical `lammpc` LAMMPS embed |

The measured rows above are a smoke with four trajectories, 1000
thermalization steps, and 2000 production steps. Rgsaddle NEB converged in
72 steps at a maximum force of 0.0008281241305277202 eV/Å; no basin stretch
was rejected. The release test, including NEB and both finite-temperature
rows, took 271 seconds after the native dependencies were installed. The smoke
intentionally used fewer trajectories than the article's stated acceptable
range and is not substituted for the full campaign.

The 0 K difference is 0.0455336205992234 eV and has no sampling uncertainty.
It therefore cannot be explained by the finite-temperature ensemble size.
The observed cause is the barrier on the converged rgsaddle NEB path; the
public image does not record the NEB implementation, convergence criterion,
or provenance of the bundled frames, so the remaining source of the path
difference cannot be resolved from the published materials.

The public W example is not itself the paper protocol:
`examples/input_python.py` requests underdamped Langevin
(`OverDamped = 0`), one repeat, and 2000 thermalization plus 2000 sampling
steps. Its image title says “Short (10 ps),” but does not record the worker
count or all parameters that produced the error bars. The ignored regression
uses the paper-range ensemble and lengths, so agreement tests the published
equilibrium values rather than bitwise trajectory identity.

Two estimator variants are explicit:

- `PafiMeanForce::ReferenceCode` is the default and records
  `(∇E·X′) ψ`, matching LAMMPS `fix pafi` and the public driver.
- `PafiMeanForce::PublishedEquation8` adds the analytic
  `kT ∂ξ ln(|ψ|/|X′|)` term in PRL Eq. (8), using the spline's analytic
  one-sided third derivative, and rejects violation of Eq. (6).

The public W numbers are tested with `ReferenceCode`, because that is the
implementation that produced them. This distinction was absent from earlier
rgsaddle PAFI.

The pinned lammpc embed exposed three generic owning-library defects: C helper
implementations were linked into a C++ translation unit without `extern "C"`
declarations; its capability schema identifier omitted the canonical `0x`
prefix required by rgpot; and its direct LAMMPS setup did not allocate
per-type atom arrays before assigning masses. The separate lammpc mboxes
under `patches/` fix those ABI and setup defects. The first passes lammpc's
four stub/schema tests; the moderate EAM smoke exercises both patches through
the real embed. No potential code is copied into rgsaddle.

## Audit of pull request 51

PR 51 (`feat/pafi-repro-land`, head
`baa493774ecc50abb32d7576eac8c98062b1dbb1`) did useful work:

- it matched the public Fe-SIA nine-gradient fixture within `0.001`;
- it sampled the public W-vacancy example at 0, 1000, and 2000 K on 9 and 33
  planes and compared rgsaddle with the public driver;
- it introduced the not-a-knot spline, basin test, and
  Leimkuhler–Matthews sampler used here.

It did **not** reproduce either PRL application, assert the three W README
values, run the path through rgsaddle NEB, or use the paper's direct
Euler–Maruyama variant. Its test-only `tests/common/eam_fs.rs` was an
independent 407-line EAM implementation, which violates this campaign's
canonical-PES requirement. This series keeps the sound generic PAFI changes,
removes that local potential route, and evaluates EAM only through
`rgpot`/`lammpc`.

## Statistical acceptance

For each public W row, the ignored test requires

`|ΔF_ours - ΔF_source| ≤ 2 sqrt(σ_ours² + σ_source² + (0.005 eV)²)`.

The last term is half of the source image's 0.01 eV display unit. The two
standard errors give a predeclared approximately 95% sampling band; this is
the derived regression tolerance, not a hand-tuned absolute threshold.

The BenchBot question is in `QUESTION.md`. A full run must add its generated
CSV values separately from the smoke values above and retain any failure
rather than widening the acceptance band.
