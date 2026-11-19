# Jóhannesson–Jónsson Section III B reproduction

Source: G. H. Jóhannesson and H. Jónsson, “Optimization of hyperplanar
transition states,” *J. Chem. Phys.* **115**, 9644–9656 (2001),
doi:10.1063/1.1415499. The checked author-hosted PDF is
<https://hj.hi.is/papers/paperOHTST-JCP01.pdf> (SHA-256
`7392008f98919d708db90c4c07aa4ea076261dea5e8b6ae00d7fc8e57188c274`).
Page numbers below are the printed journal pages.

## What the paper states

Potential (Appendix B, pp. 9655–9656):

> “The two-dimensional potentials used in the test calculations shown in
> Figs. 6(a)–6(d) are based on a LEPS plus harmonic oscillator potential,
> \(V_{\mathrm{LEPS+HO}}\), shown in Fig. 6(a).”

Equation (B1), p. 9655, is

\[
V_{\mathrm{LEPS}}(r_{AB},r_{BC}) =
\frac{Q_{AB}}{1+a}+\frac{Q_{BC}}{1+b}+\frac{Q_{AC}}{1+c}
-\left[
\frac{J_{AB}^2}{(1+a)^2}+\frac{J_{BC}^2}{(1+b)^2}
+\frac{J_{AC}^2}{(1+c)^2}
-\frac{J_{AB}J_{BC}}{(1+a)(1+b)}
-\frac{J_{BC}J_{AC}}{(1+b)(1+c)}
-\frac{J_{AB}J_{AC}}{(1+a)(1+c)}
\right]^{1/2}.
\]

The pair terms are, exactly as Eqs. (B2) and (B3), p. 9655,

\[
Q(r)=\frac d2\left(\frac32e^{-2\alpha(r-r_0)}
-e^{-\alpha(r-r_0)}\right),
\qquad
J(r)=\frac d4\left(e^{-2\alpha(r-r_0)}
-6e^{-\alpha(r-r_0)}\right).
\]

The parameter sentence on p. 9655 states:

> “The parameters used in the calculations presented here are
> \(a=c=0.05\), \(b=0.80\),
> \(r_{0,AB}=r_{0,BC}=r_{0,AC}=0.742\),
> \(\alpha_{AB}=\alpha_{BC}=\alpha_{AC}=1.942\),
> \(d_{AB}=d_{BC}=4.746\), and \(d_{AC}=3.445\).”

The coupling is Eq. (B4), p. 9656,

\[
V_{\mathrm{LEPS+HO}}(x,y)=
V_{\mathrm{LEPS}}(x,r_{AC}-x)
+2k_c\left[x-\left(r_{AC}/2-y/c\right)\right]^2,
\]

followed by:

> “where \(k_c=0.2025\), \(r_{AC}=3.742\), and \(c=1.154\).”

Here the second \(c\) is the oscillator coordinate scale, distinct from the
LEPS parameter \(c=0.05\). The paper then states on p. 9656 that
“The coordinate \((x,y)=(2.02083,-0.172881)\) corresponds to the first
order saddle point on the \(V_{\mathrm{LEPS+HO}}\) surface.”

Temperature and barrier convention:

* Section III A, p. 9650, says: “the system was simulated at a very low
  temperature, \(k_BT=10^{-7}E_A\), where \(E_A\) is the energy of the
  saddle point with respect to \(R\).”
* Section III B, p. 9651, says: “The two-dimensional potential surface
  \(V_a\) shown in Fig. 6(a) was used to calculate the free energy barrier
  at a temperature of \(k_BT=0.1E_A\).”
* The same paragraph on p. 9651 says:
  “The ratio of the configuration integrals,
  \(Q^\ddagger/Q_{Z_R}\) was found to be \(0.765\pm0.005\) for \(V_a\) at
  \(k_BT=0.1E_A\). The value of \(\langle|v|\rangle\) was found to be
  0.399, so the rate constant is estimated to be
  \(k_{\mathrm{OH-TST}}=(4.5\pm0.2)\times10^{-7}\).”
* It continues: “harmonic TST gives a very similar value for the rate
  constant \(k_{\mathrm{hTST}}=4.3\times10^{-7}\).”

Velocity and mass are Eqs. (23) and (24), p. 9650:

\[
\frac{\langle|v|\rangle}{2}=\sqrt{\frac{k_BT}{2\pi m}},
\qquad
m=\sum_i m_i(n_x^2+n_y^2+n_z^2).
\]

The text calls \(m\) “the effective mass for motion across the dividing
surface in the direction of the normal.” The paper does **not** specify the
individual masses \(m_i\), a numerical effective mass, or physical units for
the Appendix B parameters or rates. It also does not print a numerical
\(E_A\).

Sampling details that are actually stated are on p. 9650:

> “In all calculations of canonical ensemble averages we use a single
> molecular dynamics trajectory. We simulate a system at a given
> temperature by applying stochastic collisions to the trajectory.”

Equation (22), p. 9649, estimates \(Q_{Z_R}/Q_R\) from trajectory crossings,
weighted by the inverse normal displacement. Section III A, p. 9650, says
“The integral was evaluated by using the trapezoidal rule.” The paper does not
state the trajectory length, time step, collision frequency, number of
samples, or number of work planes.

## Term-by-term implementation check

`tests/common/ohtst_surfaces.rs::LepsHo` matches Eqs. (B1)–(B4):

| Paper term | Implementation |
|---|---|
| \(r_{AB}=x,\ r_{BC}=r_{AC}-x\) | `x`, `R_AC - x` |
| three \(Q/(1+a,b,c)\) terms | `qab/sa + qbc/sb + qac/sc` |
| six-term exchange square root | `s = u²+w²+z²-uw-wz-uz; sqrt(s)` |
| \(Q(r)\), \(J(r)\) | `coulomb_exchange`, including analytic derivatives |
| \(2k_c[x-(r_{AC}/2-y/c)]^2\) | `2.0*K_C*stretch*stretch` |
| all Appendix B parameters | exact decimal values listed above |

No potential mismatch was found. The Vineyard hTST rate uses the analytic
Hessian of those same terms (`LepsHo::hessian`). The quadrature references
in `tests/ohtst_leps.rs` locate the optimal line by Newton's method on
central differences of the quadrature free energy; they are test-only
cross-checks and no sampled rate depends on them.

## Regenerating every number

Every number below that is not a quotation from the paper is printed by one
command from a clean checkout of this repository, at the revision that
contains this file. The paper quotations can be checked against the PDF
named at the top:

```bash
curl -L -o paperOHTST-JCP01.pdf https://hj.hi.is/papers/paperOHTST-JCP01.pdf
sha256sum paperOHTST-JCP01.pdf
```

The reproduction is one integration test. It takes a few seconds after the
crate has been built:

```bash
cargo test --features ohtst --test ohtst_leps section_iii_b_reproduces_the_printed_rates -- --nocapture
```

`--nocapture` is required, because the test reports through standard
error. It prints these lines, and cargo then reports the test as `ok`:

```text
Section III B potential: E_A=3.6339513165591186, reactant=[0.7415206639943598, 1.303418963134847], saddle=[2.0208277343663372, -0.17290120545875307]
Section III B prose convention: kT=0.1 E_A=0.36339513165591186, <|v|>=0.4809828749490712, k_hTST=4.041014867543564e-5, 0.25 / kT=0.6879563819713403
Section III B printed-rate convention: kT=0.25, <|v|>=0.3989422804014327, k_hTST=4.331269928384644e-7, quadrature=4.201484800516092e-7
Section III B seed 1: k_OH-TST=4.5280923324556635e-7
Section III B seed 2: k_OH-TST=4.560986734779895e-7
Section III B seed 3: k_OH-TST=3.9261333126697063e-7
Section III B seed 4: k_OH-TST=4.4348413057494177e-7
Section III B seed 5: k_OH-TST=3.6507495025465244e-7
Section III B seed 6: k_OH-TST=4.7465717146380427e-7
Section III B seed 7: k_OH-TST=4.2821212936180344e-7
Section III B seed 8: k_OH-TST=4.5904921118016843e-7
Section III B seed 9: k_OH-TST=4.962147098558373e-7
Section III B seed 10: k_OH-TST=4.1531516315560214e-7
Section III B seed 11: k_OH-TST=4.583473329168407e-7
Section III B seed 12: k_OH-TST=4.216168504271633e-7
Section III B seed 13: k_OH-TST=4.317968290100397e-7
Section III B seed 14: k_OH-TST=4.488500486846323e-7
Section III B seed 15: k_OH-TST=4.2634715638012876e-7
Section III B seed 16: k_OH-TST=4.3311347874850217e-7
Section III B k_OH-TST over 16 seeds: mean=4.377250250002902e-7, sample sd=3.1345620996686165e-8, standard error=7.836405249171541e-9, lowest=3.6507495025465244e-7, highest=4.962147098558373e-7
Section III B against (4.5e-7 ± 2e-8): mean - published=-1.227497499970977e-8 (-0.6137487499854885 of the published error, -1.566403804985388 standard errors), mean - lower edge=7.72502500029025e-9 (0.985786818657309 standard errors), upper edge - mean=3.227497499970979e-8
```

The seeds are fixed (1 to 16), so the output repeats exactly on the same
toolchain. Setting `RAYON_NUM_THREADS=1` gave identical output. A different
compiler or math library may change the trailing digits of the sampled
rates.

Where the sections below quote a number, it is the printed value rounded to
the digits shown. The paper's own values (0.399, 4.3, 4.5 ± 0.2, 0.765 ± 0.005)
are quotations, not output.

## Convention decision

Evaluating Appendix B gives, in the first line of the output, the reactant
minimum \((0.74152,1.30342)\), the printed saddle, and \(E_A=3.63395\) in
the potential's own energy unit. Thus the literal \(k_BT=0.1E_A=0.363395\),
with unit mass, gives \(\langle|v|\rangle=0.480983\) and
\(k_{\mathrm{hTST}}=4.0410\times10^{-5}\) (second line), not the printed
values.

Both independently printed quantities select \(k_BT=0.25\) in the Appendix B
energy unit with unit mass (third line):

* Eq. (23): \(\sqrt{2k_BT/\pi}=0.398942\);
* analytic Vineyard hTST: \(4.33127\times10^{-7}\).

Therefore the numerical convention behind the printed rates is
\(k_BT=0.25\), unit mass, and the potential's own energy, length, and time
units. The prose \(0.1E_A\) is inconsistent with Appendix B by a factor
\(0.25/(0.1E_A)=0.688\) (the last field of the second line). The printed
\(Q^\ddagger/Q_{Z_R}=0.765\) is also mislabeled: Eq. (3), p. 9646, uses
\(Q_{Z_R}/Q_R\) in the prefactor, while \(Q^\ddagger/Q_{Z_R}\) is the
exponentially small reversible-work factor.

## How the rate is obtained from the sixteen seeds

The paper omits all sample counts, so the test declares its own. For each
seed \(s=1,\dots,16\) it makes three independent samples, each with the
BAOAB thermostat at \(k_BT=0.25\), time step 0.1, friction 1, 300 burn-in
steps and a stride of 5 (`sampler` in `tests/ohtst_leps.rs`), seeded with
\(s\):

1. plane optimization from the dimer's saddle plane: 1000 records per
   sample, at most 1000 records in all (`with_max_records`);
2. the Eq. (22) crossing estimate \(\rho_s\) of \(Q_{Z_R}/Q_R\) on the
   guideline plane through the reactant minimum, from one trajectory of
   8000 records, with the optimized plane as the wall;
3. the reversible work \(W_s\) from that plane to the optimized plane over
   17 work planes of 2000 records each, integrated by the trapezoidal rule
   (`with_simpson(false)`), as in the paper.

The seed's rate is

\[
k_s=\sqrt{\frac{k_BT}{2\pi}}\;\rho_s\,e^{-W_s/k_BT},
\]

computed by `rate_from_parts`. The quoted rate is the mean of the sixteen
\(k_s\). The error is the standard error of that mean, which is the sample
standard deviation (denominator \(n-1\)) divided by \(\sqrt{16}\). The
per-seed errors of \(\rho_s\) and \(W_s\) are not propagated into it.
BAOAB samples the same canonical distribution as the paper's Andersen
collisions; an exact collision schedule cannot be reconstructed because the
paper does not give its time step or collision frequency.

Measured per-seed rates, in units of \(10^{-7}\):

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| \(k_s\) | 4.528 | 4.561 | 3.926 | 4.435 | 3.651 | 4.747 | 4.282 | 4.590 |

| Seed | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| \(k_s\) | 4.962 | 4.153 | 4.583 | 4.216 | 4.318 | 4.489 | 4.263 | 4.331 |

The spread between seeds is large: the sample standard deviation is
\(0.313\times10^{-7}\) and the seeds range from \(3.651\times10^{-7}\) to
\(4.962\times10^{-7}\). The mean is \(4.377\times10^{-7}\) with standard
error \(0.078\times10^{-7}\), so a single seed is a poor estimate and the
mean over sixteen is what carries the result.

## Rates

| Quantity | Result | Published |
|---|---:|---:|
| \(\langle|v|\rangle\) | 0.398942 | 0.399 |
| analytic Vineyard \(k_{\mathrm{hTST}}\) | \(4.33127\times10^{-7}\) | \(4.3\times10^{-7}\) |
| optimal-line quadrature \(k_{\mathrm{OH-TST}}\) | \(4.20148\times10^{-7}\) | — |
| Eq. (22) + trapezoidal work \(k_{\mathrm{OH-TST}}\), mean of 16 seeds | \((4.377\pm0.078)\times10^{-7}\) | \((4.5\pm0.2)\times10^{-7}\) |

The quoted uncertainty is the standard error of the mean, not the spread
of one seed.

## Where the mean sits relative to the published rate

The mean, \(4.377\times10^{-7}\), is below the published central value
\(4.5\times10^{-7}\) by \(0.123\times10^{-7}\). That is 0.61 of the
published \(\pm0.2\times10^{-7}\) and 1.57 of this estimate's standard
error. It is inside the published interval \([4.3,4.7]\times10^{-7}\),
\(0.077\times10^{-7}\) above its lower edge (0.99 standard errors) and
\(0.323\times10^{-7}\) below the upper edge. Six seeds (3, 5, 7, 10, 12
and 15) lie below \(4.3\times10^{-7}\) and two (seeds 6 and 9) lie above
\(4.7\times10^{-7}\), so the interval describes the mean and not the
individual seeds. The result is therefore compatible with the published
rate, with the sampled mean on its low side; it does not reproduce the
central value, and the data here cannot say whether the 0.123 gap is a
difference in method or sampling noise. The paper does not say what its
\(\pm0.2\times10^{-7}\) measures.

The optimal-line quadrature, \(4.20148\times10^{-7}\), is below the
interval. It is the converged rate for this line and these conventions, so
it is not itself a sample of the published value, and the sampled mean lies
\(0.176\times10^{-7}\) above it (the difference of the two printed
values), which is 2.24 standard errors.

Because the estimator's population mean sits right at the \(4.3\times10^{-7}\)
lower edge (noiseless 17-node trapezoidal work gives \(4.3147\times10^{-7}\)),
an unwidened check rejects valid multi-seed means roughly half the time as the
seed count grows. The test therefore widens the published interval by twice
this estimate's standard error (`SE_COVERAGE = 2.0`, ~95% normal coverage):
`(band_low - SE_COVERAGE * error..=band_high + SE_COVERAGE * error).contains(&mean)`.

## Other checks

Run the suites with

```bash
cargo test --features capi,ohtst
cargo test --features ohtst --test ohtst_leps
```

Both must report zero failed tests. The C and C++ ABI smoke programs are
among the Rust integration tests. The Python binding has its own tests in
`tests/python/test_ohtst.py`, run with `pytest tests/python` after
`pip install .`, and the two Meson consumer modes are exercised by
`ci/meson_consumer.sh subproject target/debug/librgsaddle.so` and
`ci/meson_consumer.sh pkgconfig target/debug/librgsaddle.so` after
`cargo build --features capi`. No test counts are quoted here: they change
with every test added, and the commands above print the current ones.
