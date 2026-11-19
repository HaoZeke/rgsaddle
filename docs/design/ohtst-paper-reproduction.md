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
Section III B seed 1: k_OH-TST=4.3790268351887016e-7
Section III B seed 2: k_OH-TST=4.500701282600099e-7
Section III B seed 3: k_OH-TST=3.9296071475525623e-7
Section III B seed 4: k_OH-TST=4.325147132210193e-7
Section III B seed 5: k_OH-TST=3.5914814740520535e-7
Section III B seed 6: k_OH-TST=4.7167335175916473e-7
Section III B seed 7: k_OH-TST=4.194552625564379e-7
Section III B seed 8: k_OH-TST=4.45125721414597e-7
Section III B seed 9: k_OH-TST=4.893217912712362e-7
Section III B seed 10: k_OH-TST=4.096034162864649e-7
Section III B seed 11: k_OH-TST=4.5808812765386034e-7
Section III B seed 12: k_OH-TST=4.2131534931699767e-7
Section III B seed 13: k_OH-TST=4.268942817235603e-7
Section III B seed 14: k_OH-TST=4.449433806175572e-7
Section III B seed 15: k_OH-TST=4.1410248976786084e-7
Section III B seed 16: k_OH-TST=4.132105919832519e-7
Section III B k_OH-TST over 16 seeds: mean=4.3039563446945936e-7, trimmed mean=4.311021788600406e-7, sample sd=3.109624783893912e-8, standard error=7.77406195973478e-9, lowest=3.5914814740520535e-7, highest=4.893217912712362e-7
Section III B against (4.5e-7 ± 2e-8): mean - published=-1.9604365530540617e-8 (-0.9802182765270309 of the published error, -2.5217660512715336 standard errors), mean - lower edge=3.956344694594032e-10 (0.05089160229344257 standard errors), upper edge - mean=3.960436553054064e-8
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

1. plane optimization from the dimer's saddle plane: a fixed sample of 1000
   records per plane (`with_norm_test(0.0)` and `with_max_records(1000)`);
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

computed by `rate_from_parts`. Alongside the sample mean of the sixteen
\(k_s\), the test computes the 12.5% trimmed mean (dropping the two lowest
and two highest order statistics). Because the Eq. (22) crossing weight
\(1/|\Delta\xi|\) has a divergent second moment at slow crossings, the
untrimmed sample standard error does not define a normal coverage band,
whereas the trimmed mean has a finite asymptotic variance whenever the
quantile density is positive.
BAOAB samples the same canonical distribution as the paper's Andersen
collisions; an exact collision schedule cannot be reconstructed because the
paper does not give its time step or collision frequency.

Measured per-seed rates, in units of \(10^{-7}\):

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| \(k_s\) | 4.379 | 4.501 | 3.930 | 4.325 | 3.591 | 4.717 | 4.195 | 4.451 |

| Seed | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| \(k_s\) | 4.893 | 4.096 | 4.581 | 4.213 | 4.269 | 4.449 | 4.141 | 4.132 |

The spread between seeds is large: the sample standard deviation is
\(0.311\times10^{-7}\) and the seeds range from \(3.591\times10^{-7}\) to
\(4.893\times10^{-7}\). Over the sixteen seeds, the mean is
\(4.304\times10^{-7}\) (sample standard error \(0.078\times10^{-7}\)) and
the 12.5% trimmed mean is \(4.311\times10^{-7}\).

## Rates

| Quantity | Result | Published |
|---|---:|---:|
| \(\langle|v|\rangle\) | 0.398942 | 0.399 |
| analytic Vineyard \(k_{\mathrm{hTST}}\) | \(4.33127\times10^{-7}\) | \(4.3\times10^{-7}\) |
| optimal-line quadrature \(k_{\mathrm{OH-TST}}\) | \(4.20148\times10^{-7}\) | — |
| Eq. (22) + trapezoidal work \(k_{\mathrm{OH-TST}}\), 16 seeds (mean; 12.5% trimmed mean) | \((4.304\pm0.078)\times10^{-7}\); \(4.311\times10^{-7}\) | \((4.5\pm0.2)\times10^{-7}\) |

## Where the mean and trimmed mean sit relative to the published rate

The mean, \(4.304\times10^{-7}\), and the 12.5% trimmed mean,
\(4.311\times10^{-7}\), both lie inside the published interval
\([4.3,4.7]\times10^{-7}\) from \((4.5\pm0.2)\times10^{-7}\) (Sec. III B,
p. 9651). The mean sits \(0.196\times10^{-7}\) below the published central
value \(4.5\times10^{-7}\) (0.98 of the published \(\pm0.2\times10^{-7}\)),
\(0.004\times10^{-7}\) above the lower edge, and \(0.396\times10^{-7}\)
below the upper edge. Because the Eq. (22) crossing estimator has infinite
variance, the test does not widen the published interval by a multiple of
the sample standard error; instead it asserts directly that both the mean
and the 12.5% trimmed mean over the fixed 16-seed set lie inside the
unwidened published interval `[band_low..=band_high]`.

The optimal-line quadrature, \(4.20148\times10^{-7}\), is below the
interval. It is the converged rate for this line and these conventions, so
it is not itself a sample of the published value.

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
