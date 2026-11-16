//! OH-TST in 48 dimensions: a double well along one mode coupled to 47
//! anharmonic modes, in a rotated frame,
//!
//! `V = V₀ (q₁² - 1)² + Σ_j [(½ ω_j² + c_j q₁) q_j² + ¼ d q_j⁴]`, `x = Q q`,
//!
//! with `V₀ = 2`, `d = 0.5`, `ω_j² = 2 + 18 {j φ}`, `c_j = 0.12 ω_j² {j √2}`
//! for `j = 1..47`, `{·}` the fractional part and `φ = (√5 - 1)/2`, and
//! `Q` the product of two Householder reflections. This is the family of
//! couplings a working note on OH-TST used. Its random draw is not
//! available, so the parameters here are fixed sequences instead.
//!
//! `V` is even in every `q_j`, so the torque on any plane `q₁ = s`
//! vanishes and the optimal plane is one of them. Its partition function
//! is a product of one-dimensional integrals, and `Z_R` one more integral
//! over `q₁`: the references are exact quadrature.

use ndarray::{Array1, ArrayView1};
use rgsaddle::{
    AnharmonicConfig, HarmonicReference, Hyperplane, LangevinConfig, PointSurface, SaddleError,
    anharmonic_correction, sample_plane,
};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/ohtst_surfaces.rs"
));

use ohtst_surfaces::simpson;

const N: usize = 48;
const V0: f64 = 2.0;
const QUARTIC: f64 = 0.5;

struct CoupledModes {
    omega2: Vec<f64>,
    coupling: Vec<f64>,
    reflections: [Array1<f64>; 2],
}

impl CoupledModes {
    fn new() -> Self {
        let phi = 0.5 * (5.0f64.sqrt() - 1.0);
        let frac = |v: f64| v - v.floor();
        let omega2: Vec<f64> = (1..N).map(|j| 2.0 + 18.0 * frac(j as f64 * phi)).collect();
        let coupling = (1..N)
            .map(|j| 0.12 * omega2[j - 1] * frac(j as f64 * 2.0f64.sqrt()))
            .collect();
        let unit = |v: Array1<f64>| &v / v.dot(&v).sqrt();
        let first = unit((0..N).map(|i| (1.3 * i as f64 + 0.4).sin() + 0.2).collect());
        let second = unit((0..N).map(|i| (0.7 * (i * i) as f64 + 1.1).cos()).collect());
        Self {
            omega2,
            coupling,
            reflections: [first, second],
        }
    }

    fn reflect(h: &Array1<f64>, v: &mut Array1<f64>) {
        let c = 2.0 * h.dot(v);
        v.scaled_add(-c, h);
    }

    /// `Q v = H₁ H₂ v`.
    fn rotate(&self, mut v: Array1<f64>) -> Array1<f64> {
        Self::reflect(&self.reflections[1], &mut v);
        Self::reflect(&self.reflections[0], &mut v);
        v
    }

    /// `Qᵀ v = H₂ H₁ v`.
    fn unrotate(&self, mut v: Array1<f64>) -> Array1<f64> {
        Self::reflect(&self.reflections[0], &mut v);
        Self::reflect(&self.reflections[1], &mut v);
        v
    }

    /// The direction of `q₁` in Cartesian coordinates.
    fn axis(&self) -> Array1<f64> {
        let mut e = Array1::zeros(N);
        e[0] = 1.0;
        self.rotate(e)
    }

    /// `∫ exp(-[(½ ω_j² + c_j s) q² + ¼ d q⁴] / kT) dq` by the trapezoid
    /// rule, spectrally accurate for this smooth bump.
    fn mode_integral(&self, j: usize, s: f64, kt: f64) -> f64 {
        let quadratic = 0.5 * self.omega2[j] + self.coupling[j] * s;
        let width = (kt / (2.0 * quadratic)).sqrt();
        let points = 80;
        let half = 8.0 * width;
        let h = 2.0 * half / points as f64;
        (0..=points)
            .map(|i| {
                let q = -half + h * i as f64;
                let weight = if i == 0 || i == points { 0.5 } else { 1.0 };
                weight * (-(quadratic * q * q + 0.25 * QUARTIC * q.powi(4)) / kt).exp()
            })
            .sum::<f64>()
            * h
    }

    /// `ln Z(s)` of the plane `q₁ = s`, energies from the saddle.
    fn log_plane(&self, s: f64, kt: f64) -> f64 {
        let well = V0 * (s * s - 1.0).powi(2) - V0;
        -well / kt
            + (0..N - 1)
                .map(|j| self.mode_integral(j, s, kt).ln())
                .sum::<f64>()
    }

    /// The plane `q₁ = s` of largest free energy, by golden section.
    fn optimal_offset(&self, kt: f64) -> f64 {
        let ratio = 0.5 * (5.0f64.sqrt() - 1.0);
        let (mut a, mut b) = (-0.5, 0.5);
        while b - a > 1.0e-10 {
            let c = b - ratio * (b - a);
            let d = a + ratio * (b - a);
            if self.log_plane(c, kt) < self.log_plane(d, kt) {
                b = d;
            } else {
                a = c;
            }
        }
        0.5 * (a + b)
    }

    /// `ln k = ln(sqrt(kT/2π) Z(s) / Z_R(s))` with `Z_R` the integral of
    /// `Z` over `q₁ < s`.
    fn log_rate(&self, s: f64, kt: f64) -> f64 {
        let peak = self.log_plane(-1.0, kt);
        let scaled = |q: f64| (self.log_plane(q, kt) - peak).exp();
        let behind = simpson(&scaled, -2.5, s, 1.0e-9);
        0.5 * (kt / (2.0 * std::f64::consts::PI)).ln() + self.log_plane(s, kt) - peak - behind.ln()
    }

    /// Harmonic TST: `λ₁ = 8 V₀` and `ω_j² - 2 c_j` at the minimum,
    /// `ω_j²` across the saddle.
    fn log_harmonic_rate(&self, kt: f64) -> f64 {
        let reactant: f64 = (8.0 * V0).ln()
            + (0..N - 1)
                .map(|j| (self.omega2[j] - 2.0 * self.coupling[j]).ln())
                .sum::<f64>();
        let saddle: f64 = self.omega2.iter().map(|w| w.ln()).sum();
        0.5 * (reactant - saddle) - (2.0 * std::f64::consts::PI).ln() - V0 / kt
    }

    /// `A‡`, `A_R`, `S` and `c₁` from the contractions. At the minimum
    /// `V₁₁₁ = -24 V₀`, `V₁₁₁₁ = 24 V₀`, `V₁jj = 2 c_j`, `V_jjjj = 6 d`.
    fn contractions(&self) -> (f64, f64, f64, f64) {
        let modes = 0..N - 1;
        let plane: f64 = modes
            .clone()
            .map(|j| -6.0 * QUARTIC / (8.0 * self.omega2[j].powi(2)))
            .sum();
        let slope: f64 = modes
            .clone()
            .map(|j| 2.0 * self.coupling[j] / self.omega2[j])
            .sum();
        let (v111, v1111, mu1) = (-24.0 * V0, 24.0 * V0, 8.0 * V0);
        let mut reactant = -v1111 / (8.0 * mu1 * mu1) + v111 * v111 / (12.0 * mu1.powi(3));
        let mut trace = v111 / mu1;
        for j in modes {
            let mu = self.omega2[j] - 2.0 * self.coupling[j];
            let k = 2.0 * self.coupling[j];
            reactant += -6.0 * QUARTIC / (8.0 * mu * mu) + 3.0 * k * k / (12.0 * mu1 * mu * mu);
            trace += k / mu;
        }
        reactant += trace * trace / (8.0 * mu1);
        let coefficient = plane - reactant - slope * slope / (8.0 * 4.0 * V0);
        (plane, reactant, slope, coefficient)
    }
}

impl PointSurface for CoupledModes {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let q = self.unrotate(x.to_owned());
        let q1 = q[0];
        let mut energy = V0 * (q1 * q1 - 1.0).powi(2);
        let mut g = Array1::zeros(N);
        g[0] = 4.0 * V0 * q1 * (q1 * q1 - 1.0);
        for j in 1..N {
            let (w, c, qj) = (self.omega2[j - 1], self.coupling[j - 1], q[j]);
            energy += (0.5 * w + c * q1) * qj * qj + 0.25 * QUARTIC * qj.powi(4);
            g[0] += c * qj * qj;
            g[j] = (w + 2.0 * c * q1) * qj + QUARTIC * qj.powi(3);
        }
        Ok((energy, self.rotate(g)))
    }
}

#[test]
fn coupled_modes_gradient_and_frame() {
    let model = CoupledModes::new();
    let x: Array1<f64> = (0..N).map(|i| 0.1 * (0.37 * i as f64).sin()).collect();
    let (_, g) = model.eval(x.view()).unwrap();
    let h = 1.0e-6;
    for i in [0, 7, 23, 47] {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[i] += h;
        minus[i] -= h;
        let fd =
            (model.eval(plus.view()).unwrap().0 - model.eval(minus.view()).unwrap().0) / (2.0 * h);
        assert!((fd - g[i]).abs() < 1.0e-7, "{fd} against {}", g[i]);
    }
    let back = model.unrotate(model.rotate(x.clone()));
    assert!((&back - &x).iter().all(|v| v.abs() < 1e-14));
    let axis = model.axis();
    assert!((axis.dot(&axis) - 1.0).abs() < 1e-14);
}

/// The first-order coefficients from Hessian actions in 47 free
/// directions match the contractions within four errors. Against exact
/// quadrature, `kT S / (2|λ_u|)`, `c₁ kT` and `(A‡ - A_R) kT` are within
/// `0.5 kT²` of the optimal offset and of `ln(k / k_HTST)` for the optimal
/// and the saddle plane at `kT` = 0.05 and 0.1. With these 1000 probes
/// every coefficient came out within one error of its contraction.
#[test]
fn coupled_modes_first_order_correction() {
    let model = CoupledModes::new();
    let (plane, reactant, slope, coefficient) = model.contractions();
    let axis = model.axis();
    let saddle = Hyperplane::new(Array1::zeros(N), axis.clone()).unwrap();
    let minimum = -&axis;
    let config = AnharmonicConfig::default().with_probes(1000).with_seed(6);
    let result = anharmonic_correction(&model, minimum.view(), &saddle, &config).unwrap();
    assert!((result.curvature - 4.0 * V0).abs() < 1e-5);
    for (name, value, error, exact) in [
        ("A‡", result.saddle, result.saddle_error, plane),
        ("A_R", result.reactant, result.reactant_error, reactant),
        ("S", result.slope, result.slope_error, slope),
        (
            "c₁",
            result.coefficient,
            result.coefficient_error,
            coefficient,
        ),
    ] {
        assert!(
            (value - exact).abs() < 4.0 * error + 1.0e-6,
            "{name} {value} ± {error} against {exact}"
        );
        assert!(error < 0.05 * exact.abs(), "{name} error {error}");
    }
    for kt in [0.05, 0.1] {
        let bound = 0.5 * kt * kt;
        let s = model.optimal_offset(kt);
        let harmonic = model.log_harmonic_rate(kt);
        let optimal = model.log_rate(s, kt) - harmonic;
        let fixed = model.log_rate(0.0, kt) - harmonic;
        let (value, error) = result.log_ratio(kt);
        assert!(
            (value - optimal).abs() < bound + 4.0 * error,
            "kT {kt}: c₁ kT {value} ± {error} against {optimal}"
        );
        let (value, error) = result.fixed_log_ratio(kt);
        assert!(
            (value - fixed).abs() < bound + 4.0 * error,
            "kT {kt}: (A‡ - A_R) kT {value} ± {error} against {fixed}"
        );
        let spread = kt * result.slope_error / (2.0 * result.curvature);
        assert!((result.offset(kt) - s).abs() < bound + 4.0 * spread);
    }
}

/// At `kT = 0.2` the sampler on the exact optimal plane sees no force and
/// no torque, and the virial is `47 kT`. On the saddle plane the normal
/// force matches `kT d ln Z / ds` from quadrature.
#[test]
fn coupled_modes_planes_against_quadrature() {
    let model = CoupledModes::new();
    let kt = 0.2;
    let axis = model.axis();
    let config = LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(0.1)
        .with_friction(1.0)
        .with_burn_in(300)
        .with_stride(5)
        .with_records(4000)
        .with_seed(4);
    let optimum = model.optimal_offset(kt);
    let plane = Hyperplane::new(&axis * optimum, axis.clone()).unwrap();
    let sample = sample_plane(&model, &plane, None, &config).unwrap();
    assert!(
        sample.normal_force.abs() < 4.0 * sample.normal_force_error,
        "force {} ± {}",
        sample.normal_force,
        sample.normal_force_error
    );
    let largest = sample
        .centroid_torque
        .iter()
        .zip(&sample.centroid_torque_error)
        .map(|(t, e)| (t / e).abs())
        .fold(0.0, f64::max);
    assert!(largest < 4.5, "largest torque {largest} errors");
    assert_eq!(sample.free_dimensions, N - 1);
    assert!(
        (sample.virial - (N - 1) as f64 * kt).abs() < 4.0 * sample.virial_error,
        "virial {} ± {}",
        sample.virial,
        sample.virial_error
    );

    let saddle = Hyperplane::new(Array1::zeros(N), axis).unwrap();
    let sample = sample_plane(&model, &saddle, None, &config).unwrap();
    let h = 1.0e-4;
    let exact = kt * (model.log_plane(h, kt) - model.log_plane(-h, kt)) / (2.0 * h);
    assert!(
        (sample.normal_force - exact).abs() < 4.0 * sample.normal_force_error,
        "force {} ± {} against {exact}",
        sample.normal_force,
        sample.normal_force_error
    );
    assert!(sample.normal_force_error < 0.1 * exact.abs());
}

/// Control variates from the saddle Hessian on planes turned off the
/// symmetric ones. On the saddle plane itself `h = P H₀ ν` vanishes and
/// they change nothing. Turned by 0.3 rad they remove the part of the
/// fluctuations linear in the in-plane coordinates: the force error falls
/// to the floor of the untilted plane, set by the bath terms quadratic in
/// `q_j`, and the torque errors fall several times over. Both estimates
/// stay within the errors of the uncontrolled ones.
#[test]
fn coupled_modes_control_variates() {
    let model = CoupledModes::new();
    let kt = 0.2;
    let axis = model.axis();
    let config = LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(0.1)
        .with_burn_in(300)
        .with_stride(5)
        .with_records(4000)
        .with_seed(4)
        .with_control(HarmonicReference::new(Array1::zeros(N)));
    let mut e = Array1::zeros(N);
    e[1] = 1.0;
    let across = model.rotate(e);
    let flat = sample_plane(
        &model,
        &Hyperplane::new(Array1::zeros(N), axis.clone()).unwrap(),
        None,
        &config,
    )
    .unwrap();
    let raw = flat.uncontrolled.as_ref().unwrap();
    assert!((flat.normal_force - raw.normal_force).abs() < 1.0e-9);

    let tilt: f64 = 0.3;
    let normal = &axis * tilt.cos() + &across * tilt.sin();
    let tilted = sample_plane(
        &model,
        &Hyperplane::new(Array1::zeros(N), normal).unwrap(),
        None,
        &config,
    )
    .unwrap();
    let raw = tilted.uncontrolled.as_ref().unwrap();
    assert!(
        (tilted.normal_force - raw.normal_force).abs() < 4.0 * raw.normal_force_error,
        "force {} against {} ± {}",
        tilted.normal_force,
        raw.normal_force,
        raw.normal_force_error
    );
    assert!(
        tilted.normal_force_error < 0.6 * raw.normal_force_error,
        "force error {} against {}",
        tilted.normal_force_error,
        raw.normal_force_error
    );
    assert!(tilted.normal_force_error < 1.5 * flat.normal_force_error);
    let worst = tilted
        .centroid_torque
        .iter()
        .zip(&raw.centroid_torque)
        .zip(&raw.centroid_torque_error)
        .map(|((c, r), e)| (c - r).abs() / e)
        .fold(0.0, f64::max);
    assert!(worst < 4.5, "torque {worst} errors off");
    let ratio = tilted.centroid_torque_error.sum() / raw.centroid_torque_error.sum();
    assert!(ratio < 0.3, "torque error ratio {ratio}");
}
