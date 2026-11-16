//! OH-TST on a skewed double well whose optimal plane shifts and turns
//! with temperature: `(x² - 1)² + ½ k₀ (1 + α tanh 2x) (y - x)²` with
//! `k₀ = 4`, `α = 0.8`, unit masses. The reactant minimum is `(-1, -1)`,
//! the saddle the origin one energy unit up, and the stiffness across the
//! valley rises toward the product, so the free energy of a line through
//! the saddle changes both along and across the barrier.
//!
//! The references are quadrature of line integrals and the third and
//! fourth derivatives of the surface at its stationary points. An
//! independent adaptive quadrature outside the crate (SciPy) maximizing
//! the full ratio `Z / Z_R` over lines gave offsets 0.0041, 0.0397 and
//! 0.1451 and turns -0.235°, -2.288° and -7.123° at `kT` = 0.01, 0.1 and
//! 0.4. Maximizing `A` alone, the criterion of Jóhannesson and Jónsson,
//! J. Chem. Phys. 115, 9644 (2001), agrees to four digits up to
//! `kT = 0.1` and gives 0.1374 and -7.359° at `kT = 0.4`. The rate differs
//! by 2e-4 in `ln k` there.

use ndarray::array;
use rgsaddle::{
    AnharmonicConfig, Hyperplane, LangevinConfig, MinModeConfig, MinModeSession, MinModeStatus,
    PlaneOptimizer, PlaneOptimizerConfig, ReversibleWorkConfig, anharmonic_correction, ohtst_rate,
    reactant_density, reversible_work,
};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/ohtst_surfaces.rs"
));
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/plane_manifold.rs"
));

use ohtst_surfaces::{
    SKEWED_ALPHA, SKEWED_K0, SkewedWell, descend, eigen2, half_plane_integral, hessian,
    line_free_energy, stationary,
};
use plane_manifold::PointSphereFixture;

const REACTANT: [f64; 2] = [-1.0, -1.0];
const SADDLE_ENERGY: f64 = 1.0;
const EXTENT: f64 = 8.0;

/// Saddle eigenvalues `2 ∓ 2√5`; the unstable mode is along
/// `(2, √5 - 1)`.
fn saddle_eigenvalues() -> (f64, f64) {
    let root = 2.0 * 5.0f64.sqrt();
    (2.0 - root, 2.0 + root)
}

fn saddle_angle() -> f64 {
    ((5.0f64.sqrt() - 1.0) / 2.0).atan()
}

/// Reactant Hessian `[[8 + k, -k], [-k, k]]` with `k = k₀ (1 - α tanh 2)`.
fn reactant_eigenvalues() -> (f64, f64) {
    let k = SKEWED_K0 * (1.0 - SKEWED_ALPHA * 2.0f64.tanh());
    let (trace, det) = (8.0 + 2.0 * k, 8.0 * k);
    let root = (trace * trace - 4.0 * det).sqrt();
    (0.5 * (trace - root), 0.5 * (trace + root))
}

fn unit(theta: f64) -> [f64; 2] {
    [theta.cos(), theta.sin()]
}

fn off_saddle(offset: f64, theta: f64) -> [f64; 2] {
    let n = unit(theta);
    [offset * n[0], offset * n[1]]
}

fn free_energy(point: [f64; 2], theta: f64, kt: f64) -> f64 {
    line_free_energy(
        SkewedWell::energy,
        point,
        unit(theta),
        kt,
        SADDLE_ENERGY,
        EXTENT,
    )
}

/// The line of largest free energy by Newton's method on quadrature: its
/// offset from the saddle along its normal, its angle, and the Hessian of
/// `A / kT` in those coordinates.
fn optimal_line(kt: f64) -> (f64, f64, [[f64; 2]; 2]) {
    let f = |d: f64, t: f64| free_energy(off_saddle(d, t), t, kt) / kt;
    let (mut d, mut t) = (0.0, saddle_angle());
    let h = 1.0e-3;
    let mut curvature = [[0.0; 2]; 2];
    for _ in 0..8 {
        let f0 = f(d, t);
        let (fdp, fdm, ftp, ftm) = (f(d + h, t), f(d - h, t), f(d, t + h), f(d, t - h));
        let cross =
            (f(d + h, t + h) - f(d + h, t - h) - f(d - h, t + h) + f(d - h, t - h)) / (4.0 * h * h);
        let (gd, gt) = ((fdp - fdm) / (2.0 * h), (ftp - ftm) / (2.0 * h));
        let (hdd, htt) = (
            (fdp - 2.0 * f0 + fdm) / (h * h),
            (ftp - 2.0 * f0 + ftm) / (h * h),
        );
        let det = hdd * htt - cross * cross;
        d -= (htt * gd - cross * gt) / det;
        t -= (hdd * gt - cross * gd) / det;
        curvature = [[hdd, cross], [cross, htt]];
    }
    (d, t, curvature)
}

/// `ln k` of a line by quadrature, `k = sqrt(kT/2π) Z / Z_R` with `Z_R`
/// the integral behind it.
fn log_rate(kt: f64, offset: f64, theta: f64) -> f64 {
    let point = off_saddle(offset, theta);
    let behind = half_plane_integral(
        SkewedWell::energy,
        point,
        unit(theta),
        kt,
        SADDLE_ENERGY,
        EXTENT,
    );
    0.5 * (kt / (2.0 * std::f64::consts::PI)).ln()
        - free_energy(point, theta, kt) / kt
        - behind.ln()
}

/// `ln k_HTST = ln(√(λ_R1 λ_R2) / (2π √λ‡)) - ΔV / kT`.
fn log_harmonic_rate(kt: f64) -> f64 {
    let (r1, r2) = reactant_eigenvalues();
    let (_, stable) = saddle_eigenvalues();
    ((r1 * r2).sqrt() / (2.0 * std::f64::consts::PI * stable.sqrt())).ln() - 1.0 / kt
}

fn dimer_plane() -> (Hyperplane, f64) {
    let config = MinModeConfig {
        force_tol: 1.0e-8,
        ..Default::default()
    };
    let mut session = MinModeSession::new(config, array![0.1, -0.05], array![1.0, 0.3]).unwrap();
    for _ in 0..2000 {
        let report = session.step(&SkewedWell).unwrap();
        if report.status == MinModeStatus::Converged {
            return (
                Hyperplane::from_min_mode(&session)
                    .unwrap()
                    .oriented_toward(array![1.0, 1.0].view()),
                report.curvature,
            );
        }
    }
    panic!("the dimer did not converge");
}

fn angle_of(plane: &Hyperplane) -> f64 {
    plane.normal()[1].atan2(plane.normal()[0])
}

fn sampler(kt: f64, time_step: f64) -> LangevinConfig {
    LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(time_step)
        .with_friction(1.0)
        .with_burn_in(300)
        .with_stride(5)
        .with_seed(9)
}

/// The stationary points have their closed forms, and a dimer from near
/// the saddle finds the saddle plane OH-TST starts from.
#[test]
fn skewed_well_stationary_points_and_dimer() {
    let saddle = stationary(SkewedWell::energy, [0.05, -0.03]);
    assert!(saddle[0].abs() < 1e-9 && saddle[1].abs() < 1e-9);
    let (lowest, highest, mode) = eigen2(hessian(SkewedWell::energy, saddle));
    let (unstable, stable) = saddle_eigenvalues();
    assert!((lowest - unstable).abs() < 1e-5 && (highest - stable).abs() < 1e-5);
    let angle = mode[1].atan2(mode[0]);
    let folded = (angle - saddle_angle()).rem_euclid(std::f64::consts::PI);
    assert!(folded.min(std::f64::consts::PI - folded) < 1e-6);
    let minimum = descend(SkewedWell::energy, [-0.8, -1.2]);
    assert!((minimum[0] - REACTANT[0]).abs() < 1e-6 && (minimum[1] - REACTANT[1]).abs() < 1e-6);
    let (soft, stiff, _) = eigen2(hessian(SkewedWell::energy, REACTANT));
    let (r1, r2) = reactant_eigenvalues();
    assert!((soft - r1).abs() < 1e-5 && (stiff - r2).abs() < 1e-5);

    let (plane, curvature) = dimer_plane();
    assert!(plane.point()[0].abs() < 1e-6 && plane.point()[1].abs() < 1e-6);
    assert!((curvature - unstable).abs() < 1e-3 * unstable.abs());
    let along = (angle_of(&plane) - saddle_angle()).cos();
    assert!(1.0 - along < 1e-6, "normal off the mode by {along}");
}

/// The first-order coefficients from Hessian actions and gradients match
/// the contractions of the closed-form third and fourth derivatives in
/// the Hessian eigenbases (SymPy): `A‡ = 0.27552572`, `A_R = 0.09307962`,
/// `S = 2.05852212`, `c₁ = -0.03181767`. At `kT = 0.02` quadrature puts
/// the optimal line 0.008252 from the saddle with
/// `ln(k/k_HTST) = -0.0006827`, `+0.0037572` for the saddle plane; the
/// first-order terms, `kT S / (2|λ_u|)`, `c₁ kT` and `(A‡ - A_R) kT`, are
/// within `0.5 kT²` of those.
///
/// The plane is the closed-form saddle plane: `A‡` and `S` change at first
/// order with its normal, by about 1.1 and 1.3 per radian here, so a
/// dimer normal 7e-4 rad off moves them by 1e-3.
#[test]
fn skewed_well_first_order_correction() {
    let normal = unit(saddle_angle());
    let plane = Hyperplane::new(array![0.0, 0.0], array![normal[0], normal[1]]).unwrap();
    let config = AnharmonicConfig::default().with_probes(4000).with_seed(2);
    let reactant = array![REACTANT[0], REACTANT[1]];
    let result = anharmonic_correction(&SkewedWell, reactant.view(), &plane, &config).unwrap();
    let (unstable, _) = saddle_eigenvalues();
    assert!((result.curvature + unstable).abs() < 1e-5);
    for (name, value, error, exact) in [
        ("A‡", result.saddle, result.saddle_error, 0.27552572),
        ("A_R", result.reactant, result.reactant_error, 0.09307962),
        ("S", result.slope, result.slope_error, 2.05852212),
        (
            "c₁",
            result.coefficient,
            result.coefficient_error,
            -0.03181767,
        ),
    ] {
        assert!(
            (value - exact).abs() < 4.0 * error + 1.0e-6,
            "{name} {value} ± {error} against {exact}"
        );
        assert!(error < 0.01, "{name} error {error}");
    }

    let kt = 0.02;
    let (offset, theta, _) = optimal_line(kt);
    assert!((offset - 0.008252).abs() < 2e-6, "offset {offset}");
    let harmonic = log_harmonic_rate(kt);
    let optimal = log_rate(kt, offset, theta) - harmonic;
    let fixed = log_rate(kt, 0.0, saddle_angle()) - harmonic;
    assert!((optimal + 0.0006827).abs() < 2e-6, "ln k/k_HTST {optimal}");
    assert!((fixed - 0.0037572).abs() < 2e-6, "ln k_fix/k_HTST {fixed}");
    let bound = 0.5 * kt * kt;
    let (value, error) = result.log_ratio(kt);
    assert!((value - optimal).abs() < bound + 4.0 * error);
    let (value, error) = result.fixed_log_ratio(kt);
    assert!((value - fixed).abs() < bound + 4.0 * error);
    let spread = kt * result.slope_error / (2.0 * result.curvature);
    assert!((result.offset(kt) - offset).abs() < bound + 4.0 * spread);
}

/// At `kT = 0.4` the optimizer from the dimer's plane reaches the
/// quadrature maximum of `A` within four errors and resolves the turn
/// from the saddle plane, and the sampled rate matches the quadrature
/// rate of that line, `ln(k/k_HTST) = -0.0310`. The virial on the final
/// plane is `kT`, one free direction.
#[test]
fn skewed_well_optimum_and_rate_at_high_temperature() {
    let kt = 0.4;
    let (offset, theta, curvature) = optimal_line(kt);
    assert!((offset - 0.13736).abs() < 2e-4, "offset {offset}");
    assert!(
        ((theta - saddle_angle()).to_degrees() + 7.359).abs() < 0.01,
        "turn {}",
        (theta - saddle_angle()).to_degrees()
    );
    let exact = log_rate(kt, offset, theta);
    assert!((exact - log_harmonic_rate(kt) + 0.0310).abs() < 2e-4);

    let (seed, _) = dimer_plane();
    let config = PlaneOptimizerConfig::default()
        .with_sampler(sampler(kt, 0.15).with_records(2000))
        .with_time_step(0.5)
        .with_max_iterations(100);
    let mut optimizer = PlaneOptimizer::new(PointSphereFixture::new(2), &seed, config).unwrap();
    let optimum = optimizer.run(&SkewedWell).unwrap();
    let last = optimum.trace.last().unwrap();
    let found = angle_of(&optimum.plane);
    let shift = optimum.plane.normal().dot(&optimum.plane.point());
    let angle_error = last.rotation_error / kt / curvature[1][1].abs();
    let offset_error = last.normal_force_error / kt / curvature[0][0].abs();
    assert!(
        (found - theta).abs() < 4.0 * angle_error,
        "{}° ± {}° against {}°",
        found.to_degrees(),
        angle_error.to_degrees(),
        theta.to_degrees()
    );
    assert!(
        (shift - offset).abs() < 4.0 * offset_error,
        "{shift} ± {offset_error} against {offset}"
    );
    assert!((found - saddle_angle()).abs() > 4.0 * angle_error);
    let sample = &optimum.sample;
    assert_eq!(sample.free_dimensions, 1);
    assert!(
        (sample.virial - kt).abs() < 4.0 * sample.virial_error,
        "virial {} ± {}",
        sample.virial,
        sample.virial_error
    );

    let optimal = Hyperplane::new(
        array![shift * found.cos(), shift * found.sin()],
        array![found.cos(), found.sin()],
    )
    .unwrap();
    let reactant = array![REACTANT[0], REACTANT[1]];
    let start = Hyperplane::new(reactant.clone(), array![1.0, 1.0]).unwrap();
    let density = reactant_density(
        &SkewedWell,
        &start,
        reactant.view(),
        std::slice::from_ref(&optimal),
        &sampler(kt, 0.1).with_records(16_000),
    )
    .unwrap();
    let config = ReversibleWorkConfig::default()
        .with_sampler(sampler(kt, 0.1).with_records(2000))
        .with_nodes(9);
    let work = reversible_work(
        &SkewedWell,
        &[start, optimal],
        Some(reactant.view()),
        &config,
    )
    .unwrap();
    let rate = ohtst_rate(&density, &work).unwrap();
    let at_found = log_rate(kt, shift, found);
    assert!(
        (rate.log_rate - at_found).abs() < 4.0 * rate.log_rate_error,
        "ln k {} ± {} against {at_found}",
        rate.log_rate,
        rate.log_rate_error
    );
    assert!(rate.log_rate_error < 0.08);
}
