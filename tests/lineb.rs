//! Eckart closed forms, a constant-Hessian head-to-head of the two rate
//! evaluators, and rigid energy shifts.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1};
use rgsaddle::{
    LinebConfig, LinebSession, LinebStatus, Masses, PointSurface, SaddleError, hqtst_rate,
    hqtst_rate_from_image_hessians, hqtst_rate_interpolated_hessian,
};

struct Eckart;

impl PointSurface for Eckart {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let s = 1.0 / x[0].cosh();
        Ok((s * s, Array1::from_elem(1, -2.0 * s * s * x[0].tanh())))
    }
}

/// Inverted parabola `V = 1 − x²/2`. The Hessian is constant, so the
/// cubic interpolation of it is exact and a central Hessian-vector
/// product is exact.
struct Parabola;

impl PointSurface for Parabola {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((1.0 - 0.5 * x[0] * x[0], Array1::from_elem(1, -x[0])))
    }
}

/// An endpoint `δ` short of its turning point shortens the half period
/// by `√(2δ/|∇V|)` and feels the force `endpoint_scale * δ`. This
/// tolerance keeps both ends inside the quadrature slack of the closed
/// form test, `√(spacing / panels⁴ / |∇V|)`.
fn eckart_force_tolerance() -> (f64, LinebConfig) {
    let turning = (1.0 / 0.25_f64).sqrt().acosh();
    let spacing = turning * 2.0 / 8.0;
    let draft = LinebConfig::new(spacing, spacing);
    let panels = (2 * draft.quadrature_pairs) as f64;
    let tol = draft.endpoint_scale * spacing / (8.0 * panels.powi(4));
    (tol, draft.with_opt_tol(tol).with_perp_tol(tol))
}

#[test]
fn endpoint_convergence_allows_nonuniform_interior_spacing() {
    let turning = 2.0_f64.acosh();
    let seed = ndarray::array![[-turning], [0.25 * turning], [turning]];
    let masses = Masses::coordinates(Array1::ones(1).view()).unwrap();
    let energy = Eckart.eval(seed.row(0)).unwrap().0;
    let tolerance = 1e-8;
    for (endpoint_only, expected) in [
        (false, LinebStatus::Running),
        (true, LinebStatus::Converged),
    ] {
        let cfg = LinebConfig::new(tolerance, tolerance).with_endpoint_convergence(endpoint_only);
        let mut session = LinebSession::new(cfg, masses.clone(), seed.clone(), energy).unwrap();
        let step = session.step(&Eckart).unwrap();
        assert!(step.endpoint_max < tolerance && step.perp_max < tolerance);
        assert!(step.opt_max > tolerance);
        assert_eq!(step.status, expected);
    }
}

#[test]
fn level_observer_includes_each_seed_in_its_own_path() {
    let surface = FailingEckart {
        calls: AtomicUsize::new(0),
        fail_at: usize::MAX,
    };
    let mep = ndarray::array![[-2.], [-1.], [0.], [1.], [2.]];
    let masses = Masses::coordinates(Array1::ones(1).view()).unwrap();
    let mut observed = Vec::new();
    let mut previous = 0;
    let paths = rgsaddle::continue_from_mep_with_observer(
        &surface,
        mep.view(),
        &masses,
        &LinebConfig::new(1e-5, 1e-5).with_endpoint_convergence(true),
        &rgsaddle::Continuation::new(0.25, 5, 2).with_linear_seed(true),
        |level, path| {
            assert_eq!(level, observed.len());
            let current = surface.calls.load(Ordering::Relaxed);
            assert_eq!(current - previous, path.calls.total_with_seed());
            previous = current;
            observed.push(path.energy);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(observed, vec![0.75, 0.5]);
    assert_eq!(paths.len(), observed.len());
    assert_eq!(previous, surface.calls.load(Ordering::Relaxed));
}

fn eckart_path() -> rgsaddle::LinebPath {
    let turning = (1.0 / 0.25_f64).sqrt().acosh();
    let images = Array2::from_shape_fn((9, 1), |(i, _)| {
        let end = 0.5 * turning;
        -end + 2.0 * end * i as f64 / 8.0
    });
    let (_, config) = eckart_force_tolerance();
    let mut session = LinebSession::new(
        config,
        Masses::coordinates(Array1::ones(1).view()).unwrap(),
        images,
        0.25,
    )
    .unwrap();
    let report = session.run(&Eckart, 500).unwrap();
    assert_eq!(report.status, LinebStatus::Converged);
    session.path(&Eckart).unwrap()
}

/// Eckart that fails on its `fail_at`-th evaluation.
struct FailingEckart {
    calls: AtomicUsize,
    fail_at: usize,
}

impl PointSurface for FailingEckart {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        if self.calls.fetch_add(1, Ordering::Relaxed) + 1 == self.fail_at {
            return Err(SaddleError::Surface("injected".into()));
        }
        Eckart.eval(x)
    }
}

/// Only an image below `E + w` is retried. A surface error inside the
/// objective ends the step at the failing call.
#[test]
fn a_surface_error_inside_a_step_is_returned() {
    let turning = (1.0 / 0.25_f64).sqrt().acosh();
    let images = Array2::from_shape_fn((9, 1), |(i, _)| {
        let end = 0.5 * turning;
        -end + 2.0 * end * i as f64 / 8.0
    });
    let (_, config) = eckart_force_tolerance();
    let mut session = LinebSession::new(
        config,
        Masses::coordinates(Array1::ones(1).view()).unwrap(),
        images,
        0.25,
    )
    .unwrap();
    // Nine calls fix the turning width; the failure lands in the first
    // assembly of the objective.
    let surface = FailingEckart {
        calls: AtomicUsize::new(0),
        fail_at: 12,
    };
    let err = session.step(&surface).unwrap_err();
    assert!(matches!(err, SaddleError::Surface(_)), "{err}");
    assert_eq!(surface.calls.load(Ordering::Relaxed), 12);
}

/// The first drift from rest is one half of dt² times the endpoint force.
#[test]
fn projected_verlet_uses_half_kicks_and_reaches_the_turning_points() {
    let dt = 0.2;
    let tolerance = 1e-5;
    let config = LinebConfig::new(tolerance, tolerance)
        .with_endpoint_kappa(1.0)
        .with_time_step(dt)
        .with_projected_verlet(true);
    let mut session = LinebSession::new(
        config,
        Masses::coordinates(Array1::ones(1).view()).unwrap(),
        ndarray::array![[-0.5], [0.0], [0.5]],
        0.5,
    )
    .unwrap();
    session.step(&Parabola).unwrap();
    let force = (1.0 - 0.5 * 0.5_f64.powi(2)) - 0.5;
    assert!((session.cartesian().unwrap()[[2, 0]] - (0.5 + dt * dt * force / 2.0)).abs() < 1e-14);
    let report = session.run(&Parabola, 500).unwrap();
    assert_eq!(report.status, LinebStatus::Converged);
    assert!((session.cartesian().unwrap()[[2, 0]] - 1.0).abs() < 2.0 * tolerance);
}

#[test]
fn eckart_action_period_and_temperature_match_the_closed_form() {
    let energy = 0.25_f64;
    let path = eckart_path();
    let exact_action = std::f64::consts::PI * (2.0_f64.sqrt() - (2.0 * energy).sqrt());
    let exact_half = std::f64::consts::PI / (2.0 * energy).sqrt();
    let slope = (2.0 * energy) * (1.0 - energy).sqrt();
    let panels = (2 * path_pairs()) as f64;
    let length = path.cartesian[(path.cartesian.nrows() - 1, 0)] - path.cartesian[(0, 0)];
    let spacing = length / (path.cartesian.nrows() - 1) as f64;
    let slack = (spacing / panels.powi(4) / slope).sqrt();
    assert!(
        (path.action - exact_action).abs() <= slack,
        "action {} vs {exact_action}, slack {slack}",
        path.action
    );
    assert!(
        (path.half_period - exact_half).abs() <= slack,
        "half {} vs {exact_half}",
        path.half_period
    );
    assert!(
        (path.half_period - path.half_period_dynamics).abs() <= slack,
        "quadrature {} vs dynamics {}",
        path.half_period,
        path.half_period_dynamics
    );
    let period = 2.0 * path.half_period;
    assert!((path.temperature - 1.0 / period).abs() <= slack / (period * period));
    let line = (2.0 * path.action + energy * period) / 1.0;
    let rate = hqtst_rate(&Eckart, &path, 8).unwrap();
    assert!((rate.line_action - line).abs() <= slack);
    assert_eq!(rate.modes, 8 - 1);
    assert!(rate.temperature > 0.0);
}

fn path_pairs() -> usize {
    eckart_force_tolerance().1.quadrature_pairs
}

#[test]
fn constant_hessian_log_dets_agree() {
    // The Hessian is constant on the whole line, so the images only have
    // to stay where the spline is above the tunneling energy. A straight
    // chain inside the barrier makes the cubic the identity interpolant.
    let images = Array2::from_shape_fn((9, 1), |(i, _)| -0.3 + 0.6 * i as f64 / 8.0);
    let curvature = 1.0;
    let spacing = 0.6 / 8.0;
    let tol = curvature * spacing * f64::EPSILON.sqrt();
    let session = LinebSession::new(
        LinebConfig::new(tol, tol),
        Masses::coordinates(Array1::ones(1).view()).unwrap(),
        images,
        0.5,
    )
    .unwrap();
    let path = session.path(&Parabola).unwrap();
    let beads = 10;
    let direct = hqtst_rate(&Parabola, &path, beads).unwrap();
    let oracle = hqtst_rate_interpolated_hessian(&Parabola, &path, beads).unwrap();
    assert_eq!(direct.modes, beads - 1);
    assert_eq!(oracle.modes, direct.modes);
    // The oracle differences the gradient with step √ε · max|q| and the
    // replica products with ε^{1/3} times the path length. The absolute
    // noise of one Hessian entry is ε · |∇V| / step, at most √ε here,
    // where |∇V| is the coordinate and the curvature is 1. Each kept
    // mode can move the log-determinant by that much. The bound is
    // absolute.
    let entry_noise = f64::EPSILON.sqrt();
    let roundoff = entry_noise * direct.modes as f64;
    let blocks = vec![Array2::from_elem((1, 1), -curvature); path.cartesian.nrows()];
    let analytic = hqtst_rate_from_image_hessians(&Parabola, &path, beads, &blocks).unwrap();
    let quadrature = session.quadrature_path(&Parabola).unwrap();
    let independent =
        hqtst_rate_from_image_hessians(&Parabola, &quadrature, beads, &blocks).unwrap();
    assert_eq!(quadrature.calls.dynamics, 0);
    assert_eq!(quadrature.temperature, path.temperature);
    assert_eq!(
        independent.log_rate_over_reactant,
        analytic.log_rate_over_reactant
    );
    assert_eq!(analytic.evaluations, beads);
    assert_eq!(analytic.negative_modes, oracle.negative_modes);
    assert!(
        (analytic.log_rate_over_reactant - oracle.log_rate_over_reactant).abs() <= 0.5 * roundoff
    );
    assert!(
        (direct.log_abs_det - oracle.log_abs_det).abs() <= roundoff,
        "matrix-free {} vs interpolated {}, bound {roundoff}",
        direct.log_abs_det,
        oracle.log_abs_det
    );
    // The rate's logarithm carries half the log-determinant, with a minus.
    assert!(
        (direct.log_rate_over_reactant - oracle.log_rate_over_reactant).abs() <= 0.5 * roundoff
    );
}

/// The unoptimized nine-image chain on `[−end, end]`, measured at
/// `energy`.
fn straight_path<S: PointSurface>(surface: &S, end: f64, energy: f64) -> rgsaddle::LinebPath {
    let images = Array2::from_shape_fn((9, 1), |(i, _)| -end + 2.0 * end * i as f64 / 8.0);
    let tol = 2.0 * end / 8.0 * f64::EPSILON.sqrt();
    LinebSession::new(
        LinebConfig::new(tol, tol),
        Masses::coordinates(Array1::ones(1).view()).unwrap(),
        images,
        energy,
    )
    .unwrap()
    .path(surface)
    .unwrap()
}

/// Both end knots sit exactly on the turning points `x = ±1` of the
/// parabola at `E = 1/2`. The half period of the inverted harmonic well
/// is `π` at every energy.
#[test]
fn turning_knots_at_both_ends_give_the_harmonic_half_period() {
    let path = straight_path(&Parabola, 1.0, 0.5);
    let panels = (2 * path_pairs()) as f64;
    let slack = (2.0 / 8.0 / panels.powi(4)).sqrt();
    assert!(
        (path.half_period - std::f64::consts::PI).abs() <= slack,
        "{}",
        path.half_period
    );
}

/// Force-converged ends can sit a few `10⁻⁷` above and below `E`. The
/// end below makes end-chord samples with `u > 0` see `σ = 0`. Refining
/// those near-turning ends onto `V = E` along the full surface gradient
/// before measure recovers the period; a band whose ends are far below
/// `E` still fails as before.
#[test]
fn near_turning_endpoints_are_refined_onto_v_equals_e_before_measure() {
    let energy = 0.5;
    // Residuals matching the modified Müller–Brown direct-ring failure:
    // one end below `E`, one above, both inside a `10⁻⁵` force tolerance.
    let left = -(1.0 - 1.14e-6);
    let right = 1.0 + 2.4e-7;
    let images = Array2::from_shape_fn((9, 1), |(i, _)| left + (right - left) * i as f64 / 8.0);
    let (v_left, _) = Parabola.eval(images.row(0)).unwrap();
    let (v_right, _) = Parabola.eval(images.row(8)).unwrap();
    assert!(v_left - energy > 0.0, "{}", v_left - energy);
    assert!(v_right - energy < 0.0, "{}", v_right - energy);
    let session = LinebSession::new(
        LinebConfig::new(1e-5, 1e-5),
        Masses::coordinates(Array1::ones(1).view()).unwrap(),
        images,
        energy,
    )
    .unwrap();
    let path = session.path(&Parabola).unwrap();
    let (v0, _) = Parabola.eval(path.cartesian.row(0)).unwrap();
    let (vn, _) = Parabola
        .eval(path.cartesian.row(path.cartesian.nrows() - 1))
        .unwrap();
    assert_eq!(v0, energy);
    assert_eq!(vn, energy);
    let panels = (2 * path_pairs()) as f64;
    let slack = (2.0 / 8.0 / panels.powi(4)).sqrt();
    assert!(
        (path.half_period - std::f64::consts::PI).abs() <= slack,
        "{}",
        path.half_period
    );

    let deep = Array2::from_shape_fn((9, 1), |(i, _)| -1.1 + 2.2 * i as f64 / 8.0);
    let invalid = LinebSession::new(
        LinebConfig::new(1e-5, 1e-5),
        Masses::coordinates(Array1::ones(1).view()).unwrap(),
        deep,
        energy,
    )
    .unwrap();
    let err = invalid.path(&Parabola).unwrap_err();
    assert!(
        matches!(&err, SaddleError::Invalid(message) if message.contains("vanishes")),
        "{err}"
    );
}

/// The inverted parabola lowered by `SHIFT`, a total energy far below
/// zero.
struct LoweredParabola;

const SHIFT: f64 = -2000.0;

impl PointSurface for LoweredParabola {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient) = Parabola.eval(x)?;
        Ok((energy + SHIFT, gradient))
    }
}

/// Knots `GAP` above the turning points of the lowered parabola, where
/// `V − E` rounds at `ε |SHIFT|`, the size of `GAP`. The half period is
/// still `π`, and the clock costs no more than at the parabola's own
/// energy zero.
#[test]
fn turning_knots_on_a_lowered_surface_keep_the_clock_cheap() {
    const GAP: f64 = 1e-12;
    let raised = straight_path(&Parabola, 1.0, 0.5 - GAP);
    let lowered = straight_path(&LoweredParabola, 1.0, 0.5 + SHIFT - GAP);
    let panels = (2 * path_pairs()) as f64;
    let slack = (2.0 / 8.0 / panels.powi(4)).sqrt();
    assert!(
        (lowered.half_period - std::f64::consts::PI).abs() <= slack,
        "{}",
        lowered.half_period
    );
    assert!(
        lowered.calls.measure <= raised.calls.measure,
        "lowered {} vs raised {}",
        lowered.calls.measure,
        raised.calls.measure
    );
}

/// Lowering every energy by a constant moves `ln(k Z_r)` by `−τ SHIFT / ℏ`
/// and nothing else, even where `k Z_r` itself overflows.
#[test]
fn an_energy_offset_moves_only_the_log_rate() {
    let raised = straight_path(&Parabola, 0.3, 0.5);
    let lowered = straight_path(&LoweredParabola, 0.3, 0.5 + SHIFT);
    let beads = 10;
    let reference = hqtst_rate(&Parabola, &raised, beads).unwrap();
    let rate = hqtst_rate(&LoweredParabola, &lowered, beads).unwrap();
    assert_eq!(rate.rate_over_reactant, f64::INFINITY);
    assert!(rate.log_rate_over_reactant.is_finite());
    let expected = -rate.period * SHIFT;
    let moved = rate.log_rate_over_reactant - reference.log_rate_over_reactant;
    // Each bead energy rounds at `ε |SHIFT|`, and the period and the
    // replicas at the relative `ε |SHIFT| / (V − E)` of the gap.
    let relative = 64.0 * f64::EPSILON * SHIFT.abs();
    assert!(
        (moved - expected).abs() <= relative * expected.abs(),
        "moved {moved}, expected {expected}"
    );
    assert!(
        (rate.log_abs_det - reference.log_abs_det).abs() <= relative * reference.log_abs_det.abs(),
        "{} vs {}",
        rate.log_abs_det,
        reference.log_abs_det
    );
}

#[test]
fn interpolated_image_force_clock_recovers_harmonic_period_and_beads() {
    let masses = Masses::coordinates(Array1::ones(1).view()).unwrap();
    let images = ndarray::array![[-1.0], [-0.33], [0.21], [0.64], [1.0]];
    let path_at = |pairs| {
        let config = LinebConfig::new(1e-6, 1e-6)
            .with_quadrature_pairs(pairs)
            .with_interpolated_force_clock(true);
        LinebSession::new(config, masses.clone(), images.clone(), 0.5)
            .unwrap()
            .path(&Parabola)
            .unwrap()
    };
    let mut error = f64::INFINITY;
    for pairs in [1, 2, 4] {
        let path = path_at(pairs);
        let diff = (path.half_period - std::f64::consts::PI).abs();
        assert!(diff < error * 0.35 && diff < 1e-4, "{diff} {error}");
        error = diff;
        assert_eq!(path.half_period, path.half_period_dynamics);
        assert_eq!(path.calls.dynamics, images.nrows() + 1);
        let turning = path.turning.unwrap();
        assert!(turning.max_cartesian_displacement < 1e-7);
        assert!(turning.energy_residual.abs() < 1e-7);
        let beads = path.cartesian_replicas(17).unwrap();
        for (j, bead) in beads.column(0).iter().enumerate() {
            let angle = std::f64::consts::TAU * j as f64 / 17.;
            assert!((bead + angle.cos()).abs() < 2e-5);
        }
    }
}

#[test]
fn asymmetric_image_force_reports_the_measured_short_turn_in_either_orientation() {
    struct Asymmetric;
    impl PointSurface for Asymmetric {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let q = x[0];
            let energy = (1. - q * q) * (1. + 0.6 * q.powi(3));
            Ok((
                energy,
                ndarray::array![-2. * q + 1.8 * q * q - 3. * q.powi(4)],
            ))
        }
    }
    for direction in [-1., 1.] {
        let images = ndarray::array![[-1.], [-1. / 3.], [1. / 3.], [1.]] * direction;
        let masses = Masses::coordinates(ndarray::array![1.].view()).unwrap();
        let config = LinebConfig::new(1e-6, 1e-6).with_interpolated_force_clock(true);
        let path = LinebSession::new(config, masses, images.clone(), 0.)
            .unwrap()
            .quadrature_path(&Asymmetric)
            .unwrap();
        let t = path.turning.unwrap();
        assert_eq!(t.launch_image, if direction > 0. { 0 } else { 3 });
        let ring = path.cartesian_replicas(24).unwrap();
        let turn = ring.row(if t.launch_image == 0 { 12 } else { 0 });
        let measured = Asymmetric.eval(turn).unwrap().0;
        let missed = (1. - turn[0]).abs();
        assert!(missed > 0.02 && measured > 0.05);
        assert!((t.max_cartesian_displacement - missed).abs() < 1e-12);
        assert!((t.energy_residual - measured).abs() < 1e-12);
        assert!((t.endpoint_energy_difference - measured).abs() < 1e-12);
        assert_eq!(path.calls.dynamics, images.nrows() + 1);
    }
}

#[test]
fn continuation_from_turning_images_preserves_and_lowers_energy() {
    let masses = Masses::coordinates(Array1::ones(1).view()).unwrap();
    let images = ndarray::array![[-1.0], [-0.5], [0.0], [0.5], [1.0]];
    let config = LinebConfig::new(1e-5, 1e-5).with_interpolated_force_clock(true);
    let ladder = rgsaddle::Continuation::new(0.02, 9, 2);
    let paths = rgsaddle::continue_from_path_with_observer(
        &Parabola,
        images.view(),
        0.5,
        0.5,
        &masses,
        &config,
        &ladder,
        |_, _| Ok(()),
    )
    .unwrap();
    for (i, path) in paths.iter().enumerate() {
        assert_eq!(path.cartesian.nrows(), 9);
        assert!((path.energy - (0.5 - i as f64 * 0.02)).abs() < 1e-12);
        assert!((path.half_period - std::f64::consts::PI).abs() < 1e-4);
        assert!(path.turning.unwrap().energy_residual.abs() < 1e-7);
    }
    let resumed = rgsaddle::continue_from_path_with_observer(
        &Parabola,
        paths[0].cartesian.view(),
        paths[0].energy,
        paths[1].energy,
        &masses,
        &config,
        &rgsaddle::Continuation::new(0.02, 9, 1),
        |_, _| Ok(()),
    )
    .unwrap();
    assert_eq!(resumed[0].cartesian, paths[1].cartesian);
    assert_eq!(
        resumed[0].calls.total_with_seed(),
        paths[1].calls.total_with_seed()
    );
}
