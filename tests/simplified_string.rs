//! Simplified string method of E, Ren, and Vanden-Eijnden, J. Chem. Phys.
//! 126, 164103 (2007), doi:10.1063/1.2720838:
//!
//! 1. Mueller-Brown MEP and saddle regression (Sec. V.C), comparing
//!    not-a-knot cubic-spline vs linear reparametrization against the
//!    steepest-descent MEP from the analytic saddles and against
//!    rgsaddle's CI-NEB on the same surface.
//! 2. Optional climbing-image string, OCI-NEB min-mode refinement, and
//!    saddle-index certification on both Mueller-Brown barriers.
//! 3. Steppers (`FIRE`, `L-BFGS`, `QuickMin`, `Bb`, `ODE12r`), periodic
//!    minimum-image wrapping, fixed-atom masks, exponential metric,
//!    image threads, `band_forces`, and step guards.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView2, array};
use rgmin::{FireKind, Method, QuickMinOptions};
use rgsaddle::reparam::segment_lengths;
use rgsaddle::{
    AutoNeb, AutoNebTuning, BandConfig, BandSession, BandStatus, BandSurface, Cell, CiConfig,
    DEFAULT_FORCE_TOL, ExpBandPrecon, Reparametrization, RtrConfig, SaddleError, SaddleIndexCheck,
    SpringKind, StringConfig, band_forces, linear_path,
};

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_XX: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_XY: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_YY: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];
const MB_STATIONARY_TOL: f64 = 5e-10;
const MB_CLIMBING_ENERGY_TOL: f64 = 5e-8;
const MB_MEP_DISTANCE_TOL: f64 = 5e-6;
// Self-measured regression pins from this implementation, not values
// reported by E, Ren, and Vanden-Eijnden.
const MB_SPLINE_MEP_DISTANCE: f64 = 0.011_118;
const MB_LINEAR_MEP_DISTANCE: f64 = 0.014_425;
const MB_CINEB_MEP_DISTANCE: f64 = 0.030_509;
// The deterministic smoke run measured (iterations, rows) as spline
// (148, 6833), linear (129, 5959), CI-string S1 (139, 6422), CI-NEB
// (94, 2190), and CI-string S2 (32, 720). These ceilings add 10% and
// round upward, so they detect a work regression without pinning libm
// details to one machine.
const MB_SPLINE_MAX_WORK: (usize, usize) = (165, 7_600);
const MB_LINEAR_MAX_WORK: (usize, usize) = (145, 6_600);
const MB_CI_STRING_MAX_WORK: (usize, usize) = (155, 7_100);
const MB_CINEB_MAX_WORK: (usize, usize) = (105, 2_450);
const MB_CI_STRING_S2_MAX_WORK: (usize, usize) = (36, 800);
const MB_A_MINIMUM: ([f64; 2], f64) = (
    [-0.558_223_634_633, 1.441_725_841_805],
    -146.699_517_209_954,
);
const MB_B_MINIMUM: ([f64; 2], f64) =
    ([0.623_499_404_931, 0.028_037_758_529], -108.166_724_116_852);
const MB_C_MINIMUM: ([f64; 2], f64) =
    ([-0.050_010_822_998, 0.466_694_104_872], -80.767_818_129_659);
const MB_S1_SADDLE: ([f64; 2], f64) =
    ([-0.822_001_558_733, 0.624_312_802_815], -40.664_843_508_657);
const MB_S2_SADDLE: ([f64; 2], f64) = ([0.212_486_582_001, 0.292_988_325_107], -72.248_940_112_325);

/// Analytic energy, gradient, and 2x2 Hessian of the Mueller-Brown potential.
fn muller_brown_jet(x: f64, y: f64) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    let (mut e, mut gx, mut gy) = (0.0, 0.0, 0.0);
    let (mut hxx, mut hxy, mut hyy) = (0.0, 0.0, 0.0);
    for i in 0..4 {
        let (dx, dy) = (x - MB_X0[i], y - MB_Y0[i]);
        let q = MB_XX[i] * dx * dx + MB_XY[i] * dx * dy + MB_YY[i] * dy * dy;
        let t = MB_A[i] * q.exp();
        let qx = 2.0 * MB_XX[i] * dx + MB_XY[i] * dy;
        let qy = MB_XY[i] * dx + 2.0 * MB_YY[i] * dy;
        e += t;
        gx += t * qx;
        gy += t * qy;
        hxx += t * (qx * qx + 2.0 * MB_XX[i]);
        hxy += t * (qx * qy + MB_XY[i]);
        hyy += t * (qy * qy + 2.0 * MB_YY[i]);
    }
    (e, [gx, gy], [[hxx, hxy], [hxy, hyy]])
}

/// Exact critical point of Mueller-Brown by analytic 2D Newton iteration.
fn mb_stationary(mut x: f64, mut y: f64) -> ([f64; 2], f64) {
    let stop = f64::EPSILON.sqrt() * f64::EPSILON.cbrt();
    for _ in 0..40 {
        let (_, [gx, gy], [[hxx, hxy], [_, hyy]]) = muller_brown_jet(x, y);
        if gx.hypot(gy) <= stop {
            break;
        }
        let det = hxx * hyy - hxy * hxy;
        x -= (hyy * gx - hxy * gy) / det;
        y -= (-hxy * gx + hxx * gy) / det;
    }
    let (e, _, _) = muller_brown_jet(x, y);
    ([x, y], e)
}

#[derive(Default)]
struct MullerBrown {
    rows: AtomicUsize,
}

impl BandSurface for MullerBrown {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.rows.fetch_add(positions.nrows(), Ordering::Relaxed);
        for (i, row) in positions.outer_iter().enumerate() {
            let (e, [gx, gy], _) = muller_brown_jet(row[0], row[1]);
            energies[i] = e;
            gradients[(i, 0)] = gx;
            gradients[(i, 1)] = gy;
            gradients[(i, 2)] = 0.0;
        }
        Ok(())
    }
}

/// Reference steepest-descent MEP integrated by RK4 in arc length from both
/// analytic saddles down to minima A, C, and B.
fn analytic_mb_mep() -> Vec<[f64; 2]> {
    let ([s1x, s1y], _) = mb_stationary(-0.822, 0.624);
    let ([s2x, s2y], _) = mb_stationary(0.212, 0.293);

    let unstable_dir = |sx: f64, sy: f64| -> [f64; 2] {
        let (_, _, [[a, b], [_, c]]) = muller_brown_jet(sx, sy);
        let tr = a + c;
        let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
        let lam_min = 0.5 * (tr - disc);
        let (vx, vy) = (b, lam_min - a);
        let n = vx.hypot(vy);
        [vx / n, vy / n]
    };

    let trace = |s: [f64; 2], dir: [f64; 2], sign: f64| -> Vec<[f64; 2]> {
        let kick = f64::EPSILON.cbrt();
        let ds = 0.0005;
        let floor = f64::EPSILON.cbrt();
        let mut pts = vec![s];
        let mut p = [s[0] + sign * kick * dir[0], s[1] + sign * kick * dir[1]];
        pts.push(p);
        let unit_f = |q: [f64; 2]| -> ([f64; 2], f64) {
            let (_, [gx, gy], _) = muller_brown_jet(q[0], q[1]);
            let n = gx.hypot(gy);
            if n > 0.0 {
                ([-gx / n, -gy / n], n)
            } else {
                ([0.0, 0.0], 0.0)
            }
        };
        for _ in 0..8000 {
            let (k1, nrm) = unit_f(p);
            if nrm <= floor {
                break;
            }
            let (k2, _) = unit_f([p[0] + 0.5 * ds * k1[0], p[1] + 0.5 * ds * k1[1]]);
            let (k3, _) = unit_f([p[0] + 0.5 * ds * k2[0], p[1] + 0.5 * ds * k2[1]]);
            let (k4, _) = unit_f([p[0] + ds * k3[0], p[1] + ds * k3[1]]);
            p = [
                p[0] + (ds / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
                p[1] + (ds / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            ];
            pts.push(p);
        }
        pts
    };

    let d1 = unstable_dir(s1x, s1y);
    let d2 = unstable_dir(s2x, s2y);
    let mut all = Vec::new();
    for (s, d) in [([s1x, s1y], d1), ([s2x, s2y], d2)] {
        all.extend(trace(s, d, -1.0));
        all.extend(trace(s, d, 1.0));
    }
    all
}

fn max_distance_to_curve(band: ArrayView2<f64>, curve: &[[f64; 2]]) -> f64 {
    band.outer_iter()
        .map(|row| {
            curve
                .iter()
                .map(|p| (row[0] - p[0]).hypot(row[1] - p[1]))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0_f64, f64::max)
}

fn assert_equal_spacing(
    band: ArrayView2<f64>,
    cell: Option<&Cell>,
    scheme: Reparametrization,
    relative_tolerance: f64,
) {
    let lengths = segment_lengths(&band.to_owned(), cell, scheme);
    let mean = lengths.iter().sum::<f64>() / lengths.len() as f64;
    let spread = lengths.iter().fold(0.0_f64, |largest, length| {
        largest.max((length - mean).abs())
    });
    assert!(
        spread <= relative_tolerance * mean,
        "{scheme:?} segment lengths are not equal: {lengths:?}"
    );
}

fn assert_work_within(
    label: &str,
    iterations: usize,
    rows: usize,
    (max_iterations, max_rows): (usize, usize),
) {
    assert!(
        iterations <= max_iterations,
        "{label} took {iterations} iterations, budget {max_iterations}"
    );
    assert!(
        rows <= max_rows,
        "{label} took {rows} surface rows, budget {max_rows}"
    );
}

#[test]
fn mueller_brown_mep_and_saddle_regression_matches_cineb() {
    let ([ax, ay], ea) = mb_stationary(-0.558, 1.442);
    let ([bx, by], eb) = mb_stationary(0.623, 0.028);
    let ([cx, cy], ec) = mb_stationary(-0.050, 0.467);
    let ([s1x, s1y], es1) = mb_stationary(-0.822, 0.624);
    let ([s2x, s2y], es2) = mb_stationary(0.212, 0.293);
    // E, Ren, and Vanden-Eijnden, J. Chem. Phys. 126, 164103 (2007),
    // Sec. V.C prints this potential and uses its stationary points to
    // illustrate the string and climbing methods.
    for (name, found, expected) in [
        ("A", ([ax, ay], ea), MB_A_MINIMUM),
        ("B", ([bx, by], eb), MB_B_MINIMUM),
        ("C", ([cx, cy], ec), MB_C_MINIMUM),
        ("S1", ([s1x, s1y], es1), MB_S1_SADDLE),
        ("S2", ([s2x, s2y], es2), MB_S2_SADDLE),
    ] {
        assert!(
            (found.0[0] - expected.0[0]).abs() <= MB_STATIONARY_TOL
                && (found.0[1] - expected.0[1]).abs() <= MB_STATIONARY_TOL
                && (found.1 - expected.1).abs() <= MB_STATIONARY_TOL,
            "{name}: got {found:?}, expected {expected:?}"
        );
    }
    let mep = analytic_mb_mep();

    let min_a = array![ax, ay, 0.0];
    let min_b = array![bx, by, 0.0];
    let min_c = array![cx, cy, 0.0];
    let initial = linear_path(min_a.view(), min_b.view(), 25, None).expect("linear path");

    // 1. Plain simplified string (no climbing image): spline vs linear reparametrization.
    let spline_surface = MullerBrown::default();
    let spline_cfg = BandConfig::default()
        .with_string(StringConfig::spline())
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_max_move(0.05)
        .with_force_tol(DEFAULT_FORCE_TOL);
    let mut spline_session = BandSession::new(spline_cfg, initial.clone()).expect("spline string");
    let spline_rep = spline_session
        .run(&spline_surface, 800)
        .expect("run spline");
    assert_eq!(spline_rep.status, BandStatus::Converged);
    assert!(spline_rep.ci_index.is_none());

    let linear_surface = MullerBrown::default();
    let linear_cfg = BandConfig::default()
        .with_string(StringConfig::linear())
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_max_move(0.05)
        .with_force_tol(DEFAULT_FORCE_TOL);
    let mut linear_session = BandSession::new(linear_cfg, initial.clone()).expect("linear string");
    let linear_rep = linear_session
        .run(&linear_surface, 800)
        .expect("run linear");
    assert_eq!(linear_rep.status, BandStatus::Converged);

    let err_spline = max_distance_to_curve(spline_session.positions(), &mep);
    let err_linear = max_distance_to_curve(linear_session.positions(), &mep);
    assert_equal_spacing(
        spline_session.positions(),
        None,
        Reparametrization::NotAKnot,
        5e-3,
    );
    assert_equal_spacing(
        linear_session.positions(),
        None,
        Reparametrization::Linear,
        4e-2,
    );

    // 2. Climbing-image simplified string on A -> B (locates primary saddle S1)
    //    and on C -> B (locates secondary saddle S2), plus CI-NEB on the same surface.
    let ci_string_surface = MullerBrown::default();
    let ci_string_cfg = BandConfig::default()
        .with_string(StringConfig::spline())
        .with_climbing(CiConfig::after_rel(0.5))
        .with_min_mode(None)
        .with_saddle_check(SaddleIndexCheck::default())
        .with_max_move(0.05)
        .with_force_tol(DEFAULT_FORCE_TOL);
    let mut ci_string =
        BandSession::new(ci_string_cfg.clone(), initial.clone()).expect("ci string");
    let ci_string_rep = ci_string
        .run(&ci_string_surface, 1000)
        .expect("run ci string");
    assert_eq!(ci_string_rep.status, BandStatus::Converged);
    let ci_idx = ci_string_rep.ci_index.expect("armed climbing image");
    let str_s1 = ci_string.positions().row(ci_idx).to_owned();
    let (str_es1, [str_gx, str_gy], _) = muller_brown_jet(str_s1[0], str_s1[1]);
    let cert = ci_string_rep
        .saddle_index
        .as_ref()
        .expect("certified saddle");
    assert_eq!(cert.index, 1, "primary saddle must have Hessian index 1");

    // CI-NEB on the exact same 25-image A -> B setup (stiff spring for MB scale).
    let cineb_surface = MullerBrown::default();
    let cineb_cfg = BandConfig::default()
        .with_spring(SpringKind::Uniform { k: 200.0 })
        .with_climbing(CiConfig::after_rel(0.5))
        .with_min_mode(None)
        .with_saddle_check(SaddleIndexCheck::default())
        .with_max_move(0.05)
        .with_force_tol(DEFAULT_FORCE_TOL)
        .with_method(Method::lbfgs_with_memory(20));
    let mut cineb = BandSession::new(cineb_cfg, initial).expect("ci-neb");
    let cineb_rep = cineb.run(&cineb_surface, 1000).expect("run ci-neb");
    assert_eq!(cineb_rep.status, BandStatus::Converged);
    let neb_ci = cineb_rep.ci_index.expect("ci-neb climbing image");
    let neb_s1 = cineb.positions().row(neb_ci).to_owned();
    let (neb_es1, _, _) = muller_brown_jet(neb_s1[0], neb_s1[1]);
    let err_cineb = max_distance_to_curve(cineb.positions(), &mep);

    // Secondary barrier C -> B with climbing simplified string.
    let cb_init = linear_path(min_c.view(), min_b.view(), 13, None).expect("C -> B path");
    let cb_surface = MullerBrown::default();
    let mut cb_string = BandSession::new(ci_string_cfg, cb_init).expect("C -> B string");
    let cb_rep = cb_string.run(&cb_surface, 1000).expect("run C -> B string");
    assert_eq!(cb_rep.status, BandStatus::Converged);
    let cb_ci = cb_rep.ci_index.expect("C -> B climbing image");
    let str_s2 = cb_string.positions().row(cb_ci).to_owned();
    let (str_es2, _, _) = muller_brown_jet(str_s2[0], str_s2[1]);
    assert_eq!(cb_rep.saddle_index.as_ref().expect("certified S2").index, 1);

    let spline_rows = spline_surface.rows.load(Ordering::Relaxed);
    let linear_rows = linear_surface.rows.load(Ordering::Relaxed);
    let ci_string_rows = ci_string_surface.rows.load(Ordering::Relaxed);
    let cineb_rows = cineb_surface.rows.load(Ordering::Relaxed);
    let cb_rows = cb_surface.rows.load(Ordering::Relaxed);
    assert_work_within(
        "plain spline string",
        spline_rep.iteration,
        spline_rows,
        MB_SPLINE_MAX_WORK,
    );
    assert_work_within(
        "plain linear string",
        linear_rep.iteration,
        linear_rows,
        MB_LINEAR_MAX_WORK,
    );
    assert_work_within(
        "CI-string S1",
        ci_string_rep.iteration,
        ci_string_rows,
        MB_CI_STRING_MAX_WORK,
    );
    assert_work_within(
        "CI-NEB S1",
        cineb_rep.iteration,
        cineb_rows,
        MB_CINEB_MAX_WORK,
    );
    assert_work_within(
        "CI-string S2",
        cb_rep.iteration,
        cb_rows,
        MB_CI_STRING_S2_MAX_WORK,
    );

    eprintln!(
        "Mueller-Brown reproduction:\n  \
         exact S1 = ({s1x:.8}, {s1y:.8}), E = {es1:.8}, dE(A->S1) = {:.8}\n  \
         exact S2 = ({s2x:.8}, {s2y:.8}), E = {es2:.8}, dE(B->S2) = {:.8}\n  \
         plain string (spline, 25 img): steps = {}, rows = {}, MEP dist = {err_spline:.6}\n  \
         plain string (linear, 25 img): steps = {}, rows = {}, MEP dist = {err_linear:.6}\n  \
         CI-string S1: steps = {}, rows = {}, pos = ({:.8}, {:.8}), E = {str_es1:.8}, |g| = {:.2e}\n  \
         CI-NEB   S1: steps = {}, rows = {}, pos = ({:.8}, {:.8}), E = {neb_es1:.8}, MEP dist = {err_cineb:.6}\n  \
         CI-string S2: steps = {}, rows = {}, pos = ({:.8}, {:.8}), E = {str_es2:.8}",
        es1 - ea,
        es2 - eb,
        spline_rep.iteration,
        spline_rows,
        linear_rep.iteration,
        linear_rows,
        ci_string_rep.iteration,
        ci_string_rows,
        str_s1[0],
        str_s1[1],
        str_gx.hypot(str_gy),
        cineb_rep.iteration,
        cineb_rows,
        neb_s1[0],
        neb_s1[1],
        cb_rep.iteration,
        cb_rows,
        str_s2[0],
        str_s2[1],
    );

    // Self-measured 25-image regression values. The tolerance accommodates ordinary
    // libm last-bit variation while rejecting a changed evolution or
    // reparametrization equation.
    assert!((err_spline - MB_SPLINE_MEP_DISTANCE).abs() <= MB_MEP_DISTANCE_TOL);
    assert!((err_linear - MB_LINEAR_MEP_DISTANCE).abs() <= MB_MEP_DISTANCE_TOL);
    assert!((err_cineb - MB_CINEB_MEP_DISTANCE).abs() <= MB_MEP_DISTANCE_TOL);
    assert!(
        err_spline < err_linear,
        "spline string ({err_spline}) must improve on the polyline string ({err_linear})"
    );
    assert!((str_s1[0] - s1x).hypot(str_s1[1] - s1y) < DEFAULT_FORCE_TOL);
    assert!((str_s1[0] - neb_s1[0]).hypot(str_s1[1] - neb_s1[1]) < DEFAULT_FORCE_TOL);
    assert!((str_es1 - MB_S1_SADDLE.1).abs() < MB_CLIMBING_ENERGY_TOL);
    assert!((str_es1 - neb_es1).abs() < DEFAULT_FORCE_TOL);
    assert!((str_s2[0] - s2x).hypot(str_s2[1] - s2y) < DEFAULT_FORCE_TOL);
    assert!((str_es2 - MB_S2_SADDLE.1).abs() < MB_CLIMBING_ENERGY_TOL);
}

struct CoupledDoubleWell;

impl BandSurface for CoupledDoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, r) in positions.outer_iter().enumerate() {
            let (x, y, z) = (r[0], r[1], r[2]);
            energies[i] = (x * x - 1.0).powi(2) + 2.0 * y * y + 3.0 * z * z;
            gradients[(i, 0)] = 4.0 * x * (x * x - 1.0);
            gradients[(i, 1)] = 4.0 * y;
            gradients[(i, 2)] = 6.0 * z;
        }
        Ok(())
    }
}

struct FlatSurface;

impl BandSurface for FlatSurface {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        energies.fill(0.0);
        gradients.fill(0.0);
        assert_eq!(positions.dim(), gradients.dim());
        Ok(())
    }
}

fn reparametrize_once(
    initial: Array2<f64>,
    string: StringConfig,
    cell: Option<Cell>,
    fixed: Vec<bool>,
) -> BandSession {
    let mut config = BandConfig::default()
        .with_string(string)
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_remove_translation(false)
        .with_fixed_atoms(fixed);
    if let Some(periodic) = cell {
        config = config.with_cell(periodic);
    }
    let mut session = BandSession::new(config, initial).expect("flat string");
    session.step(&FlatSurface).expect("reparametrization step");
    session
}

#[test]
fn linear_string_reparametrizes_by_arc_length_and_differs_from_spline() {
    let initial = array![
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ];
    let linear = reparametrize_once(initial.clone(), StringConfig::linear(), None, Vec::new());
    let expected = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ];
    for (found, want) in linear.positions().iter().zip(expected.iter()) {
        assert!((found - want).abs() <= 16.0 * f64::EPSILON);
    }
    assert_equal_spacing(
        linear.positions(),
        None,
        Reparametrization::Linear,
        16.0 * f64::EPSILON,
    );

    let spline = reparametrize_once(initial, StringConfig::spline(), None, Vec::new());
    let difference = spline
        .positions()
        .iter()
        .zip(expected.iter())
        .fold(0.0_f64, |largest, (found, want)| {
            largest.max((found - want).abs())
        });
    assert!(
        difference > 0.01,
        "spline unexpectedly matched the polyline"
    );
}

#[test]
fn periodic_string_reparametrizes_across_the_cell_boundary() {
    let length = 10.0;
    let cell = Cell([[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]]);
    let initial = array![
        [9.0, 0.0, 0.0],
        [9.2, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [1.8, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ];
    let session = reparametrize_once(initial, StringConfig::linear(), Some(cell), Vec::new());
    assert_equal_spacing(
        session.positions(),
        Some(&cell),
        Reparametrization::Linear,
        64.0 * f64::EPSILON,
    );
}

#[test]
fn string_reparametrization_repins_interior_fixed_atoms() {
    let endpoint_gap = 0.5e-10;
    let initial = array![
        [0.0, 0.0, 0.0, 2.0, -0.5, 0.25],
        [0.1, 0.0, 0.0, 3.0, 1.0, 2.0],
        [1.0, 0.0, 0.0, 4.0, 2.0, 3.0],
        [1.0, 2.0, 0.0, 5.0, 3.0, 4.0],
        [2.0, 2.0, 0.0, 2.0 + endpoint_gap, -0.5, 0.25],
    ];
    assert!(
        initial
            .outer_iter()
            .skip(1)
            .take(3)
            .any(|row| row[3] != initial[(0, 3)])
    );
    let session = reparametrize_once(initial, StringConfig::linear(), None, vec![false, true]);
    let held: Vec<u64> = session
        .positions()
        .row(0)
        .iter()
        .skip(3)
        .map(|v| v.to_bits())
        .collect();
    for row in session.positions().outer_iter().skip(1).take(3) {
        let found: Vec<u64> = row.iter().skip(3).map(|v| v.to_bits()).collect();
        assert_eq!(found, held);
    }
}

#[test]
fn ode12r_unchanged_trial_still_reparametrizes() {
    let initial = array![
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ];
    let expected = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ];

    let cfg = BandConfig::default()
        .with_string(StringConfig::linear())
        .with_method(Method::Ode12r)
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_remove_translation(false);
    let mut session = BandSession::new(cfg, initial).expect("ODE12r string");
    // ODE12r retains an unchanged trial for 64 retries. The next call
    // must still pass through the same reparametrization as every other
    // stepper; deleting that shared call leaves the uneven input intact.
    for _ in 0..65 {
        session.step(&FlatSurface).expect("ODE12r step");
    }
    for (found, want) in session.positions().iter().zip(expected.iter()) {
        assert!((found - want).abs() <= 16.0 * f64::EPSILON);
    }
    assert_equal_spacing(
        session.positions(),
        None,
        Reparametrization::Linear,
        64.0 * f64::EPSILON,
    );
}

#[test]
fn all_force_driven_steppers_converge_simplified_string() {
    let initial = Array2::from_shape_fn((9, 3), |(i, k)| {
        let t = i as f64 / 8.0;
        [
            2.0 * t - 1.0,
            0.35 * (std::f64::consts::PI * t).sin(),
            -0.2 * (std::f64::consts::PI * t).sin(),
        ][k]
    });

    let methods = [
        ("fire2", Method::fire(FireKind::V2), false),
        ("fire1", Method::fire(FireKind::V1), false),
        ("lbfgs", Method::lbfgs_with_memory(15), false),
        ("quickmin-rgmin", Method::QuickMin, true),
        ("bb", Method::Bb, false),
        ("ode12r", Method::Ode12r, false),
    ];

    for (name, method, qm) in methods {
        let cfg = BandConfig::default()
            .with_string(StringConfig::spline())
            .with_method(method)
            .with_quickmin(qm)
            .with_quickmin_options(QuickMinOptions::rgmin())
            .with_climbing(CiConfig::after(0.5))
            .with_min_mode(None)
            .with_saddle_check(None)
            .with_max_move(0.1)
            .with_force_tol(DEFAULT_FORCE_TOL);
        let mut session = BandSession::new(cfg, initial.clone()).expect(name);
        let report = session.run(&CoupledDoubleWell, 600).expect(name);
        assert_eq!(report.status, BandStatus::Converged, "{name}");
        let ci = report.ci_index.expect(name);
        let saddle = session.positions().row(ci).to_owned();
        assert!(
            saddle.dot(&saddle).sqrt() < 5.0 * DEFAULT_FORCE_TOL,
            "{name}: {saddle}"
        );
        let eval = session.evaluation().expect("cached evaluation");
        assert_eq!(eval.energies.len(), 9);
        assert_eq!(eval.projected.dim(), (7, 3));
    }
}

/// Two-atom harmonic bond + double well on atom 0; atom 1 can be frozen or periodic.
struct TwoAtomWell {
    box_len: Option<f64>,
}

impl BandSurface for TwoAtomWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let cell = self
            .box_len
            .map(|l| minimage::Cell::ortho(l, l, l).expect("ortho cell"));
        for (i, r) in positions.outer_iter().enumerate() {
            let p0 = [r[0], r[1], r[2]];
            let p1 = [r[3], r[4], r[5]];
            let d = match &cell {
                Some(sim) => sim.displacement_euclidean(p0, p1),
                None => [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]],
            };
            let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let stretch = dist - 2.0;
            let well = (p0[0] * p0[0] - 1.0).powi(2) + 2.0 * p0[1] * p0[1] + 2.0 * p0[2] * p0[2];
            energies[i] = well + 2.0 * stretch * stretch;
            let bond_scale = if dist > 0.0 {
                4.0 * stretch / dist
            } else {
                0.0
            };
            gradients[(i, 0)] = 4.0 * p0[0] * (p0[0] * p0[0] - 1.0) - bond_scale * d[0];
            gradients[(i, 1)] = 4.0 * p0[1] - bond_scale * d[1];
            gradients[(i, 2)] = 4.0 * p0[2] - bond_scale * d[2];
            gradients[(i, 3)] = bond_scale * d[0];
            gradients[(i, 4)] = bond_scale * d[1];
            gradients[(i, 5)] = bond_scale * d[2];
        }
        Ok(())
    }
}

#[test]
fn fixed_atoms_periodic_cell_precon_and_image_threads_work_on_string() {
    let initial = Array2::from_shape_fn((7, 6), |(i, c)| {
        let t = i as f64 / 6.0;
        let bow = 0.25 * (std::f64::consts::PI * t).sin();
        [-1.0 + 2.0 * t, bow, 0.0, 2.0, 0.5, -0.25][c]
    });
    let fixed = vec![false, true];
    let frozen_ref: Vec<u64> = initial
        .outer_iter()
        .flat_map(|row| row.into_iter().skip(3).map(|v| v.to_bits()))
        .collect();

    let base_cfg = BandConfig::default()
        .with_string(StringConfig::spline())
        .with_fixed_atoms(fixed)
        .with_climbing(CiConfig::after(0.5))
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_max_move(0.1)
        .with_force_tol(DEFAULT_FORCE_TOL);

    let mut serial = BandSession::new(base_cfg.clone(), initial.clone()).expect("serial");
    let rep_serial = serial
        .run(&TwoAtomWell { box_len: None }, 500)
        .expect("serial run");
    assert_eq!(rep_serial.status, BandStatus::Converged);

    let frozen_after: Vec<u64> = serial
        .positions()
        .outer_iter()
        .flat_map(|row| row.into_iter().skip(3).map(|v| v.to_bits()))
        .collect();
    assert_eq!(frozen_after, frozen_ref, "fixed atom bits changed");

    // Multi-threaded image evaluation matches 1 thread bit-for-bit.
    let mut threaded = BandSession::new(base_cfg, initial.clone()).expect("threaded");
    threaded.set_image_threads(3).expect("3 image threads");
    let rep_threaded = threaded
        .run(&TwoAtomWell { box_len: None }, 500)
        .expect("threaded run");
    assert_eq!(rep_threaded.iteration, rep_serial.iteration);
    assert_eq!(rep_threaded.surface_rows, rep_serial.surface_rows);
    for (a, b) in threaded.positions().iter().zip(serial.positions().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // Periodic cell + exponential metric on two free atoms.
    let l = 10.0;
    let cell = Cell([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]);
    let periodic_init = Array2::from_shape_fn((7, 6), |(i, c)| {
        let t = i as f64 / 6.0;
        let x0 = -1.0 + 2.0 * t;
        let bow = 0.2 * (std::f64::consts::PI * t).sin();
        [x0, bow, 0.0, x0 + 2.0, bow, 0.0][c]
    });
    let precon_cfg = BandConfig::default()
        .with_string(StringConfig::spline())
        .with_cell(cell)
        .with_exp_precon(ExpBandPrecon::default())
        .with_remove_translation(false)
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_max_move(0.1)
        .with_force_tol(DEFAULT_FORCE_TOL);
    let mut precon_session = BandSession::new(precon_cfg, periodic_init).expect("precon string");
    let rep_precon = precon_session
        .run(&TwoAtomWell { box_len: Some(l) }, 600)
        .expect("precon run");
    assert_eq!(rep_precon.status, BandStatus::Converged);

    // band_forces on a string configuration omits springs and tangent projection
    // on non-climbing images and reflects the climbing image.
    let string_forces_cfg = BandConfig::default()
        .with_string(StringConfig::linear())
        .with_remove_translation(false);
    let assembled = band_forces(
        &string_forces_cfg,
        &TwoAtomWell { box_len: None },
        initial.view(),
        Some(3),
    )
    .expect("string band_forces");
    let mut raw_e = Array1::zeros(7);
    let mut raw_g = Array2::zeros((7, 6));
    TwoAtomWell { box_len: None }
        .eval(initial.view(), &mut raw_e, &mut raw_g)
        .expect("eval");
    // Image 1 is non-climbing: force equals -gradient exactly.
    for c in 0..6 {
        assert!((assembled.force[c] + raw_g[(1, c)]).abs() <= 4.0 * f64::EPSILON);
    }
}

#[test]
fn string_step_guards_and_incompatible_modes_are_checked() {
    struct ConstantForce([f64; 3]);
    impl BandSurface for ConstantForce {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            for r in 0..positions.nrows() {
                energies[r] = positions[(r, 0)];
                for c in 0..3 {
                    gradients[(r, c)] = self.0[c];
                }
            }
            Ok(())
        }
    }

    let band = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let cfg = BandConfig::default()
        .with_string(StringConfig::default())
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_force_tol(f64::EPSILON.sqrt());

    // Overflowing force norm leaves the band untouched.
    let mut session = BandSession::new(cfg.clone(), band.clone()).expect("session");
    let huge = f64::MAX.sqrt() * 2.0;
    let err = session
        .step(&ConstantForce([0.0, huge, 0.0]))
        .expect_err("overflow");
    assert!(matches!(err, SaddleError::NonFinite("band force norm")));
    assert_eq!(session.positions(), band.view());

    // Swallowed step above tolerance raises Stalled.
    let origin = 256.0 / f64::EPSILON.sqrt();
    let far = array![
        [origin, origin, origin],
        [origin + 1.0, origin, origin],
        [origin + 2.0, origin, origin]
    ];
    let mut stalled = BandSession::new(cfg.clone(), far.clone()).expect("stalled session");
    let tiny = 4.0 * f64::EPSILON.sqrt();
    let err = stalled
        .step(&ConstantForce([0.0, tiny, 0.0]))
        .expect_err("stalled");
    assert!(matches!(err, SaddleError::Stalled(_)), "{err}");
    assert_eq!(stalled.positions(), far.view());

    // Failed first callback does not poison the next step.
    struct FailOnce(AtomicBool);
    impl BandSurface for FailOnce {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            if !self.0.swap(true, Ordering::Relaxed) {
                return Err(SaddleError::Surface("transient".into()));
            }
            ConstantForce([0.0, 1.0, 0.0]).eval(positions, energies, gradients)
        }
    }
    let mut recover = BandSession::new(cfg.clone(), band.clone()).expect("recover");
    let fail = FailOnce(AtomicBool::new(false));
    assert!(matches!(recover.step(&fail), Err(SaddleError::Surface(_))));
    assert_eq!(recover.positions(), band.view());
    assert!(recover.step(&fail).is_ok());

    // RTR and string are mutually exclusive; solid-state and AutoNEB refuse string.
    assert!(matches!(
        recover.set_rtr(Some(RtrConfig::default())),
        Err(SaddleError::Invalid(_))
    ));
    let mut neb = BandSession::new(BandConfig::default(), band.clone()).expect("neb");
    neb.set_rtr(Some(RtrConfig::default())).expect("rtr");
    assert!(matches!(
        neb.set_string(Some(StringConfig::default())),
        Err(SaddleError::Invalid(_))
    ));
    let solid_cfg = cfg.clone().with_solid(rgsaddle::SolidState::new(vec![
        Cell([
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0]
        ]);
        3
    ]));
    assert!(matches!(
        BandSession::new(solid_cfg, band.clone()),
        Err(SaddleError::Invalid(_))
    ));
    let autoneb_cfg = cfg
        .with_force_tol(DEFAULT_FORCE_TOL)
        .with_autoneb(AutoNebTuning::default());
    assert!(matches!(
        AutoNeb::new(autoneb_cfg, band),
        Err(SaddleError::Invalid(_))
    ));
}
