//! Growing string method of Peters, Heyden, Bell, and Chakraborty,
//! J. Chem. Phys. 120, 7877 (2004), doi:10.1063/1.1691018:
//!
//! 1. 2D curved channel with a central repulsive obstacle where linear
//!    interpolation passes straight through the peak: GrowingStringSession
//!    grows around the obstacle, hands off to StringSession, and finds the
//!    same saddle as CI-NEB while reporting force calls.
//! 2. Mueller-Brown 2D surface (Peters et al. 2004, Figs. 3-5):
//!    GrowingStringSession finds the same primary saddle as CI-NEB.
//! 3. Image IDs and calculator batching are preserved across fragment
//!    growth and after hand-off to StringSession.
//! 4. Fail-on-revert evidence for frontier tangent constraints, the
//!    configurable node-add criterion, separate fragment relaxation,
//!    and join hand-off.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, array};
use rgmin::Method;
use rgsaddle::stencil::{ImageId, Surface};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, Cell, CiConfig, DEFAULT_FORCE_TOL,
    GrowingStringConfig, GrowingStringPhase, GrowingStringSession, Reparametrization, SaddleError,
    linear_path,
};

/// 2D curved parabolic valley `y = 1 - x^2` connecting reactant `(-1, 0)`
/// and product `(1, 0)` over the saddle near `(0, 1)`, with a repulsive
/// Gaussian peak at the linear midpoint `(0, 0)` where linear interpolation
/// is poor (`V(0, 0) = 45.0` vs barrier `~1.0` at `(0, 1)`).
#[derive(Default)]
struct CurvedChannel2D {
    rows: AtomicUsize,
    calls: AtomicUsize,
    max_energy_seen: Mutex<f64>,
}

impl CurvedChannel2D {
    fn jet(x: f64, y: f64) -> (f64, [f64; 2]) {
        let one_minus_x2 = 1.0 - x * x;
        let valley = y - one_minus_x2;
        let dy_obs = y + 0.10;
        let r2 = 4.0 * x * x + 6.0 * dy_obs * dy_obs;
        let exp_term = (-r2).exp();
        let obstacle = 16.0 * exp_term;
        let energy = one_minus_x2 * one_minus_x2 + 8.0 * valley * valley + obstacle;
        let gx = -4.0 * x * one_minus_x2 + 32.0 * x * valley - 128.0 * x * exp_term;
        let gy = 16.0 * valley - 192.0 * dy_obs * exp_term;
        (energy, [gx, gy])
    }
}

impl BandSurface for CurvedChannel2D {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.rows.fetch_add(positions.nrows(), Ordering::Relaxed);
        let mut max_e = self
            .max_energy_seen
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for (i, row) in positions.outer_iter().enumerate() {
            let (e, [gx, gy]) = Self::jet(row[0], row[1]);
            if e > *max_e {
                *max_e = e;
            }
            energies[i] = e;
            gradients[(i, 0)] = gx;
            gradients[(i, 1)] = gy;
            gradients[(i, 2)] = 2.0 * row[2];
        }
        Ok(())
    }
}

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_XX: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_XY: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_YY: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

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
            let (mut e, mut gx, mut gy) = (0.0, 0.0, 0.0);
            for k in 0..4 {
                let (dx, dy) = (row[0] - MB_X0[k], row[1] - MB_Y0[k]);
                let q = MB_XX[k] * dx * dx + MB_XY[k] * dx * dy + MB_YY[k] * dy * dy;
                let t = MB_A[k] * q.exp();
                e += t;
                gx += t * (2.0 * MB_XX[k] * dx + MB_XY[k] * dy);
                gy += t * (MB_XY[k] * dx + 2.0 * MB_YY[k] * dy);
            }
            energies[i] = e;
            gradients[(i, 0)] = gx;
            gradients[(i, 1)] = gy;
            gradients[(i, 2)] = 0.0;
        }
        Ok(())
    }
}

#[test]
fn growing_string_on_2d_curved_channel_avoids_linear_midpoint_and_matches_cineb() {
    let reactant = array![-1.0, 0.0, 0.0];
    let product = array![1.0, 0.0, 0.0];
    let n_images = 11;

    // Linear interpolation places its midpoint at (0, 0) where V(0, 0) > 18.0
    // (more than 15x the true saddle barrier ~1.07 at (0, 1.02)).
    let (e_mid_linear, _) = CurvedChannel2D::jet(0.0, 0.0);
    assert!(
        e_mid_linear > 18.0,
        "linear midpoint energy should be high: {e_mid_linear}"
    );

    let gsm_surface = CurvedChannel2D {
        rows: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
        max_energy_seen: Mutex::new(f64::NEG_INFINITY),
    };
    let gsm_cfg = GrowingStringConfig::default()
        .with_n_images(n_images)
        .with_frontier_tol(1.5)
        .with_force_tol(DEFAULT_FORCE_TOL)
        .with_max_move(0.08)
        .with_climbing(Some(CiConfig::after_rel(0.5)))
        .with_min_mode(None);
    let mut gsm =
        GrowingStringSession::new(gsm_cfg, reactant.clone(), product.clone()).expect("gsm new");

    let gsm_report = gsm.run(&gsm_surface, 800).expect("gsm run");
    assert_eq!(gsm_report.status, BandStatus::Converged);
    assert_eq!(gsm_report.phase, GrowingStringPhase::Joined);
    assert!(gsm.is_joined());
    assert!(gsm.string_session().is_some());
    assert_eq!(
        gsm_report.total_surface_rows,
        gsm_surface.rows.load(Ordering::Relaxed)
    );
    let gsm_ci = gsm_report.ci_index.expect("climbing image");
    let gsm_saddle = gsm.positions().row(gsm_ci).to_owned();
    let (gsm_barrier, [gsm_gx, gsm_gy]) = CurvedChannel2D::jet(gsm_saddle[0], gsm_saddle[1]);
    assert!(
        gsm_gx.hypot(gsm_gy) < 2e-3,
        "GSM saddle gradient norm: {}",
        gsm_gx.hypot(gsm_gy)
    );
    assert!(
        gsm_report_saddle_check(&gsm_report),
        "GSM saddle must be certified index 1"
    );

    // Growing String curves around the central obstacle, never evaluating the
    // linear midpoint spike.
    let gsm_max_e = *gsm_surface.max_energy_seen.lock().unwrap();
    assert!(
        gsm_max_e < 0.5 * e_mid_linear,
        "GSM should avoid the linear midpoint spike: max energy {gsm_max_e} vs {e_mid_linear}"
    );

    // Compare against CI-NEB started from linear interpolation on the same 2D surface.
    let initial_band =
        linear_path(reactant.view(), product.view(), n_images, None).expect("linear");
    let neb_surface = CurvedChannel2D {
        rows: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
        max_energy_seen: Mutex::new(f64::NEG_INFINITY),
    };
    let neb_cfg = BandConfig::default()
        .with_climbing(Some(CiConfig::after_rel(0.5)))
        .with_min_mode(None)
        .with_force_tol(DEFAULT_FORCE_TOL)
        .with_max_move(0.08);
    let mut neb = BandSession::new(neb_cfg, initial_band).expect("neb new");
    let neb_report = neb.run(&neb_surface, 800).expect("neb run");
    assert_eq!(neb_report.status, BandStatus::Converged);
    let neb_ci = neb_report.ci_index.expect("neb ci");
    let neb_saddle = neb.positions().row(neb_ci).to_owned();
    let (neb_barrier, _) = CurvedChannel2D::jet(neb_saddle[0], neb_saddle[1]);
    let neb_rows = neb_surface.rows.load(Ordering::Relaxed);

    eprintln!(
        "CurvedChannel2D: GSM saddle=({:.6}, {:.6}) E={:.6} rows={} steps={}; CI-NEB saddle=({:.6}, {:.6}) E={:.6} rows={} steps={}",
        gsm_saddle[0],
        gsm_saddle[1],
        gsm_barrier,
        gsm_report.total_surface_rows,
        gsm_report.iteration,
        neb_saddle[0],
        neb_saddle[1],
        neb_barrier,
        neb_rows,
        neb_report.iteration,
    );

    let dist = (&gsm_saddle - &neb_saddle)
        .dot(&(&gsm_saddle - &neb_saddle))
        .sqrt();
    assert!(
        dist < 1e-3,
        "GSM and CI-NEB must find the same saddle: dist={dist:e}, gsm={gsm_saddle:?}, neb={neb_saddle:?}"
    );
    assert!(
        (gsm_barrier - neb_barrier).abs() < 1e-5,
        "GSM and CI-NEB barriers must match: gsm={gsm_barrier}, neb={neb_barrier}"
    );
}

fn gsm_report_saddle_check(report: &rgsaddle::GrowingStringReport) -> bool {
    report
        .saddle_index
        .as_ref()
        .is_some_and(|s| s.converged && s.index == 1)
}

#[test]
fn growing_string_on_mueller_brown_matches_cineb_and_reports_force_calls() {
    let min_a = array![-0.558_223_634_633, 1.441_725_841_805, 0.0];
    let min_b = array![0.623_499_404_931, 0.028_037_758_529, 0.0];
    let n_images = 15;

    let gsm_surface = MullerBrown::default();
    let gsm_cfg = GrowingStringConfig::default()
        .with_n_images(n_images)
        .with_frontier_tol(15.0)
        .with_max_steps_per_node(Some(12))
        .with_method(Method::lbfgs_with_memory(20))
        .with_force_tol(DEFAULT_FORCE_TOL)
        .with_max_move(0.05)
        .with_climbing(Some(CiConfig::after_rel(0.5)))
        .with_min_mode(None);
    let mut gsm = GrowingStringSession::new(gsm_cfg, min_a.clone(), min_b.clone()).expect("gsm");
    let gsm_rep = gsm.run(&gsm_surface, 800).expect("gsm run");
    assert_eq!(gsm_rep.status, BandStatus::Converged);
    assert_eq!(gsm_rep.phase, GrowingStringPhase::Joined);
    let gsm_ci = gsm_rep.ci_index.expect("gsm ci");
    let gsm_saddle = gsm.positions().row(gsm_ci).to_owned();
    assert_eq!(
        gsm_rep.total_surface_rows,
        gsm_surface.rows.load(Ordering::Relaxed)
    );

    let neb_surface = MullerBrown::default();
    let initial = linear_path(min_a.view(), min_b.view(), n_images, None).expect("linear path");
    let neb_cfg = BandConfig::default()
        .with_spring(rgsaddle::SpringKind::Uniform { k: 200.0 })
        .with_method(Method::lbfgs_with_memory(20))
        .with_climbing(Some(CiConfig::after_rel(0.5)))
        .with_min_mode(None)
        .with_max_move(0.05)
        .with_force_tol(DEFAULT_FORCE_TOL);
    let mut neb = BandSession::new(neb_cfg, initial).expect("neb");
    let neb_rep = neb.run(&neb_surface, 800).expect("neb run");
    assert_eq!(neb_rep.status, BandStatus::Converged);
    let neb_ci = neb_rep.ci_index.expect("neb ci");
    let neb_saddle = neb.positions().row(neb_ci).to_owned();
    let neb_rows = neb_surface.rows.load(Ordering::Relaxed);

    eprintln!(
        "Mueller-Brown: GSM saddle=({:.6}, {:.6}) rows={} steps={}; CI-NEB saddle=({:.6}, {:.6}) rows={} steps={}",
        gsm_saddle[0],
        gsm_saddle[1],
        gsm_rep.total_surface_rows,
        gsm_rep.iteration,
        neb_saddle[0],
        neb_saddle[1],
        neb_rows,
        neb_rep.iteration,
    );

    let diff = (&gsm_saddle - &neb_saddle)
        .dot(&(&gsm_saddle - &neb_saddle))
        .sqrt();
    assert!(
        diff < 1e-3,
        "Mueller-Brown GSM and CI-NEB saddles must agree: diff={diff:e}"
    );
}

struct BatchRecordingSurface {
    batches: Mutex<Vec<Vec<ImageId>>>,
}

impl BatchRecordingSurface {
    fn new() -> Self {
        Self {
            batches: Mutex::new(Vec::new()),
        }
    }
}

impl Surface for BatchRecordingSurface {
    fn eval(
        &self,
        ids: &[ImageId],
        x: ArrayView2<f64>,
        e: &mut [f64],
        mut g: ArrayViewMut2<f64>,
    ) -> Result<(), SaddleError> {
        self.batches.lock().unwrap().push(ids.to_vec());
        for (row, r) in x.outer_iter().enumerate() {
            let (px, py, pz) = (r[0], r[1], r[2]);
            let valley = py - 0.5 * (1.0 - px * px);
            e[row] = (1.0 - px * px).powi(2) + 4.0 * valley * valley + 2.0 * pz * pz;
            g[(row, 0)] = -4.0 * px * (1.0 - px * px) + 8.0 * px * valley;
            g[(row, 1)] = 8.0 * valley;
            g[(row, 2)] = 4.0 * pz;
        }
        Ok(())
    }
}

#[test]
fn join_handoff_creates_string_session_and_preserves_image_ids_and_batching() {
    let surface = BatchRecordingSurface::new();
    let cfg = GrowingStringConfig::default()
        .with_n_images(7)
        .with_frontier_tol(0.5)
        .with_force_tol(DEFAULT_FORCE_TOL)
        .with_max_move(0.1)
        .with_min_mode(None)
        .with_saddle_check(None);
    let mut session = GrowingStringSession::new(cfg, array![-1.0, 0.0, 0.0], array![1.0, 0.0, 0.0])
        .expect("session");

    assert_eq!(session.phase(), GrowingStringPhase::Growing);
    assert_eq!(session.ids(), vec![0, 2, 3, 1]);

    let mut join_steps = 0;
    let mut saw_growing = false;
    while session.iteration() < 400 {
        let rep = session.step(&surface).expect("step");
        if rep.phase == GrowingStringPhase::Growing {
            saw_growing = true;
            assert!(session.string_session().is_none());
        }
        if rep.joined_this_step {
            join_steps += 1;
            assert_eq!(rep.phase, GrowingStringPhase::Joined);
            assert!(session.string_session().is_some());
        }
        if rep.status == BandStatus::Converged {
            break;
        }
    }
    assert!(saw_growing, "must step through Growing phase before join");
    assert_eq!(join_steps, 1, "joined_this_step must fire exactly once");

    let final_ids = session.ids();
    assert_eq!(final_ids.len(), 7);
    assert_eq!(final_ids[0], 0, "reactant endpoint keeps id 0");
    assert_eq!(final_ids[6], 1, "product endpoint keeps id 1");
    let mut sorted = final_ids.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec![0, 1, 2, 3, 4, 5, 6],
        "every image id 0..7 must be unique and preserved"
    );

    let batches = surface.batches.lock().unwrap();
    // First batch evaluates all 4 initial images [0, 2, 3, 1].
    assert_eq!(batches[0], vec![0, 2, 3, 1]);
    // Every subsequent batch evaluates only the movable interior images (never endpoints 0 or 1).
    for batch in batches.iter().skip(1) {
        assert!(!batch.is_empty());
        assert!(
            !batch.contains(&0) && !batch.contains(&1),
            "cached endpoints 0 and 1 must not be re-evaluated in interior batches: {batch:?}"
        );
    }
    // After join, interior batches carry the 5 interior image IDs in chain order.
    let expected_interior = final_ids[1..6].to_vec();
    assert_eq!(batches.last().unwrap(), &expected_interior);

    let string_session = session
        .into_string_session()
        .expect("handed-off StringSession");
    assert_eq!(string_session.ids(), final_ids.as_slice());
}

#[test]
fn fail_on_revert_frontier_tangent_constraint_and_separate_fragment_relaxation() {
    // 1. Fail-on-revert evidence for frontier tangent constraint:
    // On a surface with a strong downhill slope along the chord (x) and a
    // transverse restoring force (y), the frontier step must move strictly
    // perpendicular to the frontier tangent (dx = 0 along x), whereas an
    // unprojected force would pull the frontier backward along x.
    struct TiltedTrough;
    impl BandSurface for TiltedTrough {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            for (i, r) in positions.outer_iter().enumerate() {
                let (x, y, z) = (r[0], r[1], r[2]);
                // Strong gradient along x (parallel to tangent) and along y-0.4 (perpendicular).
                energies[i] = 5.0 * x * x + 4.0 * (y - 0.4).powi(2) + z * z;
                gradients[(i, 0)] = 10.0 * x;
                gradients[(i, 1)] = 8.0 * (y - 0.4);
                gradients[(i, 2)] = 2.0 * z;
            }
            Ok(())
        }
    }

    let cfg = GrowingStringConfig::default()
        .with_n_images(8)
        .with_frontier_tol(1e-6)
        .with_reparametrization(Reparametrization::Linear)
        .with_remove_translation(false);
    let mut session = GrowingStringSession::new(cfg, array![-1.0, 0.0, 0.0], array![1.0, 0.0, 0.0])
        .expect("session");
    let lf_before = session.left_fragment().row(1).to_owned();
    let rf_before = session.right_fragment().row(0).to_owned();
    session.step(&TiltedTrough).expect("step");
    let (t_lf, t_rf) = session.frontier_tangents().expect("frontier tangents");
    // Initial chain lies along the x axis, so frontier tangents are (1, 0, 0).
    assert!((t_lf[0] - 1.0).abs() < 1e-12 && t_lf[1].abs() < 1e-12);
    assert!((t_rf[0] - 1.0).abs() < 1e-12 && t_rf[1].abs() < 1e-12);
    let dlf = &session.left_fragment().row(1) - &lf_before;
    let drf = &session.right_fragment().row(0) - &rf_before;
    // Frontier displacements are strictly orthogonal to frontier tangents:
    assert!(
        dlf.dot(&t_lf).abs() <= 4.0 * f64::EPSILON,
        "left frontier violated tangent constraint: dot={:e}",
        dlf.dot(&t_lf)
    );
    assert!(
        drf.dot(&t_rf).abs() <= 4.0 * f64::EPSILON,
        "right frontier violated tangent constraint: dot={:e}",
        drf.dot(&t_rf)
    );
    assert!(
        dlf[1] > 0.0 && drf[1] > 0.0,
        "transverse relaxation must move toward y=0.4"
    );

    // 2. Fail-on-revert evidence for separate fragment relaxation and node-add criterion:
    // On an asymmetric surface where the left side is already on the valley floor
    // (zero transverse force) while the right side has a large transverse force,
    // only the left fragment meets frontier_tol and grows a node on step 1, while
    // only the right fragment moves transversely!
    struct AsymmetricValley;
    impl BandSurface for AsymmetricValley {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            for (i, r) in positions.outer_iter().enumerate() {
                let (x, y, z) = (r[0], r[1], r[2]);
                let target_y = if x > 0.0 { 0.5 } else { 0.0 };
                energies[i] = 4.0 * (y - target_y).powi(2) + z * z;
                gradients[(i, 0)] = 0.0;
                gradients[(i, 1)] = 8.0 * (y - target_y);
                gradients[(i, 2)] = 2.0 * z;
            }
            Ok(())
        }
    }

    let asym_cfg = GrowingStringConfig::default()
        .with_n_images(8)
        .with_frontier_tol(0.1)
        .with_min_steps_per_node(1)
        .with_remove_translation(false);
    let mut asym =
        GrowingStringSession::new(asym_cfg, array![-1.0, 0.0, 0.0], array![1.0, 0.0, 0.0])
            .expect("asym");
    let rep1 = asym.step(&AsymmetricValley).expect("step 1");
    assert_eq!(rep1.left_frontier_force, 0.0);
    assert!(rep1.right_frontier_force > 1.0);
    assert_eq!(
        rep1.nodes_added, 1,
        "only the converged left frontier may grow a node"
    );
    assert_eq!(rep1.left_nodes, 3, "left fragment grew from 2 to 3 nodes");
    assert_eq!(
        rep1.right_nodes, 2,
        "right fragment stayed at 2 nodes while relaxing"
    );
    assert!(
        asym.right_fragment().row(0)[1] > 0.0,
        "right frontier must relax toward target_y = 0.5"
    );

    // 3. Fail-on-revert evidence for min_steps_per_node:
    let delayed_cfg = GrowingStringConfig::default()
        .with_n_images(8)
        .with_frontier_tol(10.0)
        .with_min_steps_per_node(3)
        .with_remove_translation(false);
    let mut delayed =
        GrowingStringSession::new(delayed_cfg, array![-1.0, 0.0, 0.0], array![1.0, 0.0, 0.0])
            .expect("delayed");
    let d1 = delayed.step(&AsymmetricValley).expect("d1");
    let d2 = delayed.step(&AsymmetricValley).expect("d2");
    let d3 = delayed.step(&AsymmetricValley).expect("d3");
    assert_eq!(d1.nodes_added, 0);
    assert_eq!(d2.nodes_added, 0);
    assert_eq!(d3.nodes_added, 2);

    // 4. Periodic cell and fixed-atom support during growing string:
    let cell = Cell([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
    let fixed_cfg = GrowingStringConfig::default()
        .with_n_images(6)
        .with_cell(Some(cell))
        .with_fixed_atoms(vec![false, true])
        .with_frontier_tol(10.0);
    let mut fixed_sess = GrowingStringSession::new(
        fixed_cfg,
        array![-1.0, 0.0, 0.0, 3.0, 4.0, 5.0],
        array![1.0, 0.0, 0.0, 3.0, 4.0, 5.0],
    )
    .expect("fixed_sess");
    fixed_sess.step(&AsymmetricValley).expect("fixed step");
    for row in fixed_sess.positions().outer_iter() {
        assert_eq!(row[3], 3.0);
        assert_eq!(row[4], 4.0);
        assert_eq!(row[5], 5.0);
    }
}

/// Zero energy and zero gradient everywhere: no node moves, so every
/// position after a step is set by node placement alone.
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
        assert_eq!(positions.nrows(), energies.len());
        Ok(())
    }
}

fn flat_config(n_images: usize) -> GrowingStringConfig {
    GrowingStringConfig::default()
        .with_n_images(n_images)
        .with_reparametrization(Reparametrization::Linear)
        .with_remove_translation(false)
        .with_saddle_check(None)
        .with_min_mode(None)
}

/// Endpoints `(0, 0, 0)` and `(9, 0, 0)`: with ten images every node
/// sits at an integer abscissa, so each placement fraction is visible.
fn flat_session(config: GrowingStringConfig) -> GrowingStringSession {
    GrowingStringSession::new(config, array![0.0, 0.0, 0.0], array![9.0, 0.0, 0.0])
        .expect("flat session")
}

fn abscissae(rows: ArrayView2<f64>) -> Vec<f64> {
    rows.outer_iter().map(|row| row[0]).collect()
}

fn assert_abscissae(actual: Vec<f64>, expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: {actual:?}");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= 16.0 * f64::EPSILON,
            "{context}: row {index} is {got}, expected {want}: {actual:?}"
        );
    }
}

fn frontier_gap(session: &GrowingStringSession) -> f64 {
    let left = session.left_fragment();
    let right = session.right_fragment();
    let gap = &right.row(0) - &left.row(left.nrows() - 1);
    gap.dot(&gap).sqrt()
}

#[test]
fn node_placement_fractions_are_exact_on_each_growth_step() {
    let mut session = flat_session(flat_config(10));
    assert_abscissae(abscissae(session.left_fragment()), &[0.0, 1.0], "start left");
    assert_abscissae(abscissae(session.right_fragment()), &[8.0, 9.0], "start right");

    // Gap 1..8 with seven vacant slots left of the eight-slot string: the
    // left node takes 1/7 of the gap and the right node 6/7 of it.
    let first = session.step(&FlatSurface).expect("step 1");
    assert_eq!(first.nodes_added, 2);
    assert_abscissae(abscissae(session.left_fragment()), &[0.0, 1.0, 2.0], "step 1 left");
    assert_abscissae(abscissae(session.right_fragment()), &[7.0, 8.0, 9.0], "step 1 right");
    assert_eq!(session.left_ids(), &[0, 2, 4]);
    assert_eq!(session.right_ids(), &[5, 3, 1]);
    assert_eq!((first.left_nodes, first.right_nodes), (3, 3));

    // Gap 2..7 with five vacant slots: 1/5 and 4/5 of the gap.
    let second = session.step(&FlatSurface).expect("step 2");
    assert_eq!(second.nodes_added, 2);
    assert!(!second.joined_this_step);
    assert_abscissae(
        abscissae(session.left_fragment()),
        &[0.0, 1.0, 2.0, 3.0],
        "step 2 left",
    );
    assert_abscissae(
        abscissae(session.right_fragment()),
        &[6.0, 7.0, 8.0, 9.0],
        "step 2 right",
    );
    assert_eq!(session.left_ids(), &[0, 2, 4, 6]);
    assert_eq!(session.right_ids(), &[7, 5, 3, 1]);

    // Gap 3..6 with three vacant slots: 1/3 and 2/3 of the gap; the string
    // is full, so the step joins.
    let third = session.step(&FlatSurface).expect("step 3");
    assert_eq!(third.nodes_added, 2);
    assert!(third.joined_this_step);
    assert_abscissae(
        abscissae(session.positions().view()),
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "joined",
    );
    assert_eq!(session.ids(), vec![0, 2, 4, 6, 8, 9, 7, 5, 3, 1]);
}

#[test]
fn node_placement_uses_the_spline_curve_for_not_a_knot() {
    let config = flat_config(10).with_reparametrization(Reparametrization::NotAKnot);
    let mut session = flat_session(config);
    session.step(&FlatSurface).expect("step");
    assert_abscissae(abscissae(session.left_fragment()), &[0.0, 1.0, 2.0], "left");
    assert_abscissae(abscissae(session.right_fragment()), &[7.0, 8.0, 9.0], "right");
}

#[test]
fn join_tol_joins_on_the_gap_left_after_this_steps_additions() {
    // The gap is 7 before the step adds its two nodes and 5 after them.
    let config = flat_config(10).with_join_tol(Some(6.0));
    let mut session = flat_session(config);
    let report = session.step(&FlatSurface).expect("step");

    assert_eq!(report.nodes_added, 2);
    assert!(report.joined_this_step, "gap after the additions is 5 <= 6");
    assert_eq!(report.phase, GrowingStringPhase::Joined);
    assert_eq!(report.n_images, 10);
    assert!(session.is_joined());
    assert_abscissae(
        abscissae(session.positions().view()),
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "early join",
    );
    let ids = session.ids();
    assert_eq!(ids.len(), 10);
    assert_eq!(ids[0], 0);
    assert_eq!(ids[9], 1);
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    assert_eq!(&ids[..3], &[0, 2, 4], "left fragment ids survive the join");
    assert_eq!(&ids[7..], &[5, 3, 1], "right fragment ids survive the join");
}

#[test]
fn join_tol_fills_the_string_to_n_images_after_an_early_join() {
    for n_images in [6, 8, 10, 13] {
        let config = flat_config(n_images).with_join_tol(Some(1000.0));
        let mut session = flat_session(config);
        let report = session.step(&FlatSurface).expect("step");
        assert!(report.joined_this_step, "{n_images} images");
        assert_eq!(report.n_images, n_images);
        let positions = session.positions();
        assert_eq!(positions.nrows(), n_images, "string holds exactly n_images");
        let spacing = 9.0 / (n_images - 1) as f64;
        let expected: Vec<f64> = (0..n_images).map(|k| spacing * k as f64).collect();
        assert_abscissae(abscissae(positions.view()), &expected, "filled string");
        let mut ids = session.ids();
        ids.sort_unstable();
        assert_eq!(ids, (0..n_images as u64).collect::<Vec<_>>());
    }
}

#[test]
fn join_tol_boundary_is_inclusive_and_a_smaller_tolerance_keeps_growing() {
    let mut probe = flat_session(flat_config(10));
    probe.step(&FlatSurface).expect("probe step");
    let gap = frontier_gap(&probe);
    assert!((gap - 5.0).abs() <= 16.0 * f64::EPSILON, "gap after step 1: {gap}");

    let mut exact = flat_session(flat_config(10).with_join_tol(Some(gap)));
    let report = exact.step(&FlatSurface).expect("exact step");
    assert!(report.joined_this_step, "gap == join_tol joins");

    let below = gap * (1.0 - 4.0 * f64::EPSILON);
    let mut tight = flat_session(flat_config(10).with_join_tol(Some(below)));
    let report = tight.step(&FlatSurface).expect("tight step");
    assert!(!report.joined_this_step, "gap > join_tol keeps growing");
    assert_eq!(report.phase, GrowingStringPhase::Growing);
    assert_eq!((report.left_nodes, report.right_nodes), (3, 3));
    assert_eq!(report.n_images, 6);

    let mut zero = flat_session(flat_config(10).with_join_tol(Some(0.0)));
    let report = zero.step(&FlatSurface).expect("zero step");
    assert!(!report.joined_this_step, "a zero join_tol never joins a positive gap");
}

#[test]
fn join_tol_joins_even_when_no_node_is_added_on_the_step() {
    // A frontier tolerance no force can meet and a minimum of 100 steps
    // per node: the step adds nothing, so only the gap can trigger the join.
    let config = flat_config(10)
        .with_min_steps_per_node(100)
        .with_join_tol(Some(8.0));
    let mut session = flat_session(config);
    let report = session.step(&FlatSurface).expect("step");
    assert_eq!(report.nodes_added, 0);
    assert!(report.joined_this_step, "gap 7 <= 8");
    assert_eq!(session.positions().nrows(), 10);
    assert_abscissae(
        abscissae(session.positions().view()),
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "no-addition join",
    );

    let config = flat_config(10)
        .with_min_steps_per_node(100)
        .with_join_tol(Some(6.0));
    let mut session = flat_session(config);
    let report = session.step(&FlatSurface).expect("step");
    assert!(!report.joined_this_step, "gap 7 > 6 with nothing added");
    assert_eq!(report.phase, GrowingStringPhase::Growing);
}

#[test]
fn config_builders_store_every_value() {
    let cell = Cell([[7.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 9.0]]);
    let config = GrowingStringConfig::default()
        .with_reparametrization(Reparametrization::Linear)
        .with_force_tol(0.123)
        .with_max_move(0.0456)
        .with_method(Method::QuickMin)
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_cell(Some(cell))
        .with_fixed_atoms(vec![true, false])
        .with_remove_translation(false)
        .with_time_step(0.375)
        .with_quickmin(true)
        .with_n_images(13)
        .with_frontier_tol(0.31)
        .with_min_steps_per_node(4)
        .with_max_steps_per_node(Some(9))
        .with_join_tol(Some(2.5));

    let string = config.band.string.expect("string config");
    assert_eq!(string.reparametrization, Reparametrization::Linear);
    assert_eq!(config.band.force_tol, 0.123);
    assert_eq!(config.band.max_move, 0.0456);
    assert!(matches!(config.band.method, Method::QuickMin));
    assert!(config.band.climbing.is_none());
    assert!(config.band.min_mode.is_none());
    assert!(config.band.saddle_check.is_none());
    let stored = config.band.cell.expect("cell");
    assert_eq!(stored.0[0][0], 7.0);
    assert_eq!(stored.0[2][2], 9.0);
    assert_eq!(config.band.fixed_atoms, vec![true, false]);
    assert!(!config.band.remove_translation);
    assert_eq!(config.band.time_step, 0.375);
    assert!(config.band.quickmin);
    assert_eq!(config.n_images, 13);
    assert_eq!(config.frontier_tol, 0.31);
    assert_eq!(config.min_steps_per_node, 4);
    assert_eq!(config.max_steps_per_node, Some(9));
    assert_eq!(config.join_tol, Some(2.5));

    let armed = GrowingStringConfig::default()
        .with_climbing(Some(CiConfig::default().with_trigger_force(7.0)))
        .with_min_mode(Some(rgsaddle::MinModeRefine::default()))
        .with_saddle_check(Some(rgsaddle::SaddleIndexCheck::default()))
        .with_quickmin(false)
        .with_remove_translation(true);
    assert_eq!(armed.band.climbing.expect("climbing").trigger_force, 7.0);
    assert!(armed.band.min_mode.is_some());
    assert!(armed.band.saddle_check.is_some());
    assert!(!armed.band.quickmin);
    assert!(armed.band.remove_translation);
}
