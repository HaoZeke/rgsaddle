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
