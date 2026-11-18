//! Observable-behavior tests for Dimer, Lanczos, and Index-k minimum-mode
//! searches on Muller-Brown, double-well, narrow-valley, and index-2 surfaces.

use ndarray::{Array1, Array2, ArrayView1, array};
use rgmin::Method;
use rgsaddle::{
    BlockEigenKind, FiniteDifference, ForceGate, HessianProduct, IndexKConfig, IndexKSession,
    MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface, SaddleError,
    WithHessian, reflect_force, translated_force,
};

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

fn muller_brown_eval(x: f64, y: f64, z: f64) -> (f64, Array1<f64>) {
    let mut e = 2.0 * z * z;
    let (mut gx, mut gy) = (0.0, 0.0);
    for i in 0..4 {
        let dx = x - MB_X0[i];
        let dy = y - MB_Y0[i];
        let t = MB_A[i] * (MB_AA[i] * dx * dx + MB_B[i] * dx * dy + MB_C[i] * dy * dy).exp();
        e += t;
        gx += t * (2.0 * MB_AA[i] * dx + MB_B[i] * dy);
        gy += t * (MB_B[i] * dx + 2.0 * MB_C[i] * dy);
    }
    (e, array![gx, gy, 4.0 * z])
}

struct MullerBrown;

impl PointSurface for MullerBrown {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok(muller_brown_eval(x[0], x[1], x[2]))
    }
}

struct SteepValley;

impl PointSurface for SteepValley {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (a, b, c) = (x[0], x[1], x[2]);
        Ok((
            (a * a - 1.0).powi(2) + 200.0 * b * b + 200.0 * c * c,
            array![4.0 * a * (a * a - 1.0), 400.0 * b, 400.0 * c],
        ))
    }
}

struct FullExcluded;

impl PointSurface for FullExcluded {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((0.5 * x.dot(&x), x.to_owned()))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        Ok(Array2::eye(x.len()))
    }
}

struct ZExcludedDoubleWell;

impl PointSurface for ZExcludedDoubleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (a, b, c) = (x[0], x[1], x[2]);
        Ok((
            (a * a - 1.0).powi(2) + 2.0 * b * b - 5.0 * c * c,
            array![4.0 * a * (a * a - 1.0), 4.0 * b, -10.0 * c],
        ))
    }

    fn excluded_modes(&self, _x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        Ok(array![[0.0, 0.0, 1.0]])
    }
}

struct MultiAtomDoubleWell;

impl PointSurface for MultiAtomDoubleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        // Two atoms: first on double well, second tethered harmonically.
        let (a, b, c) = (x[0], x[1], x[2]);
        let (d, e, f) = (x[3], x[4], x[5]);
        Ok((
            (a * a - 1.0).powi(2) + 2.0 * b * b + 2.0 * c * c + d * d + e * e + f * f,
            array![
                4.0 * a * (a * a - 1.0),
                4.0 * b,
                4.0 * c,
                2.0 * d,
                2.0 * e,
                2.0 * f
            ],
        ))
    }
}

#[test]
fn dimer_and_lanczos_controls_converge_on_muller_brown_and_steep_valley() {
    // CG translator on Muller-Brown with Dimer.
    let cg_cfg = MinModeConfig::default()
        .with_kind(MinModeKind::Dimer)
        .with_force_tol(1e-4)
        .with_max_move(0.05)
        .with_rotation_angle_tol(0.0)
        .with_saddle_check(None);
    let mut cg_dimer =
        MinModeSession::new(cg_cfg, array![0.15, 0.25, 0.0], array![1.0, -0.5, 0.0]).unwrap();
    cg_dimer.set_cg_translate(true);
    let rep = cg_dimer.run(&MullerBrown, 300).unwrap();
    assert_eq!(rep.status, MinModeStatus::Converged);
    assert!((cg_dimer.position()[0] - 0.21248658).abs() < 1e-3);
    assert!((cg_dimer.position()[1] - 0.29298833).abs() < 1e-3);
    assert!(rep.curvature < 0.0);

    // Seeded L-BFGS + project + central Lanczos on Muller-Brown.
    let lanczos_cfg = MinModeConfig::default()
        .with_kind(MinModeKind::Lanczos)
        .with_difference(FiniteDifference::Central)
        .with_method(Method::lbfgs_with_memory(10))
        .with_force_tol(1e-4)
        .with_max_move(0.05)
        .with_krylov_dim(6)
        .with_saddle_check(None);
    let mut lan =
        MinModeSession::new(lanczos_cfg, array![0.15, 0.25, 0.0], array![1.0, -0.5, 0.0]).unwrap();
    lan.set_seed_curvature(true);
    lan.set_lbfgs_project(true);
    lan.set_lbfgs_neb_guards(true);
    lan.set_torque_window(Some(false));
    lan.set_rotation_stop_length(0.2).unwrap();
    let lrep = lan.run(&MullerBrown, 300).unwrap();
    assert_eq!(lrep.status, MinModeStatus::Converged);
    assert!((lan.position()[0] - 0.21248658).abs() < 1e-3);
    assert!((lan.position()[1] - 0.29298833).abs() < 1e-3);

    // Steep valley with FIRE curvature & stalled-FIRE fallback.
    let fire_cfg = MinModeConfig::default()
        .with_kind(MinModeKind::Dimer)
        .with_method(Method::fire(rgmin::FireKind::V2))
        .with_force_tol(1e-4)
        .with_max_move(0.02)
        .with_saddle_check(None);
    let mut fire =
        MinModeSession::new(fire_cfg, array![0.25, 0.15, -0.1], array![1.0, 0.1, 0.0]).unwrap();
    fire.set_fire_curvature(true);
    fire.set_seed_curvature(true);
    fire.set_force_gate(ForceGate::L2Norm);
    let frep = fire.run(&SteepValley, 800).unwrap();
    assert_eq!(frep.status, MinModeStatus::Converged);
    assert!(fire.position().iter().all(|c| c.abs() < 1e-3));

    // Multi-atom per-atom cap keeps every atom displacement <= max_move.
    let multi_cfg = MinModeConfig::default()
        .with_kind(MinModeKind::Dimer)
        .with_method(Method::lbfgs_with_memory(5))
        .with_force_tol(1e-4)
        .with_max_move(0.05)
        .with_saddle_check(None);
    let mut multi = MinModeSession::new(
        multi_cfg,
        array![0.3, 0.4, 0.0, 0.5, -0.5, 0.2],
        array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let prev = multi.position().to_owned();
    multi.step(&MultiAtomDoubleWell).unwrap();
    let delta = &multi.position().to_owned() - &prev;
    for chunk in delta.exact_chunks(3) {
        let norm = chunk.dot(&chunk).sqrt();
        assert!(norm <= 0.05 + 1e-12, "atom move {norm} exceeded cap");
    }

    // Parameter setter error paths.
    assert!(multi.set_force_tol(-1.0).is_err());
    assert!(multi.set_force_tol(f64::NAN).is_err());
    assert!(multi.set_lanczos_rel_tol(Some(0.0)).is_err());
    assert!(multi.set_lanczos_rel_tol(Some(f64::INFINITY)).is_err());
    assert!(multi.set_rotation_stop_length(-0.1).is_err());
}

#[test]
fn excluded_modes_reject_empty_complement_and_reseed_degenerate_lanczos_seed() {
    let cfg = MinModeConfig::default()
        .with_kind(MinModeKind::Lanczos)
        .with_saddle_check(None);
    let mut empty =
        MinModeSession::new(cfg.clone(), array![0.2, 0.1, 0.0], array![1.0, 0.0, 0.0]).unwrap();
    let err = empty.step(&FullExcluded).unwrap_err();
    assert!(matches!(err, SaddleError::Shape(_)));

    // Mode initially lies purely in excluded z direction, so Lanczos must reseed
    // inside the (x, y) complement and converge to (0, 0, 0) along x.
    let mut reseed = MinModeSession::new(
        cfg.with_force_tol(1e-4).with_max_move(0.05),
        array![0.25, 0.2, 0.0],
        array![0.0, 0.0, 1.0],
    )
    .unwrap();
    let rep = reseed.run(&ZExcludedDoubleWell, 200).unwrap();
    assert_eq!(rep.status, MinModeStatus::Converged);
    assert!(reseed.position()[0].abs() < 1e-3);
    assert!(reseed.position()[1].abs() < 1e-3);
    assert!(reseed.mode()[2].abs() < 1e-10);
}

struct CoupledIndexTwo;

impl PointSurface for CoupledIndexTwo {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (a, b, c) = (x[0], x[1], x[2]);
        // Double-double well: minima at (+-1, +-1, 0), index-2 saddle at (0, 0, 0).
        Ok((
            (a * a - 1.0).powi(2) + (b * b - 1.0).powi(2) + 2.0 * c * c,
            array![4.0 * a * (a * a - 1.0), 4.0 * b * (b * b - 1.0), 4.0 * c],
        ))
    }
}

struct CoupledHessian;

impl HessianProduct for CoupledHessian {
    fn product(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        Ok(array![
            (12.0 * x[0] * x[0] - 4.0) * v[0],
            (12.0 * x[1] * x[1] - 4.0) * v[1],
            4.0 * v[2],
        ])
    }
}

#[test]
fn indexk_climbs_from_positive_curvature_basin_with_hessian_product() {
    assert_eq!(BlockEigenKind::Lobpcg.to_abi(), 2);
    assert_eq!(BlockEigenKind::BlockLanczos.to_abi(), 1);
    let def = IndexKConfig::default();
    assert_eq!(def.k, 1);

    // Start inside the (+1, +1, 0) well where all curvatures are positive so
    // BlockSession must take the ascent_unit + convex_displacement branch
    // before entering the negative-curvature index-2 neighborhood.
    let wrapped = WithHessian {
        surface: &CoupledIndexTwo,
        hessian: &CoupledHessian,
    };
    let search = MinModeConfig::default()
        .with_kind(MinModeKind::Lanczos)
        .with_force_tol(1e-5)
        .with_max_move(0.1);
    let cfg = IndexKConfig::new(2, search).with_residual_tol(1e-8);
    let mut session = IndexKSession::new(cfg, array![0.75, 0.70, 0.1], None).unwrap();
    let report = session.run(&wrapped, 500).unwrap();
    assert_eq!(report.status, MinModeStatus::Converged);
    assert!(session.position().iter().all(|c| c.abs() < 1e-3));
    assert_eq!(session.curvatures().len(), 2);
    assert!(session.curvatures().iter().all(|c| *c < -3.0));
    assert_eq!(session.translated_force().len(), 3);

    // Error paths on set_position, reflect_force, and translated_force.
    assert!(session.set_position(array![0.0, 0.0], None).is_err());
    assert!(
        session
            .set_position(array![f64::NAN, 0.0, 0.0], None)
            .is_err()
    );
    assert!(
        session
            .set_position(array![0.0, 0.0, 0.0], Some(array![0.0]))
            .is_err()
    );
    assert!(
        session
            .set_position(array![0.0, 0.0, 0.0], Some(array![0.0, f64::INFINITY, 0.0]))
            .is_err()
    );
    assert!(reflect_force(array![1.0, 2.0].view(), Array2::zeros((3, 1)).view()).is_err());
    assert!(reflect_force(array![1.0, 2.0, 3.0].view(), Array2::zeros((3, 0)).view()).is_err());
    assert!(
        translated_force(
            array![1.0, 2.0, 3.0].view(),
            Array2::eye(3).slice(ndarray::s![.., ..2]),
            array![-1.0].view()
        )
        .is_err()
    );
    assert!(
        translated_force(
            array![1.0, 2.0, 3.0].view(),
            Array2::eye(3).slice(ndarray::s![.., ..2]),
            array![f64::NAN, -1.0].view()
        )
        .is_err()
    );
}
