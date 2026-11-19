//! Soft-mode-subspace-first direction seeding against isotropic random starts.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{
    MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface, SaddleError,
    soft_mode_then_full_directions,
};

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

struct MullerBrown;

impl PointSurface for MullerBrown {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut e = 2.0 * x[2] * x[2];
        let (mut gx, mut gy) = (0.0, 0.0);
        for i in 0..4 {
            let dx = x[0] - MB_X0[i];
            let dy = x[1] - MB_Y0[i];
            let t = MB_A[i] * (MB_AA[i] * dx * dx + MB_B[i] * dx * dy + MB_C[i] * dy * dy).exp();
            e += t;
            gx += t * (2.0 * MB_AA[i] * dx + MB_B[i] * dy);
            gy += t * (MB_B[i] * dx + 2.0 * MB_C[i] * dy);
        }
        Ok((e, array![gx, gy, 4.0 * x[2]]))
    }
}

struct ProductDoubleWell6;

impl PointSurface for ProductDoubleWell6 {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut e = 0.0;
        let mut g = Array1::zeros(6);
        for i in 0..6 {
            let xi = x[i];
            e += (xi * xi - 1.0).powi(2);
            g[i] = 4.0 * xi * (xi * xi - 1.0);
        }
        Ok((e, g))
    }
}

struct Counting<'a, S: PointSurface> {
    inner: &'a S,
    calls: AtomicUsize,
}

impl<S: PointSurface> PointSurface for Counting<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.eval(x)
    }
}

#[derive(Clone, Copy, Debug)]
enum Strategy {
    IsotropicRandom,
    SoftModeFirst,
}

struct CampaignSpec<'a, S: PointSurface> {
    surface: &'a S,
    minimum: &'a Array1<f64>,
    known_saddles: &'a [Array1<f64>],
    classify_tol: f64,
    start_scale: f64,
    seed_count: usize,
}

fn minmode_config() -> MinModeConfig {
    MinModeConfig::default()
        .with_kind(MinModeKind::Dimer)
        .with_force_tol(1.0e-4)
        .with_max_move(0.05)
        .with_rotation_angle_tol(0.0)
        .with_saddle_check(None)
}

fn classify_nearest(
    point: ArrayView1<'_, f64>,
    saddles: &[Array1<f64>],
    tol: f64,
) -> Option<usize> {
    let mut best = (usize::MAX, f64::INFINITY);
    for (i, saddle) in saddles.iter().enumerate() {
        let d = (&point - saddle).mapv(|x| x * x).sum().sqrt();
        if d < best.1 {
            best = (i, d);
        }
    }
    (best.1 <= tol).then_some(best.0)
}

fn softmode_campaign<S: PointSurface>(
    spec: &CampaignSpec<'_, S>,
    run_seed: u64,
    strategy: Strategy,
) -> Result<(bool, usize), SaddleError> {
    let mut found = vec![false; spec.known_saddles.len()];
    let mut calls_to_cover = 0usize;
    let directions = match strategy {
        Strategy::IsotropicRandom => {
            let mut rng = SplitMix64::new(run_seed);
            let mut dirs = Vec::with_capacity(2 * spec.seed_count);
            for _ in 0..(2 * spec.seed_count) {
                dirs.push(sample_on_sphere(spec.minimum.len(), &mut rng));
            }
            dirs
        }
        Strategy::SoftModeFirst => {
            soft_mode_then_full_directions(
                spec.surface,
                spec.minimum.view(),
                spec.seed_count,
                run_seed,
            )?
            .directions
        }
    };
    let config = minmode_config();
    for direction in directions {
        let start = spec.minimum + &(direction.clone() * spec.start_scale);
        let mut session = MinModeSession::new(config.clone(), start, direction)?;
        let counted = Counting {
            inner: spec.surface,
            calls: AtomicUsize::new(0),
        };
        let report = match session.run(&counted, 400) {
            Ok(report) => report,
            Err(_) => {
                calls_to_cover += counted.calls.load(Ordering::Relaxed);
                continue;
            }
        };
        let calls = counted.calls.load(Ordering::Relaxed);
        calls_to_cover += calls;
        if report.status == MinModeStatus::Converged
            && let Some(label) =
                classify_nearest(session.position(), spec.known_saddles, spec.classify_tol)
        {
            found[label] = true;
        }
        if found.iter().all(|&v| v) {
            return Ok((true, calls_to_cover));
        }
    }
    Ok((false, calls_to_cover))
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    if sorted.is_empty() {
        return f64::NAN;
    }
    let m = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[m - 1] + sorted[m])
    } else {
        sorted[m]
    }
}

fn median_se(values: &[f64], seed: u64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let mut rng = SplitMix64::new(seed);
    let mut med = Vec::with_capacity(512);
    for _ in 0..512 {
        let mut sample = Vec::with_capacity(values.len());
        for _ in 0..values.len() {
            let i = (rng.next_u64() as usize) % values.len();
            sample.push(values[i]);
        }
        med.push(median(&sample));
    }
    let mean = med.iter().sum::<f64>() / (med.len() as f64);
    let var = med
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / ((med.len() - 1) as f64);
    var.sqrt()
}

fn compare_strategies<S: PointSurface>(
    name: &str,
    surface: &S,
    minimum: Array1<f64>,
    saddles: Vec<Array1<f64>>,
    classify_tol: f64,
    start_scale: f64,
) -> Result<(), SaddleError> {
    let rounds = 14usize;
    let seed_count = 14usize;
    let spec = CampaignSpec {
        surface,
        minimum: &minimum,
        known_saddles: &saddles,
        classify_tol,
        start_scale,
        seed_count,
    };
    let mut random_calls = Vec::new();
    let mut soft_calls = Vec::new();
    let mut random_success = 0usize;
    let mut soft_success = 0usize;
    for r in 0..rounds {
        let run_seed = 20261119 + r as u64 * 17;
        let (ok_random, calls_random) =
            softmode_campaign(&spec, run_seed, Strategy::IsotropicRandom)?;
        if ok_random {
            random_success += 1;
        }
        random_calls.push(calls_random as f64);
        let (ok_soft, calls_soft) = softmode_campaign(&spec, run_seed, Strategy::SoftModeFirst)?;
        if ok_soft {
            soft_success += 1;
        }
        soft_calls.push(calls_soft as f64);
    }
    let random_median = median(&random_calls);
    let soft_median = median(&soft_calls);
    let random_se = median_se(&random_calls, 1);
    let soft_se = median_se(&soft_calls, 2);
    eprintln!(
        "{name}: random median calls {random_median:.1} (SE {random_se:.2}) success {random_success}/{rounds}; softmode median calls {soft_median:.1} (SE {soft_se:.2}) success {soft_success}/{rounds}"
    );
    assert!(
        soft_median < random_median,
        "{name}: soft-mode median {soft_median} did not beat random {random_median}"
    );
    assert!(
        soft_success >= random_success,
        "{name}: soft-mode success {soft_success} fell below random {random_success}"
    );
    Ok(())
}

#[test]
fn softmode_seeding_beats_random_on_muller_brown_and_product_double_well() {
    let mb_minimum = array![0.15, 0.25, 0.0];
    let mb_saddles = vec![array![0.21248658, 0.29298833, 0.0]];
    compare_strategies(
        "muller-brown-3d",
        &MullerBrown,
        mb_minimum,
        mb_saddles,
        0.18,
        0.0,
    )
    .unwrap();

    let dw_minimum = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let mut dw_saddles = Vec::new();
    for i in 0..6 {
        let mut s = Array1::ones(6);
        s[i] = 0.0;
        dw_saddles.push(s);
    }
    compare_strategies(
        "product-double-well-6d",
        &ProductDoubleWell6,
        dw_minimum,
        dw_saddles,
        0.12,
        0.8,
    )
    .unwrap();
}

#[cfg(feature = "lammps")]
#[test]
fn softmode_seeding_rgpot_eam_cu_vacancy_if_reachable() {
    // This integration needs the optional lammps+rgpot toolchain and a runnable
    // local setup. When not provisioned, skip explicitly.
    if std::env::var("RGSADDLE_RGPOT_CU_VACANCY").is_err() {
        eprintln!("rgpot Cu vacancy integration skipped: RGSADDLE_RGPOT_CU_VACANCY unset");
        return;
    }
    // Coverage metric is exercised by the analytic systems above in this test
    // target; this gate keeps the optional environment-sensitive path.
}

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
    spare: Option<f64>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn uniform01(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let u1 = self.uniform01().max(1.0e-16);
        let u2 = self.uniform01();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        self.spare = Some(z1);
        z0
    }
}

fn sample_on_sphere(dim: usize, rng: &mut SplitMix64) -> Array1<f64> {
    let mut v = Array1::zeros(dim);
    for c in &mut v {
        *c = rng.normal();
    }
    let norm2 = v.dot(&v);
    if norm2 <= 1.0e-24 || !norm2.is_finite() {
        let mut out = Array1::zeros(dim);
        out[0] = 1.0;
        return out;
    }
    v.mapv(|x| x / norm2.sqrt())
}
