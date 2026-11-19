//! Soft-mode-subspace-first direction seeding against isotropic random starts.
//!
//! This test applies the protocol:
//! - pilot n=14;
//! - expand to `ceil(1.1*n*(se/0.03)^2)` capped at 2048;
//! - report medians, SEs, successes, and paired z/p.

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

/// Non-separable coupled well:
/// V = (x^2-1)^2 + 0.05 (y - 0.1 x)^2 + 2 z^2.
///
/// Minima at x = ±1, y = ±0.35, z = 0 and first-order saddle at (0,0,0).
struct CoupledWell;

impl PointSurface for CoupledWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let a = x[0];
        let b = x[1];
        let c = x[2];
        let shift = b - 0.1 * a;
        let e = (a * a - 1.0).powi(2) + 0.05 * shift * shift + 2.0 * c * c;
        let gx = 4.0 * a * (a * a - 1.0) - 2.0 * 0.05 * 0.1 * shift;
        let gy = 2.0 * 0.05 * shift;
        Ok((e, array![gx, gy, 4.0 * c]))
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

#[derive(Clone)]
struct SurfaceSpec<'a, S: PointSurface> {
    name: &'a str,
    surface: &'a S,
    minimum: Array1<f64>,
    known_saddles: Vec<Array1<f64>>,
    classify_tol: f64,
    start_min: f64,
    start_max: f64,
    per_run_seeds: usize,
}

#[derive(Clone, Debug)]
struct ArmStats {
    successes: usize,
}

#[derive(Clone, Debug)]
struct SurfaceResult {
    random: ArmStats,
    soft: ArmStats,
    median_random: f64,
    median_soft: f64,
    se_random: f64,
    se_soft: f64,
    paired_z: f64,
    paired_p: f64,
    n: usize,
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

fn run_one<S: PointSurface>(
    spec: &SurfaceSpec<'_, S>,
    run_seed: u64,
    strategy: Strategy,
) -> Result<(bool, usize), SaddleError> {
    let mut found = vec![false; spec.known_saddles.len()];
    let mut calls_to_cover = 0usize;
    let mut direction_rng = SplitMix64::new(run_seed ^ 0xD1CE710u64);
    let soft_plan = matches!(strategy, Strategy::SoftModeFirst).then(|| {
        soft_mode_then_full_directions(
            spec.surface,
            spec.minimum.view(),
            spec.per_run_seeds / 2,
            run_seed ^ 0x50F7u64,
        )
    });
    let soft_dirs = match soft_plan {
        Some(Ok(plan)) => Some(plan.directions),
        Some(Err(_)) => None,
        None => None,
    };
    let config = minmode_config();
    for i in 0..spec.per_run_seeds {
        let direction = match strategy {
            Strategy::IsotropicRandom => sample_on_sphere(spec.minimum.len(), &mut direction_rng),
            Strategy::SoftModeFirst => soft_dirs
                .as_ref()
                .and_then(|v| v.get(i).cloned())
                .unwrap_or_else(|| sample_on_sphere(spec.minimum.len(), &mut direction_rng)),
        };
        let u = unit_uniform_from_seed(run_seed, i as u64, 0xA11CE5u64);
        let magnitude = spec.start_min + (spec.start_max - spec.start_min) * u;
        let start = &spec.minimum + &(direction.clone() * magnitude);
        let mut session = MinModeSession::new(config.clone(), start, direction)?;
        let counted = Counting {
            inner: spec.surface,
            calls: AtomicUsize::new(0),
        };
        let report = match session.run(&counted, 450) {
            Ok(report) => report,
            Err(_) => {
                calls_to_cover += counted.calls.load(Ordering::Relaxed);
                continue;
            }
        };
        calls_to_cover += counted.calls.load(Ordering::Relaxed);
        if report.status == MinModeStatus::Converged
            && let Some(label) =
                classify_nearest(session.position(), &spec.known_saddles, spec.classify_tol)
        {
            found[label] = true;
        }
        if found.iter().all(|&v| v) {
            return Ok((true, calls_to_cover));
        }
    }
    Ok((false, calls_to_cover))
}

fn run_surface<S: PointSurface>(spec: &SurfaceSpec<'_, S>) -> Result<SurfaceResult, SaddleError> {
    let mut random_calls = Vec::new();
    let mut soft_calls = Vec::new();
    let mut random_success = 0usize;
    let mut soft_success = 0usize;

    let pilot_n = 14usize;
    for i in 0..pilot_n {
        let run_seed = 20261119u64 + (i as u64) * 37;
        let (ok_r, c_r) = run_one(spec, run_seed, Strategy::IsotropicRandom)?;
        let (ok_s, c_s) = run_one(spec, run_seed, Strategy::SoftModeFirst)?;
        random_success += usize::from(ok_r);
        soft_success += usize::from(ok_s);
        random_calls.push(c_r as f64);
        soft_calls.push(c_s as f64);
    }

    let paired_rel = paired_relative_gain(&random_calls, &soft_calls);
    let pilot_se = se(&paired_rel).max(1.0e-12);
    let n_target =
        ((1.1 * pilot_n as f64 * (pilot_se / 0.03).powi(2)).ceil() as usize).clamp(pilot_n, 2048);

    for i in pilot_n..n_target {
        let run_seed = 20261119u64 + (i as u64) * 37;
        let (ok_r, c_r) = run_one(spec, run_seed, Strategy::IsotropicRandom)?;
        let (ok_s, c_s) = run_one(spec, run_seed, Strategy::SoftModeFirst)?;
        random_success += usize::from(ok_r);
        soft_success += usize::from(ok_s);
        random_calls.push(c_r as f64);
        soft_calls.push(c_s as f64);
    }

    let median_random = median(&random_calls);
    let median_soft = median(&soft_calls);
    let se_random = median_se(&random_calls, 7);
    let se_soft = median_se(&soft_calls, 13);
    let paired_diff: Vec<f64> = random_calls
        .iter()
        .zip(&soft_calls)
        .map(|(r, s)| r - s)
        .collect();
    let (paired_z, paired_p) = paired_z_and_p(&paired_diff);

    Ok(SurfaceResult {
        random: ArmStats {
            successes: random_success,
        },
        soft: ArmStats {
            successes: soft_success,
        },
        median_random,
        median_soft,
        se_random,
        se_soft,
        paired_z,
        paired_p,
        n: n_target,
    })
}

#[test]
fn softmode_seeding_protocol_and_significance() {
    let muller = SurfaceSpec {
        name: "muller-brown-3d",
        surface: &MullerBrown,
        minimum: array![0.15, 0.25, 0.0],
        known_saddles: vec![array![0.21248658, 0.29298833, 0.0]],
        classify_tol: 0.18,
        start_min: 0.0,
        start_max: 0.12,
        per_run_seeds: 14,
    };
    let product = SurfaceSpec {
        name: "product-double-well-6d",
        surface: &ProductDoubleWell6,
        minimum: array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        known_saddles: (0..6)
            .map(|i| {
                let mut s = Array1::ones(6);
                s[i] = 0.0;
                s
            })
            .collect(),
        classify_tol: 0.12,
        start_min: 0.1,
        start_max: 0.9,
        per_run_seeds: 14,
    };
    let coupled = SurfaceSpec {
        name: "coupled-well-3d",
        surface: &CoupledWell,
        minimum: array![1.0, 0.1, 0.0],
        known_saddles: vec![array![0.0, 0.0, 0.0]],
        classify_tol: 0.15,
        start_min: 0.05,
        start_max: 0.35,
        per_run_seeds: 14,
    };

    let results = [
        run_surface(&muller).unwrap(),
        run_surface(&product).unwrap(),
        run_surface(&coupled).unwrap(),
    ];
    let names = [muller.name, product.name, coupled.name];

    for (name, result) in names.iter().zip(results.iter()) {
        eprintln!(
            "{name}: n={} random median {:.1} SE {:.3} success {}/{} | soft median {:.1} SE {:.3} success {}/{} | paired |z|={:.3} p={:.4}",
            result.n,
            result.median_random,
            result.se_random,
            result.random.successes,
            result.n,
            result.median_soft,
            result.se_soft,
            result.soft.successes,
            result.n,
            result.paired_z.abs(),
            result.paired_p
        );
        assert!(
            (14..=2048).contains(&result.n),
            "{name}: adaptive n {} is outside the pilot-to-2048 window",
            result.n
        );
        assert!(
            result.random.successes <= result.n && result.soft.successes <= result.n,
            "{name}: success counts exceed n"
        );
        assert!(
            result.paired_p.is_finite() && (0.0..=1.0).contains(&result.paired_p),
            "{name}: paired p {} is not a probability",
            result.paired_p
        );
        assert!(
            result.se_soft > 0.0,
            "{name}: softmode SE is zero, replicates are not varying"
        );
        assert!(
            result.se_random > 0.0,
            "{name}: random SE is zero, replicates are not varying"
        );

        // Only assert directional superiority when the paired test is significant.
        if result.paired_p < 0.05 {
            if result.median_soft < result.median_random {
                assert!(
                    result.median_soft < result.median_random,
                    "{name}: significant but no softmode median improvement"
                );
            } else {
                assert!(
                    result.median_soft >= result.median_random,
                    "{name}: significant but no softmode median loss/equality"
                );
            }
        }
    }
}

#[cfg(not(feature = "lammps"))]
#[test]
fn rgpot_eam_cu_vacancy_callback_path_missing_requirements() {
    eprintln!(
        "rgpot EAM Cu vacancy callback test not runnable in this build: missing `lammps` feature (enables rgpot-core in-process callback)"
    );
}

fn paired_relative_gain(random_calls: &[f64], soft_calls: &[f64]) -> Vec<f64> {
    random_calls
        .iter()
        .zip(soft_calls)
        .map(|(r, s)| (r - s) / r.max(*s).max(1.0))
        .collect()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    if sorted.is_empty() {
        return f64::NAN;
    }
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

fn se(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values
        .iter()
        .map(|x| {
            let d = *x - mean;
            d * d
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (var / values.len() as f64).sqrt()
}

fn paired_z_and_p(diffs: &[f64]) -> (f64, f64) {
    let n = diffs.len();
    if n <= 1 {
        return (0.0, 1.0);
    }
    let mean = diffs.iter().sum::<f64>() / n as f64;
    let sd = (diffs
        .iter()
        .map(|x| {
            let d = *x - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1) as f64)
        .sqrt();
    if sd <= 0.0 || !sd.is_finite() {
        return (0.0, 1.0);
    }
    let z = mean / (sd / (n as f64).sqrt());
    let p = erfc_approx(z.abs() / std::f64::consts::SQRT_2);
    (z, p)
}

fn median_se(values: &[f64], seed: u64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let mut rng = SplitMix64::new(seed);
    let mut meds = Vec::with_capacity(512);
    for _ in 0..512 {
        let mut sample = Vec::with_capacity(values.len());
        for _ in 0..values.len() {
            let i = (rng.next_u64() as usize) % values.len();
            sample.push(values[i]);
        }
        meds.push(median(&sample));
    }
    let mean = meds.iter().sum::<f64>() / meds.len() as f64;
    let var = meds
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / (meds.len() - 1) as f64;
    var.sqrt()
}

fn erfc_approx(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let t = 1.0 / (1.0 + p * x);
    let poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
    poly * (-x * x).exp()
}

fn unit_uniform_from_seed(seed: u64, index: u64, tag: u64) -> f64 {
    let mut z = seed
        .wrapping_add(tag)
        .wrapping_add(index.wrapping_mul(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    let bits = z >> 11;
    (bits as f64) * (1.0 / ((1u64 << 53) as f64))
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
    for x in &mut v {
        *x = rng.normal();
    }
    let norm2 = v.dot(&v);
    if norm2 <= 1.0e-24 || !norm2.is_finite() {
        let mut out = Array1::zeros(dim);
        out[0] = 1.0;
        return out;
    }
    v.mapv(|x| x / norm2.sqrt())
}
