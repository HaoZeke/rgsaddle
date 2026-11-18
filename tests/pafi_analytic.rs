//! PAFI against a potential of mean force that can be integrated by hand.
//!
//! The estimator is Swinburne and Marinica, Phys. Rev. Lett. 120, 135503
//! (2018), doi:10.1103/PhysRevLett.120.135503: `dF/ds = ⟨(∇E · n)(1-ψ)⟩`.

use ndarray::{Array1, ArrayView1};
use rgsaddle::{
    ArcPath, PafiConfig, PathFrame, PointSurface, SaddleError, anchored_barriers,
    average_hyperplane, balanced_barriers, integrate_profile, pafi_profile,
    zero_temperature_barrier,
};

struct AlongX;

impl PointSurface for AlongX {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let coordinate = x[0];
        let energy = (1.0 - coordinate * coordinate).powi(2);
        let mut gradient = Array1::zeros(x.len());
        gradient[0] = -4.0 * coordinate * (1.0 - coordinate * coordinate);
        Ok((energy, gradient))
    }
}

/// `V = (1-x²)² + ½ k(x) y²`, with `k = 1.5 + 0.5 x`.
struct Widening;

impl PointSurface for Widening {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gx, gy) = widening(x[0], x[1]);
        Ok((energy, Array1::from_vec(vec![gx, gy])))
    }
}

fn widening(x: f64, y: f64) -> (f64, f64, f64) {
    let well = (1.0 - x * x).powi(2);
    let stiffness = 1.5 + 0.5 * x;
    let energy = well + 0.5 * stiffness * y * y;
    let d_well = -4.0 * x * (1.0 - x * x);
    let gx = d_well + 0.25 * y * y;
    let gy = stiffness * y;
    (energy, gx, gy)
}

fn line(samples: usize) -> Vec<Array1<f64>> {
    (0..samples)
        .map(|i| {
            let x = -1.0 + i as f64 / (samples - 1) as f64;
            Array1::from_vec(vec![x])
        })
        .collect()
}

#[test]
fn one_dimensional_barrier_is_the_potential() {
    let frames = line(41);
    let path = ArcPath::from_positions(&frames).expect("path");
    let config = PafiConfig::default()
        .with_thermal(0.02)
        .with_mobility(1.0)
        .with_time_step(0.01)
        .with_thermalize(5)
        .with_stride(1)
        .with_samples(8)
        .with_blocks(2)
        .with_seed(3)
        .with_fix_com(false)
        .with_retain(1)
        .with_target_error(None)
        .with_max_rounds(1);
    let geometry = path.geometry(0.5).expect("mid");
    // x = -0.5 sits halfway in arc length on this uniform line.
    assert!((geometry.positions[0] + 0.5).abs() < 1e-9);
    let config = config.with_samples(4);
    let plane = average_hyperplane(&AlongX, &geometry, &config).expect("plane");
    let expected = -4.0 * (-0.5) * (1.0 - 0.25);
    assert!(
        (plane.gradient - expected).abs() < 1e-9,
        "gradient {} expected {expected}",
        plane.gradient
    );
    assert!(plane.psi.abs() < 1e-9);
    assert!(plane.error < 1e-12);
    let arcs = path.arcs();
    let (profile, _) = pafi_profile(&AlongX, &path, &arcs, &config).expect("profile");
    assert!(
        (profile.barrier - 1.0).abs() < 2.0e-3,
        "barrier {}",
        profile.barrier
    );
    assert_eq!(profile.dividing, arcs.len() - 1);
}

#[test]
fn transverse_width_shifts_the_free_energy() {
    let thermal = 0.05;
    let truth = 1.0 + 0.5 * thermal * 1.5_f64.ln();
    let frames: Vec<Array1<f64>> = (0..11)
        .map(|i| {
            let x = -1.0 + i as f64 / 10.0;
            Array1::from_vec(vec![x, 0.0])
        })
        .collect();
    let path = ArcPath::from_positions(&frames).expect("path");
    let config = PafiConfig::default()
        .with_thermal(thermal)
        .with_mobility(1.0)
        .with_time_step(0.004)
        .with_thermalize(400)
        .with_stride(8)
        .with_samples(600)
        .with_blocks(6)
        .with_seed(11)
        .with_fix_com(false)
        .with_retain(2)
        .with_target_error(None)
        .with_max_rounds(1);
    let (profile, _) = pafi_profile(&Widening, &path, &path.arcs(), &config).expect("profile");
    let miss = (profile.barrier - truth).abs();
    assert!(
        miss < 0.02 && miss < 4.0 * profile.barrier_error.max(0.005),
        "barrier {} truth {truth} error {}",
        profile.barrier,
        profile.barrier_error
    );
}

#[test]
fn zero_temperature_integral_recovers_the_barrier() {
    let frames: Vec<Array1<f64>> = (0..41)
        .map(|i| {
            let x = -1.0 + 2.0 * i as f64 / 40.0;
            Array1::from_vec(vec![x])
        })
        .collect();
    let path = ArcPath::from_positions(&frames).expect("path");
    let profile = zero_temperature_barrier(&AlongX, &path, &path.arcs()).expect("integral");
    let end = profile.free_energy.last().copied().unwrap_or(f64::NAN);
    assert!(
        (profile.barrier - 1.0).abs() < 3.0e-3 && end.abs() < 2.0e-2,
        "barrier {} end {end} dividing {}",
        profile.barrier,
        profile.dividing
    );
    assert!(profile.dividing > 0 && profile.dividing < profile.arc.len() - 1);
}

/// A corner between images. The spline passes through the corner image,
/// so the integral reaches the energy of the corner instead of cutting it.
#[test]
fn zero_temperature_integral_keeps_a_corner() {
    let frames = vec![
        Array1::from_vec(vec![-1.0, 0.0]),
        Array1::from_vec(vec![0.0, 0.4]),
        Array1::from_vec(vec![1.0, 0.0]),
    ];
    let path = ArcPath::from_positions(&frames).expect("path");
    let profile = zero_temperature_barrier(&Corner, &path, &path.arcs()).expect("integral");
    let expected = 1.0 + 4.0 * 0.16;
    assert!(
        (profile.barrier - expected).abs() < 2.0e-3,
        "barrier {} expected {expected}",
        profile.barrier
    );
}

/// `V = (1-x²)² + ½ k (y - a(1-x²))²` with `a = 0.5`, `k = 4`: a valley
/// bent up from the minima at `(±1, 0)` to the saddle at `(0, a)`.
struct Bent;

impl PointSurface for Bent {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gx, gy) = bent(x[0], x[1]);
        Ok((energy, Array1::from_vec(vec![gx, gy])))
    }
}

fn bent(x: f64, y: f64) -> (f64, f64, f64) {
    let (bend, stiffness) = (0.5, 4.0);
    let floor = bend * (1.0 - x * x);
    let rise = y - floor;
    let energy = (1.0 - x * x).powi(2) + 0.5 * stiffness * rise * rise;
    let gx = -4.0 * x * (1.0 - x * x) + stiffness * rise * 2.0 * bend * x;
    (energy, gx, stiffness * rise)
}

/// `-kT ln ∫ exp(-V/kT) dl` over the line `X(s) + l m`, `m ⟂ X'(s)`, by
/// the trapezoid rule on `|l| ≤ 1.5`. These lines leave the valley well
/// inside that range.
fn line_free_energy(path: &ArcPath, arc: f64, thermal: f64) -> f64 {
    let geometry = path.geometry(arc).expect("geometry");
    let (tx, ty) = (geometry.tangent[0], geometry.tangent[1]);
    let norm = tx.hypot(ty);
    let (mx, my) = (-ty / norm, tx / norm);
    let (x, y) = (geometry.positions[0], geometry.positions[1]);
    let points = 30_001;
    let step = 3.0 / (points - 1) as f64;
    let energies: Vec<f64> = (0..points)
        .map(|k| {
            let l = -1.5 + step * k as f64;
            bent(x + l * mx, y + l * my).0
        })
        .collect();
    let low = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let weights: Vec<f64> = energies
        .iter()
        .map(|energy| (-(energy - low) / thermal).exp())
        .collect();
    let sum = weights.iter().sum::<f64>() - 0.5 * (weights[0] + weights[points - 1]);
    low - thermal * (sum * step).ln()
}

/// Five images on the floor of a bent valley at `kT = 0.1`. The gradient
/// on every plane is the derivative of the free energy of the spline
/// hyperplanes, `-kT ln Z` by quadrature over each, and the barrier is
/// the trapezoid sum of those derivatives. Hyperplanes normal to the
/// chords between the images have derivatives up to 0.18 away.
#[test]
fn bent_valley_follows_the_spline_hyperplanes() {
    let thermal = 0.1;
    let frames: Vec<Array1<f64>> = (0..5)
        .map(|i| {
            let x = -1.0 + 0.5 * i as f64;
            Array1::from_vec(vec![x, 0.5 * (1.0 - x * x)])
        })
        .collect();
    let path = ArcPath::from_positions(&frames).expect("path");
    let knots = path.arcs();
    let mut arcs = Vec::new();
    for pair in knots.windows(2) {
        arcs.push(pair[0]);
        arcs.push(0.5 * (pair[0] + pair[1]));
    }
    arcs.push(knots[knots.len() - 1]);
    let config = PafiConfig::default()
        .with_thermal(thermal)
        .with_mobility(1.0)
        .with_time_step(0.004)
        .with_thermalize(400)
        .with_stride(8)
        .with_samples(1500)
        .with_blocks(10)
        .with_seed(17)
        .with_fix_com(false)
        .with_retain(1);
    let (profile, planes) = pafi_profile(&Bent, &path, &arcs, &config).expect("profile");
    let delta = 1.0e-4;
    let exact: Vec<f64> = arcs
        .iter()
        .map(|arc| {
            (line_free_energy(&path, arc + delta, thermal)
                - line_free_energy(&path, arc - delta, thermal))
                / (2.0 * delta)
        })
        .collect();
    for (k, (plane, truth)) in planes.iter().zip(&exact).enumerate() {
        assert!(
            (plane.gradient - truth).abs() < 4.0 * plane.error + 2.0e-3 && plane.error < 0.05,
            "plane {k}: dF/ds {} exact {truth} error {}",
            plane.gradient,
            plane.error
        );
        assert!(plane.psi < 1.0);
    }
    let errors = vec![0.0; arcs.len()];
    let truth = integrate_profile(&arcs, &exact, &errors).expect("exact profile");
    assert!(
        (profile.barrier - truth.barrier).abs() < 4.0 * profile.barrier_error + 1.0e-3,
        "barrier {} exact {} error {}",
        profile.barrier,
        truth.barrier,
        profile.barrier_error
    );
}

/// Not-a-knot splines reproduce a cubic, so the path through samples of
/// one is that cubic, with its first two derivatives, at any parameter.
#[test]
fn spline_reproduces_a_cubic() {
    let cubic = |s: f64| [s * s * s - 2.0 * s + 1.0, 0.5 * s * s + s, -s];
    let slope = |s: f64| [3.0 * s * s - 2.0, s + 1.0, -1.0];
    let bend = |s: f64| [6.0 * s, 1.0, 0.0];
    for knots in [vec![0.0, 0.4, 1.1, 2.0], vec![0.0, 0.3, 0.5, 1.2, 1.6, 2.0]] {
        let frames = knots
            .iter()
            .map(|s| PathFrame {
                positions: Array1::from_vec(cubic(*s).to_vec()),
                arc: *s,
            })
            .collect();
        let path = ArcPath::new(frames).expect("path");
        for k in 0..=40 {
            let s = 2.0 * k as f64 / 40.0;
            let geometry = path.geometry(s).expect("geometry");
            for (got, want) in [
                (&geometry.positions, cubic(s)),
                (&geometry.tangent, slope(s)),
                (&geometry.curvature, bend(s)),
            ] {
                for (a, b) in got.iter().zip(want) {
                    assert!((a - b).abs() < 1.0e-12, "s {s}: {a} against {b}");
                }
            }
        }
    }
}

/// Two frames are the straight line and three the parabola through them.
#[test]
fn two_frames_are_a_line_and_three_a_parabola() {
    let line = ArcPath::from_positions(&[
        Array1::from_vec(vec![0.0, 1.0]),
        Array1::from_vec(vec![3.0, 5.0]),
    ])
    .expect("line");
    let middle = line.geometry(2.5).expect("middle");
    assert!((middle.positions[0] - 1.5).abs() < 1.0e-12);
    assert!((middle.positions[1] - 3.0).abs() < 1.0e-12);
    assert!((middle.tangent[0] - 0.6).abs() < 1.0e-12);
    assert!(middle.curvature.iter().all(|value| *value == 0.0));
    let frames = [(0.0, 0.0), (1.0, 2.0), (3.0, 0.0)]
        .iter()
        .map(|(s, y)| PathFrame {
            positions: Array1::from_vec(vec![*y]),
            arc: *s,
        })
        .collect();
    let parabola = ArcPath::new(frames).expect("parabola");
    let truth = |s: f64| s * (3.0 - s);
    for s in [0.0, 0.5, 1.5, 2.0, 3.0] {
        let geometry = parabola.geometry(s).expect("geometry");
        assert!(
            (geometry.positions[0] - truth(s)).abs() < 1.0e-12,
            "s {s}: {}",
            geometry.positions[0]
        );
        assert!((geometry.curvature[0] + 2.0).abs() < 1.0e-12);
    }
}

/// The reversed path is the same curve from the other end.
#[test]
fn reversed_path_traverses_the_same_curve() {
    let frames: Vec<Array1<f64>> = (0..6)
        .map(|i| {
            let x = -1.0 + 0.4 * i as f64;
            Array1::from_vec(vec![x, 0.5 * (1.0 - x * x) + 0.1 * x * x * x])
        })
        .collect();
    let path = ArcPath::from_positions(&frames).expect("path");
    let reversed = path.reversed().expect("reversed");
    let (first, last) = path.bounds();
    assert_eq!(reversed.bounds(), (first, last));
    for k in 0..=50 {
        let s = first + (last - first) * k as f64 / 50.0;
        let ahead = path.geometry(s).expect("ahead");
        let back = reversed.geometry(first + last - s).expect("back");
        for (a, b, sign) in [
            (&ahead.positions, &back.positions, 1.0),
            (&ahead.tangent, &back.tangent, -1.0),
            (&ahead.curvature, &back.curvature, 1.0),
        ] {
            for (x, y) in a.iter().zip(b) {
                assert!((x - sign * y).abs() < 1.0e-12, "s {s}: {x} against {y}");
            }
        }
    }
}

struct Corner;

impl PointSurface for Corner {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (x0, y) = (x[0], x[1]);
        let well = (1.0 - x0 * x0).powi(2);
        let energy = well + 4.0 * y * y;
        let mut gradient = Array1::zeros(x.len());
        gradient[0] = -4.0 * x0 * (1.0 - x0 * x0);
        gradient[1] = 8.0 * y;
        Ok((energy, gradient))
    }
}
