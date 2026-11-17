//! PAFI against a potential of mean force that can be integrated by hand.
//!
//! The estimator is Swinburne and Marinica, Phys. Rev. Lett. 120, 135503
//! (2018), doi:10.1103/PhysRevLett.120.135503: `dF/ds = ⟨(∇E · n)(1-ψ)⟩`.

use ndarray::{Array1, ArrayView1};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use rgsaddle::{
    ArcPath, PafiConfig, PathGeometry, PointSurface, SaddleError, anchored_barriers,
    average_hyperplane, balanced_barriers, integrate_profile, pafi_frames, pafi_profile,
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
fn reverse_barrier_runs_from_the_last_plane() {
    let profile = integrate_profile(
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 2.0, -1.0, -3.0],
        &[0.1, 0.1, 0.2, 0.2],
    )
    .expect("profile");
    assert_eq!(profile.dividing, 2);
    assert!((profile.barrier - 1.5).abs() < 1.0e-12);
    let (barrier, error) = profile.reverse_barrier();
    assert!((barrier - 2.0).abs() < 1.0e-12, "reverse {barrier}");
    assert!((error - 0.02_f64.sqrt()).abs() < 1.0e-12, "error {error}");
}

/// The profile above ends 0.5 below its start. Told that the ends differ
/// by 0.1, the two barriers weigh in by their errors and the remaining
/// -0.6 is drift.
#[test]
fn balanced_barriers_close_on_the_end_gap() {
    let profile = integrate_profile(
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 2.0, -1.0, -3.0],
        &[0.1, 0.1, 0.2, 0.2],
    )
    .expect("profile");
    let balanced = profile.balanced(0.1).expect("balanced");
    let (forward, reverse) = ((1.5, profile.barrier_error), profile.reverse_barrier());
    let (weight_f, weight_r) = (forward.1.powi(-2), reverse.1.powi(-2));
    let mean = (weight_f * forward.0 + weight_r * (reverse.0 + 0.1)) / (weight_f + weight_r);
    assert!(
        (balanced.drift + 0.6).abs() < 1.0e-12,
        "drift {}",
        balanced.drift
    );
    assert!((balanced.forward - mean).abs() < 1.0e-12);
    assert!((balanced.forward - balanced.reverse - 0.1).abs() < 1.0e-12);
    let error = ((weight_f + weight_r).recip() + 0.09).sqrt();
    assert!(
        (balanced.error - error).abs() < 1.0e-12,
        "error {}",
        balanced.error
    );
    let even = balanced_barriers((1.0, 0.1), (0.8, 0.1), 0.0).expect("even");
    assert!((even.forward - 0.9).abs() < 1.0e-12 && (even.reverse - 0.9).abs() < 1.0e-12);
    assert!(balanced_barriers((1.0, 0.0), (0.8, 0.1), 0.0).is_err());
    assert!(balanced_barriers((1.0, 0.1), (f64::NAN, 0.1), 0.0).is_err());
}

/// The same profile anchored on its forward barrier: the reverse barrier
/// is the forward one less the end gap, the error is the forward one, and
/// the drift is the same -0.6.
#[test]
fn anchored_barriers_keep_the_forward_barrier() {
    let profile = integrate_profile(
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 2.0, -1.0, -3.0],
        &[0.1, 0.1, 0.2, 0.2],
    )
    .expect("profile");
    let anchored = profile.anchored(0.1).expect("anchored");
    assert!((anchored.forward - 1.5).abs() < 1.0e-12);
    assert!((anchored.reverse - 1.4).abs() < 1.0e-12);
    assert!((anchored.error - profile.barrier_error).abs() < 1.0e-12);
    assert!(
        (anchored.drift + 0.6).abs() < 1.0e-12,
        "drift {}",
        anchored.drift
    );
    assert!(anchored_barriers((1.0, 0.0), (0.8, 0.1), 0.0).is_err());
    assert!(anchored_barriers((1.0, 0.1), (0.8, 0.1), f64::INFINITY).is_err());
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

/// A corner between images. The integral is along the chords, so it
/// recovers the energy of the corner instead of cutting it.
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

const RING_STIFFNESS: f64 = 20.0;
const RING_STIFFENING: f64 = 1.0;
const RING_BARRIER: f64 = 1.0;
const RING_THERMAL: f64 = 0.1;

/// `V = A (1 + a sin²(2θ)) (r - 1)² + W sin²(2θ)`: minima on the axes and
/// saddles on the diagonals of a ring, so every hop between neighbouring
/// axes is the same and its path is curved. The ring is stiffer at the
/// saddle, and the free-energy barrier is `W + ½ kT ln(1 + a)`.
struct Ring;

impl PointSurface for Ring {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let r = x[0].hypot(x[1]);
        let theta = x[1].atan2(x[0]);
        let rise = (2.0 * theta).sin().powi(2);
        let stretch = r - 1.0;
        let stiffness = RING_STIFFNESS * (1.0 + RING_STIFFENING * rise);
        let energy = stiffness * stretch * stretch + RING_BARRIER * rise;
        let radial = 2.0 * stiffness * stretch;
        let angular = 2.0
            * (4.0 * theta).sin()
            * (RING_STIFFNESS * RING_STIFFENING * stretch * stretch + RING_BARRIER)
            / (r * r);
        let gradient = Array1::from_vec(vec![
            radial * x[0] / r - angular * x[1],
            radial * x[1] / r + angular * x[0],
        ]);
        Ok((energy, gradient))
    }
}

/// Unevenly spaced frames on the ring from the x axis to the y axis, the
/// saddle among them, turned by `turn` and mirrored across the x axis
/// when asked.
fn ring_frames(turn: f64, mirror: bool) -> Vec<Array1<f64>> {
    [
        0.0, 0.06, 0.15, 0.27, 0.42, 0.6, FRAC_PI_4, 1.0, 1.15, 1.3, 1.45, FRAC_PI_2,
    ]
    .iter()
    .map(|angle| {
        let angle = if mirror { -angle } else { *angle } + turn;
        Array1::from_vec(vec![angle.cos(), angle.sin()])
    })
    .collect()
}

/// The planes through a reversed band are the same planes, and the arc
/// length runs the other way.
#[test]
fn frame_planes_do_not_depend_on_the_traversal() {
    let frames = ring_frames(0.0, false);
    let back: Vec<Array1<f64>> = frames.iter().rev().cloned().collect();
    let forward = ArcPath::from_positions(&frames)
        .expect("path")
        .frame_planes(1.0)
        .expect("planes");
    let backward = ArcPath::from_positions(&back)
        .expect("path")
        .frame_planes(1.0)
        .expect("planes");
    let last = frames.len() - 1;
    let total = forward.arcs[last];
    for k in 0..=last {
        let sum = &forward.normals[k] + &backward.normals[last - k];
        assert!(sum.iter().all(|v| v.abs() < 1e-12), "normal {k}");
        let turn = &forward.curvatures[k] - &backward.curvatures[last - k];
        assert!(turn.iter().all(|v| v.abs() < 1e-10), "curvature {k}");
        assert!((forward.arcs[k] - (total - backward.arcs[last - k])).abs() < 1e-12);
    }
}

/// On the ring the normals are the tangents of the circle and the
/// curvature points to its centre, to second order in the spacing.
#[test]
fn frame_planes_follow_the_ring() {
    let frames = ring_frames(0.0, false);
    let planes = ArcPath::from_positions(&frames)
        .expect("path")
        .frame_planes(1.0)
        .expect("planes");
    for (k, frame) in frames.iter().enumerate() {
        let tangent = Array1::from_vec(vec![-frame[1], frame[0]]);
        let tilt = 1.0 - planes.normals[k].dot(&tangent);
        let bend = &planes.curvatures[k] + frame;
        assert!(tilt < 1e-3, "frame {k} tilts by {tilt}");
        assert!(bend.dot(&bend).sqrt() < 0.1, "frame {k} bends by {bend}");
    }
}

/// An atom that barely moves is left out of the planes at a share its
/// squared steps do not reach.
#[test]
fn frame_planes_keep_the_atoms_that_carry_the_path() {
    let frames: Vec<Array1<f64>> = (0..5)
        .map(|i| {
            let t = i as f64 / 4.0;
            Array1::from_vec(vec![t, 0.0, 0.0, 5.0 + 0.1 * t, 0.0, 0.0])
        })
        .collect();
    let path = ArcPath::from_positions(&frames).expect("path");
    let local = path.frame_planes(0.95).expect("planes");
    assert_eq!(local.active, vec![0]);
    assert!(
        local
            .normals
            .iter()
            .all(|n| n[3] == 0.0 && (n[0] - 1.0).abs() < 1e-12)
    );
    assert!((local.arcs[4] - 1.0).abs() < 1e-12);
    let every = path.frame_planes(1.0).expect("planes");
    assert_eq!(every.active, vec![0, 1]);
    assert!((every.arcs[4] - 1.01_f64.sqrt()).abs() < 1e-12);
}

/// Over unevenly spaced frames the barrier is the same on the hop turned
/// onto the next axis and on the hop mirrored, a reversed band gives the
/// barrier back from the other end, and the barrier keeps the entropic
/// part `½ kT ln(1 + a)` to a tenth.
#[test]
fn frame_barrier_does_not_depend_on_the_frame() {
    let config = PafiConfig::default()
        .with_thermal(RING_THERMAL)
        .with_mobility(1.0)
        .with_time_step(0.01)
        .with_thermalize(200)
        .with_stride(1)
        .with_samples(20_000)
        .with_blocks(8)
        .with_seed(5)
        .with_fix_com(false)
        .with_retain(0)
        .with_target_error(None);
    let barrier = |frames: Vec<Array1<f64>>| {
        let path = ArcPath::from_positions(&frames).expect("path");
        let (profile, _) = pafi_frames(&Ring, &path, &config).expect("pafi");
        let back = profile.reverse_barrier();
        ((profile.barrier, profile.barrier_error), back)
    };
    let (built, built_back) = barrier(ring_frames(0.0, false));
    let (reversed, reversed_back) = barrier(ring_frames(0.0, false).into_iter().rev().collect());
    let (turned, _) = barrier(ring_frames(FRAC_PI_2, false));
    let (mirrored, _) = barrier(ring_frames(0.0, true));
    let entropic = 0.5 * RING_THERMAL * (1.0 + RING_STIFFENING).ln();
    eprintln!(
        "as built {built:?} back {built_back:?}, reversed {reversed:?} back {reversed_back:?}, turned {turned:?}, mirrored {mirrored:?}, exact {}",
        RING_BARRIER + entropic
    );
    let agree = |left: (f64, f64), right: (f64, f64)| {
        (left.0 - right.0).abs() < 3.0 * left.1.hypot(right.1)
    };
    assert!(agree(built, turned), "turned {turned:?} against {built:?}");
    assert!(
        agree(built, mirrored),
        "mirrored {mirrored:?} against {built:?}"
    );
    assert!(
        agree(built, reversed_back),
        "reversed {reversed_back:?} against {built:?}"
    );
    assert!(
        agree(built_back, reversed),
        "reversed {reversed:?} against {built_back:?}"
    );
    for (label, (value, error)) in [("as built", built), ("reversed", reversed)] {
        assert!(
            (value - RING_BARRIER - entropic).abs() < 3.0 * error + 0.1 * entropic,
            "{label}: {value} ± {error} against {}",
            RING_BARRIER + entropic
        );
    }
}

const BASIN_BARRIER: f64 = 0.4;
const BASIN_COUPLING: f64 = 0.3;
const BASIN_THERMAL: f64 = 0.1;
const BASIN_PLANE: f64 = 0.5;

/// `V = (1 - x²)² + B (y² - 1)² + λ x y`: the plane `x = ½` cuts two
/// basins in `y`, the one below the deeper, and `∂V/∂x = λ y - 4x(1 - x²)`
/// tells them apart.
struct TwoBasins;

impl PointSurface for TwoBasins {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (p, q) = (x[0], x[1]);
        let energy =
            (1.0 - p * p).powi(2) + BASIN_BARRIER * (q * q - 1.0).powi(2) + BASIN_COUPLING * p * q;
        let gradient = Array1::from_vec(vec![
            -4.0 * p * (1.0 - p * p) + BASIN_COUPLING * q,
            4.0 * BASIN_BARRIER * q * (q * q - 1.0) + BASIN_COUPLING * p,
        ]);
        Ok((energy, gradient))
    }
}

/// Boltzmann average of `∂V/∂x` on the plane over `y` in `(from, to)`.
fn basin_gradient(from: f64, to: f64) -> f64 {
    let steps = 40_000;
    let width = (to - from) / steps as f64;
    let (mut weight, mut sum) = (0.0, 0.0);
    for i in 0..=steps {
        let y = from + width * i as f64;
        let point = Array1::from_vec(vec![BASIN_PLANE, y]);
        let (energy, gradient) = TwoBasins.eval(point.view()).expect("energy");
        let w = (-energy / BASIN_THERMAL).exp();
        weight += w;
        sum += w * gradient[0];
    }
    sum / weight
}

/// On a plane that also cuts a deeper basin, the jump test keeps the chain
/// in the basin it starts from, and the average is that basin's alone.
/// Without the test the chain falls into the other one.
#[test]
fn jump_test_keeps_the_basin_of_the_path() {
    let geometry = PathGeometry {
        positions: Array1::from_vec(vec![BASIN_PLANE, 1.0]),
        tangent: Array1::from_vec(vec![1.0, 0.0]),
        curvature: Array1::zeros(2),
        arc: 0.0,
    };
    let config = PafiConfig::default()
        .with_thermal(BASIN_THERMAL)
        .with_mobility(1.0)
        .with_time_step(0.01)
        .with_thermalize(200)
        .with_stride(1)
        .with_samples(40_000)
        .with_window(200)
        .with_blocks(8)
        .with_seed(3)
        .with_fix_com(false)
        .with_retain(0);
    let kept = average_hyperplane(
        &TwoBasins,
        &geometry,
        &config.clone().with_max_jump(Some(0.5)),
    )
    .expect("kept");
    let free = average_hyperplane(&TwoBasins, &geometry, &config).expect("free");
    let mut top = 0.0;
    for _ in 0..50 {
        let slope = 4.0 * BASIN_BARRIER * top * (top * top - 1.0) + BASIN_COUPLING * BASIN_PLANE;
        let bend = 4.0 * BASIN_BARRIER * (3.0 * top * top - 1.0);
        top -= slope / bend;
    }
    let upper = basin_gradient(top, 4.0);
    let lower = basin_gradient(-4.0, top);
    eprintln!(
        "kept {:.4} ± {:.4} after {} jumps, free {:.4} ± {:.4}, upper basin {upper:.4}, lower {lower:.4}",
        kept.gradient, kept.error, kept.jumps, free.gradient, free.error
    );
    assert!(kept.jumps > 0);
    assert_eq!(free.jumps, 0);
    assert!(
        (kept.gradient - upper).abs() < 3.0 * kept.error + 0.02,
        "kept {} ± {} against {upper}",
        kept.gradient,
        kept.error
    );
    assert!((free.gradient - upper).abs() > 0.25 * (upper - lower).abs());
}
