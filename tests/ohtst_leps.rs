//! OH-TST on the LEPS plus harmonic oscillator surface of Jóhannesson and
//! Jónsson, J. Chem. Phys. 115, 9644 (2001), Appendix B, at
//! `kT = 0.1 E_A` as in their Sec. III B and at lower temperatures.
//!
//! The references are quadrature of the line free energy
//! `A = -kT ln ∫ exp(-V/kT)` and of the reactant integral behind the
//! optimal line, with unit masses, computed in each test and independent of
//! the sampler, and harmonic TST through rgsaddle's dimer and Vineyard
//! functions. The paper gives its rate constants without units, so they
//! are not compared here.

use ndarray::array;
use rgsaddle::{
    Hyperplane, LangevinConfig, MinModeConfig, MinModeSession, MinModeStatus, PlaneIntegrator,
    PlaneOptimizer, PlaneOptimizerConfig, ReversibleWorkConfig, ShootingConfig,
    angular_scale_rad_s, exact_eigh, ohtst_rate, plane_transmission, reactant_density,
    reversible_work, vineyard_prefactor,
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
    LepsHo, descend, eigen2, half_plane_integral, hessian, line_free_energy, stationary,
};
use plane_manifold::PointSphereFixture;

/// Stationary points of the surface and its barrier `E_A`.
struct Geometry {
    reactant: [f64; 2],
    product: [f64; 2],
    saddle: [f64; 2],
    saddle_energy: f64,
    barrier: f64,
}

fn geometry() -> Geometry {
    let reactant = descend(LepsHo::energy, [0.75, 1.3]);
    let product = descend(LepsHo::energy, [3.0, -1.3]);
    let saddle = stationary(LepsHo::energy, [2.02, -0.17]);
    let saddle_energy = LepsHo::energy(saddle[0], saddle[1]).0;
    Geometry {
        reactant,
        product,
        saddle,
        saddle_energy,
        barrier: saddle_energy - LepsHo::energy(reactant[0], reactant[1]).0,
    }
}

impl Geometry {
    /// Angle of the reactant-to-product guideline.
    fn guideline(&self) -> f64 {
        (self.product[1] - self.reactant[1]).atan2(self.product[0] - self.reactant[0])
    }

    /// `Z_R`: the line through the reactant normal to the guideline.
    fn reactant_line(&self) -> Hyperplane {
        line(self.reactant, self.guideline())
    }

    /// The point `offset` along the normal at `theta` from the saddle.
    fn off_saddle(&self, offset: f64, theta: f64) -> [f64; 2] {
        let n = unit(theta);
        [
            self.saddle[0] + offset * n[0],
            self.saddle[1] + offset * n[1],
        ]
    }

    /// Angle of the saddle's unstable mode.
    fn saddle_angle(&self) -> f64 {
        let (_, _, mode) = eigen2(hessian(LepsHo::energy, self.saddle));
        (mode[1] / mode[0]).atan()
    }
}

fn unit(theta: f64) -> [f64; 2] {
    [theta.cos(), theta.sin()]
}

fn line(point: [f64; 2], theta: f64) -> Hyperplane {
    let n = unit(theta);
    Hyperplane::new(array![point[0], point[1]], array![n[0], n[1]]).unwrap()
}

fn angle_of(plane: &Hyperplane) -> f64 {
    plane.normal()[1].atan2(plane.normal()[0])
}

/// Quadrature line free energy, shifted by the saddle energy.
fn free_energy(geometry: &Geometry, point: [f64; 2], theta: f64, kt: f64) -> f64 {
    line_free_energy(
        LepsHo::energy,
        point,
        unit(theta),
        kt,
        geometry.saddle_energy,
        8.0,
    )
}

/// The optimal line by Newton's method on quadrature: its offset from the
/// saddle along its normal, its angle, and the Hessian of `A / kT` in
/// those two coordinates (offset, radians).
fn optimal_line(geometry: &Geometry, kt: f64) -> (f64, f64, [[f64; 2]; 2]) {
    let f = |d: f64, t: f64| free_energy(geometry, geometry.off_saddle(d, t), t, kt) / kt;
    let (mut d, mut t) = (0.0, geometry.saddle_angle());
    let h = 1.0e-3;
    let mut curvature = [[0.0; 2]; 2];
    for _ in 0..6 {
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

/// Quadrature OH-TST rate with unit masses, `sqrt(kT/2π) Z‡ / Q_R`, with
/// `Q_R` the integral over the reactant side of the optimal line.
fn quadrature_rate(geometry: &Geometry, kt: f64, offset: f64, theta: f64) -> f64 {
    let point = geometry.off_saddle(offset, theta);
    let reactant = half_plane_integral(
        LepsHo::energy,
        point,
        unit(theta),
        kt,
        geometry.saddle_energy,
        8.0,
    );
    let plane = (-free_energy(geometry, point, theta, kt) / kt).exp();
    (kt / (2.0 * std::f64::consts::PI)).sqrt() * plane / reactant
}

/// Harmonic TST with unit masses from the Hessian eigenvalues and
/// rgsaddle's Vineyard prefactor, in the surface's time unit.
fn harmonic_rate(geometry: &Geometry, kt: f64) -> f64 {
    let eigenvalues = |at: [f64; 2]| {
        let h = hessian(LepsHo::energy, at);
        exact_eigh(ndarray::arr2(&h).view()).unwrap().0
    };
    let (minimum, saddle) = (eigenvalues(geometry.reactant), eigenvalues(geometry.saddle));
    let prefactor = vineyard_prefactor(&[minimum[0], minimum[1]], &[saddle[1]], 1.0).unwrap()
        / angular_scale_rad_s();
    prefactor * (-geometry.barrier / kt).exp()
}

/// The plane of a dimer run from near the saddle, and its curvature.
fn dimer_plane() -> (Hyperplane, f64) {
    let config = MinModeConfig {
        force_tol: 1.0e-8,
        ..Default::default()
    };
    let mut session = MinModeSession::new(config, array![1.9, -0.05], array![1.0, 0.2]).unwrap();
    for _ in 0..2000 {
        let report = session.step(&LepsHo).unwrap();
        if report.status == MinModeStatus::Converged {
            return (
                Hyperplane::from_min_mode(&session).unwrap(),
                report.curvature,
            );
        }
    }
    panic!("the dimer did not converge");
}

fn sampler(kt: f64, time_step: f64) -> LangevinConfig {
    LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(time_step)
        .with_friction(1.0)
        .with_burn_in(300)
        .with_stride(5)
        .with_seed(5)
}

/// A step of 0.3 is past the BAOAB stability limit at the repulsive
/// walls, so these runs use the exact GHMC integrator instead.
fn large_step_sampler(kt: f64) -> LangevinConfig {
    sampler(kt, 0.3)
        .with_integrator(PlaneIntegrator::Ghmc)
        .with_records(1000)
        .with_seed(11)
}

/// Simpson's rule on quadrature `dA/ds` at the nodes of a reversible-work
/// leg from `(p0, θ0)` to `(p1, θ1)`, with the point linear in `s` and the
/// normal turning at a constant rate, in units of `kT`.
fn simpson_on_leg(
    geometry: &Geometry,
    kt: f64,
    (p0, t0): ([f64; 2], f64),
    (p1, t1): ([f64; 2], f64),
    nodes: usize,
) -> f64 {
    let at = |s: f64| {
        let point = [p0[0] + s * (p1[0] - p0[0]), p0[1] + s * (p1[1] - p0[1])];
        free_energy(geometry, point, t0 + s * (t1 - t0), kt) / kt
    };
    let h = 1.0e-4;
    let slopes: Vec<f64> = (0..nodes)
        .map(|i| {
            let s = i as f64 / (nodes - 1) as f64;
            (at(s + h) - at(s - h)) / (2.0 * h)
        })
        .collect();
    let width = 1.0 / (nodes - 1) as f64;
    let inner: f64 = slopes[1..nodes - 1]
        .iter()
        .enumerate()
        .map(|(i, v)| if i % 2 == 0 { 4.0 * v } else { 2.0 * v })
        .sum();
    (slopes[0] + inner + slopes[nodes - 1]) * width / 3.0
}

/// A dimer from near the saddle converges to the paper's saddle point,
/// (2.02083, -0.172881), and its plane is normal to the unstable mode.
/// That is the plane OH-TST starts from.
#[test]
fn leps_dimer_seeds_the_saddle_plane() {
    let geometry = geometry();
    let (plane, curvature) = dimer_plane();
    let point = plane.point();
    assert!((point[0] - 2.02083).abs() < 5.0e-5 && (point[1] + 0.172881).abs() < 5.0e-5);
    let (lowest, _, mode) = eigen2(hessian(LepsHo::energy, geometry.saddle));
    assert!((curvature - lowest).abs() < 1.0e-3 * lowest.abs());
    let along = (plane.normal()[0] * mode[0] + plane.normal()[1] * mode[1]).abs();
    assert!(1.0 - along < 1.0e-8, "normal off the mode by {along}");
}

/// At `kT = 0.1 E_A` the optimizer, translating and turning the dimer's
/// plane, reaches the quadrature optimum, -4.9174° and 0.00092 from the
/// saddle along its normal, within four errors (last force and torque
/// errors over the quadrature curvature). The saddle plane is at -4.6601°;
/// the optimum is resolved from it, the turn toward the ridge that Sec.
/// III B describes.
#[test]
fn leps_optimizer_climbs_from_the_dimer_plane() {
    let geometry = geometry();
    let kt = 0.1 * geometry.barrier;
    let (offset, theta, curvature) = optimal_line(&geometry, kt);
    let (seed, _) = dimer_plane();
    let config = PlaneOptimizerConfig::default()
        .with_sampler(large_step_sampler(kt))
        .with_time_step(1.0)
        .with_max_iterations(100);
    let mut optimizer = PlaneOptimizer::new(PointSphereFixture::new(2), &seed, config).unwrap();
    let optimum = optimizer.run(&LepsHo).unwrap();
    let last = optimum.trace.last().unwrap();
    let found = angle_of(&optimum.plane);
    let shift = optimum
        .plane
        .normal()
        .dot(&(&optimum.plane.point() - &array![geometry.saddle[0], geometry.saddle[1]]));
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
    assert!((found - angle_of(&seed)).abs() > 4.0 * angle_error);
}

/// The OH-TST rate at `kT = 0.1 E_A`. The density of the reactant
/// distribution at `Z_R`, behind a wall at the optimal plane, matches
/// quadrature (0.6509); the reversible work from `Z_R` to the optimal
/// plane matches quadrature (8.3081 kT; Simpson's rule on its slopes at
/// the same nodes is within 0.01 kT of that); and their rate matches the
/// quadrature rate, 3.8589e-5 in the surface's time unit, all within four
/// standard errors. Harmonic TST from the Vineyard prefactor gives
/// 4.0410e-5, a ratio of 0.955.
///
/// The transmission coefficient shot from the optimal plane with the
/// sampler's friction is a probability.
#[test]
fn leps_rate_against_quadrature_and_harmonic_tst() {
    let geometry = geometry();
    let kt = 0.1 * geometry.barrier;
    let (offset, theta, _) = optimal_line(&geometry, kt);
    let summit = geometry.off_saddle(offset, theta);
    let optimal = line(summit, theta);
    let start = geometry.reactant_line();
    let reactant = array![geometry.reactant[0], geometry.reactant[1]];

    let exact = quadrature_rate(&geometry, kt, offset, theta);
    let rise = (free_energy(&geometry, summit, theta, kt)
        - free_energy(&geometry, geometry.reactant, geometry.guideline(), kt))
        / kt;
    let density_exact = exact / (kt / (2.0 * std::f64::consts::PI)).sqrt() * rise.exp();

    let density = reactant_density(
        &LepsHo,
        &start,
        reactant.view(),
        std::slice::from_ref(&optimal),
        &sampler(kt, 0.1).with_records(16_000),
    )
    .unwrap();
    assert!(
        (density.density - density_exact).abs() < 4.0 * density.density_error,
        "density {} ± {} against {density_exact}",
        density.density,
        density.density_error
    );

    let nodes = 17;
    let layout = simpson_on_leg(
        &geometry,
        kt,
        (geometry.reactant, geometry.guideline()),
        (summit, theta),
        nodes,
    );
    assert!((layout - rise).abs() < 0.01, "{layout} against {rise}");
    let config = ReversibleWorkConfig::default()
        .with_sampler(sampler(kt, 0.1).with_records(2000).with_retain(128))
        .with_nodes(nodes);
    let work = reversible_work(&LepsHo, &[start, optimal], Some(reactant.view()), &config).unwrap();
    let (value, error) = (work.work / kt, work.error / kt);
    assert!(
        (value - rise).abs() < 4.0 * error,
        "work {value} ± {error} kT against {rise}"
    );

    let rate = ohtst_rate(&density, &work).unwrap();
    assert!(
        (rate.log_rate - exact.ln()).abs() < 4.0 * rate.log_rate_error,
        "rate {} ± {} against {exact}",
        rate.rate,
        rate.error
    );
    let ratio = exact / harmonic_rate(&geometry, kt);
    assert!((ratio - 0.95493).abs() < 1.0e-4, "ratio {ratio}");

    let shooting = ShootingConfig {
        thermal: kt,
        friction: 1.0,
        time_step: 0.05,
        steps: 400,
        record_every: 40,
        shots: 256,
        blocks: 8,
        plateau_record: 9,
        ..ShootingConfig::default()
    };
    let kappa = plane_transmission(&LepsHo, &work.end, &shooting).unwrap();
    assert!(
        kappa.kappa > 4.0 * kappa.error && kappa.kappa < 1.0 + 4.0 * kappa.error,
        "{kappa:?}"
    );
}

/// With the orientation held at the guideline normal, as in the paper's
/// Fig. 8(b), the free energy along the guideline peaks 2.3876 from the
/// reactant, 5.7344 kT above `Z_R` by quadrature against 8.3081 kT for
/// the optimal plane. The optimizer with rotation off finds that peak from
/// the plane through the saddle, and the reversible work to it matches.
/// Most of its error comes from the last nodes, where the line crosses the
/// ridge at a slant.
#[test]
fn leps_fixed_orientation_barrier() {
    let geometry = geometry();
    let kt = 0.1 * geometry.barrier;
    let theta = geometry.guideline();
    let n = unit(theta);
    let along = |o: f64| {
        let point = [
            geometry.reactant[0] + o * n[0],
            geometry.reactant[1] + o * n[1],
        ];
        free_energy(&geometry, point, theta, kt) / kt
    };
    let (mut a, mut b) = (2.0, 2.8);
    while b - a > 1.0e-6 {
        let third = (b - a) / 3.0;
        if along(a + third) > along(b - third) {
            b -= third;
        } else {
            a += third;
        }
    }
    let peak = 0.5 * (a + b);
    let h = 1.0e-3;
    let curvature = (along(peak + h) - 2.0 * along(peak) + along(peak - h)) / (h * h);
    let rise = along(peak) - along(0.0);
    assert!((peak - 2.38761).abs() < 1.0e-4 && (rise - 5.73441).abs() < 1.0e-4);

    let saddle_offset = n[0] * (geometry.saddle[0] - geometry.reactant[0])
        + n[1] * (geometry.saddle[1] - geometry.reactant[1]);
    let through = line(
        [
            geometry.reactant[0] + saddle_offset * n[0],
            geometry.reactant[1] + saddle_offset * n[1],
        ],
        theta,
    );
    let config = PlaneOptimizerConfig::default()
        .with_sampler(large_step_sampler(kt))
        .with_rotate(false)
        .with_recenter(false)
        .with_time_step(0.3)
        .with_max_iterations(100);
    let mut optimizer = PlaneOptimizer::new(PointSphereFixture::new(2), &through, config).unwrap();
    let optimum = optimizer.run(&LepsHo).unwrap();
    let found = optimum
        .plane
        .normal()
        .dot(&(&optimum.plane.point() - &array![geometry.reactant[0], geometry.reactant[1]]));
    let error = optimum.trace.last().unwrap().normal_force_error / kt / curvature.abs();
    assert!(
        (found - peak).abs() < 4.0 * error,
        "{found} ± {error} against {peak}"
    );

    let config = ReversibleWorkConfig::default()
        .with_sampler(sampler(kt, 0.1).with_records(2000))
        .with_nodes(17);
    let work = reversible_work(
        &LepsHo,
        &[geometry.reactant_line(), optimum.plane.clone()],
        Some(array![geometry.reactant[0], geometry.reactant[1]].view()),
        &config,
    )
    .unwrap();
    let top = [
        geometry.reactant[0] + found * n[0],
        geometry.reactant[1] + found * n[1],
    ];
    let exact = along(found) - along(0.0);
    let layout = simpson_on_leg(&geometry, kt, (geometry.reactant, theta), (top, theta), 17);
    assert!((layout - exact).abs() < 0.02, "{layout} against {exact}");
    let (value, error) = (work.work / kt, work.error / kt);
    assert!(
        (value - exact).abs() < 4.0 * error,
        "work {value} ± {error} kT against {exact}"
    );
}

/// As the temperature falls the optimal line turns back to the saddle
/// line and the OH-TST rate approaches harmonic TST: by quadrature the
/// ratio is 0.955 at `kT = 0.1 E_A`, 0.979 at 0.05 and 0.992 at 0.02,
/// and the optimum is 0.053° from the saddle line at 0.02. At
/// `kT = 0.02 E_A` the sampled rate matches the quadrature rate and
/// harmonic TST within four standard errors.
#[test]
fn leps_rate_approaches_harmonic_tst_at_low_temperature() {
    let geometry = geometry();
    let saddle_angle = geometry.saddle_angle();
    let mut previous = f64::INFINITY;
    for fraction in [0.1, 0.05, 0.02] {
        let kt = fraction * geometry.barrier;
        let (offset, theta, _) = optimal_line(&geometry, kt);
        let ratio = quadrature_rate(&geometry, kt, offset, theta) / harmonic_rate(&geometry, kt);
        let gap = 1.0 - ratio;
        assert!(
            gap > 0.0 && gap < previous && gap < 0.5 * fraction,
            "ratio {ratio} at {fraction}"
        );
        previous = gap;
        let turn = (theta - saddle_angle).abs().to_degrees();
        assert!(turn < 3.0 * fraction, "{turn}° at {fraction}");
    }

    let kt = 0.02 * geometry.barrier;
    let (offset, theta, _) = optimal_line(&geometry, kt);
    let optimal = line(geometry.off_saddle(offset, theta), theta);
    let start = geometry.reactant_line();
    let reactant = array![geometry.reactant[0], geometry.reactant[1]];
    let density = reactant_density(
        &LepsHo,
        &start,
        reactant.view(),
        std::slice::from_ref(&optimal),
        &sampler(kt, 0.1).with_records(16_000),
    )
    .unwrap();
    let config = ReversibleWorkConfig::default()
        .with_sampler(sampler(kt, 0.1).with_records(4000))
        .with_nodes(17);
    let work = reversible_work(&LepsHo, &[start, optimal], Some(reactant.view()), &config).unwrap();
    let rate = ohtst_rate(&density, &work).unwrap();
    assert!(rate.log_rate_error < 0.25, "{rate:?}");
    for (reference, name) in [
        (quadrature_rate(&geometry, kt, offset, theta), "quadrature"),
        (harmonic_rate(&geometry, kt), "harmonic"),
    ] {
        assert!(
            (rate.log_rate - reference.ln()).abs() < 4.0 * rate.log_rate_error,
            "rate {} ± {} against {name} {reference}",
            rate.rate,
            rate.error
        );
    }
}
