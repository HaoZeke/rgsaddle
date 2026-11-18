//! Point-surface and hyperplane-sampler basin densities on the same
//! analytic surfaces, plus a hard-wall rate assembled from the sampler.

use ndarray::array;
use rgsaddle::{
    AnalyticSurfaceSampler, BasinConfig, BasinDensity, Hyperplane, LangevinConfig,
    ReversibleWorkConfig, WorkEstimate, basin_density, basin_density_with_sampler, ohtst_rate,
    reversible_work,
};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/ohtst_surfaces.rs"
));

use ohtst_surfaces::{LepsHo, SkewedWell, descend, half_plane_integral, line_integral, stationary};

fn agree(point: &BasinDensity, sampled: &BasinDensity, reference: f64) {
    let between = (point.density - sampled.density).abs();
    let statistical = 4.0 * point.density_error.hypot(sampled.density_error);
    let quadrature =
        point.quadrature_error * point.density + sampled.quadrature_error * sampled.density;
    assert!(
        between <= statistical + quadrature,
        "point {} ± {} against sampler {} ± {}",
        point.density,
        point.density_error,
        sampled.density,
        sampled.density_error
    );
    for result in [point, sampled] {
        let gap = (result.density - reference).abs();
        let allowance = 4.0 * result.density_error + result.quadrature_error * reference;
        assert!(
            gap <= allowance,
            "{} ± {} against {reference}",
            result.density,
            result.density_error
        );
    }
}

fn both(
    surface: &impl rgsaddle::PointSurface,
    minimum: [f64; 2],
    normal: [f64; 2],
    kt: f64,
    seed: u64,
) -> (BasinDensity, BasinDensity) {
    let config = BasinConfig::default().with_sampler(
        LangevinConfig::default()
            .with_thermal(kt)
            .with_time_step(0.1)
            .with_friction(1.0)
            .with_burn_in(200)
            .with_stride(5)
            .with_records(1200)
            .with_seed(seed),
    );
    let minimum = array![minimum[0], minimum[1]];
    let normal = array![normal[0], normal[1]];
    let point = basin_density(surface, minimum.view(), normal.view(), &config).unwrap();
    let mut engine = AnalyticSurfaceSampler::new(surface, config.sampler.clone());
    let sampled =
        basin_density_with_sampler(&mut engine, minimum.view(), normal.view(), &config).unwrap();
    (point, sampled)
}

/// Parallel planes on the skewed well, seed 9, against the half-plane
/// quadrature used by the point-surface basin test.
#[test]
fn skewed_well_sampler_matches_the_point_surface() {
    let kt = 0.1;
    let normal = [1.0, 1.0];
    let reactant = [-1.0, -1.0];
    let reference = line_integral(SkewedWell::energy, reactant, normal, kt, 0.0, 8.0)
        / half_plane_integral(SkewedWell::energy, [0.0, 0.0], normal, kt, 0.0, 8.0);
    let (point, sampled) = both(&SkewedWell, reactant, normal, kt, 9);
    agree(&point, &sampled, reference);
}

/// The same comparison on LEPS plus a harmonic oscillator, seed 5.
#[test]
fn leps_sampler_matches_the_point_surface() {
    let reactant = descend(LepsHo::energy, [0.75, 1.3]);
    let product = descend(LepsHo::energy, [3.0, -1.3]);
    let saddle = stationary(LepsHo::energy, [2.02, -0.17]);
    let barrier =
        LepsHo::energy(saddle[0], saddle[1]).0 - LepsHo::energy(reactant[0], reactant[1]).0;
    let kt = 0.1 * barrier;
    let theta = (product[1] - reactant[1]).atan2(product[0] - reactant[0]);
    let normal = [theta.cos(), theta.sin()];
    let reference = line_integral(LepsHo::energy, reactant, normal, kt, 0.0, 8.0)
        / half_plane_integral(LepsHo::energy, saddle, normal, kt, 0.0, 8.0);
    let (point, sampled) = both(&LepsHo, reactant, normal, kt, 5);
    agree(&point, &sampled, reference);
}

/// A wall parallel to the sampling normal, then reversible work and the rate.
#[test]
fn hard_wall_sampler_assembles_a_rate() {
    let kt = 0.1;
    let reactant = [-1.0, -1.0];
    let normal = [1.0, 1.0];
    let transverse = array![1.0, -1.0];
    let mut face = array![reactant[0], reactant[1]];
    face.scaled_add(0.35, &transverse);
    let wall = Hyperplane::new(face, transverse).unwrap();
    let sampling = LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(0.1)
        .with_friction(1.0)
        .with_burn_in(200)
        .with_stride(5)
        .with_records(600)
        .with_seed(9)
        .with_wall(wall);
    let config = BasinConfig::default().with_sampler(sampling.clone());
    let minimum = array![reactant[0], reactant[1]];
    let direction = array![normal[0], normal[1]];
    let point = basin_density(&SkewedWell, minimum.view(), direction.view(), &config).unwrap();
    let mut engine = AnalyticSurfaceSampler::new(&SkewedWell, sampling.clone());
    let sampled =
        basin_density_with_sampler(&mut engine, minimum.view(), direction.view(), &config).unwrap();
    let between = (point.density - sampled.density).abs();
    let allowance = 4.0 * point.density_error.hypot(sampled.density_error)
        + point.quadrature_error * point.density
        + sampled.quadrature_error * sampled.density;
    assert!(
        between <= allowance,
        "walled point {} ± {} against sampler {} ± {}",
        point.density,
        point.density_error,
        sampled.density,
        sampled.density_error
    );

    let mut shifted = sampled.plane.point().to_owned();
    shifted.scaled_add(0.15, &sampled.plane.normal());
    let destination = Hyperplane::new(shifted, sampled.plane.normal().to_owned()).unwrap();
    let work_config = ReversibleWorkConfig::default()
        .with_sampler(sampling)
        .with_nodes(3)
        .with_bar_samples(0)
        .with_estimate(WorkEstimate::Integration);
    let work = reversible_work(
        &SkewedWell,
        &[sampled.plane.clone(), destination],
        None,
        &work_config,
    )
    .unwrap();
    let rate = ohtst_rate(&sampled, &work).unwrap();
    assert!(rate.rate.is_finite() && rate.rate > 0.0, "{}", rate.rate);
}
