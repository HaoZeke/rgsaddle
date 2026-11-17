//! Shared bookkeeping for engine-backed NVE shooting.

use ndarray::Array1;

use super::{Endpoint, Hyperplane, TransmissionCoefficient, TransmissionConfig};
use crate::SaddleError;
use crate::stats::{Rng, block_mean};

/// Draw a flux-weighted Maxwell-Boltzmann Cartesian velocity and its exact reverse.
///
/// `cart_normal` is a unit normal in Cartesian coordinates, `thermal_vel` is
/// `kT` expressed in `mass * (length / time)^2`, and `sqrt_masses` has length
/// equal to `cart_normal.len()`.
#[cfg_attr(not(feature = "rgpot-session"), allow(dead_code))]
pub(crate) fn flux_velocity(
    cart_normal: &Array1<f64>,
    thermal_vel: f64,
    sqrt_masses: &Array1<f64>,
    seed: u64,
) -> Result<(Array1<f64>, Array1<f64>), SaddleError> {
    if !(thermal_vel.is_finite() && thermal_vel > 0.0) {
        return Err(SaddleError::Invalid("shooting kT must be positive".into()));
    }
    if cart_normal.len() != sqrt_masses.len() || cart_normal.is_empty() {
        return Err(SaddleError::Shape(
            "flux normal and masses must share configuration dimension".into(),
        ));
    }
    let mut flux_normal = cart_normal / sqrt_masses;
    let norm = flux_normal.dot(&flux_normal).sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(SaddleError::Invalid("flux normal vanishes".into()));
    }
    flux_normal /= norm;

    let mut rng = Rng::new(seed);
    let mut tangent: Array1<f64> = (0..cart_normal.len()).map(|_| rng.normal()).collect();
    let along = tangent.dot(&flux_normal);
    tangent.scaled_add(-along, &flux_normal);
    let rayleigh = (-2.0 * (1.0 - rng.uniform()).ln()).sqrt();
    let scale = thermal_vel.sqrt();
    let velocity = (tangent * scale + flux_normal * (rayleigh * scale)) / sqrt_masses;
    Ok((velocity.clone(), -velocity))
}

#[cfg_attr(not(feature = "rgpot-session"), allow(dead_code))]
pub(crate) fn checkpoints(config: &TransmissionConfig) -> Vec<usize> {
    let mut values: Vec<usize> = (1..=config.steps / config.check_every)
        .map(|index| index * config.check_every)
        .collect();
    if values.last() != Some(&config.steps) {
        values.push(config.steps);
    }
    values
}

#[cfg_attr(not(feature = "rgpot-session"), allow(dead_code))]
pub(crate) fn side(
    plane: &Hyperplane,
    coordinates: ndarray::ArrayView1<'_, f64>,
) -> Option<Endpoint> {
    let distance = plane.distance(coordinates);
    if distance < 0.0 {
        Some(Endpoint::Reactant)
    } else if distance > 0.0 {
        Some(Endpoint::Product)
    } else {
        None
    }
}

/// Aggregate checkpoint values with the same block standard error as OHTST.
#[cfg_attr(not(feature = "rgpot-session"), allow(dead_code))]
pub(crate) fn aggregate(
    values: &[Vec<Option<f64>>],
    config: &TransmissionConfig,
    recrossed: usize,
    evaluations: usize,
) -> Result<TransmissionCoefficient, SaddleError> {
    let checkpoint_steps = checkpoints(config);
    let last = checkpoint_steps.len() - 1;
    let finals: Vec<f64> = values.iter().filter_map(|shot| shot[last]).collect();
    if finals.len() < config.blocks {
        return Err(SaddleError::Invalid(
            "fewer completed shots than blocks".into(),
        ));
    }
    let (_, error, _, _) = block_mean(&finals, config.blocks);
    let kappa_of_t = (0..=last)
        .map(|checkpoint| {
            let complete: Vec<f64> = values.iter().filter_map(|shot| shot[checkpoint]).collect();
            if complete.is_empty() {
                f64::NAN
            } else {
                complete.iter().sum::<f64>() / complete.len() as f64
            }
        })
        .collect::<Vec<_>>();
    let reactive = finals.iter().filter(|value| **value > 0.5).count();
    Ok(TransmissionCoefficient {
        kappa: kappa_of_t[last],
        error,
        time: checkpoint_steps
            .iter()
            .map(|&step| step as f64 * config.time_step)
            .collect(),
        kappa_of_t,
        shots: finals.len(),
        unassigned: values.len() - finals.len(),
        reactive,
        recrossed,
        evaluations,
    })
}
