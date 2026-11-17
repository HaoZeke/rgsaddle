//! Shared bookkeeping for engine-backed NVE shooting.

use ndarray::Array1;

use super::{Endpoint, Hyperplane, TransmissionCoefficient, TransmissionConfig};
use crate::SaddleError;
use crate::stats::{Rng, block_mean};

/// Draw a flux-weighted Maxwell-Boltzmann velocity and its exact reverse.
pub(crate) fn flux_velocity(
    normal: &Array1<f64>,
    thermal: f64,
    seed: u64,
) -> Result<(Array1<f64>, Array1<f64>), SaddleError> {
    if !(thermal.is_finite() && thermal > 0.0) {
        return Err(SaddleError::Invalid("shooting kT must be positive".into()));
    }
    let mut rng = Rng::new(seed);
    let mut tangent: Array1<f64> = (0..normal.len()).map(|_| rng.normal()).collect();
    let along = tangent.dot(normal);
    tangent.scaled_add(-along, normal);
    let rayleigh = (-2.0 * (1.0 - rng.uniform()).ln()).sqrt();
    let velocity = tangent * thermal.sqrt() + normal * (rayleigh * thermal.sqrt());
    Ok((velocity.clone(), -velocity))
}

#[allow(dead_code)]
pub(crate) fn checkpoints(config: &TransmissionConfig) -> Vec<usize> {
    let mut values: Vec<usize> = (1..=config.steps / config.check_every)
        .map(|index| index * config.check_every)
        .collect();
    if values.last() != Some(&config.steps) {
        values.push(config.steps);
    }
    values
}

#[allow(dead_code)]
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
#[allow(dead_code)]
pub(crate) fn aggregate(
    values: &[Vec<Option<f64>>],
    config: &TransmissionConfig,
) -> Result<TransmissionCoefficient, SaddleError> {
    let last = config.steps / config.check_every;
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
            complete.iter().sum::<f64>() / complete.len() as f64
        })
        .collect::<Vec<_>>();
    let reactive = finals.iter().filter(|value| **value > 0.5).count();
    Ok(TransmissionCoefficient {
        kappa: kappa_of_t[last],
        error,
        time: (0..=last)
            .map(|index| index as f64 * config.check_every as f64 * config.time_step)
            .collect(),
        kappa_of_t,
        shots: finals.len(),
        unassigned: values.len() - finals.len(),
        reactive,
        recrossed: 0,
        evaluations: 0,
    })
}
