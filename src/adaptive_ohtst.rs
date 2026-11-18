//! Adaptive finite-temperature strings of optimal hyperplanes.
//!
//! Unlike fixed-path PAFI, this method translates and rotates interior
//! hyperplanes under their sampled OH-TST free-energy gradients.

use crate::SaddleError;
use crate::ohtst::{Hyperplane, LangevinConfig, OhTstRate, PlaneDensity, PlaneSample};
use crate::projector::HyperplaneSampler;

/// Mandatory first sample count.
pub const ADAPTIVE_OHTST_PILOT_SAMPLES: usize = 14;
/// Target standard error of the sampled log-rate contribution.
pub const ADAPTIVE_OHTST_TARGET_LOG_ERROR: f64 = 0.03;
/// Largest retained sample count at one plane.
pub const ADAPTIVE_OHTST_MAX_SAMPLES: usize = 2048;
/// Prescribed sample-growth safety factor.
pub const ADAPTIVE_OHTST_SAMPLE_SAFETY: f64 = 1.1;

/// Controls for [`adaptive_ohtst_string_rate`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct AdaptiveOhTstConfig {
    /// Shared constrained-sampler controls. The method owns `records`.
    pub sampler: LangevinConfig,
    /// Geometry-relaxation iteration cap.
    pub max_iterations: usize,
    /// Plane cap after profile refinement.
    pub max_planes: usize,
}

impl Default for AdaptiveOhTstConfig {
    fn default() -> Self {
        Self {
            sampler: LangevinConfig::default(),
            max_iterations: 64,
            max_planes: 257,
        }
    }
}

impl AdaptiveOhTstConfig {
    /// Set all sampler controls except the method-owned record count.
    #[must_use]
    pub fn with_sampler(mut self, sampler: LangevinConfig) -> Self {
        self.sampler = sampler;
        self
    }

    /// Set `kT`.
    #[must_use]
    pub fn with_thermal(mut self, thermal: f64) -> Self {
        self.sampler.thermal = thermal;
        self
    }

    /// Set the geometry-relaxation cap.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the adaptive plane cap.
    #[must_use]
    pub fn with_max_planes(mut self, max_planes: usize) -> Self {
        self.max_planes = max_planes;
        self
    }
}

/// A free-energy profile on a relaxed finite-temperature plane string.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct AdaptiveOhTstProfile {
    /// Ordered relaxed planes.
    pub planes: Vec<Hyperplane>,
    /// Free energy relative to the first plane.
    pub free_energy: Vec<f64>,
    /// Propagated standard error at every plane.
    pub free_energy_error: Vec<f64>,
    /// Largest profile value.
    pub barrier: f64,
    /// Standard error of [`Self::barrier`].
    pub barrier_error: f64,
    /// Sampling contribution to [`Self::barrier_error`].
    pub sampled_barrier_error: f64,
    /// Measured barrier change under neighbor-smoothed normals and a trust-limited step.
    pub geometry_error: f64,
    /// Measured force-integral discrepancy from grid doubling and the offset rule.
    pub quadrature_error: f64,
    /// Index of the selected dividing plane.
    pub dividing: usize,
    /// Records requested at each final plane.
    pub samples_per_plane: Vec<usize>,
    /// Shared samples supplying translation, rotation, and work.
    pub plane_samples: Vec<PlaneSample>,
}

/// Adaptive OH-TST string barrier composed with the one-way flux.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct AdaptiveOhTstEstimate {
    /// Rate and its propagated standard error.
    pub rate: OhTstRate,
    /// Relaxed path, profile, barrier, and profile uncertainty.
    pub profile: AdaptiveOhTstProfile,
    /// Plane at the profile maximum.
    pub dividing_plane: Hyperplane,
    /// Geometry-relaxation iterations.
    pub iterations: usize,
    /// Force or energy evaluations.
    pub evaluations: usize,
    /// Sampling termination status.
    pub status: AdaptiveOhTstStatus,
}

/// Whether adaptive plane sampling met its prescribed log-error target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveOhTstStatus {
    /// The total log-rate error, including geometry and quadrature, reached the target.
    TargetMet,
    /// The sample cap was reached with the total error still above the target.
    /// A sample that is already precise counts here when geometry or quadrature
    /// still misses the target: the cap, not the sample size, is what stopped the run.
    SampleCap,
    /// Sampling is already at its target, but geometry or quadrature keeps the
    /// total error above it, and the record cap was not reached. Further records
    /// would not change that.
    Unresolved,
}

/// Apply the prescribed sample-growth rule.
pub fn adaptive_ohtst_sample_count(current: usize, error: f64) -> Result<usize, SaddleError> {
    if current == 0 || current > ADAPTIVE_OHTST_MAX_SAMPLES || !(error.is_finite() && error >= 0.0)
    {
        return Err(SaddleError::Invalid(
            "adaptive sampling needs a count inside its cap and finite non-negative error".into(),
        ));
    }
    let ratio = error / ADAPTIVE_OHTST_TARGET_LOG_ERROR;
    let proposed = (ADAPTIVE_OHTST_SAMPLE_SAFETY * current as f64 * ratio * ratio).ceil() as usize;
    Ok(proposed.max(current).min(ADAPTIVE_OHTST_MAX_SAMPLES))
}

/// Relax an OH-TST hyperplane string and estimate its rate.
pub fn adaptive_ohtst_string_rate<E, D>(
    _sampler: &mut E,
    _initial: &[Hyperplane],
    _density: &D,
    _config: &AdaptiveOhTstConfig,
) -> Result<AdaptiveOhTstEstimate, SaddleError>
where
    E: HyperplaneSampler,
    D: PlaneDensity + ?Sized,
{
    Err(SaddleError::Solver(
        "adaptive OH-TST string estimator is not implemented".into(),
    ))
}
