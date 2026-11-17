//! Thermodynamic integration on one fixed hyperplane.

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::projector::{Hyperplane, HyperplaneEnsemble, HyperplaneSampleBudget, HyperplaneSampler};
use crate::stats::{EquilibrationReport, block_mean, equilibrate};

use super::{AnalyticSurfaceSampler, LangevinConfig};

/// Stable observable name for `V_full - V_harmonic`.
pub const THERMODYNAMIC_DERIVATIVE: &str = "potential.full_minus_harmonic";

/// Settings for fixed-plane lambda thermodynamic integration.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ThermodynamicIntegrationConfig {
    /// `kT` in the engine's energy unit.
    pub thermal: f64,
    /// Gauss--Legendre nodes on `[0, 1]`.
    pub nodes: usize,
    /// Minimum engine warm-up at every node.
    pub thermalize: usize,
    /// Engine steps between records.
    pub stride: usize,
    /// Records requested at every node.
    pub samples: usize,
    /// Most blocks used for each node's error.
    pub blocks: usize,
    /// First deterministic node seed.
    pub seed: u64,
    /// Named observation containing `∂V_lambda/∂lambda`.
    pub observable: String,
}

impl Default for ThermodynamicIntegrationConfig {
    fn default() -> Self {
        Self {
            thermal: 0.02585,
            nodes: 8,
            thermalize: 500,
            stride: 5,
            samples: 2_000,
            blocks: 20,
            seed: 1,
            observable: THERMODYNAMIC_DERIVATIVE.into(),
        }
    }
}

impl ThermodynamicIntegrationConfig {
    /// Set [`Self::thermal`].
    #[must_use]
    pub fn with_thermal(mut self, thermal: f64) -> Self {
        self.thermal = thermal;
        self
    }

    /// Set [`Self::nodes`].
    #[must_use]
    pub fn with_nodes(mut self, nodes: usize) -> Self {
        self.nodes = nodes;
        self
    }

    /// Set [`Self::thermalize`].
    #[must_use]
    pub fn with_thermalize(mut self, thermalize: usize) -> Self {
        self.thermalize = thermalize;
        self
    }

    /// Set [`Self::stride`].
    #[must_use]
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set [`Self::samples`].
    #[must_use]
    pub fn with_samples(mut self, samples: usize) -> Self {
        self.samples = samples;
        self
    }

    /// Set [`Self::blocks`].
    #[must_use]
    pub fn with_blocks(mut self, blocks: usize) -> Self {
        self.blocks = blocks;
        self
    }

    /// Set [`Self::seed`].
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set [`Self::observable`].
    #[must_use]
    pub fn with_observable(mut self, observable: impl Into<String>) -> Self {
        self.observable = observable.into();
        self
    }

    fn check(&self) -> Result<(), SaddleError> {
        if !(self.thermal.is_finite() && self.thermal > 0.0) {
            return Err(SaddleError::Invalid(
                "thermodynamic integration needs a finite positive temperature".into(),
            ));
        }
        if self.nodes == 0
            || self.stride == 0
            || self.blocks < 2
            || self.samples < 2 * self.blocks
            || self.observable.is_empty()
        {
            return Err(SaddleError::Invalid(
                "thermodynamic integration needs nodes, a stride, two blocks, two records per block, and an observable".into(),
            ));
        }
        Ok(())
    }
}

/// One lambda node and its block estimate.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ThermodynamicIntegrationNode {
    /// Coupling parameter in `[0, 1]`.
    pub lambda: f64,
    /// Gauss--Legendre weight on `[0, 1]`.
    pub weight: f64,
    /// Mean `∂V_lambda/∂lambda`.
    pub derivative: f64,
    /// Block standard error of [`Self::derivative`].
    pub error: f64,
    /// Integrated autocorrelation time in retained records.
    pub tau: f64,
    /// Effective retained records.
    pub effective_samples: f64,
    /// Engine evaluations reported at this node.
    pub evaluations: usize,
    /// Automated equilibration and drift report.
    pub equilibration: EquilibrationReport,
    /// Hard-boundary reflections in the region's planar-face order for
    /// this lambda window.
    pub hard_reflections: Vec<usize>,
}

/// `F_full - F_harmonic` on one fixed hyperplane.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ThermodynamicIntegrationEstimate {
    /// Gauss--Legendre estimate of the free-energy difference.
    pub free_energy: f64,
    /// Propagated independent-node standard error.
    pub error: f64,
    /// Estimates in ascending lambda order.
    pub nodes: Vec<ThermodynamicIntegrationNode>,
    /// Total engine evaluations.
    pub evaluations: usize,
}

/// Estimate `F_full - F_harmonic` from named sampler observations.
pub fn thermodynamic_integration<E: HyperplaneSampler>(
    sampler: &mut E,
    plane: &Hyperplane,
    config: &ThermodynamicIntegrationConfig,
) -> Result<ThermodynamicIntegrationEstimate, SaddleError> {
    config.check()?;
    let quadrature = gauss_legendre(config.nodes)?;
    let mut nodes = Vec::with_capacity(config.nodes);
    let mut free_energy = 0.0;
    let mut variance = 0.0;
    let mut evaluations = 0;
    for (index, (lambda, weight)) in quadrature.into_iter().enumerate() {
        let budget = HyperplaneSampleBudget::default()
            .with_thermal(config.thermal)
            .with_seed(config.seed.wrapping_add(index as u64))
            .with_thermalize(config.thermalize)
            .with_stride(config.stride)
            .with_samples(config.samples)
            .with_positions(false)
            .with_lambda(lambda)
            .with_observable(config.observable.clone());
        let ensemble = sampler.sample_hyperplane(plane, &budget)?;
        if ensemble.observations.len() != config.samples {
            return Err(SaddleError::Invalid(
                "TI sampler did not return the requested records".into(),
            ));
        }
        let values: Vec<f64> = ensemble
            .observations
            .iter()
            .map(|observation| {
                observation
                    .observables
                    .get(&config.observable)
                    .copied()
                    .ok_or_else(|| {
                        SaddleError::Invalid(format!(
                            "TI sampler omitted observable {}",
                            config.observable
                        ))
                    })
            })
            .collect::<Result<_, _>>()?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(SaddleError::NonFinite("TI observable"));
        }
        let equilibration = equilibrate(&values, config.thermalize, config.stride, config.blocks)?;
        let (derivative, error, tau, effective_samples) =
            block_mean(&values[equilibration.discard..], config.blocks);
        free_energy += weight * derivative;
        variance += (weight * error).powi(2);
        evaluations += ensemble.evaluations;
        nodes.push(ThermodynamicIntegrationNode {
            lambda,
            weight,
            derivative,
            error,
            tau,
            effective_samples,
            evaluations: ensemble.evaluations,
            equilibration,
            hard_reflections: ensemble.hard_reflections,
        });
    }
    Ok(ThermodynamicIntegrationEstimate {
        free_energy,
        error: variance.sqrt(),
        nodes,
        evaluations,
    })
}

/// Amsel-style restricted molecular-dynamics TI on a fixed plane.
///
/// This is the fixed-plane spelling of [`thermodynamic_integration`].
/// The engine samples
/// `V_lambda = (1-lambda) V_harmonic + lambda V_full` while preserving the
/// plane constraint and emits [`THERMODYNAMIC_DERIVATIVE`] (or the
/// configured observable). The estimator deliberately does not move or
/// optimize the plane: callers first choose the dividing plane, then use
/// this operation for its anharmonic free-energy factor.
pub fn restricted_thermodynamic_integration<E: HyperplaneSampler>(
    sampler: &mut E,
    plane: &Hyperplane,
    config: &ThermodynamicIntegrationConfig,
) -> Result<ThermodynamicIntegrationEstimate, SaddleError> {
    thermodynamic_integration(sampler, plane, config)
}

/// Analytic reference backend for
/// `V_lambda = (1-lambda) V_harmonic + lambda V_full`.
pub struct AnalyticThermodynamicSampler<'a, H, F> {
    harmonic: &'a H,
    full: &'a F,
    dynamics: LangevinConfig,
}

impl<'a, H: PointSurface, F: PointSurface> AnalyticThermodynamicSampler<'a, H, F> {
    /// Build the reference backend.
    pub fn new(harmonic: &'a H, full: &'a F, dynamics: LangevinConfig) -> Self {
        Self {
            harmonic,
            full,
            dynamics,
        }
    }
}

struct Interpolated<'a, H, F> {
    harmonic: &'a H,
    full: &'a F,
    lambda: f64,
}

impl<H: PointSurface, F: PointSurface> PointSurface for Interpolated<'_, H, F> {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (harmonic_energy, harmonic_gradient) = self.harmonic.eval(q)?;
        let (full_energy, full_gradient) = self.full.eval(q)?;
        let lambda = self.lambda;
        Ok((
            (1.0 - lambda) * harmonic_energy + lambda * full_energy,
            harmonic_gradient * (1.0 - lambda) + full_gradient * lambda,
        ))
    }

    fn excluded_modes(&self, q: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.full.excluded_modes(q)
    }
}

impl<H: PointSurface, F: PointSurface> HyperplaneSampler
    for AnalyticThermodynamicSampler<'_, H, F>
{
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        let lambda = budget.lambda.ok_or_else(|| {
            SaddleError::Invalid("analytic TI sampling needs a lambda request".into())
        })?;
        if !(0.0..=1.0).contains(&lambda) {
            return Err(SaddleError::Invalid(
                "thermodynamic lambda must lie in [0, 1]".into(),
            ));
        }
        let surface = Interpolated {
            harmonic: self.harmonic,
            full: self.full,
            lambda,
        };
        let mut inner_budget = budget.clone();
        inner_budget.positions = true;
        let mut sampler = AnalyticSurfaceSampler::new(&surface, self.dynamics.clone());
        let mut ensemble = sampler.sample_hyperplane(plane, &inner_budget)?;
        for observation in &mut ensemble.observations {
            let position = observation.position.as_ref().ok_or_else(|| {
                SaddleError::Invalid("analytic TI sampler omitted a position".into())
            })?;
            let (harmonic, _) = self.harmonic.eval(position.view())?;
            let (full, _) = self.full.eval(position.view())?;
            ensemble.evaluations += 2;
            observation
                .observables
                .insert(THERMODYNAMIC_DERIVATIVE.into(), full - harmonic);
            if !budget.positions {
                observation.position = None;
            }
        }
        Ok(ensemble)
    }
}

fn gauss_legendre(count: usize) -> Result<Vec<(f64, f64)>, SaddleError> {
    if count == 0 {
        return Err(SaddleError::Invalid(
            "Gauss-Legendre quadrature needs at least one node".into(),
        ));
    }
    let mut nodes = vec![(0.0, 0.0); count];
    let half = count.div_ceil(2);
    for index in 0..half {
        let mut root = (std::f64::consts::PI * (index as f64 + 0.75) / (count as f64 + 0.5)).cos();
        let mut derivative = 0.0;
        for _ in 0..64 {
            let (value, slope) = legendre(count, root);
            derivative = slope;
            let next = root - value / slope;
            if (next - root).abs() <= 4.0 * f64::EPSILON * (1.0 + root.abs()) {
                root = next;
                break;
            }
            root = next;
        }
        let weight = 1.0 / ((1.0 - root * root) * derivative * derivative);
        let low = 0.5 * (1.0 - root);
        let high = 0.5 * (1.0 + root);
        nodes[index] = (low, weight);
        nodes[count - 1 - index] = (high, weight);
    }
    Ok(nodes)
}

fn legendre(order: usize, x: f64) -> (f64, f64) {
    let (mut previous, mut current) = (1.0, x);
    if order == 1 {
        return (current, 1.0);
    }
    for degree in 2..=order {
        let next = ((2 * degree - 1) as f64 * x * current - (degree - 1) as f64 * previous)
            / degree as f64;
        previous = current;
        current = next;
    }
    let derivative = order as f64 * (previous - x * current) / (1.0 - x * x);
    (current, derivative)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::ohtst::{ExactGaussianPlaneSampler, QuadraticModel};

    struct Harmonic;

    impl PointSurface for Harmonic {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((0.5 * q[1] * q[1], array![0.0, q[1]]))
        }
    }

    struct Quartic;

    impl PointSurface for Quartic {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((0.25 * q[1].powi(4), array![0.0, q[1].powi(3)]))
        }
    }

    #[test]
    fn gauss_legendre_integrates_polynomials_through_its_exact_degree() {
        let nodes = gauss_legendre(4).unwrap();
        for power in 0_i32..=7 {
            let integral: f64 = nodes
                .iter()
                .map(|(node, weight)| weight * node.powi(power))
                .sum();
            let exact = 1.0 / (power + 1) as f64;
            assert!((integral - exact).abs() < 1.0e-13);
        }
    }

    #[test]
    fn exact_gaussian_gives_zero_for_an_identical_reference() {
        let model = QuadraticModel::new(array![0.0, 0.0], array![[1.0, 0.0], [0.0, 2.0]]);
        let difference = |_q: ArrayView1<f64>| Ok(0.0);
        let mut sampler = ExactGaussianPlaneSampler::new(model)
            .unwrap()
            .with_observable(THERMODYNAMIC_DERIVATIVE, &difference);
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let config = ThermodynamicIntegrationConfig::default()
            .with_nodes(4)
            .with_samples(200)
            .with_blocks(4);
        let estimate = thermodynamic_integration(&mut sampler, &plane, &config).unwrap();
        assert_eq!(estimate.free_energy, 0.0);
        assert_eq!(estimate.error, 0.0);
    }

    #[test]
    fn restricted_spelling_uses_the_same_sampler_estimator() {
        let model = QuadraticModel::new(array![0.0, 0.0], array![[1.0, 0.0], [0.0, 2.0]]);
        let difference = |_q: ArrayView1<f64>| Ok(0.0);
        let mut sampler = ExactGaussianPlaneSampler::new(model)
            .unwrap()
            .with_observable(THERMODYNAMIC_DERIVATIVE, &difference);
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let config = ThermodynamicIntegrationConfig::default()
            .with_nodes(2)
            .with_samples(100)
            .with_blocks(4);
        let estimate = restricted_thermodynamic_integration(&mut sampler, &plane, &config).unwrap();
        assert_eq!(estimate.free_energy, 0.0);
    }

    #[test]
    fn anharmonic_free_energy_matches_the_partition_functions() {
        let thermal = 0.5;
        let dynamics = LangevinConfig::default()
            .with_thermal(thermal)
            .with_time_step(0.03)
            .with_friction(2.0);
        let mut sampler = AnalyticThermodynamicSampler::new(&Harmonic, &Quartic, dynamics);
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let config = ThermodynamicIntegrationConfig::default()
            .with_thermal(thermal)
            .with_nodes(6)
            .with_thermalize(500)
            .with_stride(8)
            .with_samples(4_000)
            .with_blocks(16)
            .with_seed(41);
        let estimate = thermodynamic_integration(&mut sampler, &plane, &config).unwrap();
        // Direct endpoint partition integrals, evaluated independently of TI.
        let width = 8.0;
        let intervals = 200_000;
        let integrate = |potential: &dyn Fn(f64) -> f64| {
            let h = 2.0 * width / intervals as f64;
            (0..=intervals)
                .map(|index| {
                    let y = -width + index as f64 * h;
                    let coefficient = if index == 0 || index == intervals {
                        1.0
                    } else if index % 2 == 0 {
                        2.0
                    } else {
                        4.0
                    };
                    coefficient * (-potential(y) / thermal).exp()
                })
                .sum::<f64>()
                * h
                / 3.0
        };
        let harmonic = integrate(&|y| 0.5 * y * y);
        let quartic = integrate(&|y| 0.25 * y.powi(4));
        let expected = -thermal * (quartic / harmonic).ln();
        assert!(
            (estimate.free_energy - expected).abs() < 4.0 * estimate.error + 0.01,
            "{} ± {} against {expected}",
            estimate.free_energy,
            estimate.error
        );
    }
}
