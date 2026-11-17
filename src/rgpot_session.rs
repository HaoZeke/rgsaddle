//! Engine-neutral adapter for an rgpot-style molecular-dynamics session.
//!
//! The trait is intentionally narrower than any engine API. A C++ rgpot
//! binding can implement it without exposing OpenMM, GROMACS, or another
//! engine type to rgsaddle.

use ndarray::Array1;

use crate::SaddleError;
use crate::ohtst::{HyperplaneDynamics, TransmissionCoefficient, TransmissionConfig};
use crate::projector::{
    Hyperplane, HyperplaneEnsemble, HyperplaneObservation, HyperplaneSampleBudget,
    HyperplaneSampler,
};

/// State returned by a generic rgpot MD session.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct RgpotMdState {
    /// Cartesian or mass-weighted positions, as chosen by the session.
    pub positions: Array1<f64>,
    /// Velocities in the same coordinate convention.
    pub velocities: Array1<f64>,
    /// Forces in the same coordinate convention.
    pub forces: Array1<f64>,
    /// Potential energy, when the session provides it.
    pub potential_energy: Option<f64>,
    /// Kinetic energy, when the session provides it.
    pub kinetic_energy: Option<f64>,
}

/// A generic rgpot-compatible MD session.
pub trait RgpotMdSession {
    /// Set the current positions.
    fn set_positions(&mut self, positions: &Array1<f64>) -> Result<(), SaddleError>;
    /// Set the current velocities.
    fn set_velocities(&mut self, velocities: &Array1<f64>) -> Result<(), SaddleError>;
    /// Advance projected Langevin/BAOAB dynamics and return wall hits.
    #[allow(clippy::too_many_arguments)]
    fn step_langevin(
        &mut self,
        thermal: f64,
        friction: f64,
        time_step: f64,
        plane: &Hyperplane,
        walls: &[(Array1<f64>, Array1<f64>)],
        steps: usize,
        seed: u64,
    ) -> Result<usize, SaddleError>;
    /// Advance deterministic NVE dynamics.
    fn step_nve(&mut self, time_step: f64, steps: usize) -> Result<(), SaddleError>;
    /// Read the current state.
    fn state(&mut self) -> Result<RgpotMdState, SaddleError>;
}

/// Controls shared by the generic sampler and shooter.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct RgpotSessionConfig {
    /// BAOAB time step.
    pub time_step: f64,
    /// Langevin friction.
    pub friction: f64,
}

impl Default for RgpotSessionConfig {
    fn default() -> Self {
        Self {
            time_step: 0.001,
            friction: 1.0,
        }
    }
}

impl RgpotSessionConfig {
    /// Set the time step.
    #[must_use]
    pub fn with_time_step(mut self, value: f64) -> Self {
        self.time_step = value;
        self
    }
    /// Set the Langevin friction.
    #[must_use]
    pub fn with_friction(mut self, value: f64) -> Self {
        self.friction = value;
        self
    }
}

/// `HyperplaneSampler` adapter over an rgpot MD session.
pub struct RgpotHyperplaneSampler<S> {
    session: S,
    config: RgpotSessionConfig,
}

impl<S> RgpotHyperplaneSampler<S> {
    /// Bind a session to the generic adapter.
    pub fn new(session: S, config: RgpotSessionConfig) -> Self {
        Self { session, config }
    }
    /// Borrow the underlying session.
    pub fn session(&self) -> &S {
        &self.session
    }
    /// Mutably borrow the underlying session.
    pub fn session_mut(&mut self) -> &mut S {
        &mut self.session
    }
}

impl<S: RgpotMdSession> HyperplaneSampler for RgpotHyperplaneSampler<S> {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        if !(self.config.time_step.is_finite() && self.config.time_step > 0.0)
            || !(self.config.friction.is_finite() && self.config.friction >= 0.0)
            || !(budget.thermal.is_finite() && budget.thermal > 0.0)
            || budget.stride == 0
        {
            return Err(SaddleError::Invalid(
                "invalid rgpot session sampling controls".into(),
            ));
        }
        self.session.set_positions(&plane.point().to_owned())?;
        self.session.set_velocities(&Array1::zeros(plane.dim()))?;
        let walls = match budget.region.as_ref() {
            None => Vec::new(),
            Some(region) => region
                .planar_boundaries()
                .ok_or_else(|| {
                    SaddleError::Invalid("rgpot session requires planar region boundaries".into())
                })?
                .into_iter()
                .map(|wall| (wall.point().to_owned(), wall.normal().to_owned()))
                .collect(),
        };
        let mut observations = Vec::with_capacity(budget.samples);
        let mut evaluations = 0;
        let mut reflections = 0;
        for sample in 0..budget.samples {
            reflections += self.session.step_langevin(
                budget.thermal,
                self.config.friction,
                self.config.time_step,
                plane,
                &walls,
                budget.stride,
                budget.seed.wrapping_add(sample as u64),
            )?;
            let state = self.session.state()?;
            if state.positions.len() != plane.dim()
                || state.velocities.len() != plane.dim()
                || state.forces.len() != plane.dim()
            {
                return Err(SaddleError::Shape(
                    "rgpot session state differs from hyperplane".into(),
                ));
            }
            let projected_force =
                &state.forces - &(plane.normal().to_owned() * plane.normal().dot(&state.forces));
            let projected_velocity = &state.velocities
                - &(plane.normal().to_owned() * plane.normal().dot(&state.velocities));
            let mut observation = HyperplaneObservation::default()
                .with_constraint_force(-plane.normal().dot(&state.forces))
                .with_projected_force(projected_force)
                .with_projected_velocity(projected_velocity)
                .with_observable("rgpot.reflections", reflections as f64);
            if budget.positions {
                observation.position = Some(state.positions);
            }
            observations.push(observation);
            evaluations += 1;
        }
        Ok(HyperplaneEnsemble {
            observations,
            evaluations,
            acceptance: 1.0,
        })
    }
}

/// `HyperplaneDynamics` adapter over an rgpot MD session.
pub struct RgpotHyperplaneDynamics<S> {
    session: S,
    config: RgpotSessionConfig,
}

impl<S> RgpotHyperplaneDynamics<S> {
    /// Bind a session to the generic shooting adapter.
    pub fn new(session: S, config: RgpotSessionConfig) -> Self {
        Self { session, config }
    }
}

impl<S: RgpotMdSession> HyperplaneDynamics for RgpotHyperplaneDynamics<S> {
    fn shoot_hyperplane(
        &mut self,
        plane: &Hyperplane,
        starts: &[Array1<f64>],
        config: &TransmissionConfig,
    ) -> Result<TransmissionCoefficient, SaddleError> {
        if starts.is_empty() || config.shots == 0 || config.shots > starts.len() {
            return Err(SaddleError::Invalid(
                "rgpot shooting needs starts and a bounded shot count".into(),
            ));
        }
        if !(config.time_step.is_finite() && config.time_step > 0.0)
            || !(config.thermal.is_finite() && config.thermal > 0.0)
        {
            return Err(SaddleError::Invalid(
                "invalid rgpot shooting controls".into(),
            ));
        }
        let velocity = plane.normal().to_owned() * config.thermal.sqrt();
        let mut reactive = 0;
        let mut recrossed = 0;
        for start in starts.iter().take(config.shots) {
            if start.len() != plane.dim() {
                return Err(SaddleError::Shape("rgpot start differs from plane".into()));
            }
            self.session.set_positions(start)?;
            self.session.set_velocities(&velocity)?;
            self.session.step_nve(self.config.time_step, config.steps)?;
            let forward = self.session.state()?.positions;
            self.session.set_positions(start)?;
            self.session.set_velocities(&(-&velocity))?;
            self.session.step_nve(config.time_step, config.steps)?;
            let backward = self.session.state()?.positions;
            if plane.distance(forward.view()) > 0.0 && plane.distance(backward.view()) < 0.0 {
                reactive += 1;
            } else if plane.distance(forward.view()) > 0.0 || plane.distance(backward.view()) < 0.0
            {
                recrossed += 1;
            }
        }
        let shots = config.shots;
        let kappa = reactive as f64 / shots as f64;
        Ok(TransmissionCoefficient {
            kappa,
            error: 0.0,
            time: vec![config.time_step * config.steps as f64],
            kappa_of_t: vec![kappa],
            shots,
            unassigned: 0,
            reactive,
            recrossed,
            evaluations: shots * config.steps,
        })
    }
}
