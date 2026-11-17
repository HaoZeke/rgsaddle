//! Engine-neutral adapter for an rgpot-style molecular-dynamics session.
//!
//! The trait is intentionally narrower than any engine API. A C++ rgpot
//! binding can implement it without exposing OpenMM, GROMACS, or another
//! engine type to rgsaddle.

use ndarray::Array1;
use rgpot_core::md_session::{MdSession, MdSessionState};

use crate::SaddleError;
use crate::ohtst::session::flux_velocity;
use crate::ohtst::{HyperplaneDynamics, TransmissionCoefficient, TransmissionConfig};
use crate::projector::{
    Hyperplane, HyperplaneEnsemble, HyperplaneObservation, HyperplaneSampleBudget,
    HyperplaneSampler,
};

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
            // OpenMM's documented Langevin examples use a 1 fs step.
            time_step: 0.001,
            // A unit inverse-picosecond friction is the documented example
            // value for Langevin dynamics, in inverse picoseconds.
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

impl<S: MdSession> HyperplaneSampler for RgpotHyperplaneSampler<S> {
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
        self.session
            .set_positions(
                plane
                    .point()
                    .as_slice()
                    .ok_or_else(|| SaddleError::Shape("non-contiguous hyperplane point".into()))?,
            )
            .map_err(SaddleError::Solver)?;
        self.session
            .set_velocities(&vec![0.0; plane.dim()])
            .map_err(SaddleError::Solver)?;
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
            let wall_points: Vec<f64> = walls
                .iter()
                .flat_map(|(point, _)| point.iter().copied())
                .collect();
            let wall_normals: Vec<f64> = walls
                .iter()
                .flat_map(|(_, normal)| normal.iter().copied())
                .collect();
            reflections += self
                .session
                .step_langevin(
                    budget.thermal,
                    self.config.friction,
                    self.config.time_step,
                    plane.point().as_slice().ok_or_else(|| {
                        SaddleError::Shape("non-contiguous hyperplane point".into())
                    })?,
                    plane.normal().as_slice().ok_or_else(|| {
                        SaddleError::Shape("non-contiguous hyperplane normal".into())
                    })?,
                    &wall_points,
                    &wall_normals,
                    budget.stride,
                    budget.seed.wrapping_add(sample as u64),
                )
                .map_err(SaddleError::Solver)?;
            let state = self.session.state().map_err(SaddleError::Solver)?;
            if state.positions.len() != plane.dim()
                || state.velocities.len() != plane.dim()
                || state.forces.len() != plane.dim()
            {
                return Err(SaddleError::Shape(
                    "rgpot session state differs from hyperplane".into(),
                ));
            }
            let positions = Array1::from(state.positions);
            let velocities = Array1::from(state.velocities);
            let forces = Array1::from(state.forces);
            let projected_force =
                &forces - &(plane.normal().to_owned() * plane.normal().dot(&forces));
            let projected_velocity =
                &velocities - &(plane.normal().to_owned() * plane.normal().dot(&velocities));
            let mut observation = HyperplaneObservation::default()
                .with_constraint_force(-plane.normal().dot(&state.forces))
                .with_projected_force(projected_force)
                .with_projected_velocity(projected_velocity)
                .with_observable("rgpot.reflections", reflections as f64);
            if budget.positions {
                observation.position = Some(positions);
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

impl<S: MdSession> HyperplaneDynamics for RgpotHyperplaneDynamics<S> {
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
        let (velocity, reverse_velocity) =
            flux_velocity(plane.normal(), config.thermal, config.seed)?;
        let mut reactive = 0;
        let mut recrossed = 0;
        for start in starts.iter().take(config.shots) {
            if start.len() != plane.dim() {
                return Err(SaddleError::Shape("rgpot start differs from plane".into()));
            }
            self.session
                .set_positions(
                    start.as_slice().ok_or_else(|| {
                        SaddleError::Shape("non-contiguous shooting start".into())
                    })?,
                )
                .map_err(SaddleError::Solver)?;
            self.session
                .set_velocities(
                    velocity.as_slice().ok_or_else(|| {
                        SaddleError::Shape("non-contiguous shooting velocity".into())
                    })?,
                )
                .map_err(SaddleError::Solver)?;
            self.session
                .step_nve(self.config.time_step, config.steps)
                .map_err(SaddleError::Solver)?;
            let forward =
                Array1::from(self.session.state().map_err(SaddleError::Solver)?.positions);
            self.session
                .set_positions(
                    start.as_slice().ok_or_else(|| {
                        SaddleError::Shape("non-contiguous shooting start".into())
                    })?,
                )
                .map_err(SaddleError::Solver)?;
            let reverse_velocity = -&velocity;
            self.session
                .set_velocities(
                    reverse_velocity.as_slice().ok_or_else(|| {
                        SaddleError::Shape("non-contiguous shooting velocity".into())
                    })?,
                )
                .map_err(SaddleError::Solver)?;
            self.session
                .step_nve(self.config.time_step, config.steps)
                .map_err(SaddleError::Solver)?;
            let backward =
                Array1::from(self.session.state().map_err(SaddleError::Solver)?.positions);
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

#[cfg(test)]
mod tests {
    use super::*;
    use rgpot_core::md_session::MdSessionState;

    struct MockSession {
        state: MdSessionState,
    }

    impl MdSession for MockSession {
        fn set_positions(&mut self, positions: &[f64]) -> Result<(), String> {
            self.state.positions = positions.to_vec();
            Ok(())
        }

        fn set_velocities(&mut self, velocities: &[f64]) -> Result<(), String> {
            self.state.velocities = velocities.to_vec();
            Ok(())
        }

        fn step_langevin(
            &mut self,
            _k_t: f64,
            _friction: f64,
            time_step: f64,
            plane_point: &[f64],
            plane_normal: &[f64],
            wall_points: &[f64],
            wall_normals: &[f64],
            steps: usize,
            _seed: u64,
        ) -> Result<usize, String> {
            let mut hits = 0;
            for _ in 0..steps {
                for (position, velocity) in self
                    .state
                    .positions
                    .iter_mut()
                    .zip(self.state.velocities.iter())
                {
                    *position += time_step * *velocity;
                }
                let distance = self
                    .state
                    .positions
                    .iter()
                    .zip(plane_normal.iter())
                    .zip(plane_point.iter())
                    .map(|((position, normal), point)| (position - point) * normal)
                    .sum::<f64>();
                for ((position, normal), point) in self
                    .state
                    .positions
                    .iter_mut()
                    .zip(plane_normal.iter())
                    .zip(plane_point.iter())
                {
                    *position -= distance * normal;
                }
                for (wall_point, wall_normal) in wall_points
                    .chunks_exact(plane_point.len())
                    .zip(wall_normals.chunks_exact(plane_point.len()))
                {
                    let penetration = self
                        .state
                        .positions
                        .iter()
                        .zip(wall_normal.iter())
                        .zip(wall_point.iter())
                        .map(|((position, normal), point)| (position - point) * normal)
                        .sum::<f64>();
                    if penetration > 0.0 {
                        for ((position, velocity), (point, normal)) in self
                            .state
                            .positions
                            .iter_mut()
                            .zip(self.state.velocities.iter_mut())
                            .zip(wall_point.iter().zip(wall_normal.iter()))
                        {
                            *position -= penetration * normal;
                            *velocity -= 2.0 * (*velocity * normal) * normal;
                            let _ = point;
                        }
                        hits += 1;
                    }
                }
            }
            Ok(hits)
        }

        fn step_nve(&mut self, _time_step: f64, _steps: usize) -> Result<(), String> {
            Ok(())
        }

        fn state(&mut self) -> Result<MdSessionState, String> {
            Ok(self.state.clone())
        }
    }

    #[test]
    fn mock_session_confines_plane_and_reflects_wall() {
        let state = MdSessionState {
            positions: vec![0.0, 0.9],
            velocities: vec![3.0, 2.0],
            ..MdSessionState::default()
        };
        let mut wall_probe = MockSession {
            state: state.clone(),
        };
        let wall_hits = wall_probe
            .step_langevin(
                4.0,
                0.0,
                0.2,
                &[0.0, 0.0],
                &[1.0, 0.0],
                &[0.0, 1.0],
                &[0.0, 1.0],
                1,
                11,
            )
            .expect("wall step");
        assert!(wall_hits > 0);
        let mut sampler = RgpotHyperplaneSampler::new(
            MockSession { state },
            RgpotSessionConfig {
                time_step: 0.2,
                friction: 0.0,
            },
        );
        let plane = Hyperplane::new(Array1::zeros(2), Array1::from_vec(vec![1.0, 0.0]))
            .expect("valid plane");
        let budget = HyperplaneSampleBudget::default()
            .with_samples(4)
            .with_stride(1)
            .with_thermal(4.0)
            .with_seed(11);
        let ensemble = sampler.sample_hyperplane(&plane, &budget).expect("sample");
        assert!(ensemble.observations.iter().all(|observation| {
            observation
                .position
                .as_ref()
                .is_some_and(|position| position[0].abs() <= f64::EPSILON * 16.0)
        }));
        let kinetic = ensemble.observations[0]
            .projected_velocity
            .as_ref()
            .map(|velocity| 0.5 * velocity.dot(velocity))
            .expect("velocity");
        assert!(kinetic.is_finite());
    }
}
