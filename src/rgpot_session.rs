//! Engine-neutral adapter for an rgpot molecular-dynamics session.
//!
//! All controls and state vectors use rgpot's unit convention: positions in
//! Angstrom, velocities in Angstrom/ps, forces in eV/Angstrom, `kT` and
//! energies in eV, `time_step` in ps, `friction` in inverse ps, and masses in
//! amu. The crate contains no `dlopen` or engine-specific code.

use ndarray::Array1;
use rgpot_core::md_session::{AMU_A2_PS2_PER_EV, MdLangevinConfig, MdSession};

use crate::SaddleError;
use crate::ohtst::session::{aggregate, checkpoints, flux_velocity, side};
use crate::ohtst::{
    Counting, EndState, Endpoint, HyperplaneDynamics, TransmissionCoefficient, TransmissionConfig,
};
use crate::projector::{
    Hyperplane, HyperplaneCoordinates, HyperplaneEnsemble, HyperplaneObservation,
    HyperplaneSampleBudget, HyperplaneSampler, TangentProjector,
};

/// Controls shared by the generic sampler and shooter.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct RgpotSessionConfig {
    /// BAOAB time step in ps.
    pub time_step: f64,
    /// Langevin friction in inverse ps.
    pub friction: f64,
}

impl Default for RgpotSessionConfig {
    fn default() -> Self {
        Self {
            // 1 fs in ps.
            time_step: 0.001,
            // Unit inverse-picosecond Langevin friction.
            friction: 1.0,
        }
    }
}

impl RgpotSessionConfig {
    /// Set [`Self::time_step`] in ps.
    #[must_use]
    pub fn with_time_step(mut self, value: f64) -> Self {
        self.time_step = value;
        self
    }

    /// Set [`Self::friction`] in inverse ps.
    #[must_use]
    pub fn with_friction(mut self, value: f64) -> Self {
        self.friction = value;
        self
    }
}

fn coord_sqrt_masses<S: MdSession>(
    session: &S,
    expected_dim: usize,
) -> Result<Array1<f64>, SaddleError> {
    if session.dof() != expected_dim {
        return Err(SaddleError::Shape(
            "rgpot session dimension differs from hyperplane".into(),
        ));
    }
    let atom_masses = session
        .masses()
        .map_err(|err| SaddleError::Solver(err.to_string()))?;
    if atom_masses.len() * 3 != expected_dim {
        return Err(SaddleError::Shape(
            "rgpot session mass count differs from hyperplane atoms".into(),
        ));
    }
    if !atom_masses
        .iter()
        .all(|mass| mass.is_finite() && *mass > 0.0)
    {
        return Err(SaddleError::Invalid(
            "rgpot session requires positive finite particle masses".into(),
        ));
    }
    Ok(atom_masses
        .iter()
        .flat_map(|mass| [mass.sqrt(); 3])
        .collect())
}

/// `HyperplaneSampler` adapter over an rgpot [`MdSession`].
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

    /// Borrow the adapter configuration.
    pub fn config(&self) -> &RgpotSessionConfig {
        &self.config
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

        let sqrt_masses = coord_sqrt_masses(&self.session, plane.dim())?;
        let (cart_point, cart_normal) = match budget.coordinates {
            HyperplaneCoordinates::Cartesian => {
                (plane.point().to_owned(), plane.normal().to_owned())
            }
            HyperplaneCoordinates::MassWeighted => {
                let coord_masses = &sqrt_masses * &sqrt_masses;
                let slice = coord_masses
                    .as_slice()
                    .ok_or_else(|| SaddleError::Shape("non-contiguous masses".into()))?;
                plane.to_cartesian(slice)?
            }
        };

        let mut wall_points = Vec::new();
        let mut wall_normals = Vec::new();
        if let Some(region) = budget.region.as_ref() {
            let boundaries = region.planar_boundaries().ok_or_else(|| {
                SaddleError::Invalid("rgpot session requires planar region boundaries".into())
            })?;
            wall_points.reserve(boundaries.len() * plane.dim());
            wall_normals.reserve(boundaries.len() * plane.dim());
            for wall in boundaries {
                if wall.point().len() != plane.dim() || wall.normal().len() != plane.dim() {
                    return Err(SaddleError::Shape(
                        "wall boundary and hyperplane dimensions differ".into(),
                    ));
                }
                match budget.coordinates {
                    HyperplaneCoordinates::Cartesian => {
                        wall_points.extend(wall.point());
                        wall_normals.extend(wall.normal());
                    }
                    HyperplaneCoordinates::MassWeighted => {
                        let w_point = &wall.point() / &sqrt_masses;
                        let w_normal_raw = &wall.normal() * &sqrt_masses;
                        let norm = w_normal_raw.dot(&w_normal_raw).sqrt();
                        if !(norm.is_finite() && norm > 0.0) {
                            return Err(SaddleError::Invalid("wall normal vanishes".into()));
                        }
                        wall_points.extend(w_point);
                        wall_normals.extend(w_normal_raw / norm);
                    }
                }
            }
        }

        let cart_point_slice = cart_point
            .as_slice()
            .ok_or_else(|| SaddleError::Shape("non-contiguous hyperplane point".into()))?;
        let cart_normal_slice = cart_normal
            .as_slice()
            .ok_or_else(|| SaddleError::Shape("non-contiguous hyperplane normal".into()))?;

        self.session
            .set_positions(cart_point_slice)
            .map_err(|err| SaddleError::Solver(err.to_string()))?;
        self.session
            .set_velocities(&vec![0.0; plane.dim()])
            .map_err(|err| SaddleError::Solver(err.to_string()))?;

        if budget.thermalize > 0 {
            let burn_cfg = MdLangevinConfig::default()
                .with_k_t(budget.thermal)
                .with_friction(self.config.friction)
                .with_time_step(self.config.time_step)
                .with_seed(budget.seed);
            self.session
                .step_langevin(
                    &burn_cfg,
                    cart_point_slice,
                    cart_normal_slice,
                    &wall_points,
                    &wall_normals,
                    budget.thermalize,
                )
                .map_err(|err| SaddleError::Solver(err.to_string()))?;
        }

        let mut observations = Vec::with_capacity(budget.samples);
        let mut evaluations = budget.thermalize;
        for sample in 0..budget.samples {
            let step_cfg = MdLangevinConfig::default()
                .with_k_t(budget.thermal)
                .with_friction(self.config.friction)
                .with_time_step(self.config.time_step)
                .with_seed(budget.seed.wrapping_add(1 + sample as u64));
            let hits = self
                .session
                .step_langevin(
                    &step_cfg,
                    cart_point_slice,
                    cart_normal_slice,
                    &wall_points,
                    &wall_normals,
                    budget.stride,
                )
                .map_err(|err| SaddleError::Solver(err.to_string()))?;
            let state = self
                .session
                .state()
                .map_err(|err| SaddleError::Solver(err.to_string()))?;
            if state.positions.len() != plane.dim()
                || state.velocities.len() != plane.dim()
                || state.forces.len() != plane.dim()
            {
                return Err(SaddleError::Shape(
                    "rgpot session state differs from hyperplane".into(),
                ));
            }
            let cart_positions = Array1::from(state.positions);
            let cart_velocities = Array1::from(state.velocities);
            let cart_forces = Array1::from(state.forces);

            let (positions, velocities, forces) = match budget.coordinates {
                HyperplaneCoordinates::Cartesian => (cart_positions, cart_velocities, cart_forces),
                HyperplaneCoordinates::MassWeighted => (
                    &cart_positions * &sqrt_masses,
                    &cart_velocities * &sqrt_masses,
                    &cart_forces / &sqrt_masses,
                ),
            };
            let projected_force = TangentProjector::project(plane, forces.view());
            let projected_velocity = TangentProjector::project(plane, velocities.view());
            let mut observation = HyperplaneObservation::default()
                .with_constraint_force(plane.normal().dot(&forces))
                .with_projected_force(projected_force)
                .with_projected_velocity(projected_velocity)
                .with_observable("rgpot.reflections", hits as f64);
            if budget.positions {
                observation.position = Some(positions);
            }
            observations.push(observation);
            evaluations += budget.stride;
        }
        Ok(HyperplaneEnsemble {
            observations,
            evaluations,
            acceptance: 1.0,
        })
    }
}

/// `HyperplaneDynamics` adapter over an rgpot [`MdSession`].
pub struct RgpotHyperplaneDynamics<S> {
    session: S,
    config: RgpotSessionConfig,
}

impl<S> RgpotHyperplaneDynamics<S> {
    /// Bind a session to the generic shooting adapter.
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

    /// Borrow the adapter configuration.
    pub fn config(&self) -> &RgpotSessionConfig {
        &self.config
    }
}

impl<S: MdSession> HyperplaneDynamics for RgpotHyperplaneDynamics<S> {
    fn shoot_hyperplane(
        &mut self,
        plane: &Hyperplane,
        starts: &[Array1<f64>],
        config: &TransmissionConfig,
    ) -> Result<TransmissionCoefficient, SaddleError> {
        let dim = plane.dim();
        config.check(dim)?;
        if config.friction != 0.0 {
            return Err(SaddleError::Invalid(
                "rgpot NVE shooting requires zero friction".into(),
            ));
        }
        if !matches!(config.end_state, EndState::Side) {
            return Err(SaddleError::Invalid(
                "rgpot shooting currently supports EndState::Side only".into(),
            ));
        }
        if starts.is_empty() {
            return Err(SaddleError::Invalid(
                "rgpot shooting needs at least one start".into(),
            ));
        }
        if starts.iter().any(|start| start.len() != dim) {
            return Err(SaddleError::Shape("rgpot start differs from plane".into()));
        }

        let session_sqrt = coord_sqrt_masses(&self.session, dim)?;
        let (sqrt_masses, cart_plane, scale_positions) = match &config.masses {
            Some(masses) => (
                Array1::from_iter(masses.iter().map(|mass| mass.sqrt())),
                plane.clone(),
                false,
            ),
            None => {
                let coord_masses = &session_sqrt * &session_sqrt;
                let slice = coord_masses
                    .as_slice()
                    .ok_or_else(|| SaddleError::Shape("non-contiguous masses".into()))?;
                let (cart_point, cart_normal) = plane.to_cartesian(slice)?;
                (
                    session_sqrt,
                    Hyperplane::new(cart_point, cart_normal)?,
                    true,
                )
            }
        };

        let checkpoint_steps = checkpoints(config);
        let thermal_vel = config.thermal * AMU_A2_PS2_PER_EV;
        let cart_normal_owned = cart_plane.normal().to_owned();
        let product =
            |state: Option<Endpoint>| state.map(|s| f64::from(u8::from(s == Endpoint::Product)));

        let mut shot_values: Vec<Vec<Option<f64>>> = Vec::with_capacity(config.shots);
        let mut recrossed_count = 0usize;
        let mut evaluations = 0usize;

        for shot in 0..config.shots {
            let start = &starts[shot * starts.len() / config.shots];
            let cart_start = if scale_positions {
                cart_plane.project((start / &sqrt_masses).view())
            } else {
                cart_plane.project(start.view())
            };
            let cart_start_slice = cart_start
                .as_slice()
                .ok_or_else(|| SaddleError::Shape("non-contiguous shooting start".into()))?;

            let (forward_vel, backward_vel) = flux_velocity(
                &cart_normal_owned,
                thermal_vel,
                &sqrt_masses,
                config.seed.wrapping_add(shot as u64),
            )?;

            let mut integrate_half =
                |velocity: &Array1<f64>, watch_recrossing: bool| -> Result<(Vec<Option<Endpoint>>, Option<usize>), SaddleError> {
                    self.session
                        .set_positions(cart_start_slice)
                        .map_err(|err| SaddleError::Solver(err.to_string()))?;
                    self.session
                        .set_velocities(
                            velocity
                                .as_slice()
                                .ok_or_else(|| SaddleError::Shape("non-contiguous velocity".into()))?,
                        )
                        .map_err(|err| SaddleError::Solver(err.to_string()))?;

                    let mut states = vec![None; checkpoint_steps.len()];
                    let mut recrossed_at = None;
                    let mut prev_step = 0usize;
                    let mut run: (Option<Endpoint>, usize) = (None, 0);

                    for (idx, &cp_step) in checkpoint_steps.iter().enumerate() {
                        let delta = cp_step - prev_step;
                        prev_step = cp_step;
                        if delta > 0 {
                            self.session
                                .step_nve(config.time_step, delta)
                                .map_err(|err| SaddleError::Solver(err.to_string()))?;
                            evaluations += delta;
                        }
                        let pos = Array1::from(
                            self.session
                                .state()
                                .map_err(|err| SaddleError::Solver(err.to_string()))?
                                .positions,
                        );
                        let endpoint = side(&cart_plane, pos.view());
                        if watch_recrossing
                            && recrossed_at.is_none()
                            && endpoint == Some(Endpoint::Product)
                        {
                            recrossed_at = Some(idx);
                            if config.counting == Counting::FirstCrossing {
                                break;
                            }
                        }
                        states[idx] = endpoint;
                        run = match (endpoint, run.0) {
                            (Some(s), Some(r)) if s == r => (endpoint, run.1 + 1),
                            (Some(_), _) => (endpoint, 1),
                            (None, _) => (None, 0),
                        };
                        if config.commit > 0 && run.1 >= config.commit {
                            for slot in &mut states[idx + 1..] {
                                *slot = endpoint;
                            }
                            break;
                        }
                    }
                    Ok((states, recrossed_at))
                };

            let (backward_states, backward_recrossed) = integrate_half(&backward_vel, true)?;
            let (forward_states, _) = integrate_half(&forward_vel, false)?;
            if backward_recrossed.is_some() {
                recrossed_count += 1;
            }

            let values = (0..checkpoint_steps.len())
                .map(|k| match config.counting {
                    Counting::Final => {
                        Some(product(forward_states[k])? - product(backward_states[k])?)
                    }
                    Counting::FirstCrossing => {
                        if backward_recrossed.is_some_and(|idx| idx <= k) {
                            return Some(0.0);
                        }
                        let past = backward_states[k]?;
                        let future = forward_states[k]?;
                        Some(f64::from(u8::from(
                            past == Endpoint::Reactant && future == Endpoint::Product,
                        )))
                    }
                })
                .collect();
            shot_values.push(values);
        }

        aggregate(&shot_values, config, recrossed_count, evaluations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Region;
    use rgpot_core::md_session::{MdSessionError, MdSessionState};

    /// Harmonic/free Cartesian mock engine speaking rgpot units (Angstrom, eV, ps, amu).
    struct MockSession {
        positions: Vec<f64>,
        velocities: Vec<f64>,
        masses: Vec<f64>,
        /// Optional quadratic barrier curvature `V(x) = -0.5 * barrier * (x_0 - pivot)^2`.
        barrier: f64,
        last_nve_dt: f64,
    }

    impl MockSession {
        fn free(masses: Vec<f64>) -> Self {
            let n3 = masses.len() * 3;
            Self {
                positions: vec![0.0; n3],
                velocities: vec![0.0; n3],
                masses,
                barrier: 0.0,
                last_nve_dt: 0.0,
            }
        }

        fn forces(&self) -> Vec<f64> {
            let mut f = vec![0.0; self.positions.len()];
            if self.barrier != 0.0 {
                // V = -0.5 * k * x0^2 -> F_0 = +k * x0 (inverted parabola at x0=0)
                f[0] = self.barrier * self.positions[0];
            }
            f
        }
    }

    impl MdSession for MockSession {
        fn dof(&self) -> usize {
            self.positions.len()
        }

        fn masses(&self) -> Result<Vec<f64>, MdSessionError> {
            Ok(self.masses.clone())
        }

        fn set_positions(&mut self, positions: &[f64]) -> Result<(), MdSessionError> {
            self.positions = positions.to_vec();
            Ok(())
        }

        fn set_velocities(&mut self, velocities: &[f64]) -> Result<(), MdSessionError> {
            self.velocities = velocities.to_vec();
            Ok(())
        }

        fn step_langevin(
            &mut self,
            _config: &MdLangevinConfig,
            plane_point: &[f64],
            plane_normal: &[f64],
            wall_points: &[f64],
            wall_normals: &[f64],
            steps: usize,
        ) -> Result<usize, MdSessionError> {
            let n3 = self.positions.len();
            let mut hits = 0;
            let coord_m: Vec<f64> = self.masses.iter().flat_map(|m| [*m; 3]).collect();
            let alpha: f64 = plane_normal
                .iter()
                .zip(&coord_m)
                .map(|(n, m)| n * n / m)
                .sum();
            let w: Vec<f64> = plane_normal
                .iter()
                .zip(&coord_m)
                .map(|(n, m)| n / (alpha * m))
                .collect();
            for step in 0..steps {
                if _config.friction > 0.0 && self.velocities.iter().all(|v| *v == 0.0) {
                    // Kick tangent y-velocity so thermalized mock motion hits the wall.
                    self.velocities[1] = 2.0 + 0.1 * ((_config.seed + step as u64) % 3) as f64;
                }
                for (p, v) in self.positions.iter_mut().zip(&self.velocities) {
                    *p += _config.time_step * *v;
                }
                let nx: f64 = self
                    .positions
                    .iter()
                    .zip(plane_point)
                    .zip(plane_normal)
                    .map(|((x, p), n)| (x - p) * n)
                    .sum();
                for (x, wi) in self.positions.iter_mut().zip(&w) {
                    *x -= nx * wi;
                }
                for (wp, wn) in wall_points.chunks_exact(n3).zip(wall_normals.chunks_exact(n3)) {
                    let pen: f64 = self
                        .positions
                        .iter()
                        .zip(wp)
                        .zip(wn)
                        .map(|((x, p), u)| (x - p) * u)
                        .sum();
                    if pen > 0.0 {
                        let uv: f64 = self.velocities.iter().zip(wn).map(|(v, u)| v * u).sum();
                        for ((x, v), u) in self
                            .positions
                            .iter_mut()
                            .zip(self.velocities.iter_mut())
                            .zip(wn)
                        {
                            *x -= 2.0 * pen * u;
                            *v -= 2.0 * uv * u;
                        }
                        hits += 1;
                    }
                }
            }
            Ok(hits)
        }

        fn step_nve(&mut self, time_step: f64, steps: usize) -> Result<(), MdSessionError> {
            self.last_nve_dt = time_step;
            let coord_m: Vec<f64> = self.masses.iter().flat_map(|m| [*m; 3]).collect();
            let mut f = self.forces();
            for _ in 0..steps {
                for ((v, fi), m) in self.velocities.iter_mut().zip(&f).zip(&coord_m) {
                    *v += 0.5 * time_step * (*fi * AMU_A2_PS2_PER_EV / m);
                }
                for (x, v) in self.positions.iter_mut().zip(&self.velocities) {
                    *x += time_step * *v;
                }
                f = self.forces();
                for ((v, fi), m) in self.velocities.iter_mut().zip(&f).zip(&coord_m) {
                    *v += 0.5 * time_step * (*fi * AMU_A2_PS2_PER_EV / m);
                }
            }
            Ok(())
        }

        fn state(&mut self) -> Result<MdSessionState, MdSessionError> {
            let f = self.forces();
            Ok(MdSessionState::default()
                .with_positions(self.positions.clone())
                .with_velocities(self.velocities.clone())
                .with_forces(f))
        }
    }

    #[test]
    fn sampler_thermalizes_confines_3n_plane_and_reflects_walls() {
        let mut session = MockSession::free(vec![4.0]);
        session.set_velocities(&[3.0, 2.0, 0.0]).unwrap();
        let plane = Hyperplane::new(
            Array1::from_vec(vec![0.0, 1.8, 0.0]),
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
        )
        .unwrap();
        let wall_hi = std::sync::Arc::new(
            crate::PlanarBoundary::new(
                Array1::from_vec(vec![0.0, 2.0, 0.0]),
                Array1::from_vec(vec![0.0, 1.0, 0.0]),
            )
            .unwrap(),
        ) as std::sync::Arc<dyn Region>;
        let wall_lo = std::sync::Arc::new(
            crate::PlanarBoundary::new(
                Array1::from_vec(vec![0.0, 1.4, 0.0]),
                Array1::from_vec(vec![0.0, -1.0, 0.0]),
            )
            .unwrap(),
        ) as std::sync::Arc<dyn Region>;
        let region = std::sync::Arc::new(
            crate::RegionIntersection::new(vec![wall_hi, wall_lo]).unwrap(),
        ) as std::sync::Arc<dyn Region>;
        let mut sampler = RgpotHyperplaneSampler::new(
            session,
            RgpotSessionConfig::default()
                .with_time_step(0.2)
                .with_friction(1.0),
        );
        let budget = HyperplaneSampleBudget::default()
            .with_samples(4)
            .with_thermalize(2)
            .with_stride(2)
            .with_thermal(0.02585)
            .with_seed(11)
            .with_region(region);
        let ensemble = sampler.sample_hyperplane(&plane, &budget).expect("sample");
        assert_eq!(ensemble.observations.len(), 4);
        assert_eq!(ensemble.evaluations, 2 + 8);
        assert!(ensemble.observations.iter().all(|observation| {
            observation
                .position
                .as_ref()
                .is_some_and(|position| position[0].abs() <= f64::EPSILON * 16.0)
        }));
        assert!(
            ensemble
                .observations
                .iter()
                .filter_map(|obs| obs.observables.get("rgpot.reflections"))
                .sum::<f64>()
                > 0.0
        );
    }

    #[test]
    fn sampler_maps_mass_weighted_coordinates_and_constraint_force_sign() {
        struct ConstantForceSession {
            positions: Vec<f64>,
            velocities: Vec<f64>,
            masses: Vec<f64>,
            force: Vec<f64>,
        }
        impl MdSession for ConstantForceSession {
            fn dof(&self) -> usize {
                self.positions.len()
            }
            fn masses(&self) -> Result<Vec<f64>, MdSessionError> {
                Ok(self.masses.clone())
            }
            fn set_positions(&mut self, positions: &[f64]) -> Result<(), MdSessionError> {
                self.positions = positions.to_vec();
                Ok(())
            }
            fn set_velocities(&mut self, velocities: &[f64]) -> Result<(), MdSessionError> {
                self.velocities = velocities.to_vec();
                Ok(())
            }
            fn step_langevin(
                &mut self,
                _config: &MdLangevinConfig,
                _plane_point: &[f64],
                _plane_normal: &[f64],
                _wall_points: &[f64],
                _wall_normals: &[f64],
                _steps: usize,
            ) -> Result<usize, MdSessionError> {
                Ok(0)
            }
            fn step_nve(&mut self, _time_step: f64, _steps: usize) -> Result<(), MdSessionError> {
                Ok(())
            }
            fn state(&mut self) -> Result<MdSessionState, MdSessionError> {
                Ok(MdSessionState::default()
                    .with_positions(self.positions.clone())
                    .with_velocities(self.velocities.clone())
                    .with_forces(self.force.clone()))
            }
        }
        let session = ConstantForceSession {
            positions: vec![0.0; 3],
            velocities: vec![0.0; 3],
            masses: vec![4.0],
            // Physical force F = (+6.0, 0, 0) eV/A => -dV/dx = +6.0 => F_q = +3.0
            force: vec![6.0, 0.0, 0.0],
        };
        let mut sampler = RgpotHyperplaneSampler::new(session, RgpotSessionConfig::default());
        let plane = Hyperplane::new(
            Array1::from_vec(vec![2.0, 0.0, 0.0]),
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
        )
        .unwrap();
        let budget = HyperplaneSampleBudget::default()
            .with_samples(1)
            .with_thermalize(0)
            .with_stride(1)
            .with_coordinates(HyperplaneCoordinates::MassWeighted);
        let ensemble = sampler.sample_hyperplane(&plane, &budget).expect("sample");
        let obs = &ensemble.observations[0];
        // Cartesian position is q / sqrt(m) = 2.0 / 2.0 = 1.0 A; returned position is q = 2.0.
        assert!((sampler.session().positions[0] - 1.0).abs() <= 16.0 * f64::EPSILON);
        assert!((obs.position.as_ref().unwrap()[0] - 2.0).abs() <= 16.0 * f64::EPSILON);
        // constraint_force = +n . F_q = +3.0
        assert!((obs.constraint_force - 3.0).abs() <= 16.0 * f64::EPSILON);
    }

    #[test]
    fn dynamics_rejects_nonzero_friction_and_quench_end_state() {
        let session = MockSession::free(vec![1.0]);
        let mut dynamics = RgpotHyperplaneDynamics::new(session, RgpotSessionConfig::default());
        let plane = Hyperplane::new(
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
        )
        .unwrap();
        let starts = vec![Array1::from_vec(vec![0.0, 0.0, 0.0])];
        let bad_friction = TransmissionConfig::default()
            .with_friction(1.0)
            .with_shots(4)
            .with_blocks(2);
        assert!(matches!(
            dynamics.shoot_hyperplane(&plane, &starts, &bad_friction),
            Err(SaddleError::Invalid(_))
        ));
        let basins = crate::ohtst::Basins::new(
            vec![Array1::from_vec(vec![-1.0, 0.0, 0.0])],
            vec![Array1::from_vec(vec![1.0, 0.0, 0.0])],
        );
        let bad_end = TransmissionConfig::default()
            .with_end_state(EndState::from(basins))
            .with_shots(4)
            .with_blocks(2);
        assert!(matches!(
            dynamics.shoot_hyperplane(&plane, &starts, &bad_end),
            Err(SaddleError::Invalid(_))
        ));
    }

    #[test]
    fn dynamics_shoots_across_barrier_and_uses_transmission_time_step() {
        let mut session = MockSession::free(vec![1.0]);
        session.barrier = 1.0;
        let mut dynamics = RgpotHyperplaneDynamics::new(
            session,
            RgpotSessionConfig::default().with_time_step(0.999),
        );
        let plane = Hyperplane::new(
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
        )
        .unwrap();
        let starts = vec![Array1::from_vec(vec![0.0, 0.0, 0.0])];
        let tx_cfg = TransmissionConfig::default()
            .with_thermal(0.02585)
            .with_time_step(0.002)
            .with_steps(20)
            .with_check_every(10)
            .with_shots(8)
            .with_blocks(2)
            .with_counting(Counting::FirstCrossing)
            .with_seed(5);
        let report = dynamics
            .shoot_hyperplane(&plane, &starts, &tx_cfg)
            .expect("shoot");
        assert!((dynamics.session().last_nve_dt - 0.002).abs() <= f64::EPSILON);
        assert_eq!(report.shots, 8);
        assert_eq!(report.reactive, 8);
        assert!((report.kappa - 1.0).abs() <= f64::EPSILON);
        assert_eq!(report.time.len(), 2);
        assert_eq!(report.evaluations, 8 * 2 * 20);

        let final_cfg = tx_cfg.with_counting(Counting::Final);
        let final_report = dynamics
            .shoot_hyperplane(&plane, &starts, &final_cfg)
            .expect("final counting");
        assert!((final_report.kappa - 1.0).abs() <= f64::EPSILON);
    }
}
