//! Constrained molecular dynamics through the rgpot MD session ABI.
//!
//! One backend serves every engine that exports the engine-generic session
//! of rgpot (`CppCore/rgpot/MDSession/md_session_c_abi.h`): the library is
//! named by a symbol prefix, and the calls are `<prefix>_md_<call>`. GROMACS
//! nblib is `rgpot_nblib`; another engine that exports the same calls under
//! its own prefix is used by changing [`MdSessionConfig::prefix`] and nothing
//! else. Nothing here is specific to one engine.
//!
//! [`MdSessionBackend`] implements [`HyperplaneSampler`] with thermostatted
//! plane-projected steps and planar hard walls, and [`HyperplaneDynamics`]
//! with unconstrained steps from a flux-weighted velocity draw. The session
//! owns integration, forces and units: Angstrom, picosecond, eV and amu at
//! the boundary, so `kT` is in eV, friction in inverse ps and time steps in
//! ps. The library stays loaded for the life of the process, because an
//! engine may keep process-global state.
//!
//! Mass-weighted coordinates are `q = M^(1/2) x` with `x` in Angstrom and
//! `M` in amu. A plane normal in `q` maps to the gradient `M^(1/2) n_q` of
//! the Cartesian positions that the session takes.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::path::{Path, PathBuf};

use libloading::Library;
use ndarray::{Array1, ArrayView1};
use rgpot_core::profile::library_candidates;

use crate::ohtst::{
    Counting, EndState, Endpoint, HyperplaneDynamics, TransmissionCoefficient, TransmissionConfig,
};
use crate::projector::TangentProjector;
use crate::region::PlanarBoundary;
use crate::stats::{Rng, block_mean};
use crate::strain_rate::angular_scale_rad_s;
use crate::{
    Hyperplane, HyperplaneCoordinates, HyperplaneEnsemble, HyperplaneObservation,
    HyperplaneSampleBudget, HyperplaneSampler, SaddleError,
};

const MD_ABI_VERSION: c_int = 1;
const FEATURE_PROJECTED: c_uint = 2;
const PS_PER_S: f64 = 1.0e12;
/// Relative agreement of the masses a caller states and the engine's.
const MASS_MATCH: f64 = 1.0 / (1u64 << 30) as f64;

const OBSERVABLE_POTENTIAL: &str = "mdsession.potential_ev";
const OBSERVABLE_KINETIC: &str = "mdsession.kinetic_ev";

/// `sqrt(eV / amu)` in Angstrom per ps, the factor that turns a mass-weighted
/// velocity in `sqrt(eV)` into a Cartesian one.
fn velocity_unit() -> f64 {
    angular_scale_rad_s() / PS_PER_S
}

/// Atomistic start configuration for an MD session.
///
/// `spec` is the engine's creation text: a topology for nblib, a serialized
/// system for an engine that takes one. `positions` are Cartesian, three
/// values per atom, in angstrom; they are where a sampler starts and
/// are projected onto the plane by the first step.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct MdSessionSystem {
    /// Engine creation text.
    pub spec: String,
    /// Cartesian coordinates in angstrom, three per atom.
    pub positions: Array1<f64>,
    /// Optional row-major simulation cell in angstrom, lattice vectors as rows.
    pub box_matrix: Option<[f64; 9]>,
}

impl MdSessionSystem {
    /// Set [`Self::spec`].
    #[must_use]
    pub fn with_spec(mut self, spec: impl Into<String>) -> Self {
        self.spec = spec.into();
        self
    }

    /// Set [`Self::positions`].
    #[must_use]
    pub fn with_positions(mut self, positions: Array1<f64>) -> Self {
        self.positions = positions;
        self
    }

    /// Set [`Self::box_matrix`].
    #[must_use]
    pub fn with_box_matrix(mut self, box_matrix: [f64; 9]) -> Self {
        self.box_matrix = Some(box_matrix);
        self
    }

    fn check(&self) -> Result<(), SaddleError> {
        if self.spec.is_empty() {
            return Err(SaddleError::Invalid(
                "MD session creation text is empty".into(),
            ));
        }
        if self.positions.is_empty() || !self.positions.len().is_multiple_of(3) {
            return Err(SaddleError::Shape(
                "MD session positions need three coordinates per atom".into(),
            ));
        }
        if !self.positions.iter().all(|value| value.is_finite())
            || self
                .box_matrix
                .is_some_and(|cell| !cell.iter().all(|value| value.is_finite()))
        {
            return Err(SaddleError::NonFinite("MD session system"));
        }
        Ok(())
    }
}

/// Which engine library to load and how its session is stepped.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct MdSessionConfig {
    /// Symbol prefix of the engine: the calls are `<prefix>_md_<call>`, and
    /// `lib<prefix>.so` is a library candidate.
    pub prefix: String,
    /// Optional explicit library path, tried before the candidates.
    pub library_path: Option<PathBuf>,
    /// Session time step in ps, for sampling.
    pub time_step_ps: f64,
    /// Langevin friction in inverse ps, for sampling.
    pub friction_per_ps: f64,
}

impl Default for MdSessionConfig {
    fn default() -> Self {
        Self {
            prefix: String::new(),
            library_path: None,
            time_step_ps: 0.001,
            friction_per_ps: 1.0,
        }
    }
}

impl MdSessionConfig {
    /// Set [`Self::prefix`].
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Set an explicit library path.
    #[must_use]
    pub fn with_library_path(mut self, library_path: impl Into<PathBuf>) -> Self {
        self.library_path = Some(library_path.into());
        self
    }

    /// Set [`Self::time_step_ps`].
    #[must_use]
    pub fn with_time_step_ps(mut self, time_step_ps: f64) -> Self {
        self.time_step_ps = time_step_ps;
        self
    }

    /// Set [`Self::friction_per_ps`].
    #[must_use]
    pub fn with_friction_per_ps(mut self, friction_per_ps: f64) -> Self {
        self.friction_per_ps = friction_per_ps;
        self
    }

    fn check_prefix(&self) -> Result<(), SaddleError> {
        if self.prefix.is_empty()
            || !self
                .prefix
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(SaddleError::Invalid(format!(
                "MD session symbol prefix {:?} is not an identifier",
                self.prefix
            )));
        }
        Ok(())
    }

    fn check(&self) -> Result<(), SaddleError> {
        self.check_prefix()?;
        if !(self.time_step_ps.is_finite() && self.time_step_ps > 0.0) {
            return Err(SaddleError::Invalid(
                "MD session time step must be finite and positive".into(),
            ));
        }
        if !(self.friction_per_ps.is_finite() && self.friction_per_ps > 0.0) {
            return Err(SaddleError::Invalid(
                "MD session friction must be finite and positive".into(),
            ));
        }
        Ok(())
    }
}

/// [`HyperplaneSampler`] and [`HyperplaneDynamics`] over an rgpot MD session.
pub struct MdSessionBackend {
    system: MdSessionSystem,
    config: MdSessionConfig,
}

impl MdSessionBackend {
    /// Build a backend. The shared library is loaded when work begins.
    pub fn new(system: MdSessionSystem, config: MdSessionConfig) -> Self {
        Self { system, config }
    }

    /// Start configuration of each ensemble.
    pub fn system(&self) -> &MdSessionSystem {
        &self.system
    }

    /// Engine and stepping controls.
    pub fn config(&self) -> &MdSessionConfig {
        &self.config
    }
}

impl HyperplaneSampler for MdSessionBackend {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        self.system.check()?;
        self.config.check()?;
        if !(budget.thermal.is_finite() && budget.thermal > 0.0) {
            return Err(SaddleError::Invalid(
                "MD session kT must be finite and positive".into(),
            ));
        }
        if budget.stride == 0 {
            return Err(SaddleError::Invalid(
                "MD session sampling stride must be positive".into(),
            ));
        }
        let mut engine = Engine::load(&self.config, &self.system.spec, self.config.time_step_ps)?;
        engine.require(FEATURE_PROJECTED, "plane-projected steps")?;
        let dimension = engine.dimension;
        if self.system.positions.len() != dimension || plane.dim() != dimension {
            return Err(SaddleError::Shape(
                "MD session, system and hyperplane dimensions differ".into(),
            ));
        }
        let frame = Frame::new(engine.masses()?, budget.coordinates)?;
        let (point, normal) = frame.plane(plane)?;
        let walls = match &budget.region {
            Some(region) => {
                let boundaries = region.planar_boundaries().ok_or_else(|| {
                    SaddleError::Invalid(
                        "an MD session needs a region with planar boundaries".into(),
                    )
                })?;
                boundaries
                    .iter()
                    .map(|boundary| frame.wall(boundary, dimension))
                    .collect::<Result<Vec<_>, _>>()?
            }
            None => Vec::new(),
        };

        if let Some(cell) = &self.system.box_matrix {
            engine.set_box(cell)?;
        }
        let positions =
            self.system.positions.as_slice().ok_or_else(|| {
                SaddleError::Shape("MD session positions must be contiguous".into())
            })?;
        engine.set_positions(positions)?;
        engine.set_seed(budget.seed)?;
        let mut rng = Rng::new(budget.seed);
        let thermal_speed = budget.thermal.sqrt() * velocity_unit();
        let velocities: Vec<f64> = frame
            .masses
            .iter()
            .map(|mass| thermal_speed * rng.normal() / mass.sqrt())
            .collect();
        engine.set_velocities(&velocities)?;
        engine.initialize()?;

        let thermostat = Some(Thermostat {
            kt_ev: budget.thermal,
            friction_per_ps: self.config.friction_per_ps,
        });
        let projection = Some((point.as_slice(), normal.as_slice()));
        if budget.thermalize > 0 {
            engine.step(budget.thermalize, thermostat, projection, &walls)?;
        }

        let mut observations = Vec::with_capacity(budget.samples);
        let mut evaluations = 0;
        for _ in 0..budget.samples {
            engine.step(budget.stride, thermostat, projection, &walls)?;
            let state = engine.state()?;
            evaluations = state.force_evaluations;
            let force = frame.force(&state.forces);
            let constraint_force = plane.normal().dot(&force);
            let projected_force = TangentProjector::project(plane, force.view());
            let mut observation = HyperplaneObservation {
                constraint_force,
                pafi_gradient: None,
                jacobian: 0.0,
                position: budget.positions.then(|| frame.position(&state.positions)),
                projected_force: Some(projected_force),
                projected_velocity: Some(frame.velocity(&state.velocities)),
                observables: Default::default(),
            };
            observation
                .observables
                .insert(OBSERVABLE_POTENTIAL.to_string(), state.potential_ev);
            observation
                .observables
                .insert(OBSERVABLE_KINETIC.to_string(), state.kinetic_ev);
            observations.push(observation);
        }
        Ok(HyperplaneEnsemble {
            observations,
            evaluations,
            acceptance: 1.0,
        })
    }
}

impl HyperplaneDynamics for MdSessionBackend {
    fn shoot_hyperplane(
        &mut self,
        plane: &Hyperplane,
        starts: &[Array1<f64>],
        config: &TransmissionConfig,
    ) -> Result<TransmissionCoefficient, SaddleError> {
        self.system.check()?;
        self.config.check_prefix()?;
        check_shooting(config)?;
        if starts.is_empty() {
            return Err(SaddleError::Invalid(
                "shooting needs at least one start on the plane".into(),
            ));
        }
        if matches!(config.end_state, EndState::Quench(_)) {
            return Err(SaddleError::Invalid(
                "an MD session assigns the end of a half by the side of the plane; \
                 a quench needs a surface"
                    .into(),
            ));
        }
        if config.hold_excluded && config.friction > 0.0 {
            return Err(SaddleError::Invalid(
                "the session thermostat acts on every coordinate, so held rigid \
                 translations need hold_excluded false or zero friction"
                    .into(),
            ));
        }
        let n = plane.dim();
        if starts.iter().any(|start| start.len() != n) {
            return Err(SaddleError::Shape(
                "a start differs in length from the plane".into(),
            ));
        }
        let mut engine = Engine::load(&self.config, &self.system.spec, config.time_step)?;
        engine.require(FEATURE_PROJECTED, "steps with a thermostat")?;
        if engine.dimension != n {
            return Err(SaddleError::Shape(
                "MD session and hyperplane dimensions differ".into(),
            ));
        }
        let masses = engine.masses()?;
        let coordinates = match &config.masses {
            Some(stated) => {
                if stated.len() != n
                    || stated
                        .iter()
                        .zip(&masses)
                        .any(|(stated, engine)| (stated - engine).abs() > MASS_MATCH * engine)
                {
                    return Err(SaddleError::Invalid(
                        "the masses of the shooting config differ from the session's".into(),
                    ));
                }
                HyperplaneCoordinates::Cartesian
            }
            None => HyperplaneCoordinates::MassWeighted,
        };
        let frame = Frame::new(masses, coordinates)?;
        let (_, normal) = frame.plane(plane)?;
        if let Some(cell) = &self.system.box_matrix {
            engine.set_box(cell)?;
        }

        let flux = {
            let mut direction =
                Array1::from_iter(normal.iter().zip(&frame.root).map(|(n, r)| n / r));
            let norm = direction.dot(&direction).sqrt();
            direction /= norm;
            direction
        };
        let held = if config.hold_excluded {
            frame.translations()?
        } else {
            Vec::new()
        };
        let mut free = flux.clone();
        project_out(&mut free, &held);
        let free_norm = free.dot(&free).sqrt();
        if free_norm <= f64::EPSILON {
            return Err(SaddleError::Invalid(
                "the plane normal lies in the held modes".into(),
            ));
        }
        free /= free_norm;

        let mut checkpoints: Vec<usize> = (1..=config.steps / config.check_every)
            .map(|k| k * config.check_every)
            .collect();
        if checkpoints.last() != Some(&config.steps) {
            checkpoints.push(config.steps);
        }
        let thermostat = (config.friction > 0.0).then_some(Thermostat {
            kt_ev: config.thermal,
            friction_per_ps: config.friction,
        });
        let mut shooter = Shooter {
            engine: &mut engine,
            frame: &frame,
            plane,
            config,
            checkpoints: &checkpoints,
            thermostat,
            flux,
            free,
            held,
        };
        let mut shots = Vec::with_capacity(config.shots);
        for shot in 0..config.shots {
            let start = plane.project(starts[shot * starts.len() / config.shots].view());
            shots.push(shooter.shot(&start, config.seed.wrapping_add(shot as u64))?);
        }
        let evaluations = shooter.engine.force_evaluations()?;

        let last = checkpoints.len() - 1;
        let kappa_of_t: Vec<f64> = (0..checkpoints.len())
            .map(|k| {
                let values: Vec<f64> = shots.iter().filter_map(|s| s.values[k]).collect();
                if values.is_empty() {
                    f64::NAN
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                }
            })
            .collect();
        let finals: Vec<f64> = shots.iter().filter_map(|s| s.values[last]).collect();
        if finals.len() < config.blocks {
            return Err(SaddleError::Invalid(
                "fewer shots reached a state than there are blocks".into(),
            ));
        }
        let (_, error, _, _) = block_mean(&finals, config.blocks);
        Ok(TransmissionCoefficient {
            kappa: kappa_of_t[last],
            error,
            time: checkpoints
                .iter()
                .map(|&step| step as f64 * config.time_step)
                .collect(),
            kappa_of_t,
            shots: finals.len(),
            unassigned: shots.len() - finals.len(),
            reactive: finals.iter().filter(|value| **value > 0.5).count(),
            recrossed: shots.iter().filter(|s| s.recrossed).count(),
            evaluations,
        })
    }
}

fn check_shooting(config: &TransmissionConfig) -> Result<(), SaddleError> {
    if !(config.thermal.is_finite() && config.thermal > 0.0) {
        return Err(SaddleError::Invalid(
            "shooting kT must be finite and positive".into(),
        ));
    }
    if !(config.friction.is_finite() && config.friction >= 0.0) {
        return Err(SaddleError::Invalid(
            "shooting friction must be finite and non-negative".into(),
        ));
    }
    if !(config.time_step.is_finite() && config.time_step > 0.0) || config.steps == 0 {
        return Err(SaddleError::Invalid(
            "shooting needs a positive time step and at least one step".into(),
        ));
    }
    if config.check_every == 0 || config.blocks < 2 || config.shots < config.blocks {
        return Err(SaddleError::Invalid(
            "shooting needs a checkpoint interval, two blocks and a shot per block".into(),
        ));
    }
    Ok(())
}

fn project_out(vector: &mut Array1<f64>, held: &[Array1<f64>]) {
    for direction in held {
        let along = direction.dot(vector);
        vector.scaled_add(-along, direction);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Watch {
    Forward,
    Backward,
}

struct Shot {
    values: Vec<Option<f64>>,
    recrossed: bool,
}

struct Half {
    states: Vec<Option<Endpoint>>,
    recrossed: Option<usize>,
}

struct Shooter<'a> {
    engine: &'a mut Engine,
    frame: &'a Frame,
    plane: &'a Hyperplane,
    config: &'a TransmissionConfig,
    checkpoints: &'a [usize],
    thermostat: Option<Thermostat>,
    /// The flux direction in the mass-weighted frame, unit length.
    flux: Array1<f64>,
    /// `flux` without its held components, unit length.
    free: Array1<f64>,
    held: Vec<Array1<f64>>,
}

impl Shooter<'_> {
    fn shot(&mut self, start: &Array1<f64>, seed: u64) -> Result<Shot, SaddleError> {
        let config = self.config;
        let mut rng = Rng::new(seed);
        let n = start.len();
        let mut drawn: Array1<f64> = (0..n).map(|_| rng.normal()).collect();
        project_out(&mut drawn, &self.held);
        let along = drawn.dot(&self.flux);
        drawn.scaled_add(-along, &self.flux);
        let thermal = config.thermal.sqrt();
        let rayleigh = (-2.0 * (1.0 - rng.uniform()).ln()).sqrt();
        let mut weighted = &drawn * thermal;
        weighted.scaled_add(rayleigh * thermal, &self.free);
        let speed = velocity_unit();
        let velocity: Vec<f64> = weighted
            .iter()
            .zip(&self.frame.root)
            .map(|(value, root)| speed * value / root)
            .collect();
        let reversed: Vec<f64> = velocity.iter().map(|value| -value).collect();
        let origin = self.frame.cartesian_position(start.view());

        let backward = self.half(&origin, &reversed, rng.u64(), Watch::Backward)?;
        let forward = self.half(&origin, &velocity, rng.u64(), Watch::Forward)?;
        let product =
            |state: Option<Endpoint>| state.map(|s| f64::from(u8::from(s == Endpoint::Product)));
        let values = (0..self.checkpoints.len())
            .map(|k| match config.counting {
                Counting::Final => Some(product(forward.states[k])? - product(backward.states[k])?),
                Counting::FirstCrossing => {
                    if backward.recrossed.is_some_and(|r| r <= k) {
                        return Some(0.0);
                    }
                    let past = backward.states[k]?;
                    let future = forward.states[k]?;
                    Some(f64::from(u8::from(
                        past == Endpoint::Reactant && future == Endpoint::Product,
                    )))
                }
            })
            .collect();
        Ok(Shot {
            values,
            recrossed: backward.recrossed.is_some(),
        })
    }

    /// Integrate one half from `start` with `velocity`. A backward half
    /// records its first return to the product side of the plane, which
    /// ends it when only first crossings count.
    fn half(
        &mut self,
        start: &[f64],
        velocity: &[f64],
        seed: u64,
        watch: Watch,
    ) -> Result<Half, SaddleError> {
        let config = self.config;
        self.engine.set_positions(start)?;
        self.engine.set_velocities(velocity)?;
        if self.thermostat.is_some() {
            self.engine.set_seed(seed)?;
        }
        let mut states = vec![None; self.checkpoints.len()];
        let mut recrossed = None;
        let mut next = 0;
        let mut run: (Option<Endpoint>, usize) = (None, 0);
        let mut done = 0;
        let every_step = watch == Watch::Backward && config.counting == Counting::FirstCrossing;
        while done < config.steps {
            let target = if every_step {
                done + 1
            } else {
                self.checkpoints[next]
            };
            self.engine
                .step(target - done, self.thermostat, None, &[])?;
            done = target;
            let positions = self.engine.positions()?;
            if !positions.iter().all(|value| value.is_finite()) {
                return Err(SaddleError::NonFinite("shooting positions"));
            }
            let side = self.plane.distance(self.frame.position(&positions).view());
            if watch == Watch::Backward && recrossed.is_none() && side > 0.0 {
                recrossed = Some(next);
                if config.counting == Counting::FirstCrossing {
                    break;
                }
            }
            if done == self.checkpoints[next] {
                let state = if side < 0.0 {
                    Some(Endpoint::Reactant)
                } else if side > 0.0 {
                    Some(Endpoint::Product)
                } else {
                    None
                };
                states[next] = state;
                run = match (state, run.0) {
                    (Some(s), Some(r)) if s == r => (state, run.1 + 1),
                    (Some(_), _) => (state, 1),
                    (None, _) => (None, 0),
                };
                next += 1;
                if config.commit > 0 && run.1 >= config.commit {
                    for slot in &mut states[next..] {
                        *slot = state;
                    }
                    break;
                }
                if next == self.checkpoints.len() {
                    break;
                }
            }
        }
        Ok(Half { states, recrossed })
    }
}

/// The map between the Cartesian positions of the session and the
/// coordinates a hyperplane is stated in.
struct Frame {
    masses: Vec<f64>,
    root: Array1<f64>,
    coordinates: HyperplaneCoordinates,
}

impl Frame {
    fn new(masses: Vec<f64>, coordinates: HyperplaneCoordinates) -> Result<Self, SaddleError> {
        if masses.is_empty() || !masses.iter().all(|mass| mass.is_finite() && *mass > 0.0) {
            return Err(SaddleError::Invalid(
                "the session masses must be finite and positive".into(),
            ));
        }
        let root = masses.iter().map(|mass| mass.sqrt()).collect();
        Ok(Self {
            masses,
            root,
            coordinates,
        })
    }

    fn position(&self, cartesian: &[f64]) -> Array1<f64> {
        match self.coordinates {
            HyperplaneCoordinates::Cartesian => Array1::from_vec(cartesian.to_vec()),
            HyperplaneCoordinates::MassWeighted => cartesian
                .iter()
                .zip(&self.root)
                .map(|(x, root)| x * root)
                .collect(),
        }
    }

    fn cartesian_position(&self, position: ArrayView1<f64>) -> Vec<f64> {
        match self.coordinates {
            HyperplaneCoordinates::Cartesian => position.to_vec(),
            HyperplaneCoordinates::MassWeighted => position
                .iter()
                .zip(&self.root)
                .map(|(q, root)| q / root)
                .collect(),
        }
    }

    fn force(&self, cartesian: &[f64]) -> Array1<f64> {
        match self.coordinates {
            HyperplaneCoordinates::Cartesian => Array1::from_vec(cartesian.to_vec()),
            HyperplaneCoordinates::MassWeighted => cartesian
                .iter()
                .zip(&self.root)
                .map(|(f, root)| f / root)
                .collect(),
        }
    }

    fn velocity(&self, cartesian: &[f64]) -> Array1<f64> {
        self.position(cartesian)
    }

    /// The point and the unit gradient of the plane in Cartesian positions.
    fn plane(&self, plane: &Hyperplane) -> Result<(Vec<f64>, Vec<f64>), SaddleError> {
        let (point, normal) = match self.coordinates {
            HyperplaneCoordinates::Cartesian => {
                (plane.point().to_owned(), plane.normal().to_owned())
            }
            HyperplaneCoordinates::MassWeighted => plane.to_cartesian(&self.masses)?,
        };
        Ok((point.to_vec(), normal.to_vec()))
    }

    /// A planar boundary as a wall `normal . x <= offset` of Cartesian positions.
    fn wall(&self, boundary: &PlanarBoundary, dimension: usize) -> Result<Wall, SaddleError> {
        if boundary.point().len() != dimension {
            return Err(SaddleError::Shape(
                "a boundary and the session differ in dimension".into(),
            ));
        }
        let point = self.cartesian_position(boundary.point());
        let normal: Vec<f64> = match self.coordinates {
            HyperplaneCoordinates::Cartesian => boundary.normal().to_vec(),
            HyperplaneCoordinates::MassWeighted => boundary
                .normal()
                .iter()
                .zip(&self.root)
                .map(|(n, root)| n * root)
                .collect(),
        };
        let offset = normal.iter().zip(&point).map(|(n, x)| n * x).sum();
        Ok(Wall { normal, offset })
    }

    /// The rigid translations as orthonormal directions of the
    /// mass-weighted frame.
    fn translations(&self) -> Result<Vec<Array1<f64>>, SaddleError> {
        if !self.root.len().is_multiple_of(3) {
            return Err(SaddleError::Shape(
                "rigid translations need three coordinates per atom".into(),
            ));
        }
        Ok((0..3)
            .map(|axis| {
                let mut direction = Array1::zeros(self.root.len());
                for atom in 0..self.root.len() / 3 {
                    direction[3 * atom + axis] = self.root[3 * atom + axis];
                }
                let norm = direction.dot(&direction).sqrt();
                direction / norm
            })
            .collect())
    }
}

#[derive(Clone, Copy)]
struct Thermostat {
    kt_ev: f64,
    friction_per_ps: f64,
}

struct Wall {
    normal: Vec<f64>,
    offset: f64,
}

struct MdState {
    positions: Vec<f64>,
    velocities: Vec<f64>,
    forces: Vec<f64>,
    potential_ev: f64,
    kinetic_ev: f64,
    force_evaluations: usize,
}

#[repr(C)]
struct RawConfig {
    bytes: u32,
    time_step_ps: f64,
}

#[repr(C)]
struct RawInfo {
    bytes: u32,
    potential_ev: f64,
    kinetic_ev: f64,
    time_ps: f64,
    steps: u64,
    force_evaluations: u64,
    virial_ev: [f64; 9],
    cell: [f64; 9],
}

#[repr(C)]
struct RawThermostat {
    bytes: u32,
    kt_ev: f64,
    friction_per_ps: f64,
}

#[repr(C)]
struct RawPlane {
    bytes: u32,
    dimension: usize,
    point: *const f64,
    normal: *const f64,
}

#[repr(C)]
struct RawWall {
    normal: *const f64,
    offset: f64,
}

type VersionFn = unsafe extern "C" fn() -> c_int;
type FeaturesFn = unsafe extern "C" fn() -> c_uint;
type TextFn = unsafe extern "C" fn() -> *const c_char;
type CreateFn =
    unsafe extern "C" fn(*const c_char, *const RawConfig, *mut c_char, usize) -> *mut c_void;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type AtomCountFn = unsafe extern "C" fn(*const c_void) -> usize;
type MassesFn = unsafe extern "C" fn(*const c_void, *mut f64, usize) -> c_int;
type SetBoxFn = unsafe extern "C" fn(*mut c_void, *const f64) -> c_int;
type SetVectorFn = unsafe extern "C" fn(*mut c_void, *const f64, usize) -> c_int;
type SetSeedFn = unsafe extern "C" fn(*mut c_void, u64) -> c_int;
type InitializeFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type StateFn =
    unsafe extern "C" fn(*mut c_void, *mut f64, *mut f64, *mut f64, usize, *mut RawInfo) -> c_int;
type StepProjectedFn = unsafe extern "C" fn(
    *mut c_void,
    u64,
    *const RawThermostat,
    *const RawPlane,
    *const RawWall,
    usize,
) -> c_int;

struct Engine {
    session: *mut c_void,
    dimension: usize,
    features: c_uint,
    name: String,
    last_error: TextFn,
    destroy: DestroyFn,
    masses: MassesFn,
    set_box: Option<SetBoxFn>,
    set_positions: SetVectorFn,
    set_velocities: SetVectorFn,
    set_seed: Option<SetSeedFn>,
    initialize: Option<InitializeFn>,
    state: StateFn,
    step_projected: Option<StepProjectedFn>,
}

impl Engine {
    fn load(config: &MdSessionConfig, spec: &str, time_step_ps: f64) -> Result<Self, SaddleError> {
        config.check_prefix()?;
        let prefix = config.prefix.as_str();
        let explicit = config.library_path.as_deref().and_then(Path::to_str);
        let mut failures = Vec::new();
        let mut loaded = None;
        for candidate in library_candidates(prefix, explicit) {
            match unsafe { Library::new(&candidate) } {
                Ok(library) => {
                    loaded = Some(library);
                    break;
                }
                Err(error) => failures.push(format!("{candidate}: {error}")),
            }
        }
        let library = loaded.ok_or_else(|| {
            SaddleError::Surface(format!("lib{prefix} not loaded; {}", failures.join("; ")))
        })?;
        let library: &'static Library = Box::leak(Box::new(library));
        unsafe {
            let abi_version: VersionFn = resolve(library, prefix, "abi_version")?;
            let available: VersionFn = resolve(library, prefix, "available")?;
            let features: FeaturesFn = resolve(library, prefix, "features")?;
            let engine_name: TextFn = resolve(library, prefix, "engine")?;
            let last_error: TextFn = resolve(library, prefix, "last_error")?;
            let create: CreateFn = resolve(library, prefix, "create")?;
            let atom_count: AtomCountFn = resolve(library, prefix, "atom_count")?;
            if abi_version() != MD_ABI_VERSION {
                return Err(SaddleError::Surface(format!(
                    "{prefix} speaks MD session ABI {}, this host {MD_ABI_VERSION}",
                    abi_version()
                )));
            }
            if available() == 0 {
                return Err(SaddleError::Surface(format!(
                    "{prefix} was built without its engine"
                )));
            }
            let spec = CString::new(spec).map_err(|_| {
                SaddleError::Invalid("MD session creation text contains a NUL byte".into())
            })?;
            let raw = RawConfig {
                bytes: std::mem::size_of::<RawConfig>() as u32,
                time_step_ps,
            };
            let mut message = vec![0 as c_char; 512];
            let session = create(spec.as_ptr(), &raw, message.as_mut_ptr(), message.len());
            if session.is_null() {
                let reason = CStr::from_ptr(message.as_ptr())
                    .to_string_lossy()
                    .into_owned();
                return Err(SaddleError::Surface(if reason.is_empty() {
                    text(last_error)
                } else {
                    reason
                }));
            }
            let dimension = 3 * atom_count(session);
            Ok(Self {
                session,
                dimension,
                features: features(),
                name: text(engine_name),
                last_error,
                destroy: resolve(library, prefix, "destroy")?,
                masses: resolve(library, prefix, "masses")?,
                set_box: optional(library, prefix, "set_box"),
                set_positions: resolve(library, prefix, "set_positions")?,
                set_velocities: resolve(library, prefix, "set_velocities")?,
                set_seed: optional(library, prefix, "set_seed"),
                initialize: optional(library, prefix, "initialize"),
                state: resolve(library, prefix, "state")?,
                step_projected: optional(library, prefix, "step_projected"),
            })
        }
    }

    fn require(&self, feature: c_uint, what: &str) -> Result<(), SaddleError> {
        if self.features & feature == 0 || self.step_projected.is_none() {
            return Err(SaddleError::Surface(format!(
                "the {} MD session does not provide {what}",
                self.name
            )));
        }
        Ok(())
    }

    fn check(&self, status: c_int) -> Result<(), SaddleError> {
        if status == 0 {
            Ok(())
        } else {
            Err(SaddleError::Surface(text(self.last_error)))
        }
    }

    fn masses(&mut self) -> Result<Vec<f64>, SaddleError> {
        let mut masses = vec![0.0; self.dimension];
        let status = unsafe { (self.masses)(self.session, masses.as_mut_ptr(), masses.len()) };
        self.check(status)?;
        Ok(masses)
    }

    fn set_box(&mut self, cell: &[f64; 9]) -> Result<(), SaddleError> {
        let set = self.set_box.ok_or_else(|| {
            SaddleError::Surface(format!("the {} MD session cannot set a box", self.name))
        })?;
        let status = unsafe { set(self.session, cell.as_ptr()) };
        self.check(status)
    }

    fn set_positions(&mut self, positions: &[f64]) -> Result<(), SaddleError> {
        self.check_dimension(positions.len())?;
        let status =
            unsafe { (self.set_positions)(self.session, positions.as_ptr(), positions.len()) };
        self.check(status)
    }

    fn set_velocities(&mut self, velocities: &[f64]) -> Result<(), SaddleError> {
        self.check_dimension(velocities.len())?;
        let status =
            unsafe { (self.set_velocities)(self.session, velocities.as_ptr(), velocities.len()) };
        self.check(status)
    }

    fn set_seed(&mut self, seed: u64) -> Result<(), SaddleError> {
        let set = self.set_seed.ok_or_else(|| {
            SaddleError::Surface(format!("the {} MD session cannot be seeded", self.name))
        })?;
        let status = unsafe { set(self.session, seed) };
        self.check(status)
    }

    fn check_dimension(&self, length: usize) -> Result<(), SaddleError> {
        if length != self.dimension {
            return Err(SaddleError::Shape(format!(
                "the MD session has {} coordinates, got {length}",
                self.dimension
            )));
        }
        Ok(())
    }

    fn step(
        &mut self,
        steps: usize,
        thermostat: Option<Thermostat>,
        plane: Option<(&[f64], &[f64])>,
        walls: &[Wall],
    ) -> Result<(), SaddleError> {
        let step = self.step_projected.ok_or_else(|| {
            SaddleError::Surface(format!(
                "the {} MD session has no projected steps",
                self.name
            ))
        })?;
        let raw_thermostat = thermostat.map(|t| RawThermostat {
            bytes: std::mem::size_of::<RawThermostat>() as u32,
            kt_ev: t.kt_ev,
            friction_per_ps: t.friction_per_ps,
        });
        let raw_plane = plane.map(|(point, normal)| RawPlane {
            bytes: std::mem::size_of::<RawPlane>() as u32,
            dimension: point.len(),
            point: point.as_ptr(),
            normal: normal.as_ptr(),
        });
        let raw_walls: Vec<RawWall> = walls
            .iter()
            .map(|wall| RawWall {
                normal: wall.normal.as_ptr(),
                offset: wall.offset,
            })
            .collect();
        let status = unsafe {
            step(
                self.session,
                steps as u64,
                raw_thermostat
                    .as_ref()
                    .map_or(std::ptr::null(), std::ptr::from_ref),
                raw_plane
                    .as_ref()
                    .map_or(std::ptr::null(), std::ptr::from_ref),
                if raw_walls.is_empty() {
                    std::ptr::null()
                } else {
                    raw_walls.as_ptr()
                },
                raw_walls.len(),
            )
        };
        self.check(status)
    }

    fn initialize(&mut self) -> Result<(), SaddleError> {
        match self.initialize {
            Some(initialize) => {
                let status = unsafe { initialize(self.session) };
                self.check(status)
            }
            None => Ok(()),
        }
    }

    fn info() -> RawInfo {
        RawInfo {
            bytes: std::mem::size_of::<RawInfo>() as u32,
            potential_ev: 0.0,
            kinetic_ev: 0.0,
            time_ps: 0.0,
            steps: 0,
            force_evaluations: 0,
            virial_ev: [0.0; 9],
            cell: [0.0; 9],
        }
    }

    fn state(&mut self) -> Result<MdState, SaddleError> {
        let mut positions = vec![0.0; self.dimension];
        let mut velocities = vec![0.0; self.dimension];
        let mut forces = vec![0.0; self.dimension];
        let mut info = Self::info();
        let status = unsafe {
            (self.state)(
                self.session,
                positions.as_mut_ptr(),
                velocities.as_mut_ptr(),
                forces.as_mut_ptr(),
                self.dimension,
                &mut info,
            )
        };
        self.check(status)?;
        Ok(MdState {
            positions,
            velocities,
            forces,
            potential_ev: info.potential_ev,
            kinetic_ev: info.kinetic_ev,
            force_evaluations: info.force_evaluations as usize,
        })
    }

    fn positions(&mut self) -> Result<Vec<f64>, SaddleError> {
        let mut positions = vec![0.0; self.dimension];
        let status = unsafe {
            (self.state)(
                self.session,
                positions.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                self.dimension,
                std::ptr::null_mut(),
            )
        };
        self.check(status)?;
        Ok(positions)
    }

    fn force_evaluations(&mut self) -> Result<usize, SaddleError> {
        let mut info = Self::info();
        let status = unsafe {
            (self.state)(
                self.session,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                self.dimension,
                &mut info,
            )
        };
        self.check(status)?;
        Ok(info.force_evaluations as usize)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.session);
        }
        self.session = std::ptr::null_mut();
    }
}

/// Resolve `<prefix>_md_<call>`.
///
/// # Safety
///
/// `T` must be the function pointer type of that call.
unsafe fn resolve<T: Copy>(
    library: &'static Library,
    prefix: &str,
    call: &str,
) -> Result<T, SaddleError> {
    let name = format!("{prefix}_md_{call}\0");
    let symbol = unsafe { library.get::<T>(name.as_bytes()) }
        .map_err(|error| SaddleError::Surface(format!("{prefix}_md_{call}: {error}")))?;
    Ok(*symbol)
}

/// A call an engine may leave out.
///
/// # Safety
///
/// `T` must be the function pointer type of that call.
unsafe fn optional<T: Copy>(library: &'static Library, prefix: &str, call: &str) -> Option<T> {
    unsafe { resolve(library, prefix, call) }.ok()
}

fn text(function: TextFn) -> String {
    let pointer = unsafe { function() };
    if pointer.is_null() {
        "the MD session failed without a diagnostic".into()
    } else {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    fn frame(coordinates: HyperplaneCoordinates) -> Frame {
        Frame::new(vec![4.0, 4.0, 4.0, 1.0, 1.0, 1.0], coordinates).unwrap()
    }

    #[test]
    fn velocity_unit_is_the_speed_of_one_ev_per_amu() {
        // sqrt(e / u) is 9822.69 m/s, which is 98.2269 Angstrom/ps.
        assert_relative_eq!(velocity_unit(), 98.2269, max_relative = 2.0_f64.powi(-20));
    }

    #[test]
    fn configs_are_checked() {
        assert!(MdSessionConfig::default().check().is_err());
        let config = MdSessionConfig::default().with_prefix("rgpot_nblib");
        config.check().unwrap();
        assert!(config.clone().with_time_step_ps(0.0).check().is_err());
        assert!(
            config
                .clone()
                .with_friction_per_ps(f64::NAN)
                .check()
                .is_err()
        );
        assert!(config.with_prefix("a-b").check().is_err());
        assert!(MdSessionSystem::default().check().is_err());
    }

    #[test]
    fn mass_weighting_round_trips() {
        let frame = frame(HyperplaneCoordinates::MassWeighted);
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let q = frame.position(&x);
        assert_relative_eq!(q[0], 2.0);
        assert_relative_eq!(q[3], 4.0);
        let back = frame.cartesian_position(q.view());
        for (a, b) in back.iter().zip(&x) {
            assert_relative_eq!(a, b);
        }
        let f = frame.force(&x);
        assert_relative_eq!(f[0], 0.5);
        assert_relative_eq!(f[5], 6.0);
    }

    #[test]
    fn a_mass_weighted_plane_is_a_cartesian_gradient() {
        let frame = frame(HyperplaneCoordinates::MassWeighted);
        let x = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let n = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let plane = Hyperplane::from_cartesian(x.view(), n.view(), &frame.masses).unwrap();
        let (point, normal) = frame.plane(&plane).unwrap();
        for (a, b) in point.iter().zip(x.iter()) {
            assert_relative_eq!(a, b, epsilon = f64::EPSILON);
        }
        let norm = n.dot(&n).sqrt();
        for (a, b) in normal.iter().zip(n.iter()) {
            assert_relative_eq!(*a, b / norm, epsilon = f64::EPSILON);
        }
    }

    #[test]
    fn a_boundary_becomes_a_wall_of_cartesian_positions() {
        let frame = frame(HyperplaneCoordinates::MassWeighted);
        let point = array![2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let normal = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let boundary = PlanarBoundary::new(point, normal).unwrap();
        let wall = frame.wall(&boundary, 6).unwrap();
        // q_x = 2 x_x for a mass of 4, so the face x_x = 1 has the gradient 2 e_x.
        assert_relative_eq!(wall.normal[0], 2.0);
        assert_relative_eq!(wall.offset, 2.0);
        assert!(frame.wall(&boundary, 3).is_err());
    }

    #[test]
    fn translations_are_orthonormal_in_the_mass_weighted_frame() {
        let frame = frame(HyperplaneCoordinates::MassWeighted);
        let held = frame.translations().unwrap();
        assert_eq!(held.len(), 3);
        for (i, a) in held.iter().enumerate() {
            for (j, b) in held.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(a.dot(b), want, epsilon = f64::EPSILON * 8.0);
            }
        }
    }

    #[test]
    fn a_missing_library_is_a_surface_error() {
        let config = MdSessionConfig::default()
            .with_prefix("rgpot_missing_engine")
            .with_library_path("/nonexistent/librgpot_missing_engine.so");
        let error = Engine::load(&config, "spec", 0.001).err().unwrap();
        assert!(matches!(error, SaddleError::Surface(_)));
    }
}
