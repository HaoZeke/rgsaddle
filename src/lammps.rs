//! LAMMPS constrained molecular dynamics through lammpc.
//!
//! The adapter loads lammpc's in-process profile, creates one persistent
//! LAMMPS session, and advances lammpc's projected BAOAB stepper. LAMMPS
//! supplies the forces. lammpc applies the plane to position, velocity, and
//! force on every step.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::{Path, PathBuf};

use libloading::Library;
use ndarray::Array1;
use rgpot_core::profile::{ProfileRequest, encode_force_input, library_candidates};

use crate::strain_rate::{parse_json, required_f64, required_str};
use crate::{
    Hyperplane, HyperplaneCoordinates, HyperplaneEnsemble, HyperplaneObservation,
    HyperplaneSampleBudget, HyperplaneSampler, SaddleError,
};

const LAMMPC_ABI_VERSION: c_int = 0;
const LAMMPS_THERMOSTAT_RECORD_FORMAT: &str = "rgsaddle.lammps_thermostat";
const LAMMPS_OBSERVABLE_THERMOSTAT: &str = "lammps.thermostat";
const LAMMPS_OBSERVABLE_EFFECTIVE_DOF: &str = "lammps.effective_degrees_of_freedom";

/// Thermostat used by the constrained LAMMPS backend.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LammpsThermostatKind {
    /// Projected BAOAB Langevin thermostat.
    #[default]
    BaoabLangevin,
}

impl LammpsThermostatKind {
    /// Stable string value used in records.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BaoabLangevin => "baoab_langevin",
        }
    }

    fn observable_code(self) -> f64 {
        match self {
            Self::BaoabLangevin => 1.0,
        }
    }
}

impl std::str::FromStr for LammpsThermostatKind {
    type Err = SaddleError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "baoab_langevin" => Ok(Self::BaoabLangevin),
            _ => Err(SaddleError::Invalid(format!(
                "unsupported LAMMPS thermostat {text}"
            ))),
        }
    }
}

/// Thermostat metadata written with each LAMMPS sample record.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LammpsThermostatRecord {
    /// Thermostat kind used to generate the record.
    pub thermostat: LammpsThermostatKind,
    /// Effective constrained degrees of freedom, `3N - 1 - excluded_modes`.
    pub effective_degrees_of_freedom: usize,
}

impl LammpsThermostatRecord {
    /// Serialize this metadata to JSON.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"format\":\"{LAMMPS_THERMOSTAT_RECORD_FORMAT}\",\"thermostat\":\"{}\",\"effective_degrees_of_freedom\":{}}}",
            self.thermostat.as_str(),
            self.effective_degrees_of_freedom
        )
    }

    /// Parse a document written by [`Self::to_json`].
    pub fn from_json(text: &str) -> Result<Self, SaddleError> {
        let value = parse_json(text)?;
        let object = value.as_object().ok_or_else(|| {
            SaddleError::Invalid("LAMMPS thermostat record json must be an object".into())
        })?;
        if required_str(object, "format")? != LAMMPS_THERMOSTAT_RECORD_FORMAT {
            return Err(SaddleError::Invalid(
                "json format is not rgsaddle.lammps_thermostat".into(),
            ));
        }
        let thermostat = required_str(object, "thermostat")?.parse()?;
        let dof = required_f64(object, "effective_degrees_of_freedom")?;
        if !dof.is_finite() || dof < 0.0 || dof.fract() != 0.0 {
            return Err(SaddleError::Invalid(
                "effective_degrees_of_freedom must be a nonnegative integer".into(),
            ));
        }
        Ok(Self {
            thermostat,
            effective_degrees_of_freedom: dof as usize,
        })
    }

    fn observables(&self) -> [(String, f64); 2] {
        [
            (
                LAMMPS_OBSERVABLE_THERMOSTAT.to_string(),
                self.thermostat.observable_code(),
            ),
            (
                LAMMPS_OBSERVABLE_EFFECTIVE_DOF.to_string(),
                self.effective_degrees_of_freedom as f64,
            ),
        ]
    }
}

/// Atomistic geometry and lammpc interaction-model configuration.
///
/// `config_bytes` is a flat rgpot `PotentialConfig` carrying its `lammps`
/// arm. `positions` and `box_matrix` use angstrom. `masses` has one value in
/// amu per atom and must match the per-type masses in that configuration.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct LammpsSystem {
    /// Flat rgpot `PotentialConfig` bytes.
    pub config_bytes: Vec<u8>,
    /// Cartesian coordinates, three values per atom, in angstrom.
    pub positions: Array1<f64>,
    /// Atomic numbers in position order.
    pub atomic_numbers: Vec<i32>,
    /// Atomic masses in amu, one per atom.
    pub masses: Vec<f64>,
    /// Optional row-major simulation cell in angstrom.
    pub box_matrix: Option<[f64; 9]>,
}

impl LammpsSystem {
    /// Set [`Self::config_bytes`].
    #[must_use]
    pub fn with_config_bytes(mut self, config_bytes: Vec<u8>) -> Self {
        self.config_bytes = config_bytes;
        self
    }

    /// Set [`Self::positions`].
    #[must_use]
    pub fn with_positions(mut self, positions: Array1<f64>) -> Self {
        self.positions = positions;
        self
    }

    /// Set [`Self::atomic_numbers`].
    #[must_use]
    pub fn with_atomic_numbers(mut self, atomic_numbers: Vec<i32>) -> Self {
        self.atomic_numbers = atomic_numbers;
        self
    }

    /// Set [`Self::masses`].
    #[must_use]
    pub fn with_masses(mut self, masses: Vec<f64>) -> Self {
        self.masses = masses;
        self
    }

    /// Set [`Self::box_matrix`].
    #[must_use]
    pub fn with_box_matrix(mut self, box_matrix: [f64; 9]) -> Self {
        self.box_matrix = Some(box_matrix);
        self
    }

    fn check(&self) -> Result<(), SaddleError> {
        if self.config_bytes.is_empty() {
            return Err(SaddleError::Invalid(
                "LAMMPS PotentialConfig bytes are empty".into(),
            ));
        }
        if self.positions.is_empty() || !self.positions.len().is_multiple_of(3) {
            return Err(SaddleError::Shape(
                "LAMMPS positions need three coordinates per atom".into(),
            ));
        }
        let atoms = self.positions.len() / 3;
        if self.atomic_numbers.len() != atoms || self.masses.len() != atoms {
            return Err(SaddleError::Shape(
                "LAMMPS needs one atomic number and mass per atom".into(),
            ));
        }
        if !self.positions.iter().all(|value| value.is_finite())
            || !self
                .masses
                .iter()
                .all(|mass| mass.is_finite() && *mass > 0.0)
            || self
                .box_matrix
                .is_some_and(|cell| !cell.iter().all(|value| value.is_finite()))
        {
            return Err(SaddleError::NonFinite("LAMMPS system"));
        }
        Ok(())
    }

    fn coordinate_masses(&self) -> impl Iterator<Item = f64> + '_ {
        self.masses.iter().flat_map(|mass| [*mass; 3])
    }
}

/// Runtime controls for lammpc constrained dynamics.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct LammpsSamplerConfig {
    /// Optional explicit `liblammpc` path.
    pub library_path: Option<PathBuf>,
    /// BAOAB time step in ps.
    pub time_step_ps: f64,
    /// Langevin friction in inverse ps.
    pub friction_per_ps: f64,
    /// Thermostat used by the constrained dynamics.
    pub thermostat: LammpsThermostatKind,
    /// Modes held by the plane construction in addition to the plane normal.
    pub excluded_modes: usize,
}

impl Default for LammpsSamplerConfig {
    fn default() -> Self {
        Self {
            library_path: None,
            time_step_ps: 0.001,
            friction_per_ps: 1.0,
            thermostat: LammpsThermostatKind::BaoabLangevin,
            excluded_modes: 0,
        }
    }
}

impl LammpsSamplerConfig {
    /// Set an explicit `liblammpc` path.
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

    /// Set [`Self::thermostat`].
    #[must_use]
    pub fn with_thermostat(mut self, thermostat: LammpsThermostatKind) -> Self {
        self.thermostat = thermostat;
        self
    }

    /// Set [`Self::excluded_modes`].
    #[must_use]
    pub fn with_excluded_modes(mut self, excluded_modes: usize) -> Self {
        self.excluded_modes = excluded_modes;
        self
    }

    /// Effective constrained degrees of freedom, `3N - 1 - excluded_modes`.
    pub fn effective_degrees_of_freedom(&self, n_atoms: usize) -> Result<usize, SaddleError> {
        let cartesian = n_atoms
            .checked_mul(3)
            .ok_or_else(|| SaddleError::Invalid("LAMMPS atom count overflow".into()))?;
        let constrained = 1usize
            .checked_add(self.excluded_modes)
            .ok_or_else(|| SaddleError::Invalid("LAMMPS excluded-mode count overflow".into()))?;
        cartesian.checked_sub(constrained).ok_or_else(|| {
            SaddleError::Invalid(format!(
                "LAMMPS needs 3N > 1 + excluded_modes, got N={n_atoms}, excluded_modes={}",
                self.excluded_modes
            ))
        })
    }

    fn thermostat_record(&self, n_atoms: usize) -> Result<LammpsThermostatRecord, SaddleError> {
        Ok(LammpsThermostatRecord {
            thermostat: self.thermostat,
            effective_degrees_of_freedom: self.effective_degrees_of_freedom(n_atoms)?,
        })
    }

    fn check(&self) -> Result<(), SaddleError> {
        if !(self.time_step_ps.is_finite() && self.time_step_ps > 0.0) {
            return Err(SaddleError::Invalid(
                "LAMMPS time step must be finite and positive".into(),
            ));
        }
        if !(self.friction_per_ps.is_finite() && self.friction_per_ps > 0.0) {
            return Err(SaddleError::Invalid(
                "LAMMPS friction must be finite and positive".into(),
            ));
        }
        Ok(())
    }
}

/// A real-LAMMPS implementation of [`HyperplaneSampler`].
pub struct LammpsHyperplaneSampler {
    system: LammpsSystem,
    config: LammpsSamplerConfig,
}

impl LammpsHyperplaneSampler {
    /// Build a sampler. The shared library is loaded when sampling begins.
    pub fn new(system: LammpsSystem, config: LammpsSamplerConfig) -> Self {
        Self { system, config }
    }

    /// Atomistic system used to initialize each ensemble.
    pub fn system(&self) -> &LammpsSystem {
        &self.system
    }

    /// Runtime controls used by lammpc.
    pub fn config(&self) -> &LammpsSamplerConfig {
        &self.config
    }
}

impl HyperplaneSampler for LammpsHyperplaneSampler {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        self.system.check()?;
        self.config.check()?;
        if plane.dim() != self.system.positions.len() {
            return Err(SaddleError::Shape(
                "LAMMPS system and hyperplane dimensions differ".into(),
            ));
        }
        let thermostat = self.config.thermostat_record(self.system.masses.len())?;
        if !(budget.thermal.is_finite() && budget.thermal > 0.0) {
            return Err(SaddleError::Invalid(
                "LAMMPS kT must be finite and positive".into(),
            ));
        }
        if budget.stride == 0 {
            return Err(SaddleError::Invalid(
                "LAMMPS sampling stride must be positive".into(),
            ));
        }

        let request =
            ProfileRequest {
                positions: self.system.positions.as_slice().ok_or_else(|| {
                    SaddleError::Shape("LAMMPS positions must be contiguous".into())
                })?,
                atomic_numbers: &self.system.atomic_numbers,
                box_matrix: self.system.box_matrix.as_ref(),
                length_unit: "angstrom",
                energy_unit: "eV",
            };
        let force_input = encode_force_input(&request)
            .map_err(|error| SaddleError::Surface(error.to_string()))?;
        let mut engine = Engine::load(
            self.config.library_path.as_deref(),
            &self.system.config_bytes,
        )?;
        let md_config = MdConfig {
            time_step_ps: self.config.time_step_ps,
            thermal_energy_ev: budget.thermal,
            friction_per_ps: self.config.friction_per_ps,
            seed: budget.seed,
            coordinates: match budget.coordinates {
                HyperplaneCoordinates::MassWeighted => 0,
                HyperplaneCoordinates::Cartesian => 1,
            },
        };
        engine.initialize(
            &force_input,
            plane.point().as_slice().expect("hyperplane is contiguous"),
            plane.normal().as_slice().expect("hyperplane is contiguous"),
            &md_config,
        )?;
        if let Some(region) = &budget.region {
            let boundaries = region.planar_boundaries().ok_or_else(|| {
                SaddleError::Invalid(
                    "LAMMPS requires a region with engine-portable planar boundaries".into(),
                )
            })?;
            engine.set_hard_boundary(
                boundaries,
                plane.clone(),
                self.system.coordinate_masses().collect(),
                budget.coordinates,
            )?;
        }
        engine.step(budget.thermalize)?;

        let mut observations = Vec::with_capacity(budget.samples);
        let mut accounting = MdObservation::default();
        for _ in 0..budget.samples {
            engine.step(budget.stride)?;
            let state = engine.state(plane.dim())?;
            accounting = state.accounting;
            let (position, projected_force, projected_velocity) =
                convert_state(&self.system, budget.coordinates, state);
            if let Some(region) = &budget.region
                && !region.contains(position.view())?
            {
                return Err(SaddleError::Surface(
                    "lammpc returned a configuration outside the armed hard region".into(),
                ));
            }
            let observables = thermostat.observables().into_iter().collect();
            observations.push(HyperplaneObservation {
                constraint_force: accounting.constraint_force,
                pafi_gradient: None,
                jacobian: 0.0,
                position: budget.positions.then_some(position),
                projected_force: Some(projected_force),
                projected_velocity: Some(projected_velocity),
                observables,
            });
        }
        Ok(HyperplaneEnsemble {
            observations,
            evaluations: accounting.force_evaluations,
            acceptance: 1.0,
            hard_reflections: engine.hard_reflections(),
        })
    }
}

fn convert_state(
    system: &LammpsSystem,
    coordinates: HyperplaneCoordinates,
    state: MdState,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    match coordinates {
        HyperplaneCoordinates::Cartesian => (
            Array1::from_vec(state.positions),
            Array1::from_vec(state.forces),
            Array1::from_vec(state.removed_velocities),
        ),
        HyperplaneCoordinates::MassWeighted => {
            let masses: Vec<f64> = system.coordinate_masses().collect();
            let position = state
                .positions
                .iter()
                .zip(&masses)
                .map(|(value, mass)| value * mass.sqrt())
                .collect();
            let force = state
                .forces
                .iter()
                .zip(&masses)
                .map(|(value, mass)| value / mass.sqrt())
                .collect();
            let velocity = state
                .removed_velocities
                .iter()
                .zip(&masses)
                .map(|(value, mass)| value * mass.sqrt())
                .collect();
            (position, force, velocity)
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MdConfig {
    time_step_ps: f64,
    thermal_energy_ev: f64,
    friction_per_ps: f64,
    seed: u64,
    coordinates: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MdObservation {
    constraint_force: f64,
    force_evaluations: usize,
    steps: usize,
}

type AvailableFn = unsafe extern "C" fn() -> c_int;
type AbiVersionFn = unsafe extern "C" fn() -> c_int;
type LastErrorFn = unsafe extern "C" fn() -> *const c_char;
type FinalizeFn = unsafe extern "C" fn();
type SessionCreateFn = unsafe extern "C" fn(*const c_void, usize) -> *mut c_void;
type SessionDestroyFn = unsafe extern "C" fn(*mut c_void);
type MdInitializeFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    usize,
    *const f64,
    *const f64,
    usize,
    *const MdConfig,
) -> c_int;
type MdStepFn = unsafe extern "C" fn(*mut c_void, usize) -> c_int;
type FeatureFindFn = unsafe extern "C" fn(*const c_char) -> *const LammpcFeatureEntry;
type HardBoundaryFn =
    unsafe extern "C" fn(*mut c_void, c_int, *mut f64, *mut f64, *const LammpcBoundaryBox) -> c_int;
type SetHardBoundaryFn =
    unsafe extern "C" fn(*mut c_void, Option<HardBoundaryFn>, *mut c_void) -> c_int;
type MdStateFn = unsafe extern "C" fn(
    *mut c_void,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
    usize,
    *mut MdObservation,
) -> c_int;

struct MdState {
    positions: Vec<f64>,
    forces: Vec<f64>,
    removed_velocities: Vec<f64>,
    accounting: MdObservation,
}

#[repr(C)]
struct LammpcFeatureEntry {
    feature_id: *const c_char,
    kind: c_int,
    stub_applicable: c_int,
    embed_applicable: c_int,
}

#[repr(C)]
struct LammpcBoundaryBox {
    lo: [f64; 3],
    hi: [f64; 3],
    xy: f64,
    xz: f64,
    yz: f64,
    periodic: [c_int; 3],
    triclinic: c_int,
}

struct HardBoundaryContext {
    boundaries: Vec<crate::PlanarBoundary>,
    plane: Hyperplane,
    masses: Vec<f64>,
    coordinates: HyperplaneCoordinates,
    reflections: Vec<usize>,
    error: Option<String>,
}

extern "C" fn hard_boundary_callback(
    user: *mut c_void,
    n_atoms: c_int,
    positions: *mut f64,
    velocities: *mut f64,
    _box: *const LammpcBoundaryBox,
) -> c_int {
    const REJECT: c_int = 1;
    const REFLECT: c_int = 2;
    if user.is_null() || positions.is_null() || velocities.is_null() || n_atoms < 0 {
        return REJECT;
    }
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let context = unsafe { &mut *user.cast::<HardBoundaryContext>() };
        let dimension = usize::try_from(n_atoms)
            .ok()
            .and_then(|atoms| atoms.checked_mul(3))
            .ok_or_else(|| "lammpc hard-boundary atom count overflow".to_string())?;
        if dimension != context.plane.dim() || dimension != context.masses.len() {
            return Err("lammpc hard-boundary geometry differs from the region".to_string());
        }
        let positions = unsafe { std::slice::from_raw_parts_mut(positions, dimension) };
        let velocities = unsafe { std::slice::from_raw_parts_mut(velocities, dimension) };
        let mut q = Array1::from_vec(positions.to_vec());
        let mut velocity = Array1::from_vec(velocities.to_vec());
        if context.coordinates == HyperplaneCoordinates::MassWeighted {
            for ((coordinate, speed), mass) in
                q.iter_mut().zip(velocity.iter_mut()).zip(&context.masses)
            {
                let root = mass.sqrt();
                *coordinate *= root;
                *speed *= root;
            }
        }
        let mut reflected = false;
        for _ in 0..8 {
            let mut pass_reflected = false;
            for (index, boundary) in context.boundaries.iter().enumerate() {
                let penetration = boundary
                    .signed_distance(q.view())
                    .map_err(|error| error.to_string())?;
                if penetration <= 0.0 {
                    continue;
                }
                let mut tangent = boundary.normal().to_owned();
                let normal_component = tangent.dot(&context.plane.normal());
                tangent.scaled_add(-normal_component, &context.plane.normal());
                let scale = tangent.dot(&tangent).sqrt();
                if !(scale.is_finite() && scale > 0.0) {
                    return Err(
                        "hard-region boundary has no component in the hyperplane".to_string()
                    );
                }
                tangent /= scale;
                q.scaled_add(-2.0 * penetration / scale, &tangent);
                let component = velocity.dot(&tangent);
                velocity.scaled_add(-2.0 * component, &tangent);
                let plane_component = velocity.dot(&context.plane.normal());
                velocity.scaled_add(-plane_component, &context.plane.normal());
                q = context.plane.project(q.view());
                context.reflections[index] += 1;
                reflected = true;
                pass_reflected = true;
            }
            if !pass_reflected {
                break;
            }
        }
        if context
            .boundaries
            .iter()
            .map(|boundary| boundary.signed_distance(q.view()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?
            .into_iter()
            .any(|distance| distance > 0.0)
        {
            return Err("hard-region reflection did not return inside the region".to_string());
        }
        if !reflected {
            return Ok(false);
        }
        if context.coordinates == HyperplaneCoordinates::MassWeighted {
            for ((coordinate, speed), mass) in
                q.iter_mut().zip(velocity.iter_mut()).zip(&context.masses)
            {
                let root = mass.sqrt();
                *coordinate /= root;
                *speed /= root;
            }
        }
        positions.copy_from_slice(q.as_slice().expect("boundary position is contiguous"));
        velocities.copy_from_slice(
            velocity
                .as_slice()
                .expect("boundary velocity is contiguous"),
        );
        Ok(true)
    }));
    match outcome {
        Ok(Ok(false)) => 0,
        Ok(Ok(true)) => REFLECT,
        Ok(Err(message)) => {
            let context = unsafe { &mut *user.cast::<HardBoundaryContext>() };
            context.error = Some(message);
            REJECT
        }
        Err(_) => {
            let context = unsafe { &mut *user.cast::<HardBoundaryContext>() };
            context.error = Some("panic in lammpc hard-boundary callback".into());
            REJECT
        }
    }
}

struct Engine {
    session: *mut c_void,
    destroy: SessionDestroyFn,
    finalize: FinalizeFn,
    last_error: LastErrorFn,
    md_initialize: MdInitializeFn,
    md_step: MdStepFn,
    feature_find: Option<FeatureFindFn>,
    set_hard_boundary: Option<SetHardBoundaryFn>,
    hard_boundary: Option<Box<HardBoundaryContext>>,
    md_state: MdStateFn,
    _library: Library,
}

impl Engine {
    fn load(path: Option<&Path>, config: &[u8]) -> Result<Self, SaddleError> {
        let explicit = path.and_then(Path::to_str);
        let mut failures = Vec::new();
        let mut loaded = None;
        for candidate in library_candidates("lammpc", explicit) {
            match unsafe { Library::new(&candidate) } {
                Ok(library) => {
                    loaded = Some(library);
                    break;
                }
                Err(error) => failures.push(format!("{candidate}: {error}")),
            }
        }
        let library = loaded.ok_or_else(|| {
            SaddleError::Surface(format!("liblammpc not loaded; {}", failures.join("; ")))
        })?;
        unsafe {
            let available = *library
                .get::<AvailableFn>(b"lammpc_available\0")
                .map_err(load_error)?;
            let abi_version = *library
                .get::<AbiVersionFn>(b"lammpc_abi_version\0")
                .map_err(load_error)?;
            let last_error = *library
                .get::<LastErrorFn>(b"lammpc_last_error\0")
                .map_err(load_error)?;
            let finalize = *library
                .get::<FinalizeFn>(b"lammpc_finalize\0")
                .map_err(load_error)?;
            let create = *library
                .get::<SessionCreateFn>(b"lammpc_session_create_from_config\0")
                .map_err(load_error)?;
            let destroy = *library
                .get::<SessionDestroyFn>(b"lammpc_session_destroy\0")
                .map_err(load_error)?;
            let md_initialize = *library
                .get::<MdInitializeFn>(b"lammpc_session_md_initialize\0")
                .map_err(load_error)?;
            let md_step = *library
                .get::<MdStepFn>(b"lammpc_session_md_step\0")
                .map_err(load_error)?;
            let feature_find = library
                .get::<FeatureFindFn>(b"lammpc_feature_find\0")
                .ok()
                .map(|symbol| *symbol);
            let set_hard_boundary = library
                .get::<SetHardBoundaryFn>(b"lammpc_session_set_hard_boundary\0")
                .ok()
                .map(|symbol| *symbol);
            let md_state = *library
                .get::<MdStateFn>(b"lammpc_session_md_state\0")
                .map_err(load_error)?;
            if available() == 0 {
                return Err(SaddleError::Surface(
                    "lammpc was built without LAMMPS".into(),
                ));
            }
            if abi_version() != LAMMPC_ABI_VERSION {
                return Err(SaddleError::Surface(format!(
                    "lammpc ABI {} does not match {}",
                    abi_version(),
                    LAMMPC_ABI_VERSION
                )));
            }
            let session = create(config.as_ptr().cast(), config.len());
            if session.is_null() {
                return Err(SaddleError::Surface(last_error_message(last_error)));
            }
            Ok(Self {
                session,
                destroy,
                finalize,
                last_error,
                md_initialize,
                md_step,
                feature_find,
                set_hard_boundary,
                hard_boundary: None,
                md_state,
                _library: library,
            })
        }
    }

    fn initialize(
        &mut self,
        force_input: &[u8],
        point: &[f64],
        normal: &[f64],
        config: &MdConfig,
    ) -> Result<(), SaddleError> {
        let status = unsafe {
            (self.md_initialize)(
                self.session,
                force_input.as_ptr().cast(),
                force_input.len(),
                point.as_ptr(),
                normal.as_ptr(),
                point.len(),
                config,
            )
        };
        self.check(status)
    }

    fn step(&mut self, steps: usize) -> Result<(), SaddleError> {
        let status = unsafe { (self.md_step)(self.session, steps) };
        self.check(status)?;
        if let Some(message) = self
            .hard_boundary
            .as_mut()
            .and_then(|context| context.error.take())
        {
            return Err(SaddleError::Surface(message));
        }
        Ok(())
    }

    fn set_hard_boundary(
        &mut self,
        boundaries: Vec<crate::PlanarBoundary>,
        plane: Hyperplane,
        masses: Vec<f64>,
        coordinates: HyperplaneCoordinates,
    ) -> Result<(), SaddleError> {
        let find = self.feature_find.ok_or_else(|| {
            SaddleError::Surface(
                "liblammpc lacks the feature table needed to negotiate hard boundaries".into(),
            )
        })?;
        let feature = unsafe { find(c"abi.lammpc_session_set_hard_boundary".as_ptr()).as_ref() }
            .filter(|entry| entry.kind == 2 && entry.embed_applicable != 0)
            .ok_or_else(|| {
                SaddleError::Surface(
                    "liblammpc does not advertise abi.lammpc_session_set_hard_boundary".into(),
                )
            })?;
        let advertised = unsafe { CStr::from_ptr(feature.feature_id) };
        if advertised.to_bytes() != b"abi.lammpc_session_set_hard_boundary" {
            return Err(SaddleError::Surface(
                "liblammpc returned the wrong hard-boundary feature".into(),
            ));
        }
        let set = self.set_hard_boundary.ok_or_else(|| {
            SaddleError::Surface(
                "liblammpc advertises but does not export lammpc_session_set_hard_boundary".into(),
            )
        })?;
        for boundary in &boundaries {
            if boundary.point().len() != plane.dim() {
                return Err(SaddleError::Shape(
                    "LAMMPS boundary and plane dimensions differ".into(),
                ));
            }
        }
        let reflections = vec![0; boundaries.len()];
        let mut context = Box::new(HardBoundaryContext {
            boundaries,
            plane,
            masses,
            coordinates,
            reflections,
            error: None,
        });
        let user = (&mut *context as *mut HardBoundaryContext).cast();
        let status = unsafe { set(self.session, Some(hard_boundary_callback), user) };
        self.check(status)?;
        self.hard_boundary = Some(context);
        Ok(())
    }

    fn hard_reflections(&self) -> Vec<usize> {
        self.hard_boundary
            .as_ref()
            .map_or_else(Vec::new, |context| context.reflections.clone())
    }

    fn state(&mut self, dimension: usize) -> Result<MdState, SaddleError> {
        let mut positions = vec![0.0; dimension];
        let mut velocities = vec![0.0; dimension];
        let mut forces = vec![0.0; dimension];
        let mut removed_velocities = vec![0.0; dimension];
        let mut accounting = MdObservation::default();
        let status = unsafe {
            (self.md_state)(
                self.session,
                positions.as_mut_ptr(),
                velocities.as_mut_ptr(),
                forces.as_mut_ptr(),
                removed_velocities.as_mut_ptr(),
                dimension,
                &mut accounting,
            )
        };
        self.check(status)?;
        Ok(MdState {
            positions,
            forces,
            removed_velocities,
            accounting,
        })
    }

    fn check(&self, status: c_int) -> Result<(), SaddleError> {
        if status == 0 {
            Ok(())
        } else {
            Err(SaddleError::Surface(last_error_message(self.last_error)))
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.session);
            self.session = std::ptr::null_mut();
            (self.finalize)();
        }
    }
}

fn load_error(error: libloading::Error) -> SaddleError {
    SaddleError::Surface(error.to_string())
}

fn last_error_message(last_error: LastErrorFn) -> String {
    let pointer = unsafe { last_error() };
    if pointer.is_null() {
        "lammpc failed without a diagnostic".into()
    } else {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn thermostat_record_round_trips() {
        let config = LammpsSamplerConfig::default().with_excluded_modes(2);
        let record = config.thermostat_record(4).unwrap();
        assert_eq!(record.thermostat, LammpsThermostatKind::BaoabLangevin);
        assert_eq!(record.effective_degrees_of_freedom, 9);
        let text = record.to_json();
        assert_eq!(LammpsThermostatRecord::from_json(&text).unwrap(), record);
    }

    #[test]
    fn effective_degrees_of_freedom_rejects_overconstrained_systems() {
        let error = LammpsSamplerConfig::default()
            .with_excluded_modes(3)
            .effective_degrees_of_freedom(1)
            .expect_err("3N - 1 - excluded_modes must be nonnegative");
        assert!(matches!(error, SaddleError::Invalid(_)));
    }

    #[test]
    fn thermostat_metadata_maps_to_record_observables() {
        let record = LammpsThermostatRecord {
            thermostat: LammpsThermostatKind::BaoabLangevin,
            effective_degrees_of_freedom: 14,
        };
        let observables: BTreeMap<_, _> = record.observables().into_iter().collect();
        assert_eq!(observables.get(LAMMPS_OBSERVABLE_THERMOSTAT), Some(&1.0));
        assert_eq!(
            observables.get(LAMMPS_OBSERVABLE_EFFECTIVE_DOF),
            Some(&14.0)
        );
    }

    #[test]
    fn hard_boundary_callback_reflects_and_counts_each_face() {
        let boundary = crate::PlanarBoundary::new(
            ndarray::array![0.0, 0.0, 0.0],
            ndarray::array![1.0, 0.0, 0.0],
        )
        .unwrap();
        let plane = Hyperplane::new(
            ndarray::array![0.0, 0.0, 0.0],
            ndarray::array![0.0, 0.0, 1.0],
        )
        .unwrap();
        let mut context = HardBoundaryContext {
            boundaries: vec![boundary],
            plane,
            masses: vec![1.0; 3],
            coordinates: HyperplaneCoordinates::Cartesian,
            reflections: vec![0],
            error: None,
        };
        let mut position = [0.25, 0.0, 0.0];
        let mut velocity = [0.5, 0.0, 0.0];
        let status = hard_boundary_callback(
            (&mut context as *mut HardBoundaryContext).cast(),
            1,
            position.as_mut_ptr(),
            velocity.as_mut_ptr(),
            std::ptr::null(),
        );
        assert_eq!(status, 2);
        assert_eq!(position, [-0.25, 0.0, 0.0]);
        assert_eq!(velocity, [-0.5, 0.0, 0.0]);
        assert_eq!(context.reflections, vec![1]);
        assert!(context.error.is_none());
    }
}
