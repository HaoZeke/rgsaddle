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

use crate::{
    Hyperplane, HyperplaneCoordinates, HyperplaneEnsemble, HyperplaneObservation,
    HyperplaneSampleBudget, HyperplaneSampler, SaddleError,
};

const LAMMPC_ABI_VERSION: c_int = 0;

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
}

impl Default for LammpsSamplerConfig {
    fn default() -> Self {
        Self {
            library_path: None,
            time_step_ps: 0.001,
            friction_per_ps: 1.0,
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
        engine.step(budget.thermalize)?;

        let mut observations = Vec::with_capacity(budget.samples);
        let mut accounting = MdObservation::default();
        for _ in 0..budget.samples {
            engine.step(budget.stride)?;
            let state = engine.state(plane.dim())?;
            accounting = state.accounting;
            let (position, projected_force, projected_velocity) =
                convert_state(&self.system, budget.coordinates, state);
            observations.push(HyperplaneObservation {
                constraint_force: accounting.constraint_force,
                pafi_gradient: None,
                jacobian: 0.0,
                position: budget.positions.then_some(position),
                projected_force: Some(projected_force),
                projected_velocity: Some(projected_velocity),
                observables: Default::default(),
            });
        }
        Ok(HyperplaneEnsemble {
            observations,
            evaluations: accounting.force_evaluations,
            acceptance: 1.0,
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

struct Engine {
    session: *mut c_void,
    destroy: SessionDestroyFn,
    finalize: FinalizeFn,
    last_error: LastErrorFn,
    md_initialize: MdInitializeFn,
    md_step: MdStepFn,
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
        self.check(status)
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
