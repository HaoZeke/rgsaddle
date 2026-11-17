//! Unconstrained LAMMPS trajectory shooting through lammpc's NVE ABI.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::Path;

use libloading::Library;
use ndarray::Array1;
use rgpot_core::profile::{ProfileRequest, encode_force_input, library_candidates};

use crate::SaddleError;
use crate::lammps::{LammpsSamplerConfig, LammpsSystem};
use crate::ohtst::{
    Counting, EndState, Endpoint, Hyperplane, HyperplaneDynamics, TransmissionCoefficient,
    TransmissionConfig,
};
use crate::stats::{Rng, block_mean};

const LAMMPC_ABI_VERSION: c_int = 0;
const MASS_VELOCITY_TO_EV: f64 = 1.036_426_965_626_217_5e-4;

/// LAMMPS implementation of [`HyperplaneDynamics`].
///
/// This backend uses lammpc's opt-in unconstrained NVE session ABI.  Plane
/// configurations and velocities are set independently for every trajectory
/// half, so forward and time-reversed shots have exactly opposite momenta.
pub struct LammpsHyperplaneDynamics {
    system: LammpsSystem,
    config: LammpsSamplerConfig,
}

impl LammpsHyperplaneDynamics {
    /// Build a dynamics backend. The shared library is loaded when shooting.
    #[must_use]
    pub fn new(system: LammpsSystem, config: LammpsSamplerConfig) -> Self {
        Self { system, config }
    }

    /// Atomistic system used to initialize every trajectory half.
    #[must_use]
    pub fn system(&self) -> &LammpsSystem {
        &self.system
    }

    /// Runtime controls used to locate lammpc.
    #[must_use]
    pub fn config(&self) -> &LammpsSamplerConfig {
        &self.config
    }
}

impl HyperplaneDynamics for LammpsHyperplaneDynamics {
    fn shoot_hyperplane(
        &mut self,
        plane: &Hyperplane,
        starts: &[Array1<f64>],
        config: &TransmissionConfig,
    ) -> Result<TransmissionCoefficient, SaddleError> {
        self.system.check_for_dynamics()?;
        check_config(config, plane.dim())?;
        if config.friction != 0.0 {
            return Err(SaddleError::Invalid(
                "LAMMPS NVE shooting requires zero friction".into(),
            ));
        }
        if !matches!(config.end_state, EndState::Side) {
            return Err(SaddleError::Invalid(
                "LAMMPS NVE shooting currently classifies endpoints by plane side".into(),
            ));
        }
        if plane.dim() != self.system.positions.len() {
            return Err(SaddleError::Shape(
                "LAMMPS system and hyperplane dimensions differ".into(),
            ));
        }
        if starts.is_empty() {
            return Err(SaddleError::Invalid(
                "shooting needs at least one start on the plane".into(),
            ));
        }
        if starts.iter().any(|start| start.len() != plane.dim()) {
            return Err(SaddleError::Shape(
                "a start differs in length from the plane".into(),
            ));
        }

        let masses: Array1<f64> = self
            .system
            .masses
            .iter()
            .flat_map(|mass| [*mass; 3])
            .collect();
        if let Some(configured) = &config.masses
            && !configured
                .iter()
                .zip(masses.iter())
                .all(|(left, right)| left == right)
        {
            return Err(SaddleError::Invalid(
                "LAMMPS Cartesian shooting masses must match the system masses".into(),
            ));
        }
        let mass_weighted = config.masses.is_none();
        let root = masses.mapv(f64::sqrt);
        let normal = plane.normal().to_owned();
        let mut flux_normal = if mass_weighted {
            normal.clone()
        } else {
            &normal / &root
        };
        flux_normal /= flux_normal.dot(&flux_normal).sqrt();
        let checkpoints = checkpoints(config);
        let mut shots = Vec::with_capacity(config.shots);
        for shot in 0..config.shots {
            let start = plane.project(starts[shot * starts.len() / config.shots].view());
            shots.push(self.shot(
                plane,
                &start,
                config,
                &root,
                &flux_normal,
                mass_weighted,
                &checkpoints,
                config.seed.wrapping_add(shot as u64),
            )?);
        }

        let last = checkpoints.len() - 1;
        let mut kappa_of_t = Vec::with_capacity(checkpoints.len());
        for checkpoint in 0..checkpoints.len() {
            let values: Vec<f64> = shots
                .iter()
                .filter_map(|shot| shot.values[checkpoint])
                .collect();
            kappa_of_t.push(if values.is_empty() {
                f64::NAN
            } else {
                values.iter().sum::<f64>() / values.len() as f64
            });
        }
        let finals: Vec<f64> = shots.iter().filter_map(|shot| shot.values[last]).collect();
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
            recrossed: shots.iter().filter(|shot| shot.recrossed).count(),
            evaluations: shots.iter().map(|shot| shot.evaluations).sum(),
        })
    }
}

impl LammpsSystem {
    fn check_for_dynamics(&self) -> Result<(), SaddleError> {
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
}

struct Shot {
    values: Vec<Option<f64>>,
    recrossed: bool,
    evaluations: usize,
}

struct Half {
    states: Vec<Option<Endpoint>>,
    recrossed: Option<usize>,
    evaluations: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Watch {
    Forward,
    Backward,
}

impl LammpsHyperplaneDynamics {
    #[allow(clippy::too_many_arguments)]
    fn shot(
        &self,
        plane: &Hyperplane,
        start: &Array1<f64>,
        config: &TransmissionConfig,
        root: &Array1<f64>,
        flux_normal: &Array1<f64>,
        mass_weighted: bool,
        checkpoints: &[usize],
        seed: u64,
    ) -> Result<Shot, SaddleError> {
        let mut rng = Rng::new(seed);
        let mut drawn: Array1<f64> = (0..start.len()).map(|_| rng.normal()).collect();
        let along = drawn.dot(flux_normal);
        drawn.scaled_add(-along, flux_normal);
        let thermal = config.thermal.sqrt();
        let rayleigh = (-2.0 * (1.0 - rng.uniform()).ln()).sqrt();
        let mut velocity = &drawn * thermal;
        velocity.scaled_add(rayleigh * thermal, flux_normal);
        velocity /= root;
        velocity /= MASS_VELOCITY_TO_EV.sqrt();

        let backward = self.half(
            plane,
            start,
            -&velocity,
            config,
            root,
            mass_weighted,
            checkpoints,
            Watch::Backward,
        )?;
        let forward = self.half(
            plane,
            start,
            velocity,
            config,
            root,
            mass_weighted,
            checkpoints,
            Watch::Forward,
        )?;
        let product = |state: Option<Endpoint>| {
            state.map(|endpoint| f64::from(endpoint == Endpoint::Product))
        };
        let values = (0..checkpoints.len())
            .map(|index| match config.counting {
                Counting::Final => {
                    Some(product(forward.states[index])? - product(backward.states[index])?)
                }
                Counting::FirstCrossing => {
                    if backward
                        .recrossed
                        .is_some_and(|recrossed| recrossed <= index)
                    {
                        return Some(0.0);
                    }
                    let past = backward.states[index]?;
                    let future = forward.states[index]?;
                    Some(f64::from(
                        past == Endpoint::Reactant && future == Endpoint::Product,
                    ))
                }
            })
            .collect();
        Ok(Shot {
            values,
            recrossed: backward.recrossed.is_some(),
            evaluations: backward.evaluations + forward.evaluations,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn half(
        &self,
        plane: &Hyperplane,
        start: &Array1<f64>,
        velocity: Array1<f64>,
        config: &TransmissionConfig,
        root: &Array1<f64>,
        mass_weighted: bool,
        checkpoints: &[usize],
        watch: Watch,
    ) -> Result<Half, SaddleError> {
        let position = coordinates_to_cartesian(start, root, mass_weighted);
        let mut engine = NveEngine::load(
            self.config.library_path.as_deref(),
            &self.system.config_bytes,
        )?;
        engine.initialize_geometry(
            &self.system,
            &position,
            plane,
            mass_weighted,
            config.thermal,
        )?;
        engine.set_positions(&position, self.system.box_matrix.as_ref())?;
        engine.set_velocities(
            velocity
                .as_slice()
                .ok_or_else(|| SaddleError::Shape("LAMMPS velocity must be contiguous".into()))?,
        )?;

        let mut states = vec![None; checkpoints.len()];
        let mut recrossed = None;
        let mut next = 0;
        let mut run: (Option<Endpoint>, usize) = (None, 0);
        let mut executed = 0;
        for step in 1..=config.steps {
            engine.step(config.time_step)?;
            executed += 1;
            let (position, _) = engine.state(plane.dim())?;
            let coordinates = cartesian_to_coordinates(&position, root, mass_weighted);
            if watch == Watch::Backward
                && recrossed.is_none()
                && plane.distance(coordinates.view()) > 0.0
            {
                recrossed = Some(next);
                if config.counting == Counting::FirstCrossing {
                    break;
                }
            }
            if step == checkpoints[next] {
                let state = side(plane, coordinates.view());
                states[next] = state;
                run = match (state, run.0) {
                    (Some(endpoint), Some(previous)) if endpoint == previous => (state, run.1 + 1),
                    (Some(_), _) => (state, 1),
                    (None, _) => (None, 0),
                };
                next += 1;
                if config.commit > 0 && run.1 >= config.commit {
                    for state in &mut states[next..] {
                        *state = run.0;
                    }
                    break;
                }
                if next == checkpoints.len() {
                    break;
                }
            }
        }
        Ok(Half {
            states,
            recrossed,
            // lammpc does not expose force-call accounting for NVE. One
            // velocity-Verlet step requires a force update.
            evaluations: executed + 1,
        })
    }
}

fn side(plane: &Hyperplane, coordinates: ndarray::ArrayView1<'_, f64>) -> Option<Endpoint> {
    let distance = plane.distance(coordinates);
    if distance < 0.0 {
        Some(Endpoint::Reactant)
    } else if distance > 0.0 {
        Some(Endpoint::Product)
    } else {
        None
    }
}

fn coordinates_to_cartesian(
    coordinates: &Array1<f64>,
    root: &Array1<f64>,
    mass_weighted: bool,
) -> Array1<f64> {
    if mass_weighted {
        coordinates / root
    } else {
        coordinates.clone()
    }
}

fn cartesian_to_coordinates(
    positions: &[f64],
    root: &Array1<f64>,
    mass_weighted: bool,
) -> Array1<f64> {
    let positions = Array1::from_vec(positions.to_vec());
    if mass_weighted {
        positions * root
    } else {
        positions
    }
}

fn checkpoints(config: &TransmissionConfig) -> Vec<usize> {
    let mut checkpoints: Vec<usize> = (1..=config.steps / config.check_every)
        .map(|index| index * config.check_every)
        .collect();
    if checkpoints.last() != Some(&config.steps) {
        checkpoints.push(config.steps);
    }
    checkpoints
}

fn check_config(config: &TransmissionConfig, dimension: usize) -> Result<(), SaddleError> {
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
    if let Some(masses) = &config.masses
        && (masses.len() != dimension || !masses.iter().all(|mass| mass.is_finite() && *mass > 0.0))
    {
        return Err(SaddleError::Invalid(
            "shooting needs one positive mass per coordinate".into(),
        ));
    }
    Ok(())
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
struct NveObservation {
    potential_energy_ev: f64,
    kinetic_energy_ev: f64,
}

#[repr(C)]
struct FeatureEntry {
    feature_id: *const c_char,
    kind: c_int,
    stub_applicable: c_int,
    embed_applicable: c_int,
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
type NveSetPositionsFn =
    unsafe extern "C" fn(*mut c_void, *const f64, usize, *const f64, usize) -> c_int;
type NveSetVelocitiesFn = unsafe extern "C" fn(*mut c_void, *const f64, usize) -> c_int;
type NveStepFn = unsafe extern "C" fn(*mut c_void, usize, f64) -> c_int;
type NveStateFn =
    unsafe extern "C" fn(*mut c_void, *mut f64, *mut f64, usize, *mut NveObservation) -> c_int;
type FeatureFindFn = unsafe extern "C" fn(*const c_char) -> *const FeatureEntry;

struct NveEngine {
    session: *mut c_void,
    destroy: SessionDestroyFn,
    finalize: FinalizeFn,
    last_error: LastErrorFn,
    md_initialize: MdInitializeFn,
    set_positions: NveSetPositionsFn,
    set_velocities: NveSetVelocitiesFn,
    step: NveStepFn,
    state: NveStateFn,
    _library: Library,
}

impl NveEngine {
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
            let available: AvailableFn = load_symbol(&library, b"lammpc_available\0")?;
            let abi_version: AbiVersionFn = load_symbol(&library, b"lammpc_abi_version\0")?;
            let last_error: LastErrorFn = load_symbol(&library, b"lammpc_last_error\0")?;
            let finalize: FinalizeFn = load_symbol(&library, b"lammpc_finalize\0")?;
            let feature_find: FeatureFindFn = load_symbol(&library, b"lammpc_feature_find\0")
                .map_err(|_| {
                    SaddleError::Surface(
                        "liblammpc lacks NVE shooting feature discovery; update lammpc".into(),
                    )
                })?;
            for feature in NVE_FEATURES {
                let entry = feature_find(feature.as_ptr().cast());
                if entry.is_null() || (*entry).kind != 2 || (*entry).embed_applicable == 0 {
                    let name = CStr::from_bytes_with_nul(feature)
                        .map_err(|_| SaddleError::Surface("invalid NVE feature name".into()))?
                        .to_string_lossy();
                    return Err(SaddleError::Surface(format!(
                        "liblammpc lacks required NVE shooting feature {name}"
                    )));
                }
            }
            let create: SessionCreateFn =
                load_symbol(&library, b"lammpc_session_create_from_config\0")?;
            let destroy: SessionDestroyFn = load_symbol(&library, b"lammpc_session_destroy\0")?;
            let md_initialize: MdInitializeFn =
                load_symbol(&library, b"lammpc_session_md_initialize\0")?;
            let set_positions: NveSetPositionsFn =
                load_symbol(&library, b"lammpc_session_nve_set_positions\0")?;
            let set_velocities: NveSetVelocitiesFn =
                load_symbol(&library, b"lammpc_session_nve_set_velocities\0")?;
            let step: NveStepFn = load_symbol(&library, b"lammpc_session_nve_step\0")?;
            let state: NveStateFn = load_symbol(&library, b"lammpc_session_nve_state\0")?;
            if available() == 0 {
                return Err(SaddleError::Surface(
                    "lammpc was built without LAMMPS".into(),
                ));
            }
            if abi_version() != LAMMPC_ABI_VERSION {
                return Err(SaddleError::Surface(format!(
                    "lammpc ABI {} does not match {LAMMPC_ABI_VERSION}",
                    abi_version()
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
                set_positions,
                set_velocities,
                step,
                state,
                _library: library,
            })
        }
    }

    fn initialize_geometry(
        &mut self,
        system: &LammpsSystem,
        positions: &Array1<f64>,
        plane: &Hyperplane,
        mass_weighted: bool,
        thermal: f64,
    ) -> Result<(), SaddleError> {
        let positions = positions
            .as_slice()
            .ok_or_else(|| SaddleError::Shape("LAMMPS positions must be contiguous".into()))?;
        let request = ProfileRequest {
            positions,
            atomic_numbers: &system.atomic_numbers,
            box_matrix: system.box_matrix.as_ref(),
            length_unit: "angstrom",
            energy_unit: "eV",
        };
        let force_input = encode_force_input(&request)
            .map_err(|error| SaddleError::Surface(error.to_string()))?;
        let config = MdConfig {
            time_step_ps: 1.0,
            thermal_energy_ev: thermal,
            friction_per_ps: 1.0,
            seed: 0,
            coordinates: if mass_weighted { 0 } else { 1 },
        };
        let status = unsafe {
            (self.md_initialize)(
                self.session,
                force_input.as_ptr().cast(),
                force_input.len(),
                plane.point().as_ptr(),
                plane.normal().as_ptr(),
                plane.dim(),
                &config,
            )
        };
        self.check(status)
    }

    fn set_positions(
        &mut self,
        positions: &Array1<f64>,
        cell: Option<&[f64; 9]>,
    ) -> Result<(), SaddleError> {
        let (cell, length) = match cell {
            Some(cell) => (cell.as_ptr(), cell.len()),
            None => (std::ptr::null(), 0),
        };
        let status = unsafe {
            (self.set_positions)(
                self.session,
                positions
                    .as_slice()
                    .ok_or_else(|| {
                        SaddleError::Shape("LAMMPS positions must be contiguous".into())
                    })?
                    .as_ptr(),
                positions.len(),
                cell,
                length,
            )
        };
        self.check(status)
    }

    fn set_velocities(&mut self, velocities: &[f64]) -> Result<(), SaddleError> {
        let status =
            unsafe { (self.set_velocities)(self.session, velocities.as_ptr(), velocities.len()) };
        self.check(status)
    }

    fn step(&mut self, time_step: f64) -> Result<(), SaddleError> {
        let status = unsafe { (self.step)(self.session, 1, time_step) };
        self.check(status)
    }

    fn state(&mut self, dimension: usize) -> Result<(Vec<f64>, NveObservation), SaddleError> {
        let mut positions = vec![0.0; dimension];
        let mut velocities = vec![0.0; dimension];
        let mut observation = NveObservation::default();
        let status = unsafe {
            (self.state)(
                self.session,
                positions.as_mut_ptr(),
                velocities.as_mut_ptr(),
                dimension,
                &mut observation,
            )
        };
        self.check(status)?;
        if !(observation.potential_energy_ev.is_finite()
            && observation.kinetic_energy_ev.is_finite())
        {
            return Err(SaddleError::NonFinite("LAMMPS NVE energy"));
        }
        Ok((positions, observation))
    }

    fn check(&self, status: c_int) -> Result<(), SaddleError> {
        if status == 0 {
            Ok(())
        } else {
            Err(SaddleError::Surface(last_error_message(self.last_error)))
        }
    }
}

impl Drop for NveEngine {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.session);
            self.session = std::ptr::null_mut();
            (self.finalize)();
        }
    }
}

const NVE_FEATURES: [&[u8]; 4] = [
    b"abi.lammpc_session_nve_set_positions\0",
    b"abi.lammpc_session_nve_set_velocities\0",
    b"abi.lammpc_session_nve_step\0",
    b"abi.lammpc_session_nve_state\0",
];

fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, SaddleError> {
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| SaddleError::Surface(error.to_string()))
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
    use ndarray::array;

    #[test]
    fn mass_weighted_coordinates_round_trip() {
        let coordinates = array![2.0, -6.0, 12.0];
        let root = array![2.0, 3.0, 4.0];
        let cartesian = coordinates_to_cartesian(&coordinates, &root, true);
        assert_eq!(cartesian, array![1.0, -2.0, 3.0]);
        assert_eq!(
            cartesian_to_coordinates(
                cartesian
                    .as_slice()
                    .expect("constructed array is contiguous"),
                &root,
                true,
            ),
            coordinates
        );
    }

    #[test]
    fn stub_library_has_the_complete_nve_feature_surface() {
        let Some(path) = std::env::var_os("LAMMPC_STUB_LIBRARY") else {
            return;
        };
        let error = match NveEngine::load(Some(Path::new(&path)), &[]) {
            Ok(_) => panic!("the standalone stub must report unavailable"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("built without LAMMPS"));
    }
}
