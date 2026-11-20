//! One-shot C ABI for fixed-energy LI-NEB and harmonic quantum TST rates.

use std::ffi::c_void;
use std::slice;

use ndarray::{Array1, Array2};

use super::{
    CSurface, RGSADDLE_INVALID_PARAMETER, RGSADDLE_NULL_REPORT, RGSADDLE_NULL_SURFACE, RGSADDLE_OK,
    RGSADDLE_SHAPE, RGSADDLE_STALLED, RgsaddleSurfaceFn, status_of,
};
use crate::lineb::{LinebConfig, LinebSession, LinebStatus, Masses, hqtst_rate};

/// Fixed-energy LI-NEB controls for [`rgsaddle_lineb_solve`] (ABI minor 46).
///
/// Field order matches `rgsaddle_lineb_config_t` in `include/rgsaddle.h`.
/// Start from [`rgsaddle_lineb_config_default`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RgsaddleLinebConfig {
    /// Stop when every nudged force's norm is at most this.
    pub opt_tol: f64,
    /// Stop when every interior perpendicular action gradient's norm is
    /// at most this.
    pub perp_tol: f64,
    /// Spring constant `k` in the parallel spacing force.
    pub spring: f64,
    /// `κ |f|` for the endpoint equipotential force.
    pub endpoint_scale: f64,
    /// Initial FIRE / quick-min time step.
    pub time_step: f64,
    /// Largest step; zero leaves the step uncapped.
    pub max_move: f64,
    /// Turning-surface regularization width as a fraction of the highest
    /// starting image above `E`.
    pub turning_width: f64,
    /// `ℏ` in the temperature `ℏ / (k_B τ)`.
    pub hbar: f64,
    /// Boltzmann constant in the same temperature.
    pub boltzmann: f64,
    /// Simpson pairs on each chord when the period is integrated.
    pub quadrature_pairs: i64,
}

/// Path, temperature and rate summary from [`rgsaddle_lineb_solve`]
/// (ABI minor 46).
///
/// Field order matches `rgsaddle_lineb_result_t` in `include/rgsaddle.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RgsaddleLinebResult {
    /// Spline action `W`.
    pub action: f64,
    /// Half period from the action's energy derivative.
    pub half_period: f64,
    /// Half period from constrained dynamics along the path.
    pub half_period_dynamics: f64,
    /// Instanton temperature from the quadrature half period.
    pub temperature: f64,
    /// Instanton temperature from the dynamics half period.
    pub temperature_dynamics: f64,
    /// `ln(k Z_r)`, or NaN when `beads == 0`.
    pub log_rate_over_reactant: f64,
    /// Optimizer steps completed.
    pub iterations: i64,
    /// Surface evaluations inside optimizer steps.
    pub optimizer_calls: i64,
    /// Surface evaluations in the path quadrature and time table.
    pub measure_calls: i64,
    /// Surface evaluations in the half-period dynamics.
    pub dynamics_calls: i64,
    /// Surface evaluations in the rate, or 0 when `beads == 0`.
    pub rate_calls: i64,
}

fn config_from_c(cfg: &RgsaddleLinebConfig) -> Result<LinebConfig, i32> {
    if cfg.quadrature_pairs < 1 {
        return Err(RGSADDLE_INVALID_PARAMETER);
    }
    let Some(pairs) = usize::try_from(cfg.quadrature_pairs).ok() else {
        return Err(RGSADDLE_INVALID_PARAMETER);
    };
    Ok(LinebConfig::new(cfg.opt_tol, cfg.perp_tol)
        .with_spring(cfg.spring)
        .with_endpoint_scale(cfg.endpoint_scale)
        .with_time_step(cfg.time_step)
        .with_max_move(cfg.max_move)
        .with_turning_width(cfg.turning_width)
        .with_hbar(cfg.hbar)
        .with_boltzmann(cfg.boltzmann)
        .with_quadrature_pairs(pairs))
}

fn masses_from_c(atom_masses: &[f64], molecular: i32) -> Result<Masses, i32> {
    match molecular {
        0 => {
            let mut expanded = Array1::zeros(3 * atom_masses.len());
            for (atom, &mass) in atom_masses.iter().enumerate() {
                for axis in 0..3 {
                    expanded[3 * atom + axis] = mass;
                }
            }
            Masses::coordinates(expanded.view()).map_err(|err| status_of(&err))
        }
        1 => Masses::molecule(atom_masses).map_err(|err| status_of(&err)),
        _ => Err(RGSADDLE_INVALID_PARAMETER),
    }
}

fn as_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// Fill `out` with paper defaults for the given force tolerances
/// (ABI minor 46).
///
/// # Safety
/// `out` must be a valid writable pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_lineb_config_default(
    opt_tol: f64,
    perp_tol: f64,
    out: *mut RgsaddleLinebConfig,
) -> i32 {
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let cfg = LinebConfig::new(opt_tol, perp_tol);
    // SAFETY: non-null writable config.
    unsafe {
        *out = RgsaddleLinebConfig {
            opt_tol: cfg.opt_tol,
            perp_tol: cfg.perp_tol,
            spring: cfg.spring,
            endpoint_scale: cfg.endpoint_scale,
            time_step: cfg.time_step,
            max_move: cfg.max_move,
            turning_width: cfg.turning_width,
            hbar: cfg.hbar,
            boltzmann: cfg.boltzmann,
            quadrature_pairs: as_i64(cfg.quadrature_pairs),
        };
    }
    RGSADDLE_OK
}

/// Optimize a fixed-energy tunneling path and optionally its HQ-TST rate
/// (ABI minor 46).
///
/// Positions are contiguous image, atom, xyz (`n_images * n_atoms * 3`).
/// `atom_masses` holds `n_atoms` positive masses: when `molecular` is 0
/// each mass expands to the three Cartesian coordinates with no rigid
/// alignment; when `molecular` is 1, quaternion alignment is used.
/// `beads == 0` returns the path only and writes NaN into
/// `log_rate_over_reactant`. An unconverged path returns the existing
/// non-success status (`RGSADDLE_STALLED`) after writing the last
/// images into `optimized_positions`.
///
/// # Safety
/// `cfg`, `positions`, `atom_masses`, `optimized_positions` and `out`
/// must be valid for the stated sizes when non-null. `surface` is
/// called with `user`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_lineb_solve(
    cfg: *const RgsaddleLinebConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
    atom_masses: *const f64,
    molecular: i32,
    energy: f64,
    max_steps: i64,
    beads: i64,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    optimized_positions: *mut f64,
    out: *mut RgsaddleLinebResult,
) -> i32 {
    if cfg.is_null()
        || positions.is_null()
        || atom_masses.is_null()
        || optimized_positions.is_null()
    {
        return RGSADDLE_INVALID_PARAMETER;
    }
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    let Some(f) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    if n_images < 3 || n_atoms < 1 {
        return RGSADDLE_SHAPE;
    }
    if max_steps < 0 || beads < 0 || beads == 1 {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let Ok(n_images_usize) = usize::try_from(n_images) else {
        return RGSADDLE_SHAPE;
    };
    let Ok(n_atoms_usize) = usize::try_from(n_atoms) else {
        return RGSADDLE_SHAPE;
    };
    let Ok(max_steps_usize) = usize::try_from(max_steps) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let Ok(beads_usize) = usize::try_from(beads) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let dof = match n_atoms_usize.checked_mul(3) {
        Some(dof) => dof,
        None => return RGSADDLE_SHAPE,
    };
    let n_values = match n_images_usize.checked_mul(dof) {
        Some(n) => n,
        None => return RGSADDLE_SHAPE,
    };

    // SAFETY: pointers non-null; lengths checked above.
    let cfg = unsafe { &*cfg };
    let positions = unsafe { slice::from_raw_parts(positions, n_values) };
    let atom_masses = unsafe { slice::from_raw_parts(atom_masses, n_atoms_usize) };
    let optimized = unsafe { slice::from_raw_parts_mut(optimized_positions, n_values) };
    let out = unsafe { &mut *out };

    *out = RgsaddleLinebResult {
        action: f64::NAN,
        half_period: f64::NAN,
        half_period_dynamics: f64::NAN,
        temperature: f64::NAN,
        temperature_dynamics: f64::NAN,
        log_rate_over_reactant: f64::NAN,
        iterations: 0,
        optimizer_calls: 0,
        measure_calls: 0,
        dynamics_calls: 0,
        rate_calls: 0,
    };

    let config = match config_from_c(cfg) {
        Ok(config) => config,
        Err(status) => return status,
    };
    let masses = match masses_from_c(atom_masses, molecular) {
        Ok(masses) => masses,
        Err(status) => return status,
    };
    let cartesian = match Array2::from_shape_vec((n_images_usize, dof), positions.to_vec()) {
        Ok(cartesian) => cartesian,
        Err(_) => return RGSADDLE_SHAPE,
    };
    let cs = CSurface {
        f,
        user,
        n_atoms,
        n_images,
        per_image: false,
        climbing_request: false,
        rows: Default::default(),
    };
    let mut session = match LinebSession::new(config, masses, cartesian, energy) {
        Ok(session) => session,
        Err(err) => return status_of(&err),
    };
    let report = match session.run(&cs, max_steps_usize) {
        Ok(report) => report,
        Err(err) => return status_of(&err),
    };
    out.iterations = as_i64(report.iteration);
    out.optimizer_calls = as_i64(report.evaluations);
    let cartesian = match session.cartesian() {
        Ok(cartesian) => cartesian,
        Err(err) => return status_of(&err),
    };
    for (dst, src) in optimized.iter_mut().zip(cartesian.iter()) {
        *dst = *src;
    }
    if report.status != LinebStatus::Converged {
        return RGSADDLE_STALLED;
    }
    let path = match session.path(&cs) {
        Ok(path) => path,
        Err(err) => return status_of(&err),
    };
    for (dst, src) in optimized.iter_mut().zip(path.cartesian.iter()) {
        *dst = *src;
    }
    out.action = path.action;
    out.half_period = path.half_period;
    out.half_period_dynamics = path.half_period_dynamics;
    out.temperature = path.temperature;
    out.temperature_dynamics = path.temperature_dynamics;
    out.optimizer_calls = as_i64(path.calls.optimizer);
    out.measure_calls = as_i64(path.calls.measure);
    out.dynamics_calls = as_i64(path.calls.dynamics);
    if beads_usize == 0 {
        out.log_rate_over_reactant = f64::NAN;
        out.rate_calls = 0;
        return RGSADDLE_OK;
    }
    match hqtst_rate(&cs, &path, beads_usize) {
        Ok(rate) => {
            out.log_rate_over_reactant = rate.log_rate_over_reactant;
            out.rate_calls = as_i64(rate.evaluations);
            RGSADDLE_OK
        }
        Err(err) => status_of(&err),
    }
}
