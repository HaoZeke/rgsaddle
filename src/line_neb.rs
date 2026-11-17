//! Fixed-energy line-integral NEB (LI-NEB).
//!
//! This is the microcanonical path optimization of Ásgeirsson,
//! Arnaldsson, and Jónsson, J. Chem. Phys. 148, 102334 (2018),
//! doi:10.1063/1.5007180. Coordinates are converted to
//! `q_i = sqrt(m_i) x_i`; the abbreviated action and all projections
//! are evaluated in that Euclidean, mass-scaled chart.
//!
//! Units are deliberately not prescribed. Positions, potential
//! energies, masses, the reduced Planck constant, spring constant,
//! endpoint restraint, and time step must form one consistent unit
//! system. In particular, `reduced_planck` has units of
//! `energy * time`, `spring_k` maps a mass-scaled displacement to an
//! action-gradient force, and `endpoint_kappa * energy` has those same
//! force units.
//!
//! A session performs one host-visible step at a time. It has no run
//! loop and no force budget; the host decides when to stop calling
//! [`LineNebSession::step`].

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::{Control, Method, Oracle, Solver};

use crate::{BandSurface, SaddleError};

/// All dimensional controls for a fixed-energy LI-NEB session.
#[derive(Clone, Debug)]
pub struct LineNebConfig {
    /// Uniform spring constant in the mass-scaled chart.
    pub spring_k: f64,
    /// Coefficient of the endpoint energy restraint.
    pub endpoint_kappa: f64,
    /// Convergence tolerance applied to force components.
    pub force_tol: f64,
    /// Fixed tunnelling energy `E`.
    pub energy: f64,
    /// One positive mass per physical coordinate (a 3N host repeats
    /// each atomic mass three times).
    pub masses: Array1<f64>,
    /// Fixed time step of the eOn global quick-min variant.
    pub time_step: f64,
    /// Reduced Planck constant in the host's unit system.
    pub reduced_planck: f64,
}

/// State of a host-stepped LI-NEB optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineNebStatus {
    Running,
    Converged,
}

/// Diagnostics from the configuration evaluated after one step.
#[derive(Clone, Debug)]
pub struct LineNebReport {
    pub status: LineNebStatus,
    /// Largest absolute component of the transverse action force over
    /// the interior images.
    pub max_interior_perpendicular: f64,
    /// Largest absolute component of the complete optimized endpoint
    /// force (projected spring plus energy restraint).
    pub max_endpoint_optimized: f64,
    pub action: f64,
    pub iteration: usize,
    /// Number of image rows passed to [`BandSurface::eval`] this step.
    pub surface_rows: usize,
}

#[derive(Clone, Debug)]
struct Evaluation {
    action: f64,
    optimization_force: Array2<f64>,
    max_interior_perpendicular: f64,
    max_endpoint_optimized: f64,
}

fn validate_inputs(
    positions: ArrayView2<'_, f64>,
    energies: ArrayView1<'_, f64>,
    masses: ArrayView1<'_, f64>,
    energy: f64,
    reduced_planck: f64,
) -> Result<(), SaddleError> {
    if positions.nrows() < 3 || positions.ncols() == 0 {
        return Err(SaddleError::Shape(format!(
            "LI-NEB needs at least three nonempty images; got {} x {}",
            positions.nrows(),
            positions.ncols()
        )));
    }
    if energies.len() != positions.nrows() {
        return Err(SaddleError::Shape(
            "LI-NEB energies must have one value per image".into(),
        ));
    }
    if masses.len() != positions.ncols() {
        return Err(SaddleError::Shape(
            "LI-NEB masses must have one value per coordinate".into(),
        ));
    }
    if !positions.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB positions"));
    }
    if !energies.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB energies"));
    }
    if !masses.iter().all(|m| m.is_finite() && *m > 0.0) {
        return Err(SaddleError::Invalid(
            "LI-NEB masses must be positive and finite".into(),
        ));
    }
    if !energy.is_finite() {
        return Err(SaddleError::NonFinite("LI-NEB target energy"));
    }
    if !(reduced_planck.is_finite() && reduced_planck > 0.0) {
        return Err(SaddleError::Invalid(
            "LI-NEB reduced Planck constant must be positive and finite".into(),
        ));
    }
    Ok(())
}

fn mass_scaled(positions: ArrayView2<'_, f64>, masses: ArrayView1<'_, f64>) -> Array2<f64> {
    let sqrt_masses = masses.mapv(f64::sqrt);
    &positions * &sqrt_masses
}

fn action_weights(
    energies: ArrayView1<'_, f64>,
    energy: f64,
    reduced_planck: f64,
) -> Result<Array1<f64>, SaddleError> {
    let mut xi = Array1::zeros(energies.len());
    for i in 1..energies.len() - 1 {
        let excess = energies[i] - energy;
        if excess.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(SaddleError::Invalid(format!(
                "LI-NEB interior image {i} must lie above the fixed energy"
            )));
        }
        xi[i] = (2.0 * excess).sqrt() / reduced_planck;
    }
    Ok(xi)
}

fn segments(q: ArrayView2<'_, f64>) -> Result<(Vec<Array1<f64>>, Array1<f64>), SaddleError> {
    let mut unit = Vec::with_capacity(q.nrows() - 1);
    let mut lengths = Array1::zeros(q.nrows() - 1);
    for i in 0..q.nrows() - 1 {
        let displacement = &q.row(i + 1) - &q.row(i);
        let length = displacement.dot(&displacement).sqrt();
        if !(length.is_finite() && length > 0.0) {
            return Err(SaddleError::Invalid(format!(
                "LI-NEB segment {i} must have positive finite length"
            )));
        }
        unit.push(displacement / length);
        lengths[i] = length;
    }
    Ok((unit, lengths))
}

/// Trapezoidal discretization of the fixed-energy abbreviated action.
///
/// `positions` are physical coordinates and `masses` has one entry per
/// coordinate. Endpoint energies may equal `energy`; every interior
/// energy must be strictly larger.
pub fn fixed_energy_action(
    positions: ArrayView2<'_, f64>,
    energies: ArrayView1<'_, f64>,
    masses: ArrayView1<'_, f64>,
    energy: f64,
    reduced_planck: f64,
) -> Result<f64, SaddleError> {
    validate_inputs(positions, energies, masses, energy, reduced_planck)?;
    let q = mass_scaled(positions, masses);
    let xi = action_weights(energies, energy, reduced_planck)?;
    let (_, lengths) = segments(q.view())?;
    let mut action = 0.0;
    for i in 0..lengths.len() {
        action += 0.5 * (xi[i] + xi[i + 1]) * lengths[i];
    }
    if !action.is_finite() {
        return Err(SaddleError::NonFinite("LI-NEB abbreviated action"));
    }
    Ok(action)
}

fn action_force_q(
    q: ArrayView2<'_, f64>,
    energies: ArrayView1<'_, f64>,
    force_q: ArrayView2<'_, f64>,
    energy: f64,
    reduced_planck: f64,
) -> Result<Array2<f64>, SaddleError> {
    if force_q.raw_dim() != q.raw_dim() {
        return Err(SaddleError::Shape(
            "LI-NEB force shape must match positions".into(),
        ));
    }
    if !force_q.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB physical force"));
    }
    let xi = action_weights(energies, energy, reduced_planck)?;
    let (unit, lengths) = segments(q)?;
    let mut force = Array2::zeros(q.raw_dim());
    for j in 1..q.nrows() - 1 {
        let physical = force_q.row(j).to_owned()
            * ((lengths[j - 1] + lengths[j]) / (xi[j] * reduced_planck * reduced_planck));
        let curvature = &unit[j] * (xi[j + 1] + xi[j]) - &unit[j - 1] * (xi[j] + xi[j - 1]);
        force.row_mut(j).assign(&(0.5 * (physical + curvature)));
    }
    if !force.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB action force"));
    }
    Ok(force)
}

/// Negative gradient of the discretized action with respect to the
/// physical coordinates.
///
/// The endpoint rows are zero because LI-NEB replaces their action
/// gradient by the endpoint spring and iso-energy restraint.
pub fn fixed_energy_action_force(
    positions: ArrayView2<'_, f64>,
    energies: ArrayView1<'_, f64>,
    gradients: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    energy: f64,
    reduced_planck: f64,
) -> Result<Array2<f64>, SaddleError> {
    validate_inputs(positions, energies, masses, energy, reduced_planck)?;
    if gradients.raw_dim() != positions.raw_dim() {
        return Err(SaddleError::Shape(
            "LI-NEB gradient shape must match positions".into(),
        ));
    }
    if !gradients.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB gradients"));
    }
    let sqrt_masses = masses.mapv(f64::sqrt);
    let q = &positions * &sqrt_masses;
    let force_q = -&gradients / &sqrt_masses;
    let force = action_force_q(q.view(), energies, force_q.view(), energy, reduced_planck)?;
    Ok(force * &sqrt_masses)
}

fn max_abs(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0_f64, |maximum, value| maximum.max(value.abs()))
}

fn improved_tangent(
    previous: ArrayView1<'_, f64>,
    current: ArrayView1<'_, f64>,
    next: ArrayView1<'_, f64>,
    previous_energy: f64,
    current_energy: f64,
    next_energy: f64,
) -> Result<Array1<f64>, SaddleError> {
    let backward = &current - &previous;
    let forward = &next - &current;
    let mut tangent = if next_energy > current_energy && current_energy > previous_energy {
        forward
    } else if next_energy < current_energy && current_energy < previous_energy {
        backward
    } else {
        let forward_difference = (next_energy - current_energy).abs();
        let backward_difference = (previous_energy - current_energy).abs();
        let large = forward_difference.max(backward_difference);
        let small = forward_difference.min(backward_difference);
        match next_energy.total_cmp(&previous_energy) {
            std::cmp::Ordering::Greater => forward * large + backward * small,
            std::cmp::Ordering::Less => forward * small + backward * large,
            std::cmp::Ordering::Equal => (forward + backward) * large,
        }
    };
    let norm = tangent.dot(&tangent).sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(SaddleError::Invalid(
            "LI-NEB improved tangent has no direction".into(),
        ));
    }
    tangent /= norm;
    Ok(tangent)
}

fn evaluate(
    config: &LineNebConfig,
    surface: &impl BandSurface,
    q: ArrayView2<'_, f64>,
) -> Result<Evaluation, SaddleError> {
    let sqrt_masses = config.masses.mapv(f64::sqrt);
    let positions = &q / &sqrt_masses;
    let mut energies = Array1::zeros(q.nrows());
    let mut gradients = Array2::zeros(q.raw_dim());
    surface.eval(positions.view(), &mut energies, &mut gradients)?;
    if !energies.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB surface energies"));
    }
    if !gradients.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB surface gradients"));
    }
    let action = fixed_energy_action(
        positions.view(),
        energies.view(),
        config.masses.view(),
        config.energy,
        config.reduced_planck,
    )?;
    let force_q = -&gradients / &sqrt_masses;
    let action_force = action_force_q(
        q,
        energies.view(),
        force_q.view(),
        config.energy,
        config.reduced_planck,
    )?;
    let (_, lengths) = segments(q)?;
    let mut optimization_force = Array2::zeros(q.raw_dim());
    let mut max_interior_perpendicular = 0.0_f64;

    for j in 1..q.nrows() - 1 {
        let tangent = improved_tangent(
            q.row(j - 1),
            q.row(j),
            q.row(j + 1),
            energies[j - 1],
            energies[j],
            energies[j + 1],
        )?;
        let raw = action_force.row(j);
        let perpendicular = &raw - raw.dot(&tangent) * &tangent;
        max_interior_perpendicular =
            max_interior_perpendicular.max(max_abs(perpendicular.iter().copied()));
        let spring = config.spring_k * (lengths[j] - lengths[j - 1]) * tangent;
        optimization_force
            .row_mut(j)
            .assign(&(perpendicular + spring));
    }

    for (endpoint, neighbour) in [(0, 1), (q.nrows() - 1, q.nrows() - 2)] {
        let fq = force_q.row(endpoint).to_owned();
        let force_norm = fq.dot(&fq).sqrt();
        if !(force_norm.is_finite() && force_norm > 0.0) {
            return Err(SaddleError::Invalid(format!(
                "LI-NEB endpoint {endpoint} needs a nonzero physical force"
            )));
        }
        let force_hat = fq / force_norm;
        let spring = config.spring_k * (&q.row(neighbour) - &q.row(endpoint));
        let spring_perpendicular = &spring - spring.dot(&force_hat) * &force_hat;
        let restraint = config.endpoint_kappa * (energies[endpoint] - config.energy) * force_hat;
        optimization_force
            .row_mut(endpoint)
            .assign(&(spring_perpendicular + restraint));
    }

    if !optimization_force.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("LI-NEB optimization force"));
    }
    let last = q.nrows() - 1;
    let max_endpoint_optimized = max_abs(
        optimization_force
            .row(0)
            .iter()
            .chain(optimization_force.row(last).iter())
            .copied(),
    );
    Ok(Evaluation {
        action,
        optimization_force,
        max_interior_perpendicular,
        max_endpoint_optimized,
    })
}

/// Natural cubic interpolation of vector-valued data on a strict knot grid.
///
/// LI-NEB consumers use this for coordinates, parallel forces, and Hessian
/// elements. Sampling returns the value and its first two knot derivatives.
#[derive(Clone, Debug)]
pub struct NaturalCubic {
    knots: Vec<f64>,
    values: Vec<Vec<f64>>,
    second: Vec<Vec<f64>>,
}

impl NaturalCubic {
    pub fn new(knots: Vec<f64>, values: Vec<Vec<f64>>) -> Result<Self, SaddleError> {
        if knots.len() != values.len() || knots.len() < 2 {
            return Err(SaddleError::Shape(
                "cubic interpolation needs at least two equally counted knots and rows".into(),
            ));
        }
        let width = values[0].len();
        if width == 0 || values.iter().any(|row| row.len() != width) {
            return Err(SaddleError::Shape(
                "cubic interpolation rows must have one common nonzero width".into(),
            ));
        }
        let count = knots.len();
        let mut second = vec![vec![0.0; width]; count];
        if count > 2 {
            for component in 0..width {
                let interior = count - 2;
                let mut diagonal = vec![0.0; interior];
                let mut upper = vec![0.0; interior];
                let mut rhs = vec![0.0; interior];
                for row in 0..interior {
                    let i = row + 1;
                    let left = knots[i] - knots[i - 1];
                    let right = knots[i + 1] - knots[i];
                    if !(left.is_finite() && right.is_finite() && left > 0.0 && right > 0.0) {
                        return Err(SaddleError::Invalid(
                            "cubic interpolation knots must be finite and strictly increasing"
                                .into(),
                        ));
                    }
                    diagonal[row] = 2.0 * (left + right);
                    if row + 1 < interior {
                        upper[row] = right;
                    }
                    rhs[row] = 6.0
                        * ((values[i + 1][component] - values[i][component]) / right
                            - (values[i][component] - values[i - 1][component]) / left);
                    if row > 0 {
                        let factor = left / diagonal[row - 1];
                        diagonal[row] -= factor * upper[row - 1];
                        rhs[row] -= factor * rhs[row - 1];
                    }
                }
                for row in (0..interior).rev() {
                    let next = if row + 1 < interior {
                        upper[row] * second[row + 2][component]
                    } else {
                        0.0
                    };
                    second[row + 1][component] = (rhs[row] - next) / diagonal[row];
                }
            }
        }
        if !values.iter().flatten().all(|value| value.is_finite())
            || !second.iter().flatten().all(|value| value.is_finite())
        {
            return Err(SaddleError::NonFinite("cubic interpolation"));
        }
        Ok(Self {
            knots,
            values,
            second,
        })
    }

    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    pub fn sample(&self, x: f64) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), SaddleError> {
        if !x.is_finite() || x < self.knots[0] || x > self.knots[self.knots.len() - 1] {
            return Err(SaddleError::Invalid(
                "cubic interpolation sample lies outside its knot interval".into(),
            ));
        }
        let i = match self.knots.binary_search_by(|value| value.total_cmp(&x)) {
            Ok(index) => index.min(self.knots.len() - 2),
            Err(index) => index.saturating_sub(1).min(self.knots.len() - 2),
        };
        let h = self.knots[i + 1] - self.knots[i];
        let a = (self.knots[i + 1] - x) / h;
        let b = (x - self.knots[i]) / h;
        let width = self.values[i].len();
        let mut value = vec![0.0; width];
        let mut first = vec![0.0; width];
        let mut second = vec![0.0; width];
        for component in 0..width {
            let y0 = self.values[i][component];
            let y1 = self.values[i + 1][component];
            let m0 = self.second[i][component];
            let m1 = self.second[i + 1][component];
            value[component] =
                a * y0 + b * y1 + ((a * a * a - a) * m0 + (b * b * b - b) * m1) * h * h / 6.0;
            first[component] =
                (y1 - y0) / h + ((-3.0 * a * a + 1.0) * m0 + (3.0 * b * b - 1.0) * m1) * h / 6.0;
            second[component] = a * m0 + b * m1;
        }
        Ok((value, first, second))
    }
}

/// Fixed-image-count LI-NEB session using rgmin's global eOn
/// quick-min velocity projection with a fixed time step.
pub struct LineNebSession {
    config: LineNebConfig,
    q: Array2<f64>,
    solver: Solver,
    iteration: usize,
}

impl LineNebSession {
    pub fn new(config: LineNebConfig, positions: Array2<f64>) -> Result<Self, SaddleError> {
        let energies = Array1::zeros(positions.nrows());
        validate_inputs(
            positions.view(),
            energies.view(),
            config.masses.view(),
            config.energy,
            config.reduced_planck,
        )?;
        for (name, value) in [
            ("spring constant", config.spring_k),
            ("endpoint restraint", config.endpoint_kappa),
            ("force tolerance", config.force_tol),
            ("time step", config.time_step),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(SaddleError::Invalid(format!(
                    "LI-NEB {name} must be positive and finite"
                )));
            }
        }
        let q = mass_scaled(positions.view(), config.masses.view());
        segments(q.view())?;
        let dim = q.len();
        let mut solver = Solver::new(
            Method::QuickMin,
            Control {
                maxiter: usize::MAX,
                gtol: 0.0,
                istep: config.time_step,
                maxmove: None,
                ftol_rel: None,
                noise_floor: None,
            },
            dim,
        );
        solver.set_quickmin_options(rgmin::QuickMinOptions::eon());
        solver.set_atom_maxmove(0.0);
        Ok(Self {
            config,
            q,
            solver,
            iteration: 0,
        })
    }

    /// Current physical positions, including the movable endpoints.
    pub fn positions(&self) -> Array2<f64> {
        &self.q / &self.config.masses.mapv(f64::sqrt)
    }

    pub fn config(&self) -> &LineNebConfig {
        &self.config
    }

    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Evaluate through [`BandSurface`] and take one global projected
    /// quick-min step. The host owns every call and its force budget.
    pub fn step<S: BandSurface>(&mut self, surface: &S) -> Result<LineNebReport, SaddleError> {
        let shape = self.q.raw_dim();
        let config = &self.config;
        let error = Mutex::new(None);
        let latest = Mutex::new(None);
        let rows = AtomicUsize::new(0);
        let oracle = Oracle::unbounded(self.q.len(), |flat: ArrayView1<'_, f64>| {
            let q = match flat.to_owned().into_shape_with_order(shape) {
                Ok(q) => q,
                Err(err) => {
                    *error.lock().unwrap() = Some(SaddleError::Shape(err.to_string()));
                    return (f64::INFINITY, Array1::zeros(flat.len()));
                }
            };
            rows.fetch_add(q.nrows(), Ordering::Relaxed);
            match evaluate(config, surface, q.view()) {
                Ok(evaluation) => {
                    let gradient =
                        -Array1::from_iter(evaluation.optimization_force.iter().copied());
                    let action = evaluation.action;
                    *latest.lock().unwrap() = Some(evaluation);
                    (action, gradient)
                }
                Err(err) => {
                    *error.lock().unwrap() = Some(err);
                    (f64::INFINITY, Array1::zeros(flat.len()))
                }
            }
        });
        let mut flat = Array1::from_iter(self.q.iter().copied());
        let stepped = self.solver.step(&oracle, &mut flat);
        drop(oracle);
        if let Some(err) = error.into_inner().unwrap() {
            self.solver.forget_evaluation();
            return Err(err);
        }
        stepped.map_err(|err| SaddleError::Solver(err.to_string()))?;
        self.q = flat
            .into_shape_with_order(shape)
            .map_err(|err| SaddleError::Shape(err.to_string()))?;
        if !self.q.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("LI-NEB step"));
        }
        self.iteration = self
            .iteration
            .checked_add(1)
            .ok_or_else(|| SaddleError::Solver("LI-NEB iteration counter overflow".into()))?;
        let evaluation = latest
            .into_inner()
            .unwrap()
            .ok_or_else(|| SaddleError::Solver("LI-NEB step produced no evaluation".into()))?;
        let status = if evaluation.max_interior_perpendicular <= self.config.force_tol
            && evaluation.max_endpoint_optimized <= self.config.force_tol
        {
            LineNebStatus::Converged
        } else {
            LineNebStatus::Running
        };
        Ok(LineNebReport {
            status,
            max_interior_perpendicular: evaluation.max_interior_perpendicular,
            max_endpoint_optimized: evaluation.max_endpoint_optimized,
            action: evaluation.action,
            iteration: self.iteration,
            surface_rows: rows.load(Ordering::Relaxed),
        })
    }
}
