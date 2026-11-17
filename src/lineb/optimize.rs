//! Fixed-energy LI-NEB.
//!
//! One rgmin step on the nudged action gradient. The host owns the loop;
//! [`LinebSession::run`] repeats `step`. FIRE, L-BFGS (accepted on the
//! step, because the nudged force is not the gradient of the action),
//! Barzilai–Borwein, quick-min and ODE12r are the force-driven methods.

use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::{Accept, Control, Method, Oracle, Solver};

use super::action;
use super::mass::Masses;
use super::path::{self, Periods};
use crate::band::check_force_driven;
use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// Controls for a fixed-energy optimal tunneling path.
///
/// The spring `k` and the endpoint scale are the products in appendix A
/// of doi:10.1021/acs.jctc.5c00673 when the optimizer time step is one:
/// `k Δt² = 0.1` and `κ |∇V| Δt² = 0.2`. The endpoint force is
/// `κ (V − E) f̂` with `κ = endpoint_scale / |f|`.
///
/// `opt_tol` and `perp_tol` are required. They are a force scale of the
/// surface being searched (its gradient, its curvature, or the noise in
/// an evaluation), and this method does not substitute one.
///
/// Non-exhaustive: start from [`LinebConfig::new`] and chain setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LinebConfig {
    /// Spring constant `k` in the parallel spacing force.
    pub spring: f64,
    /// `κ |f|` for the endpoint equipotential force.
    pub endpoint_scale: f64,
    /// Stop when every nudged force's norm is at most this.
    pub opt_tol: f64,
    /// Stop when every interior perpendicular action gradient's norm is
    /// at most this.
    pub perp_tol: f64,
    /// Initial FIRE / quick-min time step (rgmin `Control::istep`).
    pub time_step: f64,
    /// Largest per-atom displacement when a coordinate block is three
    /// per atom. A non-positive value leaves the step uncapped.
    pub max_move: f64,
    /// `ℏ` in the temperature `ℏ / (k_B τ)`. It does not move the path.
    pub hbar: f64,
    /// Boltzmann constant in the same temperature.
    pub boltzmann: f64,
    /// Force-driven stepper. FIRE 2.0 by default.
    pub method: Method,
    /// Simpson pairs on each chord when the period is integrated.
    pub quadrature_pairs: usize,
}

with_setters!(LinebConfig {
    with_spring => spring: val f64,
    with_endpoint_scale => endpoint_scale: val f64,
    with_opt_tol => opt_tol: val f64,
    with_perp_tol => perp_tol: val f64,
    with_time_step => time_step: val f64,
    with_max_move => max_move: val f64,
    with_hbar => hbar: val f64,
    with_boltzmann => boltzmann: val f64,
    with_method => method: val Method,
    with_quadrature_pairs => quadrature_pairs: val usize,
});

impl LinebConfig {
    /// Paper spring and endpoint scale, unit `ℏ` and `k_B`, FIRE 2.0.
    ///
    /// `opt_tol` stops the nudged force and `perp_tol` stops the
    /// perpendicular action gradient. Both are absolute force norms in
    /// the mass-weighted coordinates.
    pub fn new(opt_tol: f64, perp_tol: f64) -> Self {
        Self {
            spring: 0.1,
            endpoint_scale: 0.2,
            opt_tol,
            perp_tol,
            time_step: 1.0,
            max_move: 0.2,
            hbar: 1.0,
            boltzmann: 1.0,
            method: Method::fire(rgmin::FireKind::V2),
            quadrature_pairs: 4,
        }
    }
}

impl LinebConfig {
    fn check(&self) -> Result<(), SaddleError> {
        if !(self.spring.is_finite() && self.spring > 0.0) {
            return Err(SaddleError::Invalid(
                "the tunneling spring must be finite and positive".into(),
            ));
        }
        if !(self.endpoint_scale.is_finite() && self.endpoint_scale > 0.0) {
            return Err(SaddleError::Invalid(
                "the endpoint scale must be finite and positive".into(),
            ));
        }
        if !(self.opt_tol.is_finite()
            && self.opt_tol >= 0.0
            && self.perp_tol.is_finite()
            && self.perp_tol >= 0.0)
        {
            return Err(SaddleError::Invalid(
                "tunneling force tolerances must be finite and non-negative".into(),
            ));
        }
        if !(self.time_step.is_finite() && self.time_step > 0.0) {
            return Err(SaddleError::Invalid(
                "the tunneling time step must be finite and positive".into(),
            ));
        }
        if !(self.max_move.is_finite() && self.max_move >= 0.0) {
            return Err(SaddleError::Invalid(
                "the tunneling per-atom cap must be finite and non-negative".into(),
            ));
        }
        if !(self.hbar.is_finite()
            && self.hbar > 0.0
            && self.boltzmann.is_finite()
            && self.boltzmann > 0.0)
        {
            return Err(SaddleError::Invalid(
                "hbar and the Boltzmann constant must be finite and positive".into(),
            ));
        }
        if self.quadrature_pairs < 1 {
            return Err(SaddleError::Invalid(
                "the period quadrature needs at least one Simpson pair per chord".into(),
            ));
        }
        check_force_driven(&self.method)
    }
}

/// Whether a fixed-energy path has met its force tolerances.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinebStatus {
    /// The perpendicular action gradient and the nudged force are within
    /// their tolerances.
    Converged,
    /// Another step is available.
    Running,
}

/// One step of a fixed-energy search.
#[derive(Clone, Debug)]
pub struct LinebReport {
    /// The status.
    pub status: LinebStatus,
    /// Discrete abbreviated action `Σ (σ_j + σ_{j+1}) |Δq| / 2`.
    pub action: f64,
    /// Largest norm of an interior perpendicular action gradient.
    pub perp_max: f64,
    /// Largest norm of a nudged force, endpoints included.
    pub opt_max: f64,
    /// Steps completed, counting this one.
    pub iteration: usize,
    /// Surface evaluations so far.
    pub evaluations: usize,
}

/// A converged (or inspected) optimal tunneling path and both periods.
#[derive(Clone, Debug)]
pub struct LinebPath {
    /// Cartesian images of the aligned mass-weighted path.
    pub cartesian: Array2<f64>,
    /// Tunneling energy.
    pub energy: f64,
    /// Discrete action the optimizer minimized.
    pub discrete_action: f64,
    /// Spline action `∫ √(2(V−E)) ds`.
    pub action: f64,
    /// `−∂W/∂E` along the spline, one way.
    pub half_period: f64,
    /// Half-period constrained dynamics along the same spline.
    pub half_period_dynamics: f64,
    /// `ℏ / (k_B τ)` from [`Self::half_period`].
    pub temperature: f64,
    /// `ℏ / (k_B τ)` from [`Self::half_period_dynamics`].
    pub temperature_dynamics: f64,
    pub(crate) weighted: Array2<f64>,
    pub(crate) masses: Masses,
    pub(crate) quadrature_pairs: usize,
    pub(crate) hbar: f64,
    pub(crate) boltzmann: f64,
}

/// Fixed-energy line-integral nudged elastic band.
///
/// Images are stored in mass-weighted coordinates. Every image moves,
/// including the endpoints, which are held on `V = E`.
pub struct LinebSession {
    config: LinebConfig,
    masses: Masses,
    energy: f64,
    weighted: Array2<f64>,
    solver: Solver,
    /// Per-atom cap installed at construction, restored after a
    /// rejected step has been bisected.
    atom_cap: Option<f64>,
    /// A rejected trial tightened the step length. The next step
    /// releases that length.
    cap_held: bool,
    evaluations: AtomicUsize,
    iteration: usize,
}

impl LinebSession {
    /// `cartesian` has one image per row, in the same coordinates the
    /// surface evaluates. Every image must lie strictly above `energy`.
    pub fn new(
        config: LinebConfig,
        masses: Masses,
        cartesian: Array2<f64>,
        energy: f64,
    ) -> Result<Self, SaddleError> {
        config.check()?;
        if !(energy.is_finite()) {
            return Err(SaddleError::Invalid(
                "the tunneling energy must be finite".into(),
            ));
        }
        if cartesian.nrows() < 3 || cartesian.ncols() != masses.dof() {
            return Err(SaddleError::Shape(
                "LI-NEB needs at least three images and one mass per coordinate".into(),
            ));
        }
        if cartesian.iter().any(|value| !value.is_finite()) {
            return Err(SaddleError::NonFinite("initial tunneling path"));
        }
        let mut weighted = Array2::zeros(cartesian.dim());
        for i in 0..cartesian.nrows() {
            weighted
                .row_mut(i)
                .assign(&masses.to_weighted(cartesian.row(i))?);
        }
        let dof = cartesian.nrows() * cartesian.ncols();
        let control = Control::default()
            .with_maxiter(usize::MAX)
            .with_gtol(0.0)
            .with_istep(config.time_step)
            .with_maxmove(None);
        let mut solver = Solver::new(config.method.clone(), control, dof);
        let atom_cap = if masses.dof().is_multiple_of(3) && config.max_move > 0.0 {
            solver.set_atom_maxmove(config.max_move);
            Some(config.max_move)
        } else {
            None
        };
        if matches!(config.method, Method::Lbfgs { .. }) {
            solver.set_accept(Accept::Step);
            solver.set_lbfgs_neb_guards(true);
        }
        Ok(Self {
            config,
            masses,
            energy,
            weighted,
            solver,
            atom_cap,
            cap_held: false,
            evaluations: AtomicUsize::new(0),
            iteration: 0,
        })
    }

    /// Mass-weighted images the solver holds.
    pub fn weighted(&self) -> ArrayView2<'_, f64> {
        self.weighted.view()
    }

    /// Cartesian images corresponding to [`Self::weighted`].
    pub fn cartesian(&self) -> Result<Array2<f64>, SaddleError> {
        let mut out = Array2::zeros(self.weighted.dim());
        for i in 0..self.weighted.nrows() {
            out.row_mut(i)
                .assign(&self.masses.to_cartesian(self.weighted.row(i))?);
        }
        Ok(out)
    }

    /// One optimizer step.
    ///
    /// A trial that leaves the region `V > E` is not a point of the
    /// action. The stepper's own shortening is finite, so a rejected
    /// trial is bisected until every image is back inside the region or
    /// the step no longer moves.
    ///
    /// The objective closure has to be `Sync`, and the rest of the
    /// search is serial, so the only record of an evaluation is two
    /// atomics: whether the last call stayed above the energy, and the
    /// length of that trial. A panic in the surface propagates; there
    /// is no lock to poison and no poisoned state to keep using.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<LinebReport, SaddleError> {
        let n = self.weighted.nrows();
        let dof = self.weighted.ncols();
        let flat_dof = n * dof;
        if self.cap_held {
            self.solver.set_maxmove(f64::INFINITY);
            if let Some(cap) = self.atom_cap {
                self.solver.set_atom_maxmove(cap);
            }
            self.cap_held = false;
        }
        let config = self.config.clone();
        let masses = self.masses.clone();
        let energy = self.energy;
        let calls = &self.evaluations;
        let counted = crate::lineb::Counted { surface, calls };
        // 1: the last evaluation stayed above the energy. 2: it did not.
        let phase = AtomicU8::new(0);
        let length_bits = AtomicU64::new(0.0_f64.to_bits());
        let mut x = action::pack(self.weighted.view());
        let before = x.clone();
        // `f64::INFINITY` marks "the stepper's own length", not a bound.
        let mut cap = f64::INFINITY;
        let assembly = loop {
            phase.store(0, Ordering::Relaxed);
            let origin = before.clone();
            let oracle = Oracle::unbounded(flat_dof, |trial: ArrayView1<f64>| {
                let displacement = action::l2((&trial.to_owned() - &origin).view());
                length_bits.store(displacement.to_bits(), Ordering::Relaxed);
                let images = action::unpack(trial, n, dof);
                match action::assemble(
                    &counted,
                    &masses,
                    images.view(),
                    energy,
                    config.spring,
                    config.endpoint_scale,
                ) {
                    Ok(assembly) => {
                        phase.store(1, Ordering::Relaxed);
                        let mut force = action::pack(assembly.drive.view());
                        force *= -1.0;
                        (assembly.action, force)
                    }
                    Err(_) => {
                        phase.store(2, Ordering::Relaxed);
                        (f64::INFINITY, Array1::zeros(flat_dof))
                    }
                }
            });
            let stepped = self.solver.step(&oracle, &mut x);
            drop(oracle);
            if phase.load(Ordering::Relaxed) == 1 && stepped.is_ok() {
                break action::assemble(
                    &counted,
                    &masses,
                    action::unpack(x.view(), n, dof).view(),
                    energy,
                    config.spring,
                    config.endpoint_scale,
                )?;
            }
            self.solver.forget_evaluation();
            if phase.load(Ordering::Relaxed) != 2 {
                return Err(SaddleError::Solver(match stepped {
                    Err(err) => err.to_string(),
                    Ok(_) => "the tunneling step was rejected without an evaluation".into(),
                }));
            }
            x.assign(&before);
            let attempted = f64::from_bits(length_bits.load(Ordering::Relaxed));
            let base = if cap.is_finite() {
                cap.min(attempted)
            } else {
                attempted
            };
            let shorter = 0.5 * base;
            if !(shorter > 0.0) {
                if !(attempted > 0.0) {
                    return Err(SaddleError::Invalid(
                        "a tunneling step left the region above the energy without moving".into(),
                    ));
                }
                let held = action::assemble(
                    &counted,
                    &masses,
                    self.weighted.view(),
                    energy,
                    config.spring,
                    config.endpoint_scale,
                )?;
                let converged =
                    !(held.perp_max > config.perp_tol) && !(held.opt_max > config.opt_tol);
                if converged {
                    return Ok(LinebReport {
                        status: LinebStatus::Converged,
                        action: held.action,
                        perp_max: held.perp_max,
                        opt_max: held.opt_max,
                        iteration: self.iteration,
                        evaluations: self.evaluations.load(Ordering::Relaxed),
                    });
                }
                return Err(SaddleError::Stalled(format!(
                    "no shorter step keeps the tunneling path above the energy, nudged force {}",
                    held.opt_max
                )));
            }
            cap = shorter;
            self.solver.set_maxmove(cap);
            if self.atom_cap.is_some() {
                self.solver.set_atom_maxmove(cap);
            }
            self.cap_held = true;
        };
        let delta = &x - &before;
        let displacement = action::l2(delta.view());
        self.weighted = action::unpack(x.view(), n, dof);
        self.iteration += 1;
        let converged =
            !(assembly.perp_max > config.perp_tol) && !(assembly.opt_max > config.opt_tol);
        if !(displacement > 0.0) && !converged {
            return Err(SaddleError::Stalled(format!(
                "tunneling path unchanged with nudged force {} above {}",
                assembly.opt_max, config.opt_tol
            )));
        }
        Ok(LinebReport {
            status: if converged {
                LinebStatus::Converged
            } else {
                LinebStatus::Running
            },
            action: assembly.action,
            perp_max: assembly.perp_max,
            opt_max: assembly.opt_max,
            iteration: self.iteration,
            evaluations: self.evaluations.load(Ordering::Relaxed),
        })
    }

    /// Step until the tolerances are met or `max_steps` steps have been
    /// taken. The report of the last step is returned either way.
    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<LinebReport, SaddleError> {
        let mut report = self.step(surface)?;
        while report.status == LinebStatus::Running && self.iteration < max_steps {
            report = self.step(surface)?;
        }
        Ok(report)
    }

    /// Spline action, both half periods and both temperatures of the
    /// current images.
    pub fn path<S: PointSurface>(&self, surface: &S) -> Result<LinebPath, SaddleError> {
        let assembly = action::assemble(
            surface,
            &self.masses,
            self.weighted.view(),
            self.energy,
            self.config.spring,
            self.config.endpoint_scale,
        )?;
        let periods: Periods = path::measure(
            surface,
            &self.masses,
            assembly.aligned.view(),
            self.energy,
            self.config.quadrature_pairs,
        )?;
        let mut cartesian = Array2::zeros(assembly.aligned.dim());
        for i in 0..assembly.aligned.nrows() {
            cartesian
                .row_mut(i)
                .assign(&self.masses.to_cartesian(assembly.aligned.row(i))?);
        }
        Ok(LinebPath {
            cartesian,
            energy: self.energy,
            discrete_action: assembly.action,
            action: periods.action,
            half_period: periods.half_period,
            half_period_dynamics: periods.half_period_dynamics,
            temperature: path::temperature(
                periods.half_period,
                self.config.hbar,
                self.config.boltzmann,
            )?,
            temperature_dynamics: path::temperature(
                periods.half_period_dynamics,
                self.config.hbar,
                self.config.boltzmann,
            )?,
            weighted: assembly.aligned,
            masses: self.masses.clone(),
            quadrature_pairs: self.config.quadrature_pairs,
            hbar: self.config.hbar,
            boltzmann: self.config.boltzmann,
        })
    }
}
