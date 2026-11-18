//! Projected average force integration along a path.
//!
//! Swinburne and Marinica, Phys. Rev. Lett. 120, 135503 (2018),
//! doi:10.1103/PhysRevLett.120.135503, give a path-based free-energy
//! gradient that does not need a collective variable. The path `X(s)` is
//! the spline through the images of [`ArcPath`], and the hyperplane at `s`
//! is `(x - X) · t = 0`, with `t = X'` less its center-of-mass part when
//! the center of mass is held. With `ψ = (x - X) · X'' / |t|²`,
//!
//! `dF/ds = ⟨ (∇E · t) (1 - ψ) ⟩`.
//!
//! The paper's `ψ`, its Eq. (5), is `1 - ψ` here. [`PafiMeanForce`] selects
//! either the complete Eq. (8), including its analytic Jacobian derivative,
//! or the force term alone, which is what LAMMPS `fix pafi` and Swinburne's
//! public driver record. The force-only integral is the free energy of the
//! flat hyperplanes,
//! `-kT ln ∫ δ((x - X) · t / |t|) exp(-E/kT) dx`. Sampling here is
//! overdamped Langevin (Brownian) dynamics in the plane, the paper's Eq.
//! (9). The direct
//! Euler--Maruyama discretization and the step of Leimkuhler and Matthews,
//! Appl. Math. Res. Express 2013, 34 (2013),
//! doi:10.1093/amrx/abs010, Eq. (3), are both available through
//! [`PafiIntegrator`]. The latter sums two consecutive noise draws, which
//! makes stationary averages second order in the step and exact on a
//! harmonic plane. The parallel piece of the drift and of the noise is
//! removed, and the plane is restored after every step. Holding the center
//! of mass is optional and is the `com yes` choice of `fix pafi`. The
//! integral of `dF/ds` from the reactant is the potential of mean force.
//! One trajectory uses block averages for its standard error. An ensemble
//! uses the spread of independent trajectory means, as Swinburne's
//! reference code does.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;

use rgmin::{Control, Error, Method, Oracle, Solver, StopNorm};

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::relax::RelaxConfig;
use crate::reparam::Spline;
use crate::stats::{Rng, block_mean};

/// Time discretization of the projected overdamped Langevin equation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum PafiIntegrator {
    /// Euler--Maruyama applied directly to Eq. (9) of Swinburne and
    /// Marinica (2018).
    EulerMaruyama,
    /// The correlated-noise step of Leimkuhler and Matthews (2013),
    /// second order for stationary averages.
    #[default]
    LeimkuhlerMatthews,
}

/// Instantaneous mean-force estimator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum PafiMeanForce {
    /// The modified force recorded by LAMMPS `fix pafi` and the public
    /// PAFI driver: the first term of the paper's Eq. (8).
    #[default]
    ReferenceCode,
    /// Both terms of Swinburne and Marinica (2018), Eq. (8), including
    /// `kT d/ds ln(|ψ| / |X'|)`.
    PublishedEquation8,
}

/// Overdamped hyperplane sampling.
///
/// Non-exhaustive: start from [`PafiConfig::default`] and chain the
/// `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PafiConfig {
    /// `kT` in the energy unit of the surface.
    pub thermal: f64,
    /// Mobility `μ` in `dx = -μ ∇E dt + sqrt(2 μ kT dt) ξ`.
    pub mobility: f64,
    /// Overdamped time step.
    pub time_step: f64,
    /// Steps discarded before the first recorded sample.
    pub thermalize: usize,
    /// Steps between recorded samples.
    pub stride: usize,
    /// Recorded samples. At least twice [`Self::blocks`].
    pub samples: usize,
    /// Contiguous blocks used for the standard error. At least 2.
    pub blocks: usize,
    /// Independent trajectories of equal length on each plane. With more
    /// than one, the reported error is the reference PAFI ensemble error
    /// `sqrt(sum((mean_i - mean)^2)) / trajectories`.
    pub trajectories: usize,
    /// Time discretization of the projected overdamped Langevin equation.
    pub integrator: PafiIntegrator,
    /// Instantaneous estimator to average.
    pub mean_force: PafiMeanForce,
    /// Seed of the noise.
    pub seed: u64,
    /// Remove the mean Cartesian drift when the system has at least two atoms.
    pub fix_com: bool,
    /// How many recorded configurations to keep, evenly spaced.
    pub retain: usize,
    /// Stop early when the integrated barrier error is at or below this.
    /// `None` runs one pass.
    pub target_error: Option<f64>,
    /// Extra sampling passes used to reach [`Self::target_error`].
    pub max_rounds: usize,
    /// Largest displacement of one atom, in the length unit of the
    /// surface, between the plane minimum quenched from a sampled
    /// configuration and the one quenched from the path. Past it the
    /// configuration has left the basin of the path. `None` skips the test.
    pub max_jump: Option<f64>,
    /// The in-plane quench of [`Self::max_jump`]: its method, force
    /// tolerance, step cap and per-atom step, on the force with its part
    /// along the normal (and its center-of-mass part when held) removed.
    /// Reaching the step cap, or a stall of the method, ends the quench
    /// where it is.
    pub quench: RelaxConfig,
}

with_setters!(PafiConfig {
    with_thermal => thermal: val f64,
    with_mobility => mobility: val f64,
    with_time_step => time_step: val f64,
    with_thermalize => thermalize: val usize,
    with_stride => stride: val usize,
    with_samples => samples: val usize,
    with_blocks => blocks: val usize,
    with_trajectories => trajectories: val usize,
    with_integrator => integrator: val PafiIntegrator,
    with_mean_force => mean_force: val PafiMeanForce,
    with_seed => seed: val u64,
    with_fix_com => fix_com: val bool,
    with_retain => retain: val usize,
    with_target_error => target_error: opt f64,
    with_max_rounds => max_rounds: val usize,
    with_max_jump => max_jump: opt f64,
    with_quench => quench: val RelaxConfig,
});

impl Default for PafiConfig {
    /// `kT` of 300 K in eV, unit mobility, time step 1e-3, 200
    /// thermalization steps, one trajectory of 200 samples 5 steps apart
    /// in 5 blocks, the Leimkuhler--Matthews integrator, the reference-code
    /// mean-force estimator, seed 1, the center of mass held, 16 kept
    /// configurations, one pass, and a largest jump of 0.4 (Å), the
    /// `MaxJumpThresh` of Swinburne's reference code, quenched along the
    /// steepest-descent flow ([`Method::Ode12r`]) under the other
    /// [`RelaxConfig`] defaults.
    fn default() -> Self {
        Self {
            thermal: 0.02585,
            mobility: 1.0,
            time_step: 1.0e-3,
            thermalize: 200,
            stride: 5,
            samples: 200,
            blocks: 5,
            trajectories: 1,
            integrator: PafiIntegrator::LeimkuhlerMatthews,
            mean_force: PafiMeanForce::ReferenceCode,
            seed: 1,
            fix_com: true,
            retain: 16,
            target_error: None,
            max_rounds: 4,
            max_jump: Some(0.4),
            quench: RelaxConfig::default().with_method(Method::Ode12r),
        }
    }
}

impl PafiConfig {
    fn check(&self) -> Result<(), SaddleError> {
        if !(self.thermal.is_finite() && self.thermal > 0.0) {
            return Err(SaddleError::Invalid(
                "PAFI kT must be finite and positive".into(),
            ));
        }
        if !(self.mobility.is_finite() && self.mobility > 0.0) {
            return Err(SaddleError::Invalid(
                "PAFI mobility must be finite and positive".into(),
            ));
        }
        if !(self.time_step.is_finite() && self.time_step > 0.0) {
            return Err(SaddleError::Invalid(
                "PAFI time step must be finite and positive".into(),
            ));
        }
        if self.stride == 0
            || self.blocks < 2
            || self.samples < self.blocks * 2
            || self.trajectories == 0
        {
            return Err(SaddleError::Invalid(
                "PAFI needs a positive stride and trajectory count, at least two blocks, and two samples per block".into(),
            ));
        }
        if let Some(jump) = self.max_jump {
            if !(jump.is_finite() && jump > 0.0) {
                return Err(SaddleError::Invalid(
                    "PAFI largest jump must be finite and positive".into(),
                ));
            }
            self.quench.check()?;
        }
        Ok(())
    }
}

/// One frame of a path, with its parameter.
#[derive(Clone, Debug)]
pub struct PathFrame {
    /// Cartesian coordinates of the frame.
    pub positions: Array1<f64>,
    /// Path parameter of the frame. [`ArcPath::from_positions`] uses the
    /// summed chord lengths from the first frame.
    pub arc: f64,
}

/// Cubic-spline path through ordered frames.
///
/// Every coordinate is the not-a-knot cubic spline of the frames over
/// their [`PathFrame::arc`] values: the spline interpolation of NEB images
/// of Swinburne and Marinica (2018) and the default end condition of
/// scipy's `CubicSpline`, which their reference code uses. Two frames give
/// the straight line and three the parabola through them. `X`, `X'` and
/// `X''` are continuous, so the hyperplanes turn smoothly along the path
/// and the plane through a frame is the same from either side of it.
#[derive(Clone, Debug)]
pub struct ArcPath {
    spline: Spline,
}

/// Position and the first three parameter derivatives of a path.
#[derive(Clone, Debug)]
pub struct PathGeometry {
    /// `X(s)`.
    pub positions: Array1<f64>,
    /// `X' = dX/ds`, the hyperplane normal before any center-of-mass part
    /// is removed.
    pub tangent: Array1<f64>,
    /// `X'' = d²X/ds²`.
    pub curvature: Array1<f64>,
    /// One-sided `X''' = d³X/ds³` of the cubic spline. At an interior
    /// knot, where this derivative may jump, this is the derivative from
    /// the segment on the reactant side.
    pub jerk: Array1<f64>,
    /// `s`.
    pub arc: f64,
}

impl ArcPath {
    /// Build the path from ordered frames. The parameter must strictly
    /// increase and neighboring frames must differ.
    pub fn new(frames: Vec<PathFrame>) -> Result<Self, SaddleError> {
        if frames.len() < 2 {
            return Err(SaddleError::Invalid(
                "a PAFI path needs at least two frames".into(),
            ));
        }
        let n = frames[0].positions.len();
        if n == 0 {
            return Err(SaddleError::Shape("PAFI path frame is empty".into()));
        }
        for (i, frame) in frames.iter().enumerate() {
            if frame.positions.len() != n || !frame.positions.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::Shape(
                    "PAFI path frames must share one finite length".into(),
                ));
            }
            if !frame.arc.is_finite() || (i > 0 && frame.arc <= frames[i - 1].arc) {
                return Err(SaddleError::Invalid(
                    "PAFI arc length must strictly increase".into(),
                ));
            }
            if i > 0 && frame.positions == frames[i - 1].positions {
                return Err(SaddleError::Invalid(
                    "neighboring PAFI frames must differ".into(),
                ));
            }
        }
        let arcs: Vec<f64> = frames.iter().map(|frame| frame.arc).collect();
        let mut values = Array2::zeros((frames.len(), n));
        for (mut row, frame) in values.outer_iter_mut().zip(&frames) {
            row.assign(&frame.positions);
        }
        let spline = Spline::fit(&arcs, &values)
            .ok_or_else(|| SaddleError::Invalid("the PAFI spline system is singular".into()))?;
        Ok(Self { spline })
    }

    /// The same curve traversed from the last frame, with the parameter
    /// `first + last - s`. Integrating it gives the reversal test of a
    /// profile: its forward barrier starts from the other end.
    pub fn reversed(&self) -> Result<Self, SaddleError> {
        let (first, last) = self.bounds();
        let values = self.spline.values();
        Self::new(
            self.spline
                .knots()
                .iter()
                .zip(values.outer_iter())
                .rev()
                .map(|(arc, positions)| PathFrame {
                    positions: positions.to_owned(),
                    arc: first + last - arc,
                })
                .collect(),
        )
    }

    /// Frames in order, reactant first.
    pub fn from_positions(positions: &[Array1<f64>]) -> Result<Self, SaddleError> {
        if positions.len() < 2 {
            return Err(SaddleError::Invalid(
                "a PAFI path needs at least two frames".into(),
            ));
        }
        let mut arc = 0.0;
        let mut frames = Vec::with_capacity(positions.len());
        frames.push(PathFrame {
            positions: positions[0].clone(),
            arc,
        });
        for pair in positions.windows(2) {
            let step = pair[1]
                .iter()
                .zip(pair[0].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            if !(step.is_finite() && step > 0.0) {
                return Err(SaddleError::Invalid(
                    "PAFI frames must be finite and distinct".into(),
                ));
            }
            arc += step;
            frames.push(PathFrame {
                positions: pair[1].clone(),
                arc,
            });
        }
        Self::new(frames)
    }

    /// Arc length of the reactant and of the product.
    pub fn bounds(&self) -> (f64, f64) {
        let knots = self.spline.knots();
        (knots[0], knots[knots.len() - 1])
    }

    /// Arc length of every stored frame.
    pub fn arcs(&self) -> Vec<f64> {
        self.spline.knots().to_vec()
    }

    /// Spline geometry at `arc`.
    pub fn geometry(&self, arc: f64) -> Result<PathGeometry, SaddleError> {
        if !arc.is_finite() {
            return Err(SaddleError::NonFinite("PAFI arc"));
        }
        let (first, last) = self.bounds();
        if arc < first || arc > last {
            return Err(SaddleError::Invalid(format!(
                "PAFI coordinate {arc} is outside [{first}, {last}]"
            )));
        }
        let [positions, tangent, curvature, jerk] = self.spline.derivatives(arc);
        Ok(PathGeometry {
            positions,
            tangent,
            curvature,
            jerk,
            arc,
        })
    }
}

/// One basin rejection during hyperplane sampling.
#[derive(Clone, Debug, PartialEq)]
pub struct PafiBasinRejection {
    /// Reaction-coordinate value of the sampled plane.
    pub plane: f64,
    /// Zero-based trajectory identifier on that plane.
    pub trajectory: usize,
    /// Zero-based identifier of the basin-tested stretch in that trajectory.
    pub stretch: usize,
    /// Largest per-atom displacement between the two quenched minima.
    pub max_jump: f64,
}

/// Mean free-energy gradient on one hyperplane.
#[derive(Clone, Debug)]
pub struct HyperplaneAverage {
    /// `dF/ds` in energy per arc length.
    pub gradient: f64,
    /// Standard error from the block means.
    pub error: f64,
    /// Mean of `ψ = (x - X) · X'' / |t|²`. The paper's locality condition,
    /// its Eq. (6), is `1 - ψ > 0`.
    pub psi: f64,
    /// Recorded samples in the production run.
    pub samples: usize,
    /// Integrated autocorrelation time of the recorded series, in samples.
    pub tau: f64,
    /// `samples / (2 τ)`.
    pub n_eff: f64,
    /// Force evaluations spent on this plane, the quenches of
    /// [`PafiConfig::max_jump`] included, summed over rounds.
    pub evaluations: usize,
    /// Configurations kept from the production run, on the plane.
    pub retained: Vec<Array1<f64>>,
    /// Thermalizations and stretches of the production run sampled again
    /// because they left the basin of the path ([`PafiConfig::max_jump`]),
    /// summed over rounds.
    pub rejected: usize,
    /// Context and measured jump for each rejected stretch.
    pub rejections: Vec<PafiBasinRejection>,
}

/// Potential of mean force from the first plane.
#[derive(Clone, Debug)]
pub struct FreeEnergyProfile {
    /// Arc length of each plane.
    pub arc: Vec<f64>,
    /// `dF/ds` on each plane.
    pub gradient: Vec<f64>,
    /// Standard error of `gradient`.
    pub gradient_error: Vec<f64>,
    /// `F(s)`, trapezoid-integrated from the first plane.
    pub free_energy: Vec<f64>,
    /// Propagated standard error of `free_energy`.
    pub free_energy_error: Vec<f64>,
    /// Maximum of the potential of mean force. The reactant is zero.
    pub barrier: f64,
    /// Standard error of `barrier`.
    pub barrier_error: f64,
    /// Index of that maximum.
    pub dividing: usize,
    /// Recorded samples, summed over planes and rounds.
    pub samples: usize,
}

/// Sample one hyperplane and return the PAFI gradient.
///
/// With [`PafiConfig::max_jump`] set, the production run is cut into
/// [`PafiConfig::blocks`] stretches. The thermalized configuration and the
/// end of every stretch are quenched in the plane and compared with the
/// plane minimum quenched from the path; one atom farther than `max_jump`
/// means the configuration has left the basin of the path. A stretch that
/// left it is dropped and sampled again from the end of the last kept one,
/// and a thermalization from the path. Swinburne's reference code drops
/// such a trajectory the same way. More dropped stretches than there are
/// blocks is an error: the plane does not hold the system in the basin.
pub fn average_hyperplane<S: PointSurface>(
    surface: &S,
    geometry: &PathGeometry,
    config: &PafiConfig,
) -> Result<HyperplaneAverage, SaddleError> {
    config.check()?;
    if config.trajectories == 1 {
        return average_trajectory(surface, geometry, config, 0);
    }
    let mut trajectories = Vec::with_capacity(config.trajectories);
    for index in 0..config.trajectories {
        let local = config
            .clone()
            .with_trajectories(1)
            .with_seed(config.seed.wrapping_add(index as u64));
        trajectories.push(average_trajectory(surface, geometry, &local, index)?);
    }
    let count = trajectories.len() as f64;
    let gradient = trajectories
        .iter()
        .map(|trajectory| trajectory.gradient)
        .sum::<f64>()
        / count;
    let error = trajectories
        .iter()
        .map(|trajectory| {
            let residual = trajectory.gradient - gradient;
            residual * residual
        })
        .sum::<f64>()
        .sqrt()
        / count;
    let samples = trajectories
        .iter()
        .map(|trajectory| trajectory.samples)
        .sum();
    let psi = trajectories
        .iter()
        .map(|trajectory| trajectory.psi * trajectory.samples as f64)
        .sum::<f64>()
        / samples as f64;
    let mut retained = Vec::with_capacity(config.retain);
    for trajectory in &trajectories {
        let remaining = config.retain.saturating_sub(retained.len());
        retained.extend(trajectory.retained.iter().take(remaining).cloned());
    }
    let rejections: Vec<PafiBasinRejection> = trajectories
        .iter()
        .flat_map(|trajectory| trajectory.rejections.iter().cloned())
        .collect();
    let rejected = rejections.len();
    Ok(HyperplaneAverage {
        gradient,
        error,
        psi,
        samples,
        tau: trajectories
            .iter()
            .map(|trajectory| trajectory.tau)
            .fold(0.0, f64::max),
        n_eff: trajectories.iter().map(|trajectory| trajectory.n_eff).sum(),
        evaluations: trajectories
            .iter()
            .map(|trajectory| trajectory.evaluations)
            .sum(),
        retained,
        rejected,
        rejections,
    })
}

fn average_trajectory<S: PointSurface>(
    surface: &S,
    geometry: &PathGeometry,
    config: &PafiConfig,
    trajectory: usize,
) -> Result<HyperplaneAverage, SaddleError> {
    config.check()?;
    let n = geometry.positions.len();
    if geometry.tangent.len() != n || geometry.curvature.len() != n || geometry.jerk.len() != n {
        return Err(SaddleError::Shape(
            "PAFI path derivatives must match the position".into(),
        ));
    }
    if !geometry
        .tangent
        .iter()
        .chain(geometry.curvature.iter())
        .chain(geometry.jerk.iter())
        .all(|v| v.is_finite())
    {
        return Err(SaddleError::NonFinite("PAFI path geometry"));
    }
    let plane = Plane::new(geometry, config.fix_com && n >= 6 && n.is_multiple_of(3))?;
    let mut rng = Rng::new(config.seed);
    let walk = Walk {
        surface,
        plane: &plane,
        drift_scale: config.mobility * config.time_step,
        thermal_scale: config.mobility * config.thermal * config.time_step,
        integrator: config.integrator,
    };
    let mut evaluations = 0;
    let basin = match config.max_jump {
        Some(_) => {
            let (minimum, spent) = plane.minimum(surface, &geometry.positions, &config.quench)?;
            evaluations += spent;
            Some(minimum)
        }
        None => None,
    };
    let mut rejections = Vec::new();
    let mut left =
        |x: &Array1<f64>, evaluations: &mut usize, stretch: usize| -> Result<bool, SaddleError> {
            let (Some(basin), Some(limit)) = (&basin, config.max_jump) else {
                return Ok(false);
            };
            let (minimum, spent) = plane.minimum(surface, x, &config.quench)?;
            *evaluations += spent;
            let max_jump = StopNorm::MaxAtom.of((&minimum - basin).view());
            if max_jump <= limit {
                return Ok(false);
            }
            rejections.push(PafiBasinRejection {
                plane: geometry.arc,
                trajectory,
                stretch,
                max_jump,
            });
            if rejections.len() > config.blocks {
                return Err(SaddleError::Solver(format!(
                    "PAFI plane at s = {} left the basin of the path {} times",
                    geometry.arc,
                    rejections.len()
                )));
            }
            Ok(true)
        };
    let mut stretch_id = 0;
    let (mut x, mut noise) = loop {
        let mut x = geometry.positions.clone();
        let mut noise = walk.draw(n, &mut rng);
        for _ in 0..config.thermalize {
            let gradient = walk.gradient(&x)?;
            walk.step(&mut x, gradient, &mut noise, &mut rng);
        }
        evaluations += config.thermalize;
        let rejected = left(&x, &mut evaluations, stretch_id)?;
        stretch_id += 1;
        if !rejected {
            break (x, noise);
        }
    };
    let stretch = config.samples.div_ceil(config.blocks);
    let retain_gap = config.samples.div_ceil(config.retain.max(1)).max(1);
    let mut observations = Vec::with_capacity(config.samples);
    let mut psi_sum = 0.0;
    let mut retained = Vec::new();
    let mut kept = (x.clone(), noise.clone());
    while observations.len() < config.samples {
        let count = stretch.min(config.samples - observations.len());
        let mut values = Vec::with_capacity(count);
        let mut psi = 0.0;
        let mut configurations = Vec::new();
        for _ in 0..count {
            for offset in 0..config.stride {
                let gradient = walk.gradient(&x)?;
                if offset == 0 {
                    let (value, deviation) = projected_gradient(&gradient, &x, &plane, config)?;
                    if !value.is_finite() || !deviation.is_finite() {
                        return Err(SaddleError::NonFinite("PAFI projected gradient"));
                    }
                    let recorded = observations.len() + values.len();
                    values.push(value);
                    psi += deviation;
                    if config.retain > 0
                        && recorded.is_multiple_of(retain_gap)
                        && retained.len() + configurations.len() < config.retain
                    {
                        configurations.push(x.clone());
                    }
                }
                walk.step(&mut x, gradient, &mut noise, &mut rng);
            }
        }
        evaluations += count * config.stride;
        let rejected = left(&x, &mut evaluations, stretch_id)?;
        stretch_id += 1;
        if rejected {
            (x, noise) = kept.clone();
            continue;
        }
        observations.extend(values);
        psi_sum += psi;
        retained.extend(configurations);
        kept = (x.clone(), noise.clone());
    }
    let (gradient, error, tau, n_eff) = block_mean(&observations, config.blocks);
    Ok(HyperplaneAverage {
        gradient,
        error,
        psi: psi_sum / config.samples as f64,
        samples: config.samples,
        tau,
        n_eff,
        evaluations,
        retained,
        rejected: rejections.len(),
        rejections,
    })
}

/// The hyperplane `(x - X) · t = 0` through a point of the path.
struct Plane<'a> {
    geometry: &'a PathGeometry,
    /// `t`, less its center-of-mass part when that is held.
    normal: Array1<f64>,
    /// `dt/ds`, in the same center-of-mass quotient as `normal`.
    curvature: Array1<f64>,
    /// `d²t/ds²`, in the same center-of-mass quotient as `normal`.
    jerk: Array1<f64>,
    /// `|t|²`.
    nn: f64,
    hold_com: bool,
}

impl<'a> Plane<'a> {
    fn new(geometry: &'a PathGeometry, hold_com: bool) -> Result<Self, SaddleError> {
        let mut normal = geometry.tangent.clone();
        let mut curvature = geometry.curvature.clone();
        let mut jerk = geometry.jerk.clone();
        if hold_com {
            remove_com(&mut normal);
            remove_com(&mut curvature);
            remove_com(&mut jerk);
        }
        let nn = normal.dot(&normal);
        if !(nn.is_finite() && nn > 0.0) {
            return Err(SaddleError::Invalid("PAFI tangent vanishes".into()));
        }
        Ok(Self {
            geometry,
            normal,
            curvature,
            jerk,
            nn,
            hold_com,
        })
    }

    /// Remove the part of `vector` along the normal, and its center-of-mass
    /// part when held. `normal` has none, so the two removals commute.
    fn project(&self, vector: &mut Array1<f64>) {
        if self.hold_com {
            remove_com(vector);
        }
        project_out(vector, self.normal.view());
    }

    /// Move `x` back onto the plane along the normal.
    fn restore(&self, x: &mut Array1<f64>) {
        let slip = (&*x - &self.geometry.positions).dot(&self.normal) / self.nn;
        x.scaled_add(-slip, &self.normal);
    }

    /// The minimum in the plane that `quench` reaches from `start`, and the
    /// force evaluations it took.
    fn minimum<S: PointSurface>(
        &self,
        surface: &S,
        start: &Array1<f64>,
        quench: &RelaxConfig,
    ) -> Result<(Array1<f64>, usize), SaddleError> {
        let n = start.len();
        let mut solver = Solver::new(
            quench.method.clone(),
            Control::default()
                .with_maxiter(quench.max_steps)
                .with_gtol(quench.force_tol)
                .with_stop_norm(StopNorm::MaxAtom)
                .with_istep(quench.initial_step),
            n,
        );
        solver.set_atom_maxmove(quench.max_atom_move);
        let evaluations = AtomicUsize::new(0);
        let oracle = Oracle::unbounded(n, |x: ArrayView1<f64>| {
            evaluations.fetch_add(1, Ordering::Relaxed);
            match surface.eval(x) {
                Ok((energy, mut gradient)) if gradient.len() == n => {
                    self.project(&mut gradient);
                    (energy, gradient)
                }
                _ => (f64::NAN, Array1::from_elem(n, f64::NAN)),
            }
        });
        let mut x = start.clone();
        match solver.run(&oracle, &mut x) {
            Ok(_) | Err(Error::FireStalled { .. } | Error::AcceptStalled { .. }) => {}
            Err(error) => return Err(SaddleError::Solver(format!("PAFI plane quench: {error}"))),
        }
        self.restore(&mut x);
        Ok((x, evaluations.into_inner()))
    }
}

/// Overdamped Langevin dynamics confined to a [`Plane`].
struct Walk<'a, S> {
    surface: &'a S,
    plane: &'a Plane<'a>,
    /// `μ dt`.
    drift_scale: f64,
    /// `μ kT dt`.
    thermal_scale: f64,
    integrator: PafiIntegrator,
}

impl<S: PointSurface> Walk<'_, S> {
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, SaddleError> {
        let (_energy, gradient) = self.surface.eval(x.view())?;
        if gradient.len() != x.len() || !gradient.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::Shape(
                "PAFI gradient must match the path".into(),
            ));
        }
        Ok(gradient)
    }

    /// A standard normal vector in the plane.
    fn draw(&self, len: usize, rng: &mut Rng) -> Array1<f64> {
        let mut noise = Array1::from_shape_simple_fn(len, || rng.normal());
        self.plane.project(&mut noise);
        noise
    }

    /// One projected overdamped step from `x`, where `gradient` is the
    /// gradient at the start of the step.
    fn step(
        &self,
        x: &mut Array1<f64>,
        gradient: Array1<f64>,
        noise: &mut Array1<f64>,
        rng: &mut Rng,
    ) {
        let mut drift = gradient;
        self.plane.project(&mut drift);
        let fresh = self.draw(x.len(), rng);
        let random_step = match self.integrator {
            PafiIntegrator::EulerMaruyama => &fresh * (2.0 * self.thermal_scale).sqrt(),
            PafiIntegrator::LeimkuhlerMatthews => {
                (&*noise + &fresh) * (0.5 * self.thermal_scale).sqrt()
            }
        };
        *x = &*x - &(&drift * self.drift_scale) + &random_step;
        *noise = fresh;
        self.plane.restore(x);
    }
}

/// Integrate plane gradients with the trapezoid rule.
///
/// The dividing surface is the maximum of the potential of mean force.
pub fn integrate_profile(
    arcs: &[f64],
    gradients: &[f64],
    errors: &[f64],
) -> Result<FreeEnergyProfile, SaddleError> {
    if arcs.len() < 2 || arcs.len() != gradients.len() || arcs.len() != errors.len() {
        return Err(SaddleError::Shape(
            "PAFI integration needs matching arcs, gradients and errors".into(),
        ));
    }
    let mut free = vec![0.0; arcs.len()];
    for i in 1..arcs.len() {
        let ds = arcs[i] - arcs[i - 1];
        if !(ds.is_finite() && ds > 0.0) {
            return Err(SaddleError::Invalid(
                "PAFI arcs must strictly increase".into(),
            ));
        }
        if !gradients[i].is_finite() || !gradients[i - 1].is_finite() {
            return Err(SaddleError::NonFinite("PAFI gradient"));
        }
        free[i] = free[i - 1] + 0.5 * (gradients[i - 1] + gradients[i]) * ds;
    }
    let free_error = prefix_errors(arcs, errors)?;
    let dividing = dividing_index(&free);
    Ok(FreeEnergyProfile {
        barrier: free[dividing],
        barrier_error: free_error[dividing],
        dividing,
        arc: arcs.to_vec(),
        gradient: gradients.to_vec(),
        gradient_error: errors.to_vec(),
        free_energy: free,
        free_energy_error: free_error,
        samples: 0,
    })
}

/// Sample every requested arc and integrate.
///
/// When [`PafiConfig::target_error`] is set, planes are resampled until the
/// barrier error is at or below that target or [`PafiConfig::max_rounds`]
/// is spent. Later rounds are inverse-variance combined with the first.
pub fn pafi_profile<S: PointSurface + Sync>(
    surface: &S,
    path: &ArcPath,
    arcs: &[f64],
    config: &PafiConfig,
) -> Result<(FreeEnergyProfile, Vec<HyperplaneAverage>), SaddleError> {
    if arcs.len() < 2 {
        return Err(SaddleError::Invalid(
            "PAFI needs at least two hyperplanes".into(),
        ));
    }
    let mut planes = sample_planes(surface, path, arcs, config, 0)?;
    let mut profile = profile_of(arcs, &planes)?;
    let rounds = config.max_rounds.max(1);
    let mut round = 1usize;
    while let Some(target) = config.target_error {
        if profile.barrier_error <= target || round >= rounds {
            break;
        }
        if profile.barrier_error <= 0.0 {
            break;
        }
        let requested = (profile.barrier_error / target).powi(2);
        let factor = if requested < 2.0 {
            2.0
        } else if requested > 8.0 {
            8.0
        } else {
            requested
        };
        let samples = ((config.samples as f64) * factor).ceil() as usize;
        let extra = config
            .clone()
            .with_samples(samples.max(config.blocks * 2))
            .with_thermalize(config.thermalize / 2)
            .with_target_error(None);
        let more = sample_planes(surface, path, arcs, &extra, round as u64 * 10_000)?;
        for (plane, added) in planes.iter_mut().zip(more) {
            let (gradient, error) =
                combine(plane.gradient, plane.error, added.gradient, added.error);
            plane.gradient = gradient;
            plane.error = error;
            plane.samples += added.samples;
            plane.evaluations += added.evaluations;
            plane.tau = plane.tau.max(added.tau);
            plane.n_eff += added.n_eff;
            plane.rejected += added.rejected;
            plane.rejections.extend(added.rejections);
            if plane.retained.len() < config.retain {
                plane.retained.extend(added.retained);
            }
        }
        profile = profile_of(arcs, &planes)?;
        round += 1;
    }
    profile.samples = planes.iter().map(|plane| plane.samples).sum();
    Ok((profile, planes))
}

fn sample_planes<S: PointSurface + Sync>(
    surface: &S,
    path: &ArcPath,
    arcs: &[f64],
    config: &PafiConfig,
    salt: u64,
) -> Result<Vec<HyperplaneAverage>, SaddleError> {
    arcs.par_iter()
        .enumerate()
        .map(|(index, arc)| {
            let geometry = path.geometry(*arc)?;
            let local = config.clone().with_seed(
                config
                    .seed
                    .wrapping_add(salt)
                    .wrapping_add(index as u64)
                    .wrapping_add(1),
            );
            average_hyperplane(surface, &geometry, &local)
        })
        .collect()
}

fn profile_of(
    arcs: &[f64],
    planes: &[HyperplaneAverage],
) -> Result<FreeEnergyProfile, SaddleError> {
    let gradients: Vec<f64> = planes.iter().map(|plane| plane.gradient).collect();
    let errors: Vec<f64> = planes.iter().map(|plane| plane.error).collect();
    integrate_profile(arcs, &gradients, &errors)
}

fn combine(mean_a: f64, error_a: f64, mean_b: f64, error_b: f64) -> (f64, f64) {
    let va = error_a * error_a;
    let vb = error_b * error_b;
    if va == 0.0 && vb == 0.0 {
        return (0.5 * (mean_a + mean_b), 0.0);
    }
    if va == 0.0 {
        return (mean_a, 0.0);
    }
    if vb == 0.0 {
        return (mean_b, 0.0);
    }
    let weight_a = 1.0 / va;
    let weight_b = 1.0 / vb;
    let mean = (weight_a * mean_a + weight_b * mean_b) / (weight_a + weight_b);
    let error = (1.0 / (weight_a + weight_b)).sqrt();
    (mean, error)
}

/// Integral of `∇E · dX/ds` with the system held on the path.
///
/// The integral is the energy along the spline, `E(X(s)) - E(X(s0))` up to
/// the quadrature, and passes through the image energies at the frames.
/// On a converged minimum-energy path the force in each hyperplane
/// vanishes, so this is also the zero-temperature limit of the PAFI
/// integral and reproduces the potential barrier.
pub fn zero_temperature_barrier<S: PointSurface>(
    surface: &S,
    path: &ArcPath,
    arcs: &[f64],
) -> Result<FreeEnergyProfile, SaddleError> {
    if arcs.len() < 2 {
        return Err(SaddleError::Invalid(
            "the zero-temperature integral needs at least two points".into(),
        ));
    }
    let arcs = dense_arcs(arcs, 64);
    let mut gradients = Vec::with_capacity(arcs.len());
    let errors = vec![0.0; arcs.len()];
    for arc in &arcs {
        let geometry = path.geometry(*arc)?;
        let (_energy, gradient) = surface.eval(geometry.positions.view())?;
        if gradient.len() != geometry.tangent.len() {
            return Err(SaddleError::Shape(
                "zero-temperature gradient must match the path".into(),
            ));
        }
        gradients.push(gradient.dot(&geometry.tangent));
    }
    integrate_profile(&arcs, &gradients, &errors)
}

/// `parts` samples on every span. One trapezoid per image is not enough
/// when the directional derivative curves, and the integral then sits
/// below the barrier.
fn dense_arcs(arcs: &[f64], parts: usize) -> Vec<f64> {
    let parts = parts.max(1);
    let mut dense = Vec::with_capacity(arcs.len().saturating_mul(parts));
    for window in arcs.windows(2) {
        for k in 0..parts {
            dense.push(window[0] + (window[1] - window[0]) * (k as f64) / parts as f64);
        }
    }
    if let Some(last) = arcs.last() {
        dense.push(*last);
    }
    dense
}

fn prefix_errors(arcs: &[f64], errors: &[f64]) -> Result<Vec<f64>, SaddleError> {
    let mut out = vec![0.0; arcs.len()];
    for (i, slot) in out.iter_mut().enumerate().skip(1) {
        let mut used = vec![0.0; arcs.len()];
        for k in 1..=i {
            let span = arcs[k] - arcs[k - 1];
            used[k - 1] += 0.5 * span;
            used[k] += 0.5 * span;
        }
        let mut variance = 0.0;
        for j in 0..=i {
            let sigma = errors[j];
            if !(sigma.is_finite() && sigma >= 0.0) {
                return Err(SaddleError::Invalid(
                    "PAFI gradient error must be finite and non-negative".into(),
                ));
            }
            variance += used[j] * used[j] * sigma * sigma;
        }
        *slot = variance.sqrt();
    }
    Ok(out)
}

fn dividing_index(free: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..free.len() {
        if free[i] > free[best] {
            best = i;
        }
    }
    best
}

/// The selected Eq. (8) estimator and
/// `ψ = (x - X) · X'' / |t|²` for the plane normal `t`.
fn projected_gradient(
    gradient: &Array1<f64>,
    x: &Array1<f64>,
    plane: &Plane<'_>,
    config: &PafiConfig,
) -> Result<(f64, f64), SaddleError> {
    let deviation = x - &plane.geometry.positions;
    let curvature_deviation = deviation.dot(&plane.curvature);
    let psi = curvature_deviation / plane.nn;
    let paper_psi = 1.0 - psi;
    let force = gradient.dot(&plane.normal) * paper_psi;
    match config.mean_force {
        PafiMeanForce::ReferenceCode => Ok((force, psi)),
        PafiMeanForce::PublishedEquation8 => {
            if !(paper_psi.is_finite() && paper_psi > 0.0) {
                return Err(SaddleError::Solver(format!(
                    "PAFI locality condition failed at s = {}: psi = {paper_psi}",
                    plane.geometry.arc
                )));
            }
            // At fixed x, q = (x-X)·X'', n = X'·X':
            // q' = (x-X)·X''' - X'·X'', n' = 2 X'·X''.
            let tangent_curvature = plane.normal.dot(&plane.curvature);
            let q_derivative = deviation.dot(&plane.jerk) - tangent_curvature;
            let norm_derivative = 2.0 * tangent_curvature;
            let paper_psi_derivative =
                -q_derivative / plane.nn + curvature_deviation * norm_derivative / plane.nn.powi(2);
            let jacobian =
                config.thermal * (paper_psi_derivative / paper_psi - tangent_curvature / plane.nn);
            Ok((force + jacobian, psi))
        }
    }
}

fn project_out(vector: &mut Array1<f64>, tangent: ArrayView1<f64>) {
    let nn = tangent.dot(&tangent);
    if nn == 0.0 {
        return;
    }
    let coef = vector.dot(&tangent) / nn;
    *vector -= &(tangent.to_owned() * coef);
}

pub(crate) fn remove_com(vector: &mut Array1<f64>) {
    let n = vector.len() / 3;
    if n == 0 {
        return;
    }
    let mut mean = [0.0; 3];
    for atom in 0..n {
        for c in 0..3 {
            mean[c] += vector[3 * atom + c];
        }
    }
    for value in &mut mean {
        *value /= n as f64;
    }
    for atom in 0..n {
        for c in 0..3 {
            vector[3 * atom + c] -= mean[c];
        }
    }
}

#[cfg(test)]
mod estimator_tests {
    use ndarray::Array1;

    use super::{PafiConfig, PafiMeanForce, PathGeometry, Plane, projected_gradient};

    #[test]
    fn published_estimator_includes_the_analytic_jacobian() {
        let geometry = PathGeometry {
            positions: Array1::zeros(2),
            tangent: Array1::from_vec(vec![1.0, 1.0]),
            curvature: Array1::from_vec(vec![0.0, 1.0]),
            jerk: Array1::zeros(2),
            arc: 1.0,
        };
        let plane = Plane::new(&geometry, false).expect("plane");
        let x = Array1::from_vec(vec![-0.2, 0.2]);
        let gradient = Array1::zeros(2);
        let reference = PafiConfig::default()
            .with_thermal(0.3)
            .with_mean_force(PafiMeanForce::ReferenceCode);
        let published = reference
            .clone()
            .with_mean_force(PafiMeanForce::PublishedEquation8);
        let (reference_force, reference_deviation) =
            projected_gradient(&gradient, &x, &plane, &reference).expect("reference estimator");
        let (published_force, published_deviation) =
            projected_gradient(&gradient, &x, &plane, &published).expect("published estimator");
        let roundoff = 128.0 * f64::EPSILON;
        assert!(reference_force.abs() <= roundoff);
        assert!((reference_deviation - 0.1).abs() <= roundoff);
        assert!((published_deviation - reference_deviation).abs() <= roundoff);
        assert!((published_force - 0.05).abs() <= roundoff);
    }

    #[test]
    fn published_estimator_enforces_the_paper_locality_condition() {
        let geometry = PathGeometry {
            positions: Array1::zeros(2),
            tangent: Array1::from_vec(vec![1.0, 1.0]),
            curvature: Array1::from_vec(vec![0.0, 1.0]),
            jerk: Array1::zeros(2),
            arc: 1.0,
        };
        let plane = Plane::new(&geometry, false).expect("plane");
        let config = PafiConfig::default().with_mean_force(PafiMeanForce::PublishedEquation8);
        let error = projected_gradient(
            &Array1::zeros(2),
            &Array1::from_vec(vec![-2.0, 2.0]),
            &plane,
            &config,
        )
        .expect_err("zero paper psi violates locality");
        assert!(error.to_string().contains("locality condition"));
    }
}
