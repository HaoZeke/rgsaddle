//! Projected average force integration along a path.
//!
//! Swinburne and Marinica, Phys. Rev. Lett. 120, 135503 (2018),
//! doi:10.1103/PhysRevLett.120.135503, give a path-based free-energy
//! gradient that does not need a collective variable. The chain samples
//! the Euclidean surface measure on the flat hyperplane
//! `(x - X) · ν = 0`. With `τ = dX/dr` the path derivative and `ν` the
//! unit constraint normal,
//!
//! `dF/dr = ⟨ (∇E · ν) (τ · ν - (x - X) · dν/dr) ⟩`.
//!
//! When `ν` is the unit tangent and `r` is its arc length, `τ · ν = 1`
//! and `ψ = (x - X) · dν/dr`, so this is the multiply form LAMMPS
//! `fix pafi` averages, `⟨ (∇E · ν) (1 - ψ) ⟩`. A normal that keeps only
//! some of the atoms is a different foliation: the same formula then
//! gives the free energy of that truncated plane, and the barrier depends
//! on which atoms were kept. [`pafi_frames`] therefore uses the full
//! tangent. Rigid translations are removed from it by
//! [`crate::HyperplaneConstraint::orthogonal_to`], and the estimator uses that same
//! unit normal, so the constraint and the average are one plane.
//!
//! Sampling is overdamped Langevin (Brownian) dynamics in the plane, by
//! default with the step of Leimkuhler and Matthews ([`Integrator`]). The
//! parallel piece of the drift and of the noise is removed, and the plane
//! is restored after every step. Holding the center of mass is the
//! `com yes` choice of that fix. The integral of `dF/dr` from the
//! reactant is the potential of mean force. The standard error is the
//! larger of the block-mean error and the one from the integrated
//! autocorrelation time, so a correlated series is not treated as
//! independent. When a plane needs more samples, its chain goes on from
//! where it stopped instead of starting cold.
//!
//! [`pafi_profile`] samples planes at given arc lengths of the polyline.
//! [`pafi_frames`] puts one plane on every frame and takes each choice the
//! barrier could otherwise depend on from the band itself, so that a band
//! traversed the other way, or the same hop built in a symmetry-equivalent
//! frame, gives the same barrier within its error.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::ohtst::{
    self, Hyperplane, HyperplaneConstraint, TransmissionCoefficient, TransmissionConfig,
    transmission_coefficient,
};
use crate::stats::{Rng, windowed_block_mean};

/// Discretization of the overdamped dynamics, with `h = μ dt`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Integrator {
    /// `x' = x - h ∇E + sqrt(2 h kT) ξ`. Averages carry an error of first
    /// order in `h`: a harmonic well of stiffness `k` comes out wider by
    /// `1 / (1 - h k / 2)`.
    EulerMaruyama,
    /// Leimkuhler and Matthews, Appl. Math. Res. Express 2013, 34 (2013),
    /// doi:10.1093/amrx/abs010: `x' = x - h ∇E + sqrt(h kT / 2) (ξ + ξ')`,
    /// with `ξ` the noise of the step before. It samples a harmonic well
    /// exactly at any stable step, and averages carry an error of second
    /// order in `h` otherwise.
    #[default]
    LeimkuhlerMatthews,
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
    /// Discretization of the dynamics.
    pub integrator: Integrator,
    /// Steps discarded before the first recorded sample.
    pub thermalize: usize,
    /// Steps between recorded samples.
    pub stride: usize,
    /// Recorded samples of the first pass. At least twice [`Self::blocks`].
    pub samples: usize,
    /// Contiguous blocks used for the standard error. At least 2.
    pub blocks: usize,
    /// Seed of the noise.
    pub seed: u64,
    /// Remove the mean Cartesian drift when the system has at least two atoms.
    pub fix_com: bool,
    /// How many recorded configurations to keep, evenly spaced over the
    /// first pass.
    pub retain: usize,
    /// Stop early when the integrated barrier error is at or below this.
    /// `None` runs one pass.
    pub target_error: Option<f64>,
    /// Sampling passes used to reach [`Self::target_error`], the first
    /// included.
    pub max_rounds: usize,
    /// Largest atom force left in the plane when [`pafi_frames`] relaxes a
    /// frame in its plane before sampling.
    pub relax_tol: f64,
    /// The `MaxJumpThresh` test of Swinburne's PAFI: after every
    /// [`Self::window`] recorded samples the configuration is relaxed in
    /// its plane to [`Self::jump_tol`], and when an atom ends further than
    /// this from the minimum of the plane, the chain has left the basin of
    /// the path for a nearby one that the plane also cuts. Those samples
    /// are dropped and the chain goes back to where the last kept window
    /// ended. The planes then average over the basin of the path alone.
    /// `None` keeps every sample.
    pub max_jump: Option<f64>,
    /// Largest atom force left by the relaxation of [`Self::max_jump`].
    pub jump_tol: f64,
    /// Recorded samples between two tests of [`Self::max_jump`], shorter
    /// than the time a chain takes to leave the basin of the path. A plane
    /// that keeps no window in its first pass halves its window, up to four
    /// times, and thermalizes in windows of that length too. At least 1.
    pub window: usize,
}

with_setters!(PafiConfig {
    with_thermal => thermal: val f64,
    with_mobility => mobility: val f64,
    with_time_step => time_step: val f64,
    with_integrator => integrator: val Integrator,
    with_thermalize => thermalize: val usize,
    with_stride => stride: val usize,
    with_samples => samples: val usize,
    with_blocks => blocks: val usize,
    with_seed => seed: val u64,
    with_fix_com => fix_com: val bool,
    with_retain => retain: val usize,
    with_target_error => target_error: opt f64,
    with_max_rounds => max_rounds: val usize,
    with_relax_tol => relax_tol: val f64,
    with_max_jump => max_jump: opt f64,
    with_jump_tol => jump_tol: val f64,
    with_window => window: val usize,
});

impl Default for PafiConfig {
    fn default() -> Self {
        Self {
            thermal: 0.02585,
            mobility: 1.0,
            time_step: 1.0e-3,
            integrator: Integrator::default(),
            thermalize: 200,
            stride: 5,
            samples: 200,
            blocks: 5,
            seed: 1,
            fix_com: true,
            retain: 16,
            target_error: None,
            max_rounds: 4,
            relax_tol: 1.0e-4,
            max_jump: None,
            jump_tol: 1.0e-2,
            window: 200,
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
        if self.stride == 0 || self.blocks < 2 || self.samples < self.blocks * 2 {
            return Err(SaddleError::Invalid(
                "PAFI needs a positive stride, at least two blocks, and two samples per block"
                    .into(),
            ));
        }
        if !(self.relax_tol.is_finite() && self.relax_tol > 0.0) {
            return Err(SaddleError::Invalid(
                "PAFI relaxation tolerance must be finite and positive".into(),
            ));
        }
        if self
            .max_jump
            .is_some_and(|jump| !(jump.is_finite() && jump > 0.0))
            || !(self.jump_tol.is_finite() && self.jump_tol > 0.0)
            || self.window == 0
        {
            return Err(SaddleError::Invalid(
                "PAFI jump threshold and its tolerance must be finite and positive, and its window at least one sample".into(),
            ));
        }
        Ok(())
    }
}

/// One frame of a piecewise path, with its arc length from the reactant.
#[derive(Clone, Debug)]
pub struct PathFrame {
    /// Cartesian coordinates of the frame.
    pub positions: Array1<f64>,
    /// Arc length from the first frame.
    pub arc: f64,
}

/// Cubic Hermite path. The parameter is the polyline arc length.
#[derive(Clone, Debug)]
pub struct ArcPath {
    frames: Vec<PathFrame>,
    /// Endpoint derivatives kept so a caller can rebuild the Hermite data.
    /// The geometry used for the integral is the polyline through the frames:
    /// a cubic spline between NEB images cuts the corner off a curved saddle
    /// and the zero-temperature integral then falls short of the barrier.
    #[allow(dead_code)]
    tangents: Vec<Array1<f64>>,
}

/// Position, path derivative `n = dX/ds`, and `dn/ds` at one arc length.
#[derive(Clone, Debug)]
pub struct PathGeometry {
    /// `X(s)`.
    pub positions: Array1<f64>,
    /// `n = dX/ds`.
    pub tangent: Array1<f64>,
    /// `dn/ds`.
    pub curvature: Array1<f64>,
    /// `s`.
    pub arc: f64,
}

/// The planes [`pafi_frames`] samples, one through every frame.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct FramePlanes {
    /// Unit normal at each frame. [`ArcPath::truncated_planes`] leaves it
    /// zero off [`Self::active`]; [`ArcPath::frame_planes`] does not.
    pub normals: Vec<Array1<f64>>,
    /// `dn/ds` at each frame, zero wherever the normal is.
    pub curvatures: Vec<Array1<f64>>,
    /// Arc length of each frame from the first, along the coordinates the
    /// normal uses.
    pub arcs: Vec<f64>,
    /// Atoms the normals act on, ascending. An atom is a coordinate triple,
    /// or one coordinate when the length is not a multiple of three.
    pub active: Vec<usize>,
}

impl ArcPath {
    /// Build the path from ordered frames. Arc lengths must strictly increase.
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
        }
        let mut tangents = Vec::with_capacity(frames.len());
        for i in 0..frames.len() {
            let tangent = if i == 0 {
                difference(
                    &frames[1].positions,
                    &frames[0].positions,
                    frames[1].arc - frames[0].arc,
                )
            } else if i + 1 == frames.len() {
                difference(
                    &frames[i].positions,
                    &frames[i - 1].positions,
                    frames[i].arc - frames[i - 1].arc,
                )
            } else {
                difference(
                    &frames[i + 1].positions,
                    &frames[i - 1].positions,
                    frames[i + 1].arc - frames[i - 1].arc,
                )
            };
            if tangent.iter().all(|v| *v == 0.0) {
                return Err(SaddleError::Invalid(
                    "a PAFI frame has a zero tangent".into(),
                ));
            }
            tangents.push(tangent);
        }
        Ok(Self { frames, tangents })
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
        (self.frames[0].arc, self.frames.last().expect("frames").arc)
    }

    /// Arc length of every stored frame.
    pub fn arcs(&self) -> Vec<f64> {
        self.frames.iter().map(|frame| frame.arc).collect()
    }

    /// The stored frames, reactant first.
    pub fn frames(&self) -> &[PathFrame] {
        &self.frames
    }

    /// Polyline geometry at `arc`. The tangent is the chord of the segment.
    pub fn geometry(&self, arc: f64) -> Result<PathGeometry, SaddleError> {
        if !arc.is_finite() {
            return Err(SaddleError::NonFinite("PAFI arc"));
        }
        let last = self.frames.len() - 1;
        let arc = arc.clamp(self.frames[0].arc, self.frames[last].arc);
        let mut seg = last - 1;
        for i in 0..last {
            if arc <= self.frames[i + 1].arc {
                seg = i;
                break;
            }
        }
        let s0 = self.frames[seg].arc;
        let s1 = self.frames[seg + 1].arc;
        let dt = s1 - s0;
        let t = ((arc - s0) / dt).clamp(0.0, 1.0);
        let x0 = &self.frames[seg].positions;
        let x1 = &self.frames[seg + 1].positions;
        let n = x0.len();
        let mut position = Array1::zeros(n);
        let mut tangent = Array1::zeros(n);
        for i in 0..n {
            let step = x1[i] - x0[i];
            position[i] = x0[i] + t * step;
            tangent[i] = step / dt;
        }
        Ok(PathGeometry {
            positions: position,
            tangent,
            curvature: Array1::zeros(n),
            arc,
        })
    }

    /// The planes of [`pafi_frames`].
    ///
    /// The normal is the unit tangent of the parabola through the frame
    /// and its two neighbours, in the arc length of the chords, and the
    /// curvature is the same derivative of the normals. The end frames
    /// take the parabola through themselves and the next two. Both are of
    /// second order in the spacing, where a chord tilts the plane by half
    /// the angle the path turns over it, and a reversed band gives the
    /// same planes. Every atom carries the tangent: keeping only the atoms
    /// that hold some fraction of the summed squared steps replaces this
    /// plane by the one [`Self::truncated_planes`] builds, and the barrier
    /// of that plane depends on the fraction.
    pub fn frame_planes(&self) -> Result<FramePlanes, SaddleError> {
        self.planes_on(1.0)
    }

    /// Planes whose normals are zero off the fewest atoms, largest first,
    /// that carry `share` of the summed squared steps.
    ///
    /// This is not the plane of [`pafi_frames`]. The constraint
    /// `(x - X) · ν = 0` with `ν` the renormalized masked tangent is a
    /// different foliation from `(x - X) · τ = 0`. On a harmonic transverse
    /// well the two barriers differ by [`harmonic_mask_shift`]. `share`
    /// lies in `(0, 1]`; 1 is [`Self::frame_planes`].
    pub fn truncated_planes(&self, share: f64) -> Result<FramePlanes, SaddleError> {
        self.planes_on(share)
    }

    fn planes_on(&self, share: f64) -> Result<FramePlanes, SaddleError> {
        let count = self.frames.len();
        if !(share > 0.0 && share <= 1.0) {
            return Err(SaddleError::Invalid(
                "PAFI active share must lie in (0, 1]".into(),
            ));
        }
        let n = self.frames[0].positions.len();
        let width = atom_width(n);
        let steps: Vec<Array1<f64>> = self
            .frames
            .windows(2)
            .map(|pair| &pair[1].positions - &pair[0].positions)
            .collect();
        let active = active_atoms(&steps, width, share);
        let mut mask = Array1::<f64>::zeros(n);
        for atom in &active {
            for c in 0..width {
                mask[atom * width + c] = 1.0;
            }
        }
        let local: Vec<Array1<f64>> = steps.iter().map(|step| step * &mask).collect();
        let lengths: Vec<f64> = local.iter().map(norm).collect();
        if !lengths
            .iter()
            .all(|length| length.is_finite() && *length > 0.0)
        {
            return Err(SaddleError::Invalid(
                "a PAFI frame step moves none of the constrained atoms".into(),
            ));
        }
        let mut arcs = Vec::with_capacity(count);
        arcs.push(0.0);
        for length in &lengths {
            arcs.push(arcs[arcs.len() - 1] + length);
        }
        let normals = arc_derivative(&local, &lengths)
            .into_iter()
            .map(|tangent| {
                let length = norm(&tangent);
                if length.is_finite() && length > 0.0 {
                    Ok(tangent / length)
                } else {
                    Err(SaddleError::Invalid(
                        "the path turns back on itself at a PAFI frame".into(),
                    ))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let turns: Vec<Array1<f64>> = normals.windows(2).map(|pair| &pair[1] - &pair[0]).collect();
        let curvatures = arc_derivative(&turns, &lengths);
        Ok(FramePlanes {
            normals,
            curvatures,
            arcs,
            active,
        })
    }
}

/// Mean free-energy gradient on one hyperplane.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct HyperplaneAverage {
    /// The constrained plane, with its normal toward increasing path arc.
    pub plane: Hyperplane,
    /// `dF/ds` in energy per arc length.
    pub gradient: f64,
    /// Standard error: the larger of the block-mean error and
    /// `sqrt(2 τ var / samples)`.
    pub error: f64,
    /// Mean of `ψ = (x - X) · dn/ds`.
    pub psi: f64,
    /// Recorded samples, summed over passes.
    pub samples: usize,
    /// Integrated autocorrelation time of the recorded series, in samples.
    pub tau: f64,
    /// `samples / (2 τ)`.
    pub n_eff: f64,
    /// Force evaluations spent on this plane, summed over passes.
    pub evaluations: usize,
    /// Configurations kept from the first pass, on the plane.
    pub retained: Vec<Array1<f64>>,
    /// Energy at the origin of the plane, where its chain starts.
    pub energy: f64,
    /// `∇E · n` at the origin of the plane: the zero-temperature value of
    /// [`Self::gradient`] when the origin is the minimum in the plane.
    pub zero: f64,
    /// Largest displacement of one atom from the origin of the plane over
    /// the recorded samples. An atom is a coordinate triple, or one
    /// coordinate when the length is not a multiple of three.
    pub widest: f64,
    /// Windows dropped by [`PafiConfig::max_jump`].
    pub jumps: usize,
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

impl FreeEnergyProfile {
    /// Barrier from the last plane back over the dividing plane,
    /// `F(dividing) - F(last)`, with the standard error of the trapezoid
    /// sum over the planes between them. A path whose last plane is the
    /// product minimum gives the reverse hop from the same samples.
    pub fn reverse_barrier(&self) -> (f64, f64) {
        let last = self.free_energy.len().saturating_sub(1);
        let mut weights = vec![0.0; self.arc.len()];
        for k in self.dividing + 1..=last {
            let span = self.arc[k] - self.arc[k - 1];
            weights[k - 1] += 0.5 * span;
            weights[k] += 0.5 * span;
        }
        let variance: f64 = weights
            .iter()
            .zip(&self.gradient_error)
            .map(|(w, e)| w * w * e * e)
            .sum();
        (
            self.free_energy[self.dividing] - self.free_energy[last],
            variance.sqrt(),
        )
    }

    /// [`balanced_barriers`] of [`Self::barrier`] and
    /// [`Self::reverse_barrier`] against `end_gap`, the free energy of the
    /// last end less that of the first, known apart from this profile.
    pub fn balanced(&self, end_gap: f64) -> Result<BalancedBarriers, SaddleError> {
        balanced_barriers(
            (self.barrier, self.barrier_error),
            self.reverse_barrier(),
            end_gap,
        )
    }

    /// [`anchored_barriers`] of [`Self::barrier`] and
    /// [`Self::reverse_barrier`] against `end_gap`, the free energy of the
    /// last end less that of the first, known apart from this profile.
    pub fn anchored(&self, end_gap: f64) -> Result<BalancedBarriers, SaddleError> {
        anchored_barriers(
            (self.barrier, self.barrier_error),
            self.reverse_barrier(),
            end_gap,
        )
    }
}

/// Both barriers of one path, made to differ by the free-energy
/// difference of its ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BalancedBarriers {
    /// Barrier above the first end.
    pub forward: f64,
    /// Barrier above the last end, `end_gap` below [`Self::forward`].
    pub reverse: f64,
    /// Standard error of both: with half of [`Self::drift`] in quadrature
    /// from [`balanced_barriers`], that of the forward barrier from
    /// [`anchored_barriers`].
    pub error: f64,
    /// `forward - reverse - end_gap` of the inputs: what the integral adds
    /// to the free-energy difference of the ends.
    pub drift: f64,
}

/// Combine the forward and reverse barriers of one path given the
/// free-energy difference of its ends.
///
/// The barrier above the first plane and [`FreeEnergyProfile::reverse_barrier`]
/// above the last measure one dividing surface from the two ends, so their
/// difference is `end_gap`, the free energy of the last end less that of
/// the first, plus whatever the integral drifts along the path. With
/// `end_gap` known apart from the profile, `forward` and
/// `reverse + end_gap` are two estimates of one barrier. The result is
/// their inverse-variance mean, the reverse barrier `end_gap` below it, and
/// the drift. Half the drift is added to the error in quadrature: it is
/// what the mean moves when the whole drift lies on one side of the
/// dividing surface and the two estimates weigh alike; when it is known to
/// lie past it, [`anchored_barriers`] keeps the forward barrier instead.
/// `end_gap = 0` is a path between two equivalent states. Errors must be
/// positive.
pub fn balanced_barriers(
    forward: (f64, f64),
    reverse: (f64, f64),
    end_gap: f64,
) -> Result<BalancedBarriers, SaddleError> {
    check_barriers(forward, reverse, end_gap)?;
    let (weight_f, weight_r) = (forward.1.powi(-2), reverse.1.powi(-2));
    let mean = (weight_f * forward.0 + weight_r * (reverse.0 + end_gap)) / (weight_f + weight_r);
    let drift = forward.0 - reverse.0 - end_gap;
    Ok(BalancedBarriers {
        forward: mean,
        reverse: mean - end_gap,
        error: ((weight_f + weight_r).recip() + (0.5 * drift).powi(2)).sqrt(),
        drift,
    })
}

/// Both barriers of one path from its forward barrier and the free-energy
/// difference of its ends.
///
/// The forward barrier integrates the profile from the first plane to the
/// dividing surface and [`FreeEnergyProfile::reverse_barrier`] from there
/// to the last plane, so each carries the drift of its own stretch. With
/// `end_gap`, the free energy of the last end less that of the first,
/// known apart from the profile, the reverse barrier by detailed balance
/// is `forward - end_gap`, and only the forward stretch enters either. The
/// error is that of `forward`, and [`BalancedBarriers::drift`] still
/// reports `forward - reverse - end_gap`. It is the estimate to take over
/// [`balanced_barriers`] when the drift lies past the dividing surface, as
/// a path integrated again from its other end shows: the barriers from the
/// start of each integration then agree, and those from its end do not.
/// `end_gap = 0` is a path between two equivalent states. Errors must be
/// positive.
pub fn anchored_barriers(
    forward: (f64, f64),
    reverse: (f64, f64),
    end_gap: f64,
) -> Result<BalancedBarriers, SaddleError> {
    check_barriers(forward, reverse, end_gap)?;
    Ok(BalancedBarriers {
        forward: forward.0,
        reverse: forward.0 - end_gap,
        error: forward.1,
        drift: forward.0 - reverse.0 - end_gap,
    })
}

fn check_barriers(
    forward: (f64, f64),
    reverse: (f64, f64),
    end_gap: f64,
) -> Result<(), SaddleError> {
    if ![forward.0, forward.1, reverse.0, reverse.1, end_gap]
        .iter()
        .all(|v| v.is_finite())
    {
        return Err(SaddleError::NonFinite("PAFI barrier"));
    }
    if !(forward.1 > 0.0 && reverse.1 > 0.0) {
        return Err(SaddleError::Invalid(
            "PAFI barrier errors must be positive".into(),
        ));
    }
    Ok(())
}

/// Flat-hyperplane derivative at one configuration.
///
/// `dF/dr = (∇E · ν) (τ · ν - (x - X) · dν/dr)` on the Euclidean surface
/// measure of `(x - X) · ν = 0`. `normal` is the unit constraint normal
/// `ν`, `normal_rate` is `dν/dr`, `path_tangent` is `τ = dX/dr`, and
/// `displacement` is `x - X`. When `ν` is the unit path tangent and `r`
/// its arc length, `τ · ν = 1` and this is `(∇E · ν) (1 - ψ)`.
pub fn flat_hyperplane_gradient(
    gradient: ArrayView1<f64>,
    displacement: ArrayView1<f64>,
    normal: ArrayView1<f64>,
    normal_rate: ArrayView1<f64>,
    path_tangent: ArrayView1<f64>,
) -> Result<f64, SaddleError> {
    let n = gradient.len();
    if displacement.len() != n
        || normal.len() != n
        || normal_rate.len() != n
        || path_tangent.len() != n
    {
        return Err(SaddleError::Shape(
            "PAFI flat gradient needs one length for every vector".into(),
        ));
    }
    if !gradient
        .iter()
        .chain(displacement.iter())
        .chain(normal.iter())
        .chain(normal_rate.iter())
        .chain(path_tangent.iter())
        .all(|v| v.is_finite())
    {
        return Err(SaddleError::NonFinite("PAFI flat gradient"));
    }
    let factor = path_tangent.dot(&normal) - displacement.dot(&normal_rate);
    Ok(gradient.dot(&normal) * factor)
}

/// Map a masked-plane barrier onto the barrier of another normal.
///
/// On a path whose transverse motion is harmonic, and whose planes pass
/// through the same stationary points, the flat-hyperplane barrier with
/// unit normal `n` and the one with unit normal `n*` differ by the
/// entropy of the two curvatures. With `c(n) = nᵀ H⁻¹ n` the compliance,
/// `det(H restricted to n⊥) = det(H) c(n)` for invertible `H`, so the
/// Gaussian integrals give
///
/// `ΔF(n*) = ΔF(n) + (kT/2) ln[ (c_s(n*)/c_s(n)) (c_m(n)/c_m(n*)) ]`.
///
/// `reactant_*` is `c` on the reactant plane and `dividing_*` is `c` on
/// the dividing plane. A negative curvature, as at a saddle, is carried
/// by a negative compliance; the ratio of the two dividing compliances
/// has to be positive, and likewise the reactant pair. The shift does not
/// include a slide of the plane's minimum off the path point: that energy
/// is `E(path) - E(minimum)` and is added by the caller.
pub fn harmonic_mask_shift(
    barrier: f64,
    reactant_full: f64,
    reactant_mask: f64,
    dividing_full: f64,
    dividing_mask: f64,
    thermal: f64,
) -> Result<f64, SaddleError> {
    let values = [
        barrier,
        reactant_full,
        reactant_mask,
        dividing_full,
        dividing_mask,
        thermal,
    ];
    if !values.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("PAFI mask shift"));
    }
    if thermal <= 0.0 {
        return Err(SaddleError::Invalid(
            "PAFI mask shift needs a positive kT".into(),
        ));
    }
    let ratio = (dividing_full / dividing_mask) * (reactant_mask / reactant_full);
    if !(ratio.is_finite() && ratio > 0.0) {
        return Err(SaddleError::Invalid(
            "PAFI mask compliances need a positive ratio at each end".into(),
        ));
    }
    Ok(barrier + 0.5 * thermal * ratio.ln())
}

/// Sample one hyperplane and return the PAFI gradient.
pub fn average_hyperplane<S: PointSurface>(
    surface: &S,
    geometry: &PathGeometry,
    config: &PafiConfig,
) -> Result<HyperplaneAverage, SaddleError> {
    config.check()?;
    let directions = plane_directions(surface, geometry.positions.view(), config.fix_com)?;
    let plane = Plane::new(geometry, directions.view())?;
    let mut chain = Chain::new(&plane, config.seed, config);
    chain.run(surface, &plane, config, config.thermalize, config.samples)?;
    if chain.values.is_empty() {
        return Err(SaddleError::Stalled(
            "PAFI plane left the basin of the path in every window".into(),
        ));
    }
    Ok(chain.average(config, &plane))
}

/// Shoot the maximum-free-energy plane of a sampled PAFI profile.
///
/// The starts are the constrained-equilibrium configurations retained by
/// [`pafi_profile`] or [`pafi_frames`] on the profile's dividing plane.
/// Shooting is unconstrained in both time directions and is delegated to
/// [`transmission_coefficient`]. The PAFI plane normal points toward
/// increasing path arc, the product side of a reactant-first band.
pub fn pafi_transmission<S: PointSurface>(
    surface: &S,
    profile: &FreeEnergyProfile,
    planes: &[HyperplaneAverage],
    config: &TransmissionConfig,
) -> Result<TransmissionCoefficient, SaddleError> {
    if planes.len() != profile.free_energy.len() || profile.dividing >= planes.len() {
        return Err(SaddleError::Shape(
            "PAFI transmission needs one sampled plane per profile value".into(),
        ));
    }
    let dividing = &planes[profile.dividing];
    if dividing.retained.is_empty() {
        return Err(SaddleError::Invalid(
            "PAFI transmission needs retained constrained samples; raise PafiConfig::retain".into(),
        ));
    }
    transmission_coefficient(surface, &dividing.plane, &dividing.retained, config)
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
/// When [`PafiConfig::target_error`] is set, the chains of the planes up to
/// the dividing surface, the ones the barrier error is made of, go on from
/// where they stopped until that error is at or below the target or
/// [`PafiConfig::max_rounds`] passes are spent. A pass adds the samples
/// the error scaling as one over their root asks for, at least
/// [`PafiConfig::samples`] and at most eight times that.
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
    config.check()?;
    let geometries = arcs
        .iter()
        .map(|arc| path.geometry(*arc))
        .collect::<Result<Vec<_>, _>>()?;
    sample_rounds(surface, &geometries, config, |planes| {
        profile_of(arcs, planes)
    })
}

/// One hyperplane on every frame of `path`, each sampled from its own
/// minimum, integrated against the zero-temperature energy of those minima.
///
/// The planes are [`ArcPath::frame_planes`]: the full tangent, not a
/// truncation of it. Each frame is relaxed in its plane to
/// [`PafiConfig::relax_tol`], giving `x*_k` and `E*_k`; its chain starts
/// there and `ψ` is measured from there. With
/// `g = (∇E · ν) (τ · ν - (x - X) · dν/ds)`,
///
/// `F_k = E*_k - E*_0 + Σ trapezoid(⟨g⟩ - g(x*))`
///
/// over the arc length of the path. At zero temperature the sum vanishes
/// and the barrier is the energy of the in-plane minima whatever the
/// spacing of the frames, so the trapezoid acts only on the thermal part.
/// The dividing surface is the maximum of `F`, and passes go on as in
/// [`pafi_profile`]. Each plane's [`HyperplaneAverage::evaluations`]
/// includes its relaxation.
pub fn pafi_frames<S: PointSurface + Sync>(
    surface: &S,
    path: &ArcPath,
    config: &PafiConfig,
) -> Result<(FreeEnergyProfile, Vec<HyperplaneAverage>), SaddleError> {
    let planes = path.frame_planes()?;
    sample_frame_planes(surface, path, &planes, config)
}

/// [`pafi_frames`] on the truncated planes of [`ArcPath::truncated_planes`].
///
/// The barrier is the free energy of that truncated foliation. It depends
/// on `share`. [`harmonic_mask_shift`] maps it onto the full-tangent
/// barrier when the transverse motion is harmonic and both planes pass
/// through the same points; a minimum that has slid off the path point
/// adds `E(path) - E(minimum)` at the dividing frame and subtracts it at
/// the reactant.
pub fn pafi_truncated<S: PointSurface + Sync>(
    surface: &S,
    path: &ArcPath,
    share: f64,
    config: &PafiConfig,
) -> Result<(FreeEnergyProfile, Vec<HyperplaneAverage>), SaddleError> {
    let planes = path.truncated_planes(share)?;
    sample_frame_planes(surface, path, &planes, config)
}

fn sample_frame_planes<S: PointSurface + Sync>(
    surface: &S,
    path: &ArcPath,
    planes: &FramePlanes,
    config: &PafiConfig,
) -> Result<(FreeEnergyProfile, Vec<HyperplaneAverage>), SaddleError> {
    config.check()?;
    let n = path.frames[0].positions.len();
    let relaxed = path
        .frames
        .par_iter()
        .zip(&planes.normals)
        .map(|(frame, normal)| {
            let geometry = PathGeometry {
                positions: frame.positions.clone(),
                tangent: normal.clone(),
                curvature: Array1::zeros(n),
                arc: 0.0,
            };
            let directions = plane_directions(surface, frame.positions.view(), config.fix_com)?;
            let plane = Plane::new(&geometry, directions.view())?;
            match relax_in_plane(surface, &plane, frame.positions.clone(), config.relax_tol)? {
                (minimum, evaluations, true) => Ok((minimum, evaluations)),
                _ => Err(SaddleError::Stalled(format!(
                    "PAFI in-plane relaxation did not reach {:e} in {RELAX_STEPS} steps",
                    config.relax_tol
                ))),
            }
        })
        .collect::<Result<Vec<_>, SaddleError>>()?;
    let geometries: Vec<PathGeometry> = relaxed
        .iter()
        .zip(planes.normals.iter().zip(&planes.curvatures))
        .zip(&planes.arcs)
        .map(
            |(((positions, _), (normal, curvature)), arc)| PathGeometry {
                positions: positions.clone(),
                tangent: normal.clone(),
                curvature: curvature.clone(),
                arc: *arc,
            },
        )
        .collect();
    let (profile, mut averages) = sample_rounds(surface, &geometries, config, |averages| {
        control_variate(&planes.arcs, averages)
    })?;
    for (average, (_, steps)) in averages.iter_mut().zip(&relaxed) {
        average.evaluations += steps;
    }
    Ok((profile, averages))
}

fn sample_rounds<S, F>(
    surface: &S,
    geometries: &[PathGeometry],
    config: &PafiConfig,
    integrate: F,
) -> Result<(FreeEnergyProfile, Vec<HyperplaneAverage>), SaddleError>
where
    S: PointSurface + Sync,
    F: Fn(&[HyperplaneAverage]) -> Result<FreeEnergyProfile, SaddleError>,
{
    let planes = geometries
        .iter()
        .map(|geometry| {
            let directions = plane_directions(surface, geometry.positions.view(), config.fix_com)?;
            Plane::new(geometry, directions.view())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut chains: Vec<Chain> = planes
        .iter()
        .enumerate()
        .map(|(index, plane)| {
            let seed = config.seed.wrapping_add(index as u64).wrapping_add(1);
            Chain::new(plane, seed, config)
        })
        .collect();
    chains
        .par_iter_mut()
        .zip(&planes)
        .try_for_each(|(chain, plane)| {
            chain.run(surface, plane, config, config.thermalize, config.samples)
        })?;
    if let Some(index) = chains.iter().position(|chain| chain.values.is_empty()) {
        return Err(SaddleError::Stalled(format!(
            "PAFI plane {index} left the basin of the path in every window"
        )));
    }
    let mut averages: Vec<HyperplaneAverage> = chains
        .iter()
        .zip(&planes)
        .map(|(chain, plane)| chain.average(config, plane))
        .collect();
    let mut profile = integrate(&averages)?;
    let mut round = 1usize;
    while let Some(target) = config.target_error {
        if profile.barrier_error <= target
            || round >= config.max_rounds.max(1)
            || profile.barrier_error <= 0.0
        {
            break;
        }
        let short = (profile.barrier_error / target).powi(2) - 1.0;
        let prefix = profile.dividing + 1;
        chains[..prefix]
            .par_iter_mut()
            .zip(&planes[..prefix])
            .try_for_each(|(chain, plane)| {
                let more = ((chain.values.len() as f64) * short).ceil() as usize;
                let more = more.clamp(config.samples, 8 * config.samples);
                chain.run(surface, plane, config, 0, more)
            })?;
        averages = chains
            .iter()
            .zip(&planes)
            .map(|(chain, plane)| chain.average(config, plane))
            .collect();
        profile = integrate(&averages)?;
        round += 1;
    }
    profile.samples = averages.iter().map(|plane| plane.samples).sum();
    Ok((profile, averages))
}

fn profile_of(
    arcs: &[f64],
    planes: &[HyperplaneAverage],
) -> Result<FreeEnergyProfile, SaddleError> {
    let gradients: Vec<f64> = planes.iter().map(|plane| plane.gradient).collect();
    let errors: Vec<f64> = planes.iter().map(|plane| plane.error).collect();
    integrate_profile(arcs, &gradients, &errors)
}

fn control_variate(
    arcs: &[f64],
    planes: &[HyperplaneAverage],
) -> Result<FreeEnergyProfile, SaddleError> {
    let excess: Vec<f64> = planes
        .iter()
        .map(|plane| plane.gradient - plane.zero)
        .collect();
    let errors: Vec<f64> = planes.iter().map(|plane| plane.error).collect();
    let mut profile = integrate_profile(arcs, &excess, &errors)?;
    let base = planes[0].energy;
    for (free, plane) in profile.free_energy.iter_mut().zip(planes) {
        *free += plane.energy - base;
    }
    profile.dividing = dividing_index(&profile.free_energy);
    profile.barrier = profile.free_energy[profile.dividing];
    profile.barrier_error = profile.free_energy_error[profile.dividing];
    profile.gradient = planes.iter().map(|plane| plane.gradient).collect();
    Ok(profile)
}

/// The plane `(x - X) · ν = 0` and the vectors the estimator dots with.
struct Plane<'a> {
    geometry: &'a PathGeometry,
    /// Constraint. Its normal is `ν`, rigid directions removed.
    constraint: HyperplaneConstraint,
    /// `ν (τ · ν)`, so `∇E · tangent` is `(∇E · ν)(τ · ν)`.
    tangent: Array1<f64>,
    /// `(dν/ds) / (τ · ν)`, so `(1 - (x - X) · curvature)` supplies the
    /// rest of `τ · ν - (x - X) · dν/ds`.
    curvature: Array1<f64>,
}

impl<'a> Plane<'a> {
    fn new(geometry: &'a PathGeometry, directions: ArrayView2<f64>) -> Result<Self, SaddleError> {
        let n = geometry.positions.len();
        if geometry.tangent.len() != n || geometry.curvature.len() != n {
            return Err(SaddleError::Shape(
                "PAFI tangent and curvature must match the position".into(),
            ));
        }
        if !geometry
            .positions
            .iter()
            .chain(geometry.tangent.iter())
            .chain(geometry.curvature.iter())
            .all(|v| v.is_finite())
        {
            return Err(SaddleError::NonFinite("PAFI path geometry"));
        }
        let (tangent, curvature) = if directions.nrows() == 0 {
            (geometry.tangent.clone(), geometry.curvature.clone())
        } else {
            // τ is the stored tangent and dτ/ds its stored derivative.
            // ν = Pτ / |Pτ|, and the estimator wants
            // (∇E · ν) (τ · ν - (x - X) · dν/ds).
            let (normal, rate) = HyperplaneConstraint::orthogonal_rate(
                geometry.tangent.view(),
                geometry.curvature.view(),
                directions,
            )?;
            let alpha = geometry.tangent.dot(&normal);
            if !(alpha.is_finite() && alpha != 0.0) {
                return Err(SaddleError::Invalid(
                    "PAFI tangent is orthogonal to its rigid-mode-free normal".into(),
                ));
            }
            (&normal * alpha, &rate / alpha)
        };
        let constraint =
            HyperplaneConstraint::through(geometry.positions.clone(), tangent.view(), directions)?;
        Ok(Self {
            geometry,
            constraint,
            tangent,
            curvature,
        })
    }

    fn project(&self, vector: &mut Array1<f64>) {
        self.constraint.project(vector);
    }

    fn restore(&self, x: &mut Array1<f64>) {
        self.constraint.restore(x);
    }
}

fn plane_directions<S: PointSurface>(
    surface: &S,
    positions: ArrayView1<f64>,
    fix_com: bool,
) -> Result<Array2<f64>, SaddleError> {
    if !fix_com {
        return Ok(Array2::zeros((0, positions.len())));
    }
    let modes = surface.excluded_modes(positions)?;
    if modes.nrows() > 0 {
        if modes.ncols() != positions.len() {
            return Err(SaddleError::Shape(
                "PAFI rigid modes must match the path".into(),
            ));
        }
        return Ok(modes);
    }
    let n = positions.len();
    if n >= 6 && n.is_multiple_of(3) {
        return Ok(ohtst::uniform_translations(n / 3));
    }
    Ok(Array2::zeros((0, n)))
}

/// Windows a pass may run for each one it asks for, the dropped included.
const WINDOW_TRIES: usize = 4;

/// Times a chain that keeps no window halves [`PafiConfig::window`].
const WINDOW_HALVINGS: u32 = 4;

/// One overdamped chain on one plane, kept between passes.
struct Chain {
    x: Array1<f64>,
    previous: Option<Array1<f64>>,
    rng: Rng,
    values: Vec<f64>,
    psi_sum: f64,
    retained: Vec<Array1<f64>>,
    evaluations: usize,
    widest: f64,
    origin: (f64, f64),
    /// Minimum of the plane reached from its origin, for the jump test.
    minimum: Option<Array1<f64>>,
    /// Where the last window the jump test kept ended.
    kept: Option<Array1<f64>>,
    jumps: usize,
    /// Steps between two jump tests.
    window: usize,
}

impl Chain {
    fn new(plane: &Plane, seed: u64, config: &PafiConfig) -> Self {
        let mut rng = Rng::new(seed);
        let previous =
            (config.integrator == Integrator::LeimkuhlerMatthews).then(|| plane.noise(&mut rng));
        Self {
            x: plane.geometry.positions.clone(),
            previous,
            rng,
            values: Vec::new(),
            psi_sum: 0.0,
            retained: Vec::new(),
            evaluations: 0,
            widest: 0.0,
            origin: (f64::NAN, f64::NAN),
            minimum: None,
            kept: None,
            jumps: 0,
            window: config.window,
        }
    }

    /// `thermalize` steps, then `samples` more recorded `config.stride`
    /// apart in windows of `config.window`, each put to
    /// [`PafiConfig::max_jump`]. A dropped window is run again, from where
    /// the last kept window ended with fresh noise, or from the origin,
    /// thermalized again, when no window has been kept; the pass gives up
    /// after [`WINDOW_TRIES`] times the windows it asked for. A chain that
    /// has kept no window when its pass gives up leaves the basin of the
    /// path faster than one window. It halves its window and runs the
    /// pass again from the origin, thermalizing in windows as well, and
    /// goes on halving, the pass discarded and the chain thermalized again
    /// from the origin, while a pass drops more windows than it keeps, up
    /// to [`WINDOW_HALVINGS`] times: a long window that ends in the basin
    /// may still have spent time outside.
    fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        plane: &Plane,
        config: &PafiConfig,
        thermalize: usize,
        samples: usize,
    ) -> Result<(), SaddleError> {
        let goal = self.values.len() + samples;
        let shortest = (config.window >> WINDOW_HALVINGS).max(1);
        let mut thermalize = thermalize;
        loop {
            let mark = (
                self.values.len(),
                self.retained.len(),
                self.psi_sum,
                self.widest,
                self.jumps,
            );
            let kept = self.pass(surface, plane, config, thermalize, goal)?;
            if config.max_jump.is_none() || self.window <= shortest {
                return Ok(());
            }
            if self.window == config.window {
                if !self.values.is_empty() {
                    return Ok(());
                }
            } else if kept > self.jumps - mark.4 {
                return Ok(());
            } else {
                self.values.truncate(mark.0);
                self.retained.truncate(mark.1);
                self.psi_sum = mark.2;
                self.widest = mark.3;
                self.kept = None;
                self.x.assign(&plane.geometry.positions);
                if let Some(previous) = self.previous.as_mut() {
                    *previous = plane.noise(&mut self.rng);
                }
                thermalize = config.thermalize;
            }
            self.window = self.window.div_ceil(2).max(shortest);
        }
    }

    /// One pass of [`Self::run`] in windows of the current length, giving
    /// the windows it kept.
    fn pass<S: PointSurface>(
        &mut self,
        surface: &S,
        plane: &Plane,
        config: &PafiConfig,
        thermalize: usize,
        goal: usize,
    ) -> Result<usize, SaddleError> {
        let window = self.window;
        let windowed = window < config.window;
        let mut warm = thermalize;
        let mut kept = 0;
        let mut tries = WINDOW_TRIES * (goal - self.values.len()).div_ceil(window);
        if windowed {
            tries += WINDOW_TRIES * thermalize.div_ceil(window);
        }
        while self.values.len() < goal && tries > 0 {
            tries -= 1;
            let (heat, take) = if windowed && warm > 0 {
                (warm.min(window), 0)
            } else {
                (warm, (goal - self.values.len()).min(window))
            };
            let mark = (
                self.values.len(),
                self.retained.len(),
                self.psi_sum,
                self.widest,
            );
            self.advance(surface, plane, config, heat, take)?;
            warm -= heat;
            let Some(limit) = config.max_jump else {
                continue;
            };
            if !self.jumped(surface, plane, config, limit)? {
                self.kept = Some(self.x.clone());
                kept += 1;
                continue;
            }
            self.values.truncate(mark.0);
            self.retained.truncate(mark.1);
            self.psi_sum = mark.2;
            self.widest = mark.3;
            self.jumps += 1;
            match &self.kept {
                Some(checkpoint) => {
                    self.x.assign(checkpoint);
                    warm += heat;
                }
                None => {
                    self.x.assign(&plane.geometry.positions);
                    warm = config.thermalize;
                }
            }
            if let Some(previous) = self.previous.as_mut() {
                *previous = plane.noise(&mut self.rng);
            }
        }
        Ok(kept)
    }

    /// Whether the chain, relaxed in its plane, ends more than `limit` from
    /// the minimum of the plane, or does not settle.
    fn jumped<S: PointSurface>(
        &mut self,
        surface: &S,
        plane: &Plane,
        config: &PafiConfig,
        limit: f64,
    ) -> Result<bool, SaddleError> {
        if self.minimum.is_none() {
            let (minimum, evaluations, _) = relax_in_plane(
                surface,
                plane,
                plane.geometry.positions.clone(),
                config.jump_tol,
            )?;
            self.evaluations += evaluations;
            self.minimum = Some(minimum);
        }
        let (quenched, evaluations, settled) =
            relax_in_plane(surface, plane, self.x.clone(), config.jump_tol)?;
        self.evaluations += evaluations;
        let minimum = self.minimum.as_ref().expect("minimum");
        Ok(!settled || largest_atom(&(&quenched - minimum)) > limit)
    }

    fn advance<S: PointSurface>(
        &mut self,
        surface: &S,
        plane: &Plane,
        config: &PafiConfig,
        thermalize: usize,
        samples: usize,
    ) -> Result<(), SaddleError> {
        let geometry = plane.geometry;
        let n = self.x.len();
        let h = config.mobility * config.time_step;
        let retain_gap = config.samples.div_ceil(config.retain.max(1)).max(1);
        for step in 0..thermalize + config.stride * samples {
            let (energy, gradient) = surface.eval(self.x.view())?;
            if gradient.len() != n || !gradient.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::Shape(
                    "PAFI gradient must match the path".into(),
                ));
            }
            if self.evaluations == 0 {
                self.origin = (energy, gradient.dot(&plane.tangent));
            }
            self.evaluations += 1;
            if step >= thermalize && (step - thermalize).is_multiple_of(config.stride) {
                let (value, psi) = projected_gradient(
                    &gradient,
                    &self.x,
                    &geometry.positions,
                    &plane.tangent,
                    &plane.curvature,
                );
                if !value.is_finite() || !psi.is_finite() {
                    return Err(SaddleError::NonFinite("PAFI projected gradient"));
                }
                let recorded = self.values.len();
                self.values.push(value);
                self.psi_sum += psi;
                self.widest = self
                    .widest
                    .max(largest_atom(&(&self.x - &geometry.positions)));
                if config.retain > 0
                    && recorded.is_multiple_of(retain_gap)
                    && self.retained.len() < config.retain
                {
                    self.retained.push(self.x.clone());
                }
            }
            let mut drift = gradient;
            plane.project(&mut drift);
            let fresh = plane.noise(&mut self.rng);
            self.x.scaled_add(-h, &drift);
            match self.previous.as_mut() {
                Some(previous) => {
                    let scale = (0.5 * h * config.thermal).sqrt();
                    self.x.scaled_add(scale, previous);
                    self.x.scaled_add(scale, &fresh);
                    *previous = fresh;
                }
                None => self.x.scaled_add((2.0 * h * config.thermal).sqrt(), &fresh),
            }
            plane.restore(&mut self.x);
        }
        Ok(())
    }

    fn average(&self, config: &PafiConfig, plane: &Plane) -> HyperplaneAverage {
        let (gradient, error, tau, n_eff) = windowed_block_mean(&self.values, config.blocks);
        HyperplaneAverage {
            plane: plane.constraint.plane().clone(),
            gradient,
            error,
            psi: self.psi_sum / self.values.len().max(1) as f64,
            samples: self.values.len(),
            tau,
            n_eff,
            evaluations: self.evaluations,
            retained: self.retained.clone(),
            energy: self.origin.0,
            zero: self.origin.1,
            widest: self.widest,
            jumps: self.jumps,
        }
    }
}

impl Plane<'_> {
    fn noise(&self, rng: &mut Rng) -> Array1<f64> {
        let mut noise = Array1::from_shape_fn(self.tangent.len(), |_| rng.normal());
        self.project(&mut noise);
        noise
    }
}

/// FIRE of Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006), on the force
/// projected into `plane`, from `x` until no atom feels more than `tol`
/// there. Returns where it stopped, the force evaluations spent, and
/// whether it got there within [`RELAX_STEPS`].
fn relax_in_plane<S: PointSurface>(
    surface: &S,
    plane: &Plane,
    mut x: Array1<f64>,
    tol: f64,
) -> Result<(Array1<f64>, usize, bool), SaddleError> {
    const MAX_MOVE: f64 = 0.1;
    let mut velocity = Array1::<f64>::zeros(x.len());
    let (mut dt, mut alpha, mut positive) = (0.05, 0.1, 0usize);
    for step in 0..RELAX_STEPS {
        let (_, gradient) = surface.eval(x.view())?;
        if gradient.len() != x.len() || !gradient.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::Shape(
                "PAFI gradient must match the path".into(),
            ));
        }
        let mut force = -gradient;
        plane.project(&mut force);
        if largest_atom(&force) <= tol {
            return Ok((x, step + 1, true));
        }
        if force.dot(&velocity) > 0.0 {
            let mix = alpha * norm(&velocity) / norm(&force);
            velocity *= 1.0 - alpha;
            velocity.scaled_add(mix, &force);
            if positive > 5 {
                dt = (dt * 1.1_f64).min(0.5);
                alpha *= 0.99;
            }
            positive += 1;
        } else {
            velocity.fill(0.0);
            dt *= 0.5;
            alpha = 0.1;
            positive = 0;
        }
        velocity.scaled_add(dt, &force);
        plane.project(&mut velocity);
        let mut move_by = &velocity * dt;
        let widest = largest_atom(&move_by);
        if widest > MAX_MOVE {
            move_by *= MAX_MOVE / widest;
        }
        x += &move_by;
        plane.restore(&mut x);
    }
    Ok((x, RELAX_STEPS, false))
}

const RELAX_STEPS: usize = 10_000;

/// Integral of `∇E · dX/ds` with the system held on the path.
///
/// This is the zero-temperature limit of the PAFI integral: `ψ = 0` and
/// there is no thermal average. On a converged minimum-energy path it
/// reproduces the potential barrier.
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
    let arcs = dense_arcs(arcs, 24);
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

/// Eight samples on every span. One trapezoid per image is not enough
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

fn projected_gradient(
    gradient: &Array1<f64>,
    x: &Array1<f64>,
    origin: &Array1<f64>,
    tangent: &Array1<f64>,
    curvature: &Array1<f64>,
) -> (f64, f64) {
    let psi = (x - origin).dot(curvature);
    let parallel = gradient.dot(tangent);
    (parallel * (1.0 - psi), psi)
}

fn difference(end: &Array1<f64>, start: &Array1<f64>, ds: f64) -> Array1<f64> {
    (end - start) / ds
}

fn atom_width(len: usize) -> usize {
    if len.is_multiple_of(3) { 3 } else { 1 }
}

/// Atoms whose summed squared steps reach `share` of the total, the
/// largest first, in ascending order.
fn active_atoms(steps: &[Array1<f64>], width: usize, share: f64) -> Vec<usize> {
    let atoms = steps.first().map_or(0, |step| step.len() / width);
    if share >= 1.0 {
        return (0..atoms).collect();
    }
    let mut weight = vec![0.0; atoms];
    for step in steps {
        for (atom, slot) in weight.iter_mut().enumerate() {
            *slot += (0..width)
                .map(|c| step[atom * width + c].powi(2))
                .sum::<f64>();
        }
    }
    let total: f64 = weight.iter().sum();
    let mut order: Vec<usize> = (0..atoms).collect();
    order.sort_by(|a, b| weight[*b].total_cmp(&weight[*a]));
    let mut carried = 0.0;
    let mut active = Vec::new();
    for atom in order {
        if carried >= share * total {
            break;
        }
        carried += weight[atom];
        active.push(atom);
    }
    active.sort_unstable();
    active
}

fn largest_atom(vector: &Array1<f64>) -> f64 {
    let width = atom_width(vector.len());
    vector
        .exact_chunks(width)
        .into_iter()
        .map(|atom| atom.dot(&atom).sqrt())
        .fold(0.0, f64::max)
}

fn norm(vector: &Array1<f64>) -> f64 {
    vector.dot(vector).sqrt()
}

/// Derivative along the arc at every frame of a quantity with
/// `differences[k]` between frames `k` and `k + 1` a distance `steps[k]`
/// apart: that of the parabola through the frame and its neighbours, or
/// through an end frame and the next two.
fn arc_derivative(differences: &[Array1<f64>], steps: &[f64]) -> Vec<Array1<f64>> {
    let last = differences.len();
    if last == 1 {
        return vec![&differences[0] / steps[0]; 2];
    }
    let end = |near: &Array1<f64>, far: &Array1<f64>, h1: f64, h2: f64| {
        (near * (h2 * (2.0 * h1 + h2)) - far * (h1 * h1)) / (h1 * h2 * (h1 + h2))
    };
    let mut derivatives = Vec::with_capacity(last + 1);
    derivatives.push(end(&differences[0], &differences[1], steps[0], steps[1]));
    for k in 1..last {
        let (before, after) = (steps[k - 1], steps[k]);
        derivatives.push(
            (&differences[k] * (before * before) + &differences[k - 1] * (after * after))
                / (before * after * (before + after)),
        );
    }
    derivatives.push(end(
        &differences[last - 1],
        &differences[last - 2],
        steps[last - 1],
        steps[last - 2],
    ));
    derivatives
}
