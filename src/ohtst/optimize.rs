//! The maximum of the plane free energy, on any manifold.
//!
//! The plane is the packed point `x = [p; ν]` of `R^n × S^{n-1}`. An
//! [`rgmin::manifold::Manifold`] on that packing supplies the metric (its
//! `egrad2rgrad` turns the Euclidean blocks of [`PlaneSample`] into a
//! Riemannian gradient), the retraction and the transport. The optimizer
//! owns the sampling, the step and the statistics, so a different product
//! metric changes nothing here.
//!
//! The step is rgmin's quick-min displacement with the point block and the
//! normal block projected separately, the velocity-Verlet scheme with
//! velocity projection of Jóhannesson and Jónsson, Eqs. 5–10, and each
//! block is capped as there, by a translation length and a rotation angle.
//! The force given to quick-min is the Riemannian gradient of `A`, so the
//! plane climbs to the maximum of `A`, the minimum of `-A`.
//!
//! A sample counts as converged when the translational and the rotational
//! blocks of that gradient are both within [`PlaneOptimizerConfig::confidence`]
//! standard errors of zero; [`PlaneOptimizerConfig::patience`] such
//! samples in a row end the run. The sample size follows the norm test of
//! Byrd, Chin, Nocedal and Wu, Math. Program. 134, 127 (2012),
//! doi:10.1007/s10107-012-0572-5: when the variance of the mean gradient
//! exceeds `θ² |g|²`, the next sample is longer by that ratio, up to
//! [`PlaneOptimizerConfig::max_records`]. Each sample starts from the last
//! configuration of the one before, projected onto the new plane.

use std::ops::Range;

use ndarray::{Array1, s};
use rgmin::manifold::Manifold;
use rgmin::{QuickMinState, quickmin_displacement};

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::ohtst::plane::Hyperplane;
use crate::ohtst::sample::{LangevinConfig, PlaneSample, mean_and_error, sample_plane};

/// Settings of [`PlaneOptimizer`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PlaneOptimizerConfig {
    /// Sampling of each plane. Its records are the first sample size.
    pub sampler: LangevinConfig,
    /// Burn-in of a sample that starts from the previous plane's last
    /// configuration.
    pub warm_burn_in: usize,
    /// Largest sample, in records.
    pub max_records: usize,
    /// `θ` of the norm test. Zero keeps the sample size fixed.
    pub norm_test: f64,
    /// Quick-min time step.
    pub time_step: f64,
    /// Longest move of the point in one step, in mass-weighted length.
    pub max_translation: f64,
    /// Largest turn of the normal in one step, in radians.
    pub max_rotation: f64,
    /// Move the plane along its normal.
    pub translate: bool,
    /// Turn the normal. Without translation the plane turns about its
    /// point.
    pub rotate: bool,
    /// After each step slide the point inside the new plane to the
    /// projection of the last sample's centroid, so the next turn pivots
    /// there. Applies only when [`Self::translate`] is set.
    pub recenter: bool,
    /// Standard errors within which a gradient block counts as zero.
    pub confidence: f64,
    /// Length below which a gradient block counts as zero whatever its
    /// error. A surface whose normal force has no variance on the optimal
    /// plane, such as a separable quadratic, needs it.
    pub tolerance: f64,
    /// Converged samples in a row that end [`PlaneOptimizer::run`].
    pub patience: usize,
    /// Iterations of [`PlaneOptimizer::run`].
    pub max_iterations: usize,
}

impl Default for PlaneOptimizerConfig {
    fn default() -> Self {
        Self {
            sampler: LangevinConfig::default(),
            warm_burn_in: 100,
            max_records: 16_000,
            norm_test: 0.5,
            time_step: 0.3,
            max_translation: 0.1,
            max_rotation: 0.05,
            translate: true,
            rotate: true,
            recenter: true,
            confidence: 2.0,
            tolerance: 0.0,
            patience: 3,
            max_iterations: 100,
        }
    }
}

impl PlaneOptimizerConfig {
    /// Set [`Self::sampler`].
    #[must_use]
    pub fn with_sampler(mut self, sampler: LangevinConfig) -> Self {
        self.sampler = sampler;
        self
    }

    /// Set [`Self::warm_burn_in`].
    #[must_use]
    pub fn with_warm_burn_in(mut self, warm_burn_in: usize) -> Self {
        self.warm_burn_in = warm_burn_in;
        self
    }

    /// Set [`Self::max_records`].
    #[must_use]
    pub fn with_max_records(mut self, max_records: usize) -> Self {
        self.max_records = max_records;
        self
    }

    /// Set [`Self::norm_test`].
    #[must_use]
    pub fn with_norm_test(mut self, norm_test: f64) -> Self {
        self.norm_test = norm_test;
        self
    }

    /// Set [`Self::time_step`].
    #[must_use]
    pub fn with_time_step(mut self, time_step: f64) -> Self {
        self.time_step = time_step;
        self
    }

    /// Set [`Self::max_translation`].
    #[must_use]
    pub fn with_max_translation(mut self, max_translation: f64) -> Self {
        self.max_translation = max_translation;
        self
    }

    /// Set [`Self::max_rotation`].
    #[must_use]
    pub fn with_max_rotation(mut self, max_rotation: f64) -> Self {
        self.max_rotation = max_rotation;
        self
    }

    /// Set [`Self::translate`].
    #[must_use]
    pub fn with_translate(mut self, translate: bool) -> Self {
        self.translate = translate;
        self
    }

    /// Set [`Self::rotate`].
    #[must_use]
    pub fn with_rotate(mut self, rotate: bool) -> Self {
        self.rotate = rotate;
        self
    }

    /// Set [`Self::recenter`].
    #[must_use]
    pub fn with_recenter(mut self, recenter: bool) -> Self {
        self.recenter = recenter;
        self
    }

    /// Set [`Self::confidence`].
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set [`Self::tolerance`].
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set [`Self::patience`].
    #[must_use]
    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Set [`Self::max_iterations`].
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    fn check(&self) -> Result<(), SaddleError> {
        self.sampler.check()?;
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if !(positive(self.time_step)
            && positive(self.max_translation)
            && positive(self.max_rotation)
            && positive(self.confidence))
        {
            return Err(SaddleError::Invalid(
                "plane optimizer step, caps and confidence must be finite and positive".into(),
            ));
        }
        if !(self.norm_test.is_finite()
            && self.norm_test >= 0.0
            && self.tolerance.is_finite()
            && self.tolerance >= 0.0)
        {
            return Err(SaddleError::Invalid(
                "norm-test θ and tolerance must be finite and non-negative".into(),
            ));
        }
        if !(self.translate || self.rotate) {
            return Err(SaddleError::Invalid(
                "the plane optimizer has to translate, rotate, or both".into(),
            ));
        }
        if self.patience == 0 || self.max_records < self.sampler.records {
            return Err(SaddleError::Invalid(
                "patience must be positive and max_records at least the first sample".into(),
            ));
        }
        Ok(())
    }
}

/// One iteration of [`PlaneOptimizer::step`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PlaneStep {
    /// Zero-based iteration.
    pub iteration: usize,
    /// The sample at the plane before the move.
    pub sample: PlaneSample,
    /// Length of the point block of the Riemannian gradient.
    pub translation: f64,
    /// Root-sum-square block error of that block.
    pub translation_error: f64,
    /// Length of the normal block of the Riemannian gradient.
    pub rotation: f64,
    /// Root-sum-square block error of that block.
    pub rotation_error: f64,
    /// Both active blocks are within the confidence.
    pub within_error: bool,
    /// Samples within the confidence in a row, this one included.
    pub streak: usize,
    /// Length of the point move. Zero once the run has converged.
    pub moved: f64,
    /// Tangent length of the normal turn, in radians.
    pub turned: f64,
    /// The plane after the move.
    pub plane: Hyperplane,
}

/// The light record of one iteration kept by [`PlaneOptimizer`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PlaneTrace {
    /// The sampled plane.
    pub plane: Hyperplane,
    /// Mean force along the normal.
    pub normal_force: f64,
    /// Its block error.
    pub normal_force_error: f64,
    /// See [`PlaneStep::rotation`].
    pub rotation: f64,
    /// See [`PlaneStep::rotation_error`].
    pub rotation_error: f64,
    /// Records in the sample.
    pub records: usize,
}

/// The converged plane of [`PlaneOptimizer::run`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PlaneOptimum {
    /// The plane of the last converged sample.
    pub plane: Hyperplane,
    /// That sample.
    pub sample: PlaneSample,
    /// Iterations taken.
    pub iterations: usize,
    /// Surface evaluations over the run.
    pub evaluations: usize,
    /// Every iteration.
    pub trace: Vec<PlaneTrace>,
}

/// Stochastic ascent of the plane free energy on a manifold `M` over the
/// packed `[p; ν]`. The host can drive [`Self::step`] or call [`Self::run`].
pub struct PlaneOptimizer<M> {
    manifold: M,
    config: PlaneOptimizerConfig,
    n: usize,
    x: Array1<f64>,
    quickmin: QuickMinState,
    start: Option<Array1<f64>>,
    records: usize,
    iteration: usize,
    streak: usize,
    evaluations: usize,
    trace: Vec<PlaneTrace>,
}

impl<M: Manifold> PlaneOptimizer<M> {
    /// Start at `plane`. `manifold` must accept the packed length `2n`.
    pub fn new(
        manifold: M,
        plane: &Hyperplane,
        config: PlaneOptimizerConfig,
    ) -> Result<Self, SaddleError> {
        config.check()?;
        let n = plane.dim();
        manifold.required_dim(2 * n).map_err(|want| {
            SaddleError::Shape(format!(
                "the manifold takes {want} coordinates and the packed plane has {}",
                2 * n
            ))
        })?;
        let mut quickmin = QuickMinState::reference(2 * n, config.time_step);
        quickmin.cell_at = Some(n);
        Ok(Self {
            manifold,
            records: config.sampler.records,
            config,
            n,
            x: plane.packed(),
            quickmin,
            start: None,
            iteration: 0,
            streak: 0,
            evaluations: 0,
            trace: Vec::new(),
        })
    }

    /// The current plane.
    pub fn plane(&self) -> Result<Hyperplane, SaddleError> {
        Hyperplane::from_packed(self.x.view())
    }

    /// The manifold.
    pub fn manifold(&self) -> &M {
        &self.manifold
    }

    /// Records of the next sample.
    pub fn records(&self) -> usize {
        self.records
    }

    /// Surface evaluations so far.
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Sample the current plane, test it, and move unless the run has
    /// converged.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<PlaneStep, SaddleError> {
        let plane = self.plane()?;
        let mut sampler = self.config.sampler.clone();
        sampler.records = self.records;
        sampler.seed = sampler.seed.wrapping_add(self.iteration as u64);
        if self.start.is_some() {
            sampler.burn_in = self.config.warm_burn_in;
        }
        let sample = sample_plane(
            surface,
            &plane,
            self.start.as_ref().map(|q| q.view()),
            &sampler,
        )?;
        self.evaluations += sample.evaluations;
        let n = self.n;
        let gradient = self.riemannian(&sample.packed_gradient());
        let rows: Vec<Array1<f64>> = sample
            .block_gradients
            .rows()
            .into_iter()
            .map(|row| self.riemannian(&row.to_owned()))
            .collect();
        let (translation, translation_error) = block_statistic(&gradient, &rows, 0..n);
        let (rotation, rotation_error) = block_statistic(&gradient, &rows, n..2 * n);
        let zero = |value: f64, error: f64| {
            value <= self.config.confidence * error || value <= self.config.tolerance
        };
        let within_error = (!self.config.translate || zero(translation, translation_error))
            && (!self.config.rotate || zero(rotation, rotation_error));
        self.streak = if within_error { self.streak + 1 } else { 0 };
        if self.config.norm_test > 0.0 {
            let variance = translation_error.powi(2) + rotation_error.powi(2);
            let signal = self.config.norm_test.powi(2) * (translation.powi(2) + rotation.powi(2));
            if variance > signal {
                let grown = if signal > 0.0 {
                    (self.records as f64 * variance / signal).ceil()
                } else {
                    f64::INFINITY
                };
                self.records =
                    (grown.min(self.config.max_records as f64) as usize).max(self.records);
            }
        }
        self.trace.push(PlaneTrace {
            plane: plane.clone(),
            normal_force: sample.normal_force,
            normal_force_error: sample.normal_force_error,
            rotation,
            rotation_error,
            records: sample.records,
        });
        let (moved, turned) = if self.streak >= self.config.patience {
            (0.0, 0.0)
        } else {
            self.advance(&gradient, &sample)?
        };
        self.start = Some(sample.last.clone());
        let step = PlaneStep {
            iteration: self.iteration,
            sample,
            translation,
            translation_error,
            rotation,
            rotation_error,
            within_error,
            streak: self.streak,
            moved,
            turned,
            plane: self.plane()?,
        };
        self.iteration += 1;
        Ok(step)
    }

    /// Step until [`PlaneOptimizerConfig::patience`] samples in a row are
    /// within the confidence. Running out of iterations is an error.
    pub fn run<S: PointSurface>(&mut self, surface: &S) -> Result<PlaneOptimum, SaddleError> {
        let mut last = None;
        for _ in 0..self.config.max_iterations {
            let step = self.step(surface)?;
            if step.streak >= self.config.patience {
                return Ok(PlaneOptimum {
                    plane: step.plane,
                    sample: step.sample,
                    iterations: self.iteration,
                    evaluations: self.evaluations,
                    trace: self.trace.clone(),
                });
            }
            last = Some((
                step.translation,
                step.translation_error,
                step.rotation,
                step.rotation_error,
            ));
        }
        let (t, te, r, re) = last.unwrap_or((f64::NAN, f64::NAN, f64::NAN, f64::NAN));
        Err(SaddleError::Solver(format!(
            "the plane did not converge in {} iterations: translation {t:.3e} ± {te:.3e}, rotation {r:.3e} ± {re:.3e}",
            self.config.max_iterations
        )))
    }

    fn riemannian(&self, egrad: &Array1<f64>) -> Array1<f64> {
        let mut egrad = egrad.clone();
        self.mask(&mut egrad);
        let mut rgrad = self.manifold.egrad2rgrad(&self.x, &egrad);
        self.mask(&mut rgrad);
        rgrad
    }

    fn mask(&self, v: &mut Array1<f64>) {
        let n = self.n;
        if !self.config.translate {
            v.slice_mut(s![..n]).fill(0.0);
        }
        if !self.config.rotate {
            v.slice_mut(s![n..]).fill(0.0);
        }
    }

    fn advance(
        &mut self,
        force: &Array1<f64>,
        sample: &PlaneSample,
    ) -> Result<(f64, f64), SaddleError> {
        let n = self.n;
        let mut dx = quickmin_displacement(&mut self.quickmin, force);
        let mut lengths = [0.0; 2];
        for (k, (range, cap)) in [
            (0..n, self.config.max_translation),
            (n..2 * n, self.config.max_rotation),
        ]
        .into_iter()
        .enumerate()
        {
            let length = dx
                .slice(s![range.clone()])
                .dot(&dx.slice(s![range.clone()]))
                .sqrt();
            if length > cap {
                let factor = cap / length;
                dx.slice_mut(s![range.clone()]).mapv_inplace(|v| v * factor);
                self.quickmin
                    .vel
                    .slice_mut(s![range])
                    .mapv_inplace(|v| v * factor);
                lengths[k] = cap;
            } else {
                lengths[k] = length;
            }
        }
        let mut next = self.manifold.retract(&self.x, &dx);
        let mut velocity = self.manifold.transport(&self.x, &next, &self.quickmin.vel);
        if self.config.recenter && self.config.translate {
            let centred = Hyperplane::from_packed(next.view())?
                .recentered(sample.centroid.view())
                .packed();
            velocity = self.manifold.transport(&next, &centred, &velocity);
            next = centred;
        }
        if !next.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("plane after the step"));
        }
        self.x = next;
        self.quickmin.vel = velocity;
        Ok((lengths[0], lengths[1]))
    }
}

fn block_statistic(mean: &Array1<f64>, rows: &[Array1<f64>], range: Range<usize>) -> (f64, f64) {
    let value = mean
        .slice(s![range.clone()])
        .dot(&mean.slice(s![range.clone()]))
        .sqrt();
    let mut variance = 0.0;
    for i in range {
        let column: Vec<f64> = rows.iter().map(|row| row[i]).collect();
        variance += mean_and_error(&column).1.powi(2);
    }
    (value, variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayView1, array};

    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/common/plane_manifold.rs"
    ));
    use plane_manifold::PointSphereFixture;

    /// `V = -½ a x² + ½ b y² + ½ c z²`: the free energy of a plane is
    /// `(kT/2) ln|νᵀH⁻¹ν| + (ν · p)² / (2 νᵀH⁻¹ν)`, a maximum at `ν = e_x`
    /// through the origin.
    struct HarmonicSaddle;

    impl PointSurface for HarmonicSaddle {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let g = array![-q[0], 2.0 * q[1], 3.0 * q[2]];
            Ok((
                0.5 * (-q[0] * q[0] + 2.0 * q[1] * q[1] + 3.0 * q[2] * q[2]),
                g,
            ))
        }
    }

    fn config() -> PlaneOptimizerConfig {
        PlaneOptimizerConfig::default()
            .with_sampler(
                LangevinConfig::default()
                    .with_thermal(0.2)
                    .with_time_step(0.3)
                    .with_burn_in(200)
                    .with_records(1000),
            )
            .with_time_step(0.8)
            .with_max_translation(0.1)
            .with_max_rotation(0.1)
            .with_tolerance(1.0e-6)
            .with_max_iterations(200)
    }

    #[test]
    fn harmonic_saddle_plane_is_found() {
        let start = Hyperplane::new(array![0.3, 0.1, -0.2], array![1.0, 0.3, -0.25]).unwrap();
        let mut optimizer =
            PlaneOptimizer::new(PointSphereFixture::new(3), &start, config()).unwrap();
        let optimum = optimizer.run(&HarmonicSaddle).unwrap();
        let nu = optimum.plane.normal();
        let tilt = (nu[1] * nu[1] + nu[2] * nu[2]).sqrt();
        let offset = optimum.plane.distance(array![0.0, 0.0, 0.0].view());
        assert!(tilt < 1.0e-4, "tilt {tilt}");
        assert!(offset.abs() < 1.0e-4, "offset {offset}");
        assert!(optimum.trace.len() == optimum.iterations);
    }

    #[test]
    fn rotation_alone_turns_about_the_point() {
        let start = Hyperplane::new(array![0.0, 0.4, 0.0], array![1.0, 0.2, 0.0]).unwrap();
        let mut optimizer = PlaneOptimizer::new(
            PointSphereFixture::new(3),
            &start,
            config().with_translate(false),
        )
        .unwrap();
        let optimum = optimizer.run(&HarmonicSaddle).unwrap();
        assert_eq!(optimum.plane.point(), start.point());
        assert!(optimum.plane.normal()[1].abs() < 1.0e-4);
        assert!(PlaneOptimizer::new(PointSphereFixture::new(2), &start, config()).is_err());
    }
}
