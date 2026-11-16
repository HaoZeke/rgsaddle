//! The reactant normalization by thermodynamic integration over parallel
//! planes.
//!
//! For a unit normal `n` and the planes `H(ξ)` through `x_R + ξ n`,
//! `Z_R = ∫ Z_H(ξ) dξ` over the reactant basin, and the plane free energy
//! `A_R(ξ) = -kT ln Z_H(ξ)` has slope `⟨n · ∇V⟩_ξ`. Hence
//!
//! `Z_H(0) / Z_R = [∫ exp(-(A_R(ξ) - A_R(0)) / kT) dξ]⁻¹`,
//!
//! Eq. (7.3) of the OH-TST design survey. The planes sit at evenly spaced
//! `ξ` across `±width` harmonic standard deviations, with
//! `σ_ξ² = kT nᵀ H_R⁻¹ n` from conjugate gradients on Hessian actions at
//! the minimum. A polynomial fitted to the sampled mean forces, weighted
//! by their errors, gives `A_R`, and the delta method carries the fit
//! covariance into the error. The planes must stay in the basin; at
//! `width = 4` the neglected tails are of order `e⁻⁸` for a harmonic
//! basin.
//!
//! This replaces the crossing estimator of Jóhannesson and Jónsson,
//! J. Chem. Phys. 115, 9644 (2001), Eq. 22, kept as
//! [`ReactantDensity::crossing_density`](crate::ohtst::ReactantDensity::crossing_density):
//! each crossing there is weighted by `1/|ξ̇|`, and under the flux-weighted
//! velocity distribution the second moment of that weight diverges
//! logarithmically at slow crossings.

use std::f64::consts::TAU;

use ndarray::ArrayView1;

use super::anharmonic::hessian_action;
use super::control::conjugate_gradients;
use super::plane::Hyperplane;
use super::rate::PlaneDensity;
use super::sample::{LangevinConfig, held_directions, sample_plane};
use crate::{PointSurface, SaddleError};

/// Intervals of the Simpson rule over the fitted free energy.
const INTERVALS: usize = 2000;

/// Parallel-plane integration of the reactant normalization.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BasinConfig {
    /// Sampler for every plane. Plane `i` uses the seed plus `i`.
    pub sampler: LangevinConfig,
    /// Number of planes, at least two more than [`Self::degree`].
    pub planes: usize,
    /// Half-width of the planes, in harmonic standard deviations of `ξ`.
    pub width: f64,
    /// Degree of the polynomial fitted to the mean force; at least one.
    pub degree: usize,
    /// Gradient-difference step for Hessian actions at the minimum when
    /// the surface has no analytic one.
    pub hvp_step: f64,
    /// Relative residual of the solve for `H_R⁻¹ n`.
    pub tolerance: f64,
}

impl Default for BasinConfig {
    fn default() -> Self {
        Self {
            sampler: LangevinConfig::default(),
            planes: 11,
            width: 4.0,
            degree: 5,
            hvp_step: 1.0e-4,
            tolerance: 1.0e-10,
        }
    }
}

impl BasinConfig {
    /// Set [`Self::sampler`].
    #[must_use]
    pub fn with_sampler(mut self, sampler: LangevinConfig) -> Self {
        self.sampler = sampler;
        self
    }

    /// Set [`Self::planes`].
    #[must_use]
    pub fn with_planes(mut self, planes: usize) -> Self {
        self.planes = planes;
        self
    }

    /// Set [`Self::width`].
    #[must_use]
    pub fn with_width(mut self, width: f64) -> Self {
        self.width = width;
        self
    }

    /// Set [`Self::degree`].
    #[must_use]
    pub fn with_degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Set [`Self::hvp_step`].
    #[must_use]
    pub fn with_hvp_step(mut self, hvp_step: f64) -> Self {
        self.hvp_step = hvp_step;
        self
    }

    /// Set [`Self::tolerance`].
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    fn check(&self) -> Result<(), SaddleError> {
        self.sampler.check()?;
        if self.degree == 0 || self.planes < self.degree + 2 {
            return Err(SaddleError::Invalid(
                "the basin fit needs a positive degree and two more planes than the degree".into(),
            ));
        }
        if !(self.width.is_finite() && self.width > 0.0) {
            return Err(SaddleError::Invalid(
                "the basin width must be finite and positive".into(),
            ));
        }
        if !(self.hvp_step.is_finite() && self.hvp_step > 0.0) {
            return Err(SaddleError::Invalid(
                "the basin difference step must be finite and positive".into(),
            ));
        }
        if !(self.tolerance > 0.0 && self.tolerance < 1.0) {
            return Err(SaddleError::Invalid(
                "the basin solve tolerance must lie in (0, 1)".into(),
            ));
        }
        Ok(())
    }
}

/// `Z_H / Z_R` on the plane through the minimum, from parallel planes.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BasinDensity {
    /// The plane through the minimum, with the integration normal.
    pub plane: Hyperplane,
    /// The sampling temperature `kT`.
    pub thermal: f64,
    /// `Z_H(0) / Z_R` per unit mass-weighted length along the normal.
    pub density: f64,
    /// Standard error of [`Self::density`].
    pub density_error: f64,
    /// The harmonic standard deviation `σ_ξ` of `ξ`.
    pub spread: f64,
    /// The harmonic density `1 / (√(2π) σ_ξ)`.
    pub harmonic_density: f64,
    /// Offsets `ξ` of the sampled planes.
    pub offsets: Vec<f64>,
    /// Mean forces `⟨n · ∇V⟩` on them.
    pub slopes: Vec<f64>,
    /// Block standard errors of [`Self::slopes`].
    pub slope_errors: Vec<f64>,
    /// Coefficients of the fitted slope in powers of `ξ / σ_ξ`.
    pub coefficients: Vec<f64>,
    /// Chi-square of the fit per degree of freedom. Above one, the fit
    /// covariance is scaled up by it.
    pub reduced_chi_square: f64,
    /// Surface evaluations, Hessian actions included.
    pub evaluations: usize,
}

impl PlaneDensity for BasinDensity {
    fn plane(&self) -> &Hyperplane {
        &self.plane
    }

    fn thermal(&self) -> f64 {
        self.thermal
    }

    fn density(&self) -> (f64, f64) {
        (self.density, self.density_error)
    }
}

/// Integrate the reactant normalization over planes normal to `normal`
/// around `minimum`. Excluded modes are projected out of `normal` when the
/// sampler holds them.
pub fn basin_density<S: PointSurface>(
    surface: &S,
    minimum: ArrayView1<f64>,
    normal: ArrayView1<f64>,
    config: &BasinConfig,
) -> Result<BasinDensity, SaddleError> {
    config.check()?;
    let n = minimum.len();
    if normal.len() != n {
        return Err(SaddleError::Shape(
            "the basin normal differs in length from the minimum".into(),
        ));
    }
    let excluded = held_directions(surface, minimum, None, config.sampler.hold_excluded)?;
    let mut direction = normal.to_owned();
    for t in &excluded {
        let c = t.dot(&direction);
        direction.scaled_add(-c, t);
    }
    let length = direction.dot(&direction).sqrt();
    if length.is_nan() || length <= 1.0e-12 {
        return Err(SaddleError::Invalid(
            "the basin normal lies in the excluded modes".into(),
        ));
    }
    direction /= length;
    let mut evaluations = 0;
    let solved = conjugate_gradients(
        |v| hessian_action(surface, minimum, v, config.hvp_step, &mut evaluations),
        &direction,
        &excluded,
        config.tolerance,
    )?;
    let compliance = solved.map_or(f64::NAN, |y| y.dot(&direction));
    if !(compliance.is_finite() && compliance > 0.0) {
        return Err(SaddleError::Invalid(
            "the Hessian at the minimum is not positive along the basin normal".into(),
        ));
    }
    let kt = config.sampler.thermal;
    let spread = (kt * compliance).sqrt();
    let planes = config.planes;
    let mut offsets = Vec::with_capacity(planes);
    let mut slopes = Vec::with_capacity(planes);
    let mut slope_errors = Vec::with_capacity(planes);
    for i in 0..planes {
        let x = config.width * (2.0 * i as f64 / (planes - 1) as f64 - 1.0);
        let xi = x * spread;
        let mut point = minimum.to_owned();
        point.scaled_add(xi, &direction);
        let plane = Hyperplane::new(point, direction.clone())?;
        let sampler = config
            .sampler
            .clone()
            .with_seed(config.sampler.seed.wrapping_add(i as u64));
        let sample = sample_plane(surface, &plane, None, &sampler)?;
        evaluations += sample.evaluations;
        offsets.push(xi);
        slopes.push(-sample.normal_force);
        slope_errors.push(sample.normal_force_error);
    }
    if slope_errors.iter().any(|e| !(e.is_finite() && *e > 0.0)) {
        return Err(SaddleError::Invalid(
            "a basin plane returned no error on its mean force".into(),
        ));
    }

    let terms = config.degree + 1;
    let scaled: Vec<f64> = offsets.iter().map(|xi| xi / spread).collect();
    let mut normal_matrix = vec![0.0; terms * terms];
    let mut rhs = vec![0.0; terms];
    for ((x, slope), error) in scaled.iter().zip(&slopes).zip(&slope_errors) {
        let weight = error.powi(-2);
        for j in 0..terms {
            rhs[j] += weight * x.powi(j as i32) * slope;
            for k in 0..terms {
                normal_matrix[j * terms + k] += weight * x.powi((j + k) as i32);
            }
        }
    }
    let covariance = invert(&normal_matrix, terms)?;
    let coefficients: Vec<f64> = (0..terms)
        .map(|j| (0..terms).map(|k| covariance[j * terms + k] * rhs[k]).sum())
        .collect();
    let fitted = |x: f64| -> f64 { (0..terms).map(|k| coefficients[k] * x.powi(k as i32)).sum() };
    let chi_square: f64 = scaled
        .iter()
        .zip(&slopes)
        .zip(&slope_errors)
        .map(|((x, slope), error)| ((slope - fitted(*x)) / error).powi(2))
        .sum();
    let reduced_chi_square = chi_square / (planes - terms) as f64;
    let inflation = reduced_chi_square.max(1.0);

    let rise = |x: f64| -> f64 {
        spread
            * (0..terms)
                .map(|k| coefficients[k] * x.powi(k as i32 + 1) / (k + 1) as f64)
                .sum::<f64>()
    };
    let h = 2.0 * config.width / INTERVALS as f64;
    let mut integral = 0.0;
    let mut moments = vec![0.0; terms];
    for i in 0..=INTERVALS {
        let x = -config.width + i as f64 * h;
        let weight = if i == 0 || i == INTERVALS {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        let value = weight * (-rise(x) / kt).exp();
        integral += value;
        for (k, moment) in moments.iter_mut().enumerate() {
            *moment += value * x.powi(k as i32 + 1) / (k + 1) as f64;
        }
    }
    let gradient: Vec<f64> = moments.iter().map(|m| spread / kt * m / integral).collect();
    let integral = integral * h / 3.0 * spread;
    if !(integral.is_finite() && integral > 0.0) {
        return Err(SaddleError::NonFinite("basin integral"));
    }
    let variance: f64 = (0..terms)
        .map(|j| {
            (0..terms)
                .map(|k| gradient[j] * covariance[j * terms + k] * gradient[k])
                .sum::<f64>()
        })
        .sum::<f64>()
        * inflation;
    let density = 1.0 / integral;
    Ok(BasinDensity {
        plane: Hyperplane::new(minimum.to_owned(), direction)?,
        thermal: kt,
        density,
        density_error: density * variance.sqrt(),
        spread,
        harmonic_density: 1.0 / (TAU.sqrt() * spread),
        offsets,
        slopes,
        slope_errors,
        coefficients,
        reduced_chi_square,
        evaluations,
    })
}

/// Inverse of a small symmetric positive matrix, row-major, by
/// Gauss–Jordan elimination with partial pivoting.
fn invert(matrix: &[f64], m: usize) -> Result<Vec<f64>, SaddleError> {
    let mut a = matrix.to_vec();
    let mut inverse = vec![0.0; m * m];
    for i in 0..m {
        inverse[i * m + i] = 1.0;
    }
    for col in 0..m {
        let pivot = (col..m)
            .max_by(|&i, &j| a[i * m + col].abs().total_cmp(&a[j * m + col].abs()))
            .unwrap_or(col);
        let size = a[pivot * m + col].abs();
        if size.is_nan() || size == 0.0 {
            return Err(SaddleError::Invalid(
                "the basin fit is singular; use fewer terms or more planes".into(),
            ));
        }
        for k in 0..m {
            a.swap(col * m + k, pivot * m + k);
            inverse.swap(col * m + k, pivot * m + k);
        }
        let scale = 1.0 / a[col * m + col];
        for k in 0..m {
            a[col * m + k] *= scale;
            inverse[col * m + k] *= scale;
        }
        for row in 0..m {
            if row != col {
                let factor = a[row * m + col];
                for k in 0..m {
                    a[row * m + k] -= factor * a[col * m + k];
                    inverse[row * m + k] -= factor * inverse[col * m + k];
                }
            }
        }
    }
    Ok(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    /// `V = ½ qᵀ K q` with one excluded direction.
    struct Bowl {
        k: Array2<f64>,
    }

    impl PointSurface for Bowl {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let g = self.k.dot(&q);
            Ok((0.5 * q.dot(&g), g))
        }

        fn excluded_modes(&self, _q: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
            Ok(array![[0.0, 0.0, 1.0]])
        }
    }

    #[test]
    fn inverse_of_a_small_matrix() {
        let a = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let inverse = invert(&a, 3).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let product: f64 = (0..3).map(|k| a[i * 3 + k] * inverse[k * 3 + j]).sum();
                assert!((product - f64::from(u8::from(i == j))).abs() < 1e-14);
            }
        }
        assert!(invert(&[1.0, 2.0, 2.0, 4.0], 2).is_err());
    }

    /// On a quadratic basin the density is `1 / √(2π kT nᵀK⁻¹n)` on the
    /// free directions, the excluded one held.
    #[test]
    fn density_of_a_harmonic_basin() {
        let surface = Bowl {
            k: array![[2.0, 0.6, 0.0], [0.6, 1.0, 0.0], [0.0, 0.0, 5.0]],
        };
        let kt = 0.3;
        let config = BasinConfig::default()
            .with_degree(1)
            .with_planes(5)
            .with_sampler(
                LangevinConfig::default()
                    .with_thermal(kt)
                    .with_time_step(0.2)
                    .with_records(2000)
                    .with_seed(3),
            );
        let normal = array![1.0, 1.0, 0.7];
        let minimum = array![0.0, 0.0, 0.4];
        let result = basin_density(&surface, minimum.view(), normal.view(), &config).unwrap();
        let k_inv = array![[1.0, -0.6], [-0.6, 2.0]] / (2.0 - 0.36);
        let n = array![1.0, 1.0] / 2.0f64.sqrt();
        let spread = (kt * n.dot(&k_inv.dot(&n))).sqrt();
        assert!((result.spread - spread).abs() < 1e-8 * spread);
        let exact = 1.0 / (TAU.sqrt() * spread);
        assert!((result.harmonic_density - exact).abs() < 1e-8 * exact);
        assert!(
            (result.density - exact).abs() < 4.0 * result.density_error,
            "{} ± {} against {exact}",
            result.density,
            result.density_error
        );
        assert!(result.density_error < 0.03 * exact);
        assert!((result.plane.normal()[2]).abs() < 1e-12);
        assert_eq!(result.offsets.len(), 5);
    }
}
