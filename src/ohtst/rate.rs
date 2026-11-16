//! The rate through the optimal hyperplane.
//!
//! In mass-weighted coordinates every direction carries unit mass, so the
//! one-way flux through a plane per unit density is `⟨(ν · q̇)₊⟩ = √(kT/2π)`,
//! the paper's Eq. 23 with `μ = 1`, and
//!
//! ```text
//! k = √(kT/2π) ρ exp(-ΔA/kT),
//! ```
//!
//! with `ρ = Q_Z / Q_R` from [`reactant_density`] on the first plane of
//! the work path and `ΔA` the [`reversible_work`] from that plane to the
//! optimal one (Eqs. 2, 3 and 21). The two errors add in `ln k`.
//!
//! With a Cartesian unit normal `n̂` the same flux is `√(kT/2πμ)` with
//! `μ = 1 / (n̂ᵀ M⁻¹ n̂)` ([`flux_mass`]), because `n̂ · ẋ` is Gaussian with
//! variance `kT Σ nᵢ²/mᵢ`. The paper's Eq. 24, `μ = Σ mᵢ nᵢ²`, comes from
//! the density of velocities along the ray `v n̂` instead, and the two
//! agree only when the masses that `n̂` touches are equal.
//!
//! [`reactant_density`]: crate::ohtst::reactant_density
//! [`reversible_work`]: crate::ohtst::reversible_work

use std::f64::consts::TAU;

use ndarray::ArrayView1;

use crate::error::SaddleError;
use crate::ohtst::plane::Hyperplane;
use crate::ohtst::reactant::ReactantDensity;
use crate::ohtst::work::ReversibleWork;

/// The OH-TST rate and its parts, each with its standard error.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct OhTstRate {
    /// `k`, per unit of the time that mass-weighted coordinates imply.
    pub rate: f64,
    /// First-order standard error of [`Self::rate`], `k σ(ln k)`.
    pub error: f64,
    /// `ln k`.
    pub log_rate: f64,
    /// `σ(ln k)`.
    pub log_rate_error: f64,
    /// `√(kT/2π)`.
    pub flux: f64,
    /// `Q_Z / Q_R` on the reference plane.
    pub density: f64,
    /// Its standard error.
    pub density_error: f64,
    /// `ΔA` from the reference plane to the last plane of the work path.
    pub free_energy: f64,
    /// Its standard error.
    pub free_energy_error: f64,
}

fn same_plane(a: &Hyperplane, b: &Hyperplane) -> bool {
    let scale = 1.0 + a.point().dot(&a.point()).sqrt();
    a.dim() == b.dim()
        && a.normal().dot(&b.normal()) > 1.0 - 1.0e-12
        && a.distance(b.point()).abs() <= 1.0e-9 * scale
}

/// Combine the reactant density on a reference plane with the work from
/// that plane to the optimal one. Both must come from the same `kT`, and
/// the work path must start at the density's plane.
pub fn ohtst_rate(
    density: &ReactantDensity,
    work: &ReversibleWork,
) -> Result<OhTstRate, SaddleError> {
    let kt = density.thermal;
    if (work.thermal - kt).abs() > 1.0e-12 * kt {
        return Err(SaddleError::Invalid(
            "the reactant density and the work come from different temperatures".into(),
        ));
    }
    let start = work
        .nodes
        .first()
        .ok_or_else(|| SaddleError::Invalid("the work path has no nodes".into()))?;
    if !same_plane(&start.plane, &density.plane) {
        return Err(SaddleError::Invalid(
            "the work path does not start at the plane of the reactant density".into(),
        ));
    }
    if !(density.density.is_finite() && density.density > 0.0) {
        return Err(SaddleError::Invalid(format!(
            "the reactant density {} ± {} is not positive; sample longer or move the plane toward the minimum",
            density.density, density.density_error
        )));
    }
    let flux = (kt / TAU).sqrt();
    let log_rate = flux.ln() + density.density.ln() - work.work / kt;
    let log_rate_error =
        ((density.density_error / density.density).powi(2) + (work.error / kt).powi(2)).sqrt();
    let rate = log_rate.exp();
    Ok(OhTstRate {
        rate,
        error: rate * log_rate_error,
        log_rate,
        log_rate_error,
        flux,
        density: density.density,
        density_error: density.density_error,
        free_energy: work.work,
        free_energy_error: work.error,
    })
}

/// The mass for flux along the Cartesian direction `normal`, normalized
/// here: `1 / Σ nᵢ²/mᵢ` with one mass per coordinate.
pub fn flux_mass(normal: ArrayView1<f64>, masses: ArrayView1<f64>) -> Result<f64, SaddleError> {
    if normal.len() != masses.len() || normal.is_empty() {
        return Err(SaddleError::Shape(
            "a normal and its masses differ in length".into(),
        ));
    }
    if !masses.iter().all(|m| m.is_finite() && *m > 0.0) {
        return Err(SaddleError::Invalid(
            "masses must be finite and positive".into(),
        ));
    }
    let norm2 = normal.dot(&normal);
    if !(norm2.is_finite() && norm2 > 0.0) {
        return Err(SaddleError::Invalid("the normal vanishes".into()));
    }
    let inverse: f64 = normal
        .iter()
        .zip(masses.iter())
        .map(|(n, m)| n * n / m)
        .sum::<f64>()
        / norm2;
    Ok(1.0 / inverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minmode::PointSurface;
    use crate::ohtst::sample::{LangevinConfig, sample_plane};
    use crate::ohtst::work::WorkNode;
    use ndarray::{Array1, array};

    fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, steps: usize) -> f64 {
        let h = (b - a) / steps as f64;
        let mut sum = f(a) + f(b);
        for i in 1..steps {
            sum += f(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        sum * h / 3.0
    }

    /// `⟨(n̂ · v)₊⟩` for `vᵢ ~ N(0, kT/mᵢ)` in two dimensions, with
    /// `n̂₀ > 0`. The inner integral starts where `n̂ · v` turns positive, so
    /// both integrands are smooth.
    fn mean_positive_flux(normal: [f64; 2], masses: [f64; 2], kt: f64) -> f64 {
        let c = normal[0] * (kt / masses[0]).sqrt();
        let d = normal[1] * (kt / masses[1]).sqrt();
        let phi = |t: f64| (-0.5 * t * t).exp() / TAU.sqrt();
        simpson(
            |b| {
                let lo = (-d * b / c).max(-12.0);
                if lo >= 12.0 {
                    0.0
                } else {
                    phi(b) * simpson(|a| (c * a + d * b) * phi(a), lo, 12.0, 2000)
                }
            },
            -12.0,
            12.0,
            400,
        )
    }

    #[test]
    fn flux_mass_is_the_marginal_of_the_normal_velocity() {
        let normal = [0.5f64.sqrt(), 0.5f64.sqrt()];
        let masses = [1.0, 3.0];
        let mu = flux_mass(array![1.0, 1.0].view(), array![1.0, 3.0].view()).unwrap();
        assert!((mu - 1.5).abs() < 1e-14);
        let kt = 0.7;
        let flux = mean_positive_flux(normal, masses, kt);
        assert!((flux - (kt / (TAU * mu)).sqrt()).abs() < 1e-8);
        let eq24: f64 = masses.iter().zip(normal).map(|(m, n)| m * n * n).sum();
        assert!((eq24 - 2.0).abs() < 1e-14);
        assert!((flux - (kt / (TAU * eq24)).sqrt()).abs() > 1e-2);
        let equal = flux_mass(array![0.6, 0.8].view(), array![2.0, 2.0].view()).unwrap();
        assert!((equal - 2.0).abs() < 1e-14);
    }

    struct Bowl;

    impl PointSurface for Bowl {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((0.5 * q.dot(&q), q.to_owned()))
        }
    }

    #[test]
    fn rate_combines_density_and_work() {
        let plane = Hyperplane::new(array![-0.5, 0.0], array![1.0, 0.0]).unwrap();
        let config = LangevinConfig::default().with_records(200);
        let end = sample_plane(&Bowl, &plane, None, &config).unwrap();
        let node = WorkNode {
            leg: 0,
            s: 0.0,
            plane: plane.clone(),
            derivative: 0.0,
            error: 0.0,
            work: 0.0,
        };
        let kt = 0.5;
        let work = ReversibleWork {
            work: 3.0,
            error: 0.1,
            nodes: vec![node],
            end,
            evaluations: 0,
            thermal: kt,
        };
        let density = ReactantDensity {
            plane: plane.clone(),
            thermal: kt,
            density: 0.4,
            density_error: 0.02,
            crossing_density: 0.4,
            crossing_error: 0.05,
            behind: 0.3,
            tau: 1.0,
            records: 1,
            evaluations: 0,
            acceptance: 1.0,
            wall_hits: 0,
            last: array![0.0, 0.0],
        };
        let rate = ohtst_rate(&density, &work).unwrap();
        let expected = (kt / TAU).sqrt() * 0.4 * (-3.0f64 / kt).exp();
        assert!((rate.rate / expected - 1.0).abs() < 1e-12);
        let sigma = ((0.02f64 / 0.4).powi(2) + (0.1f64 / kt).powi(2)).sqrt();
        assert!((rate.log_rate_error - sigma).abs() < 1e-12);
        assert!((rate.error - expected * sigma).abs() < 1e-12 * expected);

        let mut hot = density.clone();
        hot.thermal = 0.6;
        assert!(ohtst_rate(&hot, &work).is_err());
        let mut elsewhere = density.clone();
        elsewhere.plane = Hyperplane::new(array![-0.4, 0.0], array![1.0, 0.0]).unwrap();
        assert!(ohtst_rate(&elsewhere, &work).is_err());
        let mut empty = density;
        empty.density = -0.01;
        assert!(ohtst_rate(&empty, &work).is_err());
    }
}
