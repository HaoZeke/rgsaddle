//! Reversible work between hyperplanes.
//!
//! The free energy of a plane changes by `dA = ∂A/∂p · dp + ∂A/∂ν · dν`
//! (Jóhannesson and Jónsson, Eqs. 4, 18 and 19). [`reversible_work`]
//! integrates it along a path through given planes: on each leg the point
//! moves on a straight line and the normal turns on the great circle
//! between the end normals,
//!
//! ```text
//! p(s) = p₀ + s (p₁ - p₀),   ν(s) = cos(sΩ) ν₀ + sin(sΩ) w,
//! ```
//!
//! with `Ω = arccos(ν₀ · ν₁)` and `w` the unit part of `ν₁` orthogonal to
//! `ν₀`. A node samples its plane with [`sample_plane`] and gives the
//! derivative of `A` along `[p₁ - p₀; ν'(s)]` from the block means, so
//! the nodes are independent and the error of the sum is
//! `sqrt(Σ wᵢ² σᵢ²)` over the quadrature weights. Each node starts from
//! the last configuration of the one before. `A` is a function of the
//! plane, so any path between two planes gives the same work; the paper
//! accumulates it along the optimizer's own trajectory instead.

use ndarray::{Array1, ArrayView1, s};

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::ohtst::plane::Hyperplane;
use crate::ohtst::sample::{LangevinConfig, PlaneSample, sample_plane};

/// Settings of [`reversible_work`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ReversibleWorkConfig {
    /// Sampling of each node.
    pub sampler: LangevinConfig,
    /// Burn-in of a node that starts from the previous node's last
    /// configuration.
    pub warm_burn_in: usize,
    /// Nodes on each leg, ends included.
    pub nodes: usize,
    /// Composite Simpson weights; they need an odd node count. Otherwise
    /// trapezoid.
    pub simpson: bool,
}

impl Default for ReversibleWorkConfig {
    fn default() -> Self {
        Self {
            sampler: LangevinConfig::default(),
            warm_burn_in: 100,
            nodes: 9,
            simpson: true,
        }
    }
}

impl ReversibleWorkConfig {
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

    /// Set [`Self::nodes`].
    #[must_use]
    pub fn with_nodes(mut self, nodes: usize) -> Self {
        self.nodes = nodes;
        self
    }

    /// Set [`Self::simpson`].
    #[must_use]
    pub fn with_simpson(mut self, simpson: bool) -> Self {
        self.simpson = simpson;
        self
    }
}

/// One node of [`ReversibleWork`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct WorkNode {
    /// Leg index.
    pub leg: usize,
    /// Position on the leg, from 0 to 1.
    pub s: f64,
    /// The sampled plane.
    pub plane: Hyperplane,
    /// `dA/ds` on the leg.
    pub derivative: f64,
    /// Block standard error of [`Self::derivative`].
    pub error: f64,
    /// Work from the first plane to this node by the trapezoid rule.
    pub work: f64,
}

/// `A(last plane) - A(first plane)`.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ReversibleWork {
    /// The work.
    pub work: f64,
    /// Its standard error.
    pub error: f64,
    /// Every node in path order. A node shared by two legs is sampled
    /// once.
    pub nodes: Vec<WorkNode>,
    /// The sample at the last plane.
    pub end: PlaneSample,
    /// Surface evaluations over all nodes.
    pub evaluations: usize,
    /// The sampling temperature `kT`.
    pub thermal: f64,
}

/// One leg of the path: the plane at `s` and its derivative in `s`.
struct Leg {
    from: Hyperplane,
    to: Hyperplane,
    dp: Array1<f64>,
    w: Array1<f64>,
    angle: f64,
}

impl Leg {
    fn new(from: &Hyperplane, to: &Hyperplane) -> Result<Self, SaddleError> {
        if from.dim() != to.dim() {
            return Err(SaddleError::Shape(
                "planes on a work path differ in dimension".into(),
            ));
        }
        let cosine = from.normal().dot(&to.normal()).clamp(-1.0, 1.0);
        let mut w = &to.normal() - &(&from.normal() * cosine);
        let norm = w.dot(&w).sqrt();
        if norm <= 1.0e-12 {
            if cosine < 0.0 {
                return Err(SaddleError::Invalid(
                    "opposite normals on a work path; orient them with Hyperplane::oriented_toward"
                        .into(),
                ));
            }
            w.fill(0.0);
        } else {
            w /= norm;
        }
        Ok(Self {
            from: from.clone(),
            to: to.clone(),
            dp: &to.point() - &from.point(),
            w,
            angle: cosine.acos(),
        })
    }

    fn plane(&self, s: f64) -> Result<Hyperplane, SaddleError> {
        if s <= 0.0 {
            return Ok(self.from.clone());
        }
        if s >= 1.0 {
            return Ok(self.to.clone());
        }
        let theta = s * self.angle;
        Hyperplane::new(
            &self.from.point() + &(&self.dp * s),
            &self.from.normal() * theta.cos() + &self.w * theta.sin(),
        )
    }

    fn tangent(&self, s: f64) -> Array1<f64> {
        let n = self.dp.len();
        let theta = s * self.angle;
        let mut t = Array1::zeros(2 * n);
        t.slice_mut(s![..n]).assign(&self.dp);
        t.slice_mut(s![n..])
            .assign(&((&self.w * theta.cos() - &self.from.normal() * theta.sin()) * self.angle));
        t
    }
}

fn weights(nodes: usize, simpson: bool) -> Vec<f64> {
    let h = 1.0 / (nodes - 1) as f64;
    (0..nodes)
        .map(|i| {
            let ends = i == 0 || i == nodes - 1;
            if simpson {
                h / 3.0
                    * if ends {
                        1.0
                    } else if i % 2 == 1 {
                        4.0
                    } else {
                        2.0
                    }
            } else if ends {
                h / 2.0
            } else {
                h
            }
        })
        .collect()
}

/// The work to move from `planes[0]` to the last plane through the planes
/// between. `start`, when given, seeds the first node.
pub fn reversible_work<S: PointSurface>(
    surface: &S,
    planes: &[Hyperplane],
    start: Option<ArrayView1<f64>>,
    config: &ReversibleWorkConfig,
) -> Result<ReversibleWork, SaddleError> {
    config.sampler.check()?;
    if planes.len() < 2 {
        return Err(SaddleError::Invalid(
            "a work path needs at least two planes".into(),
        ));
    }
    if config.nodes < 2 || (config.simpson && config.nodes.is_multiple_of(2)) {
        return Err(SaddleError::Invalid(
            "a leg needs two nodes or more, and an odd count for Simpson".into(),
        ));
    }
    let legs = planes
        .windows(2)
        .map(|pair| Leg::new(&pair[0], &pair[1]))
        .collect::<Result<Vec<_>, _>>()?;
    let weights = weights(config.nodes, config.simpson);
    let h = 1.0 / (config.nodes - 1) as f64;
    let mut seed_from = start.map(|q| q.to_owned());
    let mut nodes: Vec<WorkNode> = Vec::new();
    // The latest sample and the weighted tangent it carries into the sum;
    // the node shared by two legs carries both legs' weights.
    let mut open: Option<(PlaneSample, Array1<f64>)> = None;
    let mut work = 0.0;
    let mut variance = 0.0;
    let mut evaluations = 0;
    let mut sampled = 0u64;
    for (k, leg) in legs.iter().enumerate() {
        let mut cumulative = nodes.last().map_or(0.0, |node| node.work);
        let mut before = None;
        for (i, weight) in weights.iter().enumerate() {
            let s = i as f64 * h;
            let tangent = leg.tangent(s);
            if i > 0 || k == 0 {
                if let Some((sample, direction)) = open.take() {
                    let (value, error) = sample.directional(direction.view())?;
                    work += value;
                    variance += error * error;
                }
                let mut sampler = config.sampler.clone();
                sampler.seed = sampler.seed.wrapping_add(sampled);
                if sampled > 0 {
                    sampler.burn_in = config.warm_burn_in;
                }
                let plane = leg.plane(s)?;
                let sample = sample_plane(
                    surface,
                    &plane,
                    seed_from.as_ref().map(|q| q.view()),
                    &sampler,
                )?;
                sampled += 1;
                evaluations += sample.evaluations;
                seed_from = Some(sample.last.clone());
                open = Some((sample, Array1::zeros(tangent.len())));
            }
            let (sample, direction) = open
                .as_mut()
                .ok_or_else(|| SaddleError::Invalid("a work leg has no sample".into()))?;
            direction.scaled_add(*weight, &tangent);
            let (derivative, error) = sample.directional(tangent.view())?;
            if let Some(previous) = before {
                cumulative += 0.5 * h * (previous + derivative);
            }
            before = Some(derivative);
            if i > 0 || k == 0 {
                nodes.push(WorkNode {
                    leg: k,
                    s,
                    plane: sample.plane.clone(),
                    derivative,
                    error,
                    work: cumulative,
                });
            }
        }
    }
    let (end, direction) =
        open.ok_or_else(|| SaddleError::Invalid("a work path has no sample".into()))?;
    let (value, error) = end.directional(direction.view())?;
    work += value;
    variance += error * error;
    Ok(ReversibleWork {
        work,
        error: variance.sqrt(),
        nodes,
        end,
        evaluations,
        thermal: config.sampler.thermal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// `V = ½ qᵀ K q`, where `A(p, ν) = (kT/2) ln(νᵀK⁻¹ν) + (ν·p)²/(2 νᵀK⁻¹ν)`.
    struct Bowl;

    const K: [f64; 3] = [1.0, 2.0, 0.5];

    impl PointSurface for Bowl {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let g = array![K[0] * q[0], K[1] * q[1], K[2] * q[2]];
            Ok((0.5 * q.dot(&g), g))
        }
    }

    fn exact(plane: &Hyperplane, kt: f64) -> f64 {
        let nu = plane.normal();
        let s: f64 = (0..3).map(|i| nu[i] * nu[i] / K[i]).sum();
        let a = nu.dot(&plane.point());
        0.5 * kt * s.ln() + a * a / (2.0 * s)
    }

    #[test]
    fn work_matches_the_closed_form_on_a_bent_path() {
        let kt = 0.3;
        let planes = [
            Hyperplane::new(array![-0.6, 0.1, 0.0], array![1.0, 0.0, 0.0]).unwrap(),
            Hyperplane::new(array![-0.1, 0.3, 0.2], array![1.0, 0.6, 0.2]).unwrap(),
            Hyperplane::new(array![0.3, -0.1, 0.4], array![0.6, 0.2, 1.0]).unwrap(),
        ];
        let config = ReversibleWorkConfig::default().with_sampler(
            LangevinConfig::default()
                .with_thermal(kt)
                .with_time_step(0.4)
                .with_burn_in(200)
                .with_stride(4)
                .with_records(2000),
        );
        let result = reversible_work(&Bowl, &planes, None, &config).unwrap();
        let expected = exact(&planes[2], kt) - exact(&planes[0], kt);
        assert!(
            (result.work - expected).abs() < 4.0 * result.error + 1.0e-4,
            "work {} ± {}, closed form {expected}",
            result.work,
            result.error
        );
        assert_eq!(result.nodes.len(), 2 * config.nodes - 1);
        let middle = &result.nodes[config.nodes - 1];
        let partial = exact(&planes[1], kt) - exact(&planes[0], kt);
        assert!((middle.work - partial).abs() < 0.01);
        assert_eq!(result.end.plane, planes[2]);
    }

    #[test]
    fn opposite_normals_are_rejected() {
        let a = Hyperplane::new(array![0.0, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        let b = Hyperplane::new(array![0.1, 0.0, 0.0], array![-1.0, 0.0, 0.0]).unwrap();
        let config = ReversibleWorkConfig::default();
        assert!(reversible_work(&Bowl, &[a.clone(), b], None, &config).is_err());
        assert!(reversible_work(&Bowl, &[a], None, &config).is_err());
    }
}
