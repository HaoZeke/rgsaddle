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
//!
//! With [`ReversibleWorkConfig::bar_samples`] the same nodes also give the
//! work without quadrature error, by Bennett's acceptance ratio between
//! neighbours (J. Comput. Phys. 22, 245 (1976)). The map
//!
//! ```text
//! T(q) = p_a + R (q - p_a) + P (p_b - p_a),
//! ```
//!
//! with `R` the rotation taking `ν_a` to `ν_b` in their span and `P` the
//! projector off the held modes, is rigid, takes plane `a` onto plane `b`
//! and keeps the held modes, so `Z_b / Z_a = ⟨exp(-ΔV/kT)⟩_a` with
//! `ΔV = V(T q) - V(q)`, and the reverse average runs over `b` with `T⁻¹`.
//! This is Eqs. (7.1) and (7.2) of the OH-TST design survey. BAR uses the
//! configurations each node retains, so their spacing should exceed the
//! correlation time.

use ndarray::{Array1, ArrayView1, s};

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::ohtst::plane::Hyperplane;
use crate::ohtst::sample::{LangevinConfig, PlaneSample, held_directions, sample_plane};

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
    /// Configurations each node keeps for the BAR estimate; zero turns it
    /// off. A larger [`LangevinConfig::retain`] wins.
    pub bar_samples: usize,
}

impl Default for ReversibleWorkConfig {
    fn default() -> Self {
        Self {
            sampler: LangevinConfig::default(),
            warm_burn_in: 100,
            nodes: 9,
            simpson: true,
            bar_samples: 128,
        }
    }
}

impl ReversibleWorkConfig {
    /// Set [`Self::bar_samples`].
    #[must_use]
    pub fn with_bar_samples(mut self, bar_samples: usize) -> Self {
        self.bar_samples = bar_samples;
        self
    }

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
    /// Work from the first plane to this node by BAR between neighbours.
    pub bar_work: Option<f64>,
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
    /// The work by BAR between neighbouring nodes.
    pub bar_work: Option<f64>,
    /// Asymptotic standard error of [`Self::bar_work`].
    pub bar_error: Option<f64>,
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
    let excluded = held_directions(
        surface,
        planes[0].point(),
        None,
        config.sampler.hold_excluded,
    )?;
    let kt = config.sampler.thermal;
    let mut neighbour: Option<(Hyperplane, Vec<Array1<f64>>, Vec<f64>)> = None;
    let (mut bar_work, mut bar_variance) = (0.0, 0.0);
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
                sampler.retain = sampler.retain.max(config.bar_samples);
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
                if config.bar_samples > 0 {
                    let energies = sample
                        .retained
                        .iter()
                        .map(|q| energy(surface, q, &mut evaluations))
                        .collect::<Result<Vec<_>, _>>()?;
                    if let Some((from, configurations, before)) = neighbour.take() {
                        let map = RigidMap::new(&from, &sample.plane, &excluded);
                        let mut forward = Vec::with_capacity(configurations.len());
                        for (q, e) in configurations.iter().zip(&before) {
                            let mapped = energy(surface, &map.forward(q), &mut evaluations)?;
                            forward.push((mapped - e) / kt);
                        }
                        let mut reverse = Vec::with_capacity(energies.len());
                        for (q, e) in sample.retained.iter().zip(&energies) {
                            let mapped = energy(surface, &map.backward(q), &mut evaluations)?;
                            reverse.push((mapped - e) / kt);
                        }
                        let (difference, error) = bennett(&forward, &reverse)?;
                        bar_work += kt * difference;
                        bar_variance += (kt * error).powi(2);
                    }
                    neighbour = Some((sample.plane.clone(), sample.retained.clone(), energies));
                }
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
                    bar_work: (config.bar_samples > 0).then_some(bar_work),
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
        bar_work: (config.bar_samples > 0).then_some(bar_work),
        bar_error: (config.bar_samples > 0).then(|| bar_variance.sqrt()),
    })
}

fn energy<S: PointSurface>(
    surface: &S,
    q: &Array1<f64>,
    evaluations: &mut usize,
) -> Result<f64, SaddleError> {
    let (value, _) = surface.eval(q.view())?;
    *evaluations += 1;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SaddleError::NonFinite("energy of a mapped configuration"))
    }
}

/// The rigid map from plane `a` onto plane `b`.
struct RigidMap {
    pivot: Array1<f64>,
    shift: Array1<f64>,
    from: Array1<f64>,
    across: Option<Array1<f64>>,
    cos: f64,
    sin: f64,
}

impl RigidMap {
    fn new(a: &Hyperplane, b: &Hyperplane, excluded: &[Array1<f64>]) -> Self {
        let (na, nb) = (a.normal().to_owned(), b.normal().to_owned());
        let cos = na.dot(&nb).clamp(-1.0, 1.0);
        let mut across = &nb - &(&na * cos);
        let length = across.dot(&across).sqrt();
        let across = (length > 1.0e-14).then(|| {
            across /= length;
            across
        });
        let mut shift = &b.point() - &a.point();
        for t in excluded {
            let c = t.dot(&shift);
            shift.scaled_add(-c, t);
        }
        Self {
            pivot: a.point().to_owned(),
            shift,
            from: na,
            across,
            cos,
            sin: length,
        }
    }

    /// `R v`, or `Rᵀ v` when `sign` is negative.
    fn rotate(&self, v: &mut Array1<f64>, sign: f64) {
        let Some(u) = &self.across else {
            return;
        };
        let (a, b) = (self.from.dot(v), u.dot(v));
        let sin = sign * self.sin;
        v.scaled_add(sin * a, u);
        v.scaled_add(-sin * b, &self.from);
        v.scaled_add(self.cos - 1.0, &(&self.from * a + u * b));
    }

    fn forward(&self, q: &Array1<f64>) -> Array1<f64> {
        let mut v = q - &self.pivot;
        self.rotate(&mut v, 1.0);
        v + &self.pivot + &self.shift
    }

    fn backward(&self, q: &Array1<f64>) -> Array1<f64> {
        let mut v = q - &self.pivot - &self.shift;
        self.rotate(&mut v, -1.0);
        v + &self.pivot
    }
}

/// Bennett's acceptance ratio: `ΔA / kT` from forward works `w_F` on
/// samples of `a` and reverse works `w_R` on samples of `b`, in units of
/// `kT`, with its asymptotic standard error (Bennett's Eq. 10).
fn bennett(forward: &[f64], reverse: &[f64]) -> Result<(f64, f64), SaddleError> {
    if forward.is_empty() || reverse.is_empty() {
        return Err(SaddleError::Invalid(
            "BAR needs retained configurations on both planes".into(),
        ));
    }
    let (nf, nr) = (forward.len() as f64, reverse.len() as f64);
    let m = (nf / nr).ln();
    let fermi = |x: f64| 1.0 / (1.0 + x.exp());
    let balance = |df: f64| -> f64 {
        let left: f64 = forward.iter().map(|w| fermi(m + w - df)).sum();
        let right: f64 = reverse.iter().map(|w| fermi(-m + w + df)).sum();
        left - right
    };
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let (mut low, mut high) = (-mean(reverse) - 1.0, mean(forward) + 1.0);
    if low > high {
        std::mem::swap(&mut low, &mut high);
    }
    let mut width = 1.0;
    while balance(low) > 0.0 {
        low -= width;
        width *= 2.0;
        if width > 1.0e6 {
            return Err(SaddleError::Solver("BAR did not bracket".into()));
        }
    }
    width = 1.0;
    while balance(high) < 0.0 {
        high += width;
        width *= 2.0;
        if width > 1.0e6 {
            return Err(SaddleError::Solver("BAR did not bracket".into()));
        }
    }
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        if balance(middle) > 0.0 {
            high = middle;
        } else {
            low = middle;
        }
        if high - low < 1.0e-12 * (1.0 + middle.abs()) {
            break;
        }
    }
    let df = 0.5 * (low + high);
    let spread = |values: Vec<f64>| {
        let n = values.len() as f64;
        let first = values.iter().sum::<f64>() / n;
        let second = values.iter().map(|f| f * f).sum::<f64>() / n;
        (second / (first * first) - 1.0) / n
    };
    let variance = spread(forward.iter().map(|w| fermi(m + w - df)).collect())
        + spread(reverse.iter().map(|w| fermi(-m + w + df)).collect());
    let error = if variance.is_finite() {
        variance.max(0.0).sqrt()
    } else {
        f64::INFINITY
    };
    Ok((df, error))
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
        let (bar, bar_error) = (result.bar_work.unwrap(), result.bar_error.unwrap());
        assert!(
            (bar - expected).abs() < 4.0 * bar_error,
            "BAR {bar} ± {bar_error}, closed form {expected}"
        );
        assert!(bar_error < 0.02, "{bar_error}");
        assert!((middle.bar_work.unwrap() - partial).abs() < 0.02);
    }

    #[test]
    fn the_rigid_map_takes_one_plane_onto_the_other() {
        let a = Hyperplane::new(array![0.2, -0.4, 0.1], array![1.0, 0.3, 0.0]).unwrap();
        let b = Hyperplane::new(array![0.9, 0.2, 0.5], array![0.4, 1.0, 0.0]).unwrap();
        let held = [array![0.0, 0.0, 1.0]];
        let map = RigidMap::new(&a, &b, &held);
        let q = a.project(array![1.3, -0.2, 0.7].view());
        let r = a.project(array![-0.5, 0.8, 0.7].view());
        let (tq, tr) = (map.forward(&q), map.forward(&r));
        assert!(b.distance(tq.view()).abs() < 1e-12);
        assert!((tq[2] - q[2]).abs() < 1e-12);
        let (d0, d1) = (&q - &r, &tq - &tr);
        assert!((d0.dot(&d0) - d1.dot(&d1)).abs() < 1e-12);
        let back = map.backward(&tq);
        assert!((&back - &q).iter().all(|x| x.abs() < 1e-12));
        let same = RigidMap::new(&a, &a, &held);
        assert!((&same.forward(&q) - &q).iter().all(|x| x.abs() < 1e-12));
    }

    /// Gaussian works that obey Crooks' relation, `w_F ~ N(Δ + σ²/2, σ²)`
    /// and `w_R ~ N(-Δ + σ²/2, σ²)`, give back `Δ`.
    #[test]
    fn bar_recovers_a_known_difference() {
        let mut rng = crate::stats::Rng::new(4);
        let (delta, sigma) = (1.7, 1.2);
        let draw = |rng: &mut crate::stats::Rng, centre: f64, n: usize| -> Vec<f64> {
            (0..n).map(|_| centre + sigma * rng.normal()).collect()
        };
        let forward = draw(&mut rng, delta + 0.5 * sigma * sigma, 4000);
        let reverse = draw(&mut rng, -delta + 0.5 * sigma * sigma, 3000);
        let (estimate, error) = bennett(&forward, &reverse).unwrap();
        assert!(
            (estimate - delta).abs() < 4.0 * error,
            "{estimate} ± {error}"
        );
        assert!(error < 0.05 && error > 0.0);
        assert!(bennett(&forward, &[]).is_err());
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
