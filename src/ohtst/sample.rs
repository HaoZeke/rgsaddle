//! Constant-temperature sampling inside a hyperplane.
//!
//! The dynamics is Langevin at unit mass in mass-weighted coordinates,
//! restricted to the plane and, optionally, to fixed values of the
//! surface's excluded modes. The constraint is linear, so the restricted
//! Boltzmann density needs no Fixman term: the motion, the friction and the
//! noise are projected onto the tangent space of the plane. Each step is an
//! Ornstein–Uhlenbeck refresh of the velocity followed by one
//! velocity-Verlet proposal. With [`LangevinConfig::metropolis`] the
//! proposal is accepted on the change of the total energy and a rejection
//! reverses the velocity, which is generalized hybrid Monte Carlo and
//! leaves the restricted density exact. Without it the step is the
//! second-order OBABO splitting.

use ndarray::{Array1, Array2, ArrayView1, s};

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::ohtst::plane::{Hyperplane, orthonormal_rows};
use crate::stats::{Rng, autocorr_time};

/// Random projections of `(ν · ∇V) (q - p)` whose correlation times
/// join that of `ν · ∇V` in the block width.
const PROJECTIONS: usize = 3;

/// Upper bound on the stored partial sums, so the memory does not grow
/// with the number of records.
const CHUNKS: usize = 256;

/// Langevin sampling at constant temperature.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct LangevinConfig {
    /// `kT` in the energy unit of the surface.
    pub thermal: f64,
    /// Time step in the time unit of the mass-weighted surface (10.18 fs
    /// for eV, Å and amu).
    pub time_step: f64,
    /// Langevin friction `γ` in inverse time. Must be positive.
    pub friction: f64,
    /// Accept or reject each step on the change of the total energy.
    pub metropolis: bool,
    /// Steps discarded before the first record.
    pub burn_in: usize,
    /// Steps between records.
    pub stride: usize,
    /// Records.
    pub records: usize,
    /// Most blocks for the standard error. A block is at least twice the
    /// integrated autocorrelation time.
    pub blocks: usize,
    /// Seed of the noise.
    pub seed: u64,
    /// Configurations kept, evenly spaced over the records.
    pub retain: usize,
    /// Hold the surface's excluded modes, such as rigid translations, at
    /// their starting values.
    pub hold_excluded: bool,
}

impl Default for LangevinConfig {
    fn default() -> Self {
        Self {
            thermal: 0.02585,
            time_step: 0.05,
            friction: 1.0,
            metropolis: true,
            burn_in: 500,
            stride: 5,
            records: 2000,
            blocks: 20,
            seed: 1,
            retain: 16,
            hold_excluded: true,
        }
    }
}

impl LangevinConfig {
    /// Set [`Self::thermal`].
    #[must_use]
    pub fn with_thermal(mut self, thermal: f64) -> Self {
        self.thermal = thermal;
        self
    }

    /// Set [`Self::time_step`].
    #[must_use]
    pub fn with_time_step(mut self, time_step: f64) -> Self {
        self.time_step = time_step;
        self
    }

    /// Set [`Self::friction`].
    #[must_use]
    pub fn with_friction(mut self, friction: f64) -> Self {
        self.friction = friction;
        self
    }

    /// Set [`Self::metropolis`].
    #[must_use]
    pub fn with_metropolis(mut self, metropolis: bool) -> Self {
        self.metropolis = metropolis;
        self
    }

    /// Set [`Self::burn_in`].
    #[must_use]
    pub fn with_burn_in(mut self, burn_in: usize) -> Self {
        self.burn_in = burn_in;
        self
    }

    /// Set [`Self::stride`].
    #[must_use]
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set [`Self::records`].
    #[must_use]
    pub fn with_records(mut self, records: usize) -> Self {
        self.records = records;
        self
    }

    /// Set [`Self::blocks`].
    #[must_use]
    pub fn with_blocks(mut self, blocks: usize) -> Self {
        self.blocks = blocks;
        self
    }

    /// Set [`Self::seed`].
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set [`Self::retain`].
    #[must_use]
    pub fn with_retain(mut self, retain: usize) -> Self {
        self.retain = retain;
        self
    }

    /// Set [`Self::hold_excluded`].
    #[must_use]
    pub fn with_hold_excluded(mut self, hold_excluded: bool) -> Self {
        self.hold_excluded = hold_excluded;
        self
    }

    pub(crate) fn check(&self) -> Result<(), SaddleError> {
        if !(self.thermal.is_finite() && self.thermal > 0.0) {
            return Err(SaddleError::Invalid(
                "sampler kT must be finite and positive".into(),
            ));
        }
        if !(self.time_step.is_finite() && self.time_step > 0.0) {
            return Err(SaddleError::Invalid(
                "sampler time step must be finite and positive".into(),
            ));
        }
        if !(self.friction.is_finite() && self.friction > 0.0) {
            return Err(SaddleError::Invalid(
                "sampler friction must be finite and positive".into(),
            ));
        }
        if self.stride == 0 || self.blocks < 2 || self.records < 2 * self.blocks {
            return Err(SaddleError::Invalid(
                "sampler needs a positive stride, two blocks, and two records per block".into(),
            ));
        }
        Ok(())
    }
}

/// One step of constrained Langevin dynamics with an optional Metropolis
/// test. Positions keep `anchor`'s component along every `held` direction,
/// and a proposal on the positive side of a wall is rejected.
pub(crate) struct Dynamics<'a, S> {
    surface: &'a S,
    held: Vec<Array1<f64>>,
    anchor: Array1<f64>,
    walls: &'a [Hyperplane],
    kt: f64,
    h: f64,
    decay: f64,
    kick: f64,
    metropolis: bool,
    pub(crate) q: Array1<f64>,
    pub(crate) gradient: Array1<f64>,
    energy: f64,
    v: Array1<f64>,
    rng: Rng,
    pub(crate) evaluations: usize,
    pub(crate) proposed: usize,
    pub(crate) accepted: usize,
    pub(crate) wall_hits: usize,
}

impl<'a, S: PointSurface> Dynamics<'a, S> {
    pub(crate) fn new(
        surface: &'a S,
        start: Array1<f64>,
        held: Vec<Array1<f64>>,
        walls: &'a [Hyperplane],
        config: &LangevinConfig,
        rng: Rng,
    ) -> Result<Self, SaddleError> {
        if walls.iter().any(|wall| wall.distance(start.view()) > 0.0) {
            return Err(SaddleError::Invalid(
                "the starting configuration is past a wall".into(),
            ));
        }
        let n = start.len();
        let decay = (-config.friction * config.time_step).exp();
        let mut dynamics = Self {
            surface,
            held,
            anchor: start.clone(),
            walls,
            kt: config.thermal,
            h: config.time_step,
            decay,
            kick: (config.thermal * (1.0 - decay * decay)).sqrt(),
            metropolis: config.metropolis,
            q: start,
            gradient: Array1::zeros(n),
            energy: 0.0,
            v: Array1::zeros(n),
            rng,
            evaluations: 0,
            proposed: 0,
            accepted: 0,
            wall_hits: 0,
        };
        let q0 = dynamics.q.clone();
        let (energy, gradient) = dynamics.evaluate(q0.view())?;
        dynamics.energy = energy;
        dynamics.gradient = gradient;
        let sigma = config.thermal.sqrt();
        for value in dynamics.v.iter_mut() {
            *value = sigma * dynamics.rng.normal();
        }
        let mut v = std::mem::take(&mut dynamics.v);
        dynamics.tangent(&mut v);
        dynamics.v = v;
        Ok(dynamics)
    }

    fn evaluate(&mut self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient) = self.surface.eval(q)?;
        self.evaluations += 1;
        if gradient.len() != q.len() {
            return Err(SaddleError::Shape(
                "sampler gradient length differs from the configuration".into(),
            ));
        }
        if !energy.is_finite() || !gradient.iter().all(|g| g.is_finite()) {
            return Err(SaddleError::NonFinite("sampler energy or gradient"));
        }
        Ok((energy, gradient))
    }

    fn tangent(&self, v: &mut Array1<f64>) {
        for direction in &self.held {
            let c = direction.dot(v);
            v.scaled_add(-c, direction);
        }
    }

    fn constrain(&self, q: &mut Array1<f64>) {
        for direction in &self.held {
            let c = direction.dot(q) - direction.dot(&self.anchor);
            q.scaled_add(-c, direction);
        }
    }

    /// Advance one step. Returns whether the proposal was accepted.
    pub(crate) fn step(&mut self) -> Result<bool, SaddleError> {
        for value in self.v.iter_mut() {
            *value = self.decay * *value + self.kick * self.rng.normal();
        }
        let mut v = std::mem::take(&mut self.v);
        self.tangent(&mut v);
        self.proposed += 1;
        let kinetic0 = 0.5 * v.dot(&v);
        let mut force = -&self.gradient;
        self.tangent(&mut force);
        let mut v1 = &v + &(&force * (0.5 * self.h));
        let mut q1 = &self.q + &(&v1 * self.h);
        self.constrain(&mut q1);
        if self.walls.iter().any(|wall| wall.distance(q1.view()) > 0.0) {
            self.wall_hits += 1;
            self.v = -v;
            return Ok(false);
        }
        let (energy1, gradient1) = self.evaluate(q1.view())?;
        let mut force1 = -&gradient1;
        self.tangent(&mut force1);
        v1.scaled_add(0.5 * self.h, &force1);
        let accept = if self.metropolis {
            let delta = energy1 + 0.5 * v1.dot(&v1) - self.energy - kinetic0;
            delta <= 0.0 || self.rng.uniform() < (-delta / self.kt).exp()
        } else {
            true
        };
        if accept {
            self.q = q1;
            self.gradient = gradient1;
            self.energy = energy1;
            self.v = v1;
            self.accepted += 1;
        } else {
            self.v = -v;
        }
        Ok(accept)
    }

    pub(crate) fn acceptance(&self) -> f64 {
        if self.proposed == 0 {
            1.0
        } else {
            self.accepted as f64 / self.proposed as f64
        }
    }
}

/// Averages over one hyperplane.
///
/// `grad_point` and `grad_normal` are the Euclidean gradient blocks of the
/// plane free energy `A(p, ν)`: `∂A/∂p = ⟨ν · ∇V⟩ ν` and
/// `∂A/∂ν = -⟨(ν · ∇V) (q - p)⟩`, the second taken orthogonal to `ν` and
/// to any held mode. A manifold turns them into a Riemannian gradient.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PlaneSample {
    /// The sampled plane.
    pub plane: Hyperplane,
    /// Mean force along the normal, `⟨-ν · ∇V⟩`.
    pub normal_force: f64,
    /// Block standard error of [`Self::normal_force`].
    pub normal_force_error: f64,
    /// `∂A/∂p`.
    pub grad_point: Array1<f64>,
    /// `∂A/∂ν`.
    pub grad_normal: Array1<f64>,
    /// Block standard error of each entry of [`Self::grad_normal`].
    pub grad_normal_error: Array1<f64>,
    /// Mean configuration `⟨q⟩`.
    pub centroid: Array1<f64>,
    /// Block means of the packed gradient `[∂A/∂p; ∂A/∂ν]`, one row per
    /// block, for the standard error of any linear functional.
    pub block_gradients: Array2<f64>,
    /// Largest integrated autocorrelation time of the monitored series, in
    /// records.
    pub tau: f64,
    /// Records inside the blocks.
    pub records: usize,
    /// Surface evaluations.
    pub evaluations: usize,
    /// Accepted fraction of the proposals. One without the Metropolis test
    /// unless a wall rejected a step.
    pub acceptance: f64,
    /// Configuration after the last step.
    pub last: Array1<f64>,
    /// Configurations kept from the records, evenly spaced.
    pub retained: Vec<Array1<f64>>,
}

impl PlaneSample {
    /// `[∂A/∂p; ∂A/∂ν]`, the layout of [`Hyperplane::packed`].
    pub fn packed_gradient(&self) -> Array1<f64> {
        let n = self.grad_point.len();
        let mut g = Array1::zeros(2 * n);
        g.slice_mut(s![..n]).assign(&self.grad_point);
        g.slice_mut(s![n..]).assign(&self.grad_normal);
        g
    }

    /// The derivative of `A` along a packed direction `[δp; δν]`, and its
    /// block standard error.
    pub fn directional(&self, direction: ArrayView1<f64>) -> Result<(f64, f64), SaddleError> {
        if direction.len() != self.block_gradients.ncols() {
            return Err(SaddleError::Shape(
                "a packed direction has twice the configuration length".into(),
            ));
        }
        let values: Vec<f64> = self
            .block_gradients
            .rows()
            .into_iter()
            .map(|row| row.dot(&direction))
            .collect();
        Ok(mean_and_error(&values))
    }
}

pub(crate) fn mean_and_error(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if values.len() < 2 {
        return (mean, 0.0);
    }
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    (mean, (variance / n).sqrt())
}

/// Orthonormal directions the motion keeps fixed: the normal, then the
/// surface's excluded modes when they are held.
pub(crate) fn held_directions<S: PointSurface>(
    surface: &S,
    at: ArrayView1<f64>,
    normal: Option<ArrayView1<f64>>,
    hold_excluded: bool,
) -> Result<Vec<Array1<f64>>, SaddleError> {
    let n = at.len();
    let excluded = if hold_excluded {
        let modes = surface.excluded_modes(at)?;
        if modes.nrows() > 0 && modes.ncols() != n {
            return Err(SaddleError::Shape(
                "excluded modes differ in length from the configuration".into(),
            ));
        }
        orthonormal_rows(modes.view(), n)
    } else {
        Vec::new()
    };
    let mut held = Vec::with_capacity(excluded.len() + 1);
    if let Some(normal) = normal {
        let overlap: f64 = excluded
            .iter()
            .map(|t| t.dot(&normal).powi(2))
            .sum::<f64>()
            .sqrt();
        if overlap > 1.0e-6 {
            return Err(SaddleError::Invalid(
                "the plane normal has a component along a held mode; take Hyperplane::orthogonal_to first"
                    .into(),
            ));
        }
        held.push(normal.to_owned());
    }
    held.extend(excluded);
    Ok(held)
}

/// Sample the plane at `config.thermal` and average the gradient blocks
/// of its free energy.
///
/// `start` is projected onto the plane; `None` starts at the plane's
/// point. Held modes keep the values of the start.
pub fn sample_plane<S: PointSurface>(
    surface: &S,
    plane: &Hyperplane,
    start: Option<ArrayView1<f64>>,
    config: &LangevinConfig,
) -> Result<PlaneSample, SaddleError> {
    config.check()?;
    let n = plane.dim();
    let start = match start {
        Some(start) if start.len() != n => {
            return Err(SaddleError::Shape(
                "sampler start differs in length from the plane".into(),
            ));
        }
        Some(start) => plane.project(start),
        None => plane.point().to_owned(),
    };
    let normal = plane.normal().to_owned();
    let point = plane.point().to_owned();
    let held = held_directions(
        surface,
        point.view(),
        Some(normal.view()),
        config.hold_excluded,
    )?;
    let mut rng = Rng::new(config.seed);
    let mut probes = Vec::with_capacity(PROJECTIONS);
    for _ in 0..PROJECTIONS {
        let mut w: Array1<f64> = (0..n).map(|_| rng.normal()).collect();
        for direction in &held {
            let c = direction.dot(&w);
            w.scaled_add(-c, direction);
        }
        let norm = w.dot(&w).sqrt();
        if norm > 1.0e-12 {
            probes.push(w / norm);
        }
    }
    let excluded = held[1..].to_vec();
    let walls: [Hyperplane; 0] = [];
    let mut dynamics = Dynamics::new(surface, start, held, &walls, config, rng)?;
    for _ in 0..config.burn_in {
        dynamics.step()?;
    }
    let chunk = config.records.div_ceil(CHUNKS).max(1);
    let mut chunk_sums: Vec<Array1<f64>> = Vec::with_capacity(config.records.div_ceil(chunk));
    let mut sum = Array1::zeros(2 * n + 1);
    let mut series = vec![Vec::with_capacity(config.records); 1 + probes.len()];
    let retain_every = config
        .records
        .checked_div(config.retain)
        .map_or(usize::MAX, |every| every.max(1));
    let mut retained = Vec::with_capacity(config.retain);
    for record in 0..config.records {
        for _ in 0..config.stride {
            dynamics.step()?;
        }
        let q = &dynamics.q;
        let d = normal.dot(&dynamics.gradient);
        let arm = q - &point;
        sum[0] += d;
        sum.slice_mut(s![1..=n]).scaled_add(d, &arm);
        sum.slice_mut(s![n + 1..]).scaled_add(1.0, q);
        series[0].push(d);
        for (j, probe) in probes.iter().enumerate() {
            series[1 + j].push(d * probe.dot(&arm));
        }
        if (record + 1) % chunk == 0 || record + 1 == config.records {
            chunk_sums.push(std::mem::replace(&mut sum, Array1::zeros(2 * n + 1)));
        }
        if retained.len() < config.retain && record % retain_every == retain_every - 1 {
            retained.push(q.clone());
        }
    }
    let tau = series
        .iter()
        .map(|values| autocorr_time(values))
        .fold(0.5, f64::max);
    let width = (config.records / config.blocks)
        .max((2.0 * tau).ceil() as usize)
        .div_ceil(chunk)
        .max(1);
    let n_blocks = (config.records / chunk) / width;
    if n_blocks < 2 {
        return Err(SaddleError::Invalid(format!(
            "{} records hold fewer than two blocks of twice the correlation time {tau:.1}",
            config.records
        )));
    }
    let per_block = (width * chunk) as f64;
    let mut block_gradients = Array2::zeros((n_blocks, 2 * n));
    let mut centroid = Array1::zeros(n);
    let mut normal_force = Vec::with_capacity(n_blocks);
    for block in 0..n_blocks {
        let mut total = Array1::zeros(2 * n + 1);
        for part in &chunk_sums[block * width..(block + 1) * width] {
            total += part;
        }
        total /= per_block;
        let d = total[0];
        normal_force.push(-d);
        let mut torque = -total.slice(s![1..=n]).to_owned();
        let c = normal.dot(&torque);
        torque.scaled_add(-c, &normal);
        for t in &excluded {
            let c = t.dot(&torque);
            torque.scaled_add(-c, t);
        }
        let mut row = block_gradients.row_mut(block);
        row.slice_mut(s![..n]).assign(&(&normal * d));
        row.slice_mut(s![n..]).assign(&torque);
        centroid.scaled_add(1.0 / n_blocks as f64, &total.slice(s![n + 1..]));
    }
    let (force, force_error) = mean_and_error(&normal_force);
    let mut grad_point = Array1::zeros(n);
    let mut grad_normal = Array1::zeros(n);
    let mut grad_normal_error = Array1::zeros(n);
    for i in 0..n {
        let column: Vec<f64> = block_gradients.column(i).to_vec();
        grad_point[i] = mean_and_error(&column).0;
        let column: Vec<f64> = block_gradients.column(n + i).to_vec();
        let (mean, error) = mean_and_error(&column);
        grad_normal[i] = mean;
        grad_normal_error[i] = error;
    }
    Ok(PlaneSample {
        plane: plane.clone(),
        normal_force: force,
        normal_force_error: force_error,
        grad_point,
        grad_normal,
        grad_normal_error,
        centroid,
        block_gradients,
        tau,
        records: n_blocks * width * chunk,
        evaluations: dynamics.evaluations,
        acceptance: dynamics.acceptance(),
        last: dynamics.q.clone(),
        retained,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// `V = ½ qᵀ K q` with a translation-like excluded row.
    struct Bowl {
        k: Array2<f64>,
        excluded: Option<Array1<f64>>,
    }

    impl PointSurface for Bowl {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let g = self.k.dot(&q);
            Ok((0.5 * q.dot(&g), g))
        }

        fn excluded_modes(&self, q: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
            match &self.excluded {
                Some(row) => Ok(row.clone().into_shape_with_order((1, q.len())).unwrap()),
                None => Ok(Array2::zeros((0, q.len()))),
            }
        }
    }

    fn bowl() -> Bowl {
        Bowl {
            k: array![[2.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 0.5]],
            excluded: None,
        }
    }

    #[test]
    fn samples_stay_in_the_plane_and_hold_modes() {
        let mut surface = bowl();
        surface.excluded = Some(array![0.0, 0.0, 1.0]);
        let plane = Hyperplane::new(array![0.2, 0.1, 0.4], array![1.0, 1.0, 0.0]).unwrap();
        let config = LangevinConfig::default()
            .with_thermal(0.3)
            .with_time_step(0.2)
            .with_records(200)
            .with_blocks(4)
            .with_retain(5);
        let sample = sample_plane(&surface, &plane, None, &config).unwrap();
        assert_eq!(sample.retained.len(), 5);
        for q in sample.retained.iter().chain([&sample.last]) {
            assert!(plane.distance(q.view()).abs() < 1e-12);
            assert!((q[2] - 0.4).abs() < 1e-12);
        }
        assert!(sample.grad_normal[2].abs() < 1e-15);
        assert!(sample.grad_normal.dot(&plane.normal()).abs() < 1e-12);
        assert!(sample.acceptance > 0.9);
        let tilted = Hyperplane::new(array![0.0, 0.0, 0.0], array![0.0, 0.0, 1.0]).unwrap();
        assert!(sample_plane(&surface, &tilted, None, &config).is_err());
    }

    #[test]
    fn a_seed_fixes_the_stream() {
        let surface = bowl();
        let plane = Hyperplane::new(array![0.1, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        let config = LangevinConfig::default()
            .with_thermal(0.2)
            .with_time_step(0.3)
            .with_records(100)
            .with_blocks(4);
        let a = sample_plane(&surface, &plane, None, &config).unwrap();
        let b = sample_plane(&surface, &plane, None, &config).unwrap();
        assert_eq!(a.last, b.last);
        assert_eq!(a.block_gradients, b.block_gradients);
        let c = sample_plane(&surface, &plane, None, &config.clone().with_seed(2)).unwrap();
        assert_ne!(a.last, c.last);
    }

    #[test]
    fn the_restricted_gaussian_is_sampled() {
        // On ν = e_x through p, q = (p_x, y, z) with (y, z) Gaussian of
        // precision K_⊥ and mean -K_⊥⁻¹ k_⊥x p_x, and ν · ∇V = (K q)_x.
        let surface = bowl();
        let px = 0.6;
        let plane = Hyperplane::new(array![px, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        let kt = 0.4;
        let config = LangevinConfig::default()
            .with_thermal(kt)
            .with_time_step(0.4)
            .with_friction(0.5)
            .with_records(4000)
            .with_burn_in(200);
        let sample = sample_plane(&surface, &plane, None, &config).unwrap();
        let k_perp = array![[1.0, 0.2], [0.2, 0.5]];
        let det = k_perp[(0, 0)] * k_perp[(1, 1)] - k_perp[(0, 1)] * k_perp[(1, 0)];
        let inv = array![[0.5 / det, -0.2 / det], [-0.2 / det, 1.0 / det]];
        let mean = -inv.dot(&array![0.3 * px, 0.0]);
        let d_mean = 2.0 * px + 0.3 * mean[0];
        assert!((-sample.normal_force - d_mean).abs() < 4.0 * sample.normal_force_error);
        assert!(sample.normal_force_error < 0.02);
        assert!((sample.centroid[1] - mean[0]).abs() < 0.05);
        assert!((sample.centroid[2] - mean[1]).abs() < 0.05);
    }
}
