//! Basin constrained dimer forces from Xiao, Wu, and Henkelman, J. Chem.
//! Phys. 141, 164111 (2014), doi:10.1063/1.4898664 (the kappa-dimer).
//!
//! The force is built from the gradient, a lowest-mode estimate and a
//! Hessian-vector callback. Positions are in Angstrom, gradients in
//! eV/Angstrom, Hessian-vector products in eV/Angstrom^2, all as flat `3N`
//! arrays in atom-major order `[x0, y0, z0, x1, ...]`.
//!
//! The kappa-dimer is for a host that follows the force to a saddle from
//! inside a convex basin. The ordinary dimer of ASE (`DimerControl`) and
//! eOn is [`crate::MinModeSession`].
use crate::SaddleError;
use ndarray::{Array1, ArrayView1, ArrayView2, s};
use rgmin::{ApplyHessian, EigenParams, EigensolverKind};
use std::cell::{Cell, RefCell};

#[derive(Clone, Debug)]
/// Controls of [`kappa_dimer_force`].
///
/// The defaults are `beta = 5.0` Angstrom and a dimer-kind lowest-mode
/// solve with tolerance `1e-6` and at most 100 iterations.
///
/// Non-exhaustive: start from [`KappaDimerConfig::default`] and chain
/// the `with_*` setters.
#[non_exhaustive]
pub struct KappaDimerConfig {
    /// Switching length in Angstrom; `beta * kappa` is dimensionless, with
    /// `kappa` in 1/Angstrom. Must be positive and finite. It sets the width
    /// of the switch `gamma_parallel = -tanh(beta kappa / 2)`,
    /// `gamma_perpendicular = 1 / (1 + exp(-beta kappa))`.
    pub beta: f64,
    /// Lowest-mode solve of the reduced Hessian, from `rgmin`. Its `tol`
    /// bounds the residual of the tangent eigenpair relative to
    /// `max(|H c|, 1)` and must be positive and finite.
    pub eigen: EigenParams,
}

with_setters!(KappaDimerConfig {
    with_beta => beta: val f64,
    with_eigen => eigen: val EigenParams,
});

impl Default for KappaDimerConfig {
    fn default() -> Self {
        Self {
            beta: 5.0,
            eigen: EigenParams::default()
                .with_kind(EigensolverKind::Dimer)
                .with_tol(1e-6)
                .with_max_iter(100),
        }
    }
}

#[derive(Clone, Debug)]
/// Result of [`kappa_dimer_force`].
pub struct KappaDimerForce {
    /// Force in the ASE convention, the negative of the energy gradient,
    /// length `3N`, in eV/Angstrom. It is zero at a zero gradient.
    pub force: Array1<f64>,
    /// Unit lowest-mode vector of the Hessian on the tangent space (the
    /// space orthogonal to the excluded directions and to the gradient),
    /// lifted back to the full `3N` Cartesian space. Zero when the gradient
    /// vanishes or the tangent space is empty.
    pub tangent_mode: Array1<f64>,
    /// Dimension of the tangent space. It is `3N` minus the number of
    /// independent excluded directions, less one when the gradient is
    /// nonzero.
    pub tangent_dimension: usize,
    /// Rayleigh quotient of the Hessian along `tangent_mode`, in
    /// eV/Angstrom^2. Zero when no tangent mode was computed.
    pub tangent_curvature: f64,
    /// `-tangent_curvature / |g|` in 1/Angstrom, with `g` the gradient
    /// projected off the excluded directions. It is `-inf` when the tangent
    /// space is empty and `0` at a zero gradient.
    pub kappa: f64,
    /// Coefficient of the force component along the projected mode,
    /// dimensionless: `-tanh(beta * kappa / 2)`, so it is `+1` in strongly
    /// convex regions and `-1` where the tangent curvature is strongly
    /// negative.
    pub gamma_parallel: f64,
    /// Coefficient of the force component perpendicular to the projected
    /// mode, dimensionless: `1 / (1 + exp(-beta * kappa))`, between 0 and 1.
    pub gamma_perpendicular: f64,
    /// Norm of the residual `H c - nu c` of the tangent eigenpair, in
    /// eV/Angstrom^2.
    pub residual: f64,
    /// Number of Hessian-vector callbacks spent, counting the residual check.
    pub actions: usize,
}

/// Equation (6) in the complement of the force and the excluded modes,
/// followed by the equation (7) force (Xiao, Wu, and Henkelman, J. Chem.
/// Phys. 141, 164111 (2014), doi:10.1063/1.4898664). Rows of `excluded` are
/// Cartesian symmetry or constraint directions. The Hessian callback
/// receives full Cartesian vectors; it need not assemble a matrix.
///
/// `gradient` is the energy gradient (length `3N`, eV/Angstrom). `mode` is a
/// starting estimate of the lowest mode, length `3N`, only its direction
/// matters. `excluded` is `r x 3N` with one direction per row, for example
/// rigid translations; pass an empty `0 x 3N` view for none. `apply_hessian`
/// returns `H v` in eV/Angstrom^2 for a full `3N` vector `v`.
///
/// # Errors
///
/// - [`SaddleError::Shape`] when `gradient` is empty, `mode` or the columns
///   of `excluded` are not `3N`, or `mode` lies entirely in the excluded
///   space.
/// - [`SaddleError::Solver`] when `beta` or the eigensolver tolerance is not
///   positive and finite, when the lowest-mode solve fails, or when the
///   tangent eigenpair cannot be refined to the tolerance within the
///   tangent space.
/// - [`SaddleError::NonFinite`] when an input, the tangent mode or the force
///   is not finite.
/// - Any error returned by `apply_hessian`.
///
/// With `g` the gradient projected off the excluded directions, `tau` the
/// unit projected mode and `kappa = -nu / |g|` for the lowest tangent
/// curvature `nu`, the force is
/// `-gamma_perpendicular (g - (g . tau) tau) + gamma_parallel (g . tau) tau`.
///
/// # Examples
///
/// ```
/// use ndarray::{Array1, Array2, ArrayView1, array};
/// use rgsaddle::SaddleError;
/// use rgsaddle::kappa::{KappaDimerConfig, kappa_dimer_force};
///
/// // E = (x^2 + 2 y^2 + 3 z^2) / 2 at (1, 0, 0): gradient (1, 0, 0) eV/Angstrom
/// // and Hessian diag(1, 2, 3) eV/Angstrom^2.
/// let hessian = |v: ArrayView1<f64>| -> Result<Array1<f64>, SaddleError> {
///     Ok(array![v[0], 2.0 * v[1], 3.0 * v[2]])
/// };
/// let gradient = array![1.0, 0.0, 0.0];
/// let mode = array![0.0, 1.0, 0.0];
/// let none = Array2::<f64>::zeros((0, 3));
/// let out = kappa_dimer_force(
///     gradient.view(),
///     mode.view(),
///     none.view(),
///     hessian,
///     &KappaDimerConfig::default(),
/// )?;
/// // The tangent space is the y-z plane; its lowest curvature is 2 eV/Angstrom^2.
/// assert_eq!(out.tangent_dimension, 2);
/// approx::assert_abs_diff_eq!(out.tangent_curvature, 2.0, epsilon = 0.001);
/// approx::assert_abs_diff_eq!(out.kappa, -2.0, epsilon = 0.001);
/// // Deep in the convex basin the perpendicular force is switched off.
/// assert!(out.gamma_perpendicular < 0.001);
/// # Ok::<(), SaddleError>(())
/// ```
pub fn kappa_dimer_force<F>(
    gradient: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    excluded: ArrayView2<f64>,
    apply_hessian: F,
    config: &KappaDimerConfig,
) -> Result<KappaDimerForce, SaddleError>
where
    F: Fn(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError>,
{
    let n = gradient.len();
    if n == 0 || mode.len() != n || excluded.ncols() != n {
        return Err(SaddleError::Shape(
            "kappa gradient, mode and excluded directions".into(),
        ));
    }
    if !config.beta.is_finite()
        || config.beta <= 0.0
        || !config.eigen.tol.is_finite()
        || config.eigen.tol <= 0.0
    {
        return Err(SaddleError::Solver(
            "kappa beta and eigensolver tolerance must be positive and finite".into(),
        ));
    }
    if !gradient
        .iter()
        .chain(mode.iter())
        .chain(excluded.iter())
        .all(|v| v.is_finite())
    {
        return Err(SaddleError::NonFinite("kappa input"));
    }
    let mut space = Complement::new(n);
    for direction in excluded.rows() {
        space.exclude(direction);
    }
    let g = space.project(gradient);
    let force_norm = norm(g.view());
    let mut tau = space.project(mode);
    let mode_norm = norm(tau.view());
    if mode_norm == 0.0 {
        return Err(SaddleError::Shape(
            "kappa mode has no unconstrained component".into(),
        ));
    }
    tau /= mode_norm;
    let parallel = &tau * g.dot(&tau);
    let mut result = KappaDimerForce {
        force: Array1::zeros(n),
        tangent_mode: Array1::zeros(n),
        tangent_dimension: n - space.reflectors.len(),
        tangent_curvature: 0.0,
        kappa: 0.0,
        gamma_parallel: 0.0,
        gamma_perpendicular: 0.5,
        residual: 0.0,
        actions: 0,
    };
    if force_norm == 0.0 {
        return Ok(result);
    }
    space.exclude(g.view());
    result.tangent_dimension = space.dimension();
    if space.dimension() == 0 {
        result.kappa = f64::NEG_INFINITY;
        result.gamma_parallel = 1.0;
        result.gamma_perpendicular = 0.0;
        result.force = parallel;
        return Ok(result);
    }
    let h = ReducedHessian {
        space: &space,
        action: &apply_hessian,
        failure: RefCell::new(None),
        actions: Cell::new(0),
    };
    // A mixed seed excites tangent modes even if tau is parallel to the force.
    let seed = Array1::from_iter(
        (0..space.dimension()).map(|i| ((i + 1) as f64 * 1.618033988749895).sin()),
    );
    let mut solve = config.eigen;
    let mut mode_seed = seed;
    let mut refining = false;
    let (c, nu, residual) = loop {
        let eigenpair = rgmin::lowest_mode(
            &h,
            Array1::zeros(space.dimension()).view(),
            mode_seed.view(),
            &solve,
        );
        if let Some(error) = h.failure.take() {
            return Err(error);
        }
        let eigenpair = eigenpair.map_err(|error| SaddleError::Solver(error.to_string()))?;
        let mut c = eigenpair.vector;
        let c_norm = norm(c.view());
        if !c_norm.is_finite() || c_norm == 0.0 {
            return Err(SaddleError::NonFinite("kappa tangent mode"));
        }
        c /= c_norm;
        let hc = h.apply_hessian(Array1::zeros(c.len()).view(), c.view());
        if let Some(error) = h.failure.take() {
            return Err(error);
        }
        let nu = c.dot(&hc);
        let residual = norm((&hc - &(&c * nu)).view());
        if !nu.is_finite() || !residual.is_finite() {
            return Err(SaddleError::NonFinite("kappa tangent eigenpair"));
        }
        if residual <= config.eigen.tol * norm(hc.view()).max(1.0) {
            break (c, nu, residual);
        }
        if refining && solve.krylov == space.dimension() {
            return Err(SaddleError::Solver(format!(
                "kappa tangent eigenpair residual {residual} exceeds tolerance"
            )));
        }
        // The force requires a certified contour mode. An unresolved primary
        // solve continues in increasing Rayleigh-Ritz spaces on the same
        // reduced Hessian, retaining the physical residual requirement.
        let dimension = if refining {
            solve.krylov.saturating_mul(2)
        } else if solve.krylov == 0 {
            12
        } else {
            solve.krylov
        };
        solve.kind = EigensolverKind::RayleighRitz;
        solve.krylov = dimension.max(2).min(space.dimension());
        // rgmin scales its stopping tolerance by 1 + abs(nu); half the
        // requested tolerance implies the stricter contour scale above.
        solve.tol = config.eigen.tol * 0.5;
        mode_seed = c;
        refining = true;
    };
    result.tangent_mode = space.lift(c.view());
    result.tangent_curvature = nu;
    result.kappa = -nu / force_norm;
    (result.gamma_parallel, result.gamma_perpendicular) = coefficients(config.beta * result.kappa);
    result.force =
        -(&g - &parallel) * result.gamma_perpendicular + parallel * result.gamma_parallel;
    result.residual = residual;
    result.actions = h.actions.get();
    if !result.force.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("kappa force"));
    }
    Ok(result)
}

fn norm(v: ArrayView1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |a, &b| a.hypot(b))
}

/// Equation (7), retaining small parallel coefficients around kappa=0.
fn coefficients(z: f64) -> (f64, f64) {
    if z >= 0.0 {
        let e = (-z).exp();
        ((-z).exp_m1() / (1.0 + e), 1.0 / (1.0 + e))
    } else {
        let e = z.exp();
        (-z.exp_m1() / (1.0 + e), e / (1.0 + e))
    }
}

/// The trailing columns of a product of Householder reflections. Each
/// excluded direction consumes one leading coordinate. Storage is O(n*r)
/// for r excluded directions; Hessian vectors stay in Cartesian coordinates.
pub(crate) struct Complement {
    n: usize,
    reflectors: Vec<Array1<f64>>,
}
impl Complement {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            n,
            reflectors: Vec::new(),
        }
    }
    pub(crate) fn dimension(&self) -> usize {
        self.n - self.reflectors.len()
    }
    fn reflect(v: &mut Array1<f64>, offset: usize, axis: &Array1<f64>) {
        let dot = v.slice(s![offset..]).dot(axis);
        for (i, &a) in axis.iter().enumerate() {
            v[offset + i] -= 2.0 * dot * a;
        }
    }
    fn forward(&self, v: ArrayView1<f64>) -> Array1<f64> {
        let mut out = v.to_owned();
        for (offset, axis) in self.reflectors.iter().enumerate() {
            Self::reflect(&mut out, offset, axis);
        }
        out
    }
    pub(crate) fn reduce(&self, v: ArrayView1<f64>) -> Array1<f64> {
        self.forward(v)
            .slice(s![self.reflectors.len()..])
            .to_owned()
    }
    pub(crate) fn lift(&self, v: ArrayView1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(self.n);
        out.slice_mut(s![self.reflectors.len()..]).assign(&v);
        for (offset, axis) in self.reflectors.iter().enumerate().rev() {
            Self::reflect(&mut out, offset, axis);
        }
        out
    }
    fn project(&self, v: ArrayView1<f64>) -> Array1<f64> {
        self.lift(self.reduce(v).view())
    }
    pub(crate) fn exclude(&mut self, v: ArrayView1<f64>) {
        let input_norm = norm(v);
        if input_norm == 0.0 || self.dimension() == 0 {
            return;
        }
        let unit = &v / input_norm;
        let mut axis = self.reduce(unit.view());
        let tail_norm = norm(axis.view());
        // Dependent rotations of a linear molecule do not reduce its dimension.
        if tail_norm <= 64.0 * f64::EPSILON {
            return;
        }
        axis /= tail_norm;
        axis[0] += if axis[0] >= 0.0 { 1.0 } else { -1.0 };
        let axis_norm = norm(axis.view());
        axis /= axis_norm;
        self.reflectors.push(axis);
    }
}

struct ReducedHessian<'a, F> {
    space: &'a Complement,
    action: &'a F,
    failure: RefCell<Option<SaddleError>>,
    actions: Cell<usize>,
}
impl<F> ApplyHessian for ReducedHessian<'_, F>
where
    F: Fn(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError>,
{
    fn apply_hessian(&self, _: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        self.actions.set(self.actions.get() + 1);
        let hv = (self.action)(self.space.lift(v).view()).and_then(|hv| {
            if hv.len() != self.space.n {
                return Err(SaddleError::Shape("kappa Hessian action".into()));
            }
            if !hv.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::NonFinite("kappa Hessian action"));
            }
            Ok(self.space.reduce(hv.view()))
        });
        match hv {
            Ok(v) => v,
            Err(error) => {
                self.failure.replace(Some(error));
                Array1::zeros(v.len())
            }
        }
    }
}
