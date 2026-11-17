//! Harmonic quantum transition-state theory on a line-integral path.
//!
//! Replicas sit at `t_j = j τ / P`. The ring-polymer Hessian in
//! mass-weighted coordinates is
//!
//! `(Hv)_i = κ (2 v_i − v_{i−1} − v_{i+1}) + ∇²V(q_i) v_i`,
//!
//! `κ = 1/(β_N² ℏ²)`, `β_N = τ / (ℏ P)`. Its fluctuation factor is the
//! product of the absolute eigenvalues except the permutation mode (and,
//! for a molecule, the rigid motions). [`hqtst_rate`] applies that
//! operator with Hessian-vector products only. The paper's
//! finite-difference Hessians, cubic interpolation, and dense spectrum
//! are compiled with the `lineb-oracle` feature for tests and benches.
//! They are not part of the default build.
//!
//! The rate times the reactant partition function is Richardson's
//! ring-polymer instanton expression (Int. Rev. Phys. Chem. 37, 171
//! (2018), eq. 52), which is the harmonic quantum TST prefactor:
//!
//! `k Z_r = Λ_N^{−1} (β_N ℏ)^{−1} √B_N  Π' (β_N ℏ η_k)^{−1}  exp(−β_N U)`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use super::action::l2;
use super::align::cross;
use super::mass::Masses;
use super::optimize::LinebPath;
use super::path;
#[cfg(any(test, feature = "lineb-oracle"))]
use super::path::Curve;
use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::minmode::PointSurface;
#[cfg(any(test, feature = "lineb-oracle"))]
use crate::ohtst::spline::Spline;

/// Ring-polymer instanton rate at the temperature of a tunneling path.
#[derive(Clone, Debug)]
pub struct HqtstRate {
    /// `ℏ / (k_B τ)` of the replica period.
    pub temperature: f64,
    /// Full imaginary-time period `τ`.
    pub period: f64,
    /// Euclidean action over `ℏ` from the line integral, `(2W + E τ) / ℏ`.
    pub line_action: f64,
    /// `β_N U_N` of the replicas.
    pub bead_action: f64,
    /// `Σ ln|λ|` over the kept eigenvalues of the ring-polymer Hessian.
    pub log_abs_det: f64,
    /// How many eigenvalues that sum keeps.
    pub modes: usize,
    /// `ln(k Z_r)`.
    pub log_rate_over_reactant: f64,
    /// `k Z_r`.
    pub rate_over_reactant: f64,
    /// Surface evaluations used to build the factor.
    pub evaluations: usize,
}

/// Matrix-free harmonic quantum TST rate. `beads` is `P` in `t_j = j τ / P`.
pub fn hqtst_rate<S: PointSurface>(
    surface: &S,
    path: &LinebPath,
    beads: usize,
) -> Result<HqtstRate, SaddleError> {
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let counted = crate::lineb::Counted {
        surface,
        calls: &calls,
    };
    let replicas = path::replicas(path.weighted.view(), &path.time_table, beads)?;
    let kappa = spring_constant(replicas.period, path.hbar, beads)?;
    let gauges = gauge_modes(&replicas.weighted, &path.masses)?;
    let log_abs_det = projected_log_abs_det(replicas.weighted.len(), &gauges, |vector| {
        ring_matvec(
            &counted,
            &path.masses,
            replicas.weighted.view(),
            vector,
            kappa,
        )
    })?;
    finish(
        path,
        &replicas.weighted,
        replicas.period,
        log_abs_det,
        &calls,
        &counted,
    )
}

/// The paper's finite-difference Hessians at the LI-NEB images, cubically
/// interpolated onto the replicas, then the same rate. Oracle for
/// [`hqtst_rate`]. Not built unless the `lineb-oracle` feature is on
/// (tests and benches request it).
#[cfg(any(test, feature = "lineb-oracle"))]
pub fn hqtst_rate_interpolated_hessian<S: PointSurface>(
    surface: &S,
    path: &LinebPath,
    beads: usize,
) -> Result<HqtstRate, SaddleError> {
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let counted = crate::lineb::Counted {
        surface,
        calls: &calls,
    };
    let replicas = path::replicas(path.weighted.view(), &path.time_table, beads)?;
    let kappa = spring_constant(replicas.period, path.hbar, beads)?;
    let step = coordinate_scale(path.weighted.iter().copied())?;
    let hessians = image_hessians(&counted, &path.masses, path.weighted.view(), step)?;
    let curve = Curve::new(path.weighted.view())?;
    let dense = interpolated_ring(
        &hessians,
        curve.knots(),
        &replicas.parameter,
        replicas.weighted.ncols(),
        kappa,
    )?;
    let log_abs_det = dense_kept_log_abs_det(dense.view(), &replicas.weighted, &path.masses)?;
    finish(
        path,
        &replicas.weighted,
        replicas.period,
        log_abs_det,
        &calls,
        &counted,
    )
}

fn finish<S: PointSurface>(
    path: &LinebPath,
    beads: &Array2<f64>,
    period: f64,
    log_abs_det: f64,
    calls: &std::sync::atomic::AtomicUsize,
    surface: &crate::lineb::Counted<'_, S>,
) -> Result<HqtstRate, SaddleError> {
    let mut energies = Vec::with_capacity(beads.nrows());
    for row in beads.rows() {
        let cartesian = path.masses.to_cartesian(row)?;
        let (energy, gradient) = surface.eval(cartesian.view())?;
        if !energy.is_finite() || gradient.iter().any(|component| !component.is_finite()) {
            return Err(SaddleError::NonFinite("ring-polymer replica"));
        }
        energies.push(energy);
    }
    let bead_action = bead_action(beads, &energies, period, path.hbar)?;
    let line_action = (2.0 * path.action + path.energy * period) / path.hbar;
    let modes = beads.len() - gauge_modes(beads, &path.masses)?.len();
    let log_rate = log_rate(beads, period, path.hbar, log_abs_det, modes, bead_action)?;
    let rate = log_rate.exp();
    if !rate.is_finite() {
        return Err(SaddleError::NonFinite("instanton rate"));
    }
    Ok(HqtstRate {
        temperature: path::temperature(0.5 * period, path.hbar, path.boltzmann)?,
        period,
        line_action,
        bead_action,
        log_abs_det,
        modes,
        log_rate_over_reactant: log_rate,
        rate_over_reactant: rate,
        evaluations: calls.load(std::sync::atomic::Ordering::Relaxed),
    })
}

fn spring_constant(period: f64, hbar: f64, beads: usize) -> Result<f64, SaddleError> {
    if !(period > 0.0 && hbar > 0.0) || beads < 2 {
        return Err(SaddleError::Invalid(
            "the ring polymer needs a positive period and hbar and at least two replicas".into(),
        ));
    }
    let beta_n = period / (hbar * beads as f64);
    let kappa = 1.0 / (beta_n * beta_n * hbar * hbar);
    if !kappa.is_finite() {
        return Err(SaddleError::NonFinite("ring-polymer spring"));
    }
    Ok(kappa)
}

fn bead_action(
    beads: &Array2<f64>,
    energies: &[f64],
    period: f64,
    hbar: f64,
) -> Result<f64, SaddleError> {
    let count = beads.nrows();
    let beta_n = period / (hbar * count as f64);
    let mut kinetic = 0.0;
    for bead in 0..count {
        let previous = if bead == 0 { count - 1 } else { bead - 1 };
        let mut length_sq = 0.0;
        for coordinate in 0..beads.ncols() {
            let delta = beads[(bead, coordinate)] - beads[(previous, coordinate)];
            length_sq += delta * delta;
        }
        kinetic += length_sq;
    }
    let potential: f64 = energies.iter().sum();
    let action = kinetic / (2.0 * beta_n * hbar * hbar) + beta_n * potential;
    if !action.is_finite() {
        return Err(SaddleError::NonFinite("ring-polymer action"));
    }
    Ok(action)
}

fn log_rate(
    beads: &Array2<f64>,
    period: f64,
    hbar: f64,
    log_abs_det: f64,
    modes: usize,
    bead_action: f64,
) -> Result<f64, SaddleError> {
    let count = beads.nrows();
    let beta_n = period / (hbar * count as f64);
    let mut length_sq = 0.0;
    for bead in 0..count {
        let previous = if bead == 0 { count - 1 } else { bead - 1 };
        for coordinate in 0..beads.ncols() {
            let delta = beads[(bead, coordinate)] - beads[(previous, coordinate)];
            length_sq += delta * delta;
        }
    }
    if length_sq <= 0.0 || length_sq.is_nan() {
        return Err(SaddleError::Invalid(
            "the replicas do not separate, so B_N vanishes".into(),
        ));
    }
    let beta_hbar = beta_n * hbar;
    let log_rate = -0.5 * (2.0 * std::f64::consts::PI * beta_n * hbar * hbar).ln() - beta_hbar.ln()
        + 0.5 * length_sq.ln()
        - 0.5 * log_abs_det
        - modes as f64 * beta_hbar.ln()
        - bead_action;
    if !log_rate.is_finite() {
        return Err(SaddleError::NonFinite("instanton rate"));
    }
    Ok(log_rate)
}

fn ring_matvec<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    beads: ArrayView2<f64>,
    vector: ArrayView1<f64>,
    kappa: f64,
) -> Result<Array1<f64>, SaddleError> {
    let count = beads.nrows();
    let dof = beads.ncols();
    if vector.len() != count * dof {
        return Err(SaddleError::Shape(
            "a ring-polymer vector has one block per replica".into(),
        ));
    }
    // One difference length for the whole ring, from the largest
    // coordinate. A replica at the origin still has the path's length.
    let step = coordinate_scale(beads.iter().copied())?;
    let mut out = Array1::zeros(vector.len());
    for bead in 0..count {
        let previous = if bead == 0 { count - 1 } else { bead - 1 };
        let next = if bead + 1 == count { 0 } else { bead + 1 };
        let block = vector.slice(s![bead * dof..(bead + 1) * dof]);
        let mut spring = block.to_owned();
        spring *= 2.0;
        spring -= &vector.slice(s![previous * dof..(previous + 1) * dof]);
        spring -= &vector.slice(s![next * dof..(next + 1) * dof]);
        spring *= kappa;
        let potential = weighted_hvp(surface, masses, beads.row(bead), block, step)?;
        spring += &potential;
        out.slice_mut(s![bead * dof..(bead + 1) * dof])
            .assign(&spring);
    }
    Ok(out)
}

fn weighted_hvp<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    weighted: ArrayView1<f64>,
    vector: ArrayView1<f64>,
    step: f64,
) -> Result<Array1<f64>, SaddleError> {
    let cartesian = masses.to_cartesian(weighted)?;
    let mut cartesian_vector = vector.to_owned();
    for (component, &root) in cartesian_vector.iter_mut().zip(masses.sqrt_mass.iter()) {
        *component /= root;
    }
    let product = match surface.hessian_vector(cartesian.view(), cartesian_vector.view())? {
        Some(product) => product,
        None => central_product(surface, cartesian.view(), cartesian_vector.view(), step)?,
    };
    if product.len() != cartesian.len() {
        return Err(SaddleError::Shape(
            "a Hessian-vector product must match the coordinate length".into(),
        ));
    }
    let mut weighted_product = product;
    for (component, &root) in weighted_product.iter_mut().zip(masses.sqrt_mass.iter()) {
        *component /= root;
    }
    Ok(weighted_product)
}

/// `√ε` times the largest absolute coordinate. That is the roundoff-optimal
/// central step for a second derivative on a path of this length.
fn coordinate_scale(values: impl IntoIterator<Item = f64>) -> Result<f64, SaddleError> {
    let mut scale = 0.0_f64;
    for value in values {
        scale = scale.max(value.abs());
    }
    if scale <= 0.0 || scale.is_nan() {
        return Err(SaddleError::Invalid(
            "the tunneling coordinates are all zero, so a Hessian difference has no length".into(),
        ));
    }
    Ok(scale * f64::EPSILON.sqrt())
}

/// Central Hessian-vector product. A zero vector is `H 0 = 0` exactly.
/// Any other vector is displaced by `step` along its direction.
fn central_product<S: PointSurface>(
    surface: &S,
    position: ArrayView1<f64>,
    vector: ArrayView1<f64>,
    step: f64,
) -> Result<Array1<f64>, SaddleError> {
    let norm = l2(vector);
    if norm <= 0.0 || norm.is_nan() {
        return Ok(Array1::zeros(position.len()));
    }
    if !(step.is_finite() && step > 0.0) {
        return Err(SaddleError::Invalid(
            "the Hessian difference length must be finite and positive".into(),
        ));
    }
    let eta = step / norm;
    let mut plus = position.to_owned();
    let mut minus = position.to_owned();
    plus.scaled_add(eta, &vector);
    minus.scaled_add(-eta, &vector);
    let (_, gradient_plus) = surface.eval(plus.view())?;
    let (_, gradient_minus) = surface.eval(minus.view())?;
    if gradient_plus.len() != position.len() || gradient_minus.len() != position.len() {
        return Err(SaddleError::Shape(
            "a Hessian difference changed the gradient length".into(),
        ));
    }
    let mut product = gradient_plus;
    product -= &gradient_minus;
    product /= 2.0 * eta;
    if product.iter().any(|component| !component.is_finite()) {
        return Err(SaddleError::NonFinite("Hessian-vector product"));
    }
    Ok(product)
}

fn gauge_modes(beads: &Array2<f64>, masses: &Masses) -> Result<Vec<Array1<f64>>, SaddleError> {
    let mut modes = Vec::new();
    modes.push(velocity_mode(beads)?);
    if let Some(atoms) = masses.atoms {
        modes.extend(rigid_modes(beads, &masses.sqrt_mass, atoms));
    }
    Ok(orthonormalize(modes))
}

fn velocity_mode(beads: &Array2<f64>) -> Result<Array1<f64>, SaddleError> {
    let count = beads.nrows();
    let dof = beads.ncols();
    let mut mode = Array1::zeros(count * dof);
    for bead in 0..count {
        let previous = if bead == 0 { count - 1 } else { bead - 1 };
        for coordinate in 0..dof {
            mode[bead * dof + coordinate] =
                beads[(bead, coordinate)] - beads[(previous, coordinate)];
        }
    }
    let norm = l2(mode.view());
    if norm <= 0.0 || norm.is_nan() {
        return Err(SaddleError::Invalid(
            "the replicas do not move, so the permutation mode is undefined".into(),
        ));
    }
    mode /= norm;
    Ok(mode)
}

fn rigid_modes(beads: &Array2<f64>, sqrt_mass: &Array1<f64>, atoms: usize) -> Vec<Array1<f64>> {
    let count = beads.nrows();
    let dof = beads.ncols();
    let mut modes = Vec::with_capacity(6);
    for axis in 0..3 {
        let mut mode = Array1::zeros(count * dof);
        for bead in 0..count {
            for atom in 0..atoms {
                mode[bead * dof + 3 * atom + axis] = sqrt_mass[3 * atom];
            }
        }
        modes.push(mode);
    }
    for axis in 0..3 {
        let mut mode = Array1::zeros(count * dof);
        for bead in 0..count {
            for atom in 0..atoms {
                let point = [
                    beads[(bead, 3 * atom)],
                    beads[(bead, 3 * atom + 1)],
                    beads[(bead, 3 * atom + 2)],
                ];
                let turned = cross(axis, point);
                for component in 0..3 {
                    mode[bead * dof + 3 * atom + component] = turned[component];
                }
            }
        }
        modes.push(mode);
    }
    modes
}

fn orthonormalize(modes: Vec<Array1<f64>>) -> Vec<Array1<f64>> {
    let mut kept: Vec<Array1<f64>> = Vec::new();
    for mut mode in modes {
        for earlier in &kept {
            let coeff = earlier.dot(&mode);
            mode.scaled_add(-coeff, earlier);
        }
        let norm = l2(mode.view());
        if norm > 0.0 {
            mode /= norm;
            kept.push(mode);
        }
    }
    kept
}

fn project(vector: &mut Array1<f64>, directions: &[Array1<f64>]) {
    for direction in directions {
        let coeff = direction.dot(vector);
        vector.scaled_add(-coeff, direction);
    }
}

/// Lanczos log-absolute-determinant on the complement of `gauges`.
///
/// A Krylov chain stops when its next off-diagonal is lost in the
/// roundoff of projecting the matvec (an invariant subspace, including
/// a repeated eigenvalue). The search then starts again from the first
/// coordinate axis still outside that subspace, and the determinants
/// multiply. Each chain is diagonalized as a tridiagonal.
fn projected_log_abs_det<F>(
    length: usize,
    gauges: &[Array1<f64>],
    mut apply: F,
) -> Result<f64, SaddleError>
where
    F: FnMut(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError>,
{
    let dim = length.checked_sub(gauges.len()).ok_or_else(|| {
        SaddleError::Invalid("more gauge modes than ring-polymer coordinates".into())
    })?;
    if dim < 1 {
        return Err(SaddleError::Invalid(
            "the fluctuation space is empty".into(),
        ));
    }
    let mut locked = Vec::with_capacity(dim);
    let mut log_det = 0.0;
    while locked.len() < dim {
        let start = complement_direction(length, gauges, &locked)?;
        let chain = lanczos_chain(
            length,
            gauges,
            &locked,
            start,
            dim - locked.len(),
            &mut apply,
        )?;
        log_det += tridiagonal_log_abs_det(&chain.alpha, &chain.beta)?;
        locked.extend(chain.vectors);
    }
    if !log_det.is_finite() {
        return Err(SaddleError::NonFinite("fluctuation determinant"));
    }
    Ok(log_det)
}

fn complement_direction(
    length: usize,
    gauges: &[Array1<f64>],
    locked: &[Array1<f64>],
) -> Result<Array1<f64>, SaddleError> {
    let span = gauges.len() + locked.len();
    let resolved = match span {
        0 => f64::EPSILON,
        _ => f64::EPSILON * (span as f64) * (span as f64),
    };
    for axis in 0..length {
        let mut probe = Array1::zeros(length);
        probe[axis] = 1.0;
        project(&mut probe, gauges);
        project(&mut probe, locked);
        project(&mut probe, gauges);
        project(&mut probe, locked);
        let norm = l2(probe.view());
        if norm > resolved {
            probe /= norm;
            return Ok(probe);
        }
    }
    Err(SaddleError::Invalid(
        "the fluctuation space closed before every mode was found".into(),
    ))
}

struct Chain {
    alpha: Vec<f64>,
    beta: Vec<f64>,
    vectors: Vec<Array1<f64>>,
}

fn lanczos_chain<F>(
    length: usize,
    gauges: &[Array1<f64>],
    locked: &[Array1<f64>],
    mut vector: Array1<f64>,
    room: usize,
    apply: &mut F,
) -> Result<Chain, SaddleError>
where
    F: FnMut(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError>,
{
    let mut previous = Array1::zeros(length);
    let mut beta_prev = 0.0;
    let mut chain = Chain {
        alpha: Vec::new(),
        beta: Vec::new(),
        vectors: Vec::new(),
    };
    while chain.vectors.len() < room {
        project(&mut vector, gauges);
        project(&mut vector, locked);
        project(&mut vector, &chain.vectors);
        let norm = l2(vector.view());
        if norm <= 0.0 || norm.is_nan() {
            return Err(SaddleError::Invalid(
                "a Lanczos vector collapsed onto the modes already removed".into(),
            ));
        }
        vector /= norm;
        chain.vectors.push(vector.clone());
        let mut image = apply(vector.view())?;
        if image.len() != length {
            return Err(SaddleError::Shape(
                "the fluctuation operator changed the vector length".into(),
            ));
        }
        project(&mut image, gauges);
        project(&mut image, locked);
        let applied = l2(image.view());
        let diagonal = vector.dot(&image);
        image.scaled_add(-diagonal, &vector);
        if !chain.alpha.is_empty() {
            image.scaled_add(-beta_prev, &previous);
        }
        for _pass in 0..2 {
            project(&mut image, gauges);
            project(&mut image, locked);
            project(&mut image, &chain.vectors);
        }
        if !diagonal.is_finite() || image.iter().any(|component| !component.is_finite()) {
            return Err(SaddleError::NonFinite("Lanczos projection"));
        }
        chain.alpha.push(diagonal);
        if chain.vectors.len() == room {
            break;
        }
        let off = l2(image.view());
        let removals = (gauges.len() + locked.len() + chain.vectors.len()) as f64;
        let roundoff = applied * f64::EPSILON * removals * removals;
        if off <= roundoff || off.is_nan() {
            break;
        }
        chain.beta.push(off);
        previous = vector;
        vector = image / off;
        beta_prev = off;
    }
    if chain.alpha.is_empty() {
        return Err(SaddleError::Invalid(
            "Lanczos produced no fluctuation direction".into(),
        ));
    }
    Ok(chain)
}

fn tridiagonal_log_abs_det(alpha: &[f64], beta: &[f64]) -> Result<f64, SaddleError> {
    if alpha.is_empty() || beta.len() + 1 != alpha.len() {
        return Err(SaddleError::Invalid(
            "a fluctuation determinant needs one off-diagonal between diagonals".into(),
        ));
    }
    let n = alpha.len();
    let mut tri = Array2::zeros((n, n));
    for i in 0..n {
        tri[(i, i)] = alpha[i];
        if i + 1 < n {
            tri[(i, i + 1)] = beta[i];
            tri[(i + 1, i)] = beta[i];
        }
    }
    let (values, _) = exact_eigh(tri.view())?;
    let mut sum = 0.0;
    for value in values.iter() {
        if *value == 0.0 || !value.is_finite() {
            return Err(SaddleError::Invalid(
                "the fluctuation operator has a vanishing eigenvalue".into(),
            ));
        }
        sum += value.abs().ln();
    }
    Ok(sum)
}

#[cfg(any(test, feature = "lineb-oracle"))]
fn image_hessians<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    images: ArrayView2<f64>,
    step: f64,
) -> Result<Vec<Array2<f64>>, SaddleError> {
    let mut hessians = Vec::with_capacity(images.nrows());
    for row in images.rows() {
        hessians.push(finite_difference_hessian(surface, masses, row, step)?);
    }
    Ok(hessians)
}

#[cfg(any(test, feature = "lineb-oracle"))]
fn finite_difference_hessian<S: PointSurface>(
    surface: &S,
    masses: &Masses,
    weighted: ArrayView1<f64>,
    step: f64,
) -> Result<Array2<f64>, SaddleError> {
    let cartesian = masses.to_cartesian(weighted)?;
    let dof = cartesian.len();
    let mut hessian = Array2::zeros((dof, dof));
    for column in 0..dof {
        let mut plus = cartesian.clone();
        let mut minus = cartesian.clone();
        plus[column] += step;
        minus[column] -= step;
        let (_, gradient_plus) = surface.eval(plus.view())?;
        let (_, gradient_minus) = surface.eval(minus.view())?;
        for row in 0..dof {
            hessian[(row, column)] = (gradient_plus[row] - gradient_minus[row]) / (2.0 * step);
        }
    }
    let mut weighted_hessian = Array2::zeros((dof, dof));
    for row in 0..dof {
        for column in 0..dof {
            let symmetric = 0.5 * (hessian[(row, column)] + hessian[(column, row)]);
            weighted_hessian[(row, column)] =
                symmetric / (masses.sqrt_mass[row] * masses.sqrt_mass[column]);
        }
    }
    Ok(weighted_hessian)
}

#[cfg(any(test, feature = "lineb-oracle"))]
fn interpolated_ring(
    hessians: &[Array2<f64>],
    knots: &[f64],
    parameters: &[f64],
    dof: usize,
    kappa: f64,
) -> Result<Array2<f64>, SaddleError> {
    let mut elements = Vec::with_capacity(dof * dof);
    for row in 0..dof {
        for column in 0..dof {
            let values: Vec<f64> = hessians
                .iter()
                .map(|hessian| hessian[(row, column)])
                .collect();
            elements.push(Spline::new(knots, &values));
        }
    }
    let count = parameters.len();
    let mut ring = Array2::zeros((count * dof, count * dof));
    for bead in 0..count {
        let lambda = parameters[bead];
        let previous = if bead == 0 { count - 1 } else { bead - 1 };
        let next = if bead + 1 == count { 0 } else { bead + 1 };
        for row in 0..dof {
            for column in 0..dof {
                let curvature = elements[row * dof + column].value(lambda);
                ring[(bead * dof + row, bead * dof + column)] = curvature;
            }
            ring[(bead * dof + row, bead * dof + row)] += 2.0 * kappa;
            ring[(bead * dof + row, previous * dof + row)] -= kappa;
            ring[(bead * dof + row, next * dof + row)] -= kappa;
        }
    }
    Ok(ring)
}

#[cfg(any(test, feature = "lineb-oracle"))]
fn dense_kept_log_abs_det(
    ring: ArrayView2<f64>,
    beads: &Array2<f64>,
    masses: &Masses,
) -> Result<f64, SaddleError> {
    let gauges = gauge_modes(beads, masses)?;
    let basis = fluctuation_basis(ring.nrows(), &gauges)?;
    let dim = basis.ncols();
    let mut reduced = Array2::zeros((dim, dim));
    for column in 0..dim {
        let image = ring.dot(&basis.column(column));
        for row in 0..dim {
            reduced[(row, column)] = basis.column(row).dot(&image);
        }
    }
    for row in 0..dim {
        for column in row + 1..dim {
            let mean = 0.5 * (reduced[(row, column)] + reduced[(column, row)]);
            reduced[(row, column)] = mean;
            reduced[(column, row)] = mean;
        }
    }
    let (values, _) = exact_eigh(reduced.view())?;
    let mut sum = 0.0;
    for value in values.iter() {
        if *value == 0.0 || !value.is_finite() {
            return Err(SaddleError::Invalid(
                "a kept ring-polymer eigenvalue vanishes".into(),
            ));
        }
        sum += value.abs().ln();
    }
    if !sum.is_finite() {
        return Err(SaddleError::NonFinite("fluctuation determinant"));
    }
    Ok(sum)
}

/// Orthonormal basis of the complement of `gauges`, one column each.
#[cfg(any(test, feature = "lineb-oracle"))]
fn fluctuation_basis(length: usize, gauges: &[Array1<f64>]) -> Result<Array2<f64>, SaddleError> {
    let dim = length.checked_sub(gauges.len()).ok_or_else(|| {
        SaddleError::Invalid("more gauge modes than ring-polymer coordinates".into())
    })?;
    if dim < 1 {
        return Err(SaddleError::Invalid(
            "the fluctuation space is empty".into(),
        ));
    }
    let resolved = f64::EPSILON * ((gauges.len() + dim) as f64).powi(2);
    let mut basis = Array2::zeros((length, dim));
    let mut filled = 0;
    for axis in 0..length {
        let mut probe = Array1::zeros(length);
        probe[axis] = 1.0;
        project(&mut probe, gauges);
        for column in 0..filled {
            let coeff = basis.column(column).dot(&probe);
            probe.scaled_add(-coeff, &basis.column(column));
        }
        project(&mut probe, gauges);
        let norm = l2(probe.view());
        if norm > resolved {
            probe /= norm;
            basis.column_mut(filled).assign(&probe);
            filled += 1;
            if filled == dim {
                return Ok(basis);
            }
        }
    }
    Err(SaddleError::Invalid(
        "the fluctuation space closed before every mode was found".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A free ring's spring Hessian has eigenvalues
    /// `4 κ sin²(π k / P)` on each coordinate. Removing the constant
    /// mode leaves `Π = κ^{P−1} P²`.
    #[test]
    fn spring_log_det_matches_the_sine_product() {
        let beads = 8;
        let dof = 1;
        let kappa = 2.5;
        let apply = |vector: ArrayView1<f64>| {
            let mut out = Array1::zeros(vector.len());
            for bead in 0..beads {
                let previous = if bead == 0 { beads - 1 } else { bead - 1 };
                let next = if bead + 1 == beads { 0 } else { bead + 1 };
                out[bead] = kappa * (2.0 * vector[bead] - vector[previous] - vector[next]);
            }
            Ok(out)
        };
        let mut constant = Array1::zeros(beads * dof);
        constant.fill(1.0);
        constant /= l2(constant.view());
        let log_det = projected_log_abs_det(beads, &[constant], apply).unwrap();
        let expected = (beads - 1) as f64 * kappa.ln() + 2.0 * (beads as f64).ln();
        // Relative to the expected log-determinant.
        let scale = expected.abs();
        assert!(
            (log_det - expected).abs() <= scale * f64::EPSILON * (beads as f64).powi(2),
            "{log_det} vs {expected}"
        );
    }

    /// Distinct diagonal entries have a Krylov chain of length one each.
    /// The product of the restarted chains is the product of the entries.
    #[test]
    fn restarted_lanczos_multiplies_distinct_eigenvalues() {
        let n = 6;
        let apply = |vector: ArrayView1<f64>| {
            let mut out = Array1::zeros(n);
            for index in 0..n {
                out[index] = (index + 1) as f64 * vector[index];
            }
            Ok(out)
        };
        let log_det = projected_log_abs_det(n, &[], apply).unwrap();
        let expected: f64 = (1..=n).map(|k| (k as f64).ln()).sum();
        // Relative to the expected log-determinant.
        let scale = expected.abs();
        assert!(
            (log_det - expected).abs() <= scale * f64::EPSILON * (n as f64).powi(2),
            "{log_det} vs {expected}"
        );
    }
}
