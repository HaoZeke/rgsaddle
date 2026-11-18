//! Search-direction coverage on the circle and the sphere.
//!
//! A saddle search starts from a displacement. The set of displacements
//! already tried is a set of points on `S^1` or `S^2`, and a basin of
//! attraction that none of them landed in has to sit in a gap between
//! those points. This module measures that gap and names the next
//! direction that cuts the largest one.
//!
//! Nothing here knows about rates, catalogs, or a potential. A saddle
//! search consumes the measure and the probe this file returns. The
//! file drops into rgsaddle unchanged: it depends only on `std`.

use std::fmt;

/// A direction list the coverage geometry cannot score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoverageError {
    /// Two slices that must be the same length were not.
    DimensionMismatch {
        /// Length the other slice has.
        expected: usize,
        /// Length that was supplied.
        got: usize,
    },
    /// A direction, label, or mesh failed a value check.
    InvalidInput {
        /// What failed the check.
        reason: String,
    },
}

impl fmt::Display for CoverageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidInput { reason } => write!(f, "coverage input invalid: {reason}"),
        }
    }
}

impl std::error::Error for CoverageError {}

/// Two directions closer than this are the same point on the circle.
const ANGLE_TIE: f64 = 1e-12;

/// Residual accepted from the 3×3 cone solver. It is the floating-point
/// tolerance of Cramer's rule, not a physical cutoff.
const CONE_RESIDUAL: f64 = 1e-8;

/// One arc between consecutive searched directions.
#[derive(Debug, Clone, PartialEq)]
pub struct CircleGap {
    /// Angle of the sample where the arc starts, in radians on `[0, 2π)`.
    pub start: f64,
    /// Forward length of the arc, in radians.
    pub length: f64,
    /// Label of the sample at the start of the arc.
    pub left_label: i64,
    /// Label of the sample at the end of the arc.
    pub right_label: i64,
}

/// Covered and unresolved arcs on the circle of initial directions.
#[derive(Debug, Clone, PartialEq)]
pub struct CircleCoverage {
    /// Fraction of the circle that can still hide a basin.
    pub uncovered_measure: f64,
    /// Fraction already pinned down by the searched directions.
    pub covered_measure: f64,
    /// Unresolved arcs only. A resolved arc is omitted.
    pub gaps: Vec<CircleGap>,
}

/// Covered and unresolved quadrature points on the sphere.
#[derive(Debug, Clone, PartialEq)]
pub struct SphereCoverage {
    /// Fraction of the mesh that can still hide a basin.
    pub uncovered_measure: f64,
    /// Fraction of the mesh pinned down by the searched directions.
    pub covered_measure: f64,
    /// Quadrature weight of one mesh point, `1/n`.
    pub resolution: f64,
    /// Mesh indices that stayed unresolved.
    pub unresolved: Vec<usize>,
}

/// Coverage of searched angles on the circle.
///
/// Angles are taken modulo `2π` and sorted. The arc from each sample to
/// the next is unresolved when its endpoint labels differ, because a cut
/// between two basins can hide another basin, or when the labels agree
/// and the arc is a longest arc of that kind. The longest same-label arc
/// is the complement of the shortest arc that contains every sample of
/// the label; tied longest arcs are all unresolved. The uncovered
/// measure is the total length of those arcs divided by the sum of every
/// arc, so the two measures add to one even when the lengths do not land
/// on `2π` exactly.
///
/// No searches leave the whole circle uncovered. One search leaves one
/// arc of `2π`. Two searches of the same label at `0` and `1` leave the
/// long arc. Two searches of different labels leave both arcs.
///
/// # Errors
///
/// [`CoverageError::DimensionMismatch`] when the two slices differ in
/// length. [`CoverageError::InvalidInput`] when a label is not finite in
/// the angle, or when two different labels sit on the same angle.
pub fn circle_coverage(angles: &[f64], labels: &[i64]) -> Result<CircleCoverage, CoverageError> {
    if angles.len() != labels.len() {
        return Err(CoverageError::DimensionMismatch {
            expected: angles.len(),
            got: labels.len(),
        });
    }
    if angles.is_empty() {
        return Ok(CircleCoverage {
            uncovered_measure: 1.0,
            covered_measure: 0.0,
            gaps: Vec::new(),
        });
    }
    let tau = std::f64::consts::TAU;
    let mut points = Vec::with_capacity(angles.len());
    for (&angle, &label) in angles.iter().zip(labels) {
        if !angle.is_finite() {
            return Err(CoverageError::InvalidInput {
                reason: format!("search angle {angle} is not finite"),
            });
        }
        let mut wrapped = angle.rem_euclid(tau);
        if wrapped == tau {
            wrapped = 0.0;
        }
        points.push((wrapped, label));
    }
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let n = points.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let separation = if j == 0 {
            points[0].0 + tau - points[i].0
        } else {
            points[j].0 - points[i].0
        };
        if separation <= ANGLE_TIE && points[i].1 != points[j].1 {
            return Err(CoverageError::InvalidInput {
                reason: format!(
                    "labels {} and {} occupy the same search angle",
                    points[i].1, points[j].1
                ),
            });
        }
    }
    let mut drafts = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        let mut length = if j == 0 {
            points[0].0 + tau - points[i].0
        } else {
            points[j].0 - points[i].0
        };
        if !length.is_finite() || length < 0.0 {
            length = 0.0;
        }
        drafts.push(CircleGap {
            start: points[i].0,
            length,
            left_label: points[i].1,
            right_label: points[j].1,
        });
    }
    let total: f64 = drafts.iter().map(|g| g.length).sum();
    if !(total > 0.0) {
        return Err(CoverageError::InvalidInput {
            reason: "circle gaps summed to zero".into(),
        });
    }
    let max_intra = drafts
        .iter()
        .filter(|g| g.left_label == g.right_label)
        .map(|g| g.length)
        .fold(f64::NEG_INFINITY, f64::max);
    let gaps: Vec<CircleGap> = drafts
        .into_iter()
        .filter(|g| g.left_label != g.right_label || g.length + ANGLE_TIE >= max_intra)
        .collect();
    let unresolved: f64 = gaps.iter().map(|g| g.length).sum();
    let uncovered = unresolved / total;
    Ok(CircleCoverage {
        uncovered_measure: uncovered,
        covered_measure: 1.0 - uncovered,
        gaps,
    })
}

/// Direction that splits the longest unresolved arc.
///
/// With no searches the direction is `(1, 0, 0)`, the angle the circle
/// is measured from. Otherwise it is the midpoint of a longest
/// unresolved arc, returned as `(cos θ, sin θ, 0)`.
pub fn circle_probe(coverage: &CircleCoverage) -> [f64; 3] {
    let Some(gap) = coverage.gaps.iter().max_by(|a, b| {
        a.length
            .partial_cmp(&b.length)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) else {
        return [1.0, 0.0, 0.0];
    };
    let theta = gap.start + 0.5 * gap.length;
    [theta.cos(), theta.sin(), 0.0]
}

/// Coverage of searched directions against a quadrature mesh on the sphere.
///
/// `mesh` is the caller's quadrature. Each mesh point is covered when
/// it lies in the positive cone of the samples that carry the nearest
/// sample's label, and it does not sit on the bisector with a different
/// label. Cone membership tests subsets of one, two, and three samples
/// (Carathéodory in three dimensions) by Cramer's rule. The uncovered
/// measure is the fraction of unresolved mesh points, and `resolution`
/// is the weight `1/n` of one point.
///
/// Directions are normalised here, so a finite-radius perturbation mesh
/// can be passed through unchanged.
///
/// # Errors
///
/// [`CoverageError::InvalidInput`] when `mesh` is empty, a vector has
/// zero length, or two different labels share a direction.
/// [`CoverageError::DimensionMismatch`] when the direction and label slices
/// differ in length.
pub fn sphere_coverage(
    directions: &[[f64; 3]],
    labels: &[i64],
    mesh: &[[f64; 3]],
) -> Result<SphereCoverage, CoverageError> {
    if directions.len() != labels.len() {
        return Err(CoverageError::DimensionMismatch {
            expected: directions.len(),
            got: labels.len(),
        });
    }
    if mesh.is_empty() {
        return Err(CoverageError::InvalidInput {
            reason: "sphere coverage needs the perturbation mesh".into(),
        });
    }
    let samples = normalise_labelled(directions, labels)?;
    let quadrature: Vec<[f64; 3]> = mesh
        .iter()
        .map(|v| unit(*v))
        .collect::<Result<Vec<_>, _>>()?;
    let mut unresolved = Vec::new();
    for (index, point) in quadrature.iter().enumerate() {
        if !mesh_point_covered(point, &samples) {
            unresolved.push(index);
        }
    }
    let n = quadrature.len() as f64;
    let uncovered = unresolved.len() as f64 / n;
    Ok(SphereCoverage {
        uncovered_measure: uncovered,
        covered_measure: 1.0 - uncovered,
        resolution: 1.0 / n,
        unresolved,
    })
}

/// Unresolved mesh point farthest from the samples already searched.
///
/// Distance is the axial angle `acos(|u·v| / (|u||v|))`. Points tied
/// within [`ANGLE_TIE`] are broken by a Weyl
/// draw from `seed`, so an empty search does not always restart on the
/// first mesh vector. With no unresolved point the first mesh vector is
/// returned.
pub fn sphere_probe(
    directions: &[[f64; 3]],
    mesh: &[[f64; 3]],
    unresolved: &[usize],
    seed: u64,
) -> Result<[f64; 3], CoverageError> {
    if mesh.is_empty() {
        return Err(CoverageError::InvalidInput {
            reason: "sphere coverage needs the perturbation mesh".into(),
        });
    }
    if unresolved.is_empty() {
        return unit(mesh[0]);
    }
    let samples: Vec<[f64; 3]> = directions
        .iter()
        .map(|v| unit(*v))
        .collect::<Result<Vec<_>, _>>()?;
    let mut best = f64::NEG_INFINITY;
    let mut tied: Vec<usize> = Vec::new();
    for &index in unresolved {
        let point = unit(mesh[index])?;
        let nearest = if samples.is_empty() {
            f64::NEG_INFINITY
        } else {
            samples
                .iter()
                .map(|s| axial_angle(point, *s).unwrap_or(std::f64::consts::FRAC_PI_2))
                .fold(f64::INFINITY, f64::min)
        };
        let distance = nearest;
        if distance > best + ANGLE_TIE {
            best = distance;
            tied.clear();
            tied.push(index);
        } else if distance >= best - ANGLE_TIE {
            best = best.max(distance);
            tied.push(index);
        }
    }
    if tied.is_empty() {
        return unit(mesh[unresolved[0]]);
    }
    let pick = if tied.len() == 1 {
        0
    } else {
        (weyl_uniform(seed, 0) * tied.len() as f64).floor() as usize
    };
    let pick = pick.min(tied.len() - 1);
    unit(mesh[tied[pick]])
}

/// Mesh points that no searched direction lands on.
///
/// A point is searched when it lies on the same axis as a sample, to
/// within [`CONE_RESIDUAL`]. The axis is undirected: the distance is
/// `acos(|u·v| / (|u||v|))`, so the opposite displacement is the same
/// search. This is not cone coverage: a quadrature point can sit inside
/// a sampled cone and still be a direction nobody evaluated.
///
/// # Errors
///
/// [`CoverageError::InvalidInput`] when a direction or a mesh vector has
/// zero length or a non-finite component.
pub fn unsearched_indices(
    directions: &[[f64; 3]],
    mesh: &[[f64; 3]],
) -> Result<Vec<usize>, CoverageError> {
    let samples = directions
        .iter()
        .map(|direction| unit(*direction))
        .collect::<Result<Vec<_>, _>>()?;
    let mut missed = Vec::new();
    for (index, point) in mesh.iter().enumerate() {
        let point = unit(*point)?;
        let searched = samples
            .iter()
            .any(|sample| dot(point, *sample).abs() >= 1.0 - CONE_RESIDUAL);
        if !searched {
            missed.push(index);
        }
    }
    Ok(missed)
}

/// Uniform draw on `[0, 1)` from the Weyl sequence of `seed`.
///
/// `n` is the step. The step size is the golden-ratio conjugate. The seed
/// is mixed before it becomes the offset: dividing a small integer seed by
/// `u64::MAX` puts every ordinary seed on the same sequence, so every
/// search walks the same directions.
pub fn weyl_uniform(seed: u64, n: u64) -> f64 {
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mixed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let alpha = (mixed >> 11) as f64 / ((1u64 << 53) as f64);
    (alpha + n as f64 * phi).rem_euclid(1.0)
}

/// One standard normal from two Weyl draws, by Box-Muller.
pub fn weyl_normal(seed: u64, step: u64) -> f64 {
    let u1 = weyl_uniform(seed, step).max(f64::MIN_POSITIVE);
    let u2 = weyl_uniform(seed, step + 1);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn mesh_point_covered(point: &[f64; 3], samples: &[([f64; 3], i64)]) -> bool {
    if samples.is_empty() {
        return false;
    }
    let mut best = f64::NEG_INFINITY;
    for (vector, _) in samples {
        best = best.max(dot(*point, *vector).abs());
    }
    let mut labels = Vec::new();
    for (vector, label) in samples {
        if dot(*point, *vector).abs() + ANGLE_TIE >= best && !labels.contains(label) {
            labels.push(*label);
        }
    }
    if labels.len() != 1 {
        return false;
    }
    let label = labels[0];
    let owned: Vec<[f64; 3]> = samples
        .iter()
        .filter(|(_, sample_label)| *sample_label == label)
        .map(|(vector, _)| *vector)
        .collect();
    in_positive_cone(point, &owned)
}

fn in_positive_cone(point: &[f64; 3], vectors: &[[f64; 3]]) -> bool {
    for vector in vectors {
        if dot(*point, *vector).abs() >= 1.0 - CONE_RESIDUAL {
            return true;
        }
    }
    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            if in_cone2(point, vectors[i], vectors[j]) {
                return true;
            }
        }
    }
    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            for k in (j + 1)..vectors.len() {
                if in_cone3(point, vectors[i], vectors[j], vectors[k]) {
                    return true;
                }
            }
        }
    }
    false
}

fn in_cone2(point: &[f64; 3], v: [f64; 3], w: [f64; 3]) -> bool {
    let n = cross(v, w);
    let nn = dot(n, n);
    if nn < CONE_RESIDUAL * CONE_RESIDUAL {
        return false;
    }
    if dot(*point, n).abs() > CONE_RESIDUAL {
        return false;
    }
    let a = dot(cross(*point, w), n) / nn;
    let b = dot(cross(v, *point), n) / nn;
    a >= -CONE_RESIDUAL && b >= -CONE_RESIDUAL
}

fn in_cone3(point: &[f64; 3], v: [f64; 3], w: [f64; 3], u: [f64; 3]) -> bool {
    let d = det3(v, w, u);
    if d.abs() < 1e-14 {
        return false;
    }
    let a = det3(*point, w, u) / d;
    let b = det3(v, *point, u) / d;
    let c = det3(v, w, *point) / d;
    if a < -CONE_RESIDUAL || b < -CONE_RESIDUAL || c < -CONE_RESIDUAL {
        return false;
    }
    let recon = [
        a * v[0] + b * w[0] + c * u[0],
        a * v[1] + b * w[1] + c * u[1],
        a * v[2] + b * w[2] + c * u[2],
    ];
    let residual = (0..3)
        .map(|i| recon[i] - point[i])
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    residual <= CONE_RESIDUAL
}

fn normalise_labelled(
    directions: &[[f64; 3]],
    labels: &[i64],
) -> Result<Vec<([f64; 3], i64)>, CoverageError> {
    let mut out = Vec::with_capacity(directions.len());
    for (direction, &label) in directions.iter().zip(labels) {
        let vector = unit(*direction)?;
        for (previous, previous_label) in &out {
            if dot(vector, *previous) >= 1.0 - ANGLE_TIE && label != *previous_label {
                return Err(CoverageError::InvalidInput {
                    reason: format!(
                        "labels {previous_label} and {label} occupy the same search direction"
                    ),
                });
            }
        }
        out.push((vector, label));
    }
    Ok(out)
}

fn unit(v: [f64; 3]) -> Result<[f64; 3], CoverageError> {
    if v.iter().any(|x| !x.is_finite()) {
        return Err(CoverageError::InvalidInput {
            reason: format!("direction {v:?} is not finite"),
        });
    }
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if !(norm > 0.0) {
        return Err(CoverageError::InvalidInput {
            reason: "search direction has zero length".into(),
        });
    }
    Ok([v[0] / norm, v[1] / norm, v[2] / norm])
}

/// Whether a min-mode start has crossed the inflection of the convex bowl.
///
/// Inside the bowl the Hessian is positive definite and a min-mode search
/// funnels onto the softest axis (Pedersen and Luiser, J. Chem. Phys. 141,
/// 024109 (2014); Gunde et al., J. Chem. Phys. 160, 232501 (2024); the
/// κ-dimer of Xiao, Wu and Henkelman, 2014). The boundary is the sign
/// change of the lowest curvature. A negative curvature means the start is
/// outside that bowl. Zero is the boundary itself and is not counted as a
/// crossing.
///
/// # Errors
///
/// [`CoverageError::InvalidInput`] when `curvatures` is empty or holds a
/// non-finite value.
pub fn crossed_inflection(curvatures: &[f64]) -> Result<bool, CoverageError> {
    if curvatures.is_empty() || curvatures.iter().any(|value| !value.is_finite()) {
        return Err(CoverageError::InvalidInput {
            reason: "curvatures must be a non-empty list of finite values".into(),
        });
    }
    Ok(curvatures.iter().copied().fold(f64::INFINITY, f64::min) < 0.0)
}

/// Axial angle between two displacements, in radians.
///
/// A kick and its opposite are one search direction, so the distance is
/// `acos(|u·v| / (|u||v|))` and lies in `[0, π/2]`.
///
/// # Errors
///
/// [`CoverageError::InvalidInput`] when either vector has zero length or
/// a non-finite component.
pub fn axial_angle(u: [f64; 3], v: [f64; 3]) -> Result<f64, CoverageError> {
    let u = unit(u)?;
    let v = unit(v)?;
    let cosine = dot(u, v).abs().clamp(0.0, 1.0);
    Ok(cosine.acos())
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn det3(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_label_leaves_the_long_arc() {
        let coverage = circle_coverage(&[0.0, 1.0], &[0, 0]).unwrap();
        let expected = (std::f64::consts::TAU - 1.0) / std::f64::consts::TAU;
        assert!((coverage.uncovered_measure - expected).abs() < 1e-12);
        assert!((coverage.covered_measure + coverage.uncovered_measure - 1.0).abs() < 1e-12);
        assert_eq!(coverage.gaps.len(), 1);
        assert!((coverage.gaps[0].length - (std::f64::consts::TAU - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn a_cut_leaves_both_arcs() {
        let coverage = circle_coverage(&[0.0, std::f64::consts::PI], &[0, 1]).unwrap();
        assert!((coverage.uncovered_measure - 1.0).abs() < 1e-12);
        assert_eq!(coverage.gaps.len(), 2);
    }

    #[test]
    fn tetrahedron_covers_the_axes() {
        let s = 1.0 / 3.0_f64.sqrt();
        let tet = [[s, s, s], [s, -s, -s], [-s, s, -s], [-s, -s, s]];
        let axes = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let coverage = sphere_coverage(&tet, &[0, 0, 0, 0], &axes).unwrap();
        assert_eq!(coverage.uncovered_measure, 0.0);
        assert!((coverage.resolution - 1.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn a_covered_quadrature_point_can_still_be_unsearched() {
        let s = 1.0 / 3.0_f64.sqrt();
        let tet = [[s, s, s], [s, -s, -s], [-s, s, -s], [-s, -s, s]];
        let axes = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let coverage = sphere_coverage(&tet, &[0, 0, 0, 0], &axes).unwrap();
        assert_eq!(coverage.uncovered_measure, 0.0);
        let missed = unsearched_indices(&tet, &axes).unwrap();
        assert_eq!(missed, (0..axes.len()).collect::<Vec<_>>());
        assert!(unsearched_indices(&axes, &axes).unwrap().is_empty());
    }

    #[test]
    fn adjacent_seeds_probe_different_mesh_points() {
        let mesh = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ];
        let unresolved: Vec<usize> = (0..mesh.len()).collect();
        let first = sphere_probe(&[], &mesh, &unresolved, 0).unwrap();
        let second = sphere_probe(&[], &mesh, &unresolved, 1).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn opposite_displacements_are_one_axis() {
        let angle = axial_angle([1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]).unwrap();
        assert!(angle.abs() < 1e-12);
        let right = axial_angle([1.0, 0.0, 0.0], [0.0, 3.0, 0.0]).unwrap();
        assert!((right - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        // Searching +x evaluates the axis, so -x is not an unsearched direction.
        let missed =
            unsearched_indices(&[[1.0, 0.0, 0.0]], &[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]).unwrap();
        assert!(missed.is_empty());
    }

    #[test]
    fn a_near_full_turn_uses_the_short_arc() {
        // The circle is a 1-d torus. 0 and 2π − 0.2 are 0.2 apart, not
        // almost a full turn. The unresolved gap of one label is the long arc.
        let tau = std::f64::consts::TAU;
        let coverage = circle_coverage(&[0.0, tau - 0.2], &[0, 0]).unwrap();
        assert_eq!(coverage.gaps.len(), 1);
        assert!((coverage.gaps[0].length - (tau - 0.2)).abs() < 1e-9);
    }

    #[test]
    fn an_empty_mesh_is_an_error() {
        let err = sphere_coverage(&[[1.0, 0.0, 0.0]], &[0], &[]).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("perturbation mesh"), "{text}");
    }

    #[test]
    fn a_negative_curvature_has_crossed_the_inflection() {
        assert!(!crossed_inflection(&[0.4, 1.2, 3.0]).unwrap());
        assert!(!crossed_inflection(&[0.0, 0.2]).unwrap());
        assert!(crossed_inflection(&[0.4, -1.0e-9, 2.0]).unwrap());
        assert!(crossed_inflection(&[]).is_err());
        assert!(crossed_inflection(&[1.0, f64::NAN]).is_err());
    }
}
