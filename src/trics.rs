//! Translation rotation internal coordinates for path construction.
//!
//! The primitive coordinates follow the TRIC construction of Wang and
//! Song, J. Chem. Phys. 144, 214108 (2016).  Bonded fragments carry
//! ordinary distances, angles and dihedrals together with three
//! translations and three rotations.  Eigenvectors of the primitive
//! Wilson metric form a nonredundant coordinate system.
//!
//! The segmented interpolation follows the algorithmic ideas in the
//! BSD licensed geomeTRIC reference implementation.  No geomeTRIC
//! source is included here.

use std::cmp::Ordering;

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::SaddleError;
use crate::internal::{CartAxis, Rotation, Translation};

const DEFAULT_BOND_FACTOR: f64 = 1.35;
const DEFAULT_RANK_TOLERANCE: f64 = 0.000_000_000_1;
const DIFFERENCE_STEP: f64 = 0.000_001;
const VECTOR_TOLERANCE: f64 = 0.000_000_000_001;
const JACOBI_SWEEPS_PER_DIMENSION: usize = 24;
const ANGLE_WARNING_COSINE: f64 = 0.906_307_787_036_649_9;
const ANGLE_UNSTABLE_COSINE: f64 = 0.965_925_826_289_068_3;
const ANGLE_FAILURE_COSINE: f64 = 0.996_194_698_091_745_5;
const DEFAULT_INVERSION_TOLERANCE: f64 = 0.000_000_01;
const DEFAULT_INVERSION_MOVE: f64 = 0.1;
/// Paper convergence threshold for the maximum atom splice mismatch.
pub const TRICS_SPLICE_TOLERANCE_ANGSTROM: f64 = 0.001_8;

/// Controls for constructing a nonredundant TRIC system.
///
/// The bond graph is inferred from mutual nearest neighbour scales.
/// Explicit bonds replace that inference when supplied.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TricConfig {
    /// Multiplicative threshold used by inferred bonds.
    pub bond_factor: f64,
    /// Relative eigenvalue threshold for the primitive Wilson metric.
    pub rank_tolerance: f64,
    /// Optional zero based atom pairs defining the bond graph.
    pub bonds: Option<Vec<(usize, usize)>>,
}

with_setters!(TricConfig {
    with_bond_factor => bond_factor: val f64,
    with_rank_tolerance => rank_tolerance: val f64,
    with_bonds => bonds: val Option<Vec<(usize, usize)>>,
});

impl Default for TricConfig {
    fn default() -> Self {
        Self {
            bond_factor: DEFAULT_BOND_FACTOR,
            rank_tolerance: DEFAULT_RANK_TOLERANCE,
            bonds: None,
        }
    }
}

/// Controls for iterative internal to Cartesian inversion.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TricInversionConfig {
    /// Maximum Newton iterations for one Cartesian image.
    pub max_iterations: usize,
    /// Largest internal coordinate residual accepted.
    pub tolerance: f64,
    /// Largest displacement of one atom in one iteration.
    pub max_atom_move: f64,
}

with_setters!(TricInversionConfig {
    with_max_iterations => max_iterations: val usize,
    with_tolerance => tolerance: val f64,
    with_max_atom_move => max_atom_move: val f64,
});

impl Default for TricInversionConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: DEFAULT_INVERSION_TOLERANCE,
            max_atom_move: DEFAULT_INVERSION_MOVE,
        }
    }
}

/// Controls for TRICS path construction and splicing.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TricsConfig {
    /// Coordinate construction controls.
    pub coordinates: TricConfig,
    /// Internal to Cartesian inversion controls.
    pub inversion: TricInversionConfig,
    /// Largest diagnostic level considered stable.
    pub diagnostic_threshold: u8,
    /// Maximum atom mismatch that ends endpoint splicing.
    pub splice_tolerance: f64,
    /// Maximum endpoint splice iterations.
    pub max_splice_iterations: usize,
}

with_setters!(TricsConfig {
    with_coordinates => coordinates: val TricConfig,
    with_inversion => inversion: val TricInversionConfig,
    with_diagnostic_threshold => diagnostic_threshold: val u8,
    with_splice_tolerance => splice_tolerance: val f64,
    with_max_splice_iterations => max_splice_iterations: val usize,
});

impl Default for TricsConfig {
    fn default() -> Self {
        Self {
            coordinates: TricConfig::default(),
            inversion: TricInversionConfig::default(),
            diagnostic_threshold: 1,
            splice_tolerance: TRICS_SPLICE_TOLERANCE_ANGSTROM,
            max_splice_iterations: 1000,
        }
    }
}

/// One overlapping segment selected from the diagnostic matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct TricsSegment {
    /// First image in the segment.
    pub start: usize,
    /// Last image in the segment.
    pub end: usize,
    /// Path image whose TRIC system interpolates this segment.
    pub system: usize,
}

/// A completed TRICS path and its splice metadata.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TricsReport {
    /// Interpolated Cartesian images.
    pub path: Array2<f64>,
    /// Overlapping diagnostic segments.
    pub segments: Vec<TricsSegment>,
    /// Endpoint splice iterations. Zero means one system covered the path.
    pub splice_iterations: usize,
}

/// One primitive coordinate used to span a TRIC system.
#[derive(Clone, Debug)]
pub enum TricPrimitive {
    /// Distance between two atoms.
    Distance(usize, usize),
    /// Angle `a-b-c`.
    Angle(usize, usize, usize),
    /// One of two bends spanning a linear angle.
    LinearBend(usize, usize, usize, [f64; 3]),
    /// Dihedral `a-b-c-d`.
    Dihedral(usize, usize, usize, usize),
    /// Mean fragment position along one Cartesian axis.
    Translation(Translation),
    /// Fragment orientation relative to the system reference.
    Rotation(Rotation),
}

impl TricPrimitive {
    /// Coordinate value at a packed `3N` Cartesian geometry.
    pub fn value(&self, positions: ArrayView1<'_, f64>) -> Result<f64, SaddleError> {
        check_positions(positions)?;
        match self {
            Self::Distance(a, b) => distance(positions, *a, *b),
            Self::Angle(a, b, c) => angle(positions, *a, *b, *c),
            Self::LinearBend(a, b, c, direction) => linear_bend(positions, *a, *b, *c, *direction),
            Self::Dihedral(a, b, c, d) => dihedral(positions, *a, *b, *c, *d),
            Self::Translation(coordinate) => coordinate.value(positions),
            Self::Rotation(coordinate) => coordinate.value(positions),
        }
    }

    /// Cartesian derivative of this coordinate.
    pub fn derivative(&self, positions: ArrayView1<'_, f64>) -> Result<Array1<f64>, SaddleError> {
        check_positions(positions)?;
        match self {
            Self::Distance(a, b) => distance_derivative(positions, *a, *b),
            Self::Angle(a, b, c) => angle_derivative(positions, *a, *b, *c),
            Self::Translation(coordinate) => coordinate.gradient(positions),
            Self::Rotation(coordinate) => coordinate.gradient(positions),
            Self::Dihedral(..) | Self::LinearBend(..) => central_derivative(self, positions),
        }
    }

    /// Numerical stability level from zero through three.
    ///
    /// Zero is regular. One warns that an angular primitive is close
    /// to linear, two marks it unstable under small perturbations, and
    /// three marks imminent coordinate failure.
    pub fn diagnostic(&self, positions: ArrayView1<'_, f64>) -> u8 {
        match self {
            Self::Angle(a, b, c) => angular_diagnostic(positions, &[(*a, *b, *c)]),
            Self::Dihedral(a, b, c, d) => {
                angular_diagnostic(positions, &[(*a, *b, *c), (*b, *c, *d)])
            }
            Self::LinearBend(a, b, c, _) => linear_angular_diagnostic(positions, *a, *b, *c),
            Self::Distance(a, b) => match distance(positions, *a, *b) {
                Ok(_) => 0,
                Err(_) => 3,
            },
            Self::Translation(_) | Self::Rotation(_) => 0,
        }
    }

    fn difference(
        &self,
        left: ArrayView1<'_, f64>,
        right: ArrayView1<'_, f64>,
    ) -> Result<f64, SaddleError> {
        let mut difference = self.value(left)? - self.value(right)?;
        if matches!(self, Self::Dihedral(..)) {
            let period = std::f64::consts::TAU;
            if difference > std::f64::consts::PI {
                difference -= period;
            } else if difference < -std::f64::consts::PI {
                difference += period;
            }
        }
        Ok(difference)
    }
}

/// A nonredundant TRIC basis tied to one reference geometry.
#[derive(Clone, Debug)]
pub struct TricSystem {
    reference: Array1<f64>,
    primitives: Vec<TricPrimitive>,
    basis: Array2<f64>,
}

impl TricSystem {
    /// Build a TRIC system from a packed `3N` reference geometry.
    pub fn new(reference: ArrayView1<'_, f64>, config: &TricConfig) -> Result<Self, SaddleError> {
        check_positions(reference)?;
        check_config(config)?;
        let n_atoms = reference.len() / 3;
        let bonds = match &config.bonds {
            Some(bonds) => validate_bonds(bonds, n_atoms)?,
            None => infer_bonds(reference, config.bond_factor)?,
        };
        let fragments = fragments(n_atoms, &bonds);
        let mut primitives = molecular_primitives(&bonds, n_atoms, reference)?;
        for fragment in fragments {
            for axis in CartAxis::ALL {
                primitives.push(TricPrimitive::Translation(Translation::new(
                    fragment.clone(),
                    axis,
                )?));
            }
            if fragment.len() > 1 {
                let refpos = fragment_positions(reference, &fragment);
                for axis in CartAxis::ALL {
                    primitives.push(TricPrimitive::Rotation(Rotation::new(
                        fragment.clone(),
                        axis,
                        refpos.clone(),
                    )?));
                }
            }
        }
        let primitive_b = primitive_wilson_b(&primitives, reference)?;
        let metric = primitive_b.dot(&primitive_b.t());
        let basis = nonredundant_basis(&metric, config.rank_tolerance);
        if basis.ncols() == 0 {
            return Err(SaddleError::Invalid(
                "TRIC construction found no independent coordinates".into(),
            ));
        }
        Ok(Self {
            reference: reference.to_owned(),
            primitives,
            basis,
        })
    }

    /// Reference geometry used to define rotations and delocalization.
    pub fn reference(&self) -> ArrayView1<'_, f64> {
        self.reference.view()
    }

    /// Primitive coordinates before delocalization.
    pub fn primitives(&self) -> &[TricPrimitive] {
        &self.primitives
    }

    /// Primitive to nonredundant transformation, one coordinate per column.
    pub fn basis(&self) -> &Array2<f64> {
        &self.basis
    }

    /// Number of nonredundant coordinates.
    pub fn dimension(&self) -> usize {
        self.basis.ncols()
    }

    /// Nonredundant coordinate difference `q(left) - q(right)`.
    pub fn difference(
        &self,
        left: ArrayView1<'_, f64>,
        right: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        check_same_positions(left, right)?;
        let mut primitive = Array1::zeros(self.primitives.len());
        for (slot, coordinate) in primitive.iter_mut().zip(&self.primitives) {
            *slot = coordinate.difference(left, right)?;
        }
        Ok(self.basis.t().dot(&primitive))
    }

    /// Wilson B matrix of the nonredundant coordinates.
    pub fn wilson_b(&self, positions: ArrayView1<'_, f64>) -> Result<Array2<f64>, SaddleError> {
        let primitive = primitive_wilson_b(&self.primitives, positions)?;
        Ok(self.basis.t().dot(&primitive))
    }

    /// Highest primitive stability diagnostic at `positions`.
    pub fn diagnostic(&self, positions: ArrayView1<'_, f64>) -> u8 {
        self.primitives
            .iter()
            .map(|primitive| primitive.diagnostic(positions))
            .max()
            .map_or(0, |value| value)
    }

    /// Iteratively invert an internal displacement from `start`.
    pub fn back_transform(
        &self,
        start: ArrayView1<'_, f64>,
        displacement: ArrayView1<'_, f64>,
        config: &TricInversionConfig,
    ) -> Result<Array1<f64>, SaddleError> {
        check_positions(start)?;
        check_inversion_config(config)?;
        if start.len() != self.reference.len() || displacement.len() != self.dimension() {
            return Err(SaddleError::Shape(
                "TRIC inversion dimensions do not match the coordinate system".into(),
            ));
        }
        let mut positions = start.to_owned();
        for _ in 0..config.max_iterations {
            let actual = self.difference(positions.view(), start)?;
            let residual = &displacement - &actual;
            let residual_norm = residual.dot(&residual).sqrt();
            if residual_norm <= config.tolerance {
                return Ok(positions);
            }
            let b = self.wilson_b(positions.view())?;
            let gram = b.dot(&b.t());
            let multipliers = solve_metric(&gram, &residual)?;
            let step = cap_atom_step(b.t().dot(&multipliers), config.max_atom_move);
            positions += &step;
            if positions.iter().any(|value| !value.is_finite()) {
                return Err(SaddleError::NonFinite("TRIC back transformation"));
            }
        }
        Err(SaddleError::Solver(
            "TRIC back transformation did not converge".into(),
        ))
    }
}

/// Interpolate one segment in a fixed TRIC metric.
///
/// The endpoints are copied exactly. Each interior image advances by
/// the same internal displacement and is recovered by iterative
/// internal to Cartesian inversion.
pub fn interpolate_tric_segment(
    system: &TricSystem,
    left: ArrayView1<'_, f64>,
    right: ArrayView1<'_, f64>,
    n_images: usize,
    config: &TricInversionConfig,
) -> Result<Array2<f64>, SaddleError> {
    check_same_positions(left, right)?;
    if n_images < 2 {
        return Err(SaddleError::Shape(
            "TRIC interpolation needs at least two images".into(),
        ));
    }
    match interpolate_one_direction(system, left, right, n_images, config) {
        Ok(path) => Ok(path),
        Err(forward_error) => {
            match interpolate_one_direction(system, right, left, n_images, config) {
                Ok(backward) => Ok(Array2::from_shape_fn(
                    backward.raw_dim(),
                    |(row, column)| backward[(n_images - row - 1, column)],
                )),
                Err(_) => Err(forward_error),
            }
        }
    }
}

fn interpolate_one_direction(
    system: &TricSystem,
    left: ArrayView1<'_, f64>,
    right: ArrayView1<'_, f64>,
    n_images: usize,
    config: &TricInversionConfig,
) -> Result<Array2<f64>, SaddleError> {
    let total = system.difference(right, left)?;
    let increment = total / (n_images - 1) as f64;
    let mut path = Array2::zeros((n_images, left.len()));
    path.row_mut(0).assign(&left);
    let mut current = left.to_owned();
    for image in 1..n_images - 1 {
        current = system.back_transform(current.view(), increment.view(), config)?;
        path.row_mut(image).assign(&current);
    }
    path.row_mut(n_images - 1).assign(&right);
    Ok(path)
}

/// Build one TRIC system per image of an initial path.
pub fn tric_systems(
    path: ndarray::ArrayView2<'_, f64>,
    config: &TricConfig,
) -> Result<Vec<TricSystem>, SaddleError> {
    if path.nrows() < 2 || path.ncols() == 0 || !path.ncols().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "a TRIC path needs at least two 3N images".into(),
        ));
    }
    path.outer_iter()
        .map(|image| TricSystem::new(image, config))
        .collect()
}

/// Diagnostic matrix `D`.
///
/// Row `i` evaluates the TRIC system built at image `i` on every path
/// image `j`. Entries range from zero for a regular system to three
/// for imminent failure.
pub fn diagnostic_matrix(
    systems: &[TricSystem],
    path: ndarray::ArrayView2<'_, f64>,
) -> Result<Array2<u8>, SaddleError> {
    if systems.len() != path.nrows() {
        return Err(SaddleError::Shape(
            "the TRIC diagnostic needs one system per path image".into(),
        ));
    }
    if systems
        .iter()
        .any(|system| system.reference.len() != path.ncols())
    {
        return Err(SaddleError::Shape(
            "a TRIC system and path image have different dimensions".into(),
        ));
    }
    Ok(Array2::from_shape_fn(
        (systems.len(), path.nrows()),
        |(system, image)| systems[system].diagnostic(path.row(image)),
    ))
}

/// Find overlapping low diagnostic segments through `D`.
///
/// This is the discrete car through the maze procedure. Each row
/// contributes contiguous stable intervals. The selected chain starts
/// at the first image, ends at the last, and maximizes forward progress
/// while retaining an overlap with the preceding interval.
pub fn tric_segments(
    diagnostic: ndarray::ArrayView2<'_, u8>,
    threshold: u8,
) -> Result<Vec<TricsSegment>, SaddleError> {
    if diagnostic.nrows() == 0 || diagnostic.nrows() != diagnostic.ncols() {
        return Err(SaddleError::Shape(
            "the TRIC diagnostic matrix must be nonempty and square".into(),
        ));
    }
    let n = diagnostic.nrows();
    for system in 0..n {
        if (0..n).all(|image| diagnostic[(system, image)] <= threshold) {
            return Ok(vec![TricsSegment {
                start: 0,
                end: n - 1,
                system,
            }]);
        }
    }
    let intervals = stable_intervals(diagnostic, threshold);
    let mut selected = Vec::new();
    let mut covered = 0;
    while covered + 1 < n {
        let mut best: Option<(usize, usize, usize)> = None;
        for &(start, end, system) in &intervals {
            if start > covered || end <= covered {
                continue;
            }
            let replace = match best {
                Some((best_start, best_end, _)) => {
                    end > best_end || (end == best_end && start < best_start)
                }
                None => true,
            };
            if replace {
                best = Some((start, end, system));
            }
        }
        let Some((start, end, system)) = best else {
            return Err(SaddleError::Invalid(
                "TRICS could not drive through the diagnostic matrix".into(),
            ));
        };
        if selected.is_empty() && start != 0 {
            return Err(SaddleError::Invalid(
                "TRICS diagnostic path does not start at the first image".into(),
            ));
        }
        selected.push(TricsSegment { start, end, system });
        covered = end;
    }
    Ok(selected)
}

/// Refine an initial Cartesian path with TRICS.
///
/// The supplied path may be linear, IDPP, or caller generated. One
/// nonredundant TRIC system is built per image, low diagnostic segments
/// are interpolated in their own metrics, and overlapping endpoints are
/// iterated until their maximum atom mismatch reaches the configured
/// threshold.
pub fn trics_path(
    initial: ndarray::ArrayView2<'_, f64>,
    config: &TricsConfig,
) -> Result<TricsReport, SaddleError> {
    check_trics_config(config)?;
    let systems = tric_systems(initial, &config.coordinates)?;
    let diagnostic = diagnostic_matrix(&systems, initial)?;
    let segments = tric_segments(diagnostic.view(), config.diagnostic_threshold)?;
    if segments.len() == 1 {
        let segment = segments[0];
        let path = interpolate_tric_segment(
            &systems[segment.system],
            initial.row(0),
            initial.row(initial.nrows() - 1),
            initial.nrows(),
            &config.inversion,
        )?;
        return Ok(TricsReport {
            path,
            segments,
            splice_iterations: 0,
        });
    }
    splice_path(initial, &systems, segments, config)
}

fn stable_intervals(
    diagnostic: ndarray::ArrayView2<'_, u8>,
    threshold: u8,
) -> Vec<(usize, usize, usize)> {
    let mut intervals = Vec::new();
    for system in 0..diagnostic.nrows() {
        let mut image = 0;
        while image < diagnostic.ncols() {
            if diagnostic[(system, image)] > threshold {
                image += 1;
                continue;
            }
            let start = image;
            while image + 1 < diagnostic.ncols() && diagnostic[(system, image + 1)] <= threshold {
                image += 1;
            }
            intervals.push((start, image, system));
            image += 1;
        }
    }
    intervals
}

fn splice_path(
    initial: ndarray::ArrayView2<'_, f64>,
    systems: &[TricSystem],
    segments: Vec<TricsSegment>,
    config: &TricsConfig,
) -> Result<TricsReport, SaddleError> {
    let mut endpoints: Vec<[Array1<f64>; 2]> = segments
        .iter()
        .map(|segment| {
            [
                initial.row(segment.start).to_owned(),
                initial.row(segment.end).to_owned(),
            ]
        })
        .collect();
    let mut interpolated = Vec::new();
    for iteration in 1..=config.max_splice_iterations {
        interpolated.clear();
        for (segment, endpoint) in segments.iter().zip(&endpoints) {
            interpolated.push(interpolate_tric_segment(
                &systems[segment.system],
                endpoint[0].view(),
                endpoint[1].view(),
                segment.end - segment.start + 1,
                &config.inversion,
            )?);
        }
        let mismatch = splice_mismatch(&segments, &interpolated);
        if mismatch <= config.splice_tolerance {
            let path = concatenate_segments(initial.nrows(), &segments, &interpolated);
            return Ok(TricsReport {
                path,
                segments,
                splice_iterations: iteration,
            });
        }
        let previous = endpoints.clone();
        for pair in 0..segments.len() - 1 {
            let left = segments[pair];
            let right = segments[pair + 1];
            let right_at_left_end = left.end - right.start;
            let left_at_right_start = right.start - left.start;
            endpoints[pair][1].assign(&interpolated[pair + 1].row(right_at_left_end));
            endpoints[pair + 1][0].assign(&interpolated[pair].row(left_at_right_start));
        }
        if endpoints.iter().zip(previous).all(|(next, old)| {
            max_atom_difference(&next[0], &old[0]) <= VECTOR_TOLERANCE
                && max_atom_difference(&next[1], &old[1]) <= VECTOR_TOLERANCE
        }) {
            return Err(SaddleError::Solver(
                "TRICS splicing stopped without reducing the mismatch".into(),
            ));
        }
    }
    Err(SaddleError::Solver(
        "TRICS splicing did not converge".into(),
    ))
}

fn splice_mismatch(segments: &[TricsSegment], paths: &[Array2<f64>]) -> f64 {
    let mut mismatch = 0.0_f64;
    for pair in 0..segments.len() - 1 {
        let left = segments[pair];
        let right = segments[pair + 1];
        let left_at_right_start = right.start - left.start;
        let right_at_left_end = left.end - right.start;
        mismatch = mismatch.max(max_atom_difference(
            &paths[pair].row(left_at_right_start).to_owned(),
            &paths[pair + 1].row(0).to_owned(),
        ));
        mismatch = mismatch.max(max_atom_difference(
            &paths[pair].row(paths[pair].nrows() - 1).to_owned(),
            &paths[pair + 1].row(right_at_left_end).to_owned(),
        ));
    }
    mismatch
}

fn concatenate_segments(
    n_images: usize,
    segments: &[TricsSegment],
    paths: &[Array2<f64>],
) -> Array2<f64> {
    let dof = paths[0].ncols();
    let mut result = Array2::zeros((n_images, dof));
    let mut segment = 0;
    for image in 0..n_images {
        while segment + 1 < segments.len() {
            let boundary = (segments[segment + 1].start + segments[segment].end) / 2;
            if image <= boundary {
                break;
            }
            segment += 1;
        }
        result
            .row_mut(image)
            .assign(&paths[segment].row(image - segments[segment].start));
    }
    result
}

fn max_atom_difference(left: &Array1<f64>, right: &Array1<f64>) -> f64 {
    left.iter()
        .zip(right)
        .collect::<Vec<_>>()
        .chunks(3)
        .map(|atom| {
            atom.iter()
                .map(|(left, right)| {
                    let difference = *left - *right;
                    difference * difference
                })
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0_f64, f64::max)
}

fn check_trics_config(config: &TricsConfig) -> Result<(), SaddleError> {
    check_config(&config.coordinates)?;
    check_inversion_config(&config.inversion)?;
    if config.diagnostic_threshold > 3
        || !(config.splice_tolerance.is_finite() && config.splice_tolerance > 0.0)
        || config.max_splice_iterations == 0
    {
        return Err(SaddleError::Invalid(
            "TRICS needs a diagnostic from zero to three and positive splice controls".into(),
        ));
    }
    Ok(())
}

fn check_config(config: &TricConfig) -> Result<(), SaddleError> {
    if !(config.bond_factor.is_finite() && config.bond_factor > 1.0) {
        return Err(SaddleError::Invalid(
            "TRIC bond factor must be finite and greater than one".into(),
        ));
    }
    if !(config.rank_tolerance.is_finite()
        && config.rank_tolerance > 0.0
        && config.rank_tolerance < 1.0)
    {
        return Err(SaddleError::Invalid(
            "TRIC rank tolerance must lie between zero and one".into(),
        ));
    }
    Ok(())
}

fn check_inversion_config(config: &TricInversionConfig) -> Result<(), SaddleError> {
    if config.max_iterations == 0
        || !(config.tolerance.is_finite() && config.tolerance > 0.0)
        || !(config.max_atom_move.is_finite() && config.max_atom_move > 0.0)
    {
        return Err(SaddleError::Invalid(
            "TRIC inversion needs positive finite controls".into(),
        ));
    }
    Ok(())
}

fn check_positions(positions: ArrayView1<'_, f64>) -> Result<(), SaddleError> {
    if positions.is_empty() || !positions.len().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "TRIC coordinates must be a nonempty 3N vector".into(),
        ));
    }
    if positions.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("TRIC coordinates"));
    }
    Ok(())
}

fn check_same_positions(
    left: ArrayView1<'_, f64>,
    right: ArrayView1<'_, f64>,
) -> Result<(), SaddleError> {
    check_positions(left)?;
    check_positions(right)?;
    if left.len() != right.len() {
        return Err(SaddleError::Shape(
            "TRIC geometries have different dimensions".into(),
        ));
    }
    Ok(())
}

fn atom(positions: ArrayView1<'_, f64>, index: usize) -> Result<[f64; 3], SaddleError> {
    let start = 3 * index;
    if start + 2 >= positions.len() {
        return Err(SaddleError::Shape(
            "TRIC atom index exceeds the geometry".into(),
        ));
    }
    Ok([positions[start], positions[start + 1], positions[start + 2]])
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

fn distance(positions: ArrayView1<'_, f64>, left: usize, right: usize) -> Result<f64, SaddleError> {
    let length = norm(sub(atom(positions, left)?, atom(positions, right)?));
    if length > VECTOR_TOLERANCE {
        Ok(length)
    } else {
        Err(SaddleError::Invalid(
            "TRIC distance has coincident atoms".into(),
        ))
    }
}

fn distance_derivative(
    positions: ArrayView1<'_, f64>,
    left: usize,
    right: usize,
) -> Result<Array1<f64>, SaddleError> {
    let vector = sub(atom(positions, left)?, atom(positions, right)?);
    let unit = scale(vector, 1.0 / norm(vector));
    let mut derivative = Array1::zeros(positions.len());
    for axis in 0..3 {
        derivative[3 * left + axis] = unit[axis];
        derivative[3 * right + axis] = -unit[axis];
    }
    Ok(derivative)
}

fn angle(
    positions: ArrayView1<'_, f64>,
    left: usize,
    center: usize,
    right: usize,
) -> Result<f64, SaddleError> {
    let u = sub(atom(positions, left)?, atom(positions, center)?);
    let v = sub(atom(positions, right)?, atom(positions, center)?);
    let denominator = norm(u) * norm(v);
    if denominator <= VECTOR_TOLERANCE {
        return Err(SaddleError::Invalid(
            "TRIC angle has a zero length leg".into(),
        ));
    }
    let raw_cosine = dot(u, v) / denominator;
    let cosine = if (-1.0..=1.0).contains(&raw_cosine) {
        raw_cosine
    } else {
        raw_cosine.signum()
    };
    Ok(cosine.acos())
}

fn angle_derivative(
    positions: ArrayView1<'_, f64>,
    left: usize,
    center: usize,
    right: usize,
) -> Result<Array1<f64>, SaddleError> {
    let u = sub(atom(positions, left)?, atom(positions, center)?);
    let v = sub(atom(positions, right)?, atom(positions, center)?);
    let (nu, nv) = (norm(u), norm(v));
    let cosine = dot(u, v) / (nu * nv);
    let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
    if nu <= VECTOR_TOLERANCE || nv <= VECTOR_TOLERANCE || sine <= VECTOR_TOLERANCE {
        return Err(SaddleError::Invalid(
            "TRIC angle derivative is singular".into(),
        ));
    }
    let left_gradient = scale(
        sub(scale(u, cosine / nu), scale(v, 1.0 / nv)),
        1.0 / (nu * sine),
    );
    let right_gradient = scale(
        sub(scale(v, cosine / nv), scale(u, 1.0 / nu)),
        1.0 / (nv * sine),
    );
    let center_gradient = scale(
        [
            left_gradient[0] + right_gradient[0],
            left_gradient[1] + right_gradient[1],
            left_gradient[2] + right_gradient[2],
        ],
        -1.0,
    );
    let mut derivative = Array1::zeros(positions.len());
    for axis in 0..3 {
        derivative[3 * left + axis] = left_gradient[axis];
        derivative[3 * center + axis] = center_gradient[axis];
        derivative[3 * right + axis] = right_gradient[axis];
    }
    Ok(derivative)
}

fn angular_diagnostic(positions: ArrayView1<'_, f64>, angles: &[(usize, usize, usize)]) -> u8 {
    let mut result = 0;
    for &(left, center, right) in angles {
        let Ok(value) = angle(positions, left, center, right) else {
            return 3;
        };
        let cosine = value.cos().abs();
        let level = if cosine >= ANGLE_FAILURE_COSINE {
            3
        } else if cosine >= ANGLE_UNSTABLE_COSINE {
            2
        } else if cosine >= ANGLE_WARNING_COSINE {
            1
        } else {
            0
        };
        result = result.max(level);
    }
    result
}

fn linear_bend(
    positions: ArrayView1<'_, f64>,
    left: usize,
    center: usize,
    right: usize,
    direction: [f64; 3],
) -> Result<f64, SaddleError> {
    let u = sub(atom(positions, left)?, atom(positions, center)?);
    let v = sub(atom(positions, right)?, atom(positions, center)?);
    let (nu, nv) = (norm(u), norm(v));
    if nu <= VECTOR_TOLERANCE || nv <= VECTOR_TOLERANCE {
        return Err(SaddleError::Invalid(
            "TRIC linear bend has a zero length leg".into(),
        ));
    }
    Ok(dot(
        [
            u[0] / nu + v[0] / nv,
            u[1] / nu + v[1] / nv,
            u[2] / nu + v[2] / nv,
        ],
        direction,
    ))
}

fn linear_angular_diagnostic(
    positions: ArrayView1<'_, f64>,
    left: usize,
    center: usize,
    right: usize,
) -> u8 {
    let Ok(value) = angle(positions, left, center, right) else {
        return 3;
    };
    let cosine = value.cos().abs();
    if cosine < 0.4 {
        3
    } else if cosine < 0.6 {
        2
    } else if cosine < 0.8 {
        1
    } else {
        0
    }
}

fn perpendicular_directions(axis: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let seed = if axis[0].abs() <= axis[1].abs() && axis[0].abs() <= axis[2].abs() {
        [1.0, 0.0, 0.0]
    } else if axis[1].abs() <= axis[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let first_raw = cross(axis, seed);
    let first = scale(first_raw, 1.0 / norm(first_raw));
    let second_raw = cross(axis, first);
    let second = scale(second_raw, 1.0 / norm(second_raw));
    (first, second)
}

fn dihedral(
    positions: ArrayView1<'_, f64>,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
) -> Result<f64, SaddleError> {
    let b0 = sub(atom(positions, a)?, atom(positions, b)?);
    let b1 = sub(atom(positions, c)?, atom(positions, b)?);
    let b2 = sub(atom(positions, d)?, atom(positions, c)?);
    let b1_norm = norm(b1);
    if b1_norm <= VECTOR_TOLERANCE {
        return Err(SaddleError::Invalid(
            "TRIC dihedral has a zero central bond".into(),
        ));
    }
    let axis = scale(b1, 1.0 / b1_norm);
    let v = sub(b0, scale(axis, dot(b0, axis)));
    let w = sub(b2, scale(axis, dot(b2, axis)));
    if norm(v) <= VECTOR_TOLERANCE || norm(w) <= VECTOR_TOLERANCE {
        return Err(SaddleError::Invalid(
            "TRIC dihedral is undefined at a linear angle".into(),
        ));
    }
    Ok(dot(cross(axis, v), w).atan2(dot(v, w)))
}

fn central_derivative(
    coordinate: &TricPrimitive,
    positions: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, SaddleError> {
    let mut derivative = Array1::zeros(positions.len());
    let mut trial = positions.to_owned();
    for slot in 0..positions.len() {
        trial[slot] = positions[slot] + DIFFERENCE_STEP;
        let plus = coordinate.value(trial.view())?;
        trial[slot] = positions[slot] - DIFFERENCE_STEP;
        let minus = coordinate.value(trial.view())?;
        trial[slot] = positions[slot];
        let mut difference = plus - minus;
        if matches!(coordinate, TricPrimitive::Dihedral(..)) {
            if difference > std::f64::consts::PI {
                difference -= std::f64::consts::TAU;
            } else if difference < -std::f64::consts::PI {
                difference += std::f64::consts::TAU;
            }
        }
        derivative[slot] = difference / (2.0 * DIFFERENCE_STEP);
    }
    Ok(derivative)
}

fn primitive_wilson_b(
    primitives: &[TricPrimitive],
    positions: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, SaddleError> {
    let mut matrix = Array2::zeros((primitives.len(), positions.len()));
    for (row, primitive) in primitives.iter().enumerate() {
        matrix
            .row_mut(row)
            .assign(&primitive.derivative(positions)?);
    }
    Ok(matrix)
}

fn solve_metric(matrix: &Array2<f64>, right: &Array1<f64>) -> Result<Array1<f64>, SaddleError> {
    let n = matrix.nrows();
    if matrix.ncols() != n || right.len() != n {
        return Err(SaddleError::Shape(
            "TRIC linear solve dimensions differ".into(),
        ));
    }
    let (values, vectors) = symmetric_eigen(matrix);
    let largest = values.iter().copied().fold(0.0_f64, f64::max);
    let threshold = largest * DEFAULT_RANK_TOLERANCE;
    let mut solution = Array1::zeros(n);
    for (mode, &value) in values.iter().enumerate() {
        if value <= threshold {
            continue;
        }
        let vector = vectors.column(mode);
        let coefficient = vector.dot(right) / value;
        solution += &(&vector * coefficient);
    }
    Ok(solution)
}

fn cap_atom_step(mut step: Array1<f64>, max_move: f64) -> Array1<f64> {
    let largest = step
        .as_slice()
        .map(|values| {
            values
                .chunks(3)
                .map(|atom| atom.iter().map(|value| value * value).sum::<f64>().sqrt())
                .fold(0.0_f64, f64::max)
        })
        .map_or(0.0, |value| value);
    if largest > max_move {
        step *= max_move / largest;
    }
    step
}

fn validate_bonds(
    bonds: &[(usize, usize)],
    n_atoms: usize,
) -> Result<Vec<(usize, usize)>, SaddleError> {
    let mut checked = Vec::with_capacity(bonds.len());
    for &(left, right) in bonds {
        if left == right || left >= n_atoms || right >= n_atoms {
            return Err(SaddleError::Shape(
                "TRIC bond contains an invalid atom pair".into(),
            ));
        }
        let pair = (left.min(right), left.max(right));
        if !checked.contains(&pair) {
            checked.push(pair);
        }
    }
    checked.sort_unstable();
    Ok(checked)
}

fn infer_bonds(
    positions: ArrayView1<'_, f64>,
    factor: f64,
) -> Result<Vec<(usize, usize)>, SaddleError> {
    let n_atoms = positions.len() / 3;
    if n_atoms < 2 {
        return Ok(Vec::new());
    }
    let mut distances = Array2::from_elem((n_atoms, n_atoms), f64::INFINITY);
    let mut nearest = vec![f64::INFINITY; n_atoms];
    for left in 0..n_atoms {
        for right in left + 1..n_atoms {
            let separation = distance(positions, left, right)?;
            distances[(left, right)] = separation;
            distances[(right, left)] = separation;
            nearest[left] = nearest[left].min(separation);
            nearest[right] = nearest[right].min(separation);
        }
    }
    let mut scale = nearest.clone();
    scale.sort_by(|left, right| match left.partial_cmp(right) {
        Some(ordering) => ordering,
        None => Ordering::Equal,
    });
    let median = scale[scale.len() / 2];
    let global_limit = factor * median;
    let mut bonds = Vec::new();
    for left in 0..n_atoms {
        for right in left + 1..n_atoms {
            let local_limit = factor * (nearest[left] + nearest[right]) * 0.5;
            if distances[(left, right)] <= local_limit.min(global_limit) {
                bonds.push((left, right));
            }
        }
    }
    Ok(bonds)
}

fn fragments(n_atoms: usize, bonds: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); n_atoms];
    for &(left, right) in bonds {
        adjacency[left].push(right);
        adjacency[right].push(left);
    }
    let mut seen = vec![false; n_atoms];
    let mut result = Vec::new();
    for start in 0..n_atoms {
        if seen[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut fragment = Vec::new();
        seen[start] = true;
        while let Some(atom) = stack.pop() {
            fragment.push(atom);
            for &next in &adjacency[atom] {
                if !seen[next] {
                    seen[next] = true;
                    stack.push(next);
                }
            }
        }
        fragment.sort_unstable();
        result.push(fragment);
    }
    result
}

fn molecular_primitives(
    bonds: &[(usize, usize)],
    n_atoms: usize,
    reference: ArrayView1<'_, f64>,
) -> Result<Vec<TricPrimitive>, SaddleError> {
    let mut adjacency = vec![Vec::new(); n_atoms];
    let mut primitives = Vec::new();
    for &(left, right) in bonds {
        adjacency[left].push(right);
        adjacency[right].push(left);
        primitives.push(TricPrimitive::Distance(left, right));
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }
    for (center, neighbors) in adjacency.iter().enumerate() {
        for first in 0..neighbors.len() {
            for second in first + 1..neighbors.len() {
                let left = neighbors[first];
                let right = neighbors[second];
                let value = angle(reference, left, center, right)?;
                if value.cos().abs() >= ANGLE_FAILURE_COSINE {
                    let bond = sub(atom(reference, left)?, atom(reference, center)?);
                    let axis = scale(bond, 1.0 / norm(bond));
                    let (first_direction, second_direction) = perpendicular_directions(axis);
                    primitives.push(TricPrimitive::LinearBend(
                        left,
                        center,
                        right,
                        first_direction,
                    ));
                    primitives.push(TricPrimitive::LinearBend(
                        left,
                        center,
                        right,
                        second_direction,
                    ));
                } else {
                    primitives.push(TricPrimitive::Angle(left, center, right));
                }
            }
        }
    }
    let mut dihedrals = Vec::new();
    for &(b, c) in bonds {
        for &a in &adjacency[b] {
            if a == c {
                continue;
            }
            for &d in &adjacency[c] {
                if d == b || d == a {
                    continue;
                }
                let forward = (a, b, c, d);
                let reverse = (d, c, b, a);
                let canonical = if forward <= reverse { forward } else { reverse };
                if !dihedrals.contains(&canonical) {
                    dihedrals.push(canonical);
                }
            }
        }
    }
    dihedrals.sort_unstable();
    primitives.extend(
        dihedrals
            .into_iter()
            .map(|(a, b, c, d)| TricPrimitive::Dihedral(a, b, c, d)),
    );
    Ok(primitives)
}

fn fragment_positions(positions: ArrayView1<'_, f64>, fragment: &[usize]) -> Array2<f64> {
    Array2::from_shape_fn((fragment.len(), 3), |(row, axis)| {
        positions[3 * fragment[row] + axis]
    })
}

fn nonredundant_basis(metric: &Array2<f64>, tolerance: f64) -> Array2<f64> {
    let (values, vectors) = symmetric_eigen(metric);
    let largest = values.iter().copied().fold(0.0_f64, f64::max);
    let mut indices: Vec<usize> = (0..values.len())
        .filter(|&index| values[index] > largest * tolerance)
        .collect();
    indices.sort_by(
        |&left, &right| match values[right].partial_cmp(&values[left]) {
            Some(ordering) => ordering,
            None => Ordering::Equal,
        },
    );
    Array2::from_shape_fn((metric.nrows(), indices.len()), |(row, column)| {
        vectors[(row, indices[column])]
    })
}

fn symmetric_eigen(matrix: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = matrix.nrows();
    let mut work = matrix.clone();
    let mut vectors = Array2::eye(n);
    for _ in 0..n.saturating_mul(JACOBI_SWEEPS_PER_DIMENSION) {
        let mut p = 0;
        let mut q = 0;
        let mut largest = 0.0_f64;
        for row in 0..n {
            for column in row + 1..n {
                let value = work[(row, column)].abs();
                if value > largest {
                    largest = value;
                    p = row;
                    q = column;
                }
            }
        }
        if largest <= VECTOR_TOLERANCE {
            break;
        }
        let app = work[(p, p)];
        let aqq = work[(q, q)];
        let apq = work[(p, q)];
        let angle = 0.5 * (2.0 * apq).atan2(aqq - app);
        let (sine, cosine) = angle.sin_cos();
        for row in 0..n {
            if row == p || row == q {
                continue;
            }
            let arp = work[(row, p)];
            let arq = work[(row, q)];
            work[(row, p)] = cosine * arp - sine * arq;
            work[(p, row)] = work[(row, p)];
            work[(row, q)] = sine * arp + cosine * arq;
            work[(q, row)] = work[(row, q)];
        }
        work[(p, p)] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        work[(q, q)] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        work[(p, q)] = 0.0;
        work[(q, p)] = 0.0;
        for row in 0..n {
            let vrp = vectors[(row, p)];
            let vrq = vectors[(row, q)];
            vectors[(row, p)] = cosine * vrp - sine * vrq;
            vectors[(row, q)] = sine * vrp + cosine * vrq;
        }
    }
    ((0..n).map(|index| work[(index, index)]).collect(), vectors)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    const TEST_STEP: f64 = 0.000_000_1;
    const TEST_TOLERANCE: f64 = 0.000_02;

    #[test]
    fn primitive_wilson_rows_match_finite_differences() {
        let positions = array![0.0, 0.0, 0.0, 1.2, 0.1, 0.0, 2.0, 0.9, 0.2, 2.8, 1.0, 1.1];
        let primitives = [
            TricPrimitive::Distance(0, 1),
            TricPrimitive::Angle(0, 1, 2),
            TricPrimitive::Dihedral(0, 1, 2, 3),
        ];
        for primitive in primitives {
            let derivative = primitive.derivative(positions.view()).unwrap();
            for slot in 0..positions.len() {
                let mut plus = positions.clone();
                let mut minus = positions.clone();
                plus[slot] += TEST_STEP;
                minus[slot] -= TEST_STEP;
                let numerical =
                    primitive.difference(plus.view(), minus.view()).unwrap() / (2.0 * TEST_STEP);
                assert_abs_diff_eq!(derivative[slot], numerical, epsilon = TEST_TOLERANCE);
            }
        }
    }

    #[test]
    fn a_tric_basis_is_nonredundant() {
        let water = array![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0];
        let config = TricConfig::default().with_bonds(Some(vec![(0, 1), (0, 2)]));
        let system = TricSystem::new(water.view(), &config).unwrap();
        assert_eq!(system.dimension(), water.len());
        let b = system.wilson_b(water.view()).unwrap();
        assert_eq!(b.dim(), (water.len(), water.len()));
    }

    #[test]
    fn inferred_bonds_keep_a_chain_sparse() {
        let chain = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.2, 0.0];
        let bonds = infer_bonds(chain.view(), DEFAULT_BOND_FACTOR).unwrap();
        assert_eq!(bonds, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn diagnostic_matrix_flags_a_collinear_angle() {
        let bent = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.7, 0.7, 0.0];
        let linear = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let path = ndarray::stack(ndarray::Axis(0), &[bent.view(), linear.view()]).unwrap();
        let config = TricConfig::default().with_bonds(Some(vec![(0, 1), (1, 2)]));
        let systems = tric_systems(path.view(), &config).unwrap();
        let diagnostic = diagnostic_matrix(&systems, path.view()).unwrap();
        assert_eq!(diagnostic[(0, 0)], 0);
        assert_eq!(diagnostic[(0, 1)], 3);
    }

    #[test]
    fn back_transformation_round_trips_an_internal_step() {
        let start = array![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0];
        let target = array![0.1, -0.05, 0.02, 1.02, 0.08, -0.03, -0.18, 0.98, 0.06];
        let config = TricConfig::default().with_bonds(Some(vec![(0, 1), (0, 2)]));
        let system = TricSystem::new(start.view(), &config).unwrap();
        let displacement = system.difference(target.view(), start.view()).unwrap();
        let recovered = system
            .back_transform(
                start.view(),
                displacement.view(),
                &TricInversionConfig::default(),
            )
            .unwrap();
        let error = &recovered - &target;
        assert!(error.dot(&error).sqrt() < TEST_TOLERANCE);
    }

    #[test]
    fn segment_interpolation_keeps_exact_endpoints() {
        let left = array![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0];
        let right = array![0.2, 0.0, 0.0, 1.18, 0.1, 0.0, -0.1, 1.0, 0.1];
        let config = TricConfig::default().with_bonds(Some(vec![(0, 1), (0, 2)]));
        let system = TricSystem::new(left.view(), &config).unwrap();
        let path = interpolate_tric_segment(
            &system,
            left.view(),
            right.view(),
            5,
            &TricInversionConfig::default(),
        )
        .unwrap();
        assert_eq!(path.row(0), left);
        assert_eq!(path.row(4), right);
        assert!(path.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn one_stable_system_skips_splicing() {
        let left = array![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0];
        let right = array![0.2, 0.0, 0.0, 1.18, 0.1, 0.0, -0.1, 1.0, 0.1];
        let mut initial = Array2::zeros((5, left.len()));
        for image in 0..5 {
            let fraction = image as f64 / 4.0;
            initial
                .row_mut(image)
                .assign(&(&left * (1.0 - fraction) + &right * fraction));
        }
        let config = TricsConfig::default()
            .with_coordinates(TricConfig::default().with_bonds(Some(vec![(0, 1), (0, 2)])));
        let report = trics_path(initial.view(), &config).unwrap();
        assert_eq!(report.segments.len(), 1);
        assert_eq!(report.splice_iterations, 0);
        let direct = interpolate_tric_segment(
            &TricSystem::new(left.view(), &config.coordinates).unwrap(),
            left.view(),
            right.view(),
            5,
            &config.inversion,
        )
        .unwrap();
        assert!(
            report
                .path
                .iter()
                .zip(&direct)
                .all(|(left, right)| (left - right).abs() <= TEST_TOLERANCE)
        );
    }

    #[test]
    fn splice_converges_across_a_linear_bend() {
        let angles = [50.0_f64, 30.0, 20.0, 10.0, 0.0, -10.0, -20.0, -30.0, -50.0];
        let mut initial = Array2::zeros((angles.len(), 9));
        for (image, degrees) in angles.into_iter().enumerate() {
            let radians = degrees.to_radians();
            initial.row_mut(image).assign(&array![
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                radians.cos(),
                radians.sin(),
                0.0
            ]);
        }
        let config = TricsConfig::default()
            .with_coordinates(TricConfig::default().with_bonds(Some(vec![(0, 1), (1, 2)])))
            .with_inversion(
                TricInversionConfig::default()
                    .with_max_iterations(200)
                    .with_tolerance(0.000_001)
                    .with_max_atom_move(0.05),
            )
            .with_diagnostic_threshold(0)
            .with_splice_tolerance(0.01);
        let report = trics_path(initial.view(), &config).unwrap();
        assert!(report.segments.len() > 1);
        assert!(report.splice_iterations > 0);
        assert!(report.path.iter().all(|value| value.is_finite()));
        for pair in report.path.outer_iter().collect::<Vec<_>>().windows(2) {
            let step = &pair[1] - &pair[0];
            assert!(step.dot(&step).sqrt() < 1.0);
        }
    }
}
