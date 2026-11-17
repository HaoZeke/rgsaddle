//! Hard, symmetry-aware regions for constrained sampling.
//!
//! A [`Region`] is an unbiased geometric restriction: molecular dynamics
//! reflects at its faces and Monte Carlo rejects a proposal outside it.
//! The primary implementation, [`SymmetryUniqueRegion`], is the Voronoi
//! cell of one representative among symmetry-equivalent images on a
//! hyperplane. Its degeneracy is the number of distinct images represented
//! by that cell.

use std::fmt::Debug;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1};

use crate::{Hyperplane, SaddleError};

/// The face of a hard region that a configuration crossed.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BoundaryViolation {
    /// Outward ambient unit normal at the violated face.
    pub normal: Array1<f64>,
    /// Signed distance past a planar face. Predicate-only boundaries may
    /// omit it, in which case MD returns to the previous contained point.
    pub penetration: Option<f64>,
}

impl BoundaryViolation {
    /// A locally planar violation.
    pub fn planar(normal: Array1<f64>, penetration: f64) -> Result<Self, SaddleError> {
        if !(penetration.is_finite() && penetration > 0.0) {
            return Err(SaddleError::Invalid(
                "boundary penetration must be finite and positive".into(),
            ));
        }
        Ok(Self {
            normal: unit(normal)?,
            penetration: Some(penetration),
        })
    }

    /// A predicate violation with no penetration depth.
    pub fn predicate(normal: Array1<f64>) -> Result<Self, SaddleError> {
        Ok(Self {
            normal: unit(normal)?,
            penetration: None,
        })
    }
}

/// A hard configuration-space region.
pub trait Region: Debug + Send + Sync {
    /// Whether `point` lies in the closed region.
    fn contains(&self, point: ArrayView1<'_, f64>) -> Result<bool, SaddleError> {
        Ok(self.violation(point)?.is_none())
    }

    /// The violated face and its outward normal, or `None` inside.
    fn violation(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<Option<BoundaryViolation>, SaddleError>;

    /// Number of equivalent in-plane images represented by this region.
    fn degeneracy(&self) -> usize {
        1
    }

    /// Exact planar faces, when an external engine can consume them.
    fn planar_boundaries(&self) -> Option<Vec<PlanarBoundary>> {
        None
    }
}

/// The half-space `normal · (q - point) <= 0`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanarBoundary {
    point: Array1<f64>,
    normal: Array1<f64>,
}

impl PlanarBoundary {
    /// Construct a half-space from a face point and outward normal.
    pub fn new(point: Array1<f64>, normal: Array1<f64>) -> Result<Self, SaddleError> {
        if point.is_empty() || point.len() != normal.len() {
            return Err(SaddleError::Shape(
                "boundary point and normal need one nonzero length".into(),
            ));
        }
        if !point.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("boundary point"));
        }
        Ok(Self {
            point,
            normal: unit(normal)?,
        })
    }

    /// Construct the orientation whose closed interior contains `inside`.
    pub fn containing(
        point: Array1<f64>,
        normal: Array1<f64>,
        inside: ArrayView1<'_, f64>,
    ) -> Result<Self, SaddleError> {
        let mut boundary = Self::new(point, normal)?;
        if boundary.signed_distance(inside)? > 0.0 {
            boundary.normal.mapv_inplace(|value| -value);
        }
        Ok(boundary)
    }

    /// Point on the face.
    pub fn point(&self) -> ArrayView1<'_, f64> {
        self.point.view()
    }

    /// Outward unit normal.
    pub fn normal(&self) -> ArrayView1<'_, f64> {
        self.normal.view()
    }

    /// Signed distance, positive outside.
    pub fn signed_distance(&self, point: ArrayView1<'_, f64>) -> Result<f64, SaddleError> {
        if point.len() != self.point.len() {
            return Err(SaddleError::Shape(
                "boundary and configuration dimensions differ".into(),
            ));
        }
        Ok((&point - &self.point).dot(&self.normal))
    }
}

impl Region for PlanarBoundary {
    fn violation(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<Option<BoundaryViolation>, SaddleError> {
        let penetration = self.signed_distance(point)?;
        if penetration > 0.0 {
            Ok(Some(BoundaryViolation::planar(
                self.normal.clone(),
                penetration,
            )?))
        } else {
            Ok(None)
        }
    }

    fn planar_boundaries(&self) -> Option<Vec<PlanarBoundary>> {
        Some(vec![self.clone()])
    }
}

/// Intersection of a primary symmetry region and optional unit-degeneracy
/// predicates such as a path slab or basin exclusion.
#[derive(Clone, Debug)]
pub struct RegionIntersection {
    regions: Vec<Arc<dyn Region>>,
    degeneracy: usize,
}

impl Default for RegionIntersection {
    fn default() -> Self {
        Self {
            regions: Vec::new(),
            degeneracy: 1,
        }
    }
}

impl RegionIntersection {
    /// Validate an intersection. More than one non-unit degeneracy is
    /// rejected because multiplying overlapping symmetry counts would
    /// double count configurations.
    pub fn new(regions: Vec<Arc<dyn Region>>) -> Result<Self, SaddleError> {
        let non_unit: Vec<usize> = regions
            .iter()
            .map(|region| region.degeneracy())
            .filter(|degeneracy| *degeneracy != 1)
            .collect();
        if non_unit.len() > 1 {
            return Err(SaddleError::Invalid(
                "an intersection may carry only one symmetry degeneracy".into(),
            ));
        }
        if regions.iter().any(|region| region.degeneracy() == 0) {
            return Err(SaddleError::Invalid(
                "region degeneracy must be positive".into(),
            ));
        }
        let degeneracy = match non_unit.first() {
            Some(degeneracy) => *degeneracy,
            None => 1,
        };
        Ok(Self {
            degeneracy,
            regions,
        })
    }

    /// Components in test order.
    pub fn regions(&self) -> &[Arc<dyn Region>] {
        &self.regions
    }
}

impl Region for RegionIntersection {
    fn violation(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<Option<BoundaryViolation>, SaddleError> {
        for region in &self.regions {
            if let Some(violation) = region.violation(point)? {
                return Ok(Some(violation));
            }
        }
        Ok(None)
    }

    fn degeneracy(&self) -> usize {
        self.degeneracy
    }

    fn planar_boundaries(&self) -> Option<Vec<PlanarBoundary>> {
        let mut boundaries = Vec::new();
        for region in &self.regions {
            boundaries.extend(region.planar_boundaries()?);
        }
        Some(boundaries)
    }
}

/// One affine point- or space-group operation `q -> linear q + translation`.
///
/// Non-exhaustive: start from [`AffineSymmetry::default`] and use setters,
/// or use [`AffineSymmetry::new`].
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct AffineSymmetry {
    /// Linear coordinate map.
    pub linear: Array2<f64>,
    /// Nonsymmorphic or lattice translation.
    pub translation: Array1<f64>,
}

impl AffineSymmetry {
    /// Construct one affine operation.
    pub fn new(linear: Array2<f64>, translation: Array1<f64>) -> Self {
        Self {
            linear,
            translation,
        }
    }

    /// Set [`Self::linear`].
    #[must_use]
    pub fn with_linear(mut self, linear: Array2<f64>) -> Self {
        self.linear = linear;
        self
    }

    /// Set [`Self::translation`].
    #[must_use]
    pub fn with_translation(mut self, translation: Array1<f64>) -> Self {
        self.translation = translation;
        self
    }

    fn apply(&self, point: ArrayView1<'_, f64>) -> Result<Array1<f64>, SaddleError> {
        if self.linear.nrows() != point.len()
            || self.linear.ncols() != point.len()
            || self.translation.len() != point.len()
        {
            return Err(SaddleError::Shape(
                "symmetry operation and configuration dimensions differ".into(),
            ));
        }
        if self
            .linear
            .iter()
            .chain(self.translation.iter())
            .any(|value| !value.is_finite())
        {
            return Err(SaddleError::NonFinite("symmetry operation"));
        }
        Ok(self.linear.dot(&point) + &self.translation)
    }
}

/// Generic source of point- or space-group operations.
pub trait SymmetrySource {
    /// Operations to test for preservation of the sampled plane.
    fn operations(&self) -> &[AffineSymmetry];

    /// Displacement from `reference` to an image in the source's metric.
    fn displacement(
        &self,
        reference: ArrayView1<'_, f64>,
        image: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        if reference.len() != image.len() {
            return Err(SaddleError::Shape(
                "symmetry image and representative dimensions differ".into(),
            ));
        }
        Ok(&image - &reference)
    }
}

/// Caller-supplied symmetry operations.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct SymmetryOperations {
    /// Point- or space-group operations.
    pub operations: Vec<AffineSymmetry>,
}

impl SymmetryOperations {
    /// Set [`Self::operations`].
    #[must_use]
    pub fn with_operations(mut self, operations: Vec<AffineSymmetry>) -> Self {
        self.operations = operations;
        self
    }
}

impl SymmetrySource for SymmetryOperations {
    fn operations(&self) -> &[AffineSymmetry] {
        &self.operations
    }
}

/// Symmetry operations whose atomic displacements use a periodic
/// [`minimage::Cell`].
#[derive(Clone, Debug)]
pub struct PeriodicSymmetryOperations {
    source: SymmetryOperations,
    cell: minimage::Cell,
}

impl PeriodicSymmetryOperations {
    /// Attach a periodic cell to caller-supplied operations.
    pub fn new(source: SymmetryOperations, cell: minimage::Cell) -> Self {
        Self { source, cell }
    }

    /// Periodic cell used for image displacements.
    pub fn cell(&self) -> &minimage::Cell {
        &self.cell
    }
}

impl SymmetrySource for PeriodicSymmetryOperations {
    fn operations(&self) -> &[AffineSymmetry] {
        self.source.operations()
    }

    fn displacement(
        &self,
        reference: ArrayView1<'_, f64>,
        image: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        if reference.len() != image.len() || !reference.len().is_multiple_of(3) {
            return Err(SaddleError::Shape(
                "periodic symmetry configurations need matching atom triples".into(),
            ));
        }
        let mut displacement = Array1::zeros(reference.len());
        for atom in 0..reference.len() / 3 {
            let start = 3 * atom;
            let from = [reference[start], reference[start + 1], reference[start + 2]];
            let to = [image[start], image[start + 1], image[start + 2]];
            let delta = self.cell.displacement_euclidean(from, to);
            displacement[start] = delta[0];
            displacement[start + 1] = delta[1];
            displacement[start + 2] = delta[2];
        }
        Ok(displacement)
    }
}

/// Voronoi cell of one representative among the symmetry images on a
/// hyperplane.
#[derive(Clone, Debug)]
pub struct SymmetryUniqueRegion {
    representative: Array1<f64>,
    boundaries: Vec<PlanarBoundary>,
    degeneracy: usize,
}

impl SymmetryUniqueRegion {
    /// Build the cell from operations that map `plane` onto itself.
    ///
    /// Distinctness and plane-preservation tolerances are derived from
    /// machine epsilon, dimension and coordinate scale; no physical
    /// tolerance is guessed.
    pub fn from_source<S: SymmetrySource + ?Sized>(
        plane: &Hyperplane,
        representative: Array1<f64>,
        source: &S,
    ) -> Result<Self, SaddleError> {
        if representative.len() != plane.dim() {
            return Err(SaddleError::Shape(
                "symmetry representative and plane dimensions differ".into(),
            ));
        }
        let scale = 1.0 + representative.dot(&representative).sqrt();
        let tolerance = f64::EPSILON.sqrt() * plane.dim() as f64 * scale;
        if plane.distance(representative.view()).abs() > tolerance {
            return Err(SaddleError::Invalid(
                "symmetry representative does not lie on the plane".into(),
            ));
        }
        let mut images = vec![representative.clone()];
        for operation in source.operations() {
            if operation.linear.nrows() != plane.dim() || operation.linear.ncols() != plane.dim() {
                return Err(SaddleError::Shape(
                    "symmetry operation and plane dimensions differ".into(),
                ));
            }
            let gram = operation.linear.t().dot(&operation.linear);
            let orthogonality_error = gram
                .indexed_iter()
                .map(|((row, column), value)| {
                    let target = if row == column { 1.0 } else { 0.0 };
                    (value - target).abs()
                })
                .fold(0.0_f64, f64::max);
            if orthogonality_error > tolerance / scale {
                return Err(SaddleError::Invalid(
                    "symmetry operation is not orthogonal in sampling coordinates".into(),
                ));
            }
            let mapped_point = operation.apply(plane.point())?;
            let mapped_normal = operation.linear.dot(&plane.normal());
            let normal_length = mapped_normal.dot(&mapped_normal).sqrt();
            if !(normal_length.is_finite() && normal_length > 0.0) {
                return Err(SaddleError::Invalid(
                    "symmetry operation maps the plane normal to zero".into(),
                ));
            }
            let parallel = mapped_normal.dot(&plane.normal()).abs() / normal_length;
            if plane.distance(mapped_point.view()).abs() > tolerance
                || 1.0 - parallel > tolerance / scale
            {
                continue;
            }
            let raw_image = operation.apply(representative.view())?;
            let displacement = source.displacement(representative.view(), raw_image.view())?;
            let image = &representative + &displacement;
            if plane.distance(image.view()).abs() > tolerance {
                continue;
            }
            if images
                .iter()
                .all(|known| (&image - known).dot(&(&image - known)).sqrt() > tolerance)
            {
                images.push(image);
            }
        }
        let mut boundaries = Vec::with_capacity(images.len().saturating_sub(1));
        for image in images.iter().skip(1) {
            let direction = image - &representative;
            let midpoint = 0.5 * (image + &representative);
            boundaries.push(PlanarBoundary::containing(
                midpoint,
                direction,
                representative.view(),
            )?);
        }
        Ok(Self {
            representative,
            boundaries,
            degeneracy: images.len(),
        })
    }

    /// Representative configuration of this cell.
    pub fn representative(&self) -> ArrayView1<'_, f64> {
        self.representative.view()
    }
}

impl Region for SymmetryUniqueRegion {
    fn violation(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<Option<BoundaryViolation>, SaddleError> {
        for boundary in &self.boundaries {
            if let Some(violation) = boundary.violation(point)? {
                return Ok(Some(violation));
            }
        }
        Ok(None)
    }

    fn degeneracy(&self) -> usize {
        self.degeneracy
    }

    fn planar_boundaries(&self) -> Option<Vec<PlanarBoundary>> {
        Some(self.boundaries.clone())
    }
}

/// Region between the neighbouring hyperplanes of a path.
#[derive(Clone, Debug)]
pub struct PathSlab {
    boundaries: Vec<PlanarBoundary>,
}

impl PathSlab {
    /// Orient each supplied neighbouring plane so that `inside` is in its
    /// closed half-space. One neighbour is valid at a path endpoint.
    pub fn new(
        inside: ArrayView1<'_, f64>,
        previous: Option<&Hyperplane>,
        next: Option<&Hyperplane>,
    ) -> Result<Self, SaddleError> {
        let mut boundaries = Vec::with_capacity(2);
        for neighbour in [previous, next].into_iter().flatten() {
            boundaries.push(PlanarBoundary::containing(
                neighbour.point().to_owned(),
                neighbour.normal().to_owned(),
                inside,
            )?);
        }
        if boundaries.is_empty() {
            return Err(SaddleError::Invalid(
                "a path slab needs at least one neighbouring plane".into(),
            ));
        }
        Ok(Self { boundaries })
    }
}

impl Region for PathSlab {
    fn violation(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<Option<BoundaryViolation>, SaddleError> {
        for boundary in &self.boundaries {
            if let Some(violation) = boundary.violation(point)? {
                return Ok(Some(violation));
            }
        }
        Ok(None)
    }

    fn planar_boundaries(&self) -> Option<Vec<PlanarBoundary>> {
        Some(self.boundaries.clone())
    }
}

/// Reflect `velocity` specularly in an outward normal after both have been
/// projected into the active constraint tangent space.
pub fn reflect_velocity(
    velocity: &mut Array1<f64>,
    outward_normal: ArrayView1<'_, f64>,
    constraint: &impl crate::TangentProjector,
) -> Result<(), SaddleError> {
    if velocity.len() != outward_normal.len() {
        return Err(SaddleError::Shape(
            "velocity and boundary normal dimensions differ".into(),
        ));
    }
    let tangent = constraint.project(outward_normal);
    let norm = tangent.dot(&tangent).sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(SaddleError::Invalid(
            "boundary has no normal component within the constraint".into(),
        ));
    }
    let tangent = tangent / norm;
    let component = velocity.dot(&tangent);
    velocity.scaled_add(-2.0 * component, &tangent);
    constraint.project_in_place(velocity);
    Ok(())
}

fn unit(vector: Array1<f64>) -> Result<Array1<f64>, SaddleError> {
    if vector.is_empty() || vector.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("boundary normal"));
    }
    let norm = vector.dot(&vector).sqrt();
    if norm > 0.0 {
        Ok(vector / norm)
    } else {
        Err(SaddleError::Invalid("boundary normal vanishes".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProjectedConstraint;
    use ndarray::array;

    #[test]
    fn reflection_preserves_plane_and_speed() {
        let plane = Hyperplane::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 0.0]).unwrap();
        let mut velocity = plane.project(array![0.4, -0.4, 1.7].view());
        let speed = velocity.dot(&velocity);
        reflect_velocity(&mut velocity, array![0.0, 0.0, 1.0].view(), &plane).unwrap();
        assert!(velocity.dot(&plane.normal()).abs() <= 32.0 * f64::EPSILON);
        assert!((velocity.dot(&velocity) - speed).abs() <= 64.0 * f64::EPSILON * speed);
        let mut point = array![0.3, -0.2, 4.0];
        plane.restore_in_place(&mut point);
        assert!(plane.distance(point.view()).abs() <= 32.0 * f64::EPSILON);
    }

    #[test]
    fn symmetry_cell_has_explicit_degeneracy() {
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let identity = AffineSymmetry::new(Array2::eye(2), Array1::zeros(2));
        let mirror = AffineSymmetry::new(array![[1.0, 0.0], [0.0, -1.0]], Array1::zeros(2));
        let source = SymmetryOperations::default().with_operations(vec![identity, mirror]);
        let region = SymmetryUniqueRegion::from_source(&plane, array![0.0, 1.0], &source).unwrap();
        assert_eq!(region.degeneracy(), 2);
        assert!(region.contains(array![0.0, 0.5].view()).unwrap());
        assert!(!region.contains(array![0.0, -0.5].view()).unwrap());
    }

    #[test]
    fn one_symmetric_cell_times_degeneracy_matches_the_full_line() {
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let source = SymmetryOperations::default().with_operations(vec![
            AffineSymmetry::new(Array2::eye(2), Array1::zeros(2)),
            AffineSymmetry::new(array![[1.0, 0.0], [0.0, -1.0]], Array1::zeros(2)),
        ]);
        let region = SymmetryUniqueRegion::from_source(&plane, array![0.0, 1.0], &source).unwrap();
        let thermal = 0.2;
        let steps = 20_000;
        let limit = 3.0;
        let integrate = |start: f64, end: f64| {
            let width = (end - start) / steps as f64;
            (0..steps)
                .map(|index| {
                    let y = start + (index as f64 + 0.5) * width;
                    (-((y * y - 1.0).powi(2)) / thermal).exp() * width
                })
                .sum::<f64>()
        };
        let full = integrate(-limit, limit);
        let unique = integrate(0.0, limit);
        let reconstructed = region.degeneracy() as f64 * unique;
        let quadrature_roundoff = 128.0 * f64::EPSILON * full.abs();
        assert!((reconstructed - full).abs() <= quadrature_roundoff);
    }

    #[test]
    fn an_intersection_cannot_double_count_two_symmetry_orbits() {
        #[derive(Debug)]
        struct Twice;
        impl Region for Twice {
            fn violation(
                &self,
                _point: ArrayView1<'_, f64>,
            ) -> Result<Option<BoundaryViolation>, SaddleError> {
                Ok(None)
            }
            fn degeneracy(&self) -> usize {
                2
            }
        }
        assert!(RegionIntersection::new(vec![Arc::new(Twice), Arc::new(Twice)]).is_err());
        assert_eq!(RegionIntersection::default().degeneracy(), 1);
    }

    #[test]
    fn periodic_symmetry_images_use_minimum_image_displacements() {
        let plane = Hyperplane::new(array![0.0, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        let source = SymmetryOperations::default().with_operations(vec![
            AffineSymmetry::new(Array2::eye(3), Array1::zeros(3)),
            AffineSymmetry::new(Array2::eye(3), array![0.0, 9.0, 0.0]),
        ]);
        let periodic = PeriodicSymmetryOperations::new(
            source,
            minimage::Cell::ortho(10.0, 10.0, 10.0).unwrap(),
        );
        let region =
            SymmetryUniqueRegion::from_source(&plane, array![0.0, 0.0, 0.0], &periodic).unwrap();
        assert_eq!(region.degeneracy(), 2);
        assert!(region.contains(array![0.0, 0.2, 0.0].view()).unwrap());
        assert!(!region.contains(array![0.0, -0.8, 0.0].view()).unwrap());
    }
}
