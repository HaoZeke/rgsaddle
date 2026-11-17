//! Tangent projectors and affine constraint restoration.
//!
//! A projector removes forbidden components from vectors. A constraint
//! additionally restores a point to its level set. These two operations
//! are kept separate because band and minimum-mode methods only transform
//! forces, while PAFI also restores each sampled configuration.

use ndarray::{Array1, ArrayView1, Zip};
use rgmin::Manifold;

use crate::SaddleError;

/// Projection of an ambient vector onto an allowed tangent space.
pub trait TangentProjector {
    /// Return the allowed component of `vector`.
    fn project(&self, vector: ArrayView1<'_, f64>) -> Array1<f64>;

    /// Replace `vector` by its allowed component.
    fn project_in_place(&self, vector: &mut Array1<f64>);
}

/// A tangent projector with a level set onto which points can be restored.
pub trait ProjectedConstraint: TangentProjector {
    /// Restore `point` to the constraint level set.
    fn restore_in_place(&self, point: &mut Array1<f64>);
}

/// A point-dependent [`rgmin::Manifold`] projector bound at one point.
///
/// This adapter puts Sella internal-coordinate constraints and rgmin's
/// sphere, Grassmann and quotient manifolds behind [`TangentProjector`]
/// without changing their retractions.
#[derive(Clone, Copy, Debug)]
pub struct ManifoldTangent<'a, M: ?Sized> {
    manifold: &'a M,
    point: &'a Array1<f64>,
}

impl<'a, M: Manifold + ?Sized> ManifoldTangent<'a, M> {
    /// Bind `manifold` at `point`.
    pub fn new(manifold: &'a M, point: &'a Array1<f64>) -> Self {
        Self { manifold, point }
    }
}

impl<M: Manifold + ?Sized> TangentProjector for ManifoldTangent<'_, M> {
    fn project(&self, vector: ArrayView1<'_, f64>) -> Array1<f64> {
        self.manifold.project(self.point, &vector.to_owned())
    }

    fn project_in_place(&self, vector: &mut Array1<f64>) {
        *vector = self.manifold.project(self.point, vector);
    }
}

/// The affine plane `{x : normal · (x - point) = 0}`.
///
/// `normal` need not be normalized. Keeping its norm explicit preserves
/// PAFI's path-derivative arithmetic.
#[derive(Clone, Copy, Debug)]
pub struct AffineHyperplane<'a> {
    point: ArrayView1<'a, f64>,
    normal: ArrayView1<'a, f64>,
    normal_squared: f64,
}

impl<'a> AffineHyperplane<'a> {
    /// Bind a point and a nonzero normal of equal length.
    pub fn new(
        point: ArrayView1<'a, f64>,
        normal: ArrayView1<'a, f64>,
    ) -> Result<Self, SaddleError> {
        if point.is_empty() || point.len() != normal.len() {
            return Err(SaddleError::Shape(
                "an affine hyperplane needs a point and normal of one nonzero length".into(),
            ));
        }
        if !point
            .iter()
            .chain(normal.iter())
            .all(|value| value.is_finite())
        {
            return Err(SaddleError::NonFinite("affine hyperplane"));
        }
        let normal_squared = normal.dot(&normal);
        if normal_squared == 0.0 {
            return Err(SaddleError::Invalid(
                "an affine hyperplane normal vanishes".into(),
            ));
        }
        Ok(Self {
            point,
            normal,
            normal_squared,
        })
    }

    /// Signed constraint residual divided by `normal · normal`.
    pub fn correction(&self, point: ArrayView1<'_, f64>) -> f64 {
        (&point - &self.point).dot(&self.normal) / self.normal_squared
    }
}

impl TangentProjector for AffineHyperplane<'_> {
    fn project(&self, vector: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = vector.to_owned();
        self.project_in_place(&mut out);
        out
    }

    fn project_in_place(&self, vector: &mut Array1<f64>) {
        let coefficient = vector.dot(&self.normal) / self.normal_squared;
        *vector -= &(self.normal.to_owned() * coefficient);
    }
}

impl ProjectedConstraint for AffineHyperplane<'_> {
    fn restore_in_place(&self, point: &mut Array1<f64>) {
        let correction = self.correction(point.view());
        *point -= &(self.normal.to_owned() * correction);
    }
}

/// Orthogonal complement of an axis known by its caller to have unit norm.
///
/// This adapter deliberately does not renormalize or divide by
/// `axis · axis`: band, dimer and Lanczos paths already supply unit axes,
/// and their historical arithmetic is `v - (v · axis) axis`.
#[derive(Clone, Copy, Debug)]
pub struct UnitNormal<'a> {
    axis: ArrayView1<'a, f64>,
}

impl<'a> UnitNormal<'a> {
    /// Bind the caller-validated unit `axis`.
    pub fn new(axis: ArrayView1<'a, f64>) -> Self {
        Self { axis }
    }
}

impl TangentProjector for UnitNormal<'_> {
    fn project(&self, vector: ArrayView1<'_, f64>) -> Array1<f64> {
        let parallel = vector.dot(&self.axis);
        let mut out = Array1::zeros(vector.len());
        Zip::from(out.view_mut())
            .and(vector)
            .and(self.axis)
            .for_each(|slot, &value, &axis| *slot = value - axis * parallel);
        out
    }

    fn project_in_place(&self, vector: &mut Array1<f64>) {
        let parallel = vector.dot(&self.axis);
        Zip::from(vector)
            .and(self.axis)
            .for_each(|value, &axis| *value -= axis * parallel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn affine_projection_and_restore_satisfy_the_plane() {
        let point = array![0.4, -0.2, 0.7];
        let normal = array![2.0, -1.0, 0.5];
        let plane = AffineHyperplane::new(point.view(), normal.view()).unwrap();
        let projected = plane.project(array![1.0, 3.0, -2.0].view());
        assert!(projected.dot(&normal).abs() < 1e-14);
        let mut position = array![-0.3, 1.2, 0.1];
        plane.restore_in_place(&mut position);
        assert!((&position - &point).dot(&normal).abs() < 1e-14);
    }
}
