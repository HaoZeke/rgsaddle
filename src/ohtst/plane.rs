//! The hyperplane and the mass-weighted surface it lives on.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use crate::error::SaddleError;
use crate::minmode::{MinModeSession, ModeEstimate, PointSurface};

/// A hyperplane of mass-weighted configuration space.
///
/// The plane is `{q : ν · (q - p) = 0}` with `p` a point on it and `ν` a
/// unit normal. The normal points from the reactant side to the product
/// side. Sliding `p` inside the plane leaves the plane unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct Hyperplane {
    point: Array1<f64>,
    normal: Array1<f64>,
}

impl Hyperplane {
    /// The plane through `point` normal to `normal`. The normal is
    /// normalized.
    pub fn new(point: Array1<f64>, normal: Array1<f64>) -> Result<Self, SaddleError> {
        if point.is_empty() || point.len() != normal.len() {
            return Err(SaddleError::Shape(
                "hyperplane point and normal need one nonzero length".into(),
            ));
        }
        if !point.iter().chain(normal.iter()).all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("hyperplane"));
        }
        let norm = normal.dot(&normal).sqrt();
        if norm <= 0.0 {
            return Err(SaddleError::Invalid("hyperplane normal vanishes".into()));
        }
        Ok(Self {
            point,
            normal: normal / norm,
        })
    }

    /// The plane through a minimum-mode session's position, normal to its
    /// mode. The session has to run on the mass-weighted surface.
    pub fn from_min_mode(session: &MinModeSession) -> Result<Self, SaddleError> {
        Self::new(session.position().to_owned(), session.mode().to_owned())
    }

    /// The plane through `position` normal to an estimated lowest mode.
    pub fn from_mode(position: ArrayView1<f64>, mode: &ModeEstimate) -> Result<Self, SaddleError> {
        Self::new(position.to_owned(), mode.mode.clone())
    }

    /// The Cartesian plane `{x : n · (x - x0) = 0}` in mass-weighted
    /// coordinates: `p = M^{1/2} x0` and `ν ∝ M^{-1/2} n`. `masses` has one
    /// entry per coordinate.
    pub fn from_cartesian(
        position: ArrayView1<f64>,
        normal: ArrayView1<f64>,
        masses: &[f64],
    ) -> Result<Self, SaddleError> {
        let sqrt_mass = sqrt_masses(masses, position.len())?;
        if normal.len() != position.len() {
            return Err(SaddleError::Shape(
                "Cartesian plane point and normal need one length".into(),
            ));
        }
        Self::new(&position * &sqrt_mass, &normal / &sqrt_mass)
    }

    /// The Cartesian point `M^{-1/2} p` and unit normal `∝ M^{1/2} ν` of
    /// the same plane.
    pub fn to_cartesian(&self, masses: &[f64]) -> Result<(Array1<f64>, Array1<f64>), SaddleError> {
        let sqrt_mass = sqrt_masses(masses, self.dim())?;
        let normal = &self.normal * &sqrt_mass;
        let norm = normal.dot(&normal).sqrt();
        Ok((&self.point / &sqrt_mass, normal / norm))
    }

    /// A point on the plane.
    pub fn point(&self) -> ArrayView1<'_, f64> {
        self.point.view()
    }

    /// The unit normal.
    pub fn normal(&self) -> ArrayView1<'_, f64> {
        self.normal.view()
    }

    /// Length of the configuration vector.
    pub fn dim(&self) -> usize {
        self.point.len()
    }

    /// Signed distance `ν · (q - p)`, positive on the product side.
    pub fn distance(&self, q: ArrayView1<f64>) -> f64 {
        let mut d = 0.0;
        for i in 0..self.dim() {
            d += self.normal[i] * (q[i] - self.point[i]);
        }
        d
    }

    /// Orthogonal projection of `q` onto the plane.
    pub fn project(&self, q: ArrayView1<f64>) -> Array1<f64> {
        let d = self.distance(q);
        &q - &(&self.normal * d)
    }

    /// The same plane with its point moved to the projection of `center`.
    pub fn recentered(&self, center: ArrayView1<f64>) -> Self {
        Self {
            point: self.project(center),
            normal: self.normal.clone(),
        }
    }

    /// The same plane, with the normal reversed when `toward` lies on the
    /// negative side.
    pub fn oriented_toward(mut self, toward: ArrayView1<f64>) -> Self {
        if self.distance(toward) < 0.0 {
            self.normal.mapv_inplace(|v| -v);
        }
        self
    }

    /// The same plane with its normal made orthogonal to the rows of
    /// `modes`, rigid translations for instance, and normalized again.
    pub fn orthogonal_to(&self, modes: ArrayView2<f64>) -> Result<Self, SaddleError> {
        if modes.nrows() > 0 && modes.ncols() != self.dim() {
            return Err(SaddleError::Shape(
                "modes differ in length from the plane".into(),
            ));
        }
        let mut normal = self.normal.clone();
        for direction in orthonormal_rows(modes, self.dim()) {
            let c = direction.dot(&normal);
            normal.scaled_add(-c, &direction);
        }
        Self::new(self.point.clone(), normal)
    }

    /// `[p; ν]`, length `2n`: the layout of
    /// [`PlaneSample::packed_gradient`](crate::ohtst::PlaneSample::packed_gradient)
    /// and of the directions
    /// [`PlaneSample::directional`](crate::ohtst::PlaneSample::directional)
    /// takes.
    pub fn packed(&self) -> Array1<f64> {
        let n = self.dim();
        let mut x = Array1::zeros(2 * n);
        x.slice_mut(s![..n]).assign(&self.point);
        x.slice_mut(s![n..]).assign(&self.normal);
        x
    }

    /// Inverse of [`Hyperplane::packed`]. The normal block is normalized.
    pub fn from_packed(x: ArrayView1<f64>) -> Result<Self, SaddleError> {
        if x.is_empty() || !x.len().is_multiple_of(2) {
            return Err(SaddleError::Shape(
                "a packed hyperplane has an even nonzero length".into(),
            ));
        }
        let n = x.len() / 2;
        Self::new(x.slice(s![..n]).to_owned(), x.slice(s![n..]).to_owned())
    }
}

/// Gram–Schmidt over the rows, twice for round-off. Rows that depend on
/// earlier ones are dropped.
pub(crate) fn orthonormal_rows(rows: ArrayView2<f64>, n: usize) -> Vec<Array1<f64>> {
    let mut basis: Vec<Array1<f64>> = Vec::with_capacity(rows.nrows());
    for row in rows.rows() {
        if row.len() != n {
            continue;
        }
        let scale = row.dot(&row).sqrt();
        let mut v = row.to_owned();
        for _ in 0..2 {
            for b in &basis {
                let c = b.dot(&v);
                v.scaled_add(-c, b);
            }
        }
        let norm = v.dot(&v).sqrt();
        if scale > 0.0 && norm > 1.0e-10 * scale {
            basis.push(v / norm);
        }
    }
    basis
}

fn sqrt_masses(masses: &[f64], n: usize) -> Result<Array1<f64>, SaddleError> {
    if masses.len() != n {
        return Err(SaddleError::Shape("one mass per coordinate".into()));
    }
    if !masses.iter().all(|m| m.is_finite() && *m > 0.0) {
        return Err(SaddleError::Invalid(
            "masses must be finite and positive".into(),
        ));
    }
    Ok(masses.iter().map(|m| m.sqrt()).collect())
}

/// A Cartesian surface in mass-weighted coordinates `q = M^{1/2} x`.
///
/// The energy is `V(M^{-1/2} q)` and the gradient `M^{-1/2} ∇V`. Unit mass
/// in these coordinates is the Cartesian dynamics with masses `M`, so the
/// plane sampler and the shooting step take mass 1 here. Hessian actions
/// and excluded modes are mapped when the wrapped surface has them.
pub struct MassWeighted<'a, S> {
    surface: &'a S,
    sqrt_mass: Array1<f64>,
}

impl<'a, S: PointSurface> MassWeighted<'a, S> {
    /// One mass per coordinate.
    pub fn new(surface: &'a S, masses: &[f64]) -> Result<Self, SaddleError> {
        Ok(Self {
            surface,
            sqrt_mass: sqrt_masses(masses, masses.len())?,
        })
    }

    /// One mass per atom, repeated over its three coordinates.
    pub fn per_atom(surface: &'a S, masses: &[f64]) -> Result<Self, SaddleError> {
        let expanded: Vec<f64> = masses.iter().flat_map(|m| [*m; 3]).collect();
        Self::new(surface, &expanded)
    }

    /// Masses, one per coordinate.
    pub fn masses(&self) -> Array1<f64> {
        self.sqrt_mass.mapv(|r| r * r)
    }

    /// `q = M^{1/2} x`.
    pub fn to_mass_weighted(&self, x: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        self.check(x.len())?;
        Ok(&x * &self.sqrt_mass)
    }

    /// `x = M^{-1/2} q`.
    pub fn to_cartesian(&self, q: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        self.check(q.len())?;
        Ok(&q / &self.sqrt_mass)
    }

    fn check(&self, n: usize) -> Result<(), SaddleError> {
        if n == self.sqrt_mass.len() {
            Ok(())
        } else {
            Err(SaddleError::Shape(
                "mass-weighted surface length differs from the masses".into(),
            ))
        }
    }
}

impl<S: PointSurface> PointSurface for MassWeighted<'_, S> {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let x = self.to_cartesian(q)?;
        let (energy, gradient) = self.surface.eval(x.view())?;
        if gradient.len() != q.len() {
            return Err(SaddleError::Shape(
                "surface gradient length differs from the masses".into(),
            ));
        }
        Ok((energy, gradient / &self.sqrt_mass))
    }

    fn hessian_vector(
        &self,
        q: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        let x = self.to_cartesian(q)?;
        let u = self.to_cartesian(v)?;
        Ok(self
            .surface
            .hessian_vector(x.view(), u.view())?
            .map(|hv| hv / &self.sqrt_mass))
    }

    fn excluded_modes(&self, q: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        let x = self.to_cartesian(q)?;
        let mut modes = self.surface.excluded_modes(x.view())?;
        if modes.nrows() > 0 && modes.ncols() != q.len() {
            return Err(SaddleError::Shape(
                "excluded modes differ in length from the masses".into(),
            ));
        }
        for mut row in modes.rows_mut() {
            row *= &self.sqrt_mass;
        }
        Ok(modes)
    }
}

/// A surface evaluated in one fixed periodic cell, so that planes are
/// sampled at a set volume, for example at the lattice parameter of each
/// temperature. [`PointSurface::eval`] goes to
/// [`PointSurface::eval_in_cell`] with the stored cell. Hessian actions are
/// left to gradient differences, which see the cell; an analytic action of
/// the inner surface might not.
pub struct FixedCell<'a, S> {
    surface: &'a S,
    cell: [f64; 9],
}

impl<'a, S: PointSurface> FixedCell<'a, S> {
    /// `cell` is row-major 3x3, the layout of
    /// [`PointSurface::eval_in_cell`], with a nonzero volume.
    pub fn new(surface: &'a S, cell: [f64; 9]) -> Result<Self, SaddleError> {
        if !cell.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("cell"));
        }
        let volume = cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
            - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
            + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]);
        if volume == 0.0 {
            return Err(SaddleError::Invalid("the cell has no volume".into()));
        }
        Ok(Self { surface, cell })
    }

    /// The cell, row-major.
    pub fn cell(&self) -> &[f64; 9] {
        &self.cell
    }
}

impl<S: PointSurface> PointSurface for FixedCell<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.surface.eval_in_cell(x, &self.cell)
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.surface.excluded_modes(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    struct Quadratic;

    impl PointSurface for Quadratic {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let e = 0.5 * x[0] * x[0] + x[0] * x[1] + 2.0 * x[1] * x[1];
            Ok((e, array![x[0] + x[1], x[0] + 4.0 * x[1]]))
        }

        fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
            let mut modes = Array2::zeros((1, x.len()));
            modes[(0, 0)] = 1.0;
            modes[(0, 1)] = 1.0;
            Ok(modes)
        }
    }

    #[test]
    fn cartesian_plane_maps_to_the_same_set() {
        let masses = [1.0, 3.0];
        let x0 = array![0.4, -0.2];
        let n = array![1.0, 1.0];
        let plane = Hyperplane::from_cartesian(x0.view(), n.view(), &masses).unwrap();
        let along = array![1.0, -1.0];
        for t in [-1.5, 0.3, 2.0] {
            let x = &x0 + &(&along * t);
            let q = array![x[0] * masses[0].sqrt(), x[1] * masses[1].sqrt()];
            assert!(plane.distance(q.view()).abs() < 1e-14);
        }
        let (back_point, back_normal) = plane.to_cartesian(&masses).unwrap();
        assert!((&back_point - &x0).iter().all(|d| d.abs() < 1e-14));
        let unit = &n / n.dot(&n).sqrt();
        assert!((&back_normal - &unit).iter().all(|d| d.abs() < 1e-14));
    }

    #[test]
    fn packing_recentering_and_orientation_keep_the_plane() {
        let plane = Hyperplane::new(array![1.0, 2.0, 0.0], array![0.0, 3.0, 4.0]).unwrap();
        assert!((plane.normal().dot(&plane.normal()) - 1.0).abs() < 1e-15);
        let back = Hyperplane::from_packed(plane.packed().view()).unwrap();
        assert_eq!(back, plane);
        let moved = plane.recentered(array![5.0, -1.0, 7.0].view());
        assert!(plane.distance(moved.point()).abs() < 1e-14);
        let flipped = plane.clone().oriented_toward(array![1.0, 0.0, 0.0].view());
        assert!(flipped.distance(array![1.0, 0.0, 0.0].view()) > 0.0);
        assert!(
            (&flipped.normal() + &plane.normal())
                .iter()
                .all(|d| d.abs() < 1e-15)
        );
        assert!(Hyperplane::new(array![0.0], array![0.0]).is_err());
        assert!(Hyperplane::new(array![0.0, 1.0], array![1.0]).is_err());
    }

    #[test]
    fn mass_weighted_gradient_and_modes() {
        let masses = [2.0, 0.5];
        let surface = MassWeighted::new(&Quadratic, &masses).unwrap();
        let q = array![0.3, -0.7];
        let (_, g) = surface.eval(q.view()).unwrap();
        let h = 1e-6;
        for i in 0..2 {
            let mut plus = q.clone();
            let mut minus = q.clone();
            plus[i] += h;
            minus[i] -= h;
            let fd = (surface.eval(plus.view()).unwrap().0 - surface.eval(minus.view()).unwrap().0)
                / (2.0 * h);
            assert!((fd - g[i]).abs() < 1e-9);
        }
        let modes = surface.excluded_modes(q.view()).unwrap();
        assert!((modes[(0, 0)] - 2.0f64.sqrt()).abs() < 1e-15);
        assert!((modes[(0, 1)] - 0.5f64.sqrt()).abs() < 1e-15);
        assert!(
            surface
                .hessian_vector(q.view(), q.view())
                .unwrap()
                .is_none()
        );
        assert!(MassWeighted::new(&Quadratic, &[1.0, 0.0]).is_err());
    }

    /// Energy `½|x|² + c₀₀`, gradient `x`: the first cell entry stands in
    /// for a volume-dependent energy.
    struct CellQuadratic;

    impl PointSurface for CellQuadratic {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            self.eval_in_cell(x, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        }

        fn eval_in_cell(
            &self,
            x: ArrayView1<f64>,
            cell: &[f64; 9],
        ) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((0.5 * x.dot(&x) + cell[0], x.to_owned()))
        }

        fn hessian_vector(
            &self,
            _x: ArrayView1<f64>,
            v: ArrayView1<f64>,
        ) -> Result<Option<Array1<f64>>, SaddleError> {
            Ok(Some(v.to_owned()))
        }
    }

    #[test]
    fn fixed_cell_evaluates_in_its_cell() {
        let cell = [1.1, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.1];
        let surface = FixedCell::new(&CellQuadratic, cell).unwrap();
        let x = array![0.3, -0.4];
        let (energy, gradient) = surface.eval(x.view()).unwrap();
        assert!((energy - (0.125 + 1.1)).abs() < 1e-15);
        assert_eq!(gradient, x);
        assert_eq!(surface.cell(), &cell);
        assert!(
            surface
                .hessian_vector(x.view(), x.view())
                .unwrap()
                .is_none()
        );
        let flat = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        assert!(FixedCell::new(&CellQuadratic, flat).is_err());
        let mut broken = cell;
        broken[4] = f64::NAN;
        assert!(FixedCell::new(&CellQuadratic, broken).is_err());
    }
}
