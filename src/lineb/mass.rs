//! Mass weights. Coordinates stored by the session are `q = M^{1/2} x`.

use ndarray::{Array1, ArrayView1};

use crate::error::SaddleError;

/// Per-coordinate masses, and the atom count when the system is a molecule.
#[derive(Clone, Debug)]
pub struct Masses {
    pub(crate) sqrt_mass: Array1<f64>,
    /// `Some(atoms)` turns on quaternion alignment. The three coordinates
    /// of an atom share one atomic mass.
    pub(crate) atoms: Option<usize>,
    /// Fit each image to its aligned neighbour instead of the first image.
    pub(crate) adjacent: bool,
}

impl Masses {
    /// One positive finite mass per coordinate. No rigid alignment.
    pub fn coordinates(mass: ArrayView1<f64>) -> Result<Self, SaddleError> {
        if mass.is_empty() {
            return Err(SaddleError::Shape(
                "a tunneling path needs a coordinate".into(),
            ));
        }
        let mut sqrt_mass = Array1::zeros(mass.len());
        for (i, &m) in mass.iter().enumerate() {
            if !(m.is_finite() && m > 0.0) {
                return Err(SaddleError::Invalid(
                    "coordinate masses must be finite and positive".into(),
                ));
            }
            sqrt_mass[i] = m.sqrt();
        }
        Ok(Self {
            sqrt_mass,
            atoms: None,
            adjacent: false,
        })
    }

    /// One positive finite mass per atom, expanded to three coordinates,
    /// with quaternion alignment.
    pub fn molecule(atom_mass: &[f64]) -> Result<Self, SaddleError> {
        if atom_mass.len() < 2 {
            return Err(SaddleError::Invalid(
                "quaternion alignment needs at least two atoms".into(),
            ));
        }
        let mut sqrt_mass = Array1::zeros(3 * atom_mass.len());
        for (atom, &m) in atom_mass.iter().enumerate() {
            if !(m.is_finite() && m > 0.0) {
                return Err(SaddleError::Invalid(
                    "atomic masses must be finite and positive".into(),
                ));
            }
            let root = m.sqrt();
            for axis in 0..3 {
                sqrt_mass[3 * atom + axis] = root;
            }
        }
        Ok(Self {
            sqrt_mass,
            atoms: Some(atom_mass.len()),
            adjacent: false,
        })
    }

    /// Choose the reference of the quaternion alignment for a molecule.
    ///
    /// By default every image is fitted to the first image. With
    /// `adjacent` each image is fitted to its aligned neighbour, so every
    /// chord has its minimum mass-weighted length. Ásgeirsson, Arnaldsson
    /// and Jónsson, J. Chem. Phys. 148, 102334 (2018) minimize the
    /// deviation of adjacent images in this way. The two choices agree
    /// for short paths and differ once the band turns far from its first
    /// image. The setting has no effect without atomic masses.
    pub fn with_adjacent_alignment(mut self, adjacent: bool) -> Self {
        self.adjacent = adjacent;
        self
    }

    pub(crate) fn dof(&self) -> usize {
        self.sqrt_mass.len()
    }

    pub(crate) fn to_weighted(
        &self,
        cartesian: ArrayView1<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        if cartesian.len() != self.dof() {
            return Err(SaddleError::Shape(
                "Cartesian vector length must match the masses".into(),
            ));
        }
        let mut q = cartesian.to_owned();
        for (value, &root) in q.iter_mut().zip(self.sqrt_mass.iter()) {
            *value *= root;
        }
        Ok(q)
    }

    pub(crate) fn to_cartesian(
        &self,
        weighted: ArrayView1<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        if weighted.len() != self.dof() {
            return Err(SaddleError::Shape(
                "mass-weighted vector length must match the masses".into(),
            ));
        }
        let mut x = weighted.to_owned();
        for (value, &root) in x.iter_mut().zip(self.sqrt_mass.iter()) {
            *value /= root;
        }
        Ok(x)
    }

    /// Mass-weighted force `−M^{−1/2} ∇_x V` from a Cartesian gradient.
    pub(crate) fn weighted_force(
        &self,
        gradient: ArrayView1<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        if gradient.len() != self.dof() {
            return Err(SaddleError::Shape(
                "gradient length must match the masses".into(),
            ));
        }
        let mut force = gradient.to_owned();
        for (value, &root) in force.iter_mut().zip(self.sqrt_mass.iter()) {
            *value = -*value / root;
        }
        Ok(force)
    }
}
