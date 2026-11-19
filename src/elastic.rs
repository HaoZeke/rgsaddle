//! Force multipoles of one defect, expanded about a single centre.
//!
//! The layout follows the single-centre multipole expansion of Wikfeldt,
//! Batista, Vila, and Jónsson, Phys. Chem. Chem. Phys. 15, 16542 (2013),
//! doi:10.1039/C3CP52097H: one centre per body, and the interaction is the
//! contraction of that body's moments with the field and its derivatives.
//! Here the body is a defect and the field is strain. The dipole is the
//! first force moment. The quadrupole and octupole are the second and
//! third.
//!
//! With the centre at the origin and Cartesian forces `F`,
//!
//! `P_ij = Σ_a (X_i F_j + X_j F_i) / 2`,
//!
//! `Q_ijk = Σ_a F_i X_j X_k`,
//!
//! `O_ijkl = Σ_a F_i X_j X_k X_l`.
//!
//! In a strain that varies slowly, the energy of those forces is
//!
//! `-P : ε - (1/2) Q_ijk ∂_k ε_ij - (1/6) O_ijkl ∂_l ∂_k ε_ij`.
//!
//! [`crate::StrainResponse::barrier_in`] applies the dipole and quadrupole
//! terms. The octupole is kept for the convergence of the field expansion.
//!
//! Units: positions in Angstrom, forces in eV/Angstrom, so the dipole is in
//! eV, the quadrupole in eV Angstrom and the octupole in eV Angstrom^2.
//! Strain is dimensionless and its gradient is in 1/Angstrom.

use crate::cell_log::M3;
use crate::error::SaddleError;
use crate::strain_rate::ElasticDipole;

/// `Q_ijk` with `i` the force direction and `j`, `k` the arms, in
/// eV Angstrom.
pub type Quadrupole = [[[f64; 3]; 3]; 3];

/// `O_ijkl` with `i` the force direction, in eV Angstrom^2.
pub type Octupole = [[[[f64; 3]; 3]; 3]; 3];

/// `∂_k ε_ij` stored as `gradient[i][j][k]`, in 1/Angstrom.
pub type StrainGradient = Quadrupole;

/// Strain at an event, and optionally its gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalField {
    /// Strain at the defect centre, dimensionless.
    pub strain: crate::strain_rate::Strain,
    /// `∂_k ε_ij` at the centre, in 1/Angstrom. `None` is a homogeneous field.
    pub gradient: Option<StrainGradient>,
}

impl LocalField {
    /// A homogeneous strain. The gradient is zero (stored as `None`).
    pub fn homogeneous(strain: crate::strain_rate::Strain) -> Self {
        Self {
            strain,
            gradient: None,
        }
    }
}

/// Dipole plus the higher force moments that were requested.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElasticMultipole {
    dipole: ElasticDipole,
    quadrupole: Option<Quadrupole>,
    octupole: Option<Octupole>,
}

impl ElasticMultipole {
    /// Moments of `forces` about `centre`, through `order` (2, 3, or 4).
    ///
    /// Order 2 keeps the dipole. Order 3 adds the quadrupole. Order 4 adds
    /// the octupole. The resultant and the torque must vanish.
    ///
    /// `centre` and every row of `positions` are `[x, y, z]` in Angstrom,
    /// `forces` has one `[fx, fy, fz]` row per position, in eV/Angstrom (the
    /// Kanzaki forces of a relaxed defect).
    ///
    /// # Errors
    ///
    /// - [`SaddleError::Invalid`] when `order` is not 2, 3 or 4, or when the
    ///   resultant or the torque of `forces` is not zero to within `1e-8`
    ///   times the larger of 1 and the largest raw dipole component.
    /// - [`SaddleError::Shape`] when `positions` and `forces` differ in
    ///   length.
    /// - [`SaddleError::NonFinite`] when a position or force component is
    ///   NaN or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use rgsaddle::elastic::ElasticMultipole;
    ///
    /// // A force pair along x: +1 eV/Angstrom at +1 Angstrom, -1 at -1.
    /// let positions = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
    /// let forces = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
    /// let moments = ElasticMultipole::about([0.0; 3], &positions, &forces, 3)?;
    /// assert_eq!(moments.dipole().tensor()[0][0], 2.0);
    /// assert_eq!(moments.order(), 3);
    /// # Ok::<(), rgsaddle::SaddleError>(())
    /// ```
    pub fn about(
        centre: [f64; 3],
        positions: &[[f64; 3]],
        forces: &[[f64; 3]],
        order: usize,
    ) -> Result<Self, SaddleError> {
        if !(2..=4).contains(&order) {
            return Err(SaddleError::Invalid(
                "a force multipole order is 2, 3, or 4".into(),
            ));
        }
        if positions.len() != forces.len() {
            return Err(SaddleError::Shape(
                "one force vector per atom in the multipole sum".into(),
            ));
        }
        let mut shifted = Vec::with_capacity(positions.len());
        for position in positions {
            shifted.push([
                position[0] - centre[0],
                position[1] - centre[1],
                position[2] - centre[2],
            ]);
        }
        let dipole = ElasticDipole::from_forces(&shifted, forces)?;
        let quadrupole = if order >= 3 {
            Some(quadrupole_about(&shifted, forces)?)
        } else {
            None
        };
        let octupole = if order >= 4 {
            Some(octupole_about(&shifted, forces)?)
        } else {
            None
        };
        Ok(Self {
            dipole,
            quadrupole,
            octupole,
        })
    }

    /// Rank actually stored: 2 for the dipole alone, 3 with the quadrupole,
    /// 4 with the octupole.
    pub fn order(self) -> usize {
        if self.octupole.is_some() {
            4
        } else if self.quadrupole.is_some() {
            3
        } else {
            2
        }
    }

    /// The elastic dipole `P_ij` in eV, a symmetric 3x3 tensor.
    pub fn dipole(self) -> ElasticDipole {
        self.dipole
    }

    /// `Q_ijk = Σ F_i X_j X_k` in eV Angstrom when the order is at least 3,
    /// else `None`.
    pub fn quadrupole(self) -> Option<Quadrupole> {
        self.quadrupole
    }

    /// `O_ijkl = Σ F_i X_j X_k X_l` in eV Angstrom^2 when the order is 4,
    /// else `None`.
    pub fn octupole(self) -> Option<Octupole> {
        self.octupole
    }

    /// Rotate every stored moment. `Q'_ijk = R_ia R_jb R_kc Q_abc`.
    ///
    /// `rotation` is a 3x3 orthogonal matrix, rows indexed by the new axis
    /// (`[[f64; 3]; 3]`, row-major). It is not checked for orthogonality.
    pub fn transform(self, rotation: M3) -> Self {
        Self {
            dipole: self.dipole.transform(rotation),
            quadrupole: self.quadrupole.map(|q| rotate_quadrupole(q, rotation)),
            octupole: self.octupole.map(|o| rotate_octupole(o, rotation)),
        }
    }
}

/// `(1/2) Σ Q_ijk ∂_k ε_ij`.
///
/// `quadrupole` is in eV Angstrom and `gradient` in 1/Angstrom, so the result
/// is in eV. The sign is that of the sum above; the interaction energy of
/// the module docs is its negative.
pub fn quadrupole_energy(quadrupole: Quadrupole, gradient: StrainGradient) -> f64 {
    let mut sum = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                sum += quadrupole[i][j][k] * gradient[i][j][k];
            }
        }
    }
    0.5 * sum
}

fn quadrupole_about(
    positions: &[[f64; 3]],
    forces: &[[f64; 3]],
) -> Result<Quadrupole, SaddleError> {
    let mut q = [[[0.0; 3]; 3]; 3];
    for (x, f) in positions.iter().zip(forces) {
        if !x.iter().chain(f.iter()).all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("quadrupole force"));
        }
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    q[i][j][k] += f[i] * x[j] * x[k];
                }
            }
        }
    }
    Ok(q)
}

fn octupole_about(positions: &[[f64; 3]], forces: &[[f64; 3]]) -> Result<Octupole, SaddleError> {
    let mut o = [[[[0.0; 3]; 3]; 3]; 3];
    for (x, f) in positions.iter().zip(forces) {
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        o[i][j][k][l] += f[i] * x[j] * x[k] * x[l];
                    }
                }
            }
        }
    }
    Ok(o)
}

#[allow(clippy::needless_range_loop)]
fn rotate_quadrupole(q: Quadrupole, rotation: M3) -> Quadrupole {
    let mut out = [[[0.0; 3]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let mut sum = 0.0;
                for a in 0..3 {
                    for b in 0..3 {
                        for c in 0..3 {
                            sum += rotation[i][a] * rotation[j][b] * rotation[k][c] * q[a][b][c];
                        }
                    }
                }
                out[i][j][k] = sum;
            }
        }
    }
    out
}

#[allow(clippy::needless_range_loop)]
fn rotate_octupole(o: Octupole, rotation: M3) -> Octupole {
    let mut out = [[[[0.0; 3]; 3]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    let mut sum = 0.0;
                    for a in 0..3 {
                        for b in 0..3 {
                            for c in 0..3 {
                                for d in 0..3 {
                                    sum += rotation[i][a]
                                        * rotation[j][b]
                                        * rotation[k][c]
                                        * rotation[l][d]
                                        * o[a][b][c][d];
                                }
                            }
                        }
                    }
                    out[i][j][k][l] = sum;
                }
            }
        }
    }
    out
}
