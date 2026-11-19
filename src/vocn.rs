//! Dest client of vocn auto-find.
//!
//! Bond / angle / dihedral *topology* stays in vocn. Dest loads the
//! primitive set vocn returns into a [`Constraints`] chart. Dest
//! does not generate Bond / Angle / Dihedral primitives.

use ndarray::ArrayView1;

use crate::SaddleError;
use crate::constraints::Constraints;

/// One vocn primitive. Dest stores the atom indices only.
///
/// Atom indices are zero-based rows of the host's `N x 3` position array.
/// The primitive's value (Angstrom for a bond, radian for an angle and a
/// dihedral) is read from the positions when the set is loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Primitive {
    /// A bond between atoms `[i, j]`; the coordinate is the distance in Angstrom.
    Bond([usize; 2]),
    /// A bend `[i, j, k]` with the vertex at `j`; the coordinate is the angle in radian.
    Angle([usize; 3]),
    /// A torsion about the bond `j-k` of the chain `[i, j, k, l]`; the
    /// coordinate is the dihedral angle in radian.
    Dihedral([usize; 4]),
}

/// Primitive set returned by vocn auto-find.
///
/// The set is ordered bonds, then angles, then dihedrals. It holds atom
/// indices only, so it carries no units.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Found {
    primitives: Vec<Primitive>,
}

impl Found {
    /// Empty set. [`InternalPes::from_find`][crate::InternalPes::from_find]
    /// refuses it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pack bonds, then angles, then dihedrals (Sella internals order).
    ///
    /// Each iterator yields zero-based atom-index arrays. No index is
    /// checked here; [`Found::load`] reports an index outside the chart.
    pub fn from_parts(
        bonds: impl IntoIterator<Item = [usize; 2]>,
        angles: impl IntoIterator<Item = [usize; 3]>,
        dihedrals: impl IntoIterator<Item = [usize; 4]>,
    ) -> Self {
        let mut primitives = Vec::new();
        for atoms in bonds {
            primitives.push(Primitive::Bond(atoms));
        }
        for atoms in angles {
            primitives.push(Primitive::Angle(atoms));
        }
        for atoms in dihedrals {
            primitives.push(Primitive::Dihedral(atoms));
        }
        Self { primitives }
    }

    /// Active primitives in Sella packing order: all bonds, all angles, all
    /// dihedrals, each group in the order given to [`Found::from_parts`].
    pub fn primitives(&self) -> &[Primitive] {
        &self.primitives
    }

    /// Chart length dest loads into [`crate::InternalPes`]: one internal
    /// coordinate per primitive.
    pub fn n_int(&self) -> usize {
        self.primitives.len()
    }

    /// Whether the set holds no primitive.
    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }

    /// Push the vocn set onto a chart. Targets are the current values.
    ///
    /// `x` is the flat `3N` Cartesian position vector in Angstrom, atom-major
    /// `[x0, y0, z0, x1, ...]`. Each primitive is fixed at its value in `x`
    /// (Angstrom for bonds, radian for angles and dihedrals).
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when `x` is not the `3N` frame of the chart or
    /// an atom index is outside it, and [`SaddleError::NonFinite`] when a
    /// primitive is degenerate in `x` (coincident atoms, a collinear angle
    /// or dihedral). The chart keeps the primitives loaded before the
    /// failing one.
    pub fn load(&self, chart: &mut Constraints, x: ArrayView1<f64>) -> Result<(), SaddleError> {
        for p in &self.primitives {
            match p {
                Primitive::Bond(atoms) => chart.fix_bond(*atoms, x, None)?,
                Primitive::Angle(atoms) => chart.fix_angle(*atoms, x, None)?,
                Primitive::Dihedral(atoms) => chart.fix_dihedral(*atoms, x, None)?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::pack_cart;

    #[test]
    fn from_parts_counts_sella_order() {
        let found = Found::from_parts([[0, 1], [0, 2]], [[1, 0, 2]], []);
        assert_eq!(found.n_int(), 3);
        assert!(!found.is_empty());
        match found.primitives() {
            [
                Primitive::Bond([0, 1]),
                Primitive::Bond([0, 2]),
                Primitive::Angle([1, 0, 2]),
            ] => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn load_writes_the_chart() {
        let x = pack_cart(&[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]);
        let found = Found::from_parts([[0, 1], [0, 2]], [[1, 0, 2]], []);
        let mut chart = Constraints::new(3).unwrap();
        found.load(&mut chart, x.view()).unwrap();
        let c = chart.counts();
        assert_eq!(c.nbonds, 2);
        assert_eq!(c.nangles, 1);
        assert_eq!(c.ndihedrals, 0);
        assert_eq!(chart.equalities().len(), found.n_int());
    }
}
