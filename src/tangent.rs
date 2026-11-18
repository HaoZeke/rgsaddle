//! Band tangents, ported from eOn NEBTangent.cpp with identical
//! branch structure.
//!
//! A tangent is a unit vector in the 3N configuration space of one band
//! image. It is built from the two neighbour displacements (Angstrom) and
//! the three image energies (eV). [`TangentKind::Improved`] is ASE's
//! `method="improvedtangent"` in `ase.mep.NEB` and eOn's improved
//! `NEBTangent`.

use ndarray::{Array1, ArrayView1};

use crate::error::SaddleError;

/// Which tangent estimate the band uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TangentKind {
    /// Mills, Jónsson, and Schenter, Surf. Sci. 324, 305 (1995),
    /// doi:10.1016/0039-6028(94)00731-4: the forward difference alone,
    /// `R[i+1] - R[i]`, normalised.
    Simple,
    /// Henkelman and Jónsson, J. Chem. Phys. 113, 9978 (2000),
    /// doi:10.1063/1.1323224: the segment toward the higher neighbour at
    /// monotonic points, and an energy-weighted blend of both segments at
    /// extrema. This is the default.
    #[default]
    Improved,
}

/// Unit tangent at one interior image.
///
/// `pos_diff_next` is `R[i+1] - R[i]` and `pos_diff_prev` is
/// `R[i] - R[i-1]`, both flat `3N` vectors in Angstrom, atom-major
/// `[x0, y0, z0, x1, ...]` (the host minimum-images both when periodic).
/// `energy`, `energy_prev` and `energy_next` are the energies in eV of
/// image `i`, `i-1` and `i+1`, in that order. [`TangentKind::Simple`] ignores
/// them. The result has length `3N` and unit norm.
///
/// # Errors
///
/// [`SaddleError::Invalid`] when the weighted tangent, the forward segment
/// and the backward segment all have a norm under `1e-10` Angstrom or are not
/// finite, which is an image sitting on both neighbours.
///
/// The arithmetic lives on [`crate::stencil::unit_tangent`].
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use rgsaddle::tangent::{TangentKind, compute_tangent};
///
/// // A collinear image on an uphill segment: the improved tangent is the
/// // forward segment, normalised.
/// let next = array![0.0, 0.0, 2.0];
/// let prev = array![0.0, 0.0, 1.0];
/// let t = compute_tangent(TangentKind::Improved, next.view(), prev.view(), 0.0, -1.0, 1.0)?;
/// assert_eq!(t, array![0.0, 0.0, 1.0]);
/// # Ok::<(), rgsaddle::SaddleError>(())
/// ```
pub fn compute_tangent(
    kind: TangentKind,
    pos_diff_next: ArrayView1<f64>,
    pos_diff_prev: ArrayView1<f64>,
    energy: f64,
    energy_prev: f64,
    energy_next: f64,
) -> Result<Array1<f64>, SaddleError> {
    let rule = match kind {
        TangentKind::Simple => crate::stencil::TangentRule::Simple,
        TangentKind::Improved => crate::stencil::TangentRule::Improved,
    };
    crate::stencil::unit_tangent(
        rule,
        pos_diff_next,
        pos_diff_prev,
        energy,
        energy_prev,
        energy_next,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn improved_picks_uphill_forward() {
        let next = array![1.0, 0.0];
        let prev = array![0.0, 1.0];
        let t = compute_tangent(
            TangentKind::Improved,
            next.view(),
            prev.view(),
            1.0,
            0.0,
            2.0,
        )
        .unwrap();
        assert_abs_diff_eq!(t[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(t[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn improved_picks_downhill_backward() {
        let next = array![1.0, 0.0];
        let prev = array![0.0, 1.0];
        let t = compute_tangent(
            TangentKind::Improved,
            next.view(),
            prev.view(),
            1.0,
            2.0,
            0.0,
        )
        .unwrap();
        assert_abs_diff_eq!(t[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(t[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn improved_weights_extremum_toward_steeper_side() {
        let next = array![1.0, 0.0];
        let prev = array![0.0, 1.0];
        // Maximum: prev side steeper (|dE_prev| = 2 > |dE_next| = 1).
        let t = compute_tangent(
            TangentKind::Improved,
            next.view(),
            prev.view(),
            3.0,
            1.0,
            2.0,
        )
        .unwrap();
        // energy_diff_prev (-2) < energy_diff_next (-1): next*max + prev*min.
        let expected = (array![1.0, 0.0] * 2.0 + array![0.0, 1.0] * 1.0) / 5.0f64.sqrt();
        assert_abs_diff_eq!(t[0], expected[0], epsilon = 1e-12);
        assert_abs_diff_eq!(t[1], expected[1], epsilon = 1e-12);
    }

    #[test]
    fn zero_tangent_falls_back_to_forward_difference() {
        let next = array![0.5, 0.0];
        let prev = array![-0.5, 0.0];
        // Extremum with equal, zero energy differences: combination is
        // zero-weighted; fallback normalizes the forward difference.
        let t = compute_tangent(
            TangentKind::Improved,
            next.view(),
            prev.view(),
            1.0,
            1.0,
            1.0,
        )
        .unwrap();
        assert_abs_diff_eq!(t[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn zero_forward_difference_falls_back_to_the_backward_one() {
        let next = array![0.0, 0.0];
        let prev = array![0.0, 0.3];
        let t =
            compute_tangent(TangentKind::Simple, next.view(), prev.view(), 1.0, 0.0, 2.0).unwrap();
        assert_abs_diff_eq!(t[1], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(t.dot(&t), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn coincident_images_have_no_tangent() {
        let zero = array![0.0, 0.0, 0.0];
        let err = compute_tangent(
            TangentKind::Improved,
            zero.view(),
            zero.view(),
            1.0,
            1.0,
            1.0,
        )
        .unwrap_err();
        assert!(matches!(err, SaddleError::Invalid(_)), "{err}");
    }
}
