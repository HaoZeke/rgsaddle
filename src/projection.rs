//! Force projections, ported from eOn NEBForceProjection.cpp and
//! NEBProjection.cpp.
//!
//! Every vector is a flat `3N` array over one band image, atom-major
//! `[x0, y0, z0, x1, ...]`. Forces and spring forces are in eV/Angstrom,
//! tangents are unitless unit vectors.

use crate::spring::SpringForces;
use ndarray::{Array1, ArrayView1, ArrayViewMut1, Zip};

/// Which band projection assembles the relaxed force.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProjectionKind {
    /// Mills, Jónsson, and Schenter, Surf. Sci. 324, 305 (1995),
    /// doi:10.1016/0039-6028(94)00731-4: the full spring force plus the full
    /// true force, with no projection. It cuts corners and sags on a curved
    /// path, which is why the NEB projection exists.
    PlainElasticBand,
    /// Jónsson, Mills, and Jacobsen, in Classical and Quantum Dynamics in
    /// Condensed Phase Simulations (World Scientific, 1998), pp. 385-404,
    /// doi:10.1142/9789812839664_0016: the spring force along the tangent
    /// plus the true force perpendicular to it. This is the default and the
    /// standard NEB projection.
    #[default]
    Neb,
    /// Trygubenko and Wales, J. Chem. Phys. 120, 2082 (2004),
    /// doi:10.1063/1.1636455: the NEB force plus the component of the
    /// perpendicular spring force that is perpendicular to the perpendicular
    /// true force, switched by `(2/pi) atan(|F_perp|^2 / |F_spring_perp|^2)`.
    DoublyNudged,
}

/// True force with its tangent component removed: `F - (F . t) t`.
///
/// `force` is the true force on one image in eV/Angstrom and `tangent` is its
/// unit tangent, both length `3N`. The result has length `3N`, in
/// eV/Angstrom. `tangent` must be unit length; it is not normalised here.
pub fn force_perp(force: ArrayView1<f64>, tangent: ArrayView1<f64>) -> Array1<f64> {
    crate::stencil::perp(force, tangent)
}

/// Climbing-image force: `F - 2 (F . t) t + F_dneb`.
///
/// The climbing image of Henkelman, Uberuaga, and Jónsson, J. Chem. Phys.
/// 113, 9901 (2000), doi:10.1063/1.1329672, has the spring force removed and
/// the tangent component of the true force inverted, so the image climbs
/// toward the saddle along the band. The reflection is
/// [`crate::stencil::reflect`] (Goswami, Gunde, and Jónsson, Front. Chem.
/// (2026), doi:10.3389/fchem.2026.1807063, Eq. 9).
///
/// `force` is the true force in eV/Angstrom, `tangent` the unit tangent and
/// `force_dneb` the [`dneb_component`] (pass zeros when the band is not
/// doubly nudged), all length `3N`. The result has length `3N`, in
/// eV/Angstrom.
pub fn climbing_image_force(
    force: ArrayView1<f64>,
    tangent: ArrayView1<f64>,
    force_dneb: ArrayView1<f64>,
) -> Array1<f64> {
    crate::stencil::reflect(force, tangent) + force_dneb
}

/// DNEB correction with the Trygubenko–Wales atan switching
/// (J. Chem. Phys. 120, 2082 (2004), doi:10.1063/1.1636455).
///
/// `force_spring` is the full spring force, `tangent` the unit tangent and
/// `f_perp` the true force perpendicular to the tangent ([`force_perp`]),
/// all length `3N` in eV/Angstrom (the tangent is unitless). The result has
/// length `3N`, in eV/Angstrom. It is the zero vector when the
/// perpendicular spring force or the perpendicular true force has a norm of
/// at most `1e-10` eV/Angstrom.
pub fn dneb_component(
    force_spring: ArrayView1<f64>,
    tangent: ArrayView1<f64>,
    f_perp: ArrayView1<f64>,
) -> Array1<f64> {
    let mut dneb = Array1::zeros(tangent.len());
    let mut scratch = Array1::zeros(tangent.len());
    dneb_component_into(
        force_spring,
        tangent,
        f_perp,
        dneb.view_mut(),
        scratch.view_mut(),
    );
    dneb
}

/// [`dneb_component`] written into `dneb`; `scratch` holds the unit
/// perpendicular force on the way.
pub(crate) fn dneb_component_into(
    force_spring: ArrayView1<f64>,
    tangent: ArrayView1<f64>,
    f_perp: ArrayView1<f64>,
    mut dneb: ArrayViewMut1<f64>,
    mut scratch: ArrayViewMut1<f64>,
) {
    crate::stencil::perp_into(force_spring, tangent, dneb.view_mut());
    let spring_perp_norm = dneb.dot(&dneb).sqrt();
    let force_perp_norm = f_perp.dot(&f_perp).sqrt();
    if spring_perp_norm > 1e-10 && force_perp_norm > 1e-10 {
        Zip::from(scratch.view_mut())
            .and(f_perp)
            .for_each(|h, &f| *h = f / force_perp_norm);
        let overlap = dneb.dot(&scratch);
        let switching = 2.0 / std::f64::consts::PI
            * ((force_perp_norm * force_perp_norm) / (spring_perp_norm * spring_perp_norm)).atan();
        Zip::from(dneb)
            .and(scratch.view())
            .for_each(|d, &h| *d = (*d - h * overlap) * switching);
        return;
    }
    dneb.fill(0.0);
}

impl ProjectionKind {
    /// Projected relaxation force at one non-climbing interior image.
    ///
    /// `force` is the true force (eV/Angstrom, length `3N`), `tangent` the unit
    /// tangent and `spring` the [`SpringForces`] of the same image. The
    /// result has length `3N`, in eV/Angstrom, and is the force a minimizer
    /// follows for that image. [`ProjectionKind::PlainElasticBand`] ignores
    /// `tangent`. The climbing image does not go through this method; use
    /// [`climbing_image_force`].
    pub fn project(
        &self,
        force: ArrayView1<f64>,
        tangent: ArrayView1<f64>,
        spring: &SpringForces,
    ) -> Array1<f64> {
        match self {
            ProjectionKind::PlainElasticBand => &spring.full + &force,
            ProjectionKind::Neb => {
                let fp = force_perp(force, tangent);
                &spring.parallel + &fp
            }
            ProjectionKind::DoublyNudged => {
                let fp = force_perp(force, tangent);
                let dneb = dneb_component(spring.full.view(), tangent, fp.view());
                &spring.parallel + &fp + &dneb
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn force_perp_is_orthogonal_to_tangent() {
        let f = array![1.0, 2.0, 3.0];
        let t = array![1.0, 0.0, 0.0];
        let fp = force_perp(f.view(), t.view());
        assert_abs_diff_eq!(fp.dot(&t), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fp[1], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn climbing_force_inverts_the_parallel_component() {
        let f = array![1.0, 2.0];
        let t = array![1.0, 0.0];
        let zero = array![0.0, 0.0];
        let fc = climbing_image_force(f.view(), t.view(), zero.view());
        assert_abs_diff_eq!(fc[0], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fc[1], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn dneb_switching_stays_within_the_spring_perp() {
        let spring = array![0.0, 0.4];
        let t = array![1.0, 0.0];
        let fp = array![0.0, 0.1];
        let d = dneb_component(spring.view(), t.view(), fp.view());
        // spring_perp is parallel to f_perp here: the projection
        // removes it entirely.
        assert_abs_diff_eq!(d[1], 0.0, epsilon = 1e-12);
    }
}
