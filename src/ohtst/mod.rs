//! Optimal hyperplanar transition state theory (OH-TST).
//!
//! Jóhannesson and Jónsson, J. Chem. Phys. 115, 9644 (2001),
//! doi:10.1063/1.1415499, take the dividing surface to be a hyperplane
//! and move it until its free energy is a maximum. By the variational
//! principle of transition state theory that plane gives the smallest
//! hyperplanar TST rate. Bligaard and Jónsson, Comput. Phys. Commun. 169,
//! 284 (2005), doi:10.1016/j.cpc.2005.03.064, follow the same optimum on
//! two-dimensional surfaces where the free energy of a line is a
//! one-dimensional integral.
//!
//! Coordinates are mass-weighted, `q = M^{1/2} x`. [`MassWeighted`] wraps
//! a Cartesian [`crate::PointSurface`] with its masses; a surface already
//! in these coordinates is used as it is. A [`Hyperplane`] is a point `p`
//! and a unit normal `ν` that points from the reactant to the product. Its
//! free energy is
//!
//! `A(p, ν) = -kT ln ∫ δ(ν · (q - p)) exp(-V(q) / kT) dq`.
//!
//! `A` depends on `p` only through `ν · p`, so the `n - 1` in-plane moves
//! of `p` are a gauge.
//!
//! A rate takes five calls. [`Hyperplane::from_min_mode`] seeds the plane
//! from a converged dimer. [`PlaneOptimizer`] climbs `A`. [`basin_density`]
//! gives `Z_H / Z_R` on a plane through the minimum from parallel planes.
//! [`reversible_work`] gives `ΔA` from that plane to the optimum, by
//! thermodynamic integration and by BAR. [`ohtst_rate`] returns
//! `k = sqrt(kT / 2π) (Z_H / Z_R) exp(-ΔA / kT)`, with unit mass in these
//! coordinates, and [`plane_transmission`] the Bennett–Chandler `κ`.
//! Planes are sampled by projected BAOAB ([`PlaneIntegrator`]); a
//! [`HarmonicReference`] adds Stein control variates; and
//! [`anharmonic_correction`] gives the first-order correction to harmonic
//! TST from Hessian actions alone. No full Hessian is formed anywhere.
//! These follow the decisions of Sec. 2 of the OH-TST design survey.
//!
//! Nothing here names a system. A crystal at a fixed volume or lattice
//! parameter for each temperature is a surface evaluated in that cell:
//! wrap it in [`FixedCell`] with the cell for the temperature, then in
//! [`MassWeighted`].

mod anharmonic;
mod basin;
mod control;
mod optimize;
mod plane;
mod rate;
mod reactant;
mod sample;
mod spline;
mod work;

pub use anharmonic::{AnharmonicConfig, AnharmonicCorrection, anharmonic_correction};
pub use basin::{BasinConfig, BasinDensity, BasinEnd, basin_density};
pub use control::HarmonicReference;
pub use optimize::{PlaneOptimizer, PlaneOptimizerConfig, PlaneOptimum, PlaneStep, PlaneTrace};
pub use plane::{FixedCell, Hyperplane, MassWeighted};
pub use rate::{OhTstRate, PlaneDensity, flux_mass, ohtst_rate, plane_transmission};
pub use reactant::{ReactantDensity, reactant_density};
pub use sample::{LangevinConfig, PlaneIntegrator, PlaneSample, Uncontrolled, sample_plane};
pub use work::{ReversibleWork, ReversibleWorkConfig, WorkNode, reversible_work};
