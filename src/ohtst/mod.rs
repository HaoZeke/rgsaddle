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
//! Nothing here names a system. The surface carries the cell, so a crystal
//! at a fixed volume or lattice parameter for each temperature is a
//! surface built with that cell.

mod optimize;
mod plane;
mod sample;

pub use optimize::{PlaneOptimizer, PlaneOptimizerConfig, PlaneOptimum, PlaneStep, PlaneTrace};
pub use plane::{Hyperplane, MassWeighted};
pub use sample::{LangevinConfig, PlaneSample, sample_plane};
