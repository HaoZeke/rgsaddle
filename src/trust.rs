//! Dense quadratic steps restricted to a Euclidean trust ball.
//!
//! The returned step includes its boundary and norm information.
//! The restriction controller and eigensolver are provided by rgmin.
//!
//! [`TrustRegion`] holds the trust radius and [`RestrictedStep`] is the
//! step it returned. Both are re-exported from `rgmin` and keep its units:
//! the radius and the step norm are in the length unit of the coordinates
//! (Angstrom for Cartesian positions) and the model energy change is in the
//! energy unit (eV). Their methods and errors are documented in `rgmin`.

pub use rgmin::{RestrictedStep, TrustRegion};
