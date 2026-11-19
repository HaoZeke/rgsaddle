//! Baker restricted-step partitioned RFO.
//!
//! The canonical implementation is [`crate::prfo`], where each item is
//! documented with its units, argument shapes and errors. This module
//! only re-exports [`PrfoKind`], [`partitioned_rfo_eigen`],
//! [`restricted_partitioned_rfo`] and [`restricted_prfo_displacement`] under
//! the older path.

pub use crate::prfo::{
    PrfoKind, partitioned_rfo_eigen, restricted_partitioned_rfo, restricted_prfo_displacement,
};
