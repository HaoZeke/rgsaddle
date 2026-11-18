//! A periodic band whose images sit more than half a cell apart.
//!
//! One atom rides a downhill ramp along `x` in a cell of 10 and a
//! second atom, with no gradient, sits at the same place in the first
//! two images and 4.5 above it in the last. The spring pulls the first
//! atom toward the last image, so it crosses the half-cell displacement
//! of its link to the first image at `x = 5`. A band that wraps every
//! link afresh flips the tangent and the spring force there, toward the
//! cut from both sides, and freezes on it; a band that keeps the
//! translation of each link relaxes past it.

use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, Cell, SaddleError};

struct Ramp;

impl BandSurface for Ramp {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for r in 0..positions.nrows() {
            energies[r] = -0.1 * positions[(r, 0)];
            gradients
                .row_mut(r)
                .assign(&array![-0.1, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
        Ok(())
    }
}

fn session() -> BandSession {
    let config = BandConfig::default()
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_remove_translation(false)
        .with_force_tol(1e-4)
        .with_cell(Some(Cell([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ])));
    let positions = array![
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [4.9, 0.0, 0.0, 0.0, 0.0, 0.0],
        [12.0, 0.0, 0.0, 0.0, 0.0, 4.5],
    ];
    BandSession::new(config, positions).unwrap()
}

#[test]
fn band_relaxes_past_the_half_cell_displacement() {
    let mut band = session();
    let mut report = None;
    for _ in 0..500 {
        let step = band.step(&Ramp).expect("the band must not stall");
        let done = step.status == BandStatus::Converged;
        report = Some(step);
        if done {
            break;
        }
    }
    let report = report.unwrap();
    assert_eq!(report.status, BandStatus::Converged);
    assert!(
        band.positions()[(1, 0)] > 5.5,
        "the image stopped at the cut: {}",
        band.positions()[(1, 0)]
    );
}
