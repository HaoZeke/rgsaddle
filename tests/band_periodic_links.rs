use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, Cell, SaddleError};

struct RampTwoAtoms;

impl BandSurface for RampTwoAtoms {
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

fn periodic_ramp_session() -> BandSession {
    let config = BandConfig::default()
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_remove_translation(false)
        .with_force_tol(0.0001)
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
    BandSession::new(config, positions).expect("periodic ramp session")
}

#[test]
fn band_relaxes_past_the_half_cell_displacement() {
    let mut band = periodic_ramp_session();
    let mut report = None;
    for _ in 0..500 {
        let step = band.step(&RampTwoAtoms).expect("band step");
        let done = step.status == BandStatus::Converged;
        report = Some(step);
        if done {
            break;
        }
    }
    let report = report.expect("a report");
    assert_eq!(report.status, BandStatus::Converged);
    assert!(
        band.positions()[(1, 0)] > 5.5,
        "middle image stayed near the periodic cut: {}",
        band.positions()[(1, 0)]
    );
}
