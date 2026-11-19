use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{
    AutoNeb, BandConfig, BandSession, BandStatus, BandSurface, Cell, GrowingStringConfig,
    GrowingStringSession, SaddleError,
};

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

struct Flat;

impl BandSurface for Flat {
    fn eval(
        &self,
        _positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        energies.fill(0.0);
        gradients.fill(0.0);
        Ok(())
    }
}

fn bad_cells() -> [Cell; 3] {
    [
        Cell([[f64::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        Cell([[f64::INFINITY, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        Cell([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    ]
}

#[test]
fn band_forces_reject_a_bad_cell() {
    let positions = array![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]];
    for cell in bad_cells() {
        let config = BandConfig::default().with_cell(Some(cell));
        let result = rgsaddle::band_forces(&config, &Flat, positions.view(), None);
        assert!(result.is_err(), "band_forces accepted {cell:?}");
    }
}

#[test]
fn growing_string_rejects_a_bad_cell() {
    let reactant = array![0.0, 0.0, 0.0];
    let product = array![1.0, 0.0, 0.0];
    for cell in bad_cells() {
        let config = GrowingStringConfig::default()
            .with_cell(Some(cell))
            .with_min_mode(None)
            .with_saddle_check(None);
        let result = GrowingStringSession::new(config, reactant.clone(), product.clone());
        assert!(result.is_err(), "growing string accepted {cell:?}");
    }
}

#[test]
fn autoneb_rejects_a_bad_cell() {
    let initial = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    for cell in bad_cells() {
        let config = BandConfig::default()
            .with_cell(Some(cell))
            .with_min_mode(None);
        let result = AutoNeb::new(config, initial.clone());
        assert!(result.is_err(), "AutoNEB accepted {cell:?}");
    }
}
