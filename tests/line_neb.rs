use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{
    BandSurface, LineNebConfig, LineNebSession, LineNebStatus, NaturalCubic, SaddleError,
    fixed_energy_action, fixed_energy_action_force, fixed_energy_mep_seed,
};

fn analytic_surface(positions: ArrayView2<'_, f64>) -> (Array1<f64>, Array2<f64>) {
    let mut energies = Array1::zeros(positions.nrows());
    let mut gradients = Array2::zeros(positions.raw_dim());
    for (i, row) in positions.outer_iter().enumerate() {
        let x = row[0];
        let y = row[1];
        energies[i] = 0.7 * x * x + 0.4 * x * y + 1.1 * y * y + 2.0;
        gradients[(i, 0)] = 1.4 * x + 0.4 * y;
        gradients[(i, 1)] = 0.4 * x + 2.2 * y;
    }
    (energies, gradients)
}

#[test]
fn action_and_force_match_the_discrete_li_neb_equations() {
    let positions = array![[-1.0, 0.2], [-0.35, 0.55], [0.25, 0.48], [0.9, -0.1],];
    let masses = array![2.0, 5.0];
    let target = 0.3;
    let hbar = 1.7;
    let (energies, gradients) = analytic_surface(positions.view());
    let force = fixed_energy_action_force(
        positions.view(),
        energies.view(),
        gradients.view(),
        masses.view(),
        target,
        hbar,
    )
    .unwrap();
    let action = fixed_energy_action(
        positions.view(),
        energies.view(),
        masses.view(),
        target,
        hbar,
    )
    .unwrap();
    assert!((action - 2.689_136_149_698_181_4).abs() < 2.0e-15);
    let expected = array![
        [0.0, 0.0],
        [1.095_210_982_127_138_3, -1.666_250_779_155_905_6],
        [-1.360_661_380_911_351_5, -1.015_102_671_714_826_4],
        [0.0, 0.0],
    ];
    for (actual, expected) in force.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 2.0e-15);
    }
}

#[test]
fn endpoint_restraint_may_cross_the_target_energy() {
    let positions = array![[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]];
    let energies = array![0.9, 2.0, 0.8];
    let action = fixed_energy_action(
        positions.view(),
        energies.view(),
        array![1.0, 1.0].view(),
        1.0,
        1.0,
    )
    .unwrap();
    assert!(action.is_finite());
}

#[test]
fn natural_cubic_returns_value_and_derivatives() {
    let spline =
        NaturalCubic::new(vec![0.0, 1.0, 2.0], vec![vec![1.0], vec![3.0], vec![5.0]]).unwrap();
    let sample = spline.sample(0.5).unwrap();
    assert_eq!(sample.value, vec![2.0]);
    assert_eq!(sample.first, vec![2.0]);
    assert_eq!(sample.second, vec![0.0]);
}

struct DoubleWell;

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<'_, f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let x = row[0];
            let y = row[1];
            energies[i] = (x * x - 1.0).powi(2) + y * y;
            gradients[(i, 0)] = 4.0 * x * (x * x - 1.0);
            gradients[(i, 1)] = 2.0 * y;
        }
        Ok(())
    }
}

#[test]
fn energy_sliced_seed_follows_the_double_well_mep() {
    let path = fixed_energy_mep_seed(
        &DoubleWell,
        array![0.0, 0.0].view(),
        array![1.0, 0.0].view(),
        array![1.0, 2.0].view(),
        1.0,
        0.25,
        9,
    )
    .unwrap();
    assert_eq!(path.raw_dim(), ndarray::Ix2(9, 2));
    assert!(path.column(1).iter().all(|value| *value == 0.0));
    assert!(path[(4, 0)].abs() <= f64::EPSILON);
    assert!(((path[(4, 0)].powi(2) - 1.0).powi(2) - 1.0).abs() <= f64::EPSILON);
    for endpoint in [0, 8] {
        let x = path[(endpoint, 0)];
        assert!(((x * x - 1.0).powi(2) - 0.25).abs() <= f64::EPSILON * 8.0);
    }
}

#[test]
fn fixed_energy_double_well_path_converges() {
    let n_images = 9;
    let target: f64 = 0.25;
    let initial_endpoint_energy: f64 = 0.3;
    let turning_point = (1.0_f64 - initial_endpoint_energy.sqrt()).sqrt();
    let mut positions = Array2::zeros((n_images, 2));
    for i in 0..n_images {
        let fraction = i as f64 / (n_images - 1) as f64;
        positions[(i, 0)] = -turning_point + 2.0 * turning_point * fraction;
        positions[(i, 1)] = 0.18 * (std::f64::consts::PI * fraction).sin();
    }
    let config = LineNebConfig::default()
        .with_spring_k(0.4)
        .with_endpoint_kappa(1.5)
        .with_force_tol(2.0e-6)
        .with_energy(target)
        .with_masses(array![1.0, 2.0])
        .with_time_step(0.03);
    let mut session = LineNebSession::new(config, positions).unwrap();
    let host_force_budget = 80_000;
    let mut rows = 0;
    let report = loop {
        let report = session.step(&DoubleWell).unwrap();
        rows += report.surface_rows;
        if report.status == LineNebStatus::Converged {
            break report;
        }
        assert!(
            rows < host_force_budget,
            "host force budget exhausted: {report:?}"
        );
    };

    let converged = session.positions();
    assert!(converged.column(1).iter().all(|y| y.abs() < 2.0e-5));
    assert!(converged[(n_images / 2, 0)].abs() < 2.0e-5);
    assert!(converged[(n_images / 2, 1)].abs() < 2.0e-5);
    let saddle_x = converged[(n_images / 2, 0)];
    let saddle_y = converged[(n_images / 2, 1)];
    let saddle_energy = (saddle_x * saddle_x - 1.0).powi(2) + saddle_y * saddle_y;
    assert!((saddle_energy - 1.0).abs() < 2.0e-6);
    for endpoint in [0, n_images - 1] {
        let x = converged[(endpoint, 0)];
        let y = converged[(endpoint, 1)];
        let energy = (x * x - 1.0).powi(2) + y * y;
        assert!((energy - target).abs() < 2.0e-6);
    }
    assert!(report.max_interior_perpendicular <= 2.0e-6);
    assert!(report.max_endpoint_optimized <= 2.0e-6);
    assert!(report.sum_interior_perpendicular <= 2.0e-6);
}
