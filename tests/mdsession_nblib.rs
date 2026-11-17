//! Smoke test of the rgpot MD session backend on an argon box.
//!
//! The backend is engine-generic; this test drives it through the generic
//! path with the GROMACS nblib engine of rgpot, `librgpot_nblib`, whose
//! symbol prefix is `rgpot_nblib`. Set `RGPOT_NBLIB_ENGINE` to that library;
//! without it every test returns at once.
#![cfg(feature = "mdsession")]

use std::sync::Arc;

use ndarray::Array1;
use rgsaddle::region::PlanarBoundary;
use rgsaddle::{
    Hyperplane, HyperplaneCoordinates, HyperplaneDynamics, HyperplaneEnsemble,
    HyperplaneSampleBudget, HyperplaneSampler, MdSessionBackend, MdSessionConfig, MdSessionSystem,
    SaddleError, TransmissionConfig,
};

const PREFIX: &str = "rgpot_nblib";
const CELLS: usize = 3;
const SPACING: f64 = 5.3;
const MASS: f64 = 39.948;
const KT_EV: f64 = 0.0086;
const TIME_STEP_PS: f64 = 0.004;
/// Roundoff of a projection and a force sum.
const SUM: f64 = 1.0 / (1u64 << 24) as f64;
/// Two runs of the same trajectory in different frames agree to this much.
const FRAMES: f64 = 1.0 / (1u64 << 16) as f64;

fn library() -> Option<String> {
    let path = std::env::var("RGPOT_NBLIB_ENGINE").ok()?;
    (!path.is_empty()).then_some(path)
}

fn config(library: &str) -> MdSessionConfig {
    MdSessionConfig::default()
        .with_prefix(PREFIX)
        .with_library_path(library)
        .with_time_step_ps(TIME_STEP_PS)
        .with_friction_per_ps(5.0)
}

fn argon() -> MdSessionSystem {
    let sigma_nm: f64 = 0.34;
    let epsilon_kj = 0.9961;
    let c6 = 4.0 * epsilon_kj * sigma_nm.powi(6);
    let c12 = 4.0 * epsilon_kj * sigma_nm.powi(12);
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut positions = Vec::new();
    for i in 0..CELLS {
        for j in 0..CELLS {
            for k in 0..CELLS {
                for site in &basis {
                    let origin = [i as f64 + site[0], j as f64 + site[1], k as f64 + site[2]];
                    for (axis, value) in origin.iter().enumerate() {
                        let index = positions.len() as f64;
                        let shake = 0.1 * (1.7 * index + 0.3 * axis as f64).sin();
                        positions.push(value * SPACING + shake);
                    }
                }
            }
        }
    }
    let atoms = positions.len() / 3;
    let spec = format!(
        "nblib 1\ncutoff 0.7\ncoulomb none\natomtype Ar {MASS} {c6:?} {c12:?}\n\
         molecule argon {atoms}\n  atom Ar Ar 0.0\nend\n"
    );
    let side = CELLS as f64 * SPACING;
    MdSessionSystem::default()
        .with_spec(spec)
        .with_positions(Array1::from_vec(positions))
        .with_box_matrix([side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side])
}

fn direction(length: usize) -> Array1<f64> {
    (0..length)
        .map(|index| (0.9 * index as f64 + 0.2).cos())
        .collect()
}

fn budget(coordinates: HyperplaneCoordinates) -> HyperplaneSampleBudget {
    HyperplaneSampleBudget::default()
        .with_thermal(KT_EV)
        .with_seed(3)
        .with_thermalize(20)
        .with_stride(5)
        .with_samples(8)
        .with_coordinates(coordinates)
}

fn cartesian_plane(system: &MdSessionSystem) -> Hyperplane {
    Hyperplane::new(system.positions.clone(), direction(system.positions.len())).unwrap()
}

fn mass_weighted_plane(system: &MdSessionSystem) -> Hyperplane {
    let masses = vec![MASS; system.positions.len()];
    Hyperplane::from_cartesian(
        system.positions.view(),
        direction(system.positions.len()).view(),
        &masses,
    )
    .unwrap()
}

#[test]
fn samples_stay_on_the_plane_with_tangent_forces() {
    let Some(library) = library() else { return };
    let system = argon();
    let plane = cartesian_plane(&system);
    let mut backend = MdSessionBackend::new(system, config(&library));
    let ensemble = backend
        .sample_hyperplane(&plane, &budget(HyperplaneCoordinates::Cartesian))
        .unwrap();
    assert_eq!(ensemble.observations.len(), 8);
    assert!(ensemble.evaluations > 0);
    // Equipartition over the 3N - 1 coordinates of the plane, to a factor
    // that a unit error in the thermostat would break.
    let kinetic = ensemble
        .observations
        .iter()
        .map(|o| o.observables["mdsession.kinetic_ev"])
        .sum::<f64>()
        / ensemble.observations.len() as f64;
    let expected = 0.5 * KT_EV * (plane.dim() - 1) as f64;
    assert!(
        kinetic > 0.5 * expected && kinetic < 1.5 * expected,
        "{kinetic} vs {expected}"
    );
    let scale = SPACING * CELLS as f64;
    for observation in &ensemble.observations {
        let position = observation.position.as_ref().unwrap();
        assert!(plane.distance(position.view()).abs() <= SUM * scale);
        let force = observation.projected_force.as_ref().unwrap();
        let biggest = force.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(plane.normal().dot(force).abs() <= SUM * biggest);
        assert!(observation.constraint_force.is_finite());
        assert!(observation.observables["mdsession.potential_ev"] < 0.0);
        assert!(observation.observables["mdsession.kinetic_ev"] > 0.0);
    }
}

#[test]
fn the_two_frames_describe_one_trajectory() {
    let Some(library) = library() else { return };
    let system = argon();
    let cartesian = cartesian_plane(&system);
    let weighted = mass_weighted_plane(&system);
    let first = MdSessionBackend::new(system.clone(), config(&library))
        .sample_hyperplane(&cartesian, &budget(HyperplaneCoordinates::Cartesian))
        .unwrap();
    let second = MdSessionBackend::new(system, config(&library))
        .sample_hyperplane(&weighted, &budget(HyperplaneCoordinates::MassWeighted))
        .unwrap();
    let root = MASS.sqrt();
    let scale = SPACING * CELLS as f64;
    for (a, b) in first.observations.iter().zip(&second.observations) {
        let x = a.position.as_ref().unwrap();
        let q = b.position.as_ref().unwrap();
        assert!(
            (x * root - q)
                .iter()
                .all(|d| d.abs() <= FRAMES * scale * root)
        );
        let gap = (a.constraint_force / root - b.constraint_force).abs();
        assert!(gap <= FRAMES * (a.constraint_force.abs() / root).max(SUM));
    }
}

#[test]
fn a_planar_region_holds_the_samples() {
    let Some(library) = library() else { return };
    let system = argon();
    let plane = cartesian_plane(&system);
    let mut tangent = Array1::zeros(system.positions.len());
    tangent[0] = 1.0;
    let along = plane.normal().dot(&tangent);
    tangent.scaled_add(-along, &plane.normal());
    let face =
        PlanarBoundary::new(&system.positions + &(&tangent * 0.05), tangent.clone()).unwrap();
    let budget = budget(HyperplaneCoordinates::Cartesian).with_samples(40);

    let free: HyperplaneEnsemble = MdSessionBackend::new(system.clone(), config(&library))
        .sample_hyperplane(&plane, &budget)
        .unwrap();
    let past = |ensemble: &HyperplaneEnsemble| {
        ensemble
            .observations
            .iter()
            .filter(|o| {
                face.signed_distance(o.position.as_ref().unwrap().view())
                    .unwrap()
                    > 0.0
            })
            .count()
    };
    assert!(past(&free) > 0, "the free walk never reaches the face");

    let held = MdSessionBackend::new(system, config(&library))
        .sample_hyperplane(&plane, &budget.with_region(Arc::new(face.clone())))
        .unwrap();
    assert_eq!(past(&held), 0);
}

#[test]
fn shooting_from_sampled_starts_gives_a_transmission_coefficient() {
    let Some(library) = library() else { return };
    let system = argon();
    let plane = mass_weighted_plane(&system);
    let mut backend = MdSessionBackend::new(system, config(&library));
    let starts: Vec<Array1<f64>> = backend
        .sample_hyperplane(&plane, &budget(HyperplaneCoordinates::MassWeighted))
        .unwrap()
        .observations
        .into_iter()
        .map(|o| o.position.unwrap())
        .collect();
    let shooting = TransmissionConfig::default()
        .with_thermal(KT_EV)
        .with_time_step(TIME_STEP_PS)
        .with_steps(40)
        .with_check_every(10)
        .with_shots(16)
        .with_blocks(4)
        .with_seed(11);
    let first = backend
        .shoot_hyperplane(&plane, &starts, &shooting)
        .unwrap();
    assert_eq!(first.shots + first.unassigned, 16);
    assert_eq!(first.kappa_of_t.len(), 4);
    assert_eq!(first.time.len(), 4);
    assert!(first.kappa.is_finite() && (0.0..=1.0).contains(&first.kappa));
    assert!(first.evaluations > 0);
    let again = backend
        .shoot_hyperplane(&plane, &starts, &shooting)
        .unwrap();
    assert_eq!(first.kappa_of_t, again.kappa_of_t);
    assert_eq!(first.reactive, again.reactive);

    let langevin = shooting.clone().with_friction(5.0);
    assert!(matches!(
        backend.shoot_hyperplane(&plane, &starts, &langevin),
        Err(SaddleError::Invalid(_))
    ));
    let langevin = langevin.with_hold_excluded(false);
    let result = backend
        .shoot_hyperplane(&plane, &starts, &langevin)
        .unwrap();
    assert_eq!(result.shots + result.unassigned, 16);
}

#[test]
fn a_wrong_prefix_names_the_missing_symbol() {
    let Some(library) = library() else { return };
    let system = argon();
    let plane = cartesian_plane(&system);
    let mut backend = MdSessionBackend::new(system, config(&library).with_prefix("rgpot_absent"));
    let error = backend
        .sample_hyperplane(&plane, &budget(HyperplaneCoordinates::Cartesian))
        .err()
        .unwrap();
    assert!(error.to_string().contains("rgpot_absent_md_"));
}
