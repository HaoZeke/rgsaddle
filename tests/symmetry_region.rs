use ndarray::{Array1, Array2, array};
use rgsaddle::{CrystalSymmetryConfig, CrystalSymmetryOperation, Region, SymmetryUniqueRegion};

fn operation(rotation: Array2<f64>) -> CrystalSymmetryOperation {
    CrystalSymmetryOperation::default().with_rotation(rotation)
}

fn c4v() -> Vec<CrystalSymmetryOperation> {
    let mut operations = Vec::new();
    for rotation in [
        array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        array![[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        array![[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        array![[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    ] {
        operations.push(operation(rotation.clone()));
        let reflection = array![[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
        operations.push(operation(rotation.dot(&reflection)));
    }
    operations
}

fn cubic_group(centre: Array1<f64>) -> Vec<CrystalSymmetryOperation> {
    let mut operations = Vec::new();
    for permutation in [
        [0_usize, 1_usize, 2_usize],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        for signs in 0_u8..8 {
            let mut rotation = Array2::zeros((3, 3));
            for row in 0..3 {
                rotation[[row, permutation[row]]] =
                    if signs & (1 << row) == 0 { 1.0 } else { -1.0 };
            }
            operations.push(
                CrystalSymmetryOperation::default()
                    .with_rotation(rotation)
                    .with_centre(centre.clone()),
            );
        }
    }
    operations
}

#[test]
fn analytic_c4v_orbit_has_four_saddles() {
    let reactant = array![[0.0, 0.0, 0.0]];
    let saddle = array![[1.0, 0.0, 0.0]];
    let (region, degeneracy) = SymmetryUniqueRegion::from_crystal_operations(
        reactant.view(),
        saddle.view(),
        &[0],
        &c4v(),
        &CrystalSymmetryConfig::default(),
    )
    .unwrap();
    assert_eq!(degeneracy, 4);
    assert_eq!(region.degeneracy(), 4);
    assert_eq!(region.planar_boundaries().unwrap().len(), 3);
}

#[test]
fn fcc_vacancy_nearest_neighbour_hop_has_degeneracy_twelve() {
    let centre = array![5.0, 5.0, 5.0];
    let mut reactant = Array2::zeros((12, 3));
    let mut row = 0;
    for zero_axis in 0..3 {
        for first_sign in [-1.0, 1.0] {
            for second_sign in [-1.0, 1.0] {
                let mut offset = [first_sign, second_sign, 0.0];
                offset.rotate_right(zero_axis);
                for axis in 0..3 {
                    reactant[[row, axis]] = centre[axis] + offset[axis];
                }
                row += 1;
            }
        }
    }
    let mut saddle = reactant.clone();
    for axis in 0..3 {
        saddle[[0, axis]] = 0.5 * (reactant[[0, axis]] + centre[axis]);
    }
    let config = CrystalSymmetryConfig::default()
        .with_cell(minimage::Cell::ortho(10.0, 10.0, 10.0).unwrap());
    let (region, degeneracy) = SymmetryUniqueRegion::from_crystal_operations(
        reactant.view(),
        saddle.view(),
        &[0; 12],
        &cubic_group(centre),
        &config,
    )
    .unwrap();
    assert_eq!(degeneracy, 12);
    assert_eq!(region.degeneracy(), 12);
}

#[test]
fn identity_group_gives_the_trivial_region() {
    let geometry = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let identity = operation(Array2::eye(3)).with_permutation(vec![0, 1]);
    let (region, degeneracy) = SymmetryUniqueRegion::from_crystal_operations(
        geometry.view(),
        geometry.view(),
        &[0, 0],
        &[identity],
        &CrystalSymmetryConfig::default(),
    )
    .unwrap();
    assert_eq!(degeneracy, 1);
    assert!(region.planar_boundaries().unwrap().is_empty());
    assert!(
        region
            .contains(array![9.0, -2.0, 4.0, 3.0, 7.0, 1.0].view())
            .unwrap()
    );
}

#[test]
fn invalid_operations_are_rejected() {
    let reactant = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let non_orthogonal = operation(array![[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert!(
        SymmetryUniqueRegion::from_crystal_operations(
            reactant.view(),
            reactant.view(),
            &[0, 1],
            &[non_orthogonal],
            &CrystalSymmetryConfig::default(),
        )
        .is_err()
    );

    let cross_species = operation(Array2::eye(3)).with_permutation(vec![1, 0]);
    assert!(
        SymmetryUniqueRegion::from_crystal_operations(
            reactant.view(),
            reactant.view(),
            &[0, 1],
            &[cross_species],
            &CrystalSymmetryConfig::default(),
        )
        .is_err()
    );
}
