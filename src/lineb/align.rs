//! Mass-weighted quaternion alignment of a finite system.
//!
//! The path length of a molecule is the mass-weighted distance after the
//! optimal rigid motion is removed. The rotation is the Kabsch quaternion:
//! the largest eigenvector of the 4×4 matrix built from the mass-weighted
//! covariance, the same matrix eOn and ASE use for `align_to`. A single
//! Jacobi sweep that no longer reduces the off-diagonal sum is the fixed
//! point of that eigenproblem; the iteration count is not capped.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::SaddleError;

/// Rigid placement that takes one image onto the reference orientation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Placement {
    /// Maps centered mass-weighted coordinates of this image onto the
    /// reference orientation.
    pub rotation: [[f64; 3]; 3],
}

/// Align every image to image 0. `images` is mass-weighted, one row per
/// image, length `3 * atoms`. The three coordinate masses of an atom are
/// one atomic mass.
pub(crate) fn align_images(
    images: ArrayView2<f64>,
    sqrt_mass: ArrayView1<f64>,
    atoms: usize,
) -> Result<(Array2<f64>, Vec<Placement>), SaddleError> {
    if atoms < 2 {
        return Err(SaddleError::Invalid(
            "quaternion alignment needs at least two atoms".into(),
        ));
    }
    let dof = 3 * atoms;
    if images.ncols() != dof || sqrt_mass.len() != dof {
        return Err(SaddleError::Shape(
            "alignment expects 3 coordinates and one mass weight per atom".into(),
        ));
    }
    check_atom_masses(sqrt_mass, atoms)?;
    let n = images.nrows();
    if n == 0 {
        return Err(SaddleError::Shape(
            "alignment needs at least one image".into(),
        ));
    }
    let reference = images.row(0).to_owned();
    let mut aligned = Array2::zeros((n, dof));
    let mut placements = Vec::with_capacity(n);
    for i in 0..n {
        let (row, placement) =
            align_to_reference(images.row(i), reference.view(), sqrt_mass, atoms)?;
        aligned.row_mut(i).assign(&row);
        placements.push(placement);
    }
    Ok((aligned, placements))
}

/// Rotate a mass-weighted force back to the stored orientation and remove
/// the rigid motions the alignment quotient does not feel. `aligned` is
/// the centred mass-weighted geometry the force was computed on.
pub(crate) fn restore_force(
    force: &mut Array1<f64>,
    placement: &Placement,
    aligned: ArrayView1<f64>,
    sqrt_mass: ArrayView1<f64>,
    atoms: usize,
) {
    project_rigid_at(force, aligned, sqrt_mass, atoms);
    let mut rotated = Array1::zeros(force.len());
    for atom in 0..atoms {
        let f = [force[3 * atom], force[3 * atom + 1], force[3 * atom + 2]];
        for row in 0..3 {
            let mut acc = 0.0;
            for col in 0..3 {
                acc += placement.rotation[col][row] * f[col];
            }
            rotated[3 * atom + row] = acc;
        }
    }
    *force = rotated;
}

fn align_to_reference(
    image: ArrayView1<f64>,
    reference: ArrayView1<f64>,
    sqrt_mass: ArrayView1<f64>,
    atoms: usize,
) -> Result<(Array1<f64>, Placement), SaddleError> {
    let com = centre_of_mass(image, sqrt_mass, atoms)?;
    let com_ref = centre_of_mass(reference, sqrt_mass, atoms)?;
    let centered = center(image, sqrt_mass, com);
    let centered_ref = center(reference, sqrt_mass, com_ref);
    let rotation = kabsch(&centered, &centered_ref, atoms)?;
    let mut out = Array1::zeros(image.len());
    for atom in 0..atoms {
        let p = [
            centered[3 * atom],
            centered[3 * atom + 1],
            centered[3 * atom + 2],
        ];
        for row in 0..3 {
            let mut acc = 0.0;
            for col in 0..3 {
                acc += rotation[row][col] * p[col];
            }
            out[3 * atom + row] = acc;
        }
    }
    Ok((out, Placement { rotation }))
}

fn check_atom_masses(sqrt_mass: ArrayView1<f64>, atoms: usize) -> Result<(), SaddleError> {
    for atom in 0..atoms {
        let m = sqrt_mass[3 * atom];
        if !(m.is_finite() && m > 0.0) {
            return Err(SaddleError::Invalid(
                "atomic masses must be finite and positive".into(),
            ));
        }
        for axis in 1..3 {
            let other = sqrt_mass[3 * atom + axis];
            if (other - m).abs() > 0.0 {
                return Err(SaddleError::Invalid(
                    "the three coordinate masses of an atom are one atomic mass".into(),
                ));
            }
        }
    }
    Ok(())
}

fn centre_of_mass(
    q: ArrayView1<f64>,
    sqrt_mass: ArrayView1<f64>,
    atoms: usize,
) -> Result<[f64; 3], SaddleError> {
    let mut mass = 0.0;
    let mut moment = [0.0; 3];
    for atom in 0..atoms {
        let root = sqrt_mass[3 * atom];
        let m = root * root;
        mass += m;
        for axis in 0..3 {
            moment[axis] += root * q[3 * atom + axis];
        }
    }
    if mass <= 0.0 || mass.is_nan() {
        return Err(SaddleError::Invalid(
            "a molecule needs a positive total mass".into(),
        ));
    }
    Ok([moment[0] / mass, moment[1] / mass, moment[2] / mass])
}

/// `sqrt(m) (x - com)`, which is `q - sqrt(m) com`.
fn center(q: ArrayView1<f64>, sqrt_mass: ArrayView1<f64>, com: [f64; 3]) -> Array1<f64> {
    let mut out = q.to_owned();
    for i in 0..q.len() {
        out[i] -= sqrt_mass[i] * com[i % 3];
    }
    out
}

fn kabsch(
    source: &Array1<f64>,
    target: &Array1<f64>,
    atoms: usize,
) -> Result<[[f64; 3]; 3], SaddleError> {
    let mut cov = [[0.0; 3]; 3];
    for atom in 0..atoms {
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += source[3 * atom + i] * target[3 * atom + j];
            }
        }
    }
    let (r11, r22, r33) = (cov[0][0], cov[1][1], cov[2][2]);
    let (r12, r23, r31) = (cov[0][1], cov[1][2], cov[2][0]);
    let (r13, r21, r32) = (cov[0][2], cov[1][0], cov[2][1]);
    let f = [
        [r11 + r22 + r33, r23 - r32, r31 - r13, r12 - r21],
        [r23 - r32, r11 - r22 - r33, r12 + r21, r13 + r31],
        [r31 - r13, r12 + r21, -r11 + r22 - r33, r23 + r32],
        [r12 - r21, r13 + r31, r23 + r32, -r11 - r22 + r33],
    ];
    let (_evals, evecs) = jacobi4(f)?;
    let mut best = 0;
    for k in 1..4 {
        if evecs_value(&_evals, k) > evecs_value(&_evals, best) {
            best = k;
        }
    }
    let mut q = [
        evecs[0][best],
        evecs[1][best],
        evecs[2][best],
        evecs[3][best],
    ];
    if q[0] < 0.0 {
        for component in &mut q {
            *component = -*component;
        }
    }
    let (q0, q1, q2, q3) = (q[0], q[1], q[2], q[3]);
    Ok([
        [
            q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
        ],
    ])
}

fn evecs_value(evals: &[f64; 4], k: usize) -> f64 {
    evals[k]
}

/// Cyclic Jacobi. Stops when a sweep does not reduce the off-diagonal
/// sum of squares, which is the fixed point of the iteration.
fn jacobi4(mut a: [[f64; 4]; 4]) -> Result<([f64; 4], [[f64; 4]; 4]), SaddleError> {
    let mut v = [[0.0; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let mut off = off_diagonal(&a);
    loop {
        for p in 0..4 {
            for q in (p + 1)..4 {
                let apq = a[p][q];
                if apq == 0.0 || apq.is_nan() {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                let app = a[p][p];
                let aqq = a[q][q];
                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for i in 0..4 {
                    if i == p || i == q {
                        continue;
                    }
                    let aip = a[i][p];
                    let aiq = a[i][q];
                    a[i][p] = c * aip - s * aiq;
                    a[p][i] = a[i][p];
                    a[i][q] = s * aip + c * aiq;
                    a[q][i] = a[i][q];
                }
                for row in v.iter_mut() {
                    let vip = row[p];
                    let viq = row[q];
                    row[p] = c * vip - s * viq;
                    row[q] = s * vip + c * viq;
                }
            }
        }
        let next = off_diagonal(&a);
        if next >= off || next.is_nan() {
            break;
        }
        off = next;
    }
    let evals = [a[0][0], a[1][1], a[2][2], a[3][3]];
    if !evals.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("quaternion spectrum"));
    }
    Ok((evals, v))
}

fn off_diagonal(a: &[[f64; 4]; 4]) -> f64 {
    let mut sum = 0.0;
    for (p, row) in a.iter().enumerate() {
        for value in &row[p + 1..] {
            sum += value * value;
        }
    }
    sum
}

/// Orthonormal translations and rotations at one centred mass-weighted
/// geometry. A mode with no length (a linear molecule about its axis)
/// is absent.
pub(crate) fn rigid_directions(
    positions: ArrayView1<f64>,
    sqrt_mass: ArrayView1<f64>,
    atoms: usize,
) -> Vec<Array1<f64>> {
    let mut modes = Vec::with_capacity(6);
    for axis in 0..3 {
        let mut mode = Array1::zeros(positions.len());
        for atom in 0..atoms {
            mode[3 * atom + axis] = sqrt_mass[3 * atom];
        }
        push_mode(&mut modes, mode);
    }
    for axis in 0..3 {
        let mut mode = Array1::zeros(positions.len());
        for atom in 0..atoms {
            let p = [
                positions[3 * atom],
                positions[3 * atom + 1],
                positions[3 * atom + 2],
            ];
            // q = sqrt(m) x, so the mass-weighted rotation mode is ω × q.
            let crossed = cross(axis, p);
            for component in 0..3 {
                mode[3 * atom + component] = crossed[component];
            }
        }
        push_mode(&mut modes, mode);
    }
    modes
}

/// Remove translations and rotations about `positions` (mass-weighted,
/// centred) from `force`.
pub(crate) fn project_rigid_at(
    force: &mut Array1<f64>,
    positions: ArrayView1<f64>,
    sqrt_mass: ArrayView1<f64>,
    atoms: usize,
) {
    for mode in rigid_directions(positions, sqrt_mass, atoms) {
        let coeff = mode.dot(force);
        force.scaled_add(-coeff, &mode);
    }
}

fn push_mode(modes: &mut Vec<Array1<f64>>, mut mode: Array1<f64>) {
    for earlier in modes.iter() {
        let coeff = earlier.dot(&mode);
        mode.scaled_add(-coeff, earlier);
    }
    let norm = mode.dot(&mode).sqrt();
    if norm > 0.0 {
        mode /= norm;
        modes.push(mode);
    }
}

pub(crate) fn cross(axis: usize, p: [f64; 3]) -> [f64; 3] {
    match axis {
        0 => [0.0, -p[2], p[1]],
        1 => [p[2], 0.0, -p[0]],
        _ => [-p[1], p[0], 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn a_right_angle_returns_to_the_reference() {
        let sqrt_mass = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let reference = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        // 90 degrees about z, then a translation of (4, -2, 7).
        let moved = array![4.0, -1.0, 7.0, 3.0, -2.0, 7.0, 4.0, -2.0, 8.0];
        let images = ndarray::stack(ndarray::Axis(0), &[reference.view(), moved.view()]).unwrap();
        let (aligned, _) = align_images(images.view(), sqrt_mass.view(), 3).unwrap();
        let gap = &aligned.row(1) - &aligned.row(0);
        let scale = aligned.row(0).dot(&aligned.row(0)).sqrt();
        assert!(
            gap.dot(&gap).sqrt() <= scale * f64::EPSILON * 64.0,
            "{}",
            gap.dot(&gap).sqrt()
        );
    }
}
