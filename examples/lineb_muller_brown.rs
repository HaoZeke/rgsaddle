//! Reproduce the ten-image tunneling path in Figures 2(b) and 3 of
//! Ásgeirsson et al., J. Chem. Phys. 148, 102334 (2018).

use ndarray::{Array1, Array2, ArrayView1};
use rgsaddle::{LinebConfig, LinebSession, LinebStatus, Masses, PointSurface, SaddleError};

struct ModifiedMullerBrown;

impl PointSurface for ModifiedMullerBrown {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        // Amplitude, centre and the three coefficients of the quadratic exponent.
        // The last three terms are the KNARR MullerBrownGaussWorker modifications.
        let terms = [
            [-0.200, 1.0, 0.0, -1.0, 0.0, -10.0],
            [-0.100, 0.0, 0.5, -1.0, 0.0, -10.0],
            [-0.170, -0.5, 1.5, -6.5, 11.0, -6.5],
            [0.015, -1.0, 1.0, 0.7, 0.6, 0.7],
            [-0.130, 1.0, 0.25, -12.5, 0.0, -12.5],
            [0.100, 0.6, 0.0, -12.5, 0.0, -12.5],
            [
                -0.065,
                0.8,
                0.4,
                -0.5 / 0.12_f64.powi(2),
                0.0,
                -0.5 / 0.16_f64.powi(2),
            ],
        ];
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(2);
        for [amplitude, x, y, a, b, c] in terms {
            let dx = q[0] - x;
            let dy = q[1] - y;
            let v = amplitude * (a * dx * dx + b * dx * dy + c * dy * dy).exp();
            energy += v;
            gradient[0] += v * (2.0 * a * dx + b * dy);
            gradient[1] += v * (b * dx + 2.0 * c * dy);
        }
        Ok((energy, gradient))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = [0.91155623, 0.18294581];
    let end = [-0.55822363, 1.44172584];
    // Turning energy inferred from both endpoints of Figure 2(b).
    let energy = -0.145;
    let images = Array2::from_shape_fn((10, 2), |(i, j)| {
        start[j] + (end[j] - start[j]) * i as f64 / 9.0
    });
    let masses = Masses::coordinates(Array1::ones(2).view())?;
    let mut solver = LinebSession::new(LinebConfig::new(1e-5, 1e-5), masses, images, energy)?;
    let report = solver.run(&ModifiedMullerBrown, 10000)?;
    if report.status != LinebStatus::Converged {
        return Err(format!(
            "LI-NEB residual {} after {} steps",
            report.opt_max, report.iteration
        )
        .into());
    }
    let path = solver.cartesian()?;
    let mut displacement = 0.0;
    println!("image,x,y,displacement,relative_energy");
    for (i, q) in path.outer_iter().enumerate() {
        if i > 0 {
            let delta = &q - &path.row(i - 1);
            displacement += delta.dot(&delta).sqrt();
        }
        let (v, _) = ModifiedMullerBrown.eval(q)?;
        println!(
            "{i},{:.8},{:.8},{displacement:.8},{:.8}",
            q[0],
            q[1],
            v - energy
        );
    }
    Ok(())
}
