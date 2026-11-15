//! The mode a converged search reports is the one a displacement
//! campaign pushes off along: saddle plus and minus a small step along
//! it, then two minimizations, one of which must return to the reactant
//! (Chill, Welborn, Terrell, Zhang, Berthet, Pedersen, Jónsson, and
//! Henkelman, Modelling Simul. Mater. Sci. Eng. 22, 055002 (2014),
//! doi:10.1088/0965-0393/22/5/055002). On a curved valley the lowest
//! mode turns between the reactant and the saddle, so a mode kept from
//! an early rotation points off the saddle's unstable direction.

use ndarray::{Array1, ArrayView1, array};
use rgmin::Method;
use rgsaddle::{
    MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface, SaddleError,
};

/// V = (x^2 - 1)^2 + 2 (y - x^2 / 2)^2 + 2 z^2. Minima at (+-1, 1/2, 0),
/// saddle at the origin with Hessian diag(-4, 4, 4). Near a minimum
/// the soft direction follows the parabola y = x^2 / 2, not the x axis.
struct CurvedValley;

impl CurvedValley {
    fn gradient(x: ArrayView1<f64>) -> Array1<f64> {
        let (a, b, c) = (x[0], x[1], x[2]);
        let u = b - 0.5 * a * a;
        array![4.0 * a * (a * a - 1.0) - 4.0 * a * u, 4.0 * u, 4.0 * c]
    }

    fn hessian(x: ArrayView1<f64>) -> [[f64; 3]; 3] {
        let (a, b) = (x[0], x[1]);
        let u = b - 0.5 * a * a;
        let xx = 12.0 * a * a - 4.0 - 4.0 * u + 4.0 * a * a;
        let xy = -4.0 * a;
        [[xx, xy, 0.0], [xy, 4.0, 0.0], [0.0, 0.0, 4.0]]
    }
}

impl PointSurface for CurvedValley {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (a, b, c) = (x[0], x[1], x[2]);
        let u = b - 0.5 * a * a;
        let e = (a * a - 1.0).powi(2) + 2.0 * u * u + 2.0 * c * c;
        Ok((e, Self::gradient(x)))
    }
}

/// Lowest eigenvector of the 2x2 xy block; z is stiff and decoupled.
fn lowest_xy(h: [[f64; 3]; 3]) -> (f64, [f64; 2]) {
    let (p, q, r) = (h[0][0], h[0][1], h[1][1]);
    let mean = 0.5 * (p + r);
    let lambda = mean - (0.25 * (p - r) * (p - r) + q * q).sqrt();
    let v = if q.abs() > 1e-14 {
        [q, lambda - p]
    } else if p <= r {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };
    let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
    (lambda, [v[0] / n, v[1] / n])
}

fn descend(mut x: Array1<f64>) -> Array1<f64> {
    for _ in 0..20000 {
        let g = CurvedValley::gradient(x.view());
        if g.dot(&g).sqrt() < 1e-8 {
            break;
        }
        x = &x - &(&g * 0.02);
    }
    x
}

fn pushes_off_into_both_basins(kind: MinModeKind, method: Method) {
    let reactant = array![1.0, 0.5, 0.0];
    let start = array![0.82, 0.36, 0.03];
    let seed = array![0.3, 0.9, 0.2];
    let (_, tilted) = lowest_xy(CurvedValley::hessian(start.view()));
    assert!(
        tilted[0].abs() < 0.9,
        "the start's soft mode is tilted: {tilted:?}"
    );
    let config = MinModeConfig {
        kind,
        method: method.clone(),
        force_tol: 1e-6,
        max_move: 0.1,
        ..MinModeConfig::default()
    };
    let mut session = MinModeSession::new(config, start, seed).unwrap();
    let report = session.run(&CurvedValley, 5000).unwrap();
    assert_eq!(
        report.status,
        MinModeStatus::Converged,
        "{kind:?} {method:?}"
    );
    let saddle = session.position().to_owned();
    assert!(
        saddle.dot(&saddle).sqrt() < 1e-4,
        "{kind:?} {method:?} saddle {saddle:?}"
    );

    let mode = session.mode().to_owned();
    assert!(
        (mode.dot(&mode).sqrt() - 1.0).abs() < 1e-12,
        "{kind:?} {method:?} |mode|"
    );
    let (lambda, exact) = lowest_xy(CurvedValley::hessian(saddle.view()));
    assert!(lambda < 0.0);
    let overlap = (mode[0] * exact[0] + mode[1] * exact[1]).abs();
    assert!(
        overlap > 0.999,
        "{kind:?} {method:?} overlap {overlap} mode {mode:?}"
    );
    assert!(
        (report.curvature - lambda).abs() < 0.05 * lambda.abs(),
        "{kind:?} {method:?}"
    );

    let plus = descend(&saddle + &(&mode * 0.2));
    let minus = descend(&saddle - &(&mode * 0.2));
    let home = |p: &Array1<f64>| (p - &reactant).dot(&(p - &reactant)).sqrt() < 1e-3;
    let other = array![-1.0, 0.5, 0.0];
    let away = |p: &Array1<f64>| (p - &other).dot(&(p - &other)).sqrt() < 1e-3;
    assert!(
        (home(&plus) && away(&minus)) || (home(&minus) && away(&plus)),
        "{kind:?} {method:?} endpoints {plus:?} {minus:?}"
    );
}

#[test]
fn dimer_fire_mode_pushes_off_into_both_basins() {
    pushes_off_into_both_basins(MinModeKind::Dimer, MinModeConfig::default().method);
}

#[test]
fn dimer_lbfgs_mode_pushes_off_into_both_basins() {
    pushes_off_into_both_basins(MinModeKind::Dimer, Method::lbfgs_with_memory(10));
}

#[test]
fn lanczos_lbfgs_mode_pushes_off_into_both_basins() {
    pushes_off_into_both_basins(MinModeKind::Lanczos, Method::lbfgs_with_memory(10));
}
