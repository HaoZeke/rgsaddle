//! The sampled force and torque on a hyperplane against finite differences
//! of its free energy: a closed form on a Gaussian, and quadrature of the
//! line free energy on the LEPS plus harmonic oscillator surface.

use ndarray::{Array1, Array2, ArrayView1, array, s};
use rgsaddle::{Hyperplane, LangevinConfig, PointSurface, SaddleError, sample_plane};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/ohtst_surfaces.rs"
));

use ohtst_surfaces::{LepsHo, eigen2, hessian, line_free_energy, stationary};

/// `V = ½ qᵀ K q` with `K = Uᵀ diag(λ) U`.
struct Gaussian {
    k: Array2<f64>,
    k_inv: Array2<f64>,
}

impl Gaussian {
    fn new() -> Self {
        let rotate = |a: usize, b: usize, angle: f64| {
            let mut g = Array2::eye(3);
            g[(a, a)] = angle.cos();
            g[(b, b)] = angle.cos();
            g[(a, b)] = -angle.sin();
            g[(b, a)] = angle.sin();
            g
        };
        let u = rotate(0, 1, 0.4)
            .dot(&rotate(1, 2, -0.7))
            .dot(&rotate(0, 2, 0.25));
        let lambda = [2.0, 0.6, 1.3];
        let mut k = Array2::zeros((3, 3));
        let mut k_inv = Array2::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                for m in 0..3 {
                    k[(i, j)] += u[(m, i)] * lambda[m] * u[(m, j)];
                    k_inv[(i, j)] += u[(m, i)] * u[(m, j)] / lambda[m];
                }
            }
        }
        Self { k, k_inv }
    }

    /// `A(p, ν) = (kT/2) ln(νᵀK⁻¹ν) + (ν · p)² / (2 νᵀK⁻¹ν)` up to a
    /// constant: the marginal of `ν · q` is Gaussian with variance
    /// `kT νᵀK⁻¹ν`.
    fn free_energy(&self, p: &Array1<f64>, nu: &Array1<f64>, kt: f64) -> f64 {
        let nu = nu / nu.dot(nu).sqrt();
        let s = nu.dot(&self.k_inv.dot(&nu));
        let a = nu.dot(p);
        0.5 * kt * s.ln() + a * a / (2.0 * s)
    }
}

impl PointSurface for Gaussian {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let g = self.k.dot(&q);
        Ok((0.5 * q.dot(&g), g))
    }
}

/// Central difference of `A` along the packed direction `[δp; δν]`, the
/// normal renormalized.
fn packed_difference<F: Fn(&Array1<f64>, &Array1<f64>) -> f64>(
    free_energy: F,
    plane: &Hyperplane,
    direction: &Array1<f64>,
    h: f64,
) -> f64 {
    let n = plane.dim();
    let p = plane.point().to_owned();
    let nu = plane.normal().to_owned();
    let dp = direction.slice(s![..n]).to_owned();
    let dn = direction.slice(s![n..]).to_owned();
    let plus = free_energy(&(&p + &(&dp * h)), &(&nu + &(&dn * h)));
    let minus = free_energy(&(&p - &(&dp * h)), &(&nu - &(&dn * h)));
    (plus - minus) / (2.0 * h)
}

fn packed(dp: &Array1<f64>, dn: &Array1<f64>) -> Array1<f64> {
    let mut v = Array1::zeros(dp.len() + dn.len());
    v.slice_mut(s![..dp.len()]).assign(dp);
    v.slice_mut(s![dp.len()..]).assign(dn);
    v
}

#[test]
fn gaussian_force_and_torque_match_the_closed_form() {
    let surface = Gaussian::new();
    let kt = 0.5;
    let config = LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(0.4)
        .with_friction(0.5)
        .with_burn_in(200)
        .with_stride(4)
        .with_records(6000)
        .with_seed(11);
    let plane = Hyperplane::new(array![0.4, -0.2, 0.1], array![1.0, 0.5, -0.3]).unwrap();
    let sample = sample_plane(&surface, &plane, None, &config).unwrap();
    let nu = plane.normal().to_owned();
    let w1 = {
        let v = array![0.0, 1.0, 0.0];
        let v = &v - &(&nu * nu.dot(&v));
        &v / v.dot(&v).sqrt()
    };
    let w2 = array![
        nu[1] * w1[2] - nu[2] * w1[1],
        nu[2] * w1[0] - nu[0] * w1[2],
        nu[0] * w1[1] - nu[1] * w1[0]
    ];
    let zero = Array1::zeros(3);
    let directions = [
        packed(&nu, &zero),
        packed(&zero, &w1),
        packed(&zero, &w2),
        packed(&(&w1 * 0.7 + &nu * 0.3), &(&w2 * 0.5)),
    ];
    for direction in &directions {
        let exact = packed_difference(
            |p, n| surface.free_energy(p, n, kt),
            &plane,
            direction,
            1.0e-4,
        );
        let (value, error) = sample.directional(direction.view()).unwrap();
        assert!(
            (value - exact).abs() < 4.0 * error + 1.0e-7,
            "derivative {value} ± {error}, closed form {exact}"
        );
        assert!(error < 0.03 * (1.0 + exact.abs()));
    }
    let slide = packed(&w1, &zero);
    let (value, error) = sample.directional(slide.view()).unwrap();
    assert!(value.abs() < 1.0e-12 && error < 1.0e-12);
}

#[test]
fn leps_force_and_torque_match_line_quadrature() {
    let saddle = stationary(LepsHo::energy, [2.02, -0.17]);
    let minimum = ohtst_surfaces::descend(LepsHo::energy, [0.75, 1.3]);
    let barrier = LepsHo::energy(saddle[0], saddle[1]).0 - LepsHo::energy(minimum[0], minimum[1]).0;
    let kt = 0.1 * barrier;
    let (_, _, mode) = eigen2(hessian(LepsHo::energy, saddle));
    let base = mode[1].atan2(mode[0]);
    let config = LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(0.15)
        .with_friction(1.0)
        .with_burn_in(500)
        .with_stride(5)
        .with_records(20000)
        .with_seed(3);
    for (offset, tilt) in [(0.1, 0.25), (-0.15, -0.1)] {
        let angle = base + tilt;
        let normal = [angle.cos(), angle.sin()];
        let point = [
            saddle[0] + offset * normal[0],
            saddle[1] + offset * normal[1],
        ];
        let plane =
            Hyperplane::new(array![point[0], point[1]], array![normal[0], normal[1]]).unwrap();
        let sample = sample_plane(&LepsHo, &plane, None, &config).unwrap();
        let free_energy = |p: &Array1<f64>, n: &Array1<f64>| {
            line_free_energy(LepsHo::energy, [p[0], p[1]], [n[0], n[1]], kt, 0.0, 6.0)
        };
        let nu = array![normal[0], normal[1]];
        let turn = array![-normal[1], normal[0]];
        let zero = Array1::zeros(2);
        for direction in [packed(&nu, &zero), packed(&zero, &turn)] {
            let exact = packed_difference(free_energy, &plane, &direction, 1.0e-4);
            let (value, error) = sample.directional(direction.view()).unwrap();
            assert!(
                (value - exact).abs() < 4.0 * error + 1.0e-6,
                "derivative {value} ± {error}, quadrature {exact}"
            );
            assert!(error < 0.05 * (0.1 + exact.abs()));
        }
    }
}
