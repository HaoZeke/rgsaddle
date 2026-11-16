// A Morse fcc(100) slab with a frozen bottom layer and one adatom that
// hops between neighbouring hollow sites, shared by the fixed-atom tests
// and the `fixed_slab_precon` example.
//
// Four layers of 3 x 3 atoms, periodic in x and y with a square cell of
// 3 r0, open in z. The pair is Morse (epsilon = 1, A = alpha r0 = 4,
// r0 = 2.55 A, as in Makri, Ortner and Kermode 2019), force-shifted to
// zero energy and force at 3.8 A, below half the cell. The bottom layer
// is the frozen one; without it the slab floats along z.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

const SLAB_R0: f64 = 2.55;
const SLAB_ALPHA: f64 = 4.0 / SLAB_R0;
const SLAB_EPS: f64 = 1.0;
const SLAB_CUTOFF: f64 = 3.8;
const SLAB_SIDE: usize = 3;
const SLAB_LAYERS: usize = 4;

fn slab_box() -> f64 {
    SLAB_SIDE as f64 * SLAB_R0
}

/// The periodic cell of the slab: x and y wrap, z is open.
fn slab_cell() -> rgsaddle::Cell {
    let l = slab_box();
    rgsaddle::Cell([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, 0.0]])
}

fn slab_atoms() -> usize {
    SLAB_SIDE * SLAB_SIDE * SLAB_LAYERS + 1
}

/// True for the bottom layer, which is listed first.
fn slab_fixed() -> Vec<bool> {
    let bottom = SLAB_SIDE * SLAB_SIDE;
    (0..slab_atoms()).map(|a| a < bottom).collect()
}

fn morse_pair(r: f64) -> (f64, f64) {
    let e = (-SLAB_ALPHA * (r - SLAB_R0)).exp();
    let phi = SLAB_EPS * ((1.0 - e) * (1.0 - e) - 1.0);
    let dphi = 2.0 * SLAB_EPS * SLAB_ALPHA * e * (1.0 - e);
    (phi, dphi)
}

/// Energy and gradient of one image, every atom free.
fn slab_energy(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let l = slab_box();
    let (phi_c, dphi_c) = morse_pair(SLAB_CUTOFF);
    let mut energy = 0.0;
    let mut grad = Array1::zeros(x.len());
    for i in 0..n {
        for j in i + 1..n {
            let mut d = [0.0; 3];
            for k in 0..3 {
                d[k] = x[3 * j + k] - x[3 * i + k];
                if k < 2 {
                    d[k] -= l * (d[k] / l).round();
                }
            }
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if r >= SLAB_CUTOFF {
                continue;
            }
            let (phi, dphi) = morse_pair(r);
            energy += phi - phi_c - (r - SLAB_CUTOFF) * dphi_c;
            let de = dphi - dphi_c;
            for k in 0..3 {
                let g = de * d[k] / r;
                grad[3 * j + k] += g;
                grad[3 * i + k] -= g;
            }
        }
    }
    (energy, grad)
}

/// The adatom over the hollow at column `column` of the middle row.
fn slab_unrelaxed(column: usize) -> Array1<f64> {
    let h = SLAB_R0 / 2f64.sqrt();
    let mut x = Vec::with_capacity(3 * slab_atoms());
    for layer in 0..SLAB_LAYERS {
        let shift = (layer % 2) as f64 * SLAB_R0 / 2.0;
        for i in 0..SLAB_SIDE {
            for j in 0..SLAB_SIDE {
                x.push(i as f64 * SLAB_R0 + shift);
                x.push(j as f64 * SLAB_R0 + shift);
                x.push(layer as f64 * h);
            }
        }
    }
    let shift = (SLAB_LAYERS % 2) as f64 * SLAB_R0 / 2.0;
    x.push(column as f64 * SLAB_R0 + shift);
    x.push(SLAB_R0 + shift);
    x.push(SLAB_LAYERS as f64 * h);
    Array1::from_vec(x)
}

struct SlabMinimum {
    bounds: Bounds<f64>,
    fixed: Vec<bool>,
}

impl SlabMinimum {
    fn masked(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let (e, mut g) = slab_energy(x);
        for (k, gk) in g.iter_mut().enumerate() {
            if self.fixed[k / 3] {
                *gk = 0.0;
            }
        }
        (e, g)
    }
}

impl Objective<f64> for SlabMinimum {
    fn dim(&self) -> usize {
        3 * self.fixed.len()
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        slab_energy(x).0
    }
}

impl Gradient<f64> for SlabMinimum {
    fn dim(&self) -> usize {
        3 * self.fixed.len()
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.masked(x).1
    }
}

impl DifferentiableObjective<f64> for SlabMinimum {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.masked(x)
    }
}

/// L-BFGS on the free atoms to max |g| < 1e-8; the zeroed gradient
/// keeps the bottom layer where it started.
fn slab_relax(mut x: Array1<f64>) -> Array1<f64> {
    let dim = x.len();
    let obj = SlabMinimum {
        bounds: Bounds::new(
            Array1::from_elem(dim, -1e6),
            Array1::from_elem(dim, 1e6),
            0.0,
        ),
        fixed: slab_fixed(),
    };
    let control = rgmin::Control::default()
        .with_maxiter(1)
        .with_gtol(0.0)
        .with_istep(0.05);
    let mut solver = rgmin::Solver::new(rgmin::Method::lbfgs_with_memory(10), control, dim);
    for _ in 0..5000 {
        solver.step(&obj, &mut x).expect("slab relaxation step");
        let g = obj.masked(x.view()).1;
        if g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) < 1e-8 {
            return x;
        }
    }
    panic!("the slab endpoint did not relax");
}

/// Relaxed endpoints with the adatom one hollow apart, linearly
/// interpolated over `n_images`.
fn slab_band(n_images: usize) -> Array2<f64> {
    let reactant = slab_relax(slab_unrelaxed(1));
    let product = slab_relax(slab_unrelaxed(2));
    let mut band = Array2::zeros((n_images, reactant.len()));
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        band.row_mut(i)
            .assign(&(&reactant * (1.0 - t) + &product * t));
    }
    band
}

/// Every image of the band, counting rows.
struct SlabSurface {
    rows: std::sync::atomic::AtomicUsize,
}

impl SlabSurface {
    fn new() -> Self {
        Self {
            rows: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn rows(&self) -> usize {
        self.rows.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl rgsaddle::BandSurface for SlabSurface {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), rgsaddle::SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (e, g) = slab_energy(row);
            energies[i] = e;
            gradients.row_mut(i).assign(&g);
        }
        self.rows
            .fetch_add(positions.nrows(), std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}
