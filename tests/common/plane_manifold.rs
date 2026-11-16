// TEMPORARY TEST FIXTURE. Not rgsaddle API.
//
// A minimal stand-in for the product manifold R^n × S^{n-1} that rgmin is
// to provide (weighted product metric, projection, retraction, isometric
// transport). It exists only so the OH-TST optimizer, which takes any
// `rgmin::manifold::Manifold`, can be tested before that manifold lands.
// Points are packed `[p; ν]`. The metric is `w_p |δp|² + w_ν |δν|²`, so the
// Riemannian gradient is `(g_p / w_p, (I - ν νᵀ) g_ν / w_ν)`. The
// retraction normalizes `ν + δν` and transport projects onto the new
// tangent space.

#[allow(dead_code)]
mod plane_manifold {
    use ndarray::{Array1, s};
    use rgmin::manifold::Manifold;

    pub struct PointSphereFixture {
        pub n: usize,
        pub point_weight: f64,
        pub normal_weight: f64,
    }

    impl PointSphereFixture {
        pub fn new(n: usize) -> Self {
            Self {
                n,
                point_weight: 1.0,
                normal_weight: 1.0,
            }
        }

        pub fn weighted(n: usize, point_weight: f64, normal_weight: f64) -> Self {
            Self {
                n,
                point_weight,
                normal_weight,
            }
        }

        fn tangent(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
            let n = self.n;
            let nu = x.slice(s![n..]);
            let mut out = v.clone();
            let c = nu.dot(&v.slice(s![n..]));
            out.slice_mut(s![n..]).scaled_add(-c, &nu);
            out
        }
    }

    impl Manifold for PointSphereFixture {
        fn required_dim(&self, n: usize) -> Result<(), usize> {
            if n == 2 * self.n { Ok(()) } else { Err(2 * self.n) }
        }

        fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
            self.tangent(x, v)
        }

        fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
            let n = self.n;
            let mut g = self.tangent(x, egrad);
            g.slice_mut(s![..n]).mapv_inplace(|v| v / self.point_weight);
            g.slice_mut(s![n..]).mapv_inplace(|v| v / self.normal_weight);
            g
        }

        fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
            let n = self.n;
            let mut y = x + v;
            let norm = y.slice(s![n..]).dot(&y.slice(s![n..])).sqrt();
            y.slice_mut(s![n..]).mapv_inplace(|c| c / norm);
            y
        }

        fn transport(
            &self,
            _x_from: &Array1<f64>,
            x_to: &Array1<f64>,
            v: &Array1<f64>,
        ) -> Array1<f64> {
            self.tangent(x_to, v)
        }
    }
}
