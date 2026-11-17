//! Force calls and wall time to converge a climbing-image band on a
//! Morse fcc(100) slab with its bottom layer fixed, with and without the
//! exponential metric, and with the min-mode refinement and the saddle
//! check off. One CSV row per method to stdout, and to the file named by
//! the first argument when given.
//!
//! `cargo run --release --example fixed_slab_precon -- benchmarks/fixed_slab_precon.csv`

use std::fmt::Write as _;
use std::time::Instant;

use rgsaddle::{BandConfig, BandSession, BandStatus, ExpBandPrecon};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/morse_slab.rs"
));

const N_IMAGES: usize = 7;
const FORCE_TOL: f64 = 1e-3;

fn main() {
    let band = slab_band(N_IMAGES);
    let rows = [
        ("fire", rgmin::Method::fire(rgmin::FireKind::V2)),
        ("lbfgs", rgmin::Method::lbfgs_with_memory(20)),
        ("ode12r", rgmin::Method::Ode12r),
    ];
    let mut csv = String::from(
        "method,precon,atoms,fixed,images,force_tol,converged,steps,force_calls,force_calls_per_image,wall_ms,saddle_energy\n",
    );
    for (name, method) in rows {
        for precon in [None, Some(ExpBandPrecon::default())] {
            let config = BandConfig::default()
                .with_cell(Some(slab_cell()))
                .with_fixed_atoms(slab_fixed())
                .with_force_tol(FORCE_TOL)
                .with_min_mode(None)
                .with_saddle_check(None)
                .with_method(method.clone())
                .with_exp_precon(precon);
            let surface = SlabSurface::new();
            let start = Instant::now();
            let mut session = BandSession::new(config, band.clone()).expect("session");
            let report = session.run(&surface, 5000).expect("run");
            let wall = start.elapsed().as_secs_f64() * 1e3;
            let ci = report.ci_index.unwrap_or(N_IMAGES / 2);
            let energy = slab_energy(session.positions().row(ci)).0;
            let calls = surface.rows();
            writeln!(
                csv,
                "{name},{},{},{},{N_IMAGES},{FORCE_TOL},{},{},{calls},{:.2},{wall:.1},{energy:.9}",
                if precon.is_some() { "exp" } else { "none" },
                slab_atoms(),
                slab_fixed().iter().filter(|&&f| f).count(),
                report.status == BandStatus::Converged,
                report.iteration,
                calls as f64 / (N_IMAGES - 2) as f64,
            )
            .expect("format");
        }
    }
    print!("{csv}");
    if let Some(path) = std::env::args().nth(1) {
        std::fs::write(&path, &csv).expect("write the CSV");
    }
}
