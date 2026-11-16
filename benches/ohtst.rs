//! Wall time and force calls of one converged OH-TST rate on the LEPS plus
//! harmonic oscillator surface of Jóhannesson and Jónsson, J. Chem. Phys.
//! 115, 9644 (2001), Appendix B, at `kT = 0.1 E_A`: the plane optimizer
//! from the dimer's plane, the reactant normalization over planes parallel
//! to the guideline plane through the minimum, the reversible work from
//! that plane to the optimum, and the rate.
//! The dimer runs once, outside the timed loop. The throughput criterion
//! reports is force calls per second, and the force calls of one rate,
//! with the iterations of its plane optimization, are printed before the
//! run.

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ndarray::array;
use rgsaddle::{
    BasinConfig, Hyperplane, LangevinConfig, MinModeConfig, MinModeSession, MinModeStatus,
    OhTstRate, PlaneIntegrator, PlaneOptimizer, PlaneOptimizerConfig, PlaneOptimum,
    ReversibleWorkConfig, basin_density, ohtst_rate, reversible_work,
};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/ohtst_surfaces.rs"
));

use ohtst_surfaces::{LepsHo, descend, stationary};

struct Problem {
    kt: f64,
    seed: Hyperplane,
    start: Hyperplane,
    reactant: [f64; 2],
}

fn problem() -> Problem {
    let reactant = descend(LepsHo::energy, [0.75, 1.3]);
    let product = descend(LepsHo::energy, [3.0, -1.3]);
    let saddle = stationary(LepsHo::energy, [2.02, -0.17]);
    let barrier =
        LepsHo::energy(saddle[0], saddle[1]).0 - LepsHo::energy(reactant[0], reactant[1]).0;
    let config = MinModeConfig::default().with_force_tol(1.0e-8);
    let mut session = MinModeSession::new(config, array![1.9, -0.05], array![1.0, 0.2]).unwrap();
    while session.step(&LepsHo).unwrap().status != MinModeStatus::Converged {}
    let guideline = array![product[0] - reactant[0], product[1] - reactant[1]];
    Problem {
        kt: 0.1 * barrier,
        seed: Hyperplane::from_min_mode(&session).unwrap(),
        start: Hyperplane::new(array![reactant[0], reactant[1]], guideline).unwrap(),
        reactant,
    }
}

fn sampler(kt: f64, time_step: f64, records: usize) -> LangevinConfig {
    LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(time_step)
        .with_friction(1.0)
        .with_burn_in(300)
        .with_stride(5)
        .with_records(records)
        .with_seed(5)
}

/// The rate, the force calls it took, and the plane optimization.
fn converged_rate(problem: &Problem) -> (OhTstRate, usize, PlaneOptimum) {
    let kt = problem.kt;
    let config = PlaneOptimizerConfig::default()
        .with_sampler(sampler(kt, 0.3, 1000).with_integrator(PlaneIntegrator::Ghmc));
    let optimum = PlaneOptimizer::new(&problem.seed, config)
        .unwrap()
        .run(&LepsHo)
        .unwrap();
    let reactant = array![problem.reactant[0], problem.reactant[1]];
    let density = basin_density(
        &LepsHo,
        reactant.view(),
        problem.start.normal(),
        &BasinConfig::default().with_sampler(sampler(kt, 0.1, 4000)),
    )
    .unwrap();
    let config = ReversibleWorkConfig::default()
        .with_sampler(sampler(kt, 0.1, 2000))
        .with_nodes(17);
    let work = reversible_work(
        &LepsHo,
        &[density.plane.clone(), optimum.plane.clone()],
        Some(reactant.view()),
        &config,
    )
    .unwrap();
    let rate = ohtst_rate(&density, &work).unwrap();
    let calls = optimum.evaluations + density.evaluations + work.evaluations;
    (rate, calls, optimum)
}

fn leps_rate(c: &mut Criterion) {
    let problem = problem();
    let (rate, calls, optimum) = converged_rate(&problem);
    eprintln!(
        "one converged rate: {:.4e} ± {:.1e} with {calls} force calls",
        rate.rate, rate.error
    );
    eprintln!(
        "plane optimization: {} outer and {} inner iterations, {} force calls",
        optimum.iterations, optimum.inner_iterations, optimum.evaluations
    );
    let mut group = c.benchmark_group("ohtst_leps");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Elements(calls as u64));
    group.bench_function("converged_rate", |b| b.iter(|| converged_rate(&problem)));
    group.finish();
}

criterion_group!(benches, leps_rate);
criterion_main!(benches);
