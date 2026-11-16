//! Fixed atoms on a band (`BandConfig::with_fixed_atoms`): a Morse
//! fcc(100) slab with its bottom layer frozen and an adatom hopping
//! between hollow sites. The frozen layer never moves under any stepper,
//! the exponential metric reaches the same saddle as the plain band, and
//! an all-false mask is the band without a mask.

use rgsaddle::{
    BandConfig, BandRtr, BandSession, BandStatus, BandSurface, ExpBandPrecon, RtrConfig,
    SaddleError, band_forces,
};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/morse_slab.rs"
));

const N_IMAGES: usize = 7;

fn config() -> BandConfig {
    BandConfig::default()
        .with_cell(Some(slab_cell()))
        .with_fixed_atoms(slab_fixed())
}

fn exp() -> ExpBandPrecon {
    ExpBandPrecon::default()
}

fn frozen_bits(band: ArrayView2<f64>) -> Vec<u64> {
    let fixed = slab_fixed();
    band.outer_iter()
        .flat_map(|row| {
            row.iter()
                .enumerate()
                .filter(|(k, _)| fixed[k / 3])
                .map(|(_, v)| v.to_bits())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn free_max(gradient: ndarray::ArrayView1<f64>) -> f64 {
    let fixed = slab_fixed();
    gradient
        .iter()
        .enumerate()
        .filter(|(k, _)| !fixed[k / 3])
        .fold(0.0_f64, |m, (_, v)| m.max(v.abs()))
}

/// Climbing-image energy and the free-atom gradient there after a
/// converged run.
fn converge(config: BandConfig, rtr: bool, band: &Array2<f64>) -> (f64, f64, usize) {
    let surface = SlabSurface::new();
    let mut session = BandSession::new(config, band.clone()).expect("session");
    if rtr {
        session
            .set_rtr(Some(RtrConfig::default().with_radius_max(0.2)))
            .expect("RTR");
    }
    let report = session.run(&surface, 3000).expect("run");
    assert_eq!(
        report.status,
        BandStatus::Converged,
        "max force {} after {} steps",
        report.max_force,
        report.iteration
    );
    assert_eq!(
        frozen_bits(session.positions()),
        frozen_bits(band.view()),
        "a frozen atom moved"
    );
    let ci = session.climbing_image().expect("climbing image");
    let (energy, gradient) = slab_energy(session.positions().row(ci));
    (energy, free_max(gradient.view()), surface.rows())
}

#[test]
fn frozen_atoms_are_bitwise_unchanged_by_every_stepper() {
    let band = slab_band(N_IMAGES);
    let methods = [
        ("FIRE", rgmin::Method::fire(rgmin::FireKind::V2), false),
        ("L-BFGS", rgmin::Method::lbfgs_with_memory(20), false),
        ("quick-min", rgmin::Method::QuickMin, false),
        ("ODE12r", rgmin::Method::Ode12r, false),
        ("RTR", rgmin::Method::fire(rgmin::FireKind::V2), true),
    ];
    for (name, method, rtr) in methods {
        let (energy, residual, rows) = converge(config().with_method(method), rtr, &band);
        eprintln!("{name}: saddle energy {energy:.9}, free |g| {residual:.2e}, {rows} rows");
        assert!(residual <= 1e-3, "{name} climbing image is not stationary");
    }
}

#[test]
fn exp_precon_with_a_frozen_layer_reaches_the_plain_saddle() {
    let band = slab_band(N_IMAGES);
    let tol = 1e-4;
    let plain = config().with_force_tol(tol);
    // FIRE with the min-mode refinement settles this band on a second,
    // lower stationary point with a reshaped substrate, with or without
    // the mask; the plain reference is the climbing image alone.
    let (fire, ..) = converge(plain.clone().with_min_mode(None), false, &band);
    let (lbfgs, ..) = converge(
        plain
            .clone()
            .with_method(rgmin::Method::lbfgs_with_memory(20)),
        false,
        &band,
    );
    let (ode, ode_residual, _) = converge(
        plain
            .clone()
            .with_method(rgmin::Method::Ode12r)
            .with_exp_precon(Some(exp())),
        false,
        &band,
    );
    let (fire_exp, fire_residual, _) = converge(plain.with_exp_precon(Some(exp())), false, &band);
    eprintln!(
        "saddle energies: FIRE {fire:.10}, L-BFGS {lbfgs:.10}, ODE12r-Exp {ode:.10}, FIRE-Exp {fire_exp:.10}"
    );
    assert!(ode_residual <= tol && fire_residual <= tol);
    // A free gradient of at most tol puts the climbing image within
    // tol / |lambda| of the saddle and its energy within tol^2 / |lambda|
    // of the saddle energy; the softest mode here is well above 1e-2.
    for (name, energy) in [
        ("ODE12r-Exp", ode),
        ("FIRE-Exp", fire_exp),
        ("L-BFGS", lbfgs),
    ] {
        assert!(
            (energy - fire).abs() < 1e-6,
            "{name} saddle {energy} against FIRE {fire}"
        );
    }
}

fn steps(config: BandConfig, band: &Array2<f64>, n: usize) -> (Array2<f64>, Vec<u64>) {
    let surface = SlabSurface::new();
    let mut session = BandSession::new(config, band.clone()).expect("session");
    let mut trace = Vec::new();
    for _ in 0..n {
        let report = session.step(&surface).expect("step");
        trace.push(report.max_force.to_bits());
        trace.push(report.ci_index.unwrap_or(usize::MAX) as u64);
        trace.push(report.curvature.unwrap_or(0.0).to_bits());
    }
    (session.positions().to_owned(), trace)
}

#[test]
fn an_all_false_mask_is_bitwise_the_band_without_a_mask() {
    let band = slab_band(5);
    let none = BandConfig::default().with_cell(Some(slab_cell()));
    let clear = none.clone().with_fixed_atoms(vec![false; slab_atoms()]);
    for precon in [None, Some(exp())] {
        for method in [
            rgmin::Method::fire(rgmin::FireKind::V2),
            rgmin::Method::lbfgs_with_memory(20),
        ] {
            let a = steps(
                none.clone()
                    .with_method(method.clone())
                    .with_exp_precon(precon),
                &band,
                60,
            );
            let b = steps(
                clear
                    .clone()
                    .with_method(method)
                    .with_exp_precon(precon),
                &band,
                60,
            );
            assert_eq!(a.1, b.1, "reports differ");
            let bits = |m: &Array2<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a.0), bits(&b.0), "positions differ");
        }
    }
}

/// The slab with the frozen layer's gradient zeroed by the host, the
/// only option before the mask existed.
struct HostZeroed(SlabSurface);

impl BandSurface for HostZeroed {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.0.eval(positions, energies, gradients)?;
        let fixed = slab_fixed();
        for mut row in gradients.outer_iter_mut() {
            for (k, g) in row.iter_mut().enumerate() {
                if fixed[k / 3] {
                    *g = 0.0;
                }
            }
        }
        Ok(())
    }
}

#[test]
fn host_zeroed_forces_under_the_metric_move_frozen_atoms_and_the_mask_does_not() {
    let band = slab_band(5);
    let host = BandConfig::default()
        .with_cell(Some(slab_cell()))
        .with_remove_translation(false)
        .with_exp_precon(Some(exp()));
    let surface = HostZeroed(SlabSurface::new());
    let mut session = BandSession::new(host.clone(), band.clone()).expect("session");
    for _ in 0..5 {
        session.step(&surface).expect("step");
    }
    assert_ne!(
        frozen_bits(session.positions()),
        frozen_bits(band.view()),
        "the full-system metric should couple the frozen layer to the free atoms"
    );
    let mut session =
        BandSession::new(host.with_fixed_atoms(slab_fixed()), band.clone()).expect("session");
    for _ in 0..5 {
        session.step(&surface).expect("step");
    }
    assert_eq!(frozen_bits(session.positions()), frozen_bits(band.view()));
}

#[test]
fn frozen_atoms_carry_no_force_in_the_projection_or_the_report() {
    let mut band = slab_band(5);
    for mut row in band.outer_iter_mut() {
        row[0] += 0.1;
        row[4] -= 0.1;
    }
    let surface = SlabSurface::new();
    let forces = band_forces(&config(), &surface, band.view(), None).expect("forces");
    let dof = band.ncols();
    let fixed = slab_fixed();
    let mut free = 0.0_f64;
    for (k, f) in forces.force.iter().enumerate() {
        if fixed[(k % dof) / 3] {
            assert_eq!(*f, 0.0);
        } else {
            free = free.max(f.abs());
        }
    }
    assert_eq!(forces.max_force, free);
    let mut session = BandSession::new(config(), band.clone()).expect("session");
    let report = session.step(&surface).expect("step");
    let evaluation = session.evaluation();
    let first = evaluation.as_ref().map(|e| e.gradients.clone());
    let gradients = first.expect("evaluation after a step");
    for row in gradients.outer_iter() {
        for (k, g) in row.iter().enumerate() {
            if fixed[k / 3] {
                assert_eq!(*g, 0.0, "a frozen gradient in the evaluation");
            }
        }
    }
    let frozen_gradient = slab_energy(band.row(2))
        .1
        .iter()
        .enumerate()
        .filter(|(k, _)| fixed[k / 3])
        .fold(0.0_f64, |m, (_, v)| m.max(v.abs()));
    assert!(
        frozen_gradient > 1e-2,
        "the test needs a frozen layer under load: {frozen_gradient}"
    );
    assert!(report.max_force.is_finite());
}

#[test]
fn rtr_driver_keeps_the_frozen_layer() {
    let mut band = slab_band(5);
    let start = band.clone();
    let config = config();
    let surface = SlabSurface::new();
    let mut rtr = BandRtr::new(RtrConfig::default().with_radius_max(0.2), true);
    for _ in 0..50 {
        rtr.step(&config, &surface, &mut band).expect("step");
    }
    assert_eq!(frozen_bits(band.view()), frozen_bits(start.view()));
    assert_ne!(band, start);
}

#[test]
fn endpoints_that_differ_on_a_frozen_atom_are_refused() {
    let mut band = slab_band(5);
    let last = band.nrows() - 1;
    band[(last, 2)] += 1e-6;
    let error = BandSession::new(config(), band.clone())
        .err()
        .expect("refused");
    assert!(
        matches!(&error, SaddleError::Invalid(m) if m.contains("fixed atom 0")),
        "{error}"
    );
    let mut fine = band.clone();
    fine[(last, 2)] -= 1e-6;
    fine[(last, 0)] += slab_box();
    BandSession::new(config(), fine).expect("a whole period apart is the same place");
    let mut session = BandSession::new(
        BandConfig::default().with_cell(Some(slab_cell())),
        band.clone(),
    )
    .expect("no mask");
    assert!(matches!(
        session.set_fixed_atoms(slab_fixed()),
        Err(SaddleError::Invalid(_))
    ));
    assert!(session.fixed_atoms().is_empty());
}

#[test]
fn malformed_masks_are_refused() {
    let band = slab_band(5);
    let short = BandConfig::default()
        .with_cell(Some(slab_cell()))
        .with_fixed_atoms(vec![true; slab_atoms() - 1]);
    assert!(matches!(
        BandSession::new(short, band.clone()),
        Err(SaddleError::Shape(_))
    ));
    let all = BandConfig::default()
        .with_cell(Some(slab_cell()))
        .with_fixed_atoms(vec![true; slab_atoms()]);
    assert!(matches!(
        BandSession::new(all, band),
        Err(SaddleError::Invalid(_))
    ));
}

#[test]
fn setting_the_mask_mid_run_pins_the_interior_to_the_first_image() {
    let mut band = slab_band(5);
    let free = BandConfig::default().with_cell(Some(slab_cell()));
    for i in 1..band.nrows() - 1 {
        band[(i, 0)] += 0.01 * i as f64;
    }
    let surface = SlabSurface::new();
    let mut session = BandSession::new(free, band.clone()).expect("session");
    for _ in 0..3 {
        session.step(&surface).expect("step");
    }
    session.set_fixed_atoms(slab_fixed()).expect("mask");
    assert_eq!(session.fixed_atoms(), slab_fixed().as_slice());
    let first = band.row(0).to_owned();
    let pinned = |positions: ArrayView2<f64>| {
        let fixed = slab_fixed();
        positions.outer_iter().all(|row| {
            row.iter()
                .zip(first.iter())
                .enumerate()
                .all(|(k, (a, b))| !fixed[k / 3] || a.to_bits() == b.to_bits())
        })
    };
    assert!(pinned(session.positions()));
    for _ in 0..30 {
        session.step(&surface).expect("step");
    }
    assert!(pinned(session.positions()));
    session.set_fixed_atoms(Vec::new()).expect("clear");
    assert!(session.fixed_atoms().is_empty());
}
