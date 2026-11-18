#![cfg(feature = "pafi-repro")]

//! Ignored replay of the W-vacancy example bundled with the public PAFI
//! code. This is not either application in the 2018 PRL; see
//! `benchmarks/pafi-swinburne-repro/NOTES.md`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use capnp::message::Builder;
use capnp::serialize;
use ndarray::{Array1, Array2, s};
use rgpot_core::Potentials_capnp::potential_config;
use rgsaddle::{
    ArcPath, BOLTZMANN_EV_PER_K, BandConfig, BandSession, BandStatus, Cell, FreeEnergyProfile,
    PafiConfig, PafiIntegrator, PafiMeanForce, PathFrame, PointSurface, integrate_profile,
    pafi_profile,
};

mod rgpot_profile {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/common/rgpot_profile.rs"
    ));
}

use rgpot_profile::RgpotProfileSurface;

#[derive(Clone, Copy)]
struct Published {
    temperature: f64,
    barrier: f64,
    error: f64,
}

// `doc/test_output.png` in tomswinburne/pafi at commit 1661a1d, titled
// “Short (10 ps) test for vacancy in W, EAM potential.”
const PUBLISHED: [Published; 3] = [
    Published {
        temperature: 0.0,
        barrier: 1.97,
        error: 0.00,
    },
    Published {
        temperature: 1000.0,
        barrier: 2.07,
        error: 0.04,
    },
    Published {
        temperature: 2000.0,
        barrier: 1.82,
        error: 0.09,
    },
];

struct ReferenceSystem {
    surface: RgpotProfileSurface,
    path: ArcPath,
    neb_steps: usize,
    neb_force: f64,
}

fn read_lammps_data(path: &Path) -> Result<([f64; 3], Vec<f64>), String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut atoms = None;
    let mut lengths = [0.0; 3];
    let mut lines = text.lines();
    for line in lines.by_ref() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            [count, "atoms"] => atoms = count.parse::<usize>().ok(),
            [low, high, low_name, high_name]
                if low_name.ends_with("lo") && high_name.ends_with("hi") =>
            {
                let axis = match *low_name {
                    "xlo" => 0,
                    "ylo" => 1,
                    "zlo" => 2,
                    _ => return Err(format!("{}: box line {line}", path.display())),
                };
                let low: f64 = low.parse().map_err(|_| format!("box: {line}"))?;
                let high: f64 = high.parse().map_err(|_| format!("box: {line}"))?;
                if low != 0.0 {
                    return Err(format!("{}: box does not start at zero", path.display()));
                }
                lengths[axis] = high - low;
            }
            ["Atoms", ..] => break,
            _ => {}
        }
    }
    let atoms = atoms.ok_or_else(|| format!("{}: no atom count", path.display()))?;
    let mut positions = vec![f64::NAN; 3 * atoms];
    let mut seen = 0usize;
    for line in lines {
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.is_empty() {
            if seen == 0 {
                continue;
            }
            break;
        }
        if words.len() < 5 {
            return Err(format!("{}: atom line {line}", path.display()));
        }
        let id: usize = words[0].parse().map_err(|_| format!("atom id: {line}"))?;
        if id == 0 || id > atoms {
            return Err(format!("{}: atom id {id} out of range", path.display()));
        }
        for axis in 0..3 {
            positions[3 * (id - 1) + axis] = words[2 + axis]
                .parse()
                .map_err(|_| format!("atom position: {line}"))?;
        }
        seen += 1;
    }
    if seen != atoms || positions.iter().any(|value| !value.is_finite()) {
        return Err(format!("{}: read {seen} of {atoms} atoms", path.display()));
    }
    Ok((lengths, positions))
}

fn lammpc_config(potential: &Path) -> Vec<u8> {
    let potential = potential
        .to_str()
        .expect("the potential path must be UTF-8");
    assert!(
        !potential.chars().any(char::is_whitespace),
        "lammpc pair_coeff paths cannot contain whitespace"
    );
    let mut message = Builder::new_default();
    {
        let config = message.init_root::<potential_config::Builder>();
        let mut lammps = config.init_lammps();
        lammps.set_units_style("metal");
        lammps.set_atom_style("atomic");
        lammps.set_pair_style("eam/fs");
        let coefficient = format!("* * {potential} W");
        lammps
            .reborrow()
            .init_pair_coeffs(1)
            .set(0, coefficient.as_str());
        lammps.reborrow().init_type_to_atomic_number(1).set(0, 74);
        lammps.set_newton_pair(true);
        lammps.set_boundary("p p p");
    }
    serialize::write_message_to_words(&message)
}

fn load_system(root: &Path, library: &Path) -> ReferenceSystem {
    let directory = root.join("EAM-VAC-W");
    let mut lengths = None;
    let mut frames = Vec::new();
    for index in 0..=8 {
        let image = directory.join(format!("image_{index}.dat"));
        let (box_lengths, positions) = read_lammps_data(&image).expect("reference image");
        match lengths {
            Some(known) => assert_eq!(known, box_lengths, "image box"),
            None => lengths = Some(box_lengths),
        }
        frames.push(positions);
    }
    let lengths = lengths.expect("image box");
    for atom in frames[0].chunks_mut(3) {
        for (value, length) in atom.iter_mut().zip(lengths) {
            *value = value.rem_euclid(length);
        }
    }
    let first = frames[0].clone();
    for frame in frames.iter_mut().skip(1) {
        for (atom, origin) in frame.chunks_mut(3).zip(first.chunks(3)) {
            for axis in 0..3 {
                let mut step = atom[axis] - origin[axis];
                step -= lengths[axis] * (step / lengths[axis]).round();
                atom[axis] = origin[axis] + step;
            }
        }
    }
    let atoms = first.len() / 3;
    let surface = RgpotProfileSurface::new(
        "lammpc",
        library,
        lammpc_config(&directory.join("W.eam.fs")),
        vec![74; atoms],
        [
            lengths[0], 0.0, 0.0, 0.0, lengths[1], 0.0, 0.0, 0.0, lengths[2],
        ],
    );
    let mut band = Array2::zeros((frames.len(), first.len()));
    for (mut row, frame) in band.outer_iter_mut().zip(&frames) {
        row.assign(&Array1::from_vec(frame.clone()));
    }
    let config = BandConfig::default()
        .with_cell(Some(Cell([
            [lengths[0], 0.0, 0.0],
            [0.0, lengths[1], 0.0],
            [0.0, 0.0, lengths[2]],
        ])))
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None);
    let mut neb = BandSession::new(config, band).expect("rgsaddle NEB");
    let limit = count_from_env("PAFI_NEB_STEPS", 2_000);
    eprintln!("starting rgsaddle NEB with at most {limit} steps");
    let mut last = None;
    for _ in 0..limit {
        let report = neb.step(&surface).expect("rgsaddle NEB step");
        let converged = report.status == BandStatus::Converged;
        last = Some(report);
        if converged {
            break;
        }
    }
    let report = last.expect("at least one NEB step");
    assert_eq!(
        report.status,
        BandStatus::Converged,
        "rgsaddle NEB did not converge in {limit} steps; max force {}",
        report.max_force
    );
    eprintln!(
        "rgsaddle NEB converged in {} steps at max force {} eV/A",
        report.iteration, report.max_force
    );
    let frames: Vec<Vec<f64>> = neb
        .positions()
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    let distance: Vec<f64> = frames
        .iter()
        .map(|frame| {
            frame
                .iter()
                .zip(&first)
                .map(|(value, origin)| (value - origin).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let total = *distance.last().expect("last image");
    let path = ArcPath::new(
        frames
            .into_iter()
            .zip(distance)
            .map(|(positions, distance)| PathFrame {
                positions: Array1::from_vec(positions),
                arc: distance / total,
            })
            .collect(),
    )
    .expect("reference path");
    ReferenceSystem {
        surface,
        path,
        neb_steps: report.iteration,
        neb_force: report.max_force,
    }
}

fn planes(count: usize) -> Vec<f64> {
    (0..count)
        .map(|index| index as f64 / (count - 1) as f64)
        .collect()
}

fn zero_temperature_profile<S: PointSurface>(
    surface: &S,
    path: &ArcPath,
    arcs: &[f64],
) -> FreeEnergyProfile {
    let gradient: Vec<f64> = arcs
        .iter()
        .map(|arc| {
            let geometry = path.geometry(*arc).expect("zero-temperature plane");
            let (_, gradient) = surface
                .eval(geometry.positions.view())
                .expect("zero-temperature force");
            gradient.dot(&geometry.tangent)
        })
        .collect();
    integrate_profile(arcs, &gradient, &vec![0.0; arcs.len()]).expect("zero-temperature profile")
}

fn remeshed_barrier(profile: &FreeEnergyProfile) -> (f64, f64) {
    let count = profile.arc.len();
    let basis = ArcPath::new(
        profile
            .arc
            .iter()
            .enumerate()
            .map(|(index, arc)| {
                let mut positions = Array1::zeros(count + 1);
                positions[0] = *arc;
                positions[index + 1] = 1.0;
                PathFrame {
                    positions,
                    arc: *arc,
                }
            })
            .collect(),
    )
    .expect("remesh basis");
    let dense = 10 * count;
    let points: Vec<f64> = (0..dense)
        .map(|index| index as f64 / (dense - 1) as f64)
        .collect();
    let weights: Vec<Array1<f64>> = points
        .iter()
        .map(|point| {
            basis
                .geometry(*point)
                .expect("remesh point")
                .positions
                .slice(s![1..])
                .to_owned()
        })
        .collect();
    let mut cumulative = vec![Array1::<f64>::zeros(count)];
    for index in 1..dense {
        let width = 0.5 * (points[index] - points[index - 1]);
        cumulative
            .push(&cumulative[index - 1] + &((&weights[index - 1] + &weights[index]) * width));
    }
    let gradient = Array1::from_vec(profile.gradient.clone());
    let (top, barrier) = cumulative
        .iter()
        .map(|weight| weight.dot(&gradient))
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        });
    let error = cumulative[top]
        .iter()
        .zip(&profile.gradient_error)
        .map(|(weight, plane_error)| (weight * plane_error).powi(2))
        .sum::<f64>()
        .sqrt();
    (barrier, error)
}

fn count_from_env(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(value) => match value.parse() {
            Ok(parsed) => parsed,
            Err(error) => panic!("{name}={value:?} is not a positive integer: {error}"),
        },
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => panic!("{name}: {error}"),
    }
}

#[test]
#[ignore = "full W EAM replay; needs RGSADDLE_PAFI_REFERENCE and LAMMPC_LIBRARY"]
fn w_vacancy_matches_public_code_output() {
    let root = PathBuf::from(
        std::env::var_os("RGSADDLE_PAFI_REFERENCE")
            .expect("RGSADDLE_PAFI_REFERENCE must name examples/systems"),
    );
    let library = PathBuf::from(
        std::env::var_os("LAMMPC_LIBRARY").expect("LAMMPC_LIBRARY must name liblammpc"),
    );
    let system = load_system(&root, &library);
    let trajectories = count_from_env("PAFI_TRAJECTORIES", 100);
    assert!(
        trajectories > 1,
        "ensemble errors need independent trajectories"
    );
    let thermalize = count_from_env("PAFI_THERMALIZE", 2_000);
    let samples = count_from_env("PAFI_SAMPLES", 5_000);
    // The public example's default ReactionCoordinate axis has nine planes.
    // The PRL applications initially used ten; they are separate unavailable
    // cases and are not the source of these three values.
    let arcs = planes(9);
    eprintln!("evaluating the zero-temperature profile");
    let zero = zero_temperature_profile(&system.surface, &system.path, &arcs);
    let mut rows = String::from(
        "temperature_k,published_ev,published_error_ev,ours_ev,ours_error_ev,planes,trajectories,thermalize_steps,sample_steps,integrator,mean_force,neb_steps,neb_max_force_ev_per_a,rejected_stretches\n",
    );
    let mut rejection_rows = String::from("temperature_k,plane,trajectory,stretch,max_jump\n");
    let (zero_barrier, zero_error) = remeshed_barrier(&zero);
    let mut observed = vec![(0.0, zero_barrier, zero_error, 0usize)];
    // Use the public input's numerical step. The mobility fixes this
    // overdamped implementation's time scale; the public example used an
    // underdamped thermostat, but both target the same constrained ensemble.
    let time_step = 0.001;
    for temperature in [1000.0, 2000.0] {
        eprintln!("sampling the {temperature} K profile");
        let config = PafiConfig::default()
            .with_thermal(BOLTZMANN_EV_PER_K * temperature)
            .with_mobility(1.0)
            .with_time_step(time_step)
            .with_thermalize(thermalize)
            .with_stride(1)
            .with_samples(samples)
            .with_blocks(5)
            .with_trajectories(trajectories)
            .with_integrator(PafiIntegrator::EulerMaruyama)
            .with_mean_force(PafiMeanForce::ReferenceCode)
            .with_seed(10_000 + temperature as u64)
            .with_retain(0)
            .with_max_jump(Some(0.4));
        let (profile, averages) =
            pafi_profile(&system.surface, &system.path, &arcs, &config).expect("PAFI profile");
        let (barrier, error) = remeshed_barrier(&profile);
        for average in &averages {
            for rejection in &average.rejections {
                writeln!(
                    rejection_rows,
                    "{temperature:.0},{:.17},{},{},{:.17}",
                    rejection.plane, rejection.trajectory, rejection.stretch, rejection.max_jump
                )
                .expect("rejection CSV row");
            }
        }
        let rejected = averages.iter().map(|average| average.rejected).sum();
        observed.push((temperature, barrier, error, rejected));
    }
    // The image reports two decimals, so half of one 0.01 eV display unit
    // is an independent uniform-rounding bound.
    let rounding_error: f64 = 0.01 / 2.0;
    let sigma_coverage = 2.0;
    for (published, (temperature, barrier, error, rejected)) in PUBLISHED.iter().zip(&observed) {
        writeln!(
            rows,
            "{temperature:.0},{:.2},{:.2},{barrier:.8},{error:.8},{},{trajectories},{thermalize},{samples},euler_maruyama,reference_code,{},{},{rejected}",
            published.barrier,
            published.error,
            arcs.len(),
            system.neb_steps,
            system.neb_force
        )
        .expect("CSV row");
    }
    print!("{rows}");
    print!("{rejection_rows}");
    if let Some(path) = std::env::var_os("PAFI_REPRO_OUTPUT") {
        std::fs::write(path, &rows).expect("write PAFI_REPRO_OUTPUT");
    }
    if let Some(path) = std::env::var_os("PAFI_REPRO_REJECTIONS_OUTPUT") {
        std::fs::write(path, &rejection_rows).expect("write PAFI_REPRO_REJECTIONS_OUTPUT");
    }
    for (published, (temperature, barrier, error, _)) in PUBLISHED.iter().zip(observed) {
        assert_eq!(temperature, published.temperature);
        let combined = (published.error.powi(2) + error.powi(2) + rounding_error.powi(2)).sqrt();
        assert!(
            (barrier - published.barrier).abs() <= sigma_coverage * combined,
            "{temperature} K: {barrier} ± {error} eV against {} ± {} eV",
            published.barrier,
            published.error
        );
    }
}
