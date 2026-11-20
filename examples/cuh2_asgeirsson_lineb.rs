//! H₂ / Cu(110) LI-NEB + instanton harness (Ásgeirsson 2018 method).
//!
//! Six H degrees of freedom over a fixed 216-atom Cu(110) slab on the
//! canonical rgpot CuH2 EAM. Builds a classical MEP, a fixed-energy
//! ladder, and compares direct vs interpolated-Hessian log flux, with
//! every surface evaluation counted.
//!
//! Documentation and fixture provenance:
//! `benchmarks/cuh2_asgeirsson_lineb/NOTES.md`.
//!
//! ```text
//! cargo run --features cuh2-lineb --example cuh2_asgeirsson_lineb -- fixtures
//! cargo run --release --features cuh2-lineb --example cuh2_asgeirsson_lineb -- \
//!   --lib "$RGPOT_CUH2_LIB" --all --out target/cuh2_asgeirsson
//! ```

use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use libloading::{Library, Symbol};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, CiConfig, Continuation, LinebConfig, Masses,
    PointSurface, SaddleError, continue_from_mep, hqtst_rate, hqtst_rate_interpolated_hessian,
};

/// Paper / howto tolerance: 1e-2 per Å for H → 6.44e-4 in eV·Å·amu.
const OPT_TOL: f64 = 6.44e-4;
/// ℏ such that `1e-2 * ℏ / sqrt(m_H) ≈ 6.44e-4` with `m_H = 1.00793`.
const HBAR: f64 = 0.06466;
const BOLTZMANN: f64 = 8.617_333_262_145e-5;
const H_MASS: f64 = 1.007_93;
const MEP_IMAGES: usize = 15;
const LADDER_DELTA: f64 = 0.05;
const LADDER_IMAGES: usize = 10;
const LADDER_LEVELS: usize = 9;
const RATE_BEADS: usize = 32;
/// Expected direct-rate force calls: `(P/2+1) * (1 + 2 * dof)`.
const EXPECTED_RATE_CALLS: usize = (RATE_BEADS / 2 + 1) * (1 + 2 * 6);

type EnsureParamsFn = unsafe extern "C" fn();
type ForceFn =
    unsafe extern "C" fn(i32, *const f64, *const i32, *const f64, *mut f64, *mut f64) -> i32;

struct Args {
    lib: Option<PathBuf>,
    out: PathBuf,
    fixtures_only: bool,
    smoke: bool,
    mep: bool,
    ladder: bool,
    rates: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut lib = env::var_os("RGPOT_CUH2_LIB").map(PathBuf::from);
    let mut out = PathBuf::from("cuh2_asgeirsson_out");
    let mut fixtures_only = false;
    let mut smoke = false;
    let mut mep = false;
    let mut ladder = false;
    let mut rates = false;
    let mut all = false;
    let mut argv = env::args().skip(1);
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "fixtures" | "--fixtures" => fixtures_only = true,
            "--smoke" => smoke = true,
            "--mep" => mep = true,
            "--ladder" => ladder = true,
            "--rates" => rates = true,
            "--all" => all = true,
            "--lib" => {
                lib = Some(PathBuf::from(
                    argv.next().ok_or("--lib needs a path to librgpot")?,
                ));
            }
            "--out" => {
                out = PathBuf::from(argv.next().ok_or("--out needs a directory")?);
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: cuh2_asgeirsson_lineb [fixtures|--smoke|--mep|--ladder|--rates|--all] \
                     [--lib PATH] [--out DIR]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if all {
        smoke = true;
        mep = true;
        ladder = true;
        rates = true;
    }
    if !fixtures_only && !smoke && !mep && !ladder && !rates {
        fixtures_only = true;
    }
    Ok(Args {
        lib,
        out,
        fixtures_only,
        smoke,
        mep,
        ladder,
        rates,
    })
}

fn bench_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/cuh2_asgeirsson_lineb")
}

fn fixtures_dir() -> PathBuf {
    bench_root().join("fixtures")
}

#[derive(Clone, Debug)]
struct ConSystem {
    cell: [[f64; 3]; 3],
    species: Vec<String>,
    positions: Array2<f64>,
    fixed: Vec<bool>,
    masses: Vec<f64>,
}

fn parse_con(path: &Path) -> Result<ConSystem, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 10 {
        return Err(format!("{}: too short for a CON file", path.display()));
    }
    let box_abc: Vec<f64> = lines[2]
        .split_whitespace()
        .take(3)
        .map(|s| s.parse().map_err(|e| format!("box: {e}")))
        .collect::<Result<_, _>>()?;
    let ntypes: usize = lines[6]
        .split_whitespace()
        .next()
        .ok_or("missing ntypes")?
        .parse()
        .map_err(|e| format!("ntypes: {e}"))?;
    let counts: Vec<usize> = lines[7]
        .split_whitespace()
        .take(ntypes)
        .map(|s| s.parse().map_err(|e| format!("counts: {e}")))
        .collect::<Result<_, _>>()?;
    let type_masses: Vec<f64> = lines[8]
        .split_whitespace()
        .take(ntypes)
        .map(|s| s.parse().map_err(|e| format!("masses: {e}")))
        .collect::<Result<_, _>>()?;
    let mut idx = 9;
    let mut species = Vec::new();
    let mut coords = Vec::new();
    let mut fixed = Vec::new();
    let mut masses = Vec::new();
    for (t, &count) in counts.iter().enumerate() {
        let name = lines
            .get(idx)
            .ok_or("truncated species name")?
            .trim()
            .to_string();
        idx += 2; // name + "Components of Type …"
        for _ in 0..count {
            let parts: Vec<&str> = lines
                .get(idx)
                .ok_or("truncated coordinate row")?
                .split_whitespace()
                .collect();
            if parts.len() < 4 {
                return Err(format!("bad coordinate row at line {idx}"));
            }
            coords.extend([
                parts[0].parse::<f64>().map_err(|e| e.to_string())?,
                parts[1].parse::<f64>().map_err(|e| e.to_string())?,
                parts[2].parse::<f64>().map_err(|e| e.to_string())?,
            ]);
            // eOn CON: column 4 is isFixed (1 = fixed, 0 = free).
            fixed.push(parts[3].parse::<f64>().map_err(|e| e.to_string())? > 0.5);
            species.push(name.clone());
            masses.push(type_masses[t]);
            idx += 1;
        }
    }
    let n = species.len();
    Ok(ConSystem {
        cell: [
            [box_abc[0], 0.0, 0.0],
            [0.0, box_abc[1], 0.0],
            [0.0, 0.0, box_abc[2]],
        ],
        species,
        positions: Array2::from_shape_vec((n, 3), coords).map_err(|e| e.to_string())?,
        fixed,
        masses,
    })
}

fn atomic_number(name: &str) -> Result<i32, String> {
    match name {
        "Cu" | "cu" => Ok(29),
        "H" | "h" => Ok(1),
        other => Err(format!("unsupported species {other}")),
    }
}

fn hydrogen_indices(system: &ConSystem) -> Result<[usize; 2], String> {
    let hs: Vec<usize> = system
        .species
        .iter()
        .enumerate()
        .filter_map(|(i, s)| (s == "H").then_some(i))
        .collect();
    if hs.len() != 2 {
        return Err(format!("expected 2 H atoms, found {}", hs.len()));
    }
    Ok([hs[0], hs[1]])
}

fn geometry_report(label: &str, system: &ConSystem) -> Result<(), String> {
    let [i, j] = hydrogen_indices(system)?;
    let h0 = system.positions.row(i);
    let h1 = system.positions.row(j);
    let hh = (h0[0] - h1[0]).hypot(h0[1] - h1[1]).hypot(h0[2] - h1[2]);
    let cu_z_max = system
        .positions
        .outer_iter()
        .zip(system.species.iter())
        .filter_map(|(row, s)| (s.as_str() == "Cu").then_some(row[2]))
        .fold(f64::NEG_INFINITY, f64::max);
    let mid_z = 0.5 * (h0[2] + h1[2]);
    let free_h = (!system.fixed[i]) as usize + (!system.fixed[j]) as usize;
    let fixed_cu = system
        .species
        .iter()
        .zip(system.fixed.iter())
        .filter(|(s, f)| s.as_str() == "Cu" && **f)
        .count();
    println!(
        "{label}: natoms={} HH={hh:.6} height_mid={:.6} free_H={free_h} fixed_Cu={fixed_cu} \
         H=({:.6},{:.6},{:.6}) ({:.6},{:.6},{:.6})",
        system.positions.nrows(),
        mid_z - cu_z_max,
        h0[0],
        h0[1],
        h0[2],
        h1[0],
        h1[1],
        h1[2]
    );
    Ok(())
}

fn run_fixtures_check() -> Result<(), String> {
    let dir = fixtures_dir();
    println!("fixtures directory: {}", dir.display());
    let provenance = dir.join("provenance.json");
    if !provenance.is_file() {
        return Err(format!("missing {}", provenance.display()));
    }
    println!(
        "provenance: {}",
        fs::read_to_string(&provenance).map_err(|e| e.to_string())?
    );
    let adsorbed = parse_con(&dir.join("eon_adsorbed_min2.con"))?;
    let desorbed = parse_con(&dir.join("eon_desorbed_min1.con"))?;
    let rgpot = parse_con(&dir.join("rgpot_tmp.con"))?;
    let fort = parse_con(&dir.join("fortran_cuh2_tmp.con"))?;
    geometry_report("eon adsorbed (reactant)", &adsorbed)?;
    geometry_report("eon desorbed (product)", &desorbed)?;
    geometry_report("rgpot tmp.con (near-TS fixture)", &rgpot)?;
    geometry_report("fortran_cuh2 tmp.con", &fort)?;

    let cu_a = adsorbed.positions.slice(ndarray::s![..216, ..]);
    let cu_b = desorbed.positions.slice(ndarray::s![..216, ..]);
    let cu_r = rgpot.positions.slice(ndarray::s![..216, ..]);
    let rms = |a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>| {
        let mut s = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = x - y;
            s += d * d;
        }
        (s / a.len() as f64).sqrt()
    };
    println!(
        "Cu lattice RMS eon adsorbed vs desorbed: {:.3e} (must be ~0)",
        rms(cu_a, cu_b)
    );
    println!(
        "Cu lattice RMS eon adsorbed vs rgpot_tmp: {:.3e}",
        rms(cu_a, cu_r)
    );
    println!(
        "published refs: {}",
        bench_root().join("published_refs.csv").display()
    );
    println!(
        "thesis adsorbed HH=3.05 Å, saddle HH=1.34 Å, barrier 0.545 eV; \
         howto rgpot barrier 0.469 eV; paper ~900 flux evals/T — see NOTES.md"
    );
    Ok(())
}

/// CuH2 through `rgpot_cuh2_force` (process-serial Fortran entry).
struct CuH2Lib {
    _library: Library,
    force: ForceFn,
}

impl CuH2Lib {
    unsafe fn open(path: &Path) -> Result<Self, String> {
        let library =
            unsafe { Library::new(path) }.map_err(|e| format!("load {}: {e}", path.display()))?;
        // The workspace-capable ABI exposes an explicit initializer.
        // The serial force ABI also supports plugins with internal initialization.
        if let Ok(ensure) = unsafe { library.get::<EnsureParamsFn>(b"rgpot_cuh2_ensure_params") } {
            unsafe { ensure() };
        }
        let force: Symbol<ForceFn> = unsafe { library.get(b"rgpot_cuh2_force") }
            .map_err(|e| format!("rgpot_cuh2_force: {e}"))?;
        let force = *force;
        Ok(Self {
            _library: library,
            force,
        })
    }

    fn eval(
        &self,
        positions: &[f64],
        atomic_numbers: &[i32],
        cell: &[f64; 9],
    ) -> Result<(f64, Vec<f64>), String> {
        let natoms = atomic_numbers.len();
        if positions.len() != natoms * 3 {
            return Err("position length mismatch".into());
        }
        let mut forces = vec![0.0; natoms * 3];
        let mut energy = 0.0;
        unsafe {
            let status = (self.force)(
                natoms as i32,
                positions.as_ptr(),
                atomic_numbers.as_ptr(),
                cell.as_ptr(),
                forces.as_mut_ptr(),
                &mut energy,
            );
            if status != 0 {
                return Err(format!("rgpot_cuh2_force status {status}"));
            }
        }
        Ok((energy, forces))
    }
}

/// 6-DOF H surface: Cu frozen at the CON coordinates, H from `q`.
struct FixedCuH2<'a> {
    lib: &'a CuH2Lib,
    copper: Array2<f64>,
    atomic_numbers: Vec<i32>,
    cell_flat: [f64; 9],
    calls: AtomicUsize,
    lock: Mutex<()>,
}

impl<'a> FixedCuH2<'a> {
    fn new(lib: &'a CuH2Lib, slab: &ConSystem) -> Result<Self, String> {
        if slab.positions.nrows() != 218 {
            return Err(format!(
                "expected 218 atoms (216 Cu + 2 H), got {}",
                slab.positions.nrows()
            ));
        }
        let copper = slab.positions.slice(ndarray::s![..216, ..]).to_owned();
        let mut atomic_numbers = Vec::with_capacity(218);
        for name in &slab.species {
            atomic_numbers.push(atomic_number(name)?);
        }
        let cell_flat = [
            slab.cell[0][0],
            slab.cell[0][1],
            slab.cell[0][2],
            slab.cell[1][0],
            slab.cell[1][1],
            slab.cell[1][2],
            slab.cell[2][0],
            slab.cell[2][1],
            slab.cell[2][2],
        ];
        Ok(Self {
            lib,
            copper,
            atomic_numbers,
            cell_flat,
            calls: AtomicUsize::new(0),
            lock: Mutex::new(()),
        })
    }

    fn full_positions(&self, h: ArrayView1<f64>) -> Result<Vec<f64>, SaddleError> {
        if h.len() != 6 {
            return Err(SaddleError::Solver(
                "H coordinate vector must have length 6".into(),
            ));
        }
        let mut positions = Vec::with_capacity(218 * 3);
        for row in self.copper.outer_iter() {
            positions.extend_from_slice(row.as_slice().unwrap());
        }
        positions.extend_from_slice(h.as_slice().unwrap());
        Ok(positions)
    }

    fn eval_h(&self, h: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let positions = self.full_positions(h)?;
        let _guard = self.lock.lock().unwrap();
        self.calls.fetch_add(1, Ordering::Relaxed);
        let (energy, forces) = self
            .lib
            .eval(&positions, &self.atomic_numbers, &self.cell_flat)
            .map_err(|e| SaddleError::Surface(e))?;
        // Potential returns forces; PointSurface wants gradients = −F.
        let g = array![
            -forces[216 * 3],
            -forces[216 * 3 + 1],
            -forces[216 * 3 + 2],
            -forces[217 * 3],
            -forces[217 * 3 + 1],
            -forces[217 * 3 + 2],
        ];
        Ok((energy, g))
    }

    fn take_calls(&self) -> usize {
        self.calls.swap(0, Ordering::Relaxed)
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl PointSurface for FixedCuH2<'_> {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.eval_h(q)
    }
}

impl BandSurface for FixedCuH2<'_> {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (energy, g) = self.eval_h(row)?;
            energies[i] = energy;
            gradients.row_mut(i).assign(&g);
        }
        Ok(())
    }
}

fn h_coords(system: &ConSystem) -> Result<Array1<f64>, String> {
    let [i, j] = hydrogen_indices(system)?;
    Ok(array![
        system.positions[(i, 0)],
        system.positions[(i, 1)],
        system.positions[(i, 2)],
        system.positions[(j, 0)],
        system.positions[(j, 1)],
        system.positions[(j, 2)],
    ])
}

fn lineb_config() -> LinebConfig {
    LinebConfig::new(OPT_TOL, OPT_TOL)
        .with_hbar(HBAR)
        .with_boltzmann(BOLTZMANN)
}

fn masses_h() -> Result<Masses, SaddleError> {
    Masses::coordinates(Array1::from_elem(6, H_MASS).view())
}

fn write_csv_row(file: &mut File, fields: &[String]) -> std::io::Result<()> {
    writeln!(file, "{}", fields.join(","))
}

fn run_smoke(
    surface: &FixedCuH2<'_>,
    reactant: &ConSystem,
    product: &ConSystem,
) -> Result<(), String> {
    let r = h_coords(reactant)?;
    let p = h_coords(product)?;
    let (e_r, _) = surface.eval_h(r.view()).map_err(|e| e.to_string())?;
    let (e_p, _) = surface.eval_h(p.view()).map_err(|e| e.to_string())?;
    println!("absolute energy reactant (adsorbed): {e_r:.10} eV");
    println!("absolute energy product (desorbed):  {e_p:.10} eV");
    println!("E_product - E_reactant: {:+.10} eV", e_p - e_r);
    println!("smoke surface calls: {}", surface.calls());
    Ok(())
}

fn run_mep(
    surface: &FixedCuH2<'_>,
    reactant: &ConSystem,
    product: &ConSystem,
    out: &Path,
) -> Result<Array2<f64>, String> {
    let before = surface.calls();
    let start = h_coords(reactant)?;
    let end = h_coords(product)?;
    // Two H atoms as a flat 6-vector per image (3N with N = 2).
    let mut initial = Array2::zeros((MEP_IMAGES, 6));
    for i in 0..MEP_IMAGES {
        let t = i as f64 / (MEP_IMAGES - 1) as f64;
        for k in 0..6 {
            initial[(i, k)] = (1.0 - t) * start[k] + t * end[k];
        }
    }
    let config = BandConfig::default()
        // Translation of both H atoms relative to the fixed substrate is physical.
        .with_remove_translation(false)
        .with_force_tol(1.0e-3)
        .with_max_move(0.1)
        .with_climbing(CiConfig::always());
    let mut session = BandSession::new(config, initial).map_err(|e| e.to_string())?;
    let mut converged = false;
    for step in 0..4000 {
        let report = session.step(surface).map_err(|e| e.to_string())?;
        if step % 50 == 0 || report.status == BandStatus::Converged {
            println!(
                "MEP step {step}: status={:?} max_force={:.4e}",
                report.status, report.max_force
            );
        }
        if report.status == BandStatus::Converged {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err("MEP did not converge".into());
    }
    let band = session.positions().to_owned();
    let mut energies = Array1::zeros(band.nrows());
    let mut gradients = Array2::zeros(band.raw_dim());
    BandSurface::eval(surface, band.view(), &mut energies, &mut gradients)
        .map_err(|e| e.to_string())?;
    let (saddle_i, &e_saddle) = energies
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();
    let e_reactant = energies[0];
    let saddle_force = gradients
        .row(saddle_i)
        .iter()
        .map(|g| g * g)
        .sum::<f64>()
        .sqrt();
    if saddle_i == 0 || saddle_i + 1 == band.nrows() || saddle_force > 2.0e-3 {
        return Err(format!(
            "MEP maximum at image {saddle_i} has gradient norm {saddle_force:.6e} eV/Å"
        ));
    }
    let barrier = e_saddle - e_reactant;
    let mep_calls = surface.calls() - before;
    println!(
        "MEP: saddle image {saddle_i}, E_saddle={e_saddle:.10} eV, E_reactant={e_reactant:.10} eV, \
         barrier={barrier:.10} eV (howto target 0.469 eV), band_calls={mep_calls}"
    );
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let mut file = File::create(out.join("mep.csv")).map_err(|e| e.to_string())?;
    writeln!(file, "image,energy_eV,h1x,h1y,h1z,h2x,h2y,h2z").map_err(|e| e.to_string())?;
    for (i, row) in band.outer_iter().enumerate() {
        writeln!(
            file,
            "{i},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16}",
            energies[i], row[0], row[1], row[2], row[3], row[4], row[5]
        )
        .map_err(|e| e.to_string())?;
    }
    let mut meta = File::create(out.join("mep_meta.csv")).map_err(|e| e.to_string())?;
    writeln!(meta, "quantity,value").map_err(|e| e.to_string())?;
    writeln!(meta, "barrier_eV,{barrier:.16}").map_err(|e| e.to_string())?;
    writeln!(meta, "E_reactant_eV,{e_reactant:.16}").map_err(|e| e.to_string())?;
    writeln!(meta, "E_saddle_eV,{e_saddle:.16}").map_err(|e| e.to_string())?;
    writeln!(meta, "band_force_calls,{mep_calls}").map_err(|e| e.to_string())?;
    writeln!(meta, "saddle_image,{saddle_i}").map_err(|e| e.to_string())?;
    Ok(band)
}

fn run_ladder(
    surface: &FixedCuH2<'_>,
    mep: ArrayView2<'_, f64>,
    out: &Path,
) -> Result<Vec<rgsaddle::LinebPath>, String> {
    let before = surface.calls();
    let masses = masses_h().map_err(|e| e.to_string())?;
    let config = lineb_config();
    let ladder = Continuation::new(LADDER_DELTA, LADDER_IMAGES, LADDER_LEVELS).with_max_steps(2000);
    let paths =
        continue_from_mep(surface, mep, &masses, &config, &ladder).map_err(|e| e.to_string())?;
    let after = surface.calls();
    let returned: usize = paths.iter().map(|p| p.calls.total()).sum();
    let seeds = after.saturating_sub(before + returned);
    let mut file = File::create(out.join("ladder.csv")).map_err(|e| e.to_string())?;
    writeln!(
        file,
        "level,energy_eV,action,half_period,temperature_K,half_period_dynamics,temperature_dynamics_K,\
         calls_optimizer,calls_measure,calls_dynamics,calls_total"
    )
    .map_err(|e| e.to_string())?;
    for (level, path) in paths.iter().enumerate() {
        println!(
            "ladder[{level}]: E={:.6} T={:.2} K T_dyn={:.2} K action={:.6} \
             calls opt/meas/dyn={}/{}/{}",
            path.energy,
            path.temperature,
            path.temperature_dynamics,
            path.action,
            path.calls.optimizer,
            path.calls.measure,
            path.calls.dynamics
        );
        write_csv_row(
            &mut file,
            &[
                level.to_string(),
                format!("{:.16}", path.energy),
                format!("{:.16}", path.action),
                format!("{:.16}", path.half_period),
                format!("{:.16}", path.temperature),
                format!("{:.16}", path.half_period_dynamics),
                format!("{:.16}", path.temperature_dynamics),
                path.calls.optimizer.to_string(),
                path.calls.measure.to_string(),
                path.calls.dynamics.to_string(),
                path.calls.total().to_string(),
            ],
        )
        .map_err(|e| e.to_string())?;
    }
    let mut meta = File::create(out.join("ladder_calls.csv")).map_err(|e| e.to_string())?;
    writeln!(meta, "quantity,value").map_err(|e| e.to_string())?;
    writeln!(meta, "surface_calls_span,{}", after - before).map_err(|e| e.to_string())?;
    writeln!(meta, "returned_path_calls_sum,{returned}").map_err(|e| e.to_string())?;
    writeln!(meta, "seed_calls,{seeds}").map_err(|e| e.to_string())?;
    writeln!(meta, "howto_seed_calls_reference,977").map_err(|e| e.to_string())?;
    println!(
        "ladder call ledger: surface_span={} returned_sum={returned} seeds={seeds} \
         (howto seeds reference 977)",
        after - before
    );
    Ok(paths)
}

fn run_rates(
    surface: &FixedCuH2<'_>,
    paths: &[rgsaddle::LinebPath],
    out: &Path,
) -> Result<(), String> {
    let mut file = File::create(out.join("rates.csv")).map_err(|e| e.to_string())?;
    writeln!(
        file,
        "level,energy_eV,temperature_K,beads,direct_log_rate,oracle_log_rate,delta_log,\
         direct_calls,oracle_calls,expected_direct_calls"
    )
    .map_err(|e| e.to_string())?;
    for (level, path) in paths.iter().enumerate() {
        let before = surface.calls();
        let direct = hqtst_rate(surface, path, RATE_BEADS).map_err(|e| e.to_string())?;
        let after_direct = surface.calls();
        let oracle = hqtst_rate_interpolated_hessian(surface, path, RATE_BEADS)
            .map_err(|e| e.to_string())?;
        let after_oracle = surface.calls();
        // hqtst_rate reports its own evaluation count; cross-check the wrapper.
        if direct.evaluations != EXPECTED_RATE_CALLS {
            println!(
                "warning: direct.evaluations={} expected {EXPECTED_RATE_CALLS}",
                direct.evaluations
            );
        }
        let delta = direct.log_rate_over_reactant - oracle.log_rate_over_reactant;
        println!(
            "rate[{level}]: T={:.2} K ln(kZr)_direct={:.8} ln(kZr)_oracle={:.8} Δ={:.3e} \
             direct_calls={} oracle_calls={} wrapper_delta_calls={}/{}",
            path.temperature,
            direct.log_rate_over_reactant,
            oracle.log_rate_over_reactant,
            delta,
            direct.evaluations,
            oracle.evaluations,
            after_direct - before,
            after_oracle - after_direct
        );
        write_csv_row(
            &mut file,
            &[
                level.to_string(),
                format!("{:.16}", path.energy),
                format!("{:.16}", path.temperature),
                RATE_BEADS.to_string(),
                format!("{:.16}", direct.log_rate_over_reactant),
                format!("{:.16}", oracle.log_rate_over_reactant),
                format!("{:.16}", delta),
                direct.evaluations.to_string(),
                oracle.evaluations.to_string(),
                EXPECTED_RATE_CALLS.to_string(),
            ],
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    if args.fixtures_only && !args.smoke && !args.mep && !args.ladder && !args.rates {
        run_fixtures_check()?;
        return Ok(());
    }
    run_fixtures_check()?;
    if !args.smoke && !args.mep && !args.ladder && !args.rates {
        return Ok(());
    }
    let lib_path = args.lib.ok_or_else(|| {
        "CuH2 stages need --lib PATH or RGPOT_CUH2_LIB pointing at a library exporting \
         rgpot_cuh2_force"
            .to_string()
    })?;
    let lib = unsafe { CuH2Lib::open(&lib_path) }?;
    let reactant = parse_con(&fixtures_dir().join("eon_adsorbed_min2.con"))?;
    let product = parse_con(&fixtures_dir().join("eon_desorbed_min1.con"))?;
    // Use the eOn Cu lattice (identical for adsorbed/desorbed) so the MEP
    // endpoints are consistent; freeze Cu regardless of CON flags.
    let surface = FixedCuH2::new(&lib, &reactant)?;
    fs::create_dir_all(&args.out)?;

    if args.smoke {
        run_smoke(&surface, &reactant, &product)?;
    }

    let mut mep_band = None;
    if args.mep || args.ladder || args.rates {
        mep_band = Some(run_mep(&surface, &reactant, &product, &args.out)?);
    }

    let mut paths = None;
    if args.ladder || args.rates {
        let band = mep_band.as_ref().expect("ladder/rates require an MEP band");
        paths = Some(run_ladder(&surface, band.view(), &args.out)?);
    }

    if args.rates {
        let paths = paths.ok_or("rates requires ladder paths")?;
        run_rates(&surface, &paths, &args.out)?;
    }

    println!(
        "total CuH2 surface evaluations this process: {}",
        surface.calls()
    );
    let mut total = File::create(args.out.join("total_calls.txt"))?;
    writeln!(total, "{}", surface.calls())?;
    Ok(())
}
