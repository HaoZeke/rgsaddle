//! NVT Langevin velocity Verlet for one defect pair.
//!
//! [`Langevin`] and [`EnergyGradient`] are not copper-specific. The
//! executable's test is a vacancy plus a <100> split interstitial on the
//! Mishin EAM1 that `tests/mishin_cu.rs` evaluates (Mishin, Mehl,
//! Papaconstantopoulos, Voter, and Kress, Phys. Rev. B 63, 224106 (2001),
//! Table I). Recombination is read every 0.25 ps. `MD_DETECTOR=site_occupancy`,
//! the default, calls the pair recombined when every Wigner-Seitz site of the
//! ideal lattice holds one atom for `MD_CONFIRM_PS`, and dates the capture to
//! the first sample of that run. `MD_DETECTOR=coordination` is the first
//! method of Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601
//! (2025): every atom has 12 neighbours inside 3.1 Å at one sample.
//!
//! The lattice constant at each temperature is the zero-total-pressure value
//! that `direct_md_lattice` in `tests/mishin_cu.rs` writes to
//! `validation/direct_md_lattice.csv`.
//!
//! Time is the same code unit as `npt_lattice` in the copper test.
//! `angular_scale_rad_s` converts a physical timestep so that
//! `mass_amu * acceleration = gradient` with the gradient in eV/Å.
//!
//! ```text
//! MD_TEMPS=500 MD_SEEDS=8 MD_PS=8 MD_DT_FS=2 MD_GAMMA_PER_PS=1 MD_RMAX=10.8
//! cargo run --release --example langevin_nvt
//! ```
//!
//! `MD_CHECK=1` checks the cohesive energy and times one force call.
//! `MD_BENCH=1` times whole steps and force calls of `MD_BENCH_THREADS`
//! concurrent 500 K trajectories and prints a hash of the final positions,
//! so a kernel change can be checked for bit-identical trajectories.
//! `MD_SKIN_A` (default 1 Å) is the Verlet skin; it changes the summation
//! order and therefore the trajectories.
//! A CSV row is skipped when the same temperature, shell, orientation,
//! seed, and window are already in the output.

use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::thread;
use std::time::Instant;

use rgsaddle::mic::nearest_integer;
use rgsaddle::{BOLTZMANN_EV_PER_K, Persistence, SiteOccupancy, angular_scale_rad_s};

const RC: f64 = 5.50679;
const H_CUT: f64 = 0.50037;
const E1: f64 = 2.01458e2;
const E2: f64 = 6.59288e-3;
const R01: f64 = 0.83591;
const R02: f64 = 4.46867;
const ALPHA1: f64 = 2.97758;
const ALPHA2: f64 = 1.54927;
const DELTA: f64 = 0.86225e-2;
const RS: [f64; 3] = [2.24000, 1.80000, 1.20000];
const S_REP: [f64; 3] = [4.0, 40.0, 1.15000e3];
const DENSITY_A: f64 = 3.80362;
const R03: f64 = -2.19885;
const R04: f64 = -2.61984e2;
const BETA1: f64 = 0.17394;
const BETA2: f64 = 5.35661e2;
const F0: f64 = -2.28235;
const F2: f64 = 1.35535;
const Q1: f64 = -1.27775;
const Q2: f64 = -0.86074;
const Q3: f64 = 1.78804;
const Q4: f64 = 2.97571;
const EMB_Q1: f64 = 0.40000;
const EMB_Q2: f64 = 0.30000;

const CU_MASS_AMU: f64 = 63.546;
const COORD_CUTOFF_A: f64 = 3.1;
const SAMPLE_PS: f64 = 0.25;
const DUMBBELL_OFFSET_A: f64 = 1.025;
const TABLE_H: f64 = 2.0e-3;

/// Energy and Cartesian gradient of any potential the driver can step.
pub trait EnergyGradient {
    fn energy_gradient(
        &mut self,
        positions: &[f64],
        cell: [[f64; 3]; 3],
        gradient: &mut [f64],
    ) -> f64;

    /// dE/dλ at λ = 1 for r → λ r applied to positions and cell together,
    /// at the configuration of the last `energy_gradient` call.
    fn dilation_derivative(&mut self, positions: &[f64], cell: [[f64; 3]; 3]) -> f64 {
        finite_dilation(self, positions, cell)
    }
}

fn finite_dilation<P: EnergyGradient + ?Sized>(
    potential: &mut P,
    positions: &[f64],
    cell: [[f64; 3]; 3],
) -> f64 {
    let h = 1.0e-5;
    let mut scratch = vec![0.0; positions.len()];
    let mut energy_at = |lambda: f64| {
        let moved: Vec<f64> = positions.iter().map(|value| value * lambda).collect();
        let scaled = cell.map(|row| row.map(|value| value * lambda));
        potential.energy_gradient(&moved, scaled, &mut scratch)
    };
    let up = energy_at(1.0 + h);
    let down = energy_at(1.0 - h);
    energy_at(1.0);
    (up - down) / (2.0 * h)
}

/// One NVT Langevin velocity-Verlet step in the branch's code time.
///
/// The half kicks, the exact Ornstein-Uhlenbeck update, and the two half
/// drifts are the same split as `npt_lattice` in `tests/mishin_cu.rs`.
#[derive(Clone, Copy)]
pub struct Langevin {
    pub dt: f64,
    pub gamma: f64,
    pub mass: f64,
    pub thermal: f64,
}

impl Langevin {
    pub fn from_physical(dt_fs: f64, gamma_per_ps: f64, temperature: f64, mass: f64) -> Self {
        let omega = angular_scale_rad_s();
        Self {
            dt: dt_fs * 1.0e-15 * omega,
            gamma: gamma_per_ps * 1.0e12 / omega,
            mass,
            thermal: BOLTZMANN_EV_PER_K * temperature,
        }
    }

    pub fn step<P: EnergyGradient>(
        &self,
        potential: &mut P,
        positions: &mut [f64],
        velocities: &mut [f64],
        gradient: &mut [f64],
        cell: [[f64; 3]; 3],
        rng: &mut SplitRng,
    ) -> f64 {
        let half = 0.5 * self.dt / self.mass;
        for (velocity, slope) in velocities.iter_mut().zip(gradient.iter()) {
            *velocity -= half * slope;
        }
        for (position, velocity) in positions.iter_mut().zip(velocities.iter()) {
            *position += 0.5 * self.dt * velocity;
        }
        let damp = (-self.gamma * self.dt).exp();
        let noise = (self.thermal / self.mass * (1.0 - damp * damp)).sqrt();
        for velocity in velocities.iter_mut() {
            *velocity = damp * *velocity + noise * rng.normal();
        }
        for (position, velocity) in positions.iter_mut().zip(velocities.iter()) {
            *position += 0.5 * self.dt * velocity;
        }
        let energy = potential.energy_gradient(positions, cell, gradient);
        for (velocity, slope) in velocities.iter_mut().zip(gradient.iter()) {
            *velocity -= half * slope;
        }
        energy
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DynamicsKind {
    /// Langevin at `equil_gamma_per_ps` for `equil_ps`, then at `gamma_per_ps`.
    Langevin,
    /// Isotropic NPH at `pressure_ev_a3` for `equil_ps`, then NVE in the mean box of
    /// the second half of that stage. Both stages rescale velocities to the
    /// target temperature every `rescale_ps`.
    NphNve,
}

impl DynamicsKind {
    pub fn label(self) -> &'static str {
        match self {
            DynamicsKind::Langevin => "langevin",
            DynamicsKind::NphNve => "nph_nve",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Dynamics {
    pub kind: DynamicsKind,
    pub dt_fs: f64,
    pub temperature: f64,
    pub mass: f64,
    pub gamma_per_ps: f64,
    pub equil_gamma_per_ps: f64,
    pub equil_ps: f64,
    pub rescale_ps: f64,
    /// Rescale only when |T - target| exceeds this, in K.
    pub rescale_window_k: f64,
    /// Small-oscillation period of the volume, which fixes the piston mass.
    pub piston_ps: f64,
    /// Bulk-modulus guess that turns `piston_ps` into a piston mass. Only the
    /// period depends on it, not the mean volume.
    pub bulk_ev_a3: f64,
    pub pressure_ev_a3: f64,
}

impl Dynamics {
    pub fn steps(&self, ps: f64) -> usize {
        (ps / (self.dt_fs * 1.0e-3)).round() as usize
    }

    fn langevin(&self, gamma_per_ps: f64) -> Langevin {
        Langevin::from_physical(self.dt_fs, gamma_per_ps, self.temperature, self.mass)
    }
}

/// Positions, velocities, gradient, and cell of one trajectory under [`Dynamics`].
///
/// Velocities start Maxwellian at the target temperature with the
/// centre-of-mass velocity removed; the Langevin branch draws them, and its
/// noise, from the same generator in the same order as `langevin_until` did.
pub struct MdState {
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
    pub gradient: Vec<f64>,
    pub cell: [[f64; 3]; 3],
    pub step: usize,
    rng: SplitRng,
    equil_steps: usize,
    rescale_steps: usize,
    piston_momentum: f64,
    piston_mass: f64,
    scale_sum: f64,
    scale_count: usize,
    initial_cell: [[f64; 3]; 3],
}

fn cell_volume(cell: [[f64; 3]; 3]) -> f64 {
    let [a, b, c] = cell;
    (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]))
        .abs()
}

impl MdState {
    pub fn new<P: EnergyGradient>(
        dynamics: &Dynamics,
        potential: &mut P,
        positions: Vec<f64>,
        cell: [[f64; 3]; 3],
        seed: u64,
    ) -> Self {
        let mut rng = SplitRng::new(seed);
        let sigma = (BOLTZMANN_EV_PER_K * dynamics.temperature / dynamics.mass).sqrt();
        let mut velocities: Vec<f64> = (0..positions.len()).map(|_| rng.normal() * sigma).collect();
        zero_com(&mut velocities);
        let mut gradient = vec![0.0; positions.len()];
        potential.energy_gradient(&positions, cell, &mut gradient);
        let omega = angular_scale_rad_s();
        let period = dynamics.piston_ps * 1.0e-12 * omega;
        let frequency = std::f64::consts::TAU / period;
        let piston_mass = 9.0 * cell_volume(cell) * dynamics.bulk_ev_a3 / (frequency * frequency);
        Self {
            positions,
            velocities,
            gradient,
            cell,
            step: 0,
            rng,
            equil_steps: dynamics.steps(dynamics.equil_ps),
            rescale_steps: dynamics.steps(dynamics.rescale_ps).max(1),
            piston_momentum: 0.0,
            piston_mass,
            scale_sum: 0.0,
            scale_count: 0,
            initial_cell: cell,
        }
    }

    pub fn in_equilibration(&self) -> bool {
        self.step <= self.equil_steps
    }

    /// Current cell edge over the starting one; 1 unless a barostat has run.
    pub fn dilation(&self) -> f64 {
        self.cell[0][0] / self.initial_cell[0][0]
    }

    pub fn kinetic(&self, mass: f64) -> f64 {
        0.5 * mass * self.velocities.iter().map(|v| v * v).sum::<f64>()
    }

    pub fn temperature(&self, mass: f64) -> f64 {
        let dof = (self.velocities.len() as f64 - 3.0).max(1.0);
        2.0 * self.kinetic(mass) / (dof * BOLTZMANN_EV_PER_K)
    }

    pub fn advance<P: EnergyGradient>(&mut self, dynamics: &Dynamics, potential: &mut P) {
        self.step += 1;
        match dynamics.kind {
            DynamicsKind::Langevin => {
                let gamma = if self.step <= self.equil_steps {
                    dynamics.equil_gamma_per_ps
                } else {
                    dynamics.gamma_per_ps
                };
                dynamics.langevin(gamma).step(
                    potential,
                    &mut self.positions,
                    &mut self.velocities,
                    &mut self.gradient,
                    self.cell,
                    &mut self.rng,
                );
            }
            DynamicsKind::NphNve => {
                if self.step <= self.equil_steps {
                    self.nph_step(dynamics, potential);
                    if 2 * self.step > self.equil_steps {
                        self.scale_sum += self.dilation();
                        self.scale_count += 1;
                    }
                    if self.step == self.equil_steps && self.scale_count > 0 {
                        let target = self.scale_sum / self.scale_count as f64;
                        self.scale_to(target / self.dilation());
                        potential.energy_gradient(&self.positions, self.cell, &mut self.gradient);
                    }
                } else {
                    self.verlet_step(dynamics, potential);
                }
                if self.step.is_multiple_of(self.rescale_steps) {
                    let now = self.temperature(dynamics.mass);
                    if now > 0.0 && (now - dynamics.temperature).abs() > dynamics.rescale_window_k {
                        let factor = (dynamics.temperature / now).sqrt();
                        for velocity in &mut self.velocities {
                            *velocity *= factor;
                        }
                    }
                }
            }
        }
    }

    fn scale_to(&mut self, factor: f64) {
        for value in &mut self.positions {
            *value *= factor;
        }
        self.cell = self.cell.map(|row| row.map(|value| value * factor));
    }

    fn verlet_step<P: EnergyGradient>(&mut self, dynamics: &Dynamics, potential: &mut P) {
        let dt = dynamics.langevin(0.0).dt;
        let half = 0.5 * dt / dynamics.mass;
        for (velocity, slope) in self.velocities.iter_mut().zip(self.gradient.iter()) {
            *velocity -= half * slope;
        }
        for (position, velocity) in self.positions.iter_mut().zip(self.velocities.iter()) {
            *position += dt * velocity;
        }
        potential.energy_gradient(&self.positions, self.cell, &mut self.gradient);
        for (velocity, slope) in self.velocities.iter_mut().zip(self.gradient.iter()) {
            *velocity -= half * slope;
        }
    }

    /// Velocity Verlet with an isotropic Andersen piston on ln(volume)/3.
    /// The piston force is 3V(P - P0) with P = (2K - dE/dλ) / (3V).
    fn nph_step<P: EnergyGradient>(&mut self, dynamics: &Dynamics, potential: &mut P) {
        let dt = dynamics.langevin(0.0).dt;
        let half = 0.5 * dt / dynamics.mass;
        let piston_force = |state: &mut MdState, potential: &mut P| {
            let volume = cell_volume(state.cell);
            let virial = potential.dilation_derivative(&state.positions, state.cell);
            let pressure = (2.0 * state.kinetic(dynamics.mass) - virial) / (3.0 * volume);
            3.0 * volume * (pressure - dynamics.pressure_ev_a3)
        };
        for (velocity, slope) in self.velocities.iter_mut().zip(self.gradient.iter()) {
            *velocity -= half * slope;
        }
        self.piston_momentum += 0.5 * dt * piston_force(self, potential);
        let factor = (self.piston_momentum / self.piston_mass * dt).exp();
        self.scale_to(factor);
        for (position, velocity) in self.positions.iter_mut().zip(self.velocities.iter()) {
            *position += dt * velocity;
        }
        potential.energy_gradient(&self.positions, self.cell, &mut self.gradient);
        for (velocity, slope) in self.velocities.iter_mut().zip(self.gradient.iter()) {
            *velocity -= half * slope;
        }
        self.piston_momentum += 0.5 * dt * piston_force(self, potential);
    }
}

pub struct SplitRng {
    state: u64,
}

impl SplitRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    pub fn normal(&mut self) -> f64 {
        let u1 = ((self.u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))).max(1.0e-16);
        let u2 = (self.u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

pub struct TrajectoryOutcome {
    pub captured: bool,
    pub time_ps: f64,
    pub steps: usize,
    pub initial_score: usize,
    pub final_score: usize,
    /// Final cell edge over the starting one.
    pub dilation: f64,
}

/// Maxwellian velocities, then `dynamics` until the defect count `defects`
/// has read zero for `confirm` consecutive samples, or the window ends.
///
/// The count is recorded at the start and at the end. Capture time is the
/// first sample of the confirming run, counted from the Maxwellian start,
/// equilibration included; the start itself is sample zero, so a geometry
/// that is already recombined is time zero. A run that begins inside the
/// window may finish confirming after it.
#[allow(clippy::too_many_arguments)]
pub fn run_until<P, D>(
    dynamics: &Dynamics,
    potential: &mut P,
    positions: Vec<f64>,
    cell: [[f64; 3]; 3],
    window_ps: f64,
    sample_ps: f64,
    confirm: usize,
    seed: u64,
    mut defects: D,
) -> TrajectoryOutcome
where
    P: EnergyGradient,
    D: FnMut(&[f64], [[f64; 3]; 3]) -> usize,
{
    let mut state = MdState::new(dynamics, potential, positions, cell, seed);
    let initial_score = defects(&state.positions, state.cell);
    let mut watch = Persistence::new(confirm);
    let sample_steps = dynamics.steps(sample_ps).max(1);
    let total_steps = dynamics.steps(window_ps).max(sample_steps);
    let mut captured = false;
    let mut when = total_steps;
    let mut last = initial_score;
    if let Some(began) = watch.observe(0, initial_score == 0) {
        return TrajectoryOutcome {
            captured: true,
            time_ps: 0.0,
            steps: began,
            initial_score,
            final_score: initial_score,
            dilation: 1.0,
        };
    }
    while state.step < total_steps || watch.pending() {
        state.advance(dynamics, potential);
        if !state.step.is_multiple_of(sample_steps) {
            continue;
        }
        last = defects(&state.positions, state.cell);
        if let Some(began) = watch.observe(state.step, last == 0) {
            captured = true;
            when = began;
            break;
        }
    }
    if !captured && !state.step.is_multiple_of(sample_steps) {
        last = defects(&state.positions, state.cell);
    }
    TrajectoryOutcome {
        captured,
        time_ps: when as f64 * dynamics.dt_fs * 1.0e-3,
        steps: when,
        initial_score,
        final_score: last,
        dilation: state.dilation(),
    }
}

fn zero_com(velocities: &mut [f64]) {
    let n = velocities.len() / 3;
    if n == 0 {
        return;
    }
    let mut mean = [0.0; 3];
    for atom in 0..n {
        for axis in 0..3 {
            mean[axis] += velocities[3 * atom + axis];
        }
    }
    let inv = 1.0 / n as f64;
    for atom in 0..n {
        for axis in 0..3 {
            velocities[3 * atom + axis] -= mean[axis] * inv;
        }
    }
}

fn neighbor_bins(home: [f64; 3], nside: [i32; 3]) -> ([usize; 27], usize) {
    let mut bins = [0usize; 27];
    let mut count = 0usize;
    for ox in -1..=1 {
        for oy in -1..=1 {
            for oz in -1..=1 {
                let bx = (home[0] as i32 + ox).rem_euclid(nside[0]);
                let by = (home[1] as i32 + oy).rem_euclid(nside[1]);
                let bz = (home[2] as i32 + oz).rem_euclid(nside[2]);
                let bin = ((bx * nside[1] + by) * nside[2] + bz) as usize;
                if bins[..count].contains(&bin) {
                    continue;
                }
                bins[count] = bin;
                count += 1;
            }
        }
    }
    (bins, count)
}

struct Pair {
    i: u32,
    j: u32,
    d: [f64; 3],
    inv_r: f64,
    pv: f64,
    pd: f64,
    rv: f64,
    rd: f64,
}

/// Half of the 3×3×3 bin stencil. Each unordered pair of bins is seen once
/// when the grid has at least three bins on a side.
const HALF_STENCIL: [(i32, i32, i32); 14] = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, -1),
    (0, 1, 0),
    (0, 1, 1),
    (1, -1, -1),
    (1, -1, 0),
    (1, -1, 1),
    (1, 0, -1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, -1),
    (1, 1, 0),
    (1, 1, 1),
];

fn morse(r: f64, r0: f64, alpha: f64) -> (f64, f64) {
    let z = -alpha * (r - r0);
    let e1 = z.exp();
    let e2 = (2.0 * z).exp();
    (e2 - 2.0 * e1, 2.0 * alpha * (e1 - e2))
}

fn cutoff_weight(r: f64) -> (f64, f64) {
    let x = (r - RC) / H_CUT;
    if x >= 0.0 {
        return (0.0, 0.0);
    }
    let x2 = x * x;
    let x4 = x2 * x2;
    let den = 1.0 + x4;
    let psi = x4 / den;
    let dpsi = 4.0 * x * x2 / (den * den) / H_CUT;
    (psi, dpsi)
}

fn pair_analytic(r: f64) -> (f64, f64) {
    if r >= RC {
        return (0.0, 0.0);
    }
    let (m1, dm1) = morse(r, R01, ALPHA1);
    let (m2, dm2) = morse(r, R02, ALPHA2);
    let (psi, dpsi) = cutoff_weight(r);
    let core = E1 * m1 + E2 * m2 + DELTA;
    let mut value = core * psi;
    let mut slope = (E1 * dm1 + E2 * dm2) * psi + core * dpsi;
    for (&rs, &strength) in RS.iter().zip(S_REP.iter()) {
        if r < rs {
            let gap = rs - r;
            let gap3 = gap * gap * gap;
            value -= strength * gap * gap3;
            slope += 4.0 * strength * gap3;
        }
    }
    (value, slope)
}

fn density_analytic(r: f64) -> (f64, f64) {
    if r >= RC {
        return (0.0, 0.0);
    }
    let (psi, dpsi) = cutoff_weight(r);
    let d3 = r - R03;
    let g1 = (-BETA1 * d3 * d3).exp();
    let dg1 = g1 * (-2.0 * BETA1 * d3);
    let arg = -BETA2 * (r - R04);
    let (g2, dg2) = if arg < -700.0 {
        (0.0, 0.0)
    } else {
        let g2 = arg.exp();
        (g2, -BETA2 * g2)
    };
    let inner = DENSITY_A * g1 + g2;
    let dinner = DENSITY_A * dg1 + dg2;
    (inner * psi, dinner * psi + inner * dpsi)
}

fn embed_analytic(rho: f64) -> (f64, f64) {
    let x = rho - 1.0;
    let x2 = x * x;
    let x3 = x2 * x;
    if rho < 1.0 {
        let value = F0 + 0.5 * F2 * x2 + Q1 * x3 + Q2 * x2 * x2 + Q3 * x2 * x3 + Q4 * x3 * x3;
        let slope =
            F2 * x + 3.0 * Q1 * x2 + 4.0 * Q2 * x3 + 5.0 * Q3 * x2 * x2 + 6.0 * Q4 * x2 * x3;
        (value, slope)
    } else {
        let num = F0 + 0.5 * F2 * x2 + Q1 * x3 + EMB_Q1 * x2 * x2;
        let den = 1.0 + EMB_Q2 * x3;
        let dnum = F2 * x + 3.0 * Q1 * x2 + 4.0 * EMB_Q1 * x3;
        let dden = 3.0 * EMB_Q2 * x2;
        (num / den, (dnum * den - num * dden) / (den * den))
    }
}

fn positive_image(sx: i32, sy: i32, sz: i32) -> bool {
    sx > 0 || (sx == 0 && (sy > 0 || (sy == 0 && sz > 0)))
}

/// Mishin Cu EAM1. Orthogonal cells with side longer than `2 RC` use a
/// cell list and a cubic Hermite table of the analytic pair and density.
struct MishinCu {
    table: Vec<[f64; 4]>,
    knots: usize,
    pairs: Vec<Pair>,
    rho: Vec<f64>,
    fp: Vec<f64>,
    counts: Vec<u32>,
    offsets: Vec<u32>,
    order: Vec<u32>,
    packed: Vec<[f64; 3]>,
    neigh: Vec<(u32, u32)>,
    anchor: Vec<f64>,
    dilation: Option<f64>,
    rebuilds: usize,
    skin: f64,
}

impl MishinCu {
    fn new() -> Self {
        let knots = (RC / TABLE_H).ceil() as usize + 1;
        let mut table = vec![[0.0; 4]; knots];
        for (index, knot) in table.iter_mut().enumerate() {
            let r = index as f64 * TABLE_H;
            let (pair_value, pair_slope) = pair_analytic(r);
            let (density, density_slope) = density_analytic(r);
            *knot = [pair_value, pair_slope, density, density_slope];
        }
        Self {
            table,
            knots,
            pairs: Vec::new(),
            rho: Vec::new(),
            fp: Vec::new(),
            counts: Vec::new(),
            offsets: Vec::new(),
            order: Vec::new(),
            packed: Vec::new(),
            neigh: Vec::new(),
            anchor: Vec::new(),
            dilation: None,
            rebuilds: 0,
            skin: env_f64("MD_SKIN_A", 1.0),
        }
    }

    #[inline(always)]
    fn sample(&self, r: f64) -> [f64; 4] {
        if r >= RC {
            return [0.0; 4];
        }
        let t_grid = (r / TABLE_H).max(0.0);
        // `as` truncates, which is floor for the non-negative `t_grid`.
        let mut index = t_grid as usize;
        if index + 1 >= self.knots {
            index = self.knots - 2;
        }
        let t = t_grid - index as f64;
        let one_t = 1.0 - t;
        let h00 = (1.0 + 2.0 * t) * one_t * one_t;
        let h10 = t * one_t * one_t;
        let h01 = t * t * (3.0 - 2.0 * t);
        let h11 = t * t * (t - 1.0);
        let dh00 = -6.0 * t * one_t;
        let dh10 = one_t * (1.0 - 3.0 * t);
        let dh01 = 6.0 * t * one_t;
        let dh11 = 3.0 * t * t - 2.0 * t;
        let left = self.table[index];
        let right = self.table[index + 1];
        let mut out = [0.0; 4];
        for slot in 0..2 {
            let f0 = left[2 * slot];
            let d0 = left[2 * slot + 1];
            let f1 = right[2 * slot];
            let d1 = right[2 * slot + 1];
            out[2 * slot] = h00 * f0 + h10 * TABLE_H * d0 + h01 * f1 + h11 * TABLE_H * d1;
            out[2 * slot + 1] =
                (dh00 * f0 + dh10 * TABLE_H * d0 + dh01 * f1 + dh11 * TABLE_H * d1) / TABLE_H;
        }
        out
    }

    fn push_pair(&mut self, i: usize, j: usize, d: [f64; 3], r: f64) {
        let sampled = self.sample(r);
        self.pairs.push(Pair {
            i: i as u32,
            j: j as u32,
            d,
            inv_r: 1.0 / r,
            pv: sampled[0],
            pd: sampled[1],
            rv: sampled[2],
            rd: sampled[3],
        });
    }

    fn accumulate(&mut self, n: usize, gradient: &mut [f64]) -> f64 {
        self.rho.clear();
        self.rho.resize(n, 0.0);
        for bond in &self.pairs {
            let i = bond.i as usize;
            self.rho[i] += bond.rv;
            if bond.i != bond.j {
                self.rho[bond.j as usize] += bond.rv;
            } else {
                self.rho[i] += bond.rv;
            }
        }
        let mut energy = 0.0;
        self.fp.clear();
        self.fp.resize(n, 0.0);
        for i in 0..n {
            let (value, slope) = embed_analytic(self.rho[i]);
            energy += value;
            self.fp[i] = slope;
        }
        gradient.fill(0.0);
        let mut dilation = 0.0;
        let mut dilation_exact = true;
        for bond in &self.pairs {
            let i = bond.i as usize;
            if bond.i != bond.j {
                let j = bond.j as usize;
                energy += bond.pv;
                let deriv = bond.pd + (self.fp[i] + self.fp[j]) * bond.rd;
                let scale = deriv * bond.inv_r;
                for axis in 0..3 {
                    let force = scale * bond.d[axis];
                    gradient[3 * i + axis] -= force;
                    gradient[3 * j + axis] += force;
                }
                dilation +=
                    scale * (bond.d[0] * bond.d[0] + bond.d[1] * bond.d[1] + bond.d[2] * bond.d[2]);
            } else {
                energy += bond.pv;
                dilation_exact = false;
            }
        }
        self.dilation = dilation_exact.then_some(dilation);
        energy
    }
}

impl EnergyGradient for MishinCu {
    fn energy_gradient(
        &mut self,
        positions: &[f64],
        cell: [[f64; 3]; 3],
        gradient: &mut [f64],
    ) -> f64 {
        let n = positions.len() / 3;
        let lengths = [cell[0][0].abs(), cell[1][1].abs(), cell[2][2].abs()];
        let orthogonal = cell[0][1].abs()
            + cell[0][2].abs()
            + cell[1][0].abs()
            + cell[1][2].abs()
            + cell[2][0].abs()
            + cell[2][1].abs()
            < 1.0e-8;
        let skin = self.skin;
        if orthogonal && lengths.iter().all(|length| *length > 2.0 * (RC + skin)) {
            if self.list_stale(positions, lengths, skin) {
                self.rebuilds += 1;
                self.neigh.clear();
                self.pairs_cell(positions, lengths, RC + skin, true);
                self.anchor.clear();
                self.anchor.extend_from_slice(positions);
                self.pairs.clear();
            }
            self.pairs_from_list(positions, lengths);
        } else {
            self.pairs.clear();
            self.anchor.clear();
            self.neigh.clear();
            if orthogonal {
                self.pairs_cell(positions, lengths, RC, false);
            } else {
                self.pairs_images(positions, cell);
            }
        }
        self.accumulate(n, gradient)
    }

    fn dilation_derivative(&mut self, positions: &[f64], cell: [[f64; 3]; 3]) -> f64 {
        match self.dilation {
            Some(value) => value,
            None => finite_dilation(self, positions, cell),
        }
    }
}

impl MishinCu {
    fn pairs_images(&mut self, positions: &[f64], cell: [[f64; 3]; 3]) {
        let n = positions.len() / 3;
        let lengths = [0, 1, 2].map(|row| {
            (cell[row][0] * cell[row][0]
                + cell[row][1] * cell[row][1]
                + cell[row][2] * cell[row][2])
                .sqrt()
        });
        let min_len = lengths.into_iter().fold(f64::MAX, f64::min);
        let nimg = ((RC / min_len).ceil() as i32).max(1);
        let cutoff2 = RC * RC;
        for i in 0..n {
            let ri = [positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]];
            for j in i..n {
                let rj = [positions[3 * j], positions[3 * j + 1], positions[3 * j + 2]];
                for sx in -nimg..=nimg {
                    for sy in -nimg..=nimg {
                        for sz in -nimg..=nimg {
                            if i == j && !positive_image(sx, sy, sz) {
                                continue;
                            }
                            if i == j && sx == 0 && sy == 0 && sz == 0 {
                                continue;
                            }
                            let shift = [
                                sx as f64 * cell[0][0]
                                    + sy as f64 * cell[1][0]
                                    + sz as f64 * cell[2][0],
                                sx as f64 * cell[0][1]
                                    + sy as f64 * cell[1][1]
                                    + sz as f64 * cell[2][1],
                                sx as f64 * cell[0][2]
                                    + sy as f64 * cell[1][2]
                                    + sz as f64 * cell[2][2],
                            ];
                            let d = [
                                rj[0] + shift[0] - ri[0],
                                rj[1] + shift[1] - ri[1],
                                rj[2] + shift[2] - ri[2],
                            ];
                            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                            if r2 > 1.0e-16 && r2 < cutoff2 {
                                self.push_pair(i, j, d, r2.sqrt());
                            }
                        }
                    }
                }
            }
        }
    }

    fn list_stale(&self, positions: &[f64], lengths: [f64; 3], skin: f64) -> bool {
        if self.anchor.len() != positions.len() || self.neigh.is_empty() {
            return true;
        }
        let limit = 0.25 * skin * skin;
        for atom in 0..positions.len() / 3 {
            let mut distance = 0.0;
            for axis in 0..3 {
                let mut delta = positions[3 * atom + axis] - self.anchor[3 * atom + axis];
                delta -= lengths[axis] * nearest_integer(delta / lengths[axis]);
                distance += delta * delta;
            }
            if distance > limit {
                return true;
            }
        }
        false
    }

    fn pairs_from_list(&mut self, positions: &[f64], lengths: [f64; 3]) {
        self.pairs.clear();
        let cutoff2 = RC * RC;
        let inverse = lengths.map(|length| 1.0 / length);
        let neigh = std::mem::take(&mut self.neigh);
        for &(left, right) in &neigh {
            let i = left as usize;
            let j = right as usize;
            let mut d = [0.0; 3];
            for axis in 0..3 {
                d[axis] = positions[3 * j + axis] - positions[3 * i + axis];
                d[axis] -= lengths[axis] * nearest_integer(d[axis] * inverse[axis]);
            }
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 > 1.0e-16 && r2 < cutoff2 {
                self.push_pair(i, j, d, r2.sqrt());
            }
        }
        self.neigh = neigh;
    }

    /// Every pair inside `cutoff` in cell-list order. `list_only` appends the
    /// index pairs to the Verlet list instead of sampling them into `pairs`.
    fn pairs_cell(&mut self, positions: &[f64], lengths: [f64; 3], cutoff: f64, list_only: bool) {
        let n = positions.len() / 3;
        // With two bins along an axis the +1 and -1 neighbours are the same
        // bin and the half stencil visits its pairs twice; one bin is exact.
        let nside = lengths.map(|length| match (length / cutoff).floor() as i32 {
            count if count >= 3 => count,
            _ => 1,
        });
        let bins_n = (nside[0] * nside[1] * nside[2]) as usize;
        self.counts.clear();
        self.counts.resize(bins_n, 0);
        let mut bins = vec![0usize; n];
        for i in 0..n {
            let bin = bin_of(
                [
                    positions[3 * i] / lengths[0],
                    positions[3 * i + 1] / lengths[1],
                    positions[3 * i + 2] / lengths[2],
                ],
                nside,
            );
            bins[i] = bin;
            self.counts[bin] += 1;
        }
        self.offsets.clear();
        self.offsets.resize(bins_n + 1, 0);
        for bin in 0..bins_n {
            self.offsets[bin + 1] = self.offsets[bin] + self.counts[bin];
        }
        let mut cursor = self.offsets.clone();
        self.order.clear();
        self.order.resize(n, 0);
        self.packed.clear();
        self.packed.resize(n, [0.0; 3]);
        for i in 0..n {
            let bin = bins[i];
            let slot = cursor[bin] as usize;
            self.order[slot] = i as u32;
            self.packed[slot] = [positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]];
            cursor[bin] += 1;
        }
        let cutoff2 = cutoff * cutoff;
        let inverse = lengths.map(|length| 1.0 / length);
        for slot_i in 0..n {
            let i = self.order[slot_i] as usize;
            let ri = self.packed[slot_i];
            let bin_i = bins[i] as i32;
            let ix = bin_i / (nside[1] * nside[2]);
            let rem = bin_i % (nside[1] * nside[2]);
            let iy = rem / nside[2];
            let iz = rem % nside[2];
            let mut seen = [usize::MAX; 14];
            let mut nseen = 0usize;
            for (ox, oy, oz) in HALF_STENCIL {
                let bx = (ix + ox).rem_euclid(nside[0]);
                let by = (iy + oy).rem_euclid(nside[1]);
                let bz = (iz + oz).rem_euclid(nside[2]);
                let bin = ((bx * nside[1] + by) * nside[2] + bz) as usize;
                if seen[..nseen].contains(&bin) {
                    continue;
                }
                seen[nseen] = bin;
                nseen += 1;
                let start = self.offsets[bin] as usize;
                let end = self.offsets[bin + 1] as usize;
                for slot in start..end {
                    let j = self.order[slot] as usize;
                    if ox == 0 && oy == 0 && oz == 0 && j <= i {
                        continue;
                    }
                    let rj = self.packed[slot];
                    let mut d = [rj[0] - ri[0], rj[1] - ri[1], rj[2] - ri[2]];
                    for axis in 0..3 {
                        d[axis] -= lengths[axis] * nearest_integer(d[axis] * inverse[axis]);
                    }
                    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if r2 > 1.0e-16 && r2 < cutoff2 {
                        if list_only {
                            self.neigh.push((i as u32, j as u32));
                        } else {
                            self.push_pair(i, j, d, r2.sqrt());
                        }
                    }
                }
            }
        }
    }
}

fn bin_of(frac: [f64; 3], nside: [i32; 3]) -> usize {
    let ix = (frac[0].rem_euclid(1.0) * nside[0] as f64) as i32 % nside[0];
    let iy = (frac[1].rem_euclid(1.0) * nside[1] as f64) as i32 % nside[1];
    let iz = (frac[2].rem_euclid(1.0) * nside[2] as f64) as i32 % nside[2];
    ((ix * nside[1] + iy) * nside[2] + iz) as usize
}

fn analytic_gradient(positions: &[f64], cell: [[f64; 3]; 3], gradient: &mut [f64]) -> f64 {
    let mut pot = MishinCu::new();
    pot.pairs_images(positions, cell);
    let n = positions.len() / 3;
    pot.rho.resize(n, 0.0);
    for bond in &pot.pairs {
        let r = 1.0 / bond.inv_r;
        let weight = density_analytic(r).0;
        let i = bond.i as usize;
        pot.rho[i] += weight;
        if bond.i != bond.j {
            pot.rho[bond.j as usize] += weight;
        } else {
            pot.rho[i] += weight;
        }
    }
    let mut energy = 0.0;
    pot.fp.resize(n, 0.0);
    for i in 0..n {
        let (value, slope) = embed_analytic(pot.rho[i]);
        energy += value;
        pot.fp[i] = slope;
    }
    gradient.fill(0.0);
    for bond in &pot.pairs {
        let r = 1.0 / bond.inv_r;
        let (value, dv) = pair_analytic(r);
        let drho = density_analytic(r).1;
        let i = bond.i as usize;
        if bond.i != bond.j {
            let j = bond.j as usize;
            energy += value;
            let deriv = dv + (pot.fp[i] + pot.fp[j]) * drho;
            for axis in 0..3 {
                let force = deriv * bond.d[axis] * bond.inv_r;
                gradient[3 * i + axis] -= force;
                gradient[3 * j + axis] += force;
            }
        } else {
            energy += value;
        }
    }
    energy
}

/// Atoms whose neighbor count inside 3.1 Å is not 12.
pub fn coordination_deficit(positions: &[f64], cell: [[f64; 3]; 3]) -> usize {
    let n = positions.len() / 3;
    let lengths = [cell[0][0], cell[1][1], cell[2][2]];
    let cutoff2 = COORD_CUTOFF_A * COORD_CUTOFF_A;
    let mut coord = vec![0u16; n];
    if lengths.iter().all(|length| *length > 2.0 * COORD_CUTOFF_A) {
        let nside = lengths.map(|length| ((length / COORD_CUTOFF_A).floor() as i32).max(1));
        let bins_n = (nside[0] * nside[1] * nside[2]) as usize;
        let mut members = vec![Vec::new(); bins_n];
        for i in 0..n {
            let bin = bin_of(
                [
                    positions[3 * i] / lengths[0],
                    positions[3 * i + 1] / lengths[1],
                    positions[3 * i + 2] / lengths[2],
                ],
                nside,
            );
            members[bin].push(i);
        }
        for i in 0..n {
            let ri = [positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]];
            let home = [
                (ri[0] / lengths[0]).rem_euclid(1.0) * nside[0] as f64,
                (ri[1] / lengths[1]).rem_euclid(1.0) * nside[1] as f64,
                (ri[2] / lengths[2]).rem_euclid(1.0) * nside[2] as f64,
            ];
            let (bins, bin_count) = neighbor_bins(home, nside);
            for bin in bins.into_iter().take(bin_count) {
                for &j in &members[bin] {
                    if j <= i {
                        continue;
                    }
                    let mut d = [
                        positions[3 * j] - ri[0],
                        positions[3 * j + 1] - ri[1],
                        positions[3 * j + 2] - ri[2],
                    ];
                    for axis in 0..3 {
                        d[axis] -= lengths[axis] * (d[axis] / lengths[axis]).round();
                    }
                    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if r2 < cutoff2 {
                        coord[i] += 1;
                        coord[j] += 1;
                    }
                }
            }
        }
    }
    coord.into_iter().filter(|count| *count != 12).count()
}

/// Wigner-Seitz sites holding other than one atom.
pub fn site_defects(sites: &mut SiteOccupancy, positions: &[f64], cell: [[f64; 3]; 3]) -> usize {
    sites
        .occupancy(positions, cell)
        .expect("orthorhombic cell")
        .defect_sites()
}

fn fcc(n: usize, a: f64) -> (Vec<f64>, [[f64; 3]; 3]) {
    let basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let mut values = Vec::with_capacity(n * n * n * 12);
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                for site in basis {
                    values.push((ix as f64 + site[0]) * a);
                    values.push((iy as f64 + site[1]) * a);
                    values.push((iz as f64 + site[2]) * a);
                }
            }
        }
    }
    let length = n as f64 * a;
    (
        values,
        [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
    )
}

fn shell_table(a: f64, r_max: f64) -> Vec<(f64, [f64; 3], usize)> {
    let limit = ((2.0 * r_max / a).ceil() as i32) + 1;
    let mut raw = Vec::new();
    for i in -limit..=limit {
        for j in -limit..=limit {
            for k in -limit..=limit {
                if (i + j + k) % 2 != 0 || (i == 0 && j == 0 && k == 0) {
                    continue;
                }
                let cart = [i as f64 * 0.5 * a, j as f64 * 0.5 * a, k as f64 * 0.5 * a];
                let r = (cart[0] * cart[0] + cart[1] * cart[1] + cart[2] * cart[2]).sqrt();
                if r < 0.5 * a || r > r_max {
                    continue;
                }
                raw.push((r, cart));
            }
        }
    }
    raw.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut found: Vec<(f64, [f64; 3], usize)> = Vec::new();
    for (radius, cart) in raw {
        if let Some(last) = found.last_mut()
            && (radius - last.0).abs() < 1.0e-3
        {
            last.2 += 1;
            continue;
        }
        found.push((radius, cart, 1));
    }
    found
}

/// Vacancy plus a <100> dumbbell on one shell site. Atom count is unchanged.
pub fn place_vacancy_dumbbell(
    repeats: usize,
    a: f64,
    shell: [f64; 3],
    axis: [f64; 3],
) -> (Vec<f64>, [[f64; 3]; 3]) {
    let (flat, cell) = fcc(repeats, a);
    let n = flat.len() / 3;
    let mut points = Vec::with_capacity(n);
    for atom in 0..n {
        points.push([flat[3 * atom], flat[3 * atom + 1], flat[3 * atom + 2]]);
    }
    let center = [cell[0][0] * 0.5, cell[1][1] * 0.5, cell[2][2] * 0.5];
    let vacancy = nearest(&points, center, None);
    let lengths = [cell[0][0], cell[1][1], cell[2][2]];
    let sia_guess = [
        (points[vacancy][0] + shell[0]).rem_euclid(lengths[0]),
        (points[vacancy][1] + shell[1]).rem_euclid(lengths[1]),
        (points[vacancy][2] + shell[2]).rem_euclid(lengths[2]),
    ];
    let sia = nearest(&points, sia_guess, Some(vacancy));
    let sia_pos = points[sia];
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
        .sqrt()
        .max(1.0e-8);
    let unit = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
    let mut keep = Vec::with_capacity(n);
    for (index, point) in points.iter().enumerate() {
        if index == sia || index == vacancy {
            continue;
        }
        keep.push(*point);
    }
    keep.push([
        sia_pos[0] - DUMBBELL_OFFSET_A * unit[0],
        sia_pos[1] - DUMBBELL_OFFSET_A * unit[1],
        sia_pos[2] - DUMBBELL_OFFSET_A * unit[2],
    ]);
    keep.push([
        sia_pos[0] + DUMBBELL_OFFSET_A * unit[0],
        sia_pos[1] + DUMBBELL_OFFSET_A * unit[1],
        sia_pos[2] + DUMBBELL_OFFSET_A * unit[2],
    ]);
    let mut values = Vec::with_capacity(keep.len() * 3);
    for point in keep {
        values.extend(point);
    }
    (values, cell)
}

fn nearest(points: &[[f64; 3]], target: [f64; 3], skip: Option<usize>) -> usize {
    let mut best = 0usize;
    let mut best_r = f64::MAX;
    for (index, point) in points.iter().enumerate() {
        if skip == Some(index) {
            continue;
        }
        let d = [
            point[0] - target[0],
            point[1] - target[1],
            point[2] - target[2],
        ];
        let r = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if r < best_r {
            best_r = r;
            best = index;
        }
    }
    best
}

/// Zero-total-pressure lattice constant and its standard error by temperature,
/// from `validation/direct_md_lattice.csv`.
fn zero_pressure_lattices() -> &'static BTreeMap<i64, (f64, f64)> {
    static TABLE: std::sync::OnceLock<BTreeMap<i64, (f64, f64)>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("validation/direct_md_lattice.csv");
        let text = fs::read_to_string(&path).expect("validation/direct_md_lattice.csv");
        let mut table = BTreeMap::new();
        for line in text.lines().skip(1) {
            let fields: Vec<&str> = line.split(',').collect();
            if let (Some(t), Some(a), Some(e)) = (fields.first(), fields.get(1), fields.get(2))
                && let (Ok(t), Ok(a), Ok(e)) =
                    (t.parse::<f64>(), a.parse::<f64>(), e.parse::<f64>())
            {
                table.insert(t.round() as i64, (a, e));
            }
        }
        table
    })
}

fn measured_lattice(temperature: f64) -> f64 {
    zero_pressure_lattices()
        .get(&(temperature.round() as i64))
        .unwrap_or_else(|| panic!("no zero-pressure lattice at {temperature} K"))
        .0
}

const ORIENTATIONS: [(&str, [f64; 3]); 3] = [
    ("100", [1.0, 0.0, 0.0]),
    ("010", [0.0, 1.0, 0.0]),
    ("001", [0.0, 0.0, 1.0]),
];

struct Job {
    temperature: f64,
    lattice: f64,
    shell_r: f64,
    shell: [f64; 3],
    shell_sites: usize,
    orientation: &'static str,
    axis: [f64; 3],
    seed_index: u64,
    seed: u64,
}

/// How a whole-cell trajectory decides the pair has recombined.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Detector {
    /// Every atom 12-coordinated inside 3.1 Å at one sample.
    Coordination,
    /// Every Wigner-Seitz site singly occupied for the confirmation window.
    SiteOccupancy,
}

impl Detector {
    fn label(self) -> &'static str {
        match self {
            Detector::Coordination => "coordination",
            Detector::SiteOccupancy => "site_occupancy",
        }
    }
}

struct RunSpec {
    temps: Vec<f64>,
    seeds: u64,
    window_ps: f64,
    dt_fs: f64,
    gamma_per_ps: f64,
    r_max_500: f64,
    repeats: usize,
    random_vacancy: bool,
    kind: DynamicsKind,
    equil_ps: f64,
    rescale_ps: f64,
    rescale_window_k: f64,
    piston_ps: f64,
    bulk_ev_a3: f64,
    detector: Detector,
    confirm_ps: f64,
}

impl RunSpec {
    fn dynamics(&self, temperature: f64) -> Dynamics {
        Dynamics {
            kind: self.kind,
            dt_fs: self.dt_fs,
            temperature,
            mass: CU_MASS_AMU,
            gamma_per_ps: self.gamma_per_ps,
            equil_gamma_per_ps: 1.0,
            equil_ps: self.equil_ps,
            rescale_ps: self.rescale_ps,
            rescale_window_k: self.rescale_window_k,
            piston_ps: self.piston_ps,
            bulk_ev_a3: self.bulk_ev_a3,
            pressure_ev_a3: 0.0,
        }
    }

    /// The friction column: zero for dynamics without a Langevin bath.
    fn production_gamma(&self) -> f64 {
        match self.kind {
            DynamicsKind::Langevin => self.gamma_per_ps,
            DynamicsKind::NphNve => 0.0,
        }
    }

    /// Consecutive samples that confirm a capture.
    fn confirm_samples(&self) -> usize {
        match self.detector {
            Detector::Coordination => 1,
            Detector::SiteOccupancy => ((self.confirm_ps / SAMPLE_PS).round() as usize).max(1),
        }
    }

    /// Columns after the outcome when the detector is not the coordination test.
    fn detector_suffix(&self) -> String {
        match self.detector {
            Detector::Coordination => String::new(),
            Detector::SiteOccupancy => {
                format!(",{},{:.4}", self.detector.label(), self.confirm_ps)
            }
        }
    }

    fn header(&self) -> String {
        let base = match self.detector {
            Detector::Coordination => HEADER.to_string(),
            Detector::SiteOccupancy => {
                HEADER.replace(
                    "initial_non12,final_non12",
                    "initial_defect_sites,final_defect_sites",
                ) + DETECTOR_HEADER
            }
        };
        match self.kind {
            DynamicsKind::Langevin => base,
            DynamicsKind::NphNve => base + DYNAMICS_HEADER,
        }
    }

    /// Columns appended to a row when the dynamics are not Langevin.
    fn dynamics_suffix(&self) -> String {
        match self.kind {
            DynamicsKind::Langevin => String::new(),
            DynamicsKind::NphNve => format!(
                ",{},{:.4},{:.4},{:.4},{:.4}",
                self.kind.label(),
                self.equil_ps,
                self.rescale_ps,
                self.rescale_window_k,
                self.piston_ps
            ),
        }
    }
}

const DYNAMICS_HEADER: &str = ",dynamics,equil_ps,rescale_ps,rescale_window_K,piston_ps";
const DETECTOR_HEADER: &str = ",detector,confirm_ps";

/// Detector of a trajectory row and the index of its dynamics columns.
/// Rows without detector columns predate them and used the coordination test.
fn row_detector(fields: &[&str]) -> (Detector, f64, usize) {
    match fields.get(19) {
        Some(label) if *label == Detector::SiteOccupancy.label() => (
            Detector::SiteOccupancy,
            fields
                .get(20)
                .and_then(|text| text.parse().ok())
                .unwrap_or(f64::NAN),
            21,
        ),
        _ => (Detector::Coordination, 0.0, 19),
    }
}

fn env_f64(key: &str, default: f64) -> f64 {
    env::var(key)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

fn run_spec() -> RunSpec {
    let temps = env::var("MD_TEMPS").unwrap_or_else(|_| "500".to_string());
    let temps = temps
        .split(',')
        .filter(|item| !item.is_empty())
        .map(|item| item.trim().parse::<f64>().unwrap_or(500.0))
        .collect();
    let kind = match env::var("MD_DYNAMICS").ok().as_deref() {
        Some("nph_nve") => DynamicsKind::NphNve,
        Some("langevin") | None => DynamicsKind::Langevin,
        Some(other) => panic!("MD_DYNAMICS={other} is not langevin or nph_nve"),
    };
    RunSpec {
        temps,
        seeds: env_u64("MD_SEEDS", 8),
        window_ps: env_f64("MD_PS", 8.0),
        dt_fs: env_f64("MD_DT_FS", 2.0),
        gamma_per_ps: env_f64("MD_GAMMA_PER_PS", 1.0),
        r_max_500: env_f64("MD_RMAX", 10.8),
        repeats: env_u64("MD_REPEATS", 13) as usize,
        random_vacancy: env::var("MD_RANDOM").ok().as_deref() == Some("1"),
        kind,
        equil_ps: env_f64(
            "MD_EQUIL_PS",
            if kind == DynamicsKind::NphNve {
                30.0
            } else {
                2.0
            },
        ),
        rescale_ps: env_f64("MD_RESCALE_PS", 0.1),
        rescale_window_k: env_f64("MD_RESCALE_WINDOW_K", 0.0),
        piston_ps: env_f64("MD_PISTON_PS", 2.0),
        bulk_ev_a3: env_f64("MD_BULK_EV_A3", 0.87),
        detector: match env::var("MD_DETECTOR").ok().as_deref() {
            Some("coordination") => Detector::Coordination,
            Some("site_occupancy") | None => Detector::SiteOccupancy,
            Some(other) => panic!("MD_DETECTOR={other} is not site_occupancy or coordination"),
        },
        confirm_ps: env_f64("MD_CONFIRM_PS", 1.0),
    }
}

fn lattice_points(repeats: usize, a: f64) -> (Vec<[f64; 3]>, [[f64; 3]; 3]) {
    let (flat, cell) = fcc(repeats, a);
    let mut points = Vec::with_capacity(flat.len() / 3);
    for atom in 0..flat.len() / 3 {
        points.push([flat[3 * atom], flat[3 * atom + 1], flat[3 * atom + 2]]);
    }
    (points, cell)
}

fn random_jobs(spec: &RunSpec) -> Vec<Job> {
    let mut jobs = Vec::new();
    for &temperature in &spec.temps {
        let lattice = measured_lattice(temperature);
        let (points, cell) = lattice_points(spec.repeats, lattice);
        let lengths = [cell[0][0], cell[1][1], cell[2][2]];
        let center = [lengths[0] * 0.5, lengths[1] * 0.5, lengths[2] * 0.5];
        let vacancy = nearest(&points, center, None);
        for seed_index in 0..spec.seeds {
            let seed = mix_seed(&[temperature.to_bits(), seed_index, 0x5A10_5EED]);
            let mut rng = SplitRng::new(seed);
            let mut host = vacancy;
            while host == vacancy {
                host = (rng.u64() as usize) % points.len();
            }
            let mut shell = [0.0; 3];
            for axis in 0..3 {
                let mut delta = points[host][axis] - points[vacancy][axis];
                delta -= lengths[axis] * (delta / lengths[axis]).round();
                shell[axis] = delta;
            }
            let radius = (shell[0] * shell[0] + shell[1] * shell[1] + shell[2] * shell[2]).sqrt();
            let (label, axis) = ORIENTATIONS[(rng.u64() as usize) % ORIENTATIONS.len()];
            jobs.push(Job {
                temperature,
                lattice,
                shell_r: radius,
                shell,
                shell_sites: 1,
                orientation: label,
                axis,
                seed_index,
                seed,
            });
        }
    }
    jobs
}

fn jobs_for(spec: &RunSpec) -> Vec<Job> {
    if spec.random_vacancy {
        return random_jobs(spec);
    }
    let mut jobs = Vec::new();
    let a500 = measured_lattice(500.0);
    for &temperature in &spec.temps {
        let lattice = measured_lattice(temperature);
        let r_max = spec.r_max_500 * lattice / a500;
        for (radius, vector, count) in shell_table(lattice, r_max) {
            for (label, axis) in ORIENTATIONS {
                for seed_index in 0..spec.seeds {
                    let seed = mix_seed(&[
                        temperature.to_bits(),
                        radius.to_bits(),
                        label.as_bytes()[0] as u64,
                        seed_index,
                        0xC0FFEE,
                    ]);
                    jobs.push(Job {
                        temperature,
                        lattice,
                        shell_r: radius,
                        shell: vector,
                        shell_sites: count,
                        orientation: label,
                        axis,
                        seed_index,
                        seed,
                    });
                }
            }
        }
    }
    jobs
}

fn mix_seed(parts: &[u64]) -> u64 {
    let mut hash = 0x9E37_79B9_7F4A_7C15u64;
    for part in parts {
        hash ^= part.wrapping_add(0x9E37_79B9_7F4A_7C15);
        hash = hash.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        hash ^= hash >> 30;
    }
    hash | 1
}

fn output_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("validation");
    path.push("direct_md_viv.csv");
    if let Ok(over) = env::var("MD_OUT") {
        return PathBuf::from(over);
    }
    path
}

fn note_path(csv: &std::path::Path) -> PathBuf {
    csv.with_extension("md")
}

const HEADER: &str = "temperature_K,lattice_A,repeats,n_atoms,shell_A,shell_x,shell_y,shell_z,orientation,shell_sites,seed,dt_fs,gamma_per_ps,window_ps,captured,time_ps,steps,initial_non12,final_non12";

fn settled_keys(path: &std::path::Path, target_window: f64) -> BTreeMap<String, ()> {
    let mut captured = BTreeMap::new();
    let mut reached = BTreeMap::new();
    let Ok(file) = File::open(path) else {
        return captured;
    };
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 16 {
            continue;
        }
        let Ok(temperature) = fields[0].parse::<f64>() else {
            continue;
        };
        let Ok(shell) = fields[4].parse::<f64>() else {
            continue;
        };
        let Ok(seed) = fields[10].parse::<u64>() else {
            continue;
        };
        let Ok(stored_window) = fields[13].parse::<f64>() else {
            continue;
        };
        let Ok(did_capture) = fields[14].parse::<u8>() else {
            continue;
        };
        let key = format!("{temperature:.0},{shell:.6},{},{seed}", fields[8]);
        if did_capture == 1 {
            captured.insert(key, ());
        } else {
            let far = reached.get(&key).copied().unwrap_or(0.0);
            if stored_window > far {
                reached.insert(key, stored_window);
            }
        }
    }
    for (key, window) in reached {
        if window + 1.0e-6 >= target_window {
            captured.insert(key, ());
        }
    }
    captured
}

fn run_job(potential: &mut MishinCu, spec: &RunSpec, job: &Job) -> (usize, TrajectoryOutcome) {
    let (positions, cell) = place_vacancy_dumbbell(spec.repeats, job.lattice, job.shell, job.axis);
    let n_atoms = positions.len() / 3;
    let dynamics = spec.dynamics(job.temperature);
    let outcome = match spec.detector {
        Detector::Coordination => run_until(
            &dynamics,
            potential,
            positions,
            cell,
            spec.window_ps,
            SAMPLE_PS,
            1,
            job.seed,
            coordination_deficit,
        ),
        Detector::SiteOccupancy => {
            let mut sites = SiteOccupancy::new(&fcc(spec.repeats, job.lattice).0, cell, &positions)
                .expect("orthorhombic reference");
            run_until(
                &dynamics,
                potential,
                positions,
                cell,
                spec.window_ps,
                SAMPLE_PS,
                spec.confirm_samples(),
                job.seed,
                |positions: &[f64], cell| site_defects(&mut sites, positions, cell),
            )
        }
    };
    (n_atoms, outcome)
}

fn format_row(spec: &RunSpec, job: &Job, n_atoms: usize, outcome: &TrajectoryOutcome) -> String {
    format!(
        "{:.0},{:.6},{},{n_atoms},{:.6},{:.8},{:.8},{:.8},{},{},{},{:.4},{:.4},{:.4},{},{:.4},{},{},{}{}",
        job.temperature,
        job.lattice * outcome.dilation,
        spec.repeats,
        job.shell_r,
        job.shell[0],
        job.shell[1],
        job.shell[2],
        job.orientation,
        job.shell_sites,
        job.seed_index,
        spec.dt_fs,
        spec.production_gamma(),
        spec.window_ps,
        u8::from(outcome.captured),
        outcome.time_ps,
        outcome.steps,
        outcome.initial_score,
        outcome.final_score,
        spec.detector_suffix() + &spec.dynamics_suffix(),
    )
}

struct Group {
    temperature: f64,
    window: f64,
    shell: f64,
    orientation: String,
    sites: f64,
    captured: f64,
    trials: f64,
}

const DUMBBELL_ALPHA: f64 = 2.0 / 3.0;

fn first_method_report(csv: &std::path::Path) -> String {
    #[derive(Clone)]
    struct Trial {
        temperature: f64,
        lattice: f64,
        repeats: f64,
        shell: f64,
        orientation: String,
        sites: f64,
        window: f64,
        captured: bool,
        time_ps: f64,
        gamma: f64,
        kind: DynamicsKind,
    }
    let mut trials = Vec::new();
    for line in deduped_lines(csv) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 19 {
            continue;
        }
        let Ok(temperature) = fields[0].parse::<f64>() else {
            continue;
        };
        let Ok(lattice) = fields[1].parse::<f64>() else {
            continue;
        };
        let Ok(repeats) = fields[2].parse::<f64>() else {
            continue;
        };
        let Ok(shell) = fields[4].parse::<f64>() else {
            continue;
        };
        let Ok(sites) = fields[9].parse::<f64>() else {
            continue;
        };
        let Ok(window) = fields[13].parse::<f64>() else {
            continue;
        };
        let Ok(captured) = fields[14].parse::<u8>() else {
            continue;
        };
        let Ok(time_ps) = fields[15].parse::<f64>() else {
            continue;
        };
        trials.push(Trial {
            temperature,
            lattice,
            repeats,
            shell,
            orientation: fields[8].to_string(),
            sites,
            window,
            captured: captured == 1,
            time_ps,
            gamma: fields[12].parse().unwrap_or(f64::NAN),
            kind: kind_of(&dynamics_columns(&fields, row_detector(&fields).2)),
        });
    }
    if trials.is_empty() {
        return String::new();
    }
    let mut text = String::new();
    text.push_str("## Capture fraction against time\n\n");
    text.push_str("The fraction is the number of trajectories of that shell and orientation that had recombined by that time, divided by the number of trajectories. `V_shell` is the shell sum, sites/3 times the fraction, over every shell that has all three orientations. Sites outside 10.8 Å are not in `V_shell`.\n\n");
    let mut temperatures: BTreeMap<u64, f64> = BTreeMap::new();
    for trial in &trials {
        temperatures.insert(trial.temperature.to_bits(), trial.temperature);
    }
    let marks = [
        0.25_f64, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 150.0, 200.0, 300.0, 600.0, 900.0,
        1200.0, 1500.0,
    ];
    for temperature in temperatures.values() {
        let members: Vec<&Trial> = trials
            .iter()
            .filter(|trial| trial.temperature == *temperature)
            .collect();
        let window = members.iter().map(|trial| trial.window).fold(0.0, f64::max);
        let lattice = members[0].lattice;
        let repeats = members[0].repeats;
        text.push_str(&format!("### {temperature:.0} K\n\n"));
        text.push_str("| shell (Å) | orientation |");
        for mark in marks.iter().filter(|mark| **mark <= window + 1.0e-9) {
            text.push_str(&format!(" {mark:.0} ps |"));
        }
        text.push_str(" last new capture (ps) |\n|---:|:---:|");
        for _ in marks.iter().filter(|mark| **mark <= window + 1.0e-9) {
            text.push_str("---:|");
        }
        text.push_str("---:|\n");
        let mut groups: BTreeMap<(u64, String), Vec<&Trial>> = BTreeMap::new();
        for trial in &members {
            groups
                .entry((trial.shell.to_bits(), trial.orientation.clone()))
                .or_default()
                .push(trial);
        }
        for ((_, _), group) in &groups {
            let trial = group[0];
            text.push_str(&format!(
                "| {shell:.4} | {orient} |",
                shell = trial.shell,
                orient = trial.orientation
            ));
            for mark in marks.iter().filter(|mark| **mark <= window + 1.0e-9) {
                let hit = group
                    .iter()
                    .filter(|item| item.captured && item.time_ps <= *mark)
                    .count();
                text.push_str(&format!(" {:.3} |", hit as f64 / group.len() as f64));
            }
            let last = group
                .iter()
                .filter(|item| item.captured)
                .map(|item| item.time_ps)
                .fold(0.0, f64::max);
            text.push_str(&format!(" {last:.2} |\n"));
        }
        text.push_str("\n| time (ps) | V_shell/Ω | binomial σ |\n|---:|---:|---:|\n");
        let mut shells: BTreeMap<u64, Vec<&Trial>> = BTreeMap::new();
        for trial in &members {
            shells.entry(trial.shell.to_bits()).or_default().push(trial);
        }
        let volume_at = |mark: f64| -> (f64, f64) {
            let mut volume = 0.0;
            let mut variance = 0.0;
            for group in shells.values() {
                let mut orients: BTreeMap<&str, Vec<&Trial>> = BTreeMap::new();
                for trial in group {
                    orients
                        .entry(trial.orientation.as_str())
                        .or_default()
                        .push(trial);
                }
                if orients.len() < 3 {
                    continue;
                }
                for orient in orients.values() {
                    let hit = orient
                        .iter()
                        .filter(|item| item.captured && item.time_ps <= mark)
                        .count();
                    let share = orient[0].sites / 3.0;
                    let probability = hit as f64 / orient.len() as f64;
                    volume += share * probability;
                    variance +=
                        share * share * probability * (1.0 - probability) / orient.len() as f64;
                }
            }
            (volume, variance.sqrt())
        };
        for mark in marks.iter().filter(|mark| **mark <= window + 1.0e-9) {
            let (volume, sigma) = volume_at(*mark);
            text.push_str(&format!("| {mark:.2} | {volume:.2} | {sigma:.2} |\n"));
        }
        text.push('\n');
        for later in [600.0_f64, 900.0, 1200.0, 1500.0] {
            if later > window + 1.0e-9 {
                continue;
            }
            let (now, sigma) = volume_at(later);
            let (before, _) = volume_at(later - 300.0);
            let rise = now - before;
            let settled = rise < sigma;
            text.push_str(&format!(
                "From {:.0} to {later:.0} ps the shell sum rose {rise:.2} Ω. The standard error at {later:.0} ps is {sigma:.2} Ω. {verdict}.\n",
                later - 300.0,
                verdict = if settled { "That rise is inside one standard error" } else { "That rise is larger than one standard error" }
            ));
        }
        text.push('\n');
        let diffusivity = table_diffusivity(*temperature);
        let total_sites = repeats * repeats * repeats * 4.0;
        let model_volume = (repeats * lattice).powi(3);
        let mut fit_times = Vec::new();
        for mark in marks
            .iter()
            .filter(|mark| **mark <= window + 1.0e-9 && **mark > 0.0)
        {
            let mut captured_sites = 0.0;
            for group in shells.values() {
                let mut orients: BTreeMap<&str, Vec<&Trial>> = BTreeMap::new();
                for trial in group {
                    orients
                        .entry(trial.orientation.as_str())
                        .or_default()
                        .push(trial);
                }
                if orients.len() < 3 {
                    continue;
                }
                for orient in orients.values() {
                    let hit = orient
                        .iter()
                        .filter(|item| item.captured && item.time_ps <= *mark)
                        .count();
                    captured_sites += orient[0].sites / 3.0 * hit as f64 / orient.len() as f64;
                }
            }
            let survival = (1.0 - captured_sites / total_sites).max(1.0e-12);
            fit_times.push((*mark, survival.ln()));
        }
        text.push_str("A shell-restricted survival fit uses the same through-origin slope as section 2 of Mwungeli, Liu, and Demkowicz. The slope is -k/V_model, with V_model the 13³ cell volume. n(t)/N here counts every lattice site: a simulated shell contributes its capture fraction, and a site outside 10.8 Å is counted as not yet recombined, so this volume lies below the whole-cell ensemble. The recombination volume is k a² / (α D_i) with α = 2/3, divided by Ω = a³/4.\n\n");
        if fit_times.is_empty() {
            continue;
        }
        let denominator: f64 = fit_times.iter().map(|(time, _)| time * time).sum();
        let slope = fit_times
            .iter()
            .map(|(time, log_s)| time * log_s)
            .sum::<f64>()
            / denominator;
        let residual = fit_times
            .iter()
            .map(|(time, log_s)| {
                let miss = log_s - slope * time;
                miss * miss
            })
            .sum::<f64>();
        let dof = (fit_times.len() as f64 - 1.0).max(1.0);
        let slope_sigma = (residual / dof / denominator).sqrt();
        let rate = -slope * model_volume;
        let rate_sigma = slope_sigma * model_volume;
        text.push_str(&format!(
            "Fit on {} samples up to {window:.0} ps: slope {slope:.6e} ± {slope_sigma:.6e} /ps, k {rate:.6e} ± {rate_sigma:.6e} Å³/ps.\n\n",
            fit_times.len()
        ));
        match diffusivity {
            Some((d_i, d_sigma)) if d_i > 0.0 => {
                let volume = rate * lattice * lattice / (DUMBBELL_ALPHA * d_i);
                let per_atom = volume / (lattice.powi(3) / 4.0);
                let rel = ((rate_sigma / rate).powi(2) + (d_sigma / d_i).powi(2)).sqrt();
                text.push_str(&format!(
                    "Table 1 quotes the Ref. [12] SIA diffusivity as {d_i:.3} ± {d_sigma:.3} Å²/ps at this temperature. First-method V_iv = {per_atom:.2} ± {:.2} Ω.\n\n",
                    per_atom * rel
                ));
            }
            _ => {
                text.push_str("No Table 1 diffusivity is recorded for this temperature, so k is not converted with it.\n\n");
            }
        }
        match measured_diffusivity(members[0].kind, members[0].gamma, *temperature, lattice) {
            Some((d_i, d_sigma, file)) if d_i > 0.0 => {
                let per_atom = rate_to_volume(rate, lattice, d_i);
                let rel = ((rate_sigma / rate).powi(2) + (d_sigma / d_i).powi(2)).sqrt();
                text.push_str(&format!(
                    "The free-dumbbell diffusivity measured with these dynamics (`{}`) is {d_i:.3} ± {d_sigma:.3} Å²/ps. With it, first-method V_iv = {per_atom:.2} ± {:.2} Ω.\n\n",
                    file.replace(".csv", ".md"),
                    per_atom * rel
                ));
            }
            _ => {
                text.push_str("No diffusivity measured with these dynamics is recorded for this temperature.\n\n");
            }
        }
    }
    text
}

fn table_diffusivity(temperature: f64) -> Option<(f64, f64)> {
    // Å²/ps from Table 1 of Mwungeli, Liu, and Demkowicz, quoted there from Ref. [12].
    match temperature as i32 {
        300 => Some((0.143, 0.006)),
        400 => Some((0.327, 0.005)),
        500 => Some((0.526, 0.006)),
        600 => Some((0.728, 0.010)),
        700 => Some((0.910, 0.012)),
        _ => None,
    }
}

struct RandomSet {
    trials: Vec<(bool, f64)>,
    temperature: f64,
    lattice: f64,
    repeats: f64,
    window: f64,
    gamma: f64,
    kind: DynamicsKind,
    dynamics: Option<(String, [f64; 4])>,
    /// Detector, confirmation window in ps, and the dynamics column index.
    detector: (Detector, f64, usize),
}

/// Whole-cell trajectories of one random-vacancy CSV. With NPH-then-NVE
/// dynamics the lattice is the mean production lattice of the trajectories
/// that reached NVE; otherwise it is the lattice column.
fn read_random_set(csv: &std::path::Path) -> RandomSet {
    let mut trials = Vec::new();
    let mut temperature = 0.0;
    let mut lattice = 0.0;
    let mut repeats = 13.0;
    let mut window = 0.0_f64;
    let mut gamma = f64::NAN;
    let mut dynamics = None;
    let mut detector = (Detector::Coordination, 0.0, 19);
    let mut production_lattices = Vec::new();
    for line in deduped_lines(csv) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 16 {
            continue;
        }
        let Ok(temp) = fields[0].parse::<f64>() else {
            continue;
        };
        let Ok(a) = fields[1].parse::<f64>() else {
            continue;
        };
        let Ok(rep) = fields[2].parse::<f64>() else {
            continue;
        };
        let Ok(span) = fields[13].parse::<f64>() else {
            continue;
        };
        let Ok(captured) = fields[14].parse::<u8>() else {
            continue;
        };
        let Ok(time_ps) = fields[15].parse::<f64>() else {
            continue;
        };
        temperature = temp;
        lattice = a;
        repeats = rep;
        window = window.max(span);
        gamma = fields[12].parse().unwrap_or(f64::NAN);
        dynamics = dynamics_columns(&fields, row_detector(&fields).2);
        detector = row_detector(&fields);
        if let Some((_, [equil_ps, ..])) = &dynamics
            && time_ps > *equil_ps
        {
            production_lattices.push(a);
        }
        trials.push((captured == 1, time_ps));
    }
    let kind = kind_of(&dynamics);
    if kind == DynamicsKind::NphNve && !production_lattices.is_empty() {
        lattice = production_lattices.iter().sum::<f64>() / production_lattices.len() as f64;
    }
    RandomSet {
        trials,
        temperature,
        lattice,
        repeats,
        window,
        gamma,
        kind,
        dynamics,
        detector,
    }
}

fn random_survival_note(csv: &std::path::Path) -> String {
    let RandomSet {
        trials,
        temperature,
        lattice,
        repeats,
        window,
        gamma,
        kind,
        dynamics,
        detector,
    } = read_random_set(csv);
    let measured = measured_diffusivity(kind, gamma, temperature, lattice);
    let measured_name = measured
        .map(|(_, _, file)| file.replace(".csv", ".md"))
        .unwrap_or_else(|| "no matching campaign".to_string());
    let mut text = String::new();
    text.push_str("# Random-vacancy survival, Mishin Cu\n\n");
    text.push_str("The vacancy is the lattice site nearest the cell center. The <100> dumbbell host is drawn uniformly from the other sites of the 13³ cell, and the dumbbell axis uniformly from the three orientations. Under periodic boundaries the separation is uniform over the cell, which is the ensemble in section 2 of Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025). The fit is ln S(t) = -(k/V_model) t through the origin, sampled every 25 ps, with V_model the cell volume and S the Kaplan-Meier survival, in which an uncaptured trajectory is censored at the end of its window; with equal windows S = n(t)/N. Samples after S reaches zero are left out. V_iv/Ω = 4k / (α a D_i) with α = 2/3 and Ω = a³/4.\n\n");
    let ours = match (kind, &dynamics) {
        (DynamicsKind::NphNve, Some((_, [equil_ps, rescale_ps, window_k, piston_ps]))) => format!(
            "Our k comes from the CSV's own NPH-then-NVE dynamics: {equil_ps:.0} ps of isotropic zero-pressure NPH (Andersen piston, period {piston_ps:.1} ps), then NVE in the mean box of the second half of that stage, with velocities rescaled to the target temperature every {rescale_ps:.2} ps whenever it is off by more than {window_k:.1} K. The capture clock starts at the Maxwellian start, NPH stage included, and the fit uses the mean production lattice constant {lattice:.6} Å of the trajectories that reached NVE. The self-consistent volume divides k by the free-dumbbell diffusivity measured with those same dynamics, potential, and cell, in `{measured_name}`. Both conversions are listed."
        ),
        _ => format!(
            "Our k comes from the Langevin dynamics in the CSV, so the self-consistent volume divides it by the free-dumbbell diffusivity measured with those same dynamics, potential, and cell, in `{measured_name}`. Both conversions are listed."
        ),
    };
    text.push_str(&format!("Table 1 of that paper divides its k by the D_i in its own Table 1 column, which it quotes from Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022): SIA mean-square displacements in MD. The paper's own trajectories are NPH for 30 ps, then NVE with velocity rescaling, at its lattice constant. {ours} σ_k is the standard deviation over 400 bootstrap resamples of the trajectories. The bootstrap σ column carries only that; the total σ columns add the relative error of the D_i used in quadrature.\n\n"));
    if trials.is_empty() {
        text.push_str("No trajectories.\n");
        return text;
    }
    if detector.0 == Detector::SiteOccupancy {
        text.push_str(&format!("A trajectory is captured when every Wigner-Seitz site of the ideal lattice holds one atom at {:.0} consecutive samples, {:.2} ps; the capture time is the first of them. Sites move with the cell and the centroid drift is removed. The last two outcome columns count sites holding other than one atom at the start and at the end.\n\n", (detector.1 / SAMPLE_PS).round(), detector.1));
    }
    let table_d = table_diffusivity(temperature);
    let measured_d = measured.map(|(d, sigma, _)| (d, sigma));
    let fmt_d = |d: Option<(f64, f64)>| match d {
        Some((value, error)) => format!("{value:.3} ± {error:.3} Å²/ps"),
        None => "not available".to_string(),
    };
    text.push_str(&format!(
        "At {temperature:.0} K, Table 1 D_i = {}; measured D_i = {}.\n\n",
        fmt_d(table_d),
        fmt_d(measured_d)
    ));
    let convert = |rate: f64, rate_sigma: f64, d: Option<(f64, f64)>| -> (f64, f64, f64) {
        match d {
            Some((d_i, d_sigma)) if d_i > 0.0 => {
                let volume = rate_to_volume(rate, lattice, d_i);
                let boot = volume.abs() * rate_sigma / rate.abs();
                let total =
                    volume.abs() * ((rate_sigma / rate).powi(2) + (d_sigma / d_i).powi(2)).sqrt();
                (volume, boot, total)
            }
            _ => (f64::NAN, f64::NAN, f64::NAN),
        }
    };
    let caps = [300.0_f64, 600.0, 900.0, 1200.0, 1500.0];
    text.push_str("| t_max (ps) | captured | N | k (Å³/ps) | σ_k | V_iv/Ω, Table 1 D_i | bootstrap σ | total σ | V_iv/Ω, measured D_i | total σ |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    let mut rows: Vec<(f64, f64, f64)> = Vec::new();
    for cap in caps.iter().copied().filter(|cap| *cap <= window + 1.0e-6) {
        let (rate, rate_sigma) = bootstrap_rate(&trials, cap, lattice, repeats);
        let (v_table, boot_table, total_table) = convert(rate, rate_sigma, table_d);
        let (v_measured, _, total_measured) = convert(rate, rate_sigma, measured_d);
        text.push_str(&format!(
            "| {cap:.0} | {} | {} | {rate:.2} | {rate_sigma:.2} | {v_table:.2} | {boot_table:.2} | {total_table:.2} | {v_measured:.2} | {total_measured:.2} |\n",
            trials
                .iter()
                .filter(|(hit, time)| *hit && *time <= cap)
                .count(),
            trials.len()
        ));
        rows.push((cap, rate, rate_sigma));
    }
    if let Some((cap, rate, rate_sigma)) = rows.last().copied() {
        let (v_table, boot_table, total_table) = convert(rate, rate_sigma, table_d);
        let (v_measured, _, total_measured) = convert(rate, rate_sigma, measured_d);
        text.push_str(&format!(
            "\nAt the longest window, {cap:.0} ps, k = {rate:.2} ± {rate_sigma:.2} Å³/ps. With the Table 1 D_i, V_iv = {v_table:.2} ± {boot_table:.2} Ω (bootstrap) or ± {total_table:.2} Ω (with D_i). With the measured D_i, V_iv = {v_measured:.2} ± {total_measured:.2} Ω.\n"
        ));
        if let Some((k_ml, low, high, captures)) = censored_rate(&trials, cap, lattice, repeats) {
            let volumes = |d: Option<(f64, f64)>| match d {
                Some((d_i, _)) if d_i > 0.0 => format!(
                    "{:.2} Ω (95 % {:.2} to {:.2})",
                    rate_to_volume(k_ml, lattice, d_i),
                    rate_to_volume(low, lattice, d_i),
                    rate_to_volume(high, lattice, d_i)
                ),
                _ => "not available".to_string(),
            };
            text.push_str(&format!(
                "The constant-hazard maximum likelihood over the same {cap:.0} ps, with every uncaptured trajectory censored at the end of its window, is k = {k_ml:.2} Å³/ps from {captures} captures in {} trajectories, 95 % interval {low:.2} to {high:.2} from ln k ± 1.96/√captures. V_iv is {} with the Table 1 D_i and {} with the measured D_i; these intervals leave out the D_i error.\n",
                trials.len(),
                volumes(table_d),
                volumes(measured_d),
            ));
        }
        if let (Some((k_mean, k_sigma, mean_time, mean_sigma)), Some((d_i, d_sigma)), Some(a_p)) = (
            table_volume(temperature),
            table_d,
            table_lattice(temperature),
        ) {
            // V Ω α D / a² with Ω = a³/4 undoes the paper's own conversion.
            let k_paper = k_mean * a_p * DUMBBELL_ALPHA * d_i / 4.0;
            let k_paper_sigma =
                k_paper * ((k_sigma / k_mean).powi(2) + (d_sigma / d_i).powi(2)).sqrt();
            let ratio = rate / k_paper;
            let ratio_sigma =
                ratio * ((rate_sigma / rate).powi(2) + (k_paper_sigma / k_paper).powi(2)).sqrt();
            text.push_str(&format!(
                "Table 1 at this temperature, 8788 atoms, is {k_mean:.0} ± {k_sigma:.0} Ω (k-method) and {mean_time:.0} ± {mean_sigma:.0} Ω (mean-time method). The k-method value with the Table 1 D_i and the paper's lattice constant {a_p:.3} Å implies k = {k_paper:.2} ± {k_paper_sigma:.2} Å³/ps. Our longest-window k over that is {ratio:.3} ± {ratio_sigma:.3}. The Table 1-D_i volume minus the k-method value is {:.2} Ω; the measured-D_i volume minus it is {:.2} Ω.\n",
                v_table - k_mean,
                v_measured - k_mean,
            ));
        }
        text.push('\n');
    }
    text.push_str("Table 1, 8788 atoms, k-method then mean-time method: 300 K 254 ± 21 and 286 ± 28; 500 K 185 ± 23 and 161 ± 15; 700 K 203 ± 10 and 210 ± 21. All use the Ref. [12] D_i of Table 1: 0.143, 0.526, and 0.910 Å²/ps.\n");
    text
}

fn table_volume(temperature: f64) -> Option<(f64, f64, f64, f64)> {
    // k-method, its uncertainty, mean-time method, its uncertainty. Table 1, 8788 atoms.
    match temperature as i32 {
        300 => Some((254.0, 21.0, 286.0, 28.0)),
        500 => Some((185.0, 23.0, 161.0, 15.0)),
        700 => Some((203.0, 10.0, 210.0, 21.0)),
        _ => None,
    }
}

fn bootstrap_rate(trials: &[(bool, f64)], t_max: f64, lattice: f64, repeats: f64) -> (f64, f64) {
    let point = |sample: &[(bool, f64)]| survival_rate(sample, t_max, lattice, repeats);
    let center = point(trials);
    let mut rng = SplitRng::new(0xB007);
    let mut values = Vec::with_capacity(400);
    for _ in 0..400 {
        let mut sample = Vec::with_capacity(trials.len());
        for _ in 0..trials.len() {
            let index = (rng.u64() as usize) % trials.len();
            sample.push(trials[index]);
        }
        values.push(point(&sample));
    }
    let mean = values.iter().copied().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let miss = value - mean;
            miss * miss
        })
        .sum::<f64>()
        / values.len() as f64;
    (center, variance.sqrt())
}

/// V_iv/Ω = k a² / (α D_i Ω) with Ω = a³/4.
fn rate_to_volume(rate: f64, lattice: f64, d_i: f64) -> f64 {
    rate * lattice * lattice / (DUMBBELL_ALPHA * d_i) / (lattice.powi(3) / 4.0)
}

#[cfg(test)]
fn survival_volume(
    trials: &[(bool, f64)],
    t_max: f64,
    lattice: f64,
    repeats: f64,
    d_i: f64,
) -> f64 {
    rate_to_volume(survival_rate(trials, t_max, lattice, repeats), lattice, d_i)
}

/// Kaplan-Meier survival at `time`. An uncaptured trial's time is the end of
/// its window, so it stays at risk until then and drops out uncounted.
fn kaplan_meier(trials: &[(bool, f64)], time: f64) -> f64 {
    let mut captures: Vec<f64> = trials
        .iter()
        .filter(|(hit, when)| *hit && *when <= time)
        .map(|(_, when)| *when)
        .collect();
    captures.sort_by(f64::total_cmp);
    captures.dedup();
    captures.iter().fold(1.0, |survival, &at| {
        let at_risk = trials.iter().filter(|(_, when)| *when >= at).count();
        let died = trials
            .iter()
            .filter(|(hit, when)| *hit && *when == at)
            .count();
        survival * (1.0 - died as f64 / at_risk as f64)
    })
}

/// k in Å³/ps from the through-origin fit of ln S, with S the Kaplan-Meier
/// survival sampled every 25 ps up to the last sample where S > 0.
fn survival_rate(trials: &[(bool, f64)], t_max: f64, lattice: f64, repeats: f64) -> f64 {
    let mut times = Vec::new();
    let step = 25.0_f64;
    let mut time = step;
    while time <= t_max + 1.0e-9 {
        let survival = kaplan_meier(trials, time);
        if survival <= 0.0 {
            break;
        }
        times.push((time, survival.ln()));
        time += step;
    }
    if times.is_empty() {
        return f64::NAN;
    }
    let denominator: f64 = times.iter().map(|(time, _)| time * time).sum();
    let slope = times.iter().map(|(time, log_s)| time * log_s).sum::<f64>() / denominator;
    -slope * (repeats * lattice).powi(3)
}

/// Constant-hazard maximum likelihood with right censoring at `t_max` or the
/// trial's own window: k = V_model × captures / Σ min(t_i, t_max), in Å³/ps.
/// Returns k, the 95 % interval from ln k ± 1.96/√captures, and the capture
/// count; no captures gives no estimate.
fn censored_rate(
    trials: &[(bool, f64)],
    t_max: f64,
    lattice: f64,
    repeats: f64,
) -> Option<(f64, f64, f64, usize)> {
    let captures = trials
        .iter()
        .filter(|(hit, when)| *hit && *when <= t_max)
        .count();
    let exposure: f64 = trials.iter().map(|(_, when)| when.min(t_max)).sum();
    if captures == 0 || exposure <= 0.0 {
        return None;
    }
    let rate = captures as f64 / exposure * (repeats * lattice).powi(3);
    let spread = (1.96 / (captures as f64).sqrt()).exp();
    Some((rate, rate / spread, rate * spread, captures))
}

fn deduped_lines(csv: &std::path::Path) -> Vec<String> {
    let mut best: BTreeMap<String, (f64, String)> = BTreeMap::new();
    let Ok(file) = File::open(csv) else {
        return Vec::new();
    };
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 16 {
            continue;
        }
        let Ok(window) = fields[13].parse::<f64>() else {
            continue;
        };
        let key = format!("{},{},{},{}", fields[0], fields[4], fields[8], fields[10]);
        if best
            .get(&key)
            .map(|(kept, _)| window >= *kept)
            .unwrap_or(true)
        {
            best.insert(key, (window, line));
        }
    }
    best.into_values().map(|(_, line)| line).collect()
}

fn summarize(csv: &std::path::Path) -> String {
    if csv
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.contains("random"))
    {
        return random_survival_note(csv);
    }
    let mut groups: BTreeMap<(u64, u64, String), Group> = BTreeMap::new();
    for line in deduped_lines(csv) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 19 {
            continue;
        }
        let Ok(temperature) = fields[0].parse::<f64>() else {
            continue;
        };
        let Ok(shell) = fields[4].parse::<f64>() else {
            continue;
        };
        let Ok(sites) = fields[9].parse::<f64>() else {
            continue;
        };
        let Ok(window) = fields[13].parse::<f64>() else {
            continue;
        };
        let Ok(captured) = fields[14].parse::<f64>() else {
            continue;
        };
        let key = (
            temperature.to_bits(),
            shell.to_bits(),
            fields[8].to_string(),
        );
        let group = groups.entry(key).or_insert_with(|| Group {
            temperature,
            window,
            shell,
            orientation: fields[8].to_string(),
            sites,
            captured: 0.0,
            trials: 0.0,
        });
        group.captured += captured;
        group.trials += 1.0;
        if window > group.window {
            group.window = window;
        }
    }
    let mut by_temp: BTreeMap<u64, Vec<&Group>> = BTreeMap::new();
    for group in groups.values() {
        by_temp
            .entry(group.temperature.to_bits())
            .or_default()
            .push(group);
    }
    let mut text = String::new();
    text.push_str("# Direct-MD vacancy-SIA volume, Mishin Cu\n\n");
    text.push_str("Independent NVT trajectories of one vacancy and one <100> dumbbell on the Mishin EAM1 evaluated by `tests/mishin_cu.rs`. The driver is `examples/langevin_nvt.rs`. The cell is 13×13×13 conventional cells (8788 atoms) at the branch's NPT lattice constant. Capture is the paper's 12-fold test: coordination cutoff 3.1 Å, sampled every 0.25 ps (Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025)). The shell sum counts recombined sites, with each orientation owning one third of its shell. The whole-cell survival curve is a separate run whose output name contains `random`.\n\n");
    text.push_str("The clock starts at the ideal dumbbell (offset 1.025 Å) with Maxwellian velocities. There is no separate NPH equilibration. The thermostat is the Langevin velocity Verlet in `npt_lattice`, without the barostat. Friction and timestep are the columns `gamma_per_ps` and `dt_fs`.\n\n");
    text.push_str("A 3 ps control at 500 K, 2 fs, and gamma = 1/ps, sampled every 0.25 ps, fixes what a zero in the 12-fold count means. The perfect 13³ crystal started at 0 atoms off 12-fold coordination and then fluctuated: minimum 0, mean 6.2, maximum 14, with 4 of 12 samples exactly 0. The nearest shell, a <100> dumbbell, started at 13 and fell into that same cloud (minimum 0, mean 5.7, maximum 16, 2 of 12 samples exactly 0). The outermost shell inside 10.8 Å started at 20 and stayed separated (minimum 22, mean 32.0, maximum 47, 0 of 12 samples exactly 0). Capture in the table is the first sample whose off-12 count is 0.\n\n");
    text.push_str("Binomial standard error of a capture probability is sqrt(p (1-p) / N). The volume error adds the orientation contributions in quadrature. A shell whose trajectories all agree has a binomial error of zero.\n\n");
    if groups.is_empty() {
        text.push_str("No trajectory rows were found.\n");
        return text;
    }
    for members in by_temp.values() {
        let temperature = members[0].temperature;
        let window = members.iter().map(|group| group.window).fold(0.0, f64::max);
        text.push_str(&format!("## {temperature:.0} K, window {window:.2} ps\n\n"));
        text.push_str("| shell (Å) | orientation | sites | captured | N | p | binomial σ |\n");
        text.push_str("|---:|:---:|---:|---:|---:|---:|---:|\n");
        let mut shells: BTreeMap<u64, Vec<&Group>> = BTreeMap::new();
        for group in members {
            shells.entry(group.shell.to_bits()).or_default().push(group);
        }
        let mut volume = 0.0;
        let mut variance = 0.0;
        let mut used = 0usize;
        let mut skipped = Vec::new();
        for group in members {
            let p = group.captured / group.trials;
            let sigma = (p * (1.0 - p) / group.trials).sqrt();
            text.push_str(&format!(
                "| {shell:.4} | {orient} | {sites:.0} | {cap:.0} | {n:.0} | {p:.4} | {sigma:.4} |\n",
                shell = group.shell,
                orient = group.orientation,
                sites = group.sites,
                cap = group.captured,
                n = group.trials,
                p = p,
                sigma = sigma,
            ));
        }
        text.push('\n');
        for orient_groups in shells.values() {
            let labels: BTreeMap<&str, &Group> = orient_groups
                .iter()
                .map(|group| (group.orientation.as_str(), *group))
                .collect();
            if !labels.contains_key("100")
                || !labels.contains_key("010")
                || !labels.contains_key("001")
            {
                skipped.push(orient_groups[0].shell);
                continue;
            }
            used += 1;
            for group in labels.values() {
                let p = group.captured / group.trials;
                let share = group.sites / 3.0;
                volume += share * p;
                variance += share * share * p * (1.0 - p) / group.trials;
            }
        }
        text.push_str(&format!(
            "Shell sum = {volume:.2} ± {sigma:.2} Ω over {used} shells with all three orientations. This counts sites that recombined inside the window. It is not the survival-curve volume.\n\n",
            sigma = variance.sqrt(),
        ));
        if !skipped.is_empty() {
            text.push_str("Shells left out because an orientation is missing:");
            for shell in skipped {
                text.push_str(&format!(" {shell:.4}"));
            }
            text.push_str(".\n\n");
        }
        if (temperature - 500.0).abs() < 1.0 {
            text.push_str("Commit 0a5b8661 counts eight translation channels of a <100> dumbbell and turns the axis on a hop. benchmarks/defect_pair_run12_volume.csv gives the 13³ first-passage volume as 202.4328 ± 5.1390 Ω at 300 K, 187.6200 ± 4.6448 Ω at 500 K, and 204.5416 ± 3.6445 Ω at 700 K. Table 1 of Mwungeli, Liu and Demkowicz, 8788 atoms at 500 K, is 185 ± 23 Ω (k-method) and 161 ± 15 Ω (mean-time method).\n\n");
        }
        if (temperature - 300.0).abs() < 1.0 {
            text.push_str("Table 1 at 300 K, 8788 atoms, is 254 ± 21 Ω (k-method) and 286 ± 28 Ω (mean-time method).\n\n");
        }
        if (temperature - 700.0).abs() < 1.0 {
            text.push_str("Table 1 at 700 K, 8788 atoms, is 203 ± 10 Ω (k-method) and 210 ± 21 Ω (mean-time method).\n\n");
        }
    }
    text.push_str(&first_method_report(csv));
    text.push_str("Raw per-trajectory outcomes are the CSV this note was written from.\n");
    text
}

fn defect_centroid(
    positions: &[f64],
    lengths: [f64; 3],
    previous: Option<[f64; 3]>,
) -> Option<[f64; 3]> {
    let n = positions.len() / 3;
    let mut best_gap = f64::MAX;
    let mut best_follow = f64::MAX;
    let mut pair = [0.0; 3];
    let cutoff = 2.45;
    let cutoff2 = cutoff * cutoff;
    let nside = lengths.map(|length| ((length / cutoff).floor() as i32).max(1));
    let bins_n = (nside[0] * nside[1] * nside[2]) as usize;
    let mut members = vec![Vec::new(); bins_n];
    for i in 0..n {
        let bin = bin_of(
            [
                positions[3 * i] / lengths[0],
                positions[3 * i + 1] / lengths[1],
                positions[3 * i + 2] / lengths[2],
            ],
            nside,
        );
        members[bin].push(i);
    }
    for i in 0..n {
        let ri = [positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]];
        let home = [
            (ri[0] / lengths[0]).rem_euclid(1.0) * nside[0] as f64,
            (ri[1] / lengths[1]).rem_euclid(1.0) * nside[1] as f64,
            (ri[2] / lengths[2]).rem_euclid(1.0) * nside[2] as f64,
        ];
        let (bins, count) = neighbor_bins(home, nside);
        for bin in bins.into_iter().take(count) {
            for &j in &members[bin] {
                if j <= i {
                    continue;
                }
                let mut d = [
                    positions[3 * j] - ri[0],
                    positions[3 * j + 1] - ri[1],
                    positions[3 * j + 2] - ri[2],
                ];
                for axis in 0..3 {
                    d[axis] -= lengths[axis] * (d[axis] / lengths[axis]).round();
                }
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 >= cutoff2 || r2 < 1.6 * 1.6 {
                    continue;
                }
                let bond = r2.sqrt();
                let center = [ri[0] + 0.5 * d[0], ri[1] + 0.5 * d[1], ri[2] + 0.5 * d[2]];
                let follow = if let Some(previous) = previous {
                    let mut distance = 0.0;
                    for axis in 0..3 {
                        let mut delta = center[axis] - previous[axis];
                        delta -= lengths[axis] * (delta / lengths[axis]).round();
                        distance += delta * delta;
                    }
                    distance
                } else {
                    0.0
                };
                // The dumbbell bond is ~2.05 Å. A nearest-neighbor pair is ~a/√2
                // and only looks short when it vibrates. Prefer the bond length,
                // then continuity with the previous centre.
                let score = (bond - 2.05) * (bond - 2.05) + 0.02 * follow;
                let better = score < best_follow;
                if better {
                    best_gap = r2;
                    best_follow = score;
                    pair = center;
                }
            }
        }
    }
    (best_gap < f64::MAX).then_some(pair)
}

fn isolated_dumbbell(repeats: usize, a: f64) -> (Vec<f64>, [[f64; 3]; 3]) {
    let (flat, cell) = fcc(repeats, a);
    let n = flat.len() / 3;
    let mut points = Vec::with_capacity(n);
    for atom in 0..n {
        points.push([flat[3 * atom], flat[3 * atom + 1], flat[3 * atom + 2]]);
    }
    let center = [cell[0][0] * 0.5, cell[1][1] * 0.5, cell[2][2] * 0.5];
    let host = nearest(&points, center, None);
    let host_pos = points[host];
    let mut values = Vec::with_capacity((n + 1) * 3);
    for (index, point) in points.iter().enumerate() {
        if index != host {
            values.extend(point);
        }
    }
    values.extend([host_pos[0] - DUMBBELL_OFFSET_A, host_pos[1], host_pos[2]]);
    values.extend([host_pos[0] + DUMBBELL_OFFSET_A, host_pos[1], host_pos[2]]);
    (values, cell)
}

const SIA_D_HEADER: &str = "temperature_K,lattice_A,repeats,n_atoms,dt_fs,gamma_per_ps,seed,block,tau_ps,dx_A,dy_A,dz_A,r2_A2,site_changes,unresolved_samples";
const SIA_SAMPLE_PS: f64 = 0.1;

/// Nearest fcc site in half-lattice units: integer coordinates with an even sum.
fn fcc_site(u: [f64; 3]) -> [i64; 3] {
    let mut site = u.map(|value| value.round() as i64);
    if (site[0] + site[1] + site[2]).rem_euclid(2) != 0 {
        let mut worst = 0;
        let mut miss = -1.0;
        for axis in 0..3 {
            let gap = (u[axis] - site[axis] as f64).abs();
            if gap > miss {
                miss = gap;
                worst = axis;
            }
        }
        site[worst] += if u[worst] > site[worst] as f64 { 1 } else { -1 };
    }
    site
}

/// The one doubly occupied Wigner-Seitz cell, in half-lattice units wrapped to the cell.
///
/// `shift` is the centre-of-mass displacement since time zero; it is removed
/// so a drifting crystal keeps its sites. `None` if the occupancy is not
/// exactly one cell with two atoms and none empty.
fn dumbbell_site(
    positions: &[f64],
    shift: [f64; 3],
    a: f64,
    n_half: i64,
    occupancy: &mut Vec<u8>,
) -> Option<[i64; 3]> {
    let span = n_half as usize;
    occupancy.clear();
    occupancy.resize(span * span * span, 0);
    for atom in 0..positions.len() / 3 {
        let u = [0, 1, 2].map(|axis| 2.0 * (positions[3 * atom + axis] - shift[axis]) / a);
        let site = fcc_site(u).map(|value| value.rem_euclid(n_half) as usize);
        let index = (site[0] * span + site[1]) * span + site[2];
        occupancy[index] = occupancy[index].saturating_add(1);
    }
    let mut found = None;
    for ix in 0..span {
        for iy in 0..span {
            for iz in 0..span {
                if (ix + iy + iz) % 2 != 0 {
                    continue;
                }
                match occupancy[(ix * span + iy) * span + iz] {
                    1 => {}
                    2 if found.is_none() => found = Some([ix as i64, iy as i64, iz as i64]),
                    _ => return None,
                }
            }
        }
    }
    found
}

fn center_of_mass(positions: &[f64]) -> [f64; 3] {
    let n = (positions.len() / 3).max(1) as f64;
    let mut sum = [0.0; 3];
    for atom in 0..positions.len() / 3 {
        for axis in 0..3 {
            sum[axis] += positions[3 * atom + axis];
        }
    }
    sum.map(|value| value / n)
}

fn sia_block_ps(temperature: f64) -> f64 {
    if let Ok(text) = env::var("MD_SIA_TAU")
        && let Ok(value) = text.parse::<f64>()
    {
        return value;
    }
    match temperature as i32 {
        t if t <= 300 => 40.0,
        t if t <= 500 => 20.0,
        _ => 10.0,
    }
}

/// One free <100> dumbbell under `spec`'s dynamics and the dumbbell-site
/// displacement over consecutive production blocks of `tau_ps`.
///
/// Sites are read on the ideal lattice of the current cell, so a barostat
/// stage moves the sites with the box; the centre-of-mass reference is
/// scaled the same way.
fn sia_d_trajectory(
    spec: &RunSpec,
    temperature: f64,
    seed_index: u64,
    tau_ps: f64,
    blocks: usize,
) -> Vec<String> {
    let start_lattice = measured_lattice(temperature);
    let (positions, cell) = isolated_dumbbell(spec.repeats, start_lattice);
    let n_atoms = positions.len() / 3;
    let n_half = 2 * spec.repeats as i64;
    let mut potential = MishinCu::new();
    let dynamics = spec.dynamics(temperature);
    let seed = mix_seed(&[temperature.to_bits(), seed_index, 0xD1FF_05EE]);
    let mut state = MdState::new(&dynamics, &mut potential, positions, cell, seed);
    let step_ps = spec.dt_fs * 1.0e-3;
    let sample_steps = dynamics.steps(SIA_SAMPLE_PS).max(1);
    let equil_steps = dynamics.steps(spec.equil_ps).div_ceil(sample_steps) * sample_steps;
    let block_steps = dynamics.steps(tau_ps).div_ceil(sample_steps) * sample_steps;
    let total_steps = equil_steps + blocks * block_steps;
    let com0 = center_of_mass(&state.positions);
    let mut occupancy = Vec::new();
    let mut site = dumbbell_site(
        &state.positions,
        [0.0; 3],
        start_lattice,
        n_half,
        &mut occupancy,
    )
    .expect("the starting dumbbell occupies one site");
    let mut unwrapped = [0i64; 3];
    let mut block_start = unwrapped;
    let mut changes = 0usize;
    let mut unresolved = 0usize;
    let mut rows = Vec::with_capacity(blocks);
    while state.step < total_steps {
        state.advance(&dynamics, &mut potential);
        let step = state.step;
        if !step.is_multiple_of(sample_steps) {
            continue;
        }
        let scale = state.dilation();
        let lattice = start_lattice * scale;
        let com = center_of_mass(&state.positions);
        let shift = [0, 1, 2].map(|axis| com[axis] - com0[axis] * scale);
        match dumbbell_site(&state.positions, shift, lattice, n_half, &mut occupancy) {
            Some(here) => {
                let mut moved = false;
                for axis in 0..3 {
                    let mut delta = (here[axis] - site[axis]).rem_euclid(n_half);
                    if delta > n_half / 2 {
                        delta -= n_half;
                    }
                    moved |= delta != 0;
                    unwrapped[axis] += delta;
                }
                changes += usize::from(moved && step > equil_steps);
                site = here;
            }
            None => {
                unresolved += usize::from(step > equil_steps);
            }
        }
        if step == equil_steps {
            block_start = unwrapped;
            changes = 0;
            unresolved = 0;
        } else if step > equil_steps && (step - equil_steps).is_multiple_of(block_steps) {
            let block = (step - equil_steps) / block_steps - 1;
            let d =
                [0, 1, 2].map(|axis| (unwrapped[axis] - block_start[axis]) as f64 * 0.5 * lattice);
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            rows.push(format!(
                "{temperature:.0},{lattice:.6},{},{n_atoms},{:.4},{:.4},{seed_index},{block},{:.4},{:.6},{:.6},{:.6},{r2:.6},{changes},{unresolved}{}",
                spec.repeats,
                spec.dt_fs,
                spec.production_gamma(),
                block_steps as f64 * step_ps,
                d[0],
                d[1],
                d[2],
                spec.dynamics_suffix(),
            ));
            block_start = unwrapped;
            changes = 0;
            unresolved = 0;
        }
    }
    rows
}

fn validation_file(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("validation")
        .join(name)
}

/// Output of a new free-dumbbell campaign at the current lattice table.
fn sia_d_default(kind: DynamicsKind, gamma_per_ps: f64) -> PathBuf {
    let name = match kind {
        DynamicsKind::Langevin if (gamma_per_ps - 0.1).abs() < 1.0e-9 => {
            "direct_md_sia_d_p0.csv".to_string()
        }
        DynamicsKind::Langevin => format!("direct_md_sia_d_p0_gamma{gamma_per_ps}.csv"),
        DynamicsKind::NphNve => "direct_md_sia_d_nph_nve.csv".to_string(),
    };
    validation_file(&name)
}

fn sia_d_path(spec: &RunSpec) -> PathBuf {
    if let Ok(over) = env::var("MD_SIA_OUT") {
        return PathBuf::from(over);
    }
    sia_d_default(spec.kind, spec.gamma_per_ps)
}

struct SiaCampaign {
    kind: DynamicsKind,
    gamma: f64,
    file: &'static str,
    label: &'static str,
}

/// The free-dumbbell campaigns the notes compare, in table order. The first
/// is the reference for ratios. The first two Langevin campaigns ran at the
/// earlier `npt_lattice` constants, which balanced only the configurational
/// stress; `direct_md_sia_d_p0.csv` runs at zero total pressure.
const SIA_D_CAMPAIGNS: [SiaCampaign; 4] = [
    SiaCampaign {
        kind: DynamicsKind::Langevin,
        gamma: 0.1,
        file: "direct_md_sia_d_long.csv",
        label: "Langevin, γ = 0.1/ps",
    },
    SiaCampaign {
        kind: DynamicsKind::Langevin,
        gamma: 0.01,
        file: "direct_md_sia_d_gamma0.01.csv",
        label: "Langevin, γ = 0.01/ps",
    },
    SiaCampaign {
        kind: DynamicsKind::NphNve,
        gamma: 0.0,
        file: "direct_md_sia_d_nph_nve.csv",
        label: "NPH then NVE, rescaled",
    },
    SiaCampaign {
        kind: DynamicsKind::Langevin,
        gamma: 0.1,
        file: "direct_md_sia_d_p0.csv",
        label: "Langevin, γ = 0.1/ps, zero total pressure",
    },
];

struct SiaBlock {
    temperature: f64,
    lattice: f64,
    seed: u64,
    block: usize,
    tau: f64,
    d: [f64; 3],
    r2: f64,
    changes: f64,
    unresolved: f64,
    /// `dynamics,equil_ps,rescale_ps,rescale_window_K,piston_ps` when present.
    dynamics: Option<(String, [f64; 4])>,
}

fn read_sia_blocks(csv: &std::path::Path) -> Vec<SiaBlock> {
    let Ok(file) = File::open(csv) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 15 {
            continue;
        }
        let parsed: Option<SiaBlock> = (|| {
            Some(SiaBlock {
                temperature: fields[0].parse().ok()?,
                lattice: fields[1].parse().ok()?,
                seed: fields[6].parse().ok()?,
                block: fields[7].parse().ok()?,
                tau: fields[8].parse().ok()?,
                d: [
                    fields[9].parse().ok()?,
                    fields[10].parse().ok()?,
                    fields[11].parse().ok()?,
                ],
                r2: fields[12].parse().ok()?,
                changes: fields[13].parse().ok()?,
                unresolved: fields[14].parse().ok()?,
                dynamics: dynamics_columns(&fields, 15),
            })
        })();
        if let Some(row) = parsed {
            rows.push(row);
        }
    }
    rows
}

struct SiaEstimate {
    d: f64,
    sigma: f64,
    d_pair: f64,
    sigma_pair: f64,
    seeds: usize,
    blocks: usize,
    tau: f64,
    lattice: f64,
    changes_per_ps: f64,
    unresolved_fraction: f64,
}

/// D = <|Δr|²>/(6τ) per trajectory, averaged over trajectories, with the
/// standard error across trajectories. The pair estimate joins consecutive
/// blocks into 2τ windows; it agrees with D when τ is past the correlated regime.
fn sia_estimate(rows: &[SiaBlock], temperature: f64) -> Option<SiaEstimate> {
    let mut by_seed: BTreeMap<u64, Vec<&SiaBlock>> = BTreeMap::new();
    for row in rows.iter().filter(|row| row.temperature == temperature) {
        by_seed.entry(row.seed).or_default().push(row);
    }
    if by_seed.is_empty() {
        return None;
    }
    let mut per_seed = Vec::new();
    let mut per_seed_pair = Vec::new();
    let mut blocks = 0usize;
    let mut changes = 0.0;
    let mut unresolved = 0.0;
    let mut time = 0.0;
    let mut tau = 0.0;
    let mut lattice = 0.0;
    for group in by_seed.values_mut() {
        group.sort_by_key(|row| row.block);
        tau = group[0].tau;
        lattice = group[0].lattice;
        let msd = group.iter().map(|row| row.r2).sum::<f64>() / group.len() as f64;
        per_seed.push(msd / (6.0 * tau));
        let pairs: Vec<f64> = group
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| {
                let d = [0, 1, 2].map(|axis| pair[0].d[axis] + pair[1].d[axis]);
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            })
            .collect();
        if !pairs.is_empty() {
            per_seed_pair.push(pairs.iter().sum::<f64>() / pairs.len() as f64 / (12.0 * tau));
        }
        blocks += group.len();
        changes += group.iter().map(|row| row.changes).sum::<f64>();
        unresolved += group.iter().map(|row| row.unresolved).sum::<f64>();
        time += group.len() as f64 * tau;
    }
    let mean_se = |values: &[f64]| -> (f64, f64) {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        if values.len() < 2 {
            return (mean, f64::NAN);
        }
        let var = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        (mean, (var / n).sqrt())
    };
    let (d, sigma) = mean_se(&per_seed);
    let (d_pair, sigma_pair) = if per_seed_pair.is_empty() {
        (f64::NAN, f64::NAN)
    } else {
        mean_se(&per_seed_pair)
    };
    let samples = time / SIA_SAMPLE_PS;
    Some(SiaEstimate {
        d,
        sigma,
        d_pair,
        sigma_pair,
        seeds: per_seed.len(),
        blocks,
        tau,
        lattice,
        changes_per_ps: changes / time,
        unresolved_fraction: unresolved / samples,
    })
}

fn dynamics_columns(fields: &[&str], first: usize) -> Option<(String, [f64; 4])> {
    if fields.len() < first + 5 {
        return None;
    }
    let mut values = [0.0; 4];
    for (slot, text) in values.iter_mut().zip(&fields[first + 1..first + 5]) {
        *slot = text.parse().ok()?;
    }
    Some((fields[first].to_string(), values))
}

fn kind_of(dynamics: &Option<(String, [f64; 4])>) -> DynamicsKind {
    match dynamics {
        Some((label, _)) if label == DynamicsKind::NphNve.label() => DynamicsKind::NphNve,
        _ => DynamicsKind::Langevin,
    }
}

/// The free-dumbbell D, its error, and its CSV for the given dynamics. A
/// Langevin campaign must also have run at `lattice`; NPH-then-NVE sets its
/// own lattice.
fn measured_diffusivity(
    kind: DynamicsKind,
    gamma_per_ps: f64,
    temperature: f64,
    lattice: f64,
) -> Option<(f64, f64, &'static str)> {
    SIA_D_CAMPAIGNS
        .iter()
        .filter(|campaign| campaign.kind == kind && (campaign.gamma - gamma_per_ps).abs() < 1.0e-9)
        .find_map(|campaign| {
            let rows = read_sia_blocks(&validation_file(campaign.file));
            let estimate = sia_estimate(&rows, temperature)?;
            let matches =
                kind == DynamicsKind::NphNve || (estimate.lattice - lattice).abs() < 1.0e-5;
            matches.then_some((estimate.d, estimate.sigma, campaign.file))
        })
}

fn defect_pair_diffusivity(temperature: f64) -> Option<(f64, f64)> {
    // Å²/ps, the free-dumbbell MD diffusivity quoted by the defect-pair rate work.
    match temperature as i32 {
        300 => Some((0.278, 0.033)),
        500 => Some((0.860, 0.071)),
        700 => Some((1.886, 0.188)),
        _ => None,
    }
}

fn table_lattice(temperature: f64) -> Option<f64> {
    // Å, the zero-pressure lattice parameter in Table 1 of Mwungeli, Liu, and Demkowicz.
    match temperature as i32 {
        300 => Some(3.631),
        400 => Some(3.637),
        500 => Some(3.643),
        600 => Some(3.649),
        700 => Some(3.655),
        _ => None,
    }
}

fn sia_d_note(csv: &std::path::Path) -> String {
    let rows = read_sia_blocks(csv);
    let mut text = String::new();
    let dynamics = rows.first().and_then(|row| row.dynamics.clone());
    match (kind_of(&dynamics), &dynamics) {
        (DynamicsKind::NphNve, Some((_, [equil_ps, rescale_ps, window_k, piston_ps]))) => {
            text.push_str("# Free dumbbell diffusivity, Mishin Cu, NPH then NVE\n\n");
            text.push_str(&format!("One <100> dumbbell in the perfect 13×13×13 cell (8789 atoms) on the Mishin EAM1 of `tests/mishin_cu.rs`, started at a lattice constant from `npt_lattice`. The dynamics follow the protocol Mwungeli, Liu, and Demkowicz describe: Maxwellian start, {equil_ps:.0} ps of isotropic zero-pressure NPH (Andersen piston, period {piston_ps:.1} ps), then NVE in the mean box of the second half of that stage. Both stages rescale velocities to the target temperature every {rescale_ps:.2} ps whenever it is off by more than {window_k:.1} K. The timestep is the column `dt_fs`; the lattice column is the production lattice constant after the NPH stage. The dumbbell"));
        }
        _ => {
            text.push_str("# Free dumbbell diffusivity, Mishin Cu\n\n");
            let current = rows.first().is_some_and(|row| {
                zero_pressure_lattices()
                    .get(&(row.temperature.round() as i64))
                    .is_some_and(|(a, _)| (a - row.lattice).abs() < 1.0e-5)
            });
            text.push_str(if current {
                "One <100> dumbbell in the perfect 13×13×13 cell (8789 atoms) on the Mishin EAM1 of `tests/mishin_cu.rs`, at the zero-total-pressure lattice constant of `direct_md_lattice.csv`."
            } else {
                "One <100> dumbbell in the perfect 13×13×13 cell (8789 atoms) on the Mishin EAM1 of `tests/mishin_cu.rs`, at the lattice constant `npt_lattice` produced while it balanced only the configurational stress, with no kinetic term."
            });
            text.push_str(" The dynamics are the V_iv dynamics: Maxwellian start, 2 ps of Langevin at gamma = 1/ps, then gamma and dt from the columns `gamma_per_ps` and `dt_fs`. The dumbbell");
        }
    }
    text.push_str(" is the one doubly occupied Wigner-Seitz cell of the ideal lattice, found every 0.1 ps after the centre-of-mass drift of the cell is removed. A sample with no single doubly occupied cell is skipped and counted in `unresolved_samples`. Each row is the dumbbell-site displacement over one production block of `tau_ps`.\n\n");
    text.push_str("D for one trajectory is the mean of |Δr|² over its blocks divided by 6τ. The quoted D is the mean over trajectories and σ is the standard error across trajectories. D(2τ) joins consecutive blocks; it matches D(τ) when τ is past the correlated part of the walk. Site changes per ps count samples at which the doubly occupied cell moved, so back-and-forth jumps inside 0.1 ps are not counted.\n\n");
    let mut temperatures: Vec<f64> = rows.iter().map(|row| row.temperature).collect();
    temperatures.sort_by(f64::total_cmp);
    temperatures.dedup();
    if temperatures.is_empty() {
        text.push_str("No trajectories.\n");
        return text;
    }
    text.push_str("| T (K) | a (Å) | seeds | blocks | τ (ps) | D (Å²/ps) | σ | σ/D | D(2τ) | σ(2τ) | site changes/ps | unresolved |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    let mut estimates = Vec::new();
    for temperature in &temperatures {
        let Some(estimate) = sia_estimate(&rows, *temperature) else {
            continue;
        };
        text.push_str(&format!(
            "| {temperature:.0} | {:.6} | {} | {} | {:.1} | {:.4} | {:.4} | {:.1}% | {:.4} | {:.4} | {:.3} | {:.2}% |\n",
            estimate.lattice,
            estimate.seeds,
            estimate.blocks,
            estimate.tau,
            estimate.d,
            estimate.sigma,
            100.0 * estimate.sigma / estimate.d,
            estimate.d_pair,
            estimate.sigma_pair,
            estimate.changes_per_ps,
            100.0 * estimate.unresolved_fraction,
        ));
        estimates.push((*temperature, estimate));
    }
    text.push_str("\n## Cross-check\n\n");
    text.push_str("Table 1 of Mwungeli, Liu, and Demkowicz lists D_i from their Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022), which took D_i from the mean-square displacement of SIAs in MD. Table 1 divides each measured rate coefficient by that D_i: V_iv = k a² / (α D_i) with α = 2/3. The defect-pair column is the free-dumbbell MD diffusivity quoted by the defect-pair rate work. z is the difference over the two errors in quadrature.\n\n");
    text.push_str("| T (K) | this run | Table 1 [12] | this/Table 1 | z | defect-pair MD | this/defect-pair | z |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (temperature, estimate) in &estimates {
        let cell = |other: Option<(f64, f64)>| -> String {
            match other {
                Some((value, error)) => format!(
                    "{value:.3} ± {error:.3} | {:.3} | {:.2}",
                    estimate.d / value,
                    (estimate.d - value) / (estimate.sigma.powi(2) + error.powi(2)).sqrt()
                ),
                None => "- | - | -".to_string(),
            }
        };
        text.push_str(&format!(
            "| {temperature:.0} | {:.3} ± {:.3} | {} | {} |\n",
            estimate.d,
            estimate.sigma,
            cell(table_diffusivity(*temperature)),
            cell(defect_pair_diffusivity(*temperature)),
        ));
    }
    text.push_str("\nRaw per-block displacements are the CSV this note was written from.\n");
    text
}

fn campaign_label(kind: DynamicsKind, gamma_per_ps: f64) -> String {
    match kind {
        DynamicsKind::Langevin => format!("Langevin, γ = {gamma_per_ps}/ps"),
        DynamicsKind::NphNve => "NPH then NVE, rescaled".to_string(),
    }
}

/// Whole-cell ensemble written by the site-occupancy detector.
fn random_sites_path(kind: DynamicsKind, temperature: f64) -> PathBuf {
    validation_file(&format!(
        "direct_md_random_sites_{}_{temperature:.0}K.csv",
        kind.label()
    ))
}

fn ratio_with_error(top: (f64, f64), bottom: (f64, f64)) -> (f64, f64, f64) {
    let ratio = top.0 / bottom.0;
    let error = ratio.abs() * ((top.1 / top.0).powi(2) + (bottom.1 / bottom.0).powi(2)).sqrt();
    let z = (top.0 - bottom.0) / (top.1.powi(2) + bottom.1.powi(2)).sqrt();
    (ratio, error, z)
}

/// Every free-dumbbell note, then the cross-dynamics comparison.
fn write_sia_d_notes() {
    for campaign in &SIA_D_CAMPAIGNS {
        let path = validation_file(campaign.file);
        if path.exists() {
            fs::write(note_path(&path), sia_d_note(&path)).expect("note");
        }
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("validation/direct_md_dynamics.md");
    fs::write(path, dynamics_note()).expect("note");
}

fn dynamics_note() -> String {
    let temperatures = [300.0_f64, 500.0, 700.0];
    let mut text = String::new();
    text.push_str("# Thermostat, lattice, and detector tests, Mishin Cu\n\n");
    text.push_str("Our free-dumbbell D under Langevin at γ = 0.1/ps is larger than the Table 1 D_i of Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025), which they quote from Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022), and our first whole-cell k at 500 K is larger than the k implied by their Table 1. Their trajectories are NPH for 30 ps, then NVE with velocity rescaling, at zero pressure. This note repeats the D measurement under that protocol, under Langevin with ten times weaker friction, and under Langevin at the zero-total-pressure lattice; and the whole-cell k with a Wigner-Seitz capture test under Langevin and NPH-then-NVE. Every campaign uses the same potential, 13³ cell, timestep, seeds, and blocks. Each number below is read from the CSV named in its row.\n\n");
    text.push_str("## Free dumbbell diffusivity\n\n");
    text.push_str("D for one trajectory is the block mean of |Δr|² over 6τ of the doubly occupied Wigner-Seitz cell; σ is the standard error across trajectories. Ratios carry both errors in quadrature, and z is the difference over the two errors in quadrature. The reference for the last two columns is the first campaign.\n\n");
    let sources: Vec<(&SiaCampaign, Vec<SiaBlock>)> = SIA_D_CAMPAIGNS
        .iter()
        .map(|campaign| (campaign, read_sia_blocks(&validation_file(campaign.file))))
        .collect();
    let reference_label = sources[0].0.label;
    text.push_str(&format!("| T (K) | dynamics | CSV | a (Å) | seeds | blocks | τ (ps) | D (Å²/ps) | σ | σ/D | D / Table 1 | D / {reference_label} | z |\n|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"));
    let mean_lattice = |rows: &[SiaBlock], temperature: f64| -> Option<(f64, f64, usize)> {
        let mut lattices: Vec<f64> = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for row in rows.iter().filter(|row| row.temperature == temperature) {
            if seen.insert(row.seed) {
                lattices.push(row.lattice);
            }
        }
        if lattices.is_empty() {
            return None;
        }
        let n = lattices.len() as f64;
        let mean = lattices.iter().sum::<f64>() / n;
        let sd =
            (lattices.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
        Some((mean, sd, lattices.len()))
    };
    for temperature in temperatures {
        let reference = sia_estimate(&sources[0].1, temperature).map(|e| (e.d, e.sigma));
        for (index, (campaign, rows)) in sources.iter().enumerate() {
            let Some(estimate) = sia_estimate(rows, temperature) else {
                continue;
            };
            let lattice = mean_lattice(rows, temperature).map_or(f64::NAN, |(a, _, _)| a);
            let table = table_diffusivity(temperature)
                .map(|t| {
                    let (ratio, error, _) = ratio_with_error((estimate.d, estimate.sigma), t);
                    format!("{ratio:.3} ± {error:.3}")
                })
                .unwrap_or_else(|| "-".to_string());
            let versus = match reference {
                Some(base) if index > 0 => {
                    let (ratio, error, z) = ratio_with_error((estimate.d, estimate.sigma), base);
                    format!("{ratio:.3} ± {error:.3} | {z:.2}")
                }
                _ => "- | -".to_string(),
            };
            text.push_str(&format!(
                "| {temperature:.0} | {} | `{}` | {lattice:.6} | {} | {} | {:.1} | {:.4} | {:.4} | {:.1}% | {table} | {versus} |\n",
                campaign.label,
                campaign.file,
                estimate.seeds,
                estimate.blocks,
                estimate.tau,
                estimate.d,
                estimate.sigma,
                100.0 * estimate.sigma / estimate.d,
            ));
        }
    }
    let mut table_low = f64::INFINITY;
    let mut table_high = f64::NEG_INFINITY;
    for (_, rows) in &sources {
        for temperature in temperatures {
            if let (Some(estimate), Some((d_i, _))) = (
                sia_estimate(rows, temperature),
                table_diffusivity(temperature),
            ) {
                table_low = table_low.min(estimate.d / d_i);
                table_high = table_high.max(estimate.d / d_i);
            }
        }
    }
    text.push('\n');
    for (campaign, rows) in sources.iter().skip(1) {
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        for temperature in temperatures {
            if let (Some(top), Some(base)) = (
                sia_estimate(rows, temperature),
                sia_estimate(&sources[0].1, temperature),
            ) {
                low = low.min(top.d / base.d);
                high = high.max(top.d / base.d);
            }
        }
        if low.is_finite() {
            text.push_str(&format!(
                "{} over {reference_label}: D ratio {low:.3} to {high:.3} across temperatures.\n",
                campaign.label
            ));
        }
    }
    if table_low.is_finite() {
        text.push_str(&format!(
            "Every campaign over Table 1: D ratio {table_low:.3} to {table_high:.3}.\n"
        ));
    }
    let table_line: Vec<String> = temperatures
        .iter()
        .filter_map(|t| table_diffusivity(*t).map(|(d, e)| format!("{t:.0} K {d:.3} ± {e:.3}")))
        .collect();
    text.push_str(&format!(
        "\nTable 1 D_i (Å²/ps): {}.\n\n",
        table_line.join("; ")
    ));
    text.push_str("## Lattice constant\n\n");
    text.push_str(&format!("The first two campaigns ran at the lattice constants `npt_lattice` in `tests/mishin_cu.rs` produced while its Berendsen loop balanced only the configurational stress from `evaluate`; that lattice carries the kinetic pressure N k_B T / V in the last column. `npt_lattice` now subtracts N k_B T / V from the tension, so it reaches zero total pressure; `direct_md_lattice` in that file writes the result to `direct_md_lattice.csv` (σ from eight blocks of the averaged half), and the driver reads its lattice constants from there. The NPH column is the mean and standard deviation over trajectories of the production lattice constant of `{}`, whose barostat uses the full virial pressure (2K - dE/dλ) / 3V. Table 1 lists the paper's zero-pressure lattice constants.\n\n| T (K) | configurational-stress lattice (Å) | zero total pressure (Å) | σ | after NPH (Å) | sd | trajectories | Table 1 (Å) | N k_B T / V at the configurational-stress lattice (GPa) |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n", sources[2].0.file));
    for temperature in temperatures {
        let old = mean_lattice(&sources[0].1, temperature);
        let nph = mean_lattice(&sources[2].1, temperature);
        let zero = zero_pressure_lattices()
            .get(&(temperature.round() as i64))
            .copied();
        let cell = |value: Option<f64>, digits: usize| {
            value.map_or("-".to_string(), |v| format!("{v:.digits$}"))
        };
        let kinetic_gpa = old
            .map(|(a, _, _)| 4.0 * BOLTZMANN_EV_PER_K * temperature / a.powi(3) * EV_PER_A3_IN_GPA);
        text.push_str(&format!(
            "| {temperature:.0} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            cell(old.map(|(a, _, _)| a), 6),
            cell(zero.map(|(a, _)| a), 6),
            cell(zero.map(|(_, e)| e), 6),
            cell(nph.map(|(a, _, _)| a), 6),
            cell(nph.map(|(_, sd, _)| sd), 6),
            nph.map_or("-".to_string(), |(_, _, n)| n.to_string()),
            cell(table_lattice(temperature), 3),
            cell(kinetic_gpa, 3),
        ));
    }
    text.push_str("\n## Whole-cell k\n\n");
    text.push_str("Random-vacancy ensembles as in `direct_md_random.md`: the through-origin fit of ln(n/N) every 25 ps up to the longest window, σ_k from 400 bootstrap resamples. V_iv/Ω = 4k / (α a D_i) with α = 2/3; the total σ adds the D_i error in quadrature. Own D is the free-dumbbell D measured with the same dynamics at the same lattice constant. The coordination detector calls a trajectory captured at the first sample at which every atom has 12 neighbours inside 3.1 Å; the site detector when every Wigner-Seitz site of the ideal lattice holds one atom for the confirmation window, dated to the first sample of that window. An ensemble with fewer trajectories than the others is a partial run of the same seeds. The Table 1 row backs k out of the k-method volume with the Table 1 D_i and the paper's lattice constant.\n\n");
    text.push_str("| T (K) | dynamics | detector | CSV | N | captured | t_max (ps) | a (Å) | k (Å³/ps) | σ_k | own D (Å²/ps) | V_iv/Ω, own D | σ | V_iv/Ω, Table 1 D | σ | k / Table 1 k |\n|---:|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    let mut ratios = Vec::new();
    for temperature in temperatures {
        let paper = match (
            table_volume(temperature),
            table_diffusivity(temperature),
            table_lattice(temperature),
        ) {
            (Some((k_mean, k_sigma, _, _)), Some((d_i, d_sigma)), Some(a_p)) => {
                let k_paper = k_mean * a_p * DUMBBELL_ALPHA * d_i / 4.0;
                let k_paper_sigma =
                    k_paper * ((k_sigma / k_mean).powi(2) + (d_sigma / d_i).powi(2)).sqrt();
                Some((k_mean, k_sigma, d_i, d_sigma, a_p, k_paper, k_paper_sigma))
            }
            _ => None,
        };
        let mut paths = Vec::new();
        if temperature == 500.0 {
            paths.push(validation_file("direct_md_random.csv"));
        }
        for kind in [DynamicsKind::Langevin, DynamicsKind::NphNve] {
            paths.push(random_sites_path(kind, temperature));
        }
        let mut rates = Vec::new();
        for path in paths {
            let set = read_random_set(&path);
            if set.trials.is_empty() || set.temperature != temperature {
                continue;
            }
            let t_max = set.window;
            let (rate, rate_sigma) = bootstrap_rate(&set.trials, t_max, set.lattice, set.repeats);
            let convert = |d: Option<(f64, f64)>| match d {
                Some((d_i, d_sigma)) => {
                    let volume = rate_to_volume(rate, set.lattice, d_i);
                    let total =
                        volume * ((rate_sigma / rate).powi(2) + (d_sigma / d_i).powi(2)).sqrt();
                    (
                        format!("{d_i:.3} ± {d_sigma:.3}"),
                        format!("{volume:.2} | {total:.2}"),
                    )
                }
                None => ("-".to_string(), "- | -".to_string()),
            };
            let own = measured_diffusivity(set.kind, set.gamma, temperature, set.lattice)
                .map(|(d, sigma, _)| (d, sigma));
            let (own_d, own_v) = convert(own);
            let (_, table_v) = convert(table_diffusivity(temperature));
            let versus = paper.map_or("-".to_string(), |(.., k_paper, k_paper_sigma)| {
                let (ratio, error, _) =
                    ratio_with_error((rate, rate_sigma), (k_paper, k_paper_sigma));
                format!("{ratio:.3} ± {error:.3}")
            });
            let label = campaign_label(set.kind, set.gamma);
            let detector = set.detector.0.label();
            text.push_str(&format!(
                "| {temperature:.0} | {label} | {detector} | `{}` | {} | {} | {t_max:.0} | {:.6} | {rate:.2} | {rate_sigma:.2} | {own_d} | {own_v} | {table_v} | {versus} |\n",
                path.file_name().and_then(|name| name.to_str()).unwrap_or(""),
                set.trials.len(),
                set.trials.iter().filter(|(hit, time)| *hit && *time <= t_max).count(),
                set.lattice,
            ));
            rates.push((label, set.detector.0, (rate, rate_sigma)));
        }
        if let Some((k_mean, k_sigma, d_i, d_sigma, a_p, k_paper, k_paper_sigma)) = paper {
            text.push_str(&format!(
                "| {temperature:.0} | Table 1, k-method | - | - | {PAPER_TRIALS} | - | - | {a_p:.3} | {k_paper:.2} | {k_paper_sigma:.2} | {d_i:.3} ± {d_sigma:.3} | {k_mean:.0} | {k_sigma:.0} | {k_mean:.0} | {k_sigma:.0} | - |\n"
            ));
        }
        let sites: Vec<_> = rates
            .iter()
            .filter(|(_, detector, _)| *detector == Detector::SiteOccupancy)
            .collect();
        if let [langevin, nph] = sites.as_slice() {
            let (ratio, error, z) = ratio_with_error(nph.2, langevin.2);
            ratios.push(format!(
                "At {temperature:.0} K with the site detector, {} over {}: k ratio {ratio:.3} ± {error:.3} (z = {z:.2}).",
                nph.0, langevin.0
            ));
        }
    }
    text.push('\n');
    for line in ratios {
        text.push_str(&line);
        text.push('\n');
    }
    let quoted: Vec<String> = temperatures
        .iter()
        .filter_map(|t| {
            table_volume(*t).map(|(k, ks, m, ms)| {
                format!("{t:.0} K {k:.0} ± {ks:.0} Ω (k-method), {m:.0} ± {ms:.0} Ω (mean-time)")
            })
        })
        .collect();
    text.push_str(&format!("\nTable 1, 8788 atoms: {}.\n", quoted.join("; ")));
    text
}

struct Timed<'a> {
    inner: &'a mut MishinCu,
    seconds: f64,
    calls: usize,
}

impl EnergyGradient for Timed<'_> {
    fn energy_gradient(
        &mut self,
        positions: &[f64],
        cell: [[f64; 3]; 3],
        gradient: &mut [f64],
    ) -> f64 {
        let start = Instant::now();
        let energy = self.inner.energy_gradient(positions, cell, gradient);
        self.seconds += start.elapsed().as_secs_f64();
        self.calls += 1;
        energy
    }

    fn dilation_derivative(&mut self, positions: &[f64], cell: [[f64; 3]; 3]) -> f64 {
        self.inner.dilation_derivative(positions, cell)
    }
}

/// `MD_BENCH_THREADS` independent 500 K Langevin trajectories of a vacancy and
/// a dumbbell in the `MD_REPEATS` cell, `MD_BENCH_STEPS` steps each, run at
/// once so the timing includes the contention of a production campaign.
fn bench_steps() {
    let threads = env_u64("MD_BENCH_THREADS", 8) as usize;
    let steps = env_u64("MD_BENCH_STEPS", 2000) as usize;
    let repeats = env_u64("MD_REPEATS", 13) as usize;
    let dt_fs = env_f64("MD_DT_FS", 10.0);
    let a = measured_lattice(500.0);
    let rows: Vec<(f64, f64, usize, usize, u64)> = thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|index| {
                scope.spawn(move || {
                    let (positions, cell) =
                        place_vacancy_dumbbell(repeats, a, [3.0 * a, 2.0 * a, a], [0.0, 0.0, 1.0]);
                    let dynamics = Dynamics {
                        kind: DynamicsKind::Langevin,
                        dt_fs,
                        temperature: 500.0,
                        mass: CU_MASS_AMU,
                        gamma_per_ps: 0.1,
                        equil_gamma_per_ps: 1.0,
                        equil_ps: 2.0,
                        rescale_ps: 0.1,
                        rescale_window_k: 0.0,
                        piston_ps: 2.0,
                        bulk_ev_a3: 0.87,
                        pressure_ev_a3: 0.0,
                    };
                    let mut potential = MishinCu::new();
                    let mut timed = Timed {
                        inner: &mut potential,
                        seconds: 0.0,
                        calls: 0,
                    };
                    let mut state =
                        MdState::new(&dynamics, &mut timed, positions, cell, 11 + index as u64);
                    let force_before = timed.seconds;
                    let calls_before = timed.calls;
                    let rebuilds_before = timed.inner.rebuilds;
                    let start = Instant::now();
                    for _ in 0..steps {
                        state.advance(&dynamics, &mut timed);
                    }
                    (
                        start.elapsed().as_secs_f64(),
                        timed.seconds - force_before,
                        timed.calls - calls_before,
                        timed.inner.rebuilds - rebuilds_before,
                        state
                            .positions
                            .iter()
                            .fold(0u64, |hash, x| hash.rotate_left(5) ^ x.to_bits()),
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });
    for (index, (wall, force, calls, rebuilds, hash)) in rows.iter().enumerate() {
        println!(
            "bench thread {index} steps {steps} ms/step {:.3} force ms/call {:.3} calls {calls} rebuilds {rebuilds} steps/rebuild {:.1} positions {hash:016x}",
            1.0e3 * wall / steps as f64,
            1.0e3 * force / *calls as f64,
            steps as f64 / (*rebuilds).max(1) as f64
        );
    }
    let mean = rows.iter().map(|row| row.0).sum::<f64>() / rows.len() as f64;
    println!(
        "bench threads {threads} mean ms/step {:.3}",
        1.0e3 * mean / steps as f64
    );
}

/// Trajectories per temperature in Mwungeli, Liu, and Demkowicz.
const PAPER_TRIALS: usize = 100;
const EV_PER_A3_IN_GPA: f64 = 160.217_663;

fn run_sia_d() {
    let spec = run_spec();
    let blocks = env_u64("MD_SIA_BLOCKS", 40) as usize;
    let path = sia_d_path(&spec);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("validation dir");
    }
    if !path.exists() {
        let mut header = File::create(&path).expect("csv");
        let extra = if spec.kind == DynamicsKind::Langevin {
            ""
        } else {
            DYNAMICS_HEADER
        };
        writeln!(header, "{SIA_D_HEADER}{extra}").unwrap();
    }
    let done: std::collections::BTreeSet<(i64, u64)> = read_sia_blocks(&path)
        .iter()
        .map(|row| (row.temperature as i64, row.seed))
        .collect();
    let mut jobs: Vec<(f64, u64, f64)> = Vec::new();
    for &temperature in &spec.temps {
        for seed in 0..spec.seeds {
            if !done.contains(&(temperature as i64, seed)) {
                jobs.push((temperature, seed, sia_block_ps(temperature)));
            }
        }
    }
    jobs.sort_by(|left, right| right.2.total_cmp(&left.2));
    println!(
        "sia_d jobs {} temps {:?} seeds {} blocks {blocks} dt_fs {} gamma {} dynamics {}",
        jobs.len(),
        spec.temps,
        spec.seeds,
        spec.dt_fs,
        spec.gamma_per_ps,
        spec.kind.label()
    );
    let file = Mutex::new(OpenOptions::new().append(true).open(&path).expect("append"));
    let cursor = AtomicUsize::new(0);
    let workers = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(jobs.len().max(1));
    let started = Instant::now();
    thread::scope(|scope| {
        for _ in 0..workers {
            let (jobs, cursor, file, spec) = (&jobs, &cursor, &file, &spec);
            scope.spawn(move || {
                loop {
                    let index = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some(&(temperature, seed, tau)) = jobs.get(index) else {
                        break;
                    };
                    let rows = sia_d_trajectory(spec, temperature, seed, tau, blocks);
                    let msd = rows
                        .iter()
                        .filter_map(|row| row.split(',').nth(12)?.parse::<f64>().ok())
                        .sum::<f64>()
                        / rows.len().max(1) as f64;
                    let mut handle = file.lock().expect("csv lock");
                    for row in &rows {
                        writeln!(handle, "{row}").unwrap();
                    }
                    println!(
                        "{}/{} T {temperature:.0} seed {seed} tau {tau} D {:.4}",
                        index + 1,
                        jobs.len(),
                        msd / (6.0 * tau)
                    );
                }
            });
        }
    });
    fs::write(note_path(&path), sia_d_note(&path)).expect("note");
    write_sia_d_notes();
    println!(
        "wrote {} in {:.1}s",
        path.display(),
        started.elapsed().as_secs_f64()
    );
}

fn main() {
    let (perfect, cell) = fcc(1, 3.615);
    let mut gradient = vec![0.0; perfect.len()];
    let energy = analytic_gradient(&perfect, cell, &mut gradient);
    let per_atom = energy / 4.0;
    let max_force = gradient.iter().map(|value| value.abs()).fold(0.0, f64::max);
    println!("cohesive_eV {per_atom:.6} max_force {max_force:.3e}");
    if (per_atom + 3.54).abs() > 5.0e-4 || max_force > 1.0e-6 {
        eprintln!("Mishin crystal check failed");
        std::process::exit(1);
    }
    if env::var("MD_BENCH").ok().as_deref() == Some("1") {
        bench_steps();
        return;
    }
    if env::var("MD_STABILITY").ok().as_deref() == Some("1") {
        let spec = run_spec();
        let a = measured_lattice(spec.temps[0]);
        let thermostat =
            Langevin::from_physical(spec.dt_fs, spec.gamma_per_ps, spec.temps[0], CU_MASS_AMU);
        let mut pot = MishinCu::new();
        let shells = shell_table(a, 10.8);
        let cases = [
            ("perfect", None),
            ("near", Some(shells[0].1)),
            ("far", Some(shells[shells.len() - 1].1)),
        ];
        for (label, shell) in cases {
            let (mut positions, cell) = if let Some(shell) = shell {
                place_vacancy_dumbbell(spec.repeats, a, shell, [1.0, 0.0, 0.0])
            } else {
                fcc(spec.repeats, a)
            };
            let mut velocities = vec![0.0; positions.len()];
            let mut rng = SplitRng::new(7);
            let sigma = (thermostat.thermal / thermostat.mass).sqrt();
            for velocity in &mut velocities {
                *velocity = rng.normal() * sigma;
            }
            zero_com(&mut velocities);
            let mut gradient = vec![0.0; positions.len()];
            pot.energy_gradient(&positions, cell, &mut gradient);
            let initial = coordination_deficit(&positions, cell);
            let sample = ((SAMPLE_PS / (spec.dt_fs * 1.0e-3)).round() as usize).max(1);
            let steps =
                ((spec.window_ps.min(20.0) / (spec.dt_fs * 1.0e-3)).round() as usize).max(1);
            let mut min_non12 = usize::MAX;
            let mut max_non12 = 0usize;
            let mut sum_non12 = 0.0;
            let mut zeros = 0usize;
            let mut samples = 0usize;
            for step in 1..=steps {
                thermostat.step(
                    &mut pot,
                    &mut positions,
                    &mut velocities,
                    &mut gradient,
                    cell,
                    &mut rng,
                );
                if step % sample == 0 {
                    let deficit = coordination_deficit(&positions, cell);
                    min_non12 = min_non12.min(deficit);
                    max_non12 = max_non12.max(deficit);
                    sum_non12 += deficit as f64;
                    zeros += usize::from(deficit == 0);
                    samples += 1;
                }
            }
            let kinetic =
                velocities.iter().map(|value| value * value).sum::<f64>() * 0.5 * CU_MASS_AMU;
            let target = 1.5 * (positions.len() / 3) as f64 * BOLTZMANN_EV_PER_K * spec.temps[0];
            println!(
                "coord {label} initial {initial} min {min_non12} mean {:.1} max {max_non12} zeros {zeros}/{samples} kinetic {kinetic:.1} target {target:.1}",
                sum_non12 / samples as f64
            );
        }
        return;
    }
    if env::var("MD_CHECK").ok().as_deref() == Some("1") {
        let mut check = MishinCu::new();
        let a = measured_lattice(500.0);
        let (big, big_cell) = fcc(13, a);
        let mut big_gradient = vec![0.0; big.len()];
        for _ in 0..2 {
            check.energy_gradient(&big, big_cell, &mut big_gradient);
        }
        let started = Instant::now();
        let calls = 10usize;
        for _ in 0..calls {
            check.energy_gradient(&big, big_cell, &mut big_gradient);
        }
        let each = started.elapsed().as_secs_f64() / calls as f64;
        println!(
            "force_call_s {each:.4} atoms {} pairs {}",
            big.len() / 3,
            check.pairs.len()
        );
        return;
    }

    if env::var("MD_SIA_D").ok().as_deref() == Some("1") {
        run_sia_d();
        return;
    }
    if env::var("MD_SIA_D_NOTE").ok().as_deref() == Some("1") {
        write_sia_d_notes();
        return;
    }
    if env::var("MD_MSD").ok().as_deref() == Some("1") {
        let spec = run_spec();
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("validation/direct_md_sia_d.csv");
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("validation dir");
        }
        let mut out = File::create(&path).expect("diffusivity");
        writeln!(
            out,
            "temperature_K,D_A2_per_ps,error_A2_per_ps,seed,window_ps,hops,channel_Hz,rejected"
        )
        .unwrap();
        for &temperature in &spec.temps {
            let lattice = measured_lattice(temperature);
            for seed in 0..spec.seeds {
                let (mut positions, cell) = isolated_dumbbell(spec.repeats, lattice);
                let mut potential = MishinCu::new();
                let production = Langevin::from_physical(
                    spec.dt_fs,
                    spec.gamma_per_ps,
                    temperature,
                    CU_MASS_AMU,
                );
                let equil = Langevin::from_physical(spec.dt_fs, 1.0, temperature, CU_MASS_AMU);
                let mut velocities = vec![0.0; positions.len()];
                let mut rng = SplitRng::new(
                    seed.wrapping_mul(0xD1FF)
                        .wrapping_add(temperature.to_bits()),
                );
                let sigma = (production.thermal / production.mass).sqrt();
                for velocity in &mut velocities {
                    *velocity = rng.normal() * sigma;
                }
                zero_com(&mut velocities);
                let mut gradient = vec![0.0; positions.len()];
                potential.energy_gradient(&positions, cell, &mut gradient);
                let lengths = [cell[0][0], cell[1][1], cell[2][2]];
                let origin = defect_centroid(&positions, lengths, None).unwrap_or([0.0; 3]);
                let mut cursor = origin;
                let mut unwrapped = origin;
                let window_ps = spec.window_ps.max(40.0);
                let steps = ((window_ps / (spec.dt_fs * 1.0e-3)).round() as usize).max(1);
                let equil_steps = (2.0 / (spec.dt_fs * 1.0e-3)).round() as usize;
                let sample_steps = ((0.1 / (spec.dt_fs * 1.0e-3)).round() as usize).max(1);
                let mut start = unwrapped;
                let mut started = false;
                let mut rejected = 0usize;
                let mut hops = 0usize;
                let mut samples_pos: Vec<[f64; 3]> = Vec::new();
                for step in 1..=steps {
                    let phase = if step <= equil_steps {
                        equil
                    } else {
                        production
                    };
                    phase.step(
                        &mut potential,
                        &mut positions,
                        &mut velocities,
                        &mut gradient,
                        cell,
                        &mut rng,
                    );
                    if step % sample_steps != 0 && step != equil_steps {
                        continue;
                    }
                    let Some(here) = defect_centroid(&positions, lengths, Some(cursor)) else {
                        continue;
                    };
                    let mut jump = 0.0;
                    for axis in 0..3 {
                        let mut delta = here[axis] - cursor[axis];
                        delta -= lengths[axis] * (delta / lengths[axis]).round();
                        jump += delta * delta;
                    }
                    let hop = jump.sqrt();
                    if hop > 3.6 {
                        rejected += 1;
                        continue;
                    }
                    if started && hop > 1.5 {
                        hops += 1;
                    }
                    for axis in 0..3 {
                        let mut delta = here[axis] - cursor[axis];
                        delta -= lengths[axis] * (delta / lengths[axis]).round();
                        unwrapped[axis] += delta;
                    }
                    cursor = here;
                    if step == equil_steps {
                        start = unwrapped;
                        started = true;
                    }
                    if started {
                        samples_pos.push(unwrapped);
                    }
                }
                let elapsed = if started { window_ps - 2.0 } else { window_ps };
                let msd = samples_pos
                    .last()
                    .map(|end| {
                        let d = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
                        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
                    })
                    .unwrap_or(0.0);
                let d_i = msd / (6.0 * elapsed);
                let blocks = 4usize;
                let block_n = (samples_pos.len() / blocks).max(1);
                let mut block_d = Vec::new();
                for block in 0..blocks {
                    let end = ((block + 1) * block_n).min(samples_pos.len());
                    let begin = block * block_n;
                    if end <= begin + 1 || begin >= samples_pos.len() {
                        continue;
                    }
                    let a0 = samples_pos[begin];
                    let a1 = samples_pos[end - 1];
                    let d = [a1[0] - a0[0], a1[1] - a0[1], a1[2] - a0[2]];
                    let block_time =
                        elapsed * (end - begin) as f64 / samples_pos.len().max(1) as f64;
                    block_d.push((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / (6.0 * block_time));
                }
                let mean = if block_d.is_empty() {
                    d_i
                } else {
                    block_d.iter().sum::<f64>() / block_d.len() as f64
                };
                let error = if block_d.len() < 2 {
                    0.0
                } else {
                    let var = block_d
                        .iter()
                        .map(|value| (value - mean).powi(2))
                        .sum::<f64>()
                        / (block_d.len() as f64 * (block_d.len() - 1) as f64);
                    var.sqrt()
                };
                let channel = hops as f64 / elapsed * 1.0e12 / 8.0;
                writeln!(out, "{temperature:.0},{d_i:.8e},{error:.8e},{seed},{elapsed:.2},{hops},{channel:.6e},{rejected}").unwrap();
                println!(
                    "msd T {temperature:.0} seed {seed} D {d_i:.4} ± {error:.4} hops {hops} channel {channel:.3e} rejected {rejected}"
                );
            }
        }
        return;
    }
    if env::var("MD_NOTE_ONLY").ok().as_deref() == Some("1") {
        let path = output_path();
        fs::write(note_path(&path), summarize(&path)).expect("note");
        return;
    }

    let spec = run_spec();
    let jobs = jobs_for(&spec);
    let path = output_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("validation dir");
    }
    if !path.exists() {
        let mut header = File::create(&path).expect("csv");
        writeln!(header, "{}", spec.header()).unwrap();
    }
    let done = settled_keys(&path, spec.window_ps);
    let pending: Vec<&Job> = jobs
        .iter()
        .filter(|job| {
            !done.contains_key(&format!(
                "{:.0},{:.6},{},{}",
                job.temperature, job.shell_r, job.orientation, job.seed_index
            ))
        })
        .collect();
    println!(
        "jobs {} pending {} temps {:?} seeds {} window_ps {} dt_fs {} rmax {}",
        jobs.len(),
        pending.len(),
        spec.temps,
        spec.seeds,
        spec.window_ps,
        spec.dt_fs,
        spec.r_max_500
    );
    let file = Mutex::new(OpenOptions::new().append(true).open(&path).expect("append"));
    let cursor = AtomicUsize::new(0);
    let workers = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(pending.len().max(1));
    let started = Instant::now();
    thread::scope(|scope| {
        for _ in 0..workers {
            let pending = &pending;
            let cursor = &cursor;
            let file = &file;
            let spec = &spec;
            scope.spawn(move || {
                let mut potential = MishinCu::new();
                loop {
                    let index = cursor.fetch_add(1, Ordering::Relaxed);
                    if index >= pending.len() {
                        break;
                    }
                    let job = pending[index];
                    let (n_atoms, outcome) = run_job(&mut potential, spec, job);
                    let line = format_row(spec, job, n_atoms, &outcome);
                    let mut handle: MutexGuard<'_, File> = file.lock().expect("csv lock");
                    writeln!(handle, "{line}").unwrap();
                    println!(
                        "{}/{} T {:.0} shell {:.3} {} seed {} captured {} t {:.3} ps defects {} -> {}",
                        index + 1,
                        pending.len(),
                        job.temperature,
                        job.shell_r,
                        job.orientation,
                        job.seed_index,
                        u8::from(outcome.captured),
                        outcome.time_ps,
                        outcome.initial_score,
                        outcome.final_score
                    );
                }
            });
        }
    });
    let note = summarize(&path);
    fs::write(note_path(&path), note).expect("note");
    println!(
        "wrote {} in {:.1}s",
        path.display(),
        started.elapsed().as_secs_f64()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Harmonic {
        k: f64,
    }

    impl EnergyGradient for Harmonic {
        fn energy_gradient(
            &mut self,
            positions: &[f64],
            _cell: [[f64; 3]; 3],
            gradient: &mut [f64],
        ) -> f64 {
            let mut energy = 0.0;
            for (slope, position) in gradient.iter_mut().zip(positions.iter()) {
                *slope = self.k * position;
                energy += 0.5 * self.k * position * position;
            }
            energy
        }
    }

    #[test]
    fn mishin_cohesive_energy_matches_the_copper_test() {
        let (positions, cell) = fcc(1, 3.615);
        let mut gradient = vec![0.0; positions.len()];
        let energy = analytic_gradient(&positions, cell, &mut gradient);
        assert!((energy / 4.0 + 3.54).abs() < 5.0e-4, "{}", energy / 4.0);
        assert!(gradient.iter().all(|value| value.abs() < 1.0e-6));
    }

    #[test]
    fn tabulated_cell_list_matches_the_analytic_gradient() {
        let (mut positions, cell) = fcc(6, 3.638666);
        positions[0] += 0.07;
        positions[4] -= 0.04;
        let mut analytic = vec![0.0; positions.len()];
        let mut table = vec![0.0; positions.len()];
        let energy_a = analytic_gradient(&positions, cell, &mut analytic);
        let mut pot = MishinCu::new();
        let energy_t = pot.energy_gradient(&positions, cell, &mut table);
        assert!(
            (energy_a - energy_t).abs() < 1.0e-6,
            "{energy_a} {energy_t}"
        );
        for (left, right) in analytic.iter().zip(table.iter()) {
            assert!((left - right).abs() < 1.0e-6, "{left} {right}");
        }
    }

    #[test]
    fn velocity_verlet_holds_the_harmonic_energy() {
        let mut potential = Harmonic { k: 1.0 };
        let cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let mut positions = vec![1.0, 0.0, 0.0];
        let mut velocities = vec![0.0, 0.0, 0.0];
        let mut gradient = vec![0.0; 3];
        potential.energy_gradient(&positions, cell, &mut gradient);
        let thermostat = Langevin {
            dt: 0.01,
            gamma: 0.0,
            mass: 1.0,
            thermal: 0.0,
        };
        let mut rng = SplitRng::new(1);
        let mut worst = 0.0_f64;
        for _ in 0..2_000 {
            let energy = thermostat.step(
                &mut potential,
                &mut positions,
                &mut velocities,
                &mut gradient,
                cell,
                &mut rng,
            );
            let kinetic = 0.5 * velocities.iter().map(|v| v * v).sum::<f64>();
            worst = worst.max((energy + kinetic - 0.5).abs());
        }
        assert!(worst < 1.0e-4, "{worst}");
    }

    #[test]
    fn random_separations_cover_the_cell_past_the_shell_cutoff() {
        let spec = RunSpec {
            temps: vec![500.0],
            seeds: 192,
            window_ps: 1500.0,
            dt_fs: 10.0,
            gamma_per_ps: 0.1,
            r_max_500: 10.8,
            repeats: 13,
            random_vacancy: true,
            kind: DynamicsKind::Langevin,
            equil_ps: 2.0,
            rescale_ps: 0.1,
            rescale_window_k: 0.0,
            piston_ps: 2.0,
            bulk_ev_a3: 0.87,
            detector: Detector::SiteOccupancy,
            confirm_ps: 1.0,
        };
        let jobs = random_jobs(&spec);
        assert_eq!(jobs.len(), 192);
        let far = jobs.iter().filter(|job| job.shell_r > 10.8).count();
        assert!(far > 100, "{far}");
        let lattice = measured_lattice(500.0);
        let half_diagonal = 0.5 * 13.0 * lattice * 3.0_f64.sqrt();
        let max_r = jobs.iter().map(|job| job.shell_r).fold(0.0, f64::max);
        assert!(max_r > 20.0 && max_r <= half_diagonal + 1.0e-6, "{max_r}");
    }

    #[test]
    fn survival_volume_recovers_an_exponential_rate() {
        let lattice = 3.638666;
        let repeats = 13.0;
        let d_i = 0.526;
        let lambda = 5.0e-4;
        let n = 400usize;
        let mut trials = Vec::with_capacity(n);
        for index in 0..n {
            let alive = 1.0 - (index as f64 + 0.5) / n as f64;
            trials.push((true, -alive.ln() / lambda));
        }
        let volume = survival_volume(&trials, 1500.0, lattice, repeats, d_i);
        let model = (repeats * lattice).powi(3);
        let expected = 4.0 * (lambda * model) / (DUMBBELL_ALPHA * lattice * d_i);
        let miss = (volume - expected).abs() / expected;
        assert!(miss < 0.03, "{volume} {expected} {miss}");
    }

    #[test]
    fn kaplan_meier_censors_short_windows() {
        let trials = [
            (true, 10.0),
            (false, 20.0),
            (true, 30.0),
            (true, 30.0),
            (false, 40.0),
        ];
        assert_eq!(kaplan_meier(&trials, 5.0), 1.0);
        assert!((kaplan_meier(&trials, 25.0) - 0.8).abs() < 1.0e-15);
        assert!((kaplan_meier(&trials, 35.0) - 0.8 / 3.0).abs() < 1.0e-15);
        let equal = [(true, 10.0), (false, 50.0), (true, 30.0), (false, 50.0)];
        assert!((kaplan_meier(&equal, 35.0) - 0.5).abs() < 1.0e-15);
    }

    fn quantile_trials(lambda: f64, n: usize, window: f64) -> Vec<(bool, f64)> {
        (0..n)
            .map(|index| {
                let alive = 1.0 - (index as f64 + 0.5) / n as f64;
                let time = -alive.ln() / lambda;
                if time <= window {
                    (true, time)
                } else {
                    (false, window)
                }
            })
            .collect()
    }

    #[test]
    fn survival_rate_stops_when_every_trial_is_captured() {
        let (lattice, repeats, lambda): (f64, f64, f64) = (3.644035, 13.0, 0.01);
        let trials = quantile_trials(lambda, 96, 1500.0);
        assert!(trials.iter().all(|(hit, _)| *hit));
        let model = (repeats * lattice).powi(3);
        let rate = survival_rate(&trials, 1500.0, lattice, repeats);
        let miss = (rate / (lambda * model) - 1.0).abs();
        assert!(miss < 0.1, "{rate} {miss}");
    }

    #[test]
    fn censored_rate_recovers_the_hazard_with_censoring() {
        let (lattice, repeats, lambda): (f64, f64, f64) = (3.644035, 13.0, 1.0e-3);
        let trials = quantile_trials(lambda, 400, 1500.0);
        let censored = trials.iter().filter(|(hit, _)| !hit).count();
        assert!(censored > 50, "{censored}");
        let model = (repeats * lattice).powi(3);
        let (rate, low, high, captures) =
            censored_rate(&trials, 1500.0, lattice, repeats).expect("captures");
        assert_eq!(captures, 400 - censored);
        assert!((rate / (lambda * model) - 1.0).abs() < 0.02, "{rate}");
        assert!(
            low < lambda * model && lambda * model < high,
            "{low} {high}"
        );
        let shorter = censored_rate(&trials, 500.0, lattice, repeats).expect("captures");
        assert!(
            (shorter.0 / (lambda * model) - 1.0).abs() < 0.03,
            "{}",
            shorter.0
        );
        assert!(censored_rate(&[(false, 1500.0)], 1500.0, lattice, repeats).is_none());
    }

    #[test]
    fn placed_pair_keeps_8788_atoms_and_is_not_already_a_crystal() {
        let a = measured_lattice(500.0);
        let shells = shell_table(a, 10.8 * a / measured_lattice(500.0));
        assert!(shells.len() > 4);
        let (positions, cell) = place_vacancy_dumbbell(13, a, shells[0].1, [0.0, 0.0, 1.0]);
        assert_eq!(positions.len() / 3, 8788);
        assert!(coordination_deficit(&positions, cell) > 0);
    }

    fn small_dynamics(kind: DynamicsKind, temperature: f64, equil_ps: f64) -> Dynamics {
        Dynamics {
            kind,
            dt_fs: 10.0,
            temperature,
            mass: CU_MASS_AMU,
            gamma_per_ps: 0.1,
            equil_gamma_per_ps: 1.0,
            equil_ps,
            rescale_ps: 0.1,
            rescale_window_k: 0.0,
            piston_ps: 2.0,
            bulk_ev_a3: 0.87,
            pressure_ev_a3: 0.0,
        }
    }

    #[test]
    fn mishin_dilation_matches_the_finite_difference() {
        let (mut positions, cell) = fcc(4, 3.62);
        let mut rng = SplitRng::new(11);
        for value in &mut positions {
            *value += 0.05 * rng.normal();
        }
        let mut potential = MishinCu::new();
        let mut gradient = vec![0.0; positions.len()];
        potential.energy_gradient(&positions, cell, &mut gradient);
        let analytic = potential.dilation_derivative(&positions, cell);
        let numeric = finite_dilation(&mut potential, &positions, cell);
        assert!(potential.dilation.is_some());
        assert!(
            (analytic - numeric).abs() < 1.0e-5 * analytic.abs().max(1.0),
            "{analytic} {numeric}"
        );
    }

    #[test]
    fn small_cells_count_each_pair_once() {
        let (unit, unit_cell) = fcc(1, 3.615);
        let mut scratch = vec![0.0; unit.len()];
        let reference = analytic_gradient(&unit, unit_cell, &mut scratch) / 4.0;
        let mut potential = MishinCu::new();
        for repeats in [4usize, 5, 6, 8] {
            let (positions, cell) = fcc(repeats, 3.615);
            let mut gradient = vec![0.0; positions.len()];
            let per_atom = potential.energy_gradient(&positions, cell, &mut gradient)
                / (positions.len() / 3) as f64;
            assert!(
                (per_atom - reference).abs() < 1.0e-6,
                "{repeats} {per_atom} {reference}"
            );
        }
    }

    #[test]
    fn nve_holds_the_energy_at_ten_femtoseconds() {
        let (positions, cell) = fcc(4, 3.62);
        let mut dynamics = small_dynamics(DynamicsKind::NphNve, 500.0, 0.0);
        dynamics.rescale_window_k = f64::INFINITY;
        let mut potential = MishinCu::new();
        let mut state = MdState::new(&dynamics, &mut potential, positions, cell, 5);
        let mut samples = Vec::new();
        for _ in 0..1000 {
            state.advance(&dynamics, &mut potential);
            if state.step % 50 == 0 {
                let mut scratch = vec![0.0; state.positions.len()];
                let potential_energy =
                    potential.energy_gradient(&state.positions, state.cell, &mut scratch);
                samples.push(potential_energy + state.kinetic(CU_MASS_AMU));
            }
        }
        let atoms = (state.positions.len() / 3) as f64;
        let low = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let high = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (high - low) / atoms < 3.0e-4,
            "{} eV/atom",
            (high - low) / atoms
        );
    }

    #[test]
    fn nph_relaxes_a_compressed_crystal_to_zero_pressure() {
        let (positions, cell) = fcc(4, 3.58);
        let dynamics = small_dynamics(DynamicsKind::NphNve, 50.0, 30.0);
        let mut potential = MishinCu::new();
        let mut state = MdState::new(&dynamics, &mut potential, positions, cell, 9);
        for _ in 0..dynamics.steps(dynamics.equil_ps) {
            state.advance(&dynamics, &mut potential);
        }
        let lattice = state.cell[0][0] / 4.0;
        assert!((lattice - 3.616).abs() < 4.0e-3, "{lattice}");
        for _ in 0..200 {
            state.advance(&dynamics, &mut potential);
        }
        assert_eq!(state.cell[0][0] / 4.0, lattice);
        let temperature = state.temperature(CU_MASS_AMU);
        assert!((temperature - 50.0).abs() < 1.0e-6, "{temperature}");
    }

    #[test]
    fn a_hot_perfect_crystal_is_captured_at_the_first_sample() {
        let a = measured_lattice(500.0);
        let (positions, cell) = fcc(6, a);
        let mut sites = SiteOccupancy::new(&positions, cell, &positions).unwrap();
        let dynamics = small_dynamics(DynamicsKind::Langevin, 500.0, 2.0);
        let mut potential = MishinCu::new();
        let outcome = run_until(
            &dynamics,
            &mut potential,
            positions,
            cell,
            5.0,
            SAMPLE_PS,
            4,
            3,
            |positions: &[f64], cell| site_defects(&mut sites, positions, cell),
        );
        assert!(outcome.captured);
        assert_eq!(outcome.time_ps, 0.0);
        assert_eq!(outcome.final_score, 0);
    }

    #[test]
    fn a_frozen_frenkel_pair_is_never_captured() {
        let a = measured_lattice(500.0);
        let (perfect, _) = fcc(6, a);
        let (positions, cell) =
            place_vacancy_dumbbell(6, a, [3.0 * a, 3.0 * a, 0.0], [0.0, 0.0, 1.0]);
        let mut sites = SiteOccupancy::new(&perfect, cell, &positions).unwrap();
        assert_eq!(site_defects(&mut sites, &positions, cell), 2);
        let dynamics = small_dynamics(DynamicsKind::Langevin, 5.0, 2.0);
        let mut potential = MishinCu::new();
        let outcome = run_until(
            &dynamics,
            &mut potential,
            positions,
            cell,
            3.0,
            SAMPLE_PS,
            4,
            5,
            |positions: &[f64], cell| site_defects(&mut sites, positions, cell),
        );
        assert!(!outcome.captured);
        assert_eq!(outcome.initial_score, 2);
        assert_eq!(outcome.final_score, 2);
    }

    #[test]
    fn a_capture_that_does_not_persist_is_not_counted() {
        let (positions, cell) = fcc(4, 3.6);
        let dynamics = small_dynamics(DynamicsKind::Langevin, 10.0, 0.0);
        let mut potential = MishinCu::new();
        let readings = [3usize, 0, 0, 2, 0, 0, 0, 0, 1];
        let mut index = 0;
        let outcome = run_until(
            &dynamics,
            &mut potential,
            positions,
            cell,
            1.5,
            SAMPLE_PS,
            4,
            1,
            |_: &[f64], _| {
                let value = readings[index.min(readings.len() - 1)];
                index += 1;
                value
            },
        );
        assert!(outcome.captured);
        assert_eq!(outcome.time_ps, 4.0 * SAMPLE_PS);
    }

    #[test]
    fn fcc_site_rounds_to_an_even_parity_site() {
        assert_eq!(fcc_site([0.1, -0.2, 0.05]), [0, 0, 0]);
        assert_eq!(fcc_site([1.1, 0.9, 0.2]), [1, 1, 0]);
        // (1, 0, 0) is not an fcc site; z has the largest residual and flips.
        assert_eq!(fcc_site([0.9, 0.1, 0.4]), [1, 0, 1]);
        assert_eq!(fcc_site([3.0, -1.2, 0.95]), [3, -2, 1]);
    }

    #[test]
    fn dumbbell_site_finds_the_doubly_occupied_cell() {
        let a = measured_lattice(500.0);
        let (mut positions, cell) = isolated_dumbbell(13, a);
        let mut occupancy = Vec::new();
        let center = [cell[0][0] * 0.5, cell[1][1] * 0.5, cell[2][2] * 0.5];
        let expected =
            fcc_site(center.map(|value| 2.0 * value / a)).map(|value| value.rem_euclid(26));
        assert_eq!(
            dumbbell_site(&positions, [0.0; 3], a, 26, &mut occupancy),
            Some(expected)
        );
        let shift = [0.3 * a, -0.2 * a, 0.1 * a];
        for atom in 0..positions.len() / 3 {
            for axis in 0..3 {
                positions[3 * atom + axis] += shift[axis];
            }
        }
        assert_eq!(
            dumbbell_site(&positions, shift, a, 26, &mut occupancy),
            Some(expected)
        );
        let (perfect, _) = fcc(13, a);
        assert_eq!(
            dumbbell_site(&perfect, [0.0; 3], a, 26, &mut occupancy),
            None
        );
    }
}
