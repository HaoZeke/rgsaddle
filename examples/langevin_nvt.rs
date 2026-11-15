//! NVT Langevin velocity Verlet for one defect pair.
//!
//! [`Langevin`] and [`EnergyGradient`] are not copper-specific. The
//! executable's test is a vacancy plus a <100> split interstitial on the
//! Mishin EAM1 that `tests/mishin_cu.rs` evaluates (Mishin, Mehl,
//! Papaconstantopoulos, Voter, and Kress, Phys. Rev. B 63, 224106 (2001),
//! Table I). Recombination follows the first method of Mwungeli, Liu, and
//! Demkowicz, Phys. Rev. Materials 9, 023601 (2025): an atom is
//! recombined-crystal when its coordination, inside 3.1 Å, is 12, sampled
//! every 0.25 ps. The volume is the shell sum of those capture
//! probabilities, not the survival-curve slope in that paper.
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

use rgsaddle::{BOLTZMANN_EV_PER_K, angular_scale_rad_s};

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
}

/// Maxwellian velocities, then velocity Verlet until `recombined` or the window.
///
/// `score` is recorded at the start and at the end. Capture time is the
/// first sample at which `recombined` is true. Sampling starts after the
/// first interval, so a geometry that is already recombined is time zero.
pub fn langevin_until<P, F, S>(
    potential: &mut P,
    positions: &mut [f64],
    cell: [[f64; 3]; 3],
    thermostat: Langevin,
    equil: Langevin,
    equil_ps: f64,
    window_ps: f64,
    sample_ps: f64,
    dt_fs: f64,
    seed: u64,
    mut recombined: F,
    mut score: S,
) -> TrajectoryOutcome
where
    P: EnergyGradient,
    F: FnMut(&[f64], [[f64; 3]; 3]) -> bool,
    S: FnMut(&[f64], [[f64; 3]; 3]) -> usize,
{
    let mut velocities = vec![0.0; positions.len()];
    let mut rng = SplitRng::new(seed);
    let sigma = (thermostat.thermal / thermostat.mass).sqrt();
    for velocity in &mut velocities {
        *velocity = rng.normal() * sigma;
    }
    zero_com(&mut velocities);
    let mut gradient = vec![0.0; positions.len()];
    potential.energy_gradient(positions, cell, &mut gradient);
    let initial_score = score(positions, cell);
    if recombined(positions, cell) {
        return TrajectoryOutcome {
            captured: true,
            time_ps: 0.0,
            steps: 0,
            initial_score,
            final_score: initial_score,
        };
    }
    let sample_steps = ((sample_ps / (dt_fs * 1.0e-3)).round() as usize).max(1);
    let total_steps = ((window_ps / (dt_fs * 1.0e-3)).round() as usize).max(sample_steps);
    let equil_steps = ((equil_ps / (dt_fs * 1.0e-3)).round() as usize).min(total_steps);
    let mut captured = false;
    let mut when = total_steps;
    for step in 1..=total_steps {
        let phase = if step <= equil_steps {
            equil
        } else {
            thermostat
        };
        phase.step(
            potential,
            positions,
            &mut velocities,
            &mut gradient,
            cell,
            &mut rng,
        );
        if step % sample_steps == 0 && recombined(positions, cell) {
            captured = true;
            when = step;
            break;
        }
    }
    let final_score = score(positions, cell);
    TrajectoryOutcome {
        captured,
        time_ps: when as f64 * dt_fs * 1.0e-3,
        steps: when,
        initial_score,
        final_score,
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
        }
    }

    #[inline(always)]
    fn sample(&self, r: f64) -> [f64; 4] {
        if r >= RC {
            return [0.0; 4];
        }
        let t_grid = (r / TABLE_H).max(0.0);
        let mut index = t_grid.floor() as usize;
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
            } else {
                energy += bond.pv;
            }
        }
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
        let skin = 1.0;
        if orthogonal && lengths.iter().all(|length| *length > 2.0 * (RC + skin)) {
            if self.list_stale(positions, lengths, skin) {
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
                delta -= lengths[axis] * (delta / lengths[axis]).round();
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
                d[axis] -= lengths[axis] * (d[axis] * inverse[axis]).round();
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
                        d[axis] -= lengths[axis] * (d[axis] * inverse[axis]).round();
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

pub fn crystal_recombined(positions: &[f64], cell: [[f64; 3]; 3]) -> bool {
    coordination_deficit(positions, cell) == 0
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

fn measured_lattice(temperature: f64) -> f64 {
    match temperature as i32 {
        300 => 3.628311,
        400 => 3.633713,
        500 => 3.638666,
        600 => 3.644160,
        700 => 3.649788,
        _ => 3.638666,
    }
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

struct RunSpec {
    temps: Vec<f64>,
    seeds: u64,
    window_ps: f64,
    dt_fs: f64,
    gamma_per_ps: f64,
    r_max_500: f64,
    repeats: usize,
    random_vacancy: bool,
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
    RunSpec {
        temps,
        seeds: env_u64("MD_SEEDS", 8),
        window_ps: env_f64("MD_PS", 8.0),
        dt_fs: env_f64("MD_DT_FS", 2.0),
        gamma_per_ps: env_f64("MD_GAMMA_PER_PS", 1.0),
        r_max_500: env_f64("MD_RMAX", 10.8),
        repeats: env_u64("MD_REPEATS", 13) as usize,
        random_vacancy: env::var("MD_RANDOM").ok().as_deref() == Some("1"),
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
    let (mut positions, cell) =
        place_vacancy_dumbbell(spec.repeats, job.lattice, job.shell, job.axis);
    let n_atoms = positions.len() / 3;
    let thermostat =
        Langevin::from_physical(spec.dt_fs, spec.gamma_per_ps, job.temperature, CU_MASS_AMU);
    let equil = Langevin::from_physical(spec.dt_fs, 1.0, job.temperature, CU_MASS_AMU);
    let outcome = langevin_until(
        potential,
        &mut positions,
        cell,
        thermostat,
        equil,
        2.0,
        spec.window_ps,
        SAMPLE_PS,
        spec.dt_fs,
        job.seed,
        crystal_recombined,
        coordination_deficit,
    );
    (n_atoms, outcome)
}

fn format_row(spec: &RunSpec, job: &Job, n_atoms: usize, outcome: &TrajectoryOutcome) -> String {
    format!(
        "{:.0},{:.6},{},{n_atoms},{:.6},{:.8},{:.8},{:.8},{},{},{},{:.4},{:.4},{:.4},{},{:.4},{},{},{}",
        job.temperature,
        job.lattice,
        spec.repeats,
        job.shell_r,
        job.shell[0],
        job.shell[1],
        job.shell[2],
        job.orientation,
        job.shell_sites,
        job.seed_index,
        spec.dt_fs,
        spec.gamma_per_ps,
        spec.window_ps,
        u8::from(outcome.captured),
        outcome.time_ps,
        outcome.steps,
        outcome.initial_score,
        outcome.final_score,
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
        match measured_diffusivity(*temperature) {
            Some((d_i, d_sigma)) if d_i > 0.0 => {
                let per_atom = rate_to_volume(rate, lattice, d_i);
                let rel = ((rate_sigma / rate).powi(2) + (d_sigma / d_i).powi(2)).sqrt();
                text.push_str(&format!(
                    "The free-dumbbell diffusivity measured with these dynamics (`direct_md_sia_d_long.md`) is {d_i:.3} ± {d_sigma:.3} Å²/ps. With it, first-method V_iv = {per_atom:.2} ± {:.2} Ω.\n\n",
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

fn random_survival_note(csv: &std::path::Path) -> String {
    let mut trials = Vec::new();
    let mut temperature = 0.0;
    let mut lattice = 0.0;
    let mut repeats = 13.0;
    let mut window = 0.0_f64;
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
        trials.push((captured == 1, time_ps));
    }
    let mut text = String::new();
    text.push_str("# Random-vacancy survival, Mishin Cu\n\n");
    text.push_str("The vacancy is the lattice site nearest the cell center. The <100> dumbbell host is drawn uniformly from the other sites of the 13³ cell, and the dumbbell axis uniformly from the three orientations. Under periodic boundaries the separation is uniform over the cell, which is the ensemble in section 2 of Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025). The fit is ln(n(t)/N) = -(k/V_model) t through the origin, sampled every 25 ps, with V_model the cell volume. V_iv/Ω = 4k / (α a D_i) with α = 2/3 and Ω = a³/4.\n\n");
    text.push_str("Table 1 of that paper divides its k by the D_i in its own Table 1 column, which it quotes from Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022): SIA mean-square displacements in MD. The paper's own trajectories are NPH for 30 ps, then NVE with velocity rescaling, at its lattice constant. Our k comes from the Langevin dynamics in the CSV, so the self-consistent volume divides it by the free-dumbbell diffusivity measured with those same dynamics, potential, and cell, in `direct_md_sia_d_long.md`. Both conversions are listed. σ_k is the standard deviation over 400 bootstrap resamples of the trajectories. The bootstrap σ column carries only that; the total σ columns add the relative error of the D_i used in quadrature.\n\n");
    if trials.is_empty() {
        text.push_str("No trajectories.\n");
        return text;
    }
    let table_d = table_diffusivity(temperature);
    let measured_d = measured_diffusivity(temperature);
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

fn survival_volume(
    trials: &[(bool, f64)],
    t_max: f64,
    lattice: f64,
    repeats: f64,
    d_i: f64,
) -> f64 {
    rate_to_volume(survival_rate(trials, t_max, lattice, repeats), lattice, d_i)
}

/// k in Å³/ps from the through-origin fit of ln(n/N) sampled every 25 ps.
fn survival_rate(trials: &[(bool, f64)], t_max: f64, lattice: f64, repeats: f64) -> f64 {
    let mut times = Vec::new();
    let step = 25.0_f64;
    let mut time = step;
    while time <= t_max + 1.0e-9 {
        let alive = trials
            .iter()
            .filter(|(hit, when)| !(*hit && *when <= time))
            .count();
        let survival = (alive as f64 / trials.len() as f64).max(1.0e-12);
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
    for (_, members) in &by_temp {
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
        for (_shell, orient_groups) in &shells {
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

fn centroid_of(positions: &[f64], atoms: [usize; 2]) -> [f64; 3] {
    let mut center = [0.0; 3];
    for atom in atoms {
        for axis in 0..3 {
            center[axis] += positions[3 * atom + axis];
        }
    }
    [center[0] * 0.5, center[1] * 0.5, center[2] * 0.5]
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

/// One free <100> dumbbell, the V_iv dynamics, and the dumbbell-site
/// displacement over consecutive production blocks of `tau_ps`.
fn sia_d_trajectory(
    spec: &RunSpec,
    temperature: f64,
    seed_index: u64,
    tau_ps: f64,
    blocks: usize,
) -> Vec<String> {
    let lattice = measured_lattice(temperature);
    let (mut positions, cell) = isolated_dumbbell(spec.repeats, lattice);
    let n_atoms = positions.len() / 3;
    let n_half = 2 * spec.repeats as i64;
    let mut potential = MishinCu::new();
    let production =
        Langevin::from_physical(spec.dt_fs, spec.gamma_per_ps, temperature, CU_MASS_AMU);
    let equil = Langevin::from_physical(spec.dt_fs, 1.0, temperature, CU_MASS_AMU);
    let seed = mix_seed(&[temperature.to_bits(), seed_index, 0xD1FF_05EE]);
    let mut rng = SplitRng::new(seed);
    let sigma = (production.thermal / production.mass).sqrt();
    let mut velocities: Vec<f64> = (0..positions.len()).map(|_| rng.normal() * sigma).collect();
    zero_com(&mut velocities);
    let mut gradient = vec![0.0; positions.len()];
    potential.energy_gradient(&positions, cell, &mut gradient);
    let step_ps = spec.dt_fs * 1.0e-3;
    let sample_steps = ((SIA_SAMPLE_PS / step_ps).round() as usize).max(1);
    let equil_steps = ((2.0 / step_ps).round() as usize).div_ceil(sample_steps) * sample_steps;
    let block_steps = ((tau_ps / step_ps).round() as usize).div_ceil(sample_steps) * sample_steps;
    let total_steps = equil_steps + blocks * block_steps;
    let com0 = center_of_mass(&positions);
    let mut occupancy = Vec::new();
    let mut site = dumbbell_site(&positions, [0.0; 3], lattice, n_half, &mut occupancy)
        .expect("the starting dumbbell occupies one site");
    let mut unwrapped = [0i64; 3];
    let mut block_start = unwrapped;
    let mut changes = 0usize;
    let mut unresolved = 0usize;
    let mut rows = Vec::with_capacity(blocks);
    for step in 1..=total_steps {
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
        if step % sample_steps != 0 {
            continue;
        }
        let com = center_of_mass(&positions);
        let shift = [com[0] - com0[0], com[1] - com0[1], com[2] - com0[2]];
        match dumbbell_site(&positions, shift, lattice, n_half, &mut occupancy) {
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
        } else if step > equil_steps && (step - equil_steps) % block_steps == 0 {
            let block = (step - equil_steps) / block_steps - 1;
            let d =
                [0, 1, 2].map(|axis| (unwrapped[axis] - block_start[axis]) as f64 * 0.5 * lattice);
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            rows.push(format!(
                "{temperature:.0},{lattice:.6},{},{n_atoms},{:.4},{:.4},{seed_index},{block},{:.4},{:.6},{:.6},{:.6},{r2:.6},{changes},{unresolved}",
                spec.repeats,
                spec.dt_fs,
                spec.gamma_per_ps,
                block_steps as f64 * step_ps,
                d[0],
                d[1],
                d[2],
            ));
            block_start = unwrapped;
            changes = 0;
            unresolved = 0;
        }
    }
    rows
}

fn sia_d_path() -> PathBuf {
    if let Ok(over) = env::var("MD_SIA_OUT") {
        return PathBuf::from(over);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("validation/direct_md_sia_d_long.csv")
}

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
            .chunks_exact(2)
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

fn measured_diffusivity(temperature: f64) -> Option<(f64, f64)> {
    let rows = read_sia_blocks(&sia_d_path());
    sia_estimate(&rows, temperature).map(|estimate| (estimate.d, estimate.sigma))
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
    text.push_str("# Free dumbbell diffusivity, Mishin Cu\n\n");
    text.push_str("One <100> dumbbell in the perfect 13×13×13 cell (8789 atoms) on the Mishin EAM1 of `tests/mishin_cu.rs`, at the lattice constant of the V_iv runs. The dynamics are the V_iv dynamics: Maxwellian start, 2 ps of Langevin at gamma = 1/ps, then gamma and dt from the columns `gamma_per_ps` and `dt_fs`. The dumbbell is the one doubly occupied Wigner-Seitz cell of the ideal lattice, found every 0.1 ps after the centre-of-mass drift of the cell is removed. A sample with no single doubly occupied cell is skipped and counted in `unresolved_samples`. Each row is the dumbbell-site displacement over one production block of `tau_ps`.\n\n");
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

fn run_sia_d() {
    let spec = run_spec();
    let blocks = env_u64("MD_SIA_BLOCKS", 40) as usize;
    let path = sia_d_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("validation dir");
    }
    if !path.exists() {
        let mut header = File::create(&path).expect("csv");
        writeln!(header, "{SIA_D_HEADER}").unwrap();
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
        "sia_d jobs {} temps {:?} seeds {} blocks {blocks} dt_fs {} gamma {}",
        jobs.len(),
        spec.temps,
        spec.seeds,
        spec.dt_fs,
        spec.gamma_per_ps
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
        let path = sia_d_path();
        fs::write(note_path(&path), sia_d_note(&path)).expect("note");
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
                let mut origin = defect_centroid(&positions, lengths, None).unwrap_or([0.0; 3]);
                let mut cursor = origin;
                let mut unwrapped = origin;
                let window_ps = spec.window_ps.max(40.0);
                let steps = ((window_ps / (spec.dt_fs * 1.0e-3)).round() as usize).max(1);
                let equil_steps = ((2.0 / (spec.dt_fs * 1.0e-3)).round() as usize);
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
                        origin = unwrapped;
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
        writeln!(header, "{HEADER}").unwrap();
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
                        "{}/{} T {:.0} shell {:.3} {} seed {} captured {} t {:.3} ps non12 {} -> {}",
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
    fn placed_pair_keeps_8788_atoms_and_is_not_already_a_crystal() {
        let a = measured_lattice(500.0);
        let shells = shell_table(a, 10.8 * a / measured_lattice(500.0));
        assert!(shells.len() > 4);
        let (positions, cell) = place_vacancy_dumbbell(13, a, shells[0].1, [0.0, 0.0, 1.0]);
        assert_eq!(positions.len() / 3, 8788);
        assert!(coordination_deficit(&positions, cell) > 0);
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
